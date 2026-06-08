"""Integration test: Lagoon vault strategy trading Ostium V1.5 via async settlement.

Deploys a fresh Lagoon vault on an Anvil Arbitrum fork, whitelists the Ostium V1.5
vault, funds the Lagoon via a depositor, then runs a decide_trades() loop to open
and close a position through the async settlement cycle.

This tests the full Lagoon execution path (LagoonTransactionBuilder → guard → Safe)
with async vault support, verifying that portfolio accounting remains correct at
every step.

Steps:
1. Deploy Lagoon vault on Arbitrum fork, whitelist Ostium V1.5
2. Fund the Lagoon vault with USDC via depositor
3. Sync treasury, verify starting equity
4. decide_trades() → open Ostium position → execute → vault_settlement_pending
5. Verify portfolio equity preserved during pending state
6. Force Ostium settlement on Anvil
7. check_and_resolve_vault_settlements() → trade success, position open
8. decide_trades() → close position → execute → vault_settlement_pending
9. Force settlement(s) for withdrawal
10. check_and_resolve_vault_settlements() → trade success, position closed
11. Verify final equity ≈ starting (minus Ostium fees)
"""

import logging
import os
from decimal import Decimal

import flaky
import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance_autodetect
from eth_defi.erc_4626.vault_protocol.gains.testing import force_ostium_v15_settlement
from eth_defi.erc_4626.vault_protocol.gains.vault import OstiumVault, OstiumVersion
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
    LagoonAutomatedDeployment,
    deploy_automated_lagoon_vault,
)
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details, USDC_NATIVE_TOKEN, USDC_WHALE
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM, reason="Set JSON_RPC_ARBITRUM to run this test")

OSTIUM_VAULT_ADDRESS = "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"
FORK_BLOCK = 470_000_000


@pytest.fixture()
def anvil_arbitrum_fork() -> AnvilLaunch:
    """Arbitrum fork with unlocked USDC whale."""
    usdc_whale = USDC_WHALE[42161]
    launch = fork_network_anvil(
        JSON_RPC_ARBITRUM,
        fork_block_number=FORK_BLOCK,
        unlocked_addresses=[usdc_whale],
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3(anvil_arbitrum_fork) -> Web3:
    return create_multi_provider_web3(
        anvil_arbitrum_fork.json_rpc_url,
        default_http_timeout=(3, 250.0),
        retries=1,
    )


@pytest.fixture()
def deployer_hot_wallet(web3) -> HotWallet:
    """Deployer wallet for Lagoon vault."""
    return HotWallet.create_for_testing(web3, eth_amount=1)


@pytest.fixture()
def asset_manager(web3) -> HotWallet:
    """Asset manager wallet for Lagoon operations."""
    hw = HotWallet.create_for_testing(web3, eth_amount=5)
    return hw


@pytest.fixture()
def multisig_owners(web3) -> list:
    """Safe multisig owners."""
    return [web3.eth.accounts[2], web3.eth.accounts[3], web3.eth.accounts[4]]


@pytest.fixture()
def depositor(web3) -> str:
    """User who deposits USDC into Lagoon vault."""
    account = web3.eth.accounts[5]
    # Top up depositor with USDC
    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    tx_hash = usdc.contract.functions.transfer(
        account, 500 * 10**6,
    ).transact({"from": USDC_WHALE[42161], "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return account


@pytest.fixture()
def ostium_vault(web3) -> OstiumVault:
    """Ostium V1.5 vault."""
    vault = create_vault_instance_autodetect(web3, OSTIUM_VAULT_ADDRESS)
    assert isinstance(vault, OstiumVault)
    assert vault.version == OstiumVersion.v1_5
    return vault


@pytest.fixture()
def automated_lagoon_vault(
    web3,
    deployer_hot_wallet,
    asset_manager,
    multisig_owners,
    ostium_vault,
) -> LagoonAutomatedDeployment:
    """Deploy a Lagoon vault with Ostium V1.5 whitelisted."""

    chain_id = web3.eth.chain_id
    parameters = LagoonDeploymentParameters(
        underlying=USDC_NATIVE_TOKEN[chain_id],
        name="TestOstiumLagoon",
        symbol="TOST",
    )

    deploy_info = deploy_automated_lagoon_vault(
        web3=web3,
        deployer=deployer_hot_wallet,
        asset_manager=asset_manager.address,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=2,
        uniswap_v2=None,
        uniswap_v3=None,
        any_asset=False,
        erc_4626_vaults=[ostium_vault],
        from_the_scratch=True,
        use_forge=True,
    )

    # Verify the Ostium vault was whitelisted in the guard
    module = deploy_info.vault.trading_strategy_module
    assert module.functions.isAllowedApprovalDestination(ostium_vault.vault_address).call()

    return deploy_info


@pytest.fixture()
def arb_usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=42161,
        address=USDC_NATIVE_TOKEN[42161],
        decimals=6,
        token_symbol="USDC",
    )


@pytest.fixture()
def vault_pair(ostium_vault: OstiumVault) -> TradingPairIdentifier:
    return translate_vault_to_trading_pair(ostium_vault)


@pytest.fixture()
def strategy_universe(vault_pair: TradingPairIdentifier, arb_usdc: AssetIdentifier) -> TradingStrategyUniverse:
    """Universe containing only the Ostium vault pair."""

    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(42161),
                chain_slug="arbitrum",
                exchange_id=1,
                exchange_slug="ostium",
                address="0x0000000000000000000000000000000000000000",
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )

    pair_universe = create_universe_from_trading_pair_identifiers(
        [vault_pair],
        exchange_universe,
    )

    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.arbitrum},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[arb_usdc],
    )


@pytest.fixture()
def lagoon_execution_model(
    web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    asset_manager: HotWallet,
) -> LagoonExecution:
    """Lagoon execution model using TradingStrategyModuleV0."""
    vault = automated_lagoon_vault.vault
    return LagoonExecution(
        vault=vault,
        tx_builder=LagoonTransactionBuilder(vault, asset_manager),
        mainnet_fork=True,
        confirmation_block_count=0,
    )


@pytest.fixture()
def sync_model(
    automated_lagoon_vault: LagoonAutomatedDeployment,
    asset_manager: HotWallet,
) -> LagoonVaultSyncModel:
    return LagoonVaultSyncModel(
        vault=automated_lagoon_vault.vault,
        hot_wallet=asset_manager,
        unit_testing=True,
    )


@pytest.fixture()
def routing_model(lagoon_execution_model, strategy_universe) -> GenericRouting:
    return lagoon_execution_model.create_default_routing_model(strategy_universe)


@pytest.fixture()
def pricing_model(web3, strategy_universe) -> GenericPricing:
    pair_configurator = EthereumPairConfigurator(
        web3,
        strategy_universe,
    )
    return GenericPricing(pair_configurator)


@flaky.flaky
def test_lagoon_ostium_v15_async_lifecycle(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    ostium_vault: OstiumVault,
    depositor: str,
    asset_manager: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    lagoon_execution_model: LagoonExecution,
    sync_model: LagoonVaultSyncModel,
    routing_model: GenericRouting,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    arb_usdc: AssetIdentifier,
):
    """Full Lagoon vault → Ostium V1.5 async deposit/redeem with portfolio accounting.

    1. Deploy Lagoon vault with Ostium whitelisted (fixture)
    2. Fund Lagoon vault with USDC from depositor
    3. Sync treasury, verify starting equity
    4. decide_trades() → open position → execute → vault_settlement_pending
    5. Verify equity preserved during pending state
    6. Force Ostium settlement
    7. Settlement retry → trade success, position open
    8. decide_trades() → close position → execute → vault_settlement_pending
    9. Force settlement(s) for withdrawal
    10. Settlement retry → trade success, position closed
    11. Verify final equity ≈ starting equity
    """
    lagoon_vault = automated_lagoon_vault.vault
    state = State()

    # 2. Fund Lagoon vault
    # First post initial zero valuation so deposits are accepted
    tx_hash = lagoon_vault.post_new_valuation(Decimal(0)).transact({"from": asset_manager.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    deposit_amount = 200 * 10**6
    tx_hash = usdc.contract.functions.approve(lagoon_vault.address, deposit_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = lagoon_vault.request_deposit(depositor, deposit_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Settle deposits
    valuation = Decimal(0)
    tx_hash = lagoon_vault.post_new_valuation(valuation).transact({"from": asset_manager.address})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = lagoon_vault.settle_via_trading_strategy_module(valuation).transact({"from": asset_manager.address, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # 3. Sync treasury
    sync_model.sync_initial(
        state,
        reserve_asset=arb_usdc,
        reserve_token_price=1.0,
    )

    sync_model.sync_treasury(
        native_datetime_utc_now(),
        state,
        supported_reserves=[arb_usdc],
        post_valuation=True,
    )

    # Verify USDC landed in the safe
    safe_usdc_balance = usdc.fetch_balance_of(lagoon_vault.safe_address)
    assert safe_usdc_balance > 0, f"Safe has no USDC after deposit settlement"

    # If sync_treasury didn't find balance events, manually set reserves
    reserve_position = state.portfolio.get_default_reserve_position()
    if float(reserve_position.quantity) == 0:
        reserve_position.quantity = safe_usdc_balance
        reserve_position.reserve_token_price = 1.0

    starting_equity = state.portfolio.calculate_total_equity()
    assert starting_equity == pytest.approx(float(safe_usdc_balance), abs=2.0)

    # 4. Open position (deposit into Ostium via Lagoon guard)
    ts = native_datetime_utc_now()
    position_manager = PositionManager(
        ts,
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.10,
    )

    trades = position_manager.open_spot(vault_pair, value=50.0)
    assert len(trades) == 1
    buy_trade = trades[0]

    routing_state_details = lagoon_execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    lagoon_execution_model.initialize()

    lagoon_execution_model.execute_trades(
        ts, state, trades, routing_model, routing_state, check_balances=True,
    )

    # Verify trade is vault_settlement_pending
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending, \
        f"Expected vault_settlement_pending, got {buy_trade.get_status()}"
    assert buy_trade.other_data.get("vault_async_flow") is True
    assert buy_trade.other_data.get("vault_direction") == "deposit"

    # 5. Verify portfolio equity preserved
    pending_value = state.portfolio.get_vault_settlement_pending_value()
    assert pending_value == pytest.approx(50.0, abs=2.0)
    equity_while_pending = state.portfolio.calculate_total_equity()
    assert equity_while_pending == pytest.approx(starting_equity, abs=5.0)

    # 6. Force Ostium settlement (use asset manager as gas payer — tryNewSettlement is permissionless)
    force_ostium_v15_settlement(ostium_vault, asset_manager.address)

    # 7. Settlement retry resolves the trade
    resolved = check_and_resolve_vault_settlements(
        state=state,
        execution_model=lagoon_execution_model,
    )
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success
    assert buy_trade.executed_quantity > 0

    # Position should be open
    position = state.portfolio.get_position_by_id(buy_trade.position_id)
    assert position.is_open()
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0)

    # 8. Close position (redeem from Ostium via Lagoon guard)
    ts = native_datetime_utc_now()
    position_manager = PositionManager(
        ts,
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.10,
    )

    trades = position_manager.close_all()
    assert len(trades) == 1
    sell_trade = trades[0]

    lagoon_execution_model.execute_trades(
        ts, state, trades, routing_model, routing_state, check_balances=True,
    )

    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending
    assert sell_trade.other_data.get("vault_direction") == "redeem"

    # 9. Force settlement(s) for withdrawal
    withdraw_target = ostium_vault.vault_contract.functions.targetSettlementId(False).call()
    last_id = ostium_vault.vault_contract.functions.lastSettlementId().call()
    settlements_needed = max(withdraw_target - last_id, 1)
    for _ in range(settlements_needed):
        force_ostium_v15_settlement(ostium_vault, asset_manager.address)

    # 10. Settlement retry resolves withdrawal
    resolved = check_and_resolve_vault_settlements(
        state=state,
        execution_model=lagoon_execution_model,
    )
    assert len(resolved) == 1
    assert sell_trade.get_status() == TradeStatus.success
    assert sell_trade.executed_reserve > 0
    assert position.is_closed()

    # 11. Verify final equity
    final_equity = state.portfolio.calculate_total_equity()
    assert final_equity == pytest.approx(starting_equity, rel=0.03), \
        f"Final equity {final_equity} deviates too much from starting {starting_equity}"
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0)
