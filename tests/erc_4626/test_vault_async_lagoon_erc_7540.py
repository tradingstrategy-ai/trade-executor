"""Integration test for the Lagoon ERC-7540 async vault deposit/redeem lifecycle.

A hot-wallet strategy deposits into, then redeems from, a real ERC-7540 Lagoon
vault on an Anvil fork of Base. The test plays the role of the **target vault's
asset manager** (impersonated via an unlocked address): between strategy decision
cycles it either pushes the settlement queue forward (``force_lagoon_settle``) or
deliberately holds it, simulating the unknown live settlement delay.

Settlements are resolved through the same polymorphic hook ``StrategyRunner.tick()``
calls each cycle for live execution models
(``ExecutionModel.resolve_pending_vault_settlements`` ->
``check_and_resolve_vault_settlements``), and so exercise the Lagoon
``ERC7540DepositManager`` ticket serialise/reconstruct/status overrides added in
eth_defi. The full ``tick()`` wiring of the hook is separately covered by the
backtest tests in ``tests/backtest/test_backtest_async_vault.py``, which run the
whole ``ExecutionLoop``. This test uses the direct-driver pattern of
``test_vault_async_ostium_v15.py`` and the real 722-capital ERC-7540 vault used by
eth_defi's own Lagoon 7540 test, because a freshly deployed vault is not in the
trading-strategy dataset the CLI universe loader expects, and
``deploy_automated_lagoon_vault`` produces a v0.5.0 Safe vault rather than an
ERC-7540 vault.

Steps:
1. decide_trades() deposits into the vault -> requestDeposit -> vault_settlement_pending.
2. Hold the queue: settlement retry keeps it pending (no double-deposit).
3. Settle as asset manager -> settlement retry claims -> position open, shares on-chain.
4. decide_trades() redeems -> requestRedeem -> vault_settlement_pending; shares escrowed.
5. Settle as asset manager -> settlement retry claims -> position closed.
6. Final equity approximately equals starting equity.
"""

import logging
import os
from decimal import Decimal
from pathlib import Path

import flaky
import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig, LagoonDeploymentParameters, deploy_automated_lagoon_vault
from eth_defi.erc_4626.vault_protocol.lagoon.testing import force_lagoon_settle
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, fund_erc20_on_anvil, launch_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details, USDC_NATIVE_TOKEN
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.cli.close_position import close_single_or_all_positions
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.account_correction import calculate_account_corrections
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import revalue_state
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="Set JSON_RPC_BASE to run this test")

#: Real ERC-7540 Lagoon vault on Base (722 capital), as used by eth_defi's Lagoon 7540 test.
LAGOON_7540_VAULT = "0xb09f761cb13baca8ec087ac476647361b6314f98"
#: The vault's real asset manager — impersonated on the fork to settle the queue.
TARGET_VAULT_ASSET_MANAGER = "0x3B95C7cD4075B72ecbC4559AF99211C2B6591b2E"
ANVIL_DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
FORK_BLOCK = 41_950_000
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_WHALE_BASE = "0x40EbC1Ac8d4Fedd2E144b75fe9C0420BE82750c6"
DEPOSIT_VALUE = 50.0


@pytest.fixture()
def anvil_base_fork() -> AnvilLaunch:
    """Fork Base with the USDC whale and the target vault's asset manager unlocked."""
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        fork_block_number=FORK_BLOCK,
        unlocked_addresses=[USDC_WHALE_BASE, TARGET_VAULT_ASSET_MANAGER],
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_arbitrum_fork() -> AnvilLaunch:
    """Arbitrum fork used as the source Lagoon chain for satellite settlement tests."""
    assert JSON_RPC_ARBITRUM, "JSON_RPC_ARBITRUM not set"
    launch = fork_network_anvil(JSON_RPC_ARBITRUM)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_home_chain() -> AnvilLaunch:
    """Launch a separate home-chain Anvil used to model multichain settlement signing."""
    launch = launch_anvil()
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3(anvil_base_fork) -> Web3:
    return create_multi_provider_web3(anvil_base_fork.json_rpc_url, default_http_timeout=(3, 250.0), retries=1)


@pytest.fixture()
def arbitrum_web3(anvil_arbitrum_fork) -> Web3:
    web3 = create_multi_provider_web3(anvil_arbitrum_fork.json_rpc_url, default_http_timeout=(3, 250.0), retries=1)
    assert web3.eth.chain_id == ChainId.arbitrum.value
    return web3


@pytest.fixture()
def base_usdc() -> AssetIdentifier:
    return AssetIdentifier(chain_id=8453, address=BASE_USDC_ADDRESS.lower(), token_symbol="USDC", decimals=6)


@pytest.fixture()
def strategy_hot_wallet(web3) -> HotWallet:
    """Hot wallet acting as the strategy depositor, funded with ETH + USDC."""
    hw = HotWallet.create_for_testing(web3, test_account_n=1, eth_amount=5)
    hw.sync_nonce(web3)
    usdc = fetch_erc20_details(web3, BASE_USDC_ADDRESS)
    tx_hash = usdc.contract.functions.transfer(hw.address, 200 * 10**6).transact({"from": USDC_WHALE_BASE, "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return hw


@pytest.fixture()
def asset_manager_address(web3) -> str:
    """The impersonated target-vault asset manager, topped up with ETH to send settle txs."""
    web3.eth.send_transaction({"from": web3.eth.accounts[0], "to": TARGET_VAULT_ASSET_MANAGER, "value": 5 * 10**18})
    return TARGET_VAULT_ASSET_MANAGER


@pytest.fixture()
def target_vault(web3):
    return create_vault_instance(
        web3,
        LAGOON_7540_VAULT,
        features={ERC4626Feature.lagoon_like, ERC4626Feature.erc_7540_like},
    )


@pytest.fixture()
def vault_pair(target_vault) -> TradingPairIdentifier:
    return translate_vault_to_trading_pair(target_vault)


@pytest.fixture()
def strategy_universe(vault_pair, base_usdc) -> TradingStrategyUniverse:
    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(8453),
                chain_slug="base",
                exchange_id=1,
                exchange_slug="lagoon",
                address="0x0000000000000000000000000000000000000000",
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )
    pair_universe = create_universe_from_trading_pair_identifiers([vault_pair], exchange_universe)
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.base},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
    )
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[base_usdc])


@pytest.fixture()
def execution_model(web3, strategy_hot_wallet) -> EthereumExecution:
    return EthereumExecution(
        HotWalletTransactionBuilder(web3, strategy_hot_wallet),
        mainnet_fork=True,
        confirmation_block_count=0,
    )


@pytest.fixture()
def sync_model(web3, strategy_hot_wallet) -> HotWalletSyncModel:
    return HotWalletSyncModel(web3, strategy_hot_wallet)


@pytest.fixture()
def routing_model(execution_model, strategy_universe):
    return execution_model.create_default_routing_model(strategy_universe)


@pytest.fixture()
def pair_configurator(web3, strategy_universe, execution_model) -> EthereumPairConfigurator:
    return EthereumPairConfigurator(web3, strategy_universe, execution_model=execution_model)


@pytest.fixture()
def pricing_model(pair_configurator) -> GenericPricing:
    return GenericPricing(pair_configurator)


@pytest.fixture()
def valuation_model(pair_configurator) -> GenericValuation:
    return GenericValuation(pair_configurator)


def _execute(execution_model, routing_model, strategy_universe, state, trades):
    """Broadcast a batch of trades through the live execution model."""
    routing_state = routing_model.create_routing_state(strategy_universe, execution_model.get_routing_state_details())
    execution_model.execute_trades(native_datetime_utc_now(), state, trades, routing_model, routing_state, check_balances=True)


def _open_position_and_request_redeem(
    asset_manager_address: str,
    target_vault,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
) -> tuple[State, Decimal, object, object]:
    """Open a settled ERC-7540 position and leave its redeem request pending."""
    state = State()
    sync_model.sync_initial(state, reserve_asset=base_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[base_usdc])
    starting_equity = state.portfolio.calculate_total_equity()

    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    buy_trades = pm.open_spot(vault_pair, value=DEPOSIT_VALUE)
    _execute(execution_model, routing_model, strategy_universe, state, buy_trades)
    assert buy_trades[0].get_status() == TradeStatus.vault_settlement_pending
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1

    position = state.portfolio.get_position_by_id(buy_trades[0].position_id)
    assert position.is_open()

    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    sell_trades = pm.close_all()
    _execute(execution_model, routing_model, strategy_universe, state, sell_trades)
    sell_trade = sell_trades[0]
    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending

    return state, starting_equity, position, sell_trade


class _SingleChainWeb3Config:
    """Minimal Web3Config-like object for settlement retry tests."""

    def __init__(self, chain_id: ChainId, web3: Web3):
        self.chain_id = chain_id
        self.web3 = web3

    def get_connection(self, chain_id: ChainId) -> Web3:
        assert chain_id == self.chain_id
        return self.web3


class _MultiChainWeb3Config:
    """Minimal Web3Config-like object for multichain settlement retry tests."""

    def __init__(self, connections: dict[ChainId, Web3]):
        self.connections = connections

    def get_connection(self, chain_id: ChainId) -> Web3:
        return self.connections[chain_id]


def _deploy_source_and_satellite_lagoon(
    arbitrum_web3: Web3,
    base_web3: Web3,
    target_vault,
) -> tuple[object, object, HotWallet]:
    """Deploy a source Lagoon vault on Arbitrum and a Base satellite Safe/module."""
    deployer = HotWallet.from_private_key(ANVIL_DEPLOYER_PRIVATE_KEY)
    for chain_web3 in (arbitrum_web3, base_web3):
        chain_web3.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])

    asset_manager = deployer
    safe_salt_nonce = 7540

    source_parameters = LagoonDeploymentParameters(
        underlying=USDC_NATIVE_TOKEN[ChainId.arbitrum.value],
        name="ERC7540SettlementSource",
        symbol="E7540S",
    )
    deployer.sync_nonce(arbitrum_web3)
    source_deployment = deploy_automated_lagoon_vault(
        web3=arbitrum_web3,
        deployer=deployer,
        config=LagoonConfig(
            parameters=source_parameters,
            safe_owners=[deployer.address],
            safe_threshold=1,
            asset_manager=asset_manager.address,
            any_asset=True,
            from_the_scratch=False,
            use_forge=False,
            safe_salt_nonce=safe_salt_nonce,
        ),
    )

    satellite_parameters = LagoonDeploymentParameters(
        underlying=USDC_NATIVE_TOKEN[ChainId.base.value],
        name="ERC7540SettlementSatellite",
        symbol="E7540B",
    )
    deployer.sync_nonce(base_web3)
    satellite_deployment = deploy_automated_lagoon_vault(
        web3=base_web3,
        deployer=deployer,
        config=LagoonConfig(
            parameters=satellite_parameters,
            safe_owners=[deployer.address],
            safe_threshold=1,
            asset_manager=asset_manager.address,
            any_asset=True,
            erc_4626_vaults=[target_vault],
            from_the_scratch=False,
            use_forge=False,
            safe_salt_nonce=safe_salt_nonce,
            satellite_chain=True,
        ),
    )

    assert source_deployment.safe_address == satellite_deployment.safe_address
    return source_deployment, satellite_deployment, asset_manager


@flaky.flaky
def test_lagoon_erc_7540_async_deposit_redeem_lifecycle(
    web3: Web3,
    asset_manager_address: str,
    target_vault,
    strategy_hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model,
    pricing_model: GenericPricing,
    valuation_model: GenericValuation,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
    tmp_path,
):
    """Full Lagoon ERC-7540 async deposit/redeem lifecycle with on-chain + state checks.

    1. Deposit: requestDeposit -> vault_settlement_pending, pending on-chain.
    2. While the deposit is pending: revaluation values the quantity-0 position to zero,
       check-accounts reports no mismatch, and the settlement retry keeps the trade
       pending while the queue is held (no double-deposit).
    3. Settle as asset manager: settlement retry claims -> position open, shares on-chain.
    4. Redeem: requestRedeem -> pending, shares escrowed on-chain (owner balance zero).
    5. While the redeem is pending: check-accounts reports no mismatch even though the
       owner wallet no longer holds the shares (escrow subtracted from expected), and
       revaluation keeps the share equity.
    6. Restart simulation: persist the state JSON mid-pending and reload it, proving the
       serialised settlement ticket reconstructs from disk.
    7. Settle as asset manager: settlement retry resolves against the RELOADED state ->
       position closed.
    8. Final equity approximately equals starting equity.
    """

    owner = strategy_hot_wallet.address

    # State init and starting equity.
    state = State()
    sync_model.sync_initial(state, reserve_asset=base_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[base_usdc])
    starting_equity = state.portfolio.calculate_total_equity()
    assert starting_equity == pytest.approx(200.0, abs=1.0)

    # 1. Deposit cycle.
    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    buy_trades = pm.open_spot(vault_pair, value=DEPOSIT_VALUE)
    _execute(execution_model, routing_model, strategy_universe, state, buy_trades)
    buy_trade = buy_trades[0]
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending
    assert buy_trade.other_data["vault_direction"] == "deposit"
    assert target_vault.vault_contract.functions.pendingDepositRequest(0, owner).call() > 0
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(DEPOSIT_VALUE, abs=1.0)

    # 2. While the deposit is pending: revaluation values the quantity-0 position to
    #    zero without crashing (the runner revalues every tick before the resolver).
    revalue_state(state, native_datetime_utc_now(), valuation_model)
    pending_position = state.portfolio.get_position_by_id(buy_trade.position_id)
    assert pending_position.get_value() == pytest.approx(0.0, abs=1e-6)

    # 2. Check-accounts during the pending deposit: state reserve and position quantity
    #    both already exclude the committed capital, so no mismatch.
    corrections = list(calculate_account_corrections(
        pair_universe=strategy_universe.data_universe.pairs,
        reserve_assets=strategy_universe.reserve_assets,
        state=state,
        sync_model=sync_model,
        all_balances=True,
    ))
    mismatches = [c for c in corrections if c.mismatch]
    assert len(mismatches) == 0, f"Mismatches during pending deposit: {mismatches}"

    # 2. Hold the queue — settlement retry leaves it pending. Resolve through the
    #    polymorphic execution-model hook, the exact call StrategyRunner.tick()
    #    makes each cycle for live execution models.
    assert execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now()) == []
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending

    # 3. Settle as asset manager, then resolve.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success
    position = state.portfolio.open_positions[buy_trade.position_id]
    assert position.is_open()
    assert target_vault.share_token.fetch_balance_of(owner) > 0

    # 4. Redeem cycle.
    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    sell_trades = pm.close_all()
    _execute(execution_model, routing_model, strategy_universe, state, sell_trades)
    sell_trade = sell_trades[0]
    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending
    assert sell_trade.other_data["vault_direction"] == "redeem"
    assert target_vault.vault_contract.functions.pendingRedeemRequest(0, owner).call() > 0
    # requestRedeem() escrowed the shares: the owner wallet balance is zero on-chain
    # while our state still counts the full position quantity.
    assert target_vault.share_token.fetch_balance_of(owner) == 0
    assert position.get_quantity() > 0

    # 5. Check-accounts during the pending redeem: the expected on-chain balance must
    #    subtract the escrowed shares, otherwise this reports a false mismatch and
    #    correct-accounts would wrongly close the position mid-settlement.
    corrections = list(calculate_account_corrections(
        pair_universe=strategy_universe.data_universe.pairs,
        reserve_assets=strategy_universe.reserve_assets,
        state=state,
        sync_model=sync_model,
        all_balances=True,
    ))
    mismatches = [c for c in corrections if c.mismatch]
    assert len(mismatches) == 0, f"Mismatches during pending redeem: {mismatches}"

    # 5. Revaluation during the pending redeem keeps the share equity.
    revalue_state(state, native_datetime_utc_now(), valuation_model)
    assert position.get_value() == pytest.approx(DEPOSIT_VALUE, rel=0.05)

    # 6. Restart simulation: persist the state mid-pending and reload it. The
    #    settlement ticket (including the ERC-7540 request id) must reconstruct
    #    from the serialised JSON.
    state_file = tmp_path / "lagoon-erc-7540-pending.json"
    state.write_json_file(state_file)
    state2 = State.read_json_file(state_file)
    pending_trades2 = [
        t
        for p in state2.portfolio.open_positions.values()
        for t in p.trades.values()
        if t.get_status() == TradeStatus.vault_settlement_pending
    ]
    assert len(pending_trades2) == 1
    sell_trade2 = pending_trades2[0]
    assert sell_trade2.other_data["vault_request_id"] == sell_trade.other_data["vault_request_id"]

    # 7. Settle redeem as asset manager, then resolve against the RELOADED state
    #    through the same polymorphic execution-model hook the runner calls.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state2, ts=native_datetime_utc_now())
    assert len(resolved) == 1
    assert sell_trade2.get_status() == TradeStatus.success
    position2 = state2.portfolio.get_position_by_id(sell_trade2.position_id)
    assert position2.is_closed()

    # 8. Final equity approximately equals starting equity (on the reloaded state).
    assert state2.portfolio.calculate_total_equity() == pytest.approx(starting_equity, rel=0.05)
    assert state2.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


@flaky.flaky
def test_lagoon_erc_7540_close_position_rerun(
    web3: Web3,
    asset_manager_address: str,
    target_vault,
    strategy_hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model,
    pricing_model: GenericPricing,
    valuation_model: GenericValuation,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
    mocker,
):
    """close-position walks an ERC-7540 redeem through the queue across re-runs.

    An ERC-7540 settlement can take days, so the close-position/close-all CLI
    command must never wait for it. This test exercises the re-run pattern with
    Anvil force-settling disabled (is_anvil patched to False), so the command
    behaves as it would on a real chain.

    1. Deposit into the vault, settle as asset manager and resolve -> position open.
    2. First close-position run: requestRedeem broadcast, trade left in
       vault_settlement_pending, command exits cleanly, position stays open.
    3. Second run while the queue is still unsettled: the in-flight settlement is
       recognised, no new trade is created, command exits cleanly.
    4. Settle the queue as asset manager.
    5. Third run: the pre-flight settlement sweep claims the redeem and the
       position is closed without any new on-chain request.
    6. Final equity approximately equals starting equity.
    """

    owner = strategy_hot_wallet.address

    # 1. Deposit into the vault, settle and resolve -> position open.
    state = State()
    sync_model.sync_initial(state, reserve_asset=base_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[base_usdc])
    starting_equity = state.portfolio.calculate_total_equity()

    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    buy_trades = pm.open_spot(vault_pair, value=DEPOSIT_VALUE)
    _execute(execution_model, routing_model, strategy_universe, state, buy_trades)
    assert buy_trades[0].get_status() == TradeStatus.vault_settlement_pending
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1
    position = state.portfolio.get_position_by_id(buy_trades[0].position_id)
    assert position.is_open()
    position_id = position.position_id

    # Behave as on a real chain: no force-settling inside close-position.
    mocker.patch("tradeexecutor.cli.close_position.is_anvil", return_value=False)

    execution_context = ExecutionContext(mode=ExecutionMode.one_off)
    routing_state = routing_model.create_routing_state(strategy_universe, execution_model.get_routing_state_details())

    def _run_close():
        close_single_or_all_positions(
            web3=web3,
            execution_model=execution_model,
            execution_context=execution_context,
            pricing_model=pricing_model,
            sync_model=sync_model,
            state=state,
            universe=strategy_universe,
            routing_model=routing_model,
            routing_state=routing_state,
            valuation_model=valuation_model,
            slippage_tolerance=0.10,
            interactive=False,
            position_id=position_id,
            unit_testing=True,
        )

    # 2. First run: requestRedeem goes on-chain, settlement pending, no crash.
    _run_close()
    sell_trades = [t for t in position.trades.values() if t.is_sell()]
    assert len(sell_trades) == 1
    assert sell_trades[0].get_status() == TradeStatus.vault_settlement_pending
    assert position.is_open()
    assert target_vault.vault_contract.functions.pendingRedeemRequest(0, owner).call() > 0

    # 3. Second run while the queue is still unsettled: the in-flight settlement
    #    is recognised and no duplicate redeem request is created.
    _run_close()
    sell_trades = [t for t in position.trades.values() if t.is_sell()]
    assert len(sell_trades) == 1
    assert sell_trades[0].get_status() == TradeStatus.vault_settlement_pending
    assert position.is_open()

    # 4. Settle the queue as asset manager.
    force_lagoon_settle(target_vault, asset_manager_address)

    # 5. Third run: the pre-flight sweep claims the redeem and closes the position.
    _run_close()
    assert sell_trades[0].get_status() == TradeStatus.success
    assert position.is_closed()
    assert position_id in state.portfolio.closed_positions

    # 6. Final equity approximately equals starting equity.
    assert state.portfolio.calculate_total_equity() == pytest.approx(starting_equity, rel=0.05)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


def test_lagoon_erc_7540_settlement_claim_restart_after_state_loss(
    asset_manager_address: str,
    target_vault,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
    tmp_path: Path,
):
    """A stale state file after a successful claim must not strand ERC-7540 settlement.

    1. Open a Lagoon ERC-7540 position and request a redeem, leaving it pending.
    2. Save the pending state as the version a restarted CLI process would reload.
    3. Settle the vault queue and claim once against the in-memory state.
    4. Reload the stale pending state and run settlement resolution again.
    5. Verify the stale state is reconciled instead of staying pending forever.
    """

    # 1. Open a Lagoon ERC-7540 position and request a redeem, leaving it pending.
    state, starting_equity, _position, sell_trade = _open_position_and_request_redeem(
        asset_manager_address,
        target_vault,
        strategy_universe,
        execution_model,
        sync_model,
        routing_model,
        pricing_model,
        vault_pair,
        base_usdc,
    )

    # 2. Save the pending state as the version a restarted CLI process would reload.
    stale_state_file = tmp_path / "stale-pending-redeem.json"
    state.write_json_file(stale_state_file)

    # 3. Settle the vault queue and claim once against the in-memory state.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1
    assert sell_trade.get_status() == TradeStatus.success

    # 4. Reload the stale pending state and run settlement resolution again.
    stale_state = State.read_json_file(stale_state_file)
    stale_sell_trade = next(
        t
        for p in stale_state.portfolio.open_positions.values()
        for t in p.trades.values()
        if t.get_status() == TradeStatus.vault_settlement_pending
    )
    resolved = execution_model.resolve_pending_vault_settlements(state=stale_state, ts=native_datetime_utc_now())

    # 5. Verify the stale state is reconciled instead of staying pending forever.
    assert len(resolved) == 1
    assert stale_sell_trade.get_status() == TradeStatus.success
    assert stale_state.portfolio.calculate_total_equity() == pytest.approx(starting_equity, rel=0.05)
    assert stale_state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


def test_lagoon_erc_7540_satellite_settlement_uses_satellite_chain_signer(
    anvil_home_chain: AnvilLaunch,
    web3: Web3,
    asset_manager_address: str,
    target_vault,
    strategy_hot_wallet: HotWallet,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
):
    """A satellite-chain ERC-7540 claim must be signed for the satellite chain.

    1. Open a Base Lagoon ERC-7540 position and request a redeem, leaving it pending.
    2. Replace the default execution tx builder with a separate home-chain Anvil builder.
    3. Attach a Web3Config-like object that maps the pending vault chain back to Base.
    4. Settle the Base vault queue and resolve the pending settlement.
    5. Verify the claim succeeds and the recorded claim transaction targets Base.
    """

    # 1. Open a Base Lagoon ERC-7540 position and request a redeem, leaving it pending.
    state, _starting_equity, _position, sell_trade = _open_position_and_request_redeem(
        asset_manager_address,
        target_vault,
        strategy_universe,
        execution_model,
        sync_model,
        routing_model,
        pricing_model,
        vault_pair,
        base_usdc,
    )

    # 2. Replace the default execution tx builder with a separate home-chain Anvil builder.
    home_web3 = create_multi_provider_web3(anvil_home_chain.json_rpc_url, default_http_timeout=(3, 250.0), retries=1)
    home_web3.eth.send_transaction({"from": home_web3.eth.accounts[0], "to": strategy_hot_wallet.address, "value": 5 * 10**18})
    execution_model.tx_builder = HotWalletTransactionBuilder(home_web3, strategy_hot_wallet)
    execution_model.tx_builder.init()

    # 3. Attach a Web3Config-like object that maps the pending vault chain back to Base.
    execution_model.web3config = _SingleChainWeb3Config(ChainId.base, web3)

    # 4. Settle the Base vault queue and resolve the pending settlement.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())

    # 5. Verify the claim succeeds and the recorded claim transaction targets Base.
    assert len(resolved) == 1
    assert sell_trade.get_status() == TradeStatus.success
    assert sell_trade.blockchain_transactions[-1].chain_id == ChainId.base.value


@pytest.mark.skipif(not JSON_RPC_ARBITRUM, reason="Set JSON_RPC_ARBITRUM to run the Lagoon satellite Safe settlement test")
@pytest.mark.timeout(900)
def test_lagoon_erc_7540_satellite_safe_settlement_uses_satellite_module(
    arbitrum_web3: Web3,
    web3: Web3,
    asset_manager_address: str,
    target_vault,
    strategy_universe: TradingStrategyUniverse,
    vault_pair: TradingPairIdentifier,
    base_usdc: AssetIdentifier,
):
    """Satellite ERC-7540 settlement must use deployed Lagoon Safe modules.

    1. Deploy a source Lagoon vault on Arbitrum and a Base satellite Safe/module.
    2. Fund the Base satellite Safe with USDC and initialise matching state reserves.
    3. Open the Base ERC-7540 vault position through the Base TradingStrategyModuleV0.
    4. Settle the Base vault queue and claim the deposit through the satellite module.
    5. Request redeem through the satellite module and leave it pending.
    6. Settle the Base vault queue and claim the redeem through the satellite module.
    7. Verify both claim transactions target Base and the Base satellite module.
    """

    # 1. Deploy a source Lagoon vault on Arbitrum and a Base satellite Safe/module.
    source_deployment, satellite_deployment, asset_manager = _deploy_source_and_satellite_lagoon(
        arbitrum_web3,
        web3,
        target_vault,
    )
    satellite_vault = satellite_deployment.vault
    satellite_module = satellite_vault.trading_strategy_module_address

    execution_model = LagoonExecution(
        vault=source_deployment.vault,
        tx_builder=LagoonTransactionBuilder(source_deployment.vault, asset_manager),
        satellite_vaults={ChainId.base.value: satellite_vault},
        mainnet_fork=True,
        confirmation_block_count=0,
    )
    execution_model.web3config = _MultiChainWeb3Config({
        ChainId.arbitrum: arbitrum_web3,
        ChainId.base: web3,
    })
    routing_model = execution_model.create_default_routing_model(strategy_universe)
    pricing_model = GenericPricing(
        EthereumPairConfigurator(
            arbitrum_web3,
            strategy_universe,
            execution_model=execution_model,
        )
    )

    # 2. Fund the Base satellite Safe with USDC and initialise matching state reserves.
    fund_erc20_on_anvil(
        web3,
        BASE_USDC_ADDRESS,
        satellite_vault.safe_address,
        int(200 * 10**6),
    )
    state = State()
    state.portfolio.initialise_reserves(base_usdc, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(base_usdc, Decimal(str(DEPOSIT_VALUE)), "Test Base satellite Safe USDC balance")

    # 3. Open the Base ERC-7540 vault position through the Base TradingStrategyModuleV0.
    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    buy_trades = pm.open_spot(vault_pair, value=DEPOSIT_VALUE)
    _execute(execution_model, routing_model, strategy_universe, state, buy_trades)
    buy_trade = buy_trades[0]
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending
    assert buy_trade.blockchain_transactions[-1].chain_id == ChainId.base.value
    assert buy_trade.blockchain_transactions[-1].contract_address.lower() == satellite_module.lower()

    # 4. Settle the Base vault queue and claim the deposit through the satellite module.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success
    deposit_claim_tx = buy_trade.blockchain_transactions[-1]
    assert deposit_claim_tx.chain_id == ChainId.base.value
    assert deposit_claim_tx.contract_address.lower() == satellite_module.lower()
    assert deposit_claim_tx.other["vault_settlement_action"] == "claim"

    # 5. Request redeem through the satellite module and leave it pending.
    pm = PositionManager(native_datetime_utc_now(), strategy_universe, state, pricing_model)
    sell_trades = pm.close_all()
    _execute(execution_model, routing_model, strategy_universe, state, sell_trades)
    sell_trade = sell_trades[0]
    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending
    assert sell_trade.blockchain_transactions[-1].chain_id == ChainId.base.value
    assert sell_trade.blockchain_transactions[-1].contract_address.lower() == satellite_module.lower()

    # 6. Settle the Base vault queue and claim the redeem through the satellite module.
    force_lagoon_settle(target_vault, asset_manager_address)
    resolved = execution_model.resolve_pending_vault_settlements(state=state, ts=native_datetime_utc_now())
    assert len(resolved) == 1

    # 7. Verify both claim transactions target Base and the Base satellite module.
    assert sell_trade.get_status() == TradeStatus.success
    redeem_claim_tx = sell_trade.blockchain_transactions[-1]
    assert redeem_claim_tx.chain_id == ChainId.base.value
    assert redeem_claim_tx.contract_address.lower() == satellite_module.lower()
    assert redeem_claim_tx.other["vault_settlement_action"] == "claim"
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)
