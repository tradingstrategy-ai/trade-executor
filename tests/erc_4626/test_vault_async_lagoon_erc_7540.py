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

import flaky
import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.testing import force_lagoon_settle
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.account_correction import calculate_account_corrections
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
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="Set JSON_RPC_BASE to run this test")

#: Real ERC-7540 Lagoon vault on Base (722 capital), as used by eth_defi's Lagoon 7540 test.
LAGOON_7540_VAULT = "0xb09f761cb13baca8ec087ac476647361b6314f98"
#: The vault's real asset manager — impersonated on the fork to settle the queue.
TARGET_VAULT_ASSET_MANAGER = "0x3B95C7cD4075B72ecbC4559AF99211C2B6591b2E"
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
def web3(anvil_base_fork) -> Web3:
    return create_multi_provider_web3(anvil_base_fork.json_rpc_url, default_http_timeout=(3, 250.0), retries=1)


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
