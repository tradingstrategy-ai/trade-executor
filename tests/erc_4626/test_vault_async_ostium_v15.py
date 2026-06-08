"""Integration test for Ostium V1.5 async vault deposit/redemption lifecycle.

Uses a real Anvil fork of Arbitrum with the Ostium V1.5 vault to test the
full async deposit → settlement → claim → redeem → settlement → claim cycle
via hot wallet, with portfolio accounting checks at every step.

Steps:
1. Initialise state with reserves, record starting equity
2. decide_trades() opens a position (vault deposit)
3. execute_trades() broadcasts requestDeposit(), marks vault_settlement_pending
4. Verify portfolio accounting: pending value included in equity
5. Force settlement on Anvil
6. Settlement retry → trade success, position open
7. decide_trades() closes position (vault redeem)
8. execute_trades() broadcasts requestWithdraw(), marks vault_settlement_pending
9. Force settlement(s) for withdrawal
10. Settlement retry → trade success, position closed
11. Verify final equity approximately equals starting equity
"""

import logging
import os

import flaky
import pytest
from web3 import Web3

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance_autodetect
from eth_defi.erc_4626.vault_protocol.gains.testing import force_ostium_v15_settlement
from eth_defi.erc_4626.vault_protocol.gains.vault import OstiumVault, OstiumVersion
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import fetch_erc20_details, USDC_NATIVE_TOKEN, USDC_WHALE
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.ethereum.vault.vault_utils import translate_vault_to_trading_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.repair import find_trades_to_be_repaired
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.account_correction import calculate_account_corrections
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
    """Create an Anvil fork of Arbitrum at a post-V1.5 block."""
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
def hot_wallet(web3) -> HotWallet:
    """A test wallet with USDC."""
    hw = HotWallet.create_for_testing(web3, test_account_n=1, eth_amount=10)
    hw.sync_nonce(web3)

    usdc = fetch_erc20_details(web3, USDC_NATIVE_TOKEN[42161])
    tx_hash = usdc.contract.functions.transfer(
        hw.address, 500 * 10**6,
    ).transact({"from": USDC_WHALE[42161], "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return hw


@pytest.fixture()
def ostium_vault(web3) -> OstiumVault:
    """The Ostium V1.5 vault instance."""
    vault = create_vault_instance_autodetect(web3, OSTIUM_VAULT_ADDRESS)
    assert isinstance(vault, OstiumVault)
    assert vault.version == OstiumVersion.v1_5
    return vault


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
    """A minimal universe containing only the Ostium vault pair."""

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
def execution_model(web3, hot_wallet) -> EthereumExecution:
    return EthereumExecution(
        HotWalletTransactionBuilder(web3, hot_wallet),
        mainnet_fork=True,
        confirmation_block_count=0,
    )


@pytest.fixture()
def sync_model(web3, hot_wallet) -> HotWalletSyncModel:
    return HotWalletSyncModel(web3, hot_wallet)


@pytest.fixture()
def routing_model(execution_model, strategy_universe) -> GenericRouting:
    return execution_model.create_default_routing_model(strategy_universe)


@pytest.fixture()
def pricing_model(web3, strategy_universe, execution_model) -> GenericPricing:
    pair_configurator = EthereumPairConfigurator(
        web3,
        strategy_universe,
        execution_model=execution_model,
    )
    return GenericPricing(pair_configurator)


@flaky.flaky
def test_ostium_v15_async_deposit_redeem_lifecycle(
    web3: Web3,
    hot_wallet: HotWallet,
    ostium_vault: OstiumVault,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    arb_usdc: AssetIdentifier,
):
    """Full async deposit/redeem lifecycle with portfolio accounting checks at each step.

    1. Initialise state with reserves, record starting equity
    2. decide_trades() opens a position (vault deposit)
    3. execute_trades() broadcasts requestDeposit(), marks vault_settlement_pending
    4. Verify portfolio accounting: pending value included in equity
    5. Force settlement on Anvil
    6. Settlement retry resolves buy, position open
    7. decide_trades() closes position (vault redeem)
    8. execute_trades() broadcasts requestWithdraw(), marks vault_settlement_pending
    9. Force settlement(s) for withdrawal
    10. Settlement retry resolves sell, position closed
    11. Verify final equity approximately equals starting equity
    """
    state = State()

    # 1. Initialise state and sync on-chain reserves
    sync_model.sync_initial(state, reserve_asset=arb_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[arb_usdc])
    reserve_position = state.portfolio.get_default_reserve_position()
    starting_cash = float(reserve_position.quantity)
    assert starting_cash == pytest.approx(500.0, abs=1.0)
    starting_equity = state.portfolio.calculate_total_equity()

    # 2. Cycle 1: decide_trades() opens position
    ts = native_datetime_utc_now()
    pm = PositionManager(ts, universe=strategy_universe, state=state, pricing_model=pricing_model, default_slippage_tolerance=0.10)
    trades = pm.open_spot(vault_pair, value=50.0)
    assert len(trades) == 1
    buy_trade = trades[0]

    # 3. execute_trades() detects async vault, calls requestDeposit() instead of deposit()
    # The vault routing layer (vault_routing.py) recognises Ostium V1.5 and routes through
    # VaultDepositManager.requestDeposit(). On success it calls state.mark_vault_settlement_pending()
    # which sets vault_settlement_pending_at timestamp and stores ticket metadata in other_data.
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    execution_model.initialize()
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending
    assert buy_trade.other_data["vault_async_flow"] is True
    assert buy_trade.other_data["vault_direction"] == "deposit"

    # 4. Verify portfolio accounting during pending state
    # get_vault_settlement_pending_value() counts planned_reserve of pending buy trades
    # so equity = cash (reduced) + pending_value (50) ≈ starting_equity
    pending_value = state.portfolio.get_vault_settlement_pending_value()
    assert pending_value == pytest.approx(50.0, abs=1.0)
    equity_while_pending = state.portfolio.calculate_total_equity()
    assert equity_while_pending == pytest.approx(starting_equity, abs=2.0)

    # 5. Force settlement on Anvil (simulates Ostium keeper running a settlement epoch)
    # In production this happens off-chain every ~24h. On Anvil we can force it.
    owner_address = buy_trade.other_data["vault_owner_address"]
    force_ostium_v15_settlement(ostium_vault, owner_address)

    # 6. Settlement retry checks on-chain status and broadcasts claimDeposit()
    # This is what the runner calls at the start of each tick to resolve pending trades.
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success
    assert buy_trade.executed_quantity > 0
    position = state.portfolio.get_position_by_id(buy_trade.position_id)
    assert position.is_open()
    # After claim, pending value drops to 0 — capital is now in the position
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0)

    # 7. Cycle 2: decide_trades() closes position (redeem vault shares)
    ts = native_datetime_utc_now()
    pm = PositionManager(ts, universe=strategy_universe, state=state, pricing_model=pricing_model, default_slippage_tolerance=0.10)
    trades = pm.close_all()
    assert len(trades) == 1
    sell_trade = trades[0]

    # 8. execute_trades() calls requestWithdraw() for async redeem
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)
    assert sell_trade.get_status() == TradeStatus.vault_settlement_pending
    assert sell_trade.other_data["vault_direction"] == "redeem"
    # Sell trades are NOT counted in pending_value (only buys lock capital)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0)

    # 9. Force settlement(s) for withdrawal
    # Withdrawals may need multiple settlements to reach the target settlement ID
    withdraw_target = ostium_vault.vault_contract.functions.targetSettlementId(False).call()
    last_id = ostium_vault.vault_contract.functions.lastSettlementId().call()
    for _ in range(max(withdraw_target - last_id, 1)):
        force_ostium_v15_settlement(ostium_vault, owner_address)

    # 10. Settlement retry broadcasts claimWithdraw() and marks trade success
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)
    assert len(resolved) == 1
    assert sell_trade.get_status() == TradeStatus.success
    assert sell_trade.executed_reserve > 0
    assert position.is_closed()

    # 11. Verify final equity ≈ starting equity (minus Ostium vault fees)
    final_equity = state.portfolio.calculate_total_equity()
    assert final_equity == pytest.approx(starting_equity, rel=0.02), \
        f"Final equity {final_equity} deviates too much from starting {starting_equity}"
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0)
    assert state.portfolio.get_in_transit_value() == pytest.approx(0.0)


@flaky.flaky
def test_ostium_v15_check_accounts_during_pending_settlement(
    web3: Web3,
    hot_wallet: HotWallet,
    ostium_vault: OstiumVault,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    arb_usdc: AssetIdentifier,
):
    """Verify check-accounts reports no mismatch when a vault deposit is pending.

    1. Initialise state with reserves
    2. Execute a vault deposit → trade enters vault_settlement_pending
    3. Run calculate_account_corrections (same logic as check-accounts CLI)
    4. Verify no accounting mismatches are reported
    5. Verify repair does not touch the pending trade
    """
    state = State()

    # 1. Initialise state
    sync_model.sync_initial(state, reserve_asset=arb_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[arb_usdc])

    # 2. Execute vault deposit → pending
    ts = native_datetime_utc_now()
    pm = PositionManager(ts, universe=strategy_universe, state=state, pricing_model=pricing_model, default_slippage_tolerance=0.10)
    trades = pm.open_spot(vault_pair, value=50.0)
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    execution_model.initialize()
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    buy_trade = trades[0]
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending

    # 3. Run check-accounts logic (same as check-accounts CLI internally)
    # Why this should show no mismatch:
    # - Reserve (USDC): state was debited at trade start, on-chain USDC was sent to vault
    #   via requestDeposit(). Both reduced by same amount → match.
    # - Position (vault token): get_quantity() returns 0 (only counts is_success() trades),
    #   on-chain vault token balance is also 0 (not yet claimed) → match.
    pair_universe = strategy_universe.data_universe.pairs
    corrections = list(calculate_account_corrections(
        pair_universe=pair_universe,
        reserve_assets=strategy_universe.reserve_assets,
        state=state,
        sync_model=sync_model,
        all_balances=True,
    ))

    # 4. No accounting mismatches — on-chain balances match state
    mismatches = [c for c in corrections if c.mismatch]
    assert len(mismatches) == 0, f"Unexpected accounting mismatches during pending settlement: {mismatches}"

    # 5. Repair does not touch pending trades
    # - find_trades_to_be_repaired() only targets is_failed() → skips vault_settlement_pending
    # - repair_zero_quantity() explicitly skips positions with vault_settlement_pending trades
    repair_trades = find_trades_to_be_repaired(state)
    assert len(repair_trades) == 0, "Repair should not target vault_settlement_pending trades"


@flaky.flaky
def test_ostium_v15_check_accounts_after_settlement(
    web3: Web3,
    hot_wallet: HotWallet,
    ostium_vault: OstiumVault,
    strategy_universe: TradingStrategyUniverse,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    pricing_model: GenericPricing,
    vault_pair: TradingPairIdentifier,
    arb_usdc: AssetIdentifier,
):
    """Verify check-accounts reports no mismatch after settlement resolves a vault deposit.

    1. Initialise state, execute vault deposit → pending
    2. Force settlement, resolve via settlement retry
    3. Run check-accounts logic
    4. Verify balances match (position now holds vault tokens)
    """
    state = State()

    # 1. Setup and deposit
    sync_model.sync_initial(state, reserve_asset=arb_usdc, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, supported_reserves=[arb_usdc])

    ts = native_datetime_utc_now()
    pm = PositionManager(ts, universe=strategy_universe, state=state, pricing_model=pricing_model, default_slippage_tolerance=0.10)
    trades = pm.open_spot(vault_pair, value=50.0)
    routing_state_details = execution_model.get_routing_state_details()
    routing_state = routing_model.create_routing_state(strategy_universe, routing_state_details)
    execution_model.initialize()
    execution_model.execute_trades(ts, state, trades, routing_model, routing_state, check_balances=True)

    buy_trade = trades[0]
    assert buy_trade.get_status() == TradeStatus.vault_settlement_pending

    # 2. Force settlement and resolve
    owner_address = buy_trade.other_data["vault_owner_address"]
    force_ostium_v15_settlement(ostium_vault, owner_address)
    resolved = check_and_resolve_vault_settlements(state=state, execution_model=execution_model)
    assert len(resolved) == 1
    assert buy_trade.get_status() == TradeStatus.success

    # 3. Run check-accounts logic
    pair_universe = strategy_universe.data_universe.pairs
    corrections = list(calculate_account_corrections(
        pair_universe=pair_universe,
        reserve_assets=strategy_universe.reserve_assets,
        state=state,
        sync_model=sync_model,
        all_balances=True,
    ))

    # 4. No mismatches — position quantity matches on-chain vault tokens
    mismatches = [c for c in corrections if c.mismatch]
    assert len(mismatches) == 0, f"Unexpected accounting mismatches after settlement: {mismatches}"

    # Position should have vault tokens
    position = state.portfolio.get_position_by_id(buy_trade.position_id)
    assert position.get_quantity() > 0
