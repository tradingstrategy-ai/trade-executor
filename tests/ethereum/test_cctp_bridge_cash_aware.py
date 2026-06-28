"""Cash-aware CCTP bridge-out sizing regression tests.

The bridge-out branch of :py:func:`inject_cctp_bridge_trades` used to size a
bridge-out purely from the alpha model's net planned buy on a satellite chain,
ignoring:

- capital already idle on the satellite chain (tracked as available bridge
  capital on the chain's bridge position), and
- whether the primary chain can actually fund the bridge-out.

This let a cross-chain backtest over-bridge and drive the primary-chain reserve
negative, crashing deep in ``simulate_bridge()`` with ``OutOfSimulatedBalance``
(surfaced as ``BacktestExecutionFailed``).

These tests pin the cash-aware behaviour:

- ``test_bridge_out_funds_from_idle_satellite_capital`` reproduces the crash
  via the simulated wallet and asserts no redundant bridge-out is injected.
- ``test_bridge_out_nets_against_partial_idle_capital`` checks only the
  shortfall is bridged.
- ``test_bridge_out_raises_clearly_when_underfunded`` checks an explicit early
  :py:class:`NotEnoughMoney` instead of a deep execution crash.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from tradeexecutor.analysis.cctp import assert_bridge_coverage
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.strategy.account_correction import calculate_total_assets
from tradeexecutor.strategy.asset import get_asset_amounts
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.portfolio import NotEnoughMoney
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType

#: Arbitrum native USDC — the portfolio reserve / primary chain
USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cc2239327c5edb3a432268e5831"

#: Base native USDC — satellite chain stablecoin
USDC_BASE_ADDRESS = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"

#: Optimism native USDC — second satellite chain stablecoin
USDC_OPTIMISM_ADDRESS = "0x0b2c639c533813f4aa9d7837caf62653d097ff85"

PRIMARY_CHAIN_ID = 42161  # Arbitrum
SATELLITE_CHAIN_ID = 8453  # Base
SECOND_SATELLITE_CHAIN_ID = 10  # Optimism

TS = datetime.datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum — the reserve currency on the primary chain."""
    return AssetIdentifier(
        chain_id=PRIMARY_CHAIN_ID,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    """USDC on Base — the satellite chain stablecoin."""
    return AssetIdentifier(
        chain_id=SATELLITE_CHAIN_ID,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_optimism() -> AssetIdentifier:
    """USDC on Optimism — another satellite chain stablecoin."""
    return AssetIdentifier(
        chain_id=SECOND_SATELLITE_CHAIN_ID,
        address=USDC_OPTIMISM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        exchange_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5d",
        internal_id=1,
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def optimism_cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_optimism: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Optimism."""
    return TradingPairIdentifier(
        base=usdc_optimism,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5e",
        exchange_address="0x28b5a0e9c621a5badaa536219b3a228c8168cf5e",
        internal_id=5,
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


@pytest.fixture()
def satellite_pair(usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """A satellite-chain spot pair on Base, quoted in native USDC.

    Priced 1:1 so reserve == quantity, which keeps the bridge-capital
    accounting easy to assert.
    """
    base = AssetIdentifier(
        chain_id=SATELLITE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000011",
        token_symbol="satBASE",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=base,
        quote=usdc_base,
        pool_address="0x0000000000000000000000000000000000000022",
        exchange_address="0x0000000000000000000000000000000000000033",
        internal_id=2,
        fee=0,
        kind=TradingPairKind.spot_market_hold,
    )


@pytest.fixture()
def primary_pair(usdc_arbitrum: AssetIdentifier) -> TradingPairIdentifier:
    """A primary-chain spot pair on Arbitrum, quoted in native USDC."""
    base = AssetIdentifier(
        chain_id=PRIMARY_CHAIN_ID,
        address="0x0000000000000000000000000000000000000077",
        token_symbol="primaryARB",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=base,
        quote=usdc_arbitrum,
        pool_address="0x0000000000000000000000000000000000000088",
        exchange_address="0x0000000000000000000000000000000000000099",
        internal_id=4,
        fee=0,
        kind=TradingPairKind.spot_market_hold,
    )


@pytest.fixture()
def satellite_vault_pair(usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """An override-only async satellite vault pair on Base."""
    share = AssetIdentifier(
        chain_id=SATELLITE_CHAIN_ID,
        address="0x0000000000000000000000000000000000000044",
        token_symbol="vBASE",
        decimals=18,
    )
    return TradingPairIdentifier(
        base=share,
        quote=usdc_base,
        pool_address="0x0000000000000000000000000000000000000055",
        exchange_address="0x0000000000000000000000000000000000000066",
        internal_id=3,
        fee=0,
        kind=TradingPairKind.vault,
        other_data={"vault_protocol": "test_async_vault"},
    )


def _make_mock_universe(pairs: list[TradingPairIdentifier]):
    """Build a mock strategy universe that yields the given pairs."""
    mock = MagicMock()
    mock.iterate_pairs.return_value = pairs
    return mock


def _make_state(usdc_arbitrum: AssetIdentifier, reserve_amount: Decimal) -> State:
    """Create a fresh state with reserves on the primary chain."""
    state = State()
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = reserve_amount
    reserve.reserve_token_price = 1.0
    return state


def _routing(wallet: SimulatedWallet, usdc_arbitrum: AssetIdentifier):
    """Minimal routing model/state for the backtest executor."""
    routing_model = BacktestRoutingIgnoredModel(reserve_token_address=usdc_arbitrum.address)
    routing_state = BacktestRoutingState(pair_universe=None, wallet=wallet)
    return routing_model, routing_state


def _establish_idle_satellite_capital(
    state: State,
    execution: BacktestExecution,
    wallet: SimulatedWallet,
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    amount: Decimal,
):
    """Bridge ``amount`` from primary to satellite and settle it.

    Leaves the satellite chain holding ``amount`` idle USDC, represented as an
    open bridge position with ``amount`` available bridge capital.
    """
    _, bridge_out, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=cctp_pair,
        quantity=None,
        reserve=amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[bridge_out],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    assert bridge_out.is_success()


def _create_satellite_buy(
    state: State,
    satellite_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    reserve: Decimal,
) -> TradeExecution:
    """Create (but do not execute) a satellite-chain spot buy."""
    _, buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=None,
        reserve=reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    return buy


def _create_primary_buy(
    state: State,
    primary_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
    reserve: Decimal,
) -> TradeExecution:
    """Create (but do not execute) a primary-chain spot buy."""
    _, buy, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=primary_pair,
        quantity=None,
        reserve=reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    return buy


def _async_vault_execution(wallet: SimulatedWallet, satellite_vault_pair: TradingPairIdentifier) -> BacktestExecution:
    """Create a backtest executor that treats ``satellite_vault_pair`` as async."""
    return BacktestExecution(
        wallet=wallet,
        vault_settlement_delay_overrides={
            satellite_vault_pair.pool_address: datetime.timedelta(days=2),
        },
    )


def test_bridge_out_funds_from_idle_satellite_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Idle satellite capital funds the buy — no redundant bridge-out, no crash.

    1. Seed 25_000 primary reserve and bridge 20_000 to the satellite, leaving
       5_000 primary reserve and 20_000 idle on the satellite.
    2. Ask for an 11_500 satellite buy and inject bridge trades.
    3. Execute the result.

    On the buggy code a second 11_500 bridge-out is injected, which drains the
    5_000 primary reserve and raises ``BacktestExecutionFailed`` during step 3.
    With the fix, no bridge-out is injected; the buy allocates from the idle
    bridge capital and the primary reserve is untouched.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Park 20_000 idle on the satellite; primary reserve falls to 5_000.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(20_000))
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity == Decimal(5_000)

    # 2. Inject bridge trades for an 11_500 satellite buy.
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Execute — must not raise (reproduces the crash on buggy code).
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )

    # No redundant bridge-out was injected.
    assert [t for t in result if t.pair.is_cctp_bridge()] == []
    # The buy executed and was funded from the idle bridge capital.
    assert buy.is_success()
    assert reserve.quantity == Decimal(5_000)
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.bridge_capital_allocated == Decimal(11_500)


def test_bridge_out_uses_bridge_pair_decimals_when_reserve_asset_decimals_are_wrong(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A generated 6-decimal CCTP pair works with a wrong 18-decimal reserve asset.

    This covers vault-only universes where the reserve asset can inherit
    fallback 18-decimal metadata, while generated native USDC bridge pairs are
    correctly normalised to 6 decimals.

    1. Create a reserve asset with the same primary USDC address but wrong 18-decimal metadata.
    2. Plan a satellite buy that requires bridge-out funding with sub-raw-unit dust.
    3. Inject CCTP bridge trades.
    4. Assert the bridge-out is created and floored with bridge-pair USDC precision.
    """

    # 1. Create a reserve asset with the same primary USDC address but wrong 18-decimal metadata.
    wrong_decimals_reserve = AssetIdentifier(
        chain_id=usdc_arbitrum.chain_id,
        address=usdc_arbitrum.address,
        token_symbol=usdc_arbitrum.token_symbol,
        decimals=18,
    )
    state = _make_state(wrong_decimals_reserve, Decimal(2_000))

    # 2. Plan a satellite buy that requires bridge-out funding with sub-raw-unit dust.
    buy = _create_satellite_buy(
        state,
        satellite_pair,
        wrong_decimals_reserve,
        Decimal("1000.0000009"),
    )
    universe = _make_mock_universe([cctp_pair, satellite_pair])

    # 3. Inject CCTP bridge trades.
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=wrong_decimals_reserve,
    )

    # 4. Assert the bridge-out is created and floored with bridge-pair USDC precision.
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].planned_reserve == Decimal("1000")


def test_bridge_out_nets_against_partial_idle_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Only the shortfall above idle satellite capital is bridged out.

    Idle 4_000 on the satellite, an 11_500 buy, ample primary reserve -> a
    single 7_500 bridge-out (= 11_500 - 4_000).
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)

    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(4_000))

    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    bridge_out = bridge_trades[0]
    assert bridge_out.is_buy()
    assert bridge_out.planned_reserve == Decimal(7_500)
    assert bridge_out.pair.get_destination_chain_id() == SATELLITE_CHAIN_ID


@pytest.mark.timeout(300)
def test_bridge_out_nets_against_same_cycle_satellite_spot_sell(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """A synchronous satellite spot sell reduces the same-cycle vault deposit bridge-out.

    1. Bridge 500 USDC to Base and deploy it into a satellite spot position,
       leaving only 500 USDC on the primary chain.
    2. In one later cycle, plan a non-closing 500 USDC spot sell and a 1_000
       USDC async satellite vault deposit.
    3. Inject CCTP trades and assert only the missing 500 USDC is bridged out.
    4. Execute the cycle and assert the vault request consumes the sell proceeds
       and bridge-out proceeds without leaving extra deployable Base USDC.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(1_000))
    state = _make_state(usdc_arbitrum, Decimal(1_000))
    execution = _async_vault_execution(wallet, satellite_vault_pair)

    # 1. Bridge 500 USDC to Base and deploy it into a satellite spot position.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(500))
    spot_buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(500))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[spot_buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    reserve = state.portfolio.get_default_reserve_position()
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert reserve.quantity == Decimal(500)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)

    # 2. Plan a non-closing spot sell and a larger async satellite vault deposit.
    spot_position = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, spot_sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-spot_position.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=spot_position,
        closing=False,
    )
    vault_buy = _create_satellite_buy(state, satellite_vault_pair, usdc_arbitrum, Decimal(1_000))
    universe = _make_mock_universe([cctp_pair, satellite_pair, satellite_vault_pair])

    # 3. Inject CCTP trades and assert only the missing 500 USDC is bridged out.
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[spot_sell, vault_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].is_buy()
    assert bridge_trades[0].planned_reserve == Decimal(500)

    # 4. Execute the cycle and assert no extra deployable Base USDC remains.
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    assert vault_buy.get_status() == TradeStatus.vault_settlement_pending
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(1_000.0, abs=1e-6)


def test_bridge_out_raises_clearly_when_underfunded(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A bridge-out that exceeds fundable primary reserve raises NotEnoughMoney.

    No idle satellite capital, an 11_500 buy, only 5_000 fundable primary
    reserve -> an explicit early :py:class:`NotEnoughMoney`, not a deep
    ``BacktestExecutionFailed``/``OutOfSimulatedBalance`` during execution.
    """
    state = _make_state(usdc_arbitrum, Decimal(5_000))

    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(11_500))
    universe = _make_mock_universe([cctp_pair, satellite_pair])

    with pytest.raises(NotEnoughMoney):
        inject_cctp_bridge_trades(
            state=state,
            trades=[buy],
            strategy_universe=universe,
            primary_chain_id=PRIMARY_CHAIN_ID,
            ts=TS,
            reserve_asset=usdc_arbitrum,
        )


def test_bridge_back_capped_to_available_bridge_capital(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A bridge-back is capped when a satellite sell executes after bridge-backs.

    Non-closing spot sells sort after CCTP bridge-backs. Their proceeds are not
    available for same-cycle bridge-back sizing, so only the currently idle
    bridge capital may be bridged back.

    1. Bridge 10_000 to the satellite, then deploy 7_000 into a satellite buy,
       leaving 3_000 available bridge capital.
    2. Ask to sell the whole 7_000 position as a non-closing late sell.
    3. Assert the injected bridge-back is capped to 3_000, not 7_000.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Park 10_000 on the satellite, deploy 7_000 -> 3_000 available.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(7_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.get_available_bridge_capital() == Decimal(3_000)

    # 2. Net sell of the whole 7_000 satellite position as a non-closing late sell.
    sat_pos = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-sat_pos.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=sat_pos,
        closing=False,
    )
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Bridge-back capped to the 3_000 available, not the full 7_000 net sell.
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].is_sell()
    assert bridge_backs[0].planned_quantity == Decimal(-3_000)


@pytest.mark.timeout(300)
def test_closing_satellite_sell_can_fund_same_cycle_bridge_back(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A closing satellite sell can be bridged back in the same cycle.

    1. Bridge 10_000 to the satellite, then deploy 7_000 into a satellite buy,
       leaving 3_000 available bridge capital.
    2. Ask to close the whole 7_000 satellite position.
    3. Assert the injected bridge-back includes the 7_000 closing sell proceeds.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(50_000))
    state = _make_state(usdc_arbitrum, Decimal(50_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Park 10_000 on the satellite, deploy 7_000 -> 3_000 available.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(7_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=False,
    )
    bridge_pos = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_pos.get_available_bridge_capital() == Decimal(3_000)

    # 2. Close the whole 7_000 satellite position.
    sat_pos = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-sat_pos.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=sat_pos,
        closing=True,
    )
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 3. Bridge-back includes the 7_000 closing sell proceeds.
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].is_sell()
    assert bridge_backs[0].planned_quantity == Decimal(-7_000)


@pytest.mark.timeout(300)
def test_satellite_sell_reopens_closed_bridge_position(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """A satellite sell reopens a closed bridge position for returned USDC.

    1. Bridge 1_000 USDC to Base and deploy it all into a satellite position.
    2. Close the now-zero-available bridge position to mimic earlier accounting.
    3. Sell 250 of the satellite position.
    4. Assert the sell credits Base USDC and reopens the bridge position with
       250 available bridge capital.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(2_000))
    state = _make_state(usdc_arbitrum, Decimal(2_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 1_000 USDC to Base and deploy it all into a satellite position.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(1_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(1_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)

    # 2. Close the now-zero-available bridge position to mimic earlier accounting.
    state.portfolio.close_position(bridge_position, TS)
    assert state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID) is None

    # 3. Sell 250 of the satellite position.
    satellite_position = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=Decimal(-250),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=satellite_position,
        closing=False,
    )
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[sell],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert the bridge position was reopened for the returned Base USDC.
    reopened_bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert sell.is_success()
    assert wallet.get_balance(usdc_base) == Decimal(250)
    assert reopened_bridge_position is bridge_position
    assert reopened_bridge_position.get_available_bridge_capital() == Decimal(250)
    assert calculate_total_assets(state.portfolio)[usdc_base] == Decimal(250)


@pytest.mark.timeout(300)
def test_primary_buy_funded_from_idle_satellite_bridge_capital(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    primary_pair: TradingPairIdentifier,
):
    """Idle satellite bridge capital funds a primary-chain buy shortfall.

    1. Bridge 5_000 USDC to Base, leaving no primary-chain reserve.
    2. Plan a 4_000 primary-chain buy and inject CCTP trades.
    3. Execute the bridge-back and primary buy.
    4. Assert only the missing primary amount was bridged back and the buy did
       not drive the primary USDC wallet negative.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(5_000))
    state = _make_state(usdc_arbitrum, Decimal(5_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 5_000 USDC to Base, leaving no primary-chain reserve.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(5_000))
    reserve = state.portfolio.get_default_reserve_position()
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert reserve.quantity == Decimal(0)
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(5_000)

    # 2. Plan a 4_000 primary-chain buy and inject CCTP trades.
    buy = _create_primary_buy(state, primary_pair, usdc_arbitrum, Decimal(4_000))
    universe = _make_mock_universe([cctp_pair, primary_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].is_sell()
    assert bridge_trades[0].planned_quantity == Decimal(-4_000)

    # 3. Execute the bridge-back and primary buy.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert request-time wallet and bridge-capital accounting.
    assert buy.is_success()
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)
    assert wallet.get_balance(usdc_base) == Decimal(1_000)
    assert reserve.quantity == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(1_000)


@pytest.mark.timeout(300)
def test_cross_satellite_bridge_out_funded_from_other_idle_satellite_capital(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    usdc_optimism: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    optimism_cctp_pair: TradingPairIdentifier,
    primary_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Idle capital on one satellite can fund another satellite's bridge-out.

    1. Bridge 8_000 USDC to Optimism, leaving 2_000 primary-chain reserve.
    2. Plan a 3_000 primary buy and a 6_000 Base buy.
    3. Inject CCTP trades.
    4. Execute all trades.
    5. Assert Optimism bridged back 7_000, Base bridged out 6_000, and both
       buys executed without a primary wallet underflow.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(10_000))
    state = _make_state(usdc_arbitrum, Decimal(10_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 8_000 USDC to Optimism, leaving 2_000 primary-chain reserve.
    _establish_idle_satellite_capital(state, execution, wallet, optimism_cctp_pair, usdc_arbitrum, Decimal(8_000))
    reserve = state.portfolio.get_default_reserve_position()
    optimism_bridge_position = state.portfolio.get_bridge_position_for_chain(SECOND_SATELLITE_CHAIN_ID)
    assert reserve.quantity == Decimal(2_000)
    assert optimism_bridge_position.get_available_bridge_capital() == Decimal(8_000)

    # 2. Plan a 3_000 primary buy and a 6_000 Base buy.
    primary_buy = _create_primary_buy(state, primary_pair, usdc_arbitrum, Decimal(3_000))
    base_buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(6_000))

    # 3. Inject CCTP trades.
    universe = _make_mock_universe([cctp_pair, optimism_cctp_pair, primary_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[primary_buy, base_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    bridge_backs = [t for t in bridge_trades if t.is_sell()]
    bridge_outs = [t for t in bridge_trades if t.is_buy()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].pair.get_destination_chain_id() == SECOND_SATELLITE_CHAIN_ID
    assert bridge_backs[0].planned_quantity == Decimal(-7_000)
    assert len(bridge_outs) == 1
    assert bridge_outs[0].pair.get_destination_chain_id() == SATELLITE_CHAIN_ID
    assert bridge_outs[0].planned_reserve == Decimal(6_000)

    # 4. Execute all trades.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 5. Assert bridge capital and wallet accounting.
    base_bridge_position = state.portfolio.get_position_by_id(bridge_outs[0].position_id)
    assert primary_buy.is_success()
    assert base_buy.is_success()
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert wallet.get_balance(usdc_optimism) == Decimal(1_000)
    assert reserve.quantity == Decimal(0)
    assert base_bridge_position.get_available_bridge_capital() == Decimal(0)
    assert optimism_bridge_position.get_available_bridge_capital() == Decimal(1_000)


@pytest.mark.timeout(300)
def test_dust_available_bridge_capital_does_not_create_bridge_back(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
):
    """Sub-raw-unit available bridge capital is ignored instead of creating a bridge-back.

    1. Bridge 1_000 USDC to Base and deploy it all into a satellite position.
    2. Leave only sub-raw-unit available bridge capital to mimic Decimal dust.
    3. Plan a non-closing satellite sell whose proceeds sort after bridge-backs.
    4. Assert the planner skips the dust bridge-back trade.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(2_000))
    state = _make_state(usdc_arbitrum, Decimal(2_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 1_000 USDC to Base and deploy it all into a satellite position.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(1_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(1_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 2. Leave only sub-raw-unit available bridge capital to mimic Decimal dust.
    dust = Decimal("0.000000000000712654913188")
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    bridge_position.bridge_capital_allocated = bridge_position.get_quantity() - dust
    assert bridge_position.get_available_bridge_capital() == dust

    # 3. Plan a non-closing satellite sell whose proceeds sort after bridge-backs.
    satellite_position = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-satellite_position.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=satellite_position,
        closing=False,
    )
    universe = _make_mock_universe([cctp_pair, satellite_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert the planner skips the dust bridge-back trade.
    assert [t for t in result if t.pair.is_cctp_bridge()] == []


@pytest.mark.timeout(300)
def test_primary_shortfall_dust_after_bridge_back_is_ignored(
    usdc_base: AssetIdentifier,
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    primary_pair: TradingPairIdentifier,
):
    """Sub-raw-unit primary shortfall dust after bridge-back does not fail execution.

    1. Bridge 1_000 USDC to Base, leaving no primary reserve.
    2. Plan a primary buy whose reserve need is 1_000 plus sub-raw-unit dust.
    3. Inject CCTP trades.
    4. Execute the bridge-back and primary buy.
    5. Assert the planner bridged back the real 1_000 amount and the buy was
       snapped to the actually available raw-token amount.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(1_000))
    state = _make_state(usdc_arbitrum, Decimal(1_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 1_000 USDC to Base, leaving no primary reserve.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(1_000))
    reserve = state.portfolio.get_default_reserve_position()
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert reserve.quantity == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(1_000)

    # 2. Plan a primary buy whose reserve need is 1_000 plus sub-raw-unit dust.
    dust = Decimal("0.000000000000712654913188")
    buy = _create_primary_buy(
        state,
        primary_pair,
        usdc_arbitrum,
        Decimal(1_000) + dust,
    )

    # 3. Inject CCTP trades.
    universe = _make_mock_universe([cctp_pair, primary_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Execute the bridge-back and primary buy.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 5. Assert the bridge-back and buy used the raw-token amount.
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge() and t.is_sell()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-1_000)
    assert buy.is_success()
    assert buy.planned_reserve == Decimal(1_000)
    assert buy.reserve_currency_allocated == Decimal(0)
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert reserve.quantity == Decimal(0)


@pytest.mark.timeout(300)
def test_closing_satellite_sell_can_fund_same_cycle_primary_buy(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
    primary_pair: TradingPairIdentifier,
):
    """A closing satellite sell can bridge back and fund a primary-chain buy.

    1. Bridge 10_000 USDC to Base and deploy it all into a satellite position,
       leaving no primary reserve.
    2. In one cycle, close the satellite position and plan a 6_000 primary buy.
    3. Inject and execute CCTP trades.
    4. Assert the primary buy succeeds from the bridge-back proceeds.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(10_000))
    state = _make_state(usdc_arbitrum, Decimal(10_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 10_000 USDC to Base and deploy it all into a satellite position.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(10_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity == Decimal(0)
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)

    # 2. In one cycle, close the satellite position and plan a 6_000 primary buy.
    satellite_position = state.portfolio.get_open_position_for_pair(satellite_pair)
    _, sell, _ = state.create_trade(
        strategy_cycle_at=TS,
        pair=satellite_pair,
        quantity=-satellite_position.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=satellite_position,
        closing=True,
    )
    primary_buy = _create_primary_buy(state, primary_pair, usdc_arbitrum, Decimal(6_000))

    # 3. Inject and execute CCTP trades.
    universe = _make_mock_universe([cctp_pair, satellite_pair, primary_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[sell, primary_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_backs = [t for t in result if t.pair.is_cctp_bridge() and t.is_sell()]
    assert len(bridge_backs) == 1
    assert bridge_backs[0].planned_quantity == Decimal(-10_000)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert the primary buy succeeds from the bridge-back proceeds.
    assert sell.is_success()
    assert primary_buy.is_success()
    assert wallet.get_balance(usdc_arbitrum) == Decimal(4_000)
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert reserve.quantity == Decimal(4_000)


def test_assert_bridge_coverage_allows_satellite_vault_without_same_cycle_bridge(
    usdc_arbitrum: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """Bridge coverage analysis allows idle-capital satellite vault cycles.

    1. Bridge 5_000 USDC to Base in one cycle.
    2. Plan a later Base vault deposit that reuses the idle bridge capital.
    3. Run the notebook bridge coverage assertion.
    4. Assert the later vault cycle is flagged as having no same-cycle bridge,
       but the assertion still passes because the run has CCTP coverage.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(10_000))
    state = _make_state(usdc_arbitrum, Decimal(10_000))
    execution = BacktestExecution(wallet=wallet)

    # 1. Bridge 5_000 USDC to Base in one cycle.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(5_000))

    # 2. Plan a later Base vault deposit that reuses the idle bridge capital.
    later_ts = TS + datetime.timedelta(days=1)
    state.create_trade(
        strategy_cycle_at=later_ts,
        pair=satellite_vault_pair,
        quantity=None,
        reserve=Decimal(3_000),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )

    # 3. Run the notebook bridge coverage assertion.
    cycle_df = assert_bridge_coverage(
        state.portfolio.get_all_trades(),
        primary_chain_id=PRIMARY_CHAIN_ID,
    )

    # 4. Assert the later vault cycle is flagged as having no same-cycle bridge.
    later_cycle = cycle_df.loc[cycle_df["cycle"] == later_ts].iloc[0]
    assert later_cycle["bridge_trades"] == 0
    assert later_cycle["satellite_vault_trades"] == 1
    assert bool(later_cycle["satellite_vaults_without_same_cycle_bridge"]) is True


@pytest.mark.timeout(300)
def test_async_vault_deposit_uses_idle_satellite_bridge_capital(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """An async satellite vault deposit consumes only missing bridge capital.

    1. Bridge 12_000 USDC to Base and leave it idle in the bridge position.
    2. Plan a 10_000 Base vault deposit and inject CCTP trades.
    3. Execute the deposit request.
    4. Assert no redundant bridge-out was injected and the request debited the
       simulated Base USDC wallet immediately, leaving only idle bridge capital.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = _async_vault_execution(wallet, satellite_vault_pair)

    # 1. Bridge 12_000 USDC to Base and leave it idle in the bridge position.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(12_000))
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert bridge_position.get_available_bridge_capital() == Decimal(12_000)

    # 2. Plan a 10_000 Base vault deposit and inject CCTP trades.
    buy = _create_satellite_buy(state, satellite_vault_pair, usdc_arbitrum, Decimal(10_000))
    universe = _make_mock_universe([cctp_pair, satellite_vault_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    assert [t for t in result if t.pair.is_cctp_bridge()] == []

    # 3. Execute the deposit request.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=result,
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert request-time wallet and bridge-capital accounting.
    assert buy.get_status() == TradeStatus.vault_settlement_pending
    assert wallet.get_balance(usdc_base) == Decimal(2_000)
    assert wallet.get_balance(satellite_vault_pair.base) == Decimal(0)
    assert bridge_position.bridge_capital_allocated == Decimal(10_000)
    assert bridge_position.get_available_bridge_capital() == Decimal(2_000)
    assert calculate_total_assets(state.portfolio)[usdc_base] == Decimal(2_000)
    assert dict(get_asset_amounts(bridge_position))[usdc_base] == Decimal(2_000)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(10_000.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_async_vault_deposit_bridges_only_missing_satellite_capital(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """An async satellite vault deposit bridges the missing amount, then queues it.

    1. Start with primary reserve only and plan a 10_000 Base vault deposit.
    2. Inject CCTP trades.
    3. Execute the bridge-out and vault deposit request in one batch.
    4. Assert the bridge-out equals the missing amount and the deposit request
       leaves no deployable Base USDC in the simulated wallet.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(25_000))
    state = _make_state(usdc_arbitrum, Decimal(25_000))
    execution = _async_vault_execution(wallet, satellite_vault_pair)

    # 1. Start with primary reserve only and plan a 10_000 Base vault deposit.
    buy = _create_satellite_buy(state, satellite_vault_pair, usdc_arbitrum, Decimal(10_000))
    universe = _make_mock_universe([cctp_pair, satellite_vault_pair])

    # 2. Inject CCTP trades.
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].is_buy()
    assert bridge_trades[0].planned_reserve == Decimal(10_000)

    # 3. Execute the bridge-out and vault deposit request in one batch.
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert request-time wallet and bridge-capital accounting.
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert buy.get_status() == TradeStatus.vault_settlement_pending
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert wallet.get_balance(satellite_vault_pair.base) == Decimal(0)
    assert bridge_position.bridge_capital_allocated == Decimal(10_000)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)
    assert usdc_base not in calculate_total_assets(state.portfolio)
    assert get_asset_amounts(bridge_position) == []
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(10_000.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_async_vault_deposit_snaps_sub_raw_unit_bridge_shortfall(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """An async vault deposit with sub-raw-unit dust debits only deployable reserve.

    1. Start with exactly 1_000 primary-chain USDC reserve.
    2. Plan a Base vault deposit for 1_000 USDC plus sub-raw-unit dust.
    3. Inject CCTP trades and execute the bridge-out and deposit request.
    4. Assert the bridge-out, allocation and pending deposit request all use
       1_000 USDC and the Base reserve wallet is debited immediately.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(1_000))
    state = _make_state(usdc_arbitrum, Decimal(1_000))
    execution = _async_vault_execution(wallet, satellite_vault_pair)

    # 1. Start with exactly 1_000 primary-chain USDC reserve.
    reserve = state.portfolio.get_default_reserve_position()
    assert reserve.quantity == Decimal(1_000)

    # 2. Plan a Base vault deposit for 1_000 USDC plus sub-raw-unit dust.
    dust = Decimal("0.000000000000712654913188")
    buy = _create_satellite_buy(state, satellite_vault_pair, usdc_arbitrum, Decimal(1_000) + dust)
    universe = _make_mock_universe([cctp_pair, satellite_vault_pair])

    # 3. Inject CCTP trades and execute the bridge-out and deposit request.
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=TS,
        reserve_asset=usdc_arbitrum,
    )
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].is_buy()
    assert bridge_trades[0].planned_reserve == Decimal(1_000)

    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 4. Assert allocation and request-time wallet accounting use the deployable amount.
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert buy.get_status() == TradeStatus.vault_settlement_pending
    assert buy.planned_reserve == Decimal(1_000)
    assert buy.bridge_currency_allocated == Decimal(1_000)
    assert buy.get_vault_settlement_request_reserve() == Decimal(1_000)
    assert wallet.get_balance(usdc_arbitrum) == Decimal(0)
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert wallet.get_balance(satellite_vault_pair.base) == Decimal(0)
    assert reserve.quantity == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(1_000.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_async_vault_redeem_does_not_fund_same_cycle_satellite_buy(
    usdc_arbitrum: AssetIdentifier,
    usdc_base: AssetIdentifier,
    cctp_pair: TradingPairIdentifier,
    satellite_pair: TradingPairIdentifier,
    satellite_vault_pair: TradingPairIdentifier,
):
    """An async vault redeem cannot fund another satellite buy in the same cycle.

    1. Bridge 10_000 USDC to Base, request an 8_000 async vault deposit and
       settle it to establish an override-only async vault position.
    2. In one later cycle, request a full async redeem and a 6_000 Base spot buy.
    3. Inject CCTP trades.
    4. Assert the planner ignores the async redeem proceeds and bridges only the
       4_000 shortfall above the 2_000 idle Base bridge capital.
    5. Execute the cycle and assert the redeem request debited wallet shares
       immediately while no Base USDC proceeds appeared yet.
    """
    wallet = SimulatedWallet()
    wallet.set_balance(usdc_arbitrum, Decimal(30_000))
    state = _make_state(usdc_arbitrum, Decimal(30_000))
    execution = _async_vault_execution(wallet, satellite_vault_pair)

    # 1. Bridge 10_000 USDC to Base, request an 8_000 async vault deposit and settle it.
    _establish_idle_satellite_capital(state, execution, wallet, cctp_pair, usdc_arbitrum, Decimal(10_000))
    vault_buy = _create_satellite_buy(state, satellite_vault_pair, usdc_arbitrum, Decimal(8_000))
    routing_model, routing_state = _routing(wallet, usdc_arbitrum)
    execution.execute_trades(
        ts=TS,
        state=state,
        trades=[vault_buy],
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    execution.resolve_pending_vault_settlements(
        state=state,
        ts=TS + datetime.timedelta(days=3),
        pricing_model=None,
    )
    vault_position = state.portfolio.get_open_position_for_pair(satellite_vault_pair)
    bridge_position = state.portfolio.get_bridge_position_for_chain(SATELLITE_CHAIN_ID)
    assert vault_buy.is_success()
    assert vault_position.has_async_vault_flow()
    assert wallet.get_balance(satellite_vault_pair.base) == Decimal(8_000)
    assert bridge_position.get_available_bridge_capital() == Decimal(2_000)

    # 2. In one later cycle, request a full async redeem and a 6_000 Base spot buy.
    cycle_ts = TS + datetime.timedelta(days=4)
    _, vault_sell, _ = state.create_trade(
        strategy_cycle_at=cycle_ts,
        pair=satellite_vault_pair,
        quantity=-vault_position.get_quantity(),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=vault_position,
        closing=True,
    )
    spot_buy = _create_satellite_buy(state, satellite_pair, usdc_arbitrum, Decimal(6_000))

    # 3. Inject CCTP trades.
    universe = _make_mock_universe([cctp_pair, satellite_pair, satellite_vault_pair])
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[vault_sell, spot_buy],
        strategy_universe=universe,
        primary_chain_id=PRIMARY_CHAIN_ID,
        ts=cycle_ts,
        reserve_asset=usdc_arbitrum,
    )

    # 4. Assert the planner ignores the async redeem proceeds.
    bridge_trades = [t for t in result if t.pair.is_cctp_bridge()]
    assert len(bridge_trades) == 1
    assert bridge_trades[0].is_buy()
    assert bridge_trades[0].planned_reserve == Decimal(4_000)

    # 5. Execute the cycle and assert request-time redeem accounting.
    execution.execute_trades(
        ts=cycle_ts,
        state=state,
        trades=sorted(result, key=lambda t: t.get_execution_sort_position()),
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )
    assert vault_sell.get_status() == TradeStatus.vault_settlement_pending
    assert wallet.get_balance(satellite_vault_pair.base) == Decimal(0)
    assert wallet.get_balance(usdc_base) == Decimal(0)
    assert bridge_position.get_available_bridge_capital() == Decimal(0)
    assert dict(get_asset_amounts(vault_position))[satellite_vault_pair.base] == Decimal(0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)
