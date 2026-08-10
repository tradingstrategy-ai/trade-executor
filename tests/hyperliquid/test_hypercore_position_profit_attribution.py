"""Realised plus unrealised profit must equal what a Hypercore vault position actually made.

Regression tests for two defects in profit attribution for positions that value themselves with
the internal share-price model - :py:meth:`TradingPosition.is_using_internal_share_price_profit`.

Closed positions reported unrealised profit: ``get_unrealised_profit_usd()`` returned the
share-price model's whole-position profit regardless of whether anything was still held, so any
caller summing realised + unrealised counted a closed position's entire P&L twice.

Open positions mixed two accounting bases: realised came from lifetime average cost while
unrealised came from the share-price model, which measures profit on the currently outstanding
internal supply. Those do not complement each other once a position has been partially sold and
rebought at other prices, and the drift accumulated with every additional trade. A backtest
position traded 59 times reported 13,028 USD of profit against 6,185 USD implied by its own cash
flows, which made position-level P&L unusable for exactly the heavily traded positions a
rebalancing strategy produces.

Ground truth throughout is the position's cash flows - proceeds, plus holdings, minus cost -
which is definitional and model-independent.
"""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType


@pytest.fixture()
def pair() -> TradingPairIdentifier:
    """A Hypercore vault pair, homed on chain 9999 so ``is_hyperliquid_vault()`` matches."""
    base = AssetIdentifier(
        chain_id=9999,
        address="0x4dec0a851849056e259128464ef28ce78afa27f6",
        token_symbol="pmalt",
        decimals=6,
    )
    quote = AssetIdentifier(
        chain_id=9999,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address="0x4dec0a851849056e259128464ef28ce78afa27f6",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
    )


@pytest.fixture()
def state(pair: TradingPairIdentifier) -> State:
    """Funded portfolio to trade the vault pair in."""
    state = State()
    state.portfolio.initialise_reserves(pair.quote, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(pair.quote, Decimal("1_000_000"))
    return state


def execute(state: State, pair: TradingPairIdentifier, quantity, price: float, at: datetime.datetime) -> TradingPosition:
    """Drive one trade through the real trade path. Negative quantity sells."""
    quantity = Decimal(str(quantity))
    position, trade, _created = state.create_trade(
        strategy_cycle_at=at,
        pair=pair,
        quantity=quantity,
        reserve=None,
        assumed_price=price,
        trade_type=TradeType.rebalance,
        reserve_currency=pair.quote,
        reserve_currency_price=1.0,
    )
    state.start_execution(at, trade)
    trade.mark_broadcasted(at)
    state.mark_trade_success(
        executed_at=at,
        trade=trade,
        executed_price=price,
        executed_amount=quantity,
        executed_reserve=abs(quantity) * Decimal(str(price)),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.last_token_price = price
    return position


def cash_flow_profit(position: TradingPosition) -> float:
    """Ground truth: sell proceeds, plus the value still held, minus what was paid."""
    bought = sum(
        abs(float(t.executed_reserve or 0))
        for t in position.trades.values()
        if t.is_buy() and t.is_success()
    )
    sold = sum(
        abs(float(t.executed_reserve or 0))
        for t in position.trades.values()
        if t.is_sell() and t.is_success()
    )
    held = float(position.get_value() or 0.0) if position.is_open() else 0.0
    return sold + held - bought


def test_profit_attribution_across_a_position_lifecycle(state: State, pair: TradingPairIdentifier):
    """Realised and unrealised profit must sum to the truth at every stage of a position's life.

    Average-cost accounting and the share-price model agree while a position is only bought and
    then partially sold once. They diverge as soon as it is rebought at a different price, which
    is what a rebalancing strategy does continuously, so the lifecycle is walked in order.

    1. Buy and mark up - all profit is unrealised, nothing is realised
    2. Sell half - realised and unrealised must still sum to the cash-flow profit
    3. Rebuy higher and sell again, so average cost and share price diverge
    4. Close the position - unrealised must be zero and total profit must equal the truth
    """
    start = datetime.datetime(2026, 1, 1)
    day = datetime.timedelta(days=1)

    # 1. Buy and mark up - all profit is unrealised, nothing is realised
    position = execute(state, pair, 200, 10.0, start)
    position.last_token_price = 12.0
    assert cash_flow_profit(position) == pytest.approx(400.0)
    assert (position.get_realised_profit_usd() or 0.0) == pytest.approx(0.0)
    assert position.get_unrealised_profit_usd() == pytest.approx(400.0)
    assert position.get_total_profit_usd() == pytest.approx(400.0)

    # 2. Sell half - realised and unrealised must still sum to the cash-flow profit
    position = execute(state, pair, -100, 12.0, start + 5 * day)
    truth = cash_flow_profit(position)
    assert truth == pytest.approx(400.0)
    assert position.get_realised_profit_usd() == pytest.approx(200.0)
    assert position.get_unrealised_profit_usd() == pytest.approx(200.0)

    # 3. Rebuy higher and sell again, so average cost and share price diverge
    position = execute(state, pair, 100, 20.0, start + 10 * day)
    position = execute(state, pair, -100, 18.0, start + 15 * day)
    position = execute(state, pair, 150, 15.0, start + 20 * day)
    position.last_token_price = 16.0
    truth = cash_flow_profit(position)
    assert truth == pytest.approx(750.0)
    realised = position.get_realised_profit_usd() or 0.0
    unrealised = position.get_unrealised_profit_usd() or 0.0
    assert realised + unrealised == pytest.approx(truth, abs=0.01)
    assert position.get_total_profit_usd() == pytest.approx(truth, abs=0.01)

    # 4. Close the position - unrealised must be zero and total profit must equal the truth
    position = execute(state, pair, -250, 16.0, start + 25 * day)
    assert not position.is_open()
    assert float(position.get_quantity()) == pytest.approx(0.0)
    assert position.get_unrealised_profit_usd() == pytest.approx(0.0)
    assert position.get_realised_profit_usd() == pytest.approx(750.0, abs=0.01)
    assert position.get_total_profit_usd() == pytest.approx(750.0, abs=0.01)


def test_attribution_error_does_not_grow_with_trade_count(state: State, pair: TradingPairIdentifier):
    """A heavily traded position must not accumulate attribution error.

    This is the property that mattered in practice. The old code drifted a little on every round
    trip, so a vault traded three times looked fine while one traded fifty-nine times was wrong by
    thousands of dollars - and an equity-versus-P&L audit built on it rejected high-turnover
    strategy configurations as though their books were broken.

    1. Trade a position twenty-five times, alternating trims and top-ups at rising prices
    2. Assert realised plus unrealised still equals the cash-flow profit
    """
    start = datetime.datetime(2026, 1, 1)
    day = datetime.timedelta(days=1)

    # 1. Trade a position twenty-five times, alternating trims and top-ups at rising prices
    position = execute(state, pair, 400, 10.0, start)
    price = 10.0
    for step in range(12):
        price = 10.0 + step
        position = execute(state, pair, -50, price, start + (2 * step + 1) * day)
        position = execute(state, pair, 60, price + 0.5, start + (2 * step + 2) * day)
    position.last_token_price = price

    # 2. Assert realised plus unrealised still equals the cash-flow profit
    assert len(position.trades) == 25
    realised = position.get_realised_profit_usd() or 0.0
    unrealised = position.get_unrealised_profit_usd() or 0.0
    assert realised + unrealised == pytest.approx(cash_flow_profit(position), abs=0.01)
    assert position.get_total_profit_usd() == pytest.approx(cash_flow_profit(position), abs=0.01)
