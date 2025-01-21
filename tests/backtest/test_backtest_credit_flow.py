"""Test PositionManager.manage_credit_flow()."""

import datetime
import random
from decimal import Decimal

import pandas as pd
import pytest
from typing import List

from tradeexecutor.analysis.credit import calculate_credit_metrics
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe


start_at = datetime.datetime(2023, 1, 1)
end_at = datetime.datetime(2023, 6, 1)


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    time_bucket = TimeBucket.d1

    # Set up fake assets
    chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="uniswap-v2"
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


@pytest.fixture(scope="module")
def strategy_universe(universe):
    """Legacy alias. Use strategy_universe."""
    return universe


# Use deterministic random numbers in decide_trades()
rng = random.Random(0)


def decide_trades(input: StrategyInput) -> List[TradeExecution]:
    """Example decide_trades function using partial take profits and trailing stoploss."""
    position_manager = input.get_position_manager()
    trading_pair = input.get_default_pair()
    cash = input.state.portfolio.get_cash()

    volatile_position = position_manager.get_current_position_for_pair(trading_pair)

    # Generate randomish trades
    trades = []
    if input.cycle % 3 == 0:

        if rng.randint(0, 4) == 0:
            # Randomly close everything
            position_manager.log("Decide to close all")
            trades += position_manager.close_all(
                credit_supply=False,
            )
        else:
            if not volatile_position:
                position_manager.log("Decide to open spot")
                trades += position_manager.open_spot(
                    trading_pair,
                    cash * (rng.random() * 0.7 + 0.1),
                )
            else:
                position = volatile_position
                assert position.is_open()
                position_manager.log("Decide something else")
                if rng.randint(0, 4) == 0:
                    # Randomly increase of decrease the position
                    portion = 0.05 * rng.choice([1, -1])
                    if portion > 0:
                        dollar_delta = position.get_value() * portion
                        quantity_delta = None
                        position_manager.log("Decide to increase spot")
                    else:
                        quantity_delta = position.get_quantity() * Decimal(portion)
                        dollar_delta = position.get_value() * portion
                        position_manager.log("Decide to decrease spot")

                    assert trading_pair == position.pair, f"trading_pair: {trading_pair}, positio.pair: {position.pair}"

                    trades += position_manager.adjust_position(
                        trading_pair,
                        dollar_delta=dollar_delta,
                        quantity_delta=quantity_delta,
                        weight=1,
                        position=volatile_position,
                    )
    else:
        position_manager.log("Do nothing cycle")

    credit_flow = position_manager.calculate_credit_flow_needed(
        trades,
        allocation_pct=0.95,
    )
    trades += position_manager.manage_credit_flow(credit_flow)
    return trades


class Parameters:
    cycle_duration = TimeBucket.d1
    slippage_tolerance = 0.01
    initial_cash = 10_000


def test_backtest_credit_flow(
    strategy_universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open both long and short in the same cycle
    - Backtest shouldn't raise any exceptions
    """

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.5",
        indicator_combinations=set(),
        parameters=StrategyParameters.from_class(Parameters)
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) >= 1

    # Calculate total interest gained
    credit_positions = [p for p in portfolio.get_all_positions() if p.is_credit_supply()]
    assert len(credit_positions) == 10
    total_interest_gained = sum(p.get_total_profit_usd() for p in credit_positions)
    assert total_interest_gained == pytest.approx(76.8234503286251 rel=0.01)

    interest_metrics = calculate_credit_metrics(state)
    assert isinstance(interest_metrics, pd.DataFrame)
