"""Profile a single backtest to find performance bottlenecks.

Run with: poetry run python scripts/profile_backtest.py
"""
import cProfile
import pstats
import logging
import random
import datetime
from typing import List, Dict

import pandas as pd
from pandas_ta.overlap import ema

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting


position_size = 0.10
time_bucket = TimeBucket.d1
batch_size = 90
slow_ema_candle_count = 20
fast_ema_candle_count = 5


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    pair = universe.pairs.get_single()
    cash = state.portfolio.get_cash()
    candles = universe.candles.get_single_pair_data(timestamp, sample_count=batch_size, raise_on_not_enough_data=False)
    close = candles["close"]
    slow_ema_series = ema(close, length=slow_ema_candle_count)
    fast_ema_series = ema(close, length=fast_ema_candle_count)
    if slow_ema_series is None or fast_ema_series is None:
        return []
    slow_ema = slow_ema_series.iloc[-1]
    fast_ema = fast_ema_series.iloc[-1]
    current_price = close.iloc[-1]
    trades = []
    position_manager = PositionManager(timestamp, universe, state, pricing_model)
    if current_price >= slow_ema:
        if not position_manager.is_any_open():
            buy_amount = cash * position_size
            trades += position_manager.open_spot(pair, buy_amount)
    elif fast_ema >= slow_ema:
        if position_manager.is_any_open():
            trades += position_manager.close_all()
    return trades


def create_universe():
    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth, usdc, generate_random_ethereum_address(), mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id, fee=0.0030)
    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])
    candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)
    universe = Universe(
        time_bucket=time_bucket, chains={mock_chain_id}, exchanges={mock_exchange},
        pairs=pair_universe, candles=candle_universe, liquidity=None)
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe
    return strategy_universe


def run_profiled_backtest():
    strategy_universe = create_universe()
    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()
    routing_model = generate_simple_routing_model(strategy_universe)

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )
    print(f"Backtest completed: {len(debug_dump)} cycles")


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run_profiled_backtest()
    pr.disable()

    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    print("\n=== TOP 80 BY CUMULATIVE TIME ===")
    stats.print_stats(80)

    print("\n=== TOP 40 BY SELF TIME (tottime) ===")
    stats.sort_stats("tottime")
    stats.print_stats(40)
