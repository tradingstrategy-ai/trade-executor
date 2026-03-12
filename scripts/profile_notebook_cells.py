"""Profile each cell of the v49 optimiser notebook to find bottlenecks.

Skips backtest/optimiser cells (already optimised).

Usage:
    poetry run python scripts/profile_notebook_cells.py
"""

import cProfile
import pstats
import io
import time
import datetime
import logging

logging.basicConfig(level=logging.WARNING)


def profile_cell(name, func):
    """Profile a cell and print results."""
    print(f"\n{'='*80}")
    print(f"CELL: {name}")
    print(f"{'='*80}")

    profiler = cProfile.Profile()
    wall_start = time.perf_counter()
    profiler.enable()
    result = func()
    profiler.disable()
    wall_elapsed = time.perf_counter() - wall_start

    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)

    # Cumulative time
    ps.sort_stats("cumulative")
    ps.print_stats(30)
    print(stream.getvalue())

    # Self time
    stream2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=stream2)
    ps2.sort_stats("tottime")
    ps2.print_stats(20)
    print("--- Top by self time ---")
    print(stream2.getvalue())

    print(f"Wall time: {wall_elapsed:.3f}s")
    return result


# =============================================================================
# Cell 1: Client + charting setup
# =============================================================================
def cell_1_setup():
    from tradeexecutor.utils.notebook import setup_charting_and_output, OutputMode
    from tradingstrategy.client import Client

    client = Client.create_jupyter_client()
    setup_charting_and_output(OutputMode.static, image_format="png", width=1500, height=1000)
    return client


# =============================================================================
# Cell 3: Parameters (just class definition — expect ~0s)
# =============================================================================
def cell_3_parameters():
    from skopt.space import Integer, Real, Categorical
    from tradeexecutor.strategy.default_routing_options import TradeRouting
    from tradingstrategy.chain import ChainId
    from tradingstrategy.timebucket import TimeBucket
    from tradingstrategy.pair import HumanReadableTradingPairDescription
    from tradeexecutor.strategy.execution_context import ExecutionMode
    from tradeexecutor.strategy.cycle import CycleDuration
    return True


# =============================================================================
# Cell 5: create_binance_universe
# =============================================================================
def cell_5_universe(client):
    from tradingstrategy.timebucket import TimeBucket
    from tradeexecutor.utils.binance import create_binance_universe

    strategy_universe = create_binance_universe(
        ["BTCUSDT", "ETHUSDT"],
        candle_time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=TimeBucket.h1,
        start_at=datetime.datetime(2020, 1, 1),
        end_at=datetime.datetime(2024, 8, 10),
        include_lending=False,
    )
    return strategy_universe


# =============================================================================
# Cell 7: Display candle info
# =============================================================================
def cell_7_display(strategy_universe):
    pairs = strategy_universe.data_universe.pairs
    candles = strategy_universe.data_universe.candles

    count = candles.get_candle_count()
    for pair in pairs.iterate_pairs():
        pair_candles = candles.get_candles_by_pair(pair)
        first_close = pair_candles.iloc[0]["close"]
        first_close_at = pair_candles.index[0]
    return count


# =============================================================================
# Cell 9: Indicator definitions (just function defs — expect ~0s)
# =============================================================================
def cell_9_indicators():
    import pandas as pd
    import pandas_ta
    from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
    from tradingstrategy.utils.groupeduniverse import resample_price_series
    from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
    from tradeexecutor.strategy.execution_context import ExecutionContext
    from tradeexecutor.strategy.parameters import StrategyParameters
    return True


# =============================================================================
# Cell 11: Trading algorithm definition (just function def — expect ~0s)
# =============================================================================
def cell_11_algorithm():
    from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
    from tradeexecutor.strategy.weighting import weight_passthrouh
    from tradeexecutor.strategy.alpha_model import AlphaModel
    from tradeexecutor.state.visualisation import PlotKind, PlotShape, PlotLabel
    from tradeexecutor.state.trade import TradeExecution
    return True


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Profiling v49 notebook cells (excluding backtest/optimiser)")
    print("=" * 80)

    client = profile_cell("1: Client + charting setup", cell_1_setup)
    profile_cell("3: Parameter imports", cell_3_parameters)
    universe = profile_cell("5: create_binance_universe()", lambda: cell_5_universe(client))
    profile_cell("7: Display candle info", lambda: cell_7_display(universe))
    profile_cell("9: Indicator imports", cell_9_indicators)
    profile_cell("11: Algorithm imports", cell_11_algorithm)

    print("\n" + "=" * 80)
    print("DONE — Summary")
    print("=" * 80)
