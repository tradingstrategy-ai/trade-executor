"""Run backtest for a single strategy module."""
import inspect
import logging
import os
import datetime
import sys
from pathlib import Path

import pandas as pd

from tradeexecutor.analysis.pair import display_strategy_universe
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe, BacktestResult
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import standalone_backtest_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import DiskIndicatorStorage
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.cpu import get_safe_max_workers_count
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


def run_backtest_for_module(
    strategy_file: Path,
    cache_path: Path | None = None,
    trading_strategy_api_key: str = None,
    execution_context=standalone_backtest_execution_context,
    max_workers: int | None = None,
    verbose=True,
) -> BacktestResult:
    """Run a backtest described in the strategy module.

    - Designed for notebooks and console

    - Load all data and run a backtest

    - Will display multiple tqdm porgress bars and may print output

    :param strategy_file:
        Path to the strategy module

    :param cache_path:
        Path to the indicator cache

    :param trading_strategy_api_key:
        If not given, attempt load from a setting file or environment

    :param verbose:
        Print CLI console output

    :return:
        (state, universe, diagnostics data tuple)
    """

    assert strategy_file.exists(), f"Does not exist: {strategy_file.resolve()}"

    mod = read_strategy_module(strategy_file)

    assert mod.is_version_greater_or_equal_than(0, 2, 0), f"trading_strategy_engine_version must be 0.2.0 or newer for {strategy_file}"
    name = mod.name
    if name is None:
        name = os.path.basename(strategy_file)

    if trading_strategy_api_key is None:
        trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY")

    if execution_context.jupyter:
        client = Client.create_jupyter_client()
    elif execution_context.mode.is_unit_testing():
        client = Client.create_test_client(
            cache_path=cache_path,
        )
    else:
        # Load TRADING_STRATEGY_API_KEY from a settings file
        client = Client.create_live_client(
            trading_strategy_api_key,
            cache_path=cache_path,
        )

    universe_options = mod.get_universe_options()

    logger.info("Using cache path %s", client.transport.cache_path)
    logger.info("Loading backtesting universe data for %s", universe_options)

    universe = mod.create_trading_universe(
        datetime.datetime.utcnow(),
        client,
        execution_context,
        universe_options,
    )

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 140):
        universe_df = display_strategy_universe(universe)
        print("Loaded strategy universe pairs are:\n%s", str(universe_df))

    initial_cash = mod.initial_cash
    assert initial_cash is not None, f"Strategy module does not set initial_cash needed to backtest"
    assert mod.backtest_start, f"Strategy module does not set backtest_start"
    assert mod.backtest_end, f"Strategy module does not set backtest_end"

    # Todo:

    # Don't start at T+0 because we have not any data for that day yet
    backtest_start_at = universe_options.start_at + mod.trading_strategy_cycle.to_timedelta()
    backtest_end_at = mod.backtest_end
    logger.info("Backtest starts at %s", backtest_start_at)

    if verbose:
        print("Strategy universe OHLCV data is between", universe.data_universe.candles.get_timestamp_range())
        print("Backtesting period is", backtest_start_at, backtest_end_at)

    indicator_storage = DiskIndicatorStorage(Path(client.transport.cache_path) / "indicators", universe_key=universe.get_cache_key())
    inside_ipython = any(frame for frame in inspect.stack() if frame.function == "start_ipython")

    if not max_workers:
        # ipython command fails with multiprocessing module
        if inside_ipython:
            max_workers = 1

    backtest_setup = setup_backtest_for_universe(
        mod,
        start_at=backtest_start_at,
        end_at=backtest_end_at,
        cycle_duration=mod.trading_strategy_cycle,
        initial_deposit=initial_cash,
        name=name,
        universe=universe,
        universe_options=universe_options,
        create_indicators=mod.create_indicators,
        parameters=mod.parameters,
        indicator_storage=indicator_storage,
        max_workers=max_workers or get_safe_max_workers_count(),
    )

    assert backtest_setup.trading_strategy_engine_version
    assert backtest_setup.name

    result = run_backtest(
        backtest_setup,
        client=client,
    )

    return result