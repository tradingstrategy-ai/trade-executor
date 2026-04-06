"""Helpers to preload webhook-facing runtime data."""

import datetime
import logging

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import (
    CreateIndicatorsProtocol,
    MemoryIndicatorStorage,
    calculate_and_load_indicators,
    call_create_indicators,
)
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import CreateChartsProtocol
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


def populate_webhook_run_state(
    run_state: RunState,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
    parameters,
    *,
    create_charts: CreateChartsProtocol | None = None,
    create_indicators: CreateIndicatorsProtocol | None = None,
    timestamp: datetime.datetime | None = None,
) -> tuple[object | None, StrategyInputIndicators | None]:
    """Populate in-memory webhook data structures for a strategy universe."""

    timestamp = timestamp or native_datetime_utc_now()

    chart_registry = None
    if create_charts is not None:
        logger.info("Creating chart registry for webhook preload")
        chart_registry = create_charts(
            timestamp=timestamp,
            parameters=parameters,
            strategy_universe=strategy_universe,
            execution_context=execution_context,
        )
        run_state.chart_registry = chart_registry
    else:
        logger.info("Strategy does not provide create_charts(); skipping chart registry preload")

    strategy_input_indicators = None
    if create_indicators is not None:
        logger.info("Creating indicators for webhook preload")
        storage = MemoryIndicatorStorage(strategy_universe.get_cache_key())
        indicators = call_create_indicators(
            create_indicators,
            parameters,
            strategy_universe,
            execution_context,
            timestamp,
        )
        indicator_results = calculate_and_load_indicators(
            strategy_universe=strategy_universe,
            storage=storage,
            execution_context=execution_context,
            indicators=indicators,
            parameters=parameters,
            strategy_cycle_timestamp=timestamp,
        )
        strategy_input_indicators = StrategyInputIndicators(
            strategy_universe,
            indicator_results=indicator_results,
            available_indicators=indicators,
        )
        run_state.latest_indicators = strategy_input_indicators
    else:
        logger.info("Strategy does not provide create_indicators(); skipping indicator preload")

    return chart_registry, strategy_input_indicators
