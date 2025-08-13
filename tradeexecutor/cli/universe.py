"""Command line universe construction helpers"""
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import logging

from packaging import version

from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.parameters import dump_parameters, StrategyParameters
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client


logger = logging.getLogger(__name__)


@dataclass
class UniverseInitData:
    client: Client
    universe_model: TradingStrategyUniverseModel
    universe_options: UniverseOptions
    max_data_delay: datetime.timedelta
    run_description: StrategyExecutionDescription
    execution_model: ExecutionModel
    strategy_parameters: StrategyParameters


def setup_universe(
    trading_strategy_api_key: str,
    cache_path: Optional[Path],
    max_data_delay_minutes: int,
    strategy_factory: Callable,
    execution_context: ExecutionContext,
    unit_testing=False,
) -> UniverseInitData:
    """Setup universe loading for a trading strategy.

    - Intended to bootstrap CLI scripts
    - We need strategy module loaded and decoded
    - Extract create_universe() factory from the module
    - Don't call the factory yet
    """

    client = Client.create_live_client(
        trading_strategy_api_key,
        cache_path=cache_path,
        settings_path=None,
    )

    if not unit_testing:
        client.clear_caches()

    max_data_delay = datetime.timedelta(minutes=max_data_delay_minutes)

    run_description: StrategyExecutionDescription = strategy_factory(
        execution_model=None,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_model=None,
        valuation_model_factory=None,
        pricing_model_factory=None,
        approval_model=None,
        client=client,
        run_state=RunState(),
    )

    parameters = run_description.runner.parameters
    if parameters:
        logger.info(
            "Strategy parameters:\n%s",
            dump_parameters(parameters)
        )
        universe_options = UniverseOptions.from_strategy_parameters_class(
            parameters,
            execution_context,
        )
    else:
        universe_options = UniverseOptions()

    # Check that Parameters gives us period how much history we need
    engine_version = run_description.trading_strategy_engine_version
    if engine_version:
        if version.parse(engine_version) >= version.parse("0.5"):
            parameters = run_description.runner.parameters
            assert parameters, f"Engine version {engine_version}, but runner lacks decoded strategy parameters"
            assert "required_history_period" in parameters, f"Strategy lacks Parameters.required_history_period. We have {parameters}"

    # Deconstruct strategy input
    universe_model: TradingStrategyUniverseModel = run_description.universe_model

    return UniverseInitData(
        client=client,
        universe_model=universe_model,
        universe_options=universe_options,
        max_data_delay=max_data_delay,
        run_description=run_description,
        strategy_parameters=parameters,
        execution_model=run_description.execution_model,
    )
