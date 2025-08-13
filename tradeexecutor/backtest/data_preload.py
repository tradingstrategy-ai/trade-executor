"""Backtesting dataset load progress baring."""

import logging

import pandas as pd

from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.strategy_module import CreateTradingUniverseProtocol, get_create_trading_universe_version
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.client import Client, BaseClient
from tradingstrategy.environment.default_environment import download_with_tqdm_progress_bar

from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task


logger = logging.getLogger(__name__)


def preload_data(
    client: BaseClient,
    create_trading_universe: CreateTradingUniverseProtocol,
    universe_options: UniverseOptions,
    execution_context: ExecutionContext | None = None,
    execution_model: ExecutionModel | None = None,
    strategy_parameters: StrategyParameters | None = None,
) -> TradingStrategyUniverse:
    """Show nice progress bar for setting up data fees for backtesting trading universe.

    - We trigger call to `create_trading_universe` before the actual backtesting begins

    - The client is in a mode that it will display dataset download progress bars.
      We do not display these progress bars by default, as it could a bit noisy.
    """

    # Switch to the progress bar downloader
    # TODO: Make this cleaner
    if isinstance(client, Client):
        client.transport.download_func = download_with_tqdm_progress_bar

    # Create new execution context that signals data preload
    if execution_context:
        engine_version = execution_context.engine_version
    else:
        engine_version = None

    logger.info("Preloading data, in execution context %s", execution_context)

    if execution_context is None:
        execution_context = ExecutionContext(
            mode=ExecutionMode.data_preload,
            timed_task_context_manager=timed_task,
            engine_version=engine_version,
        )

    version = get_create_trading_universe_version(create_trading_universe)

    match version:
        case 1:
            return create_trading_universe(
                pd.Timestamp.now(),
                client,
                execution_context,
                universe_options=universe_options,
            )
        case 2:
            input = CreateTradingUniverseInput(
                client=client,
                timestamp=pd.Timestamp.now(),
                parameters=strategy_parameters,
                execution_context=execution_context,
                execution_model=execution_model,
                universe_options=universe_options,
            )
            return create_trading_universe(input)

