"""Backtesting dataset load progress baring."""

from typing import Optional, Callable

import pandas as pd

from tradingstrategy.client import Client
from tradingstrategy.environment.jupyter import download_with_tqdm_progress_bar

from tradeexecutor.strategy.execution_context import ExecutionMode, ExecutionContext
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task


def preload_data(
        client: Client,
        create_trading_universe: Callable,
        universe_options: UniverseOptions,
):
    """Show nice progress bar for setting up data fees for backtesting trading universe.

    - We trigger call to `create_trading_universe` before the actual backtesting begins

    - The client is in a mode that it will display dataset download progress bars.
      We do not display these progress bars by default, as it could a bit noisy.
    """

    # Switch to the progress bar downloader
    # TODO: Make this cleaner
    client.transport.download_func = download_with_tqdm_progress_bar

    execution_context = ExecutionContext(
        mode=ExecutionMode.data_preload,
        timed_task_context_manager=timed_task,
    )

    create_trading_universe(
        pd.Timestamp.now(),
        client,
        execution_context,
        universe_options=universe_options,
    )

