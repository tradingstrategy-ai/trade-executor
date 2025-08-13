"""Input for creating a trading universe in PandasTrader."""
import datetime
from dataclasses import dataclass

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.parameters import StrategyParameters

from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client


@dataclass
class CreateTradingUniverseInput:
    """Input for creating a trading universe using create_trading_universe() callback.

    Example:

    .. code-block:: python

        def create_trading_universe(
            input: CreateTradingUniverseInput,
        ) -> TradingStrategyUniverse:

            execution_context = input.execution_context
            client = input.client
            timestamp = input.timestamp
            parameters = input.parameters
            universe_options = input.universe_options

            if execution_context.live_trading:
                # Live trading, send strategy universe formation details
                # to logs
                debug_printer = logger.info
            else:
                # Notebook node
                debug_printer = print

            chain_id = parameters.chain_id

            debug_printer(f"Preparing trading universe on chain {chain_id.get_name()}")

            exchange_universe = client.fetch_exchange_universe()
            targeted_exchanges = [exchange_universe.get_by_chain_and_slug(chain_id, slug) for slug in parameters.exchanges]

            ...
    """

    #: Our main download client
    client: Client

    #: When this universe is being created.
    #:
    #: - Wall clock time (live execution)
    #: - Backtest start time (backtesting)
    timestamp: datetime.datetime | None

    #: Strategy parameters
    parameters: StrategyParameters

    #: Current execution context
    execution_context: ExecutionContext

    #: Execution model with web3 connection
    execution_model: ExecutionModel

    #: Universe options
    universe_options: UniverseOptions
