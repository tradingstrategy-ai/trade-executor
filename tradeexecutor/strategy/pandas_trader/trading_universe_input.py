"""Input for creating a trading universe in PandasTrader."""
import datetime
import inspect
from dataclasses import dataclass
from typing import Callable, Protocol, Optional

from tradeexecutor.strategy.execution_context import ExecutionContext
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
    #:
    #: Maybe None if strategy parameters are not used or passed.
    #:
    parameters: "tradeexecutor.strategy.parameters.StrategyParameters | None"

    #: Current execution context
    execution_context: "tradeexecutor.strategy.execution_context.ExecutionContext"

    #: Execution model with web3 connection.
    #:
    #: Maybe None if called directly or not passed.
    execution_model: "tradeexecutor.strategy.execution_model.ExecutionModel | None"

    #: Universe options
    universe_options: "tradeexecutor.strategy.universe_model.UniverseOptions"


class CreateTradingUniverseProtocol(Protocol):
    """A call signature protocol for user's create_trading_universe() functions.

    This describes the `create_trading_universe` function in trading strategies
    using Python's `callback protocol <https://peps.python.org/pep-0544/#callback-protocols>`_ feature.

    See also :ref:`strategy examples`.

    Example `create_trading_universe` function:

    .. code-block:: python

        def create_trading_universe(
                ts: datetime.datetime,
                client: Client,
                execution_context: ExecutionContext,
                candle_time_frame_override: Optional[TimeBucket]=None,
        ) -> TradingStrategyUniverse:

            # Load all datas we can get for our candle time bucket
            dataset = load_pair_data_for_single_exchange(
                client,
                execution_context,
                candle_time_bucket,
                chain_id,
                exchange_slug,
                [trading_pair_ticker],
                stop_loss_time_bucket=stop_loss_time_bucket,
                )

            # Filter down to the single pair we are interested in
            universe = TradingStrategyUniverse.create_single_pair_universe(
                dataset,
                chain_id,
                exchange_slug,
                trading_pair_ticker[0],
                trading_pair_ticker[1],
            )

            return universe
    """

    def __call__(self,
            timestamp: datetime.datetime,
            client: Optional[Client],
            execution_context: ExecutionContext,
            universe_options: UniverseOptions) -> "tradeexecutor.trading_strategy_universe.TradingStrategyUniverse":
        """Creates the trading universe where the strategy trades.

        See also :ref:`strategy examples`

        If `execution_context.live_trading` is true then this function is called for
        every execution cycle. If we are backtesting, then this function is
        called only once at the start of backtesting and the `decide_trades`
        need to deal with new and deprecated trading pairs.

        As we are only trading a single pair, load data for the single pair only.

        :param ts:
            The timestamp of the trading cycle. For live trading,
            `create_trading_universe` is called on every cycle.
            For backtesting, it is only called at the start

        :param client:
            Trading Strategy Python client instance.

        :param execution_context:
            Information how the strategy is executed. E.g.
            if we are live trading or not.

        :param options:
            Allow the backtest framework override what candle size is used to backtest the strategy
            without editing the strategy Python source code file.

        :return:
            This function must return :py:class:`TradingStrategyUniverse` instance
            filled with the data for exchanges, pairs and candles needed to decide trades.
            The trading universe also contains information about the reserve asset,
            usually stablecoin, we use for the strategy.
            """


class CreateTradingUniverseProtocolV2(Protocol):
    """A callback to create the trading universe for the strategy."""

    def __call__(
        self,
        input: CreateTradingUniverseInput,
    ) -> "tradeexecutor.trading_strategy_universe.TradingStrategyUniverse":
        pass


def get_create_trading_universe_version(
    func: Callable,
) -> int:
    """Get the version number of create_trading_universe() function"""
    arg_names = [param.name for param in inspect.signature(func).parameters.values()]

    if arg_names == ["input"]:
        return 2
    elif arg_names == ["timestamp", "client", "execution_context", "universe_options"]:
        return 1
    else:
        raise RuntimeError(f"Unknown create_trading_universe() args: {arg_names}")
