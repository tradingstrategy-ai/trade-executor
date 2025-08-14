import logging

import pandas as pd

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.trading_universe_input import get_create_trading_universe_version, CreateTradingUniverseProtocol, CreateTradingUniverseProtocolV2
from tradeexecutor.strategy.universe_model import UniverseOptions


logger = logging.getLogger(__name__)


def call_create_trading_universe(
    create_trading_universe: CreateTradingUniverseProtocol | CreateTradingUniverseProtocolV2,
    client,
    universe_options: UniverseOptions,
    execution_context: ExecutionContext | None = None,
    execution_model: "tradeexecutor.strategy.execution_model.ExecutionModel | None" = None,
    strategy_parameters: StrategyParameters | None = None,
    timestamp: pd.Timestamp | None = None,
) -> "tradeexecutor.strategy.TradingStrategyUniverse":
    """Call the create_trading_universe function to preload data."""

    version = get_create_trading_universe_version(create_trading_universe)

    logger.info(
        "call_create_trading_universe(), version %s, execution model %s, timestamp %s, mode %s",
        version,
        execution_model,
        timestamp,
        execution_context.mode.name,
    )

    match version:
        case 1:
            return create_trading_universe(
                timestamp or pd.Timestamp.now(),
                client,
                execution_context,
                universe_options=universe_options,
            )
        case 2:
            input = CreateTradingUniverseInput(
                client=client,
                timestamp=timestamp or pd.Timestamp.now(),
                parameters=strategy_parameters,
                execution_context=execution_context,
                execution_model=execution_model,
                universe_options=universe_options,
            )
            return create_trading_universe(input)