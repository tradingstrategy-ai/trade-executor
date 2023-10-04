import datetime

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_trading_and_lending_data, TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions


def test_load_lending_market_data(persistent_test_client: Client):
    """Load lending market data and see if it is ok."""

    client = persistent_test_client
    start_at = datetime.datetime(2023, 9, 1)
    end_at = datetime.datetime(2023, 10, 1)

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_trading_and_lending_data(
        client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions(start_at=start_at, end_at=end_at),
        chain_id=ChainId.polygon,
        exchange_slug="uniswap-v3",
    )

    universe = TradingStrategyUniverse.create_from_dataset(dataset)





