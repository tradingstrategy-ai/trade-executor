"""Data forward fill tests."""
import datetime

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


def test_forward_fill_spot_only(persistent_test_client: Client):
    """Forward-will spot market data.

    - When dataset is loaded, forward-fill any gaps when the dataset is transformed to the trading universe
    """
    client = persistent_test_client

    # The pairs we are rading
    pair_ids = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),  # This data has a lot of gaps
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    start_at = datetime.datetime(2022, 1, 1)
    end_at = datetime.datetime(2024, 3, 15)

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.d1,
        pairs=pair_ids,
        execution_context=unit_test_execution_context,
        liquidity=False,
        stop_loss_time_bucket=TimeBucket.h1,
        start_at=start_at,
        end_at=end_at,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x22177148e681a6ca5242c9888ace170ee7ec47bd",
        forward_fill=True,
    )

    # Check there are no gaps in the data
    import ipdb ; ipdb.set_trace()
