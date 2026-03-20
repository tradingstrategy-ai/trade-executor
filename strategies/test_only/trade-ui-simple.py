"""Minimal strategy for trade-ui CLI integration tests.

Uses two hardcoded pairs on Base (WETH/USDC on Uniswap v2 and v3)
with no external data dependencies like Coingecko or TokenSniffer.
"""

import datetime

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket


trading_strategy_engine_version = "0.5"


class Parameters:
    id = "trade-ui-simple"
    cycle_duration = CycleDuration.cycle_4h
    candle_time_bucket = TimeBucket.h1
    chain_id = ChainId.base
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=7)
    slippage_tolerance = 0.015
    backtest_start = datetime.datetime(2024, 1, 1)
    backtest_end = datetime.datetime(2024, 3, 1)
    initial_cash = 10_000


PAIRS: list[HumanReadableTradingPairDescription] = [
    (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),
]


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a minimal two-pair universe for testing."""

    pairs_df = client.fetch_pair_universe().to_pandas()
    chain_mask = pairs_df["chain_id"] == Parameters.chain_id.value
    pairs_df = pairs_df[chain_mask]

    exchange_universe = client.fetch_exchange_universe()
    from tradingstrategy.pair import PandasPairUniverse
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=exchange_universe)

    our_pair_ids = [pair_universe.get_pair_by_human_description(desc).pair_id for desc in PAIRS]
    pairs_df = pairs_df[pairs_df["pair_id"].isin(our_pair_ids)]

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
    )

    return TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        forward_fill=True,
        forward_fill_until=timestamp,
    )


def create_indicators(
    timestamp,
    parameters,
    strategy_universe,
    execution_context,
):
    """No indicators needed."""
    indicators = IndicatorRegistry()
    return indicators


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """No-op strategy — we only need the universe for test trades."""
    return []
