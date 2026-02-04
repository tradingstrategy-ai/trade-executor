"""Minimal test strategy for Derive CLI integration.

A simple strategy that does nothing but runs through init/start CLI commands.
Uses Anvil chain and deployed USDC for testing.

Environment variables:
- TEST_USDC_ADDRESS: Address of the USDC token contract deployed on Anvil
"""

import datetime
import os
from typing import List

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.ignore
reserve_currency = ReserveCurrency.usdc

# Chain ID for local Anvil testing
CHAIN_ID = ChainId.anvil


class Parameters:
    """Strategy parameters."""
    chain_id = ChainId.anvil
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.ignore
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a minimal trading universe.

    Has USDC as reserve and a dummy trading pair.
    """
    # Use real USDC address from environment
    usdc_address = os.environ.get("TEST_USDC_ADDRESS", "0x0000000000000000000000000000000000000001")
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=usdc_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Create a dummy base token for the pair
    # Use real token address from environment if available (for Anvil testing)
    dummy_token_address = os.environ.get("TEST_DUMMY_TOKEN_ADDRESS", "0x0000000000000000000000000000000000000002")
    dummy_token = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=dummy_token_address,
        token_symbol="DUMMY",
        decimals=18,
    )

    # Create a dummy trading pair (won't be traded)
    dummy_pair = TradingPairIdentifier(
        base=dummy_token,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.003,
    )

    # Create pair universe with the dummy pair
    pair_universe = create_pair_universe_from_code(CHAIN_ID, [dummy_pair])

    # Create mock exchange (required by Universe)
    mock_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="anvil",
        exchange_id=1,
        exchange_slug="mock",
        address="0x0000000000000000000000000000000000000004",
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    # Create minimal candle universe
    candles = pd.DataFrame({
        "pair_id": [1],
        "timestamp": [pd.Timestamp(ts)],
        "open": [1.0],
        "high": [1.0],
        "low": [1.0],
        "close": [1.0],
        "volume": [0.0],
    })
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    # Create the universe
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )

    strategy_universe = TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
    )

    return strategy_universe


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    """No indicators needed."""
    pass


def decide_trades(
    timestamp: pd.Timestamp,
    universe: Universe,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict,
) -> List[TradeExecution]:
    """No trades - just a passive strategy."""
    return []
