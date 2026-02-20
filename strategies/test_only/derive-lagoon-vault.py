"""Minimal strategy config for Derive chain lagoon vault.

Used with the console command to perform test deposits and other
administrative operations on a lagoon vault deployed on Derive (chain ID 957).

Not a real trading strategy - just provides the minimum scaffolding
needed for the console command to boot.
"""

import datetime
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
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
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

CHAIN_ID = ChainId.derive

# USDC on Derive
USDC_ADDRESS = "0x6879287835A86F50f784313dBEd5E5cCC5bb8481"


class Parameters:
    """Strategy parameters."""
    chain_id = ChainId.derive
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.ignore
    initial_cash = 10_000
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a minimal trading universe with USDC as reserve."""

    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )

    # Dummy base token - we are not actually trading
    dummy_token = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="DUMMY",
        decimals=18,
    )

    dummy_pair = TradingPairIdentifier(
        base=dummy_token,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000002",
        exchange_address="0x0000000000000000000000000000000000000003",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.003,
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [dummy_pair])

    mock_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="derive",
        exchange_id=1,
        exchange_slug="mock",
        address="0x0000000000000000000000000000000000000003",
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

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

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
    )


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
    """No trades - this is just for vault admin operations."""
    return []
