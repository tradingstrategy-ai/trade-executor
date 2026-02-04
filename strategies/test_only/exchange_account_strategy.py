"""Test strategy for exchange account position.

A minimal strategy that creates a single exchange account position
for testing sync and valuation flows.
"""

import datetime
from typing import List

import pandas as pd

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
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

# Chain ID for the exchange account (synthetic)
CHAIN_ID = ChainId.ethereum


class Parameters:
    """Strategy parameters for exchange account monitoring."""
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.ignore
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe with a single exchange account position.

    The position represents capital deployed to an external perp DEX
    (e.g., Derive, Hyperliquid).
    """

    # Create synthetic assets for the exchange account
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )

    # The "base" asset represents the exchange account value
    exchange_account = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    # Create the exchange account trading pair
    exchange_account_pair = TradingPairIdentifier(
        base=exchange_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": True,
        },
    )

    # Create pair universe
    pair_universe = create_pair_universe_from_code(CHAIN_ID, [exchange_account_pair])

    # Create mock exchange (required by Universe)
    mock_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="derive",
        address="0x0000000000000000000000000000000000000004",
        exchange_type=ExchangeType.uniswap_v2,  # Doesn't matter for exchange account
        pair_count=1,
    )

    # Create minimal candle universe (not used but required)
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
    """No indicators needed for exchange account monitoring."""
    pass


def decide_trades(
    timestamp: pd.Timestamp,
    universe: Universe,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict,
) -> List[TradeExecution]:
    """Passive strategy - no trades.

    Exchange account positions are managed externally on the perp DEX.
    This strategy just monitors the account value via sync.
    """
    return []
