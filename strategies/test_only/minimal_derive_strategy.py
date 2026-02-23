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

    Has USDC as reserve and a Derive exchange account pair.
    """
    # Use real USDC address from environment
    usdc_address = os.environ.get("TEST_USDC_ADDRESS", "0x0000000000000000000000000000000000000001")
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=usdc_address,
        token_symbol="USDC",
        decimals=6,
    )

    derive_account_asset = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000D371E0",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    subaccount_id = int(os.environ.get("TEST_DERIVE_SUBACCOUNT_ID", "0"))

    derive_account_pair = TradingPairIdentifier(
        base=derive_account_asset,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000D371E1",
        exchange_address="0x0000000000000000000000000000000000D371E2",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": subaccount_id,
            "exchange_is_testnet": True,
        },
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [derive_account_pair])

    derive_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="anvil",
        exchange_id=1,
        exchange_slug="derive",
        address="0x0000000000000000000000000000000000D371E2",
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={derive_exchange},
        pairs=pair_universe,
        candles=None,
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
