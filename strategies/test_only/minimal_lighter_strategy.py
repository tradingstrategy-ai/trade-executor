"""Minimal Ethereum strategy for Lagoon Lighter deployment tests.

The exchange-account pair is deliberately synthetic: trade-executor reads the
public Lighter account index from the pair and obtains its equity from the
unauthenticated public API. No Lighter API key is part of the strategy.
"""

import datetime
import os

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
)


trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

CHAIN_ID = ChainId.ethereum
LIGHTER_ACCOUNT_INDEX = int(os.environ.get("LIGHTER_ACCOUNT_INDEX", "123"))


class Parameters:
    """Strategy parameters."""

    chain_id = CHAIN_ID
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create a single Ethereum Lighter exchange-account pair."""
    del input
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_NATIVE_TOKEN[CHAIN_ID.value],
        token_symbol="USDC",
        decimals=6,
    )
    lighter_pair = create_lighter_exchange_account_pair(
        quote=usdc,
        account_index=LIGHTER_ACCOUNT_INDEX,
    )
    pair_universe = create_pair_universe_from_code(CHAIN_ID, [lighter_pair])
    exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={exchange},
        pairs=pair_universe,
        candles=None,
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
) -> None:
    """No indicators are needed for deployment or account monitoring."""
    del parameters, indicators, strategy_universe, execution_context


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Passive strategy used only to exercise the deployment/runtime wiring."""
    del input
    return []
