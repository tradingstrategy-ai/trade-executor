"""Systematic multi-strategy vault trading Lighter perpetual futures.

Lighter AI combines trend-following, momentum breakouts, mean reversion, and
macro regime signals with fully automated dynamic capital allocation.
Complementary sub-strategies target different market conditions, with capital
redistributed according to the prevailing market environment.

Multi-timeframe signals from 5-minute to daily charts provide layered
confirmation for long and short positions across liquid Lighter perpetuals.
Risk management uses diversification across uncorrelated strategies, adaptive
stop losses, and continuous liquidity and momentum screening.

This module tracks the Safe-owned Lighter account for an Ethereum Lagoon vault.
It creates one exchange-account position and leaves public equity valuation and
NAV posting to the executor. A separate automated trading process runs the
sub-strategies and submits orders; this module holds no Lighter API key.

The strategy description is adapted from OpenCZ:
https://tradingstrategy.ai/strategies/opencz/description

See deploy/README-lighter-ai.md for deployment and accounting instructions.
"""

import datetime
import os
from decimal import Decimal

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import (
    CreateTradingUniverseInput,
)
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
)

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_15m
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

#: Lighter custody and the Lagoon vault both reside on Ethereum mainnet.
CHAIN_ID = ChainId.ethereum


class Parameters:
    """Strategy parameters."""

    chain_id = CHAIN_ID
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_15m
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create a single Ethereum Lighter exchange-account pair."""
    # Read at universe creation so metadata can be loaded before deployment.
    # Never fall back to a test index: this value determines the posted NAV.
    raw_account_index = os.environ.get("LIGHTER_ACCOUNT_INDEX", "").strip()
    if not raw_account_index.isascii() or not raw_account_index.isdecimal():
        raise ValueError(
            "Set LIGHTER_ACCOUNT_INDEX to the non-negative integer account index "
            "from the Lighter deployment report"
        )
    account_index = int(raw_account_index)
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_NATIVE_TOKEN[CHAIN_ID.value],
        token_symbol="USDC",
        decimals=6,
    )
    lighter_pair = create_lighter_exchange_account_pair(
        quote=usdc,
        account_index=account_index,
    )
    pair_universe = create_pair_universe_from_code(CHAIN_ID, [lighter_pair])
    exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        # Shared external-account exchange type, as in the upstream template.
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
    """Open the account position once; sync handles subsequent equity updates."""
    if any(pos.is_exchange_account() for pos in input.state.portfolio.open_positions.values()):
        return []

    pair = next(input.strategy_universe.iterate_pairs())
    open_exchange_account_position(
        state=input.state,
        strategy_cycle_at=input.timestamp.to_pydatetime(),
        pair=pair,
        reserve_currency=input.strategy_universe.reserve_assets[0],
        reserve_amount=Decimal(0),
        notes="Lighter exchange account position",
    )
    # The helper records a synthetic trade directly; never send it to execution.
    return []


tags = {StrategyTag.beta, StrategyTag.exchange_account_strategy}
name = "Lighter AI"
# Adapted from https://tradingstrategy.ai/strategies/opencz/description
short_description = "Systematic multi-strategy vault combining trend-following, momentum breakouts, mean reversion, and macro regime signals with dynamic capital allocation on Lighter perpetual futures."
icon = ""
long_description = """
Lighter AI is a systematic multi-strategy vault for Lighter perpetual futures,
combining multiple complementary trading strategies with fully automated
dynamic capital allocation.

## How it works

Multiple systematic sub-strategies run in parallel — trend-following, momentum
breakouts, mean reversion, and macro regime filtering — each targeting a
different market condition. A dynamic allocation engine continuously
redistributes capital across strategies based on prevailing market environment.

Multi-timeframe signals from 5-minute to daily charts provide layered
confirmation before entering positions. The vault trades both long and short
across the most liquid perpetual futures on Lighter.

All execution is fully automated.

## Risk framework

- Dynamic capital allocation across uncorrelated strategies
- Adaptive stop losses
- Continuous liquidity and momentum screening

## Backtesting results

TODO

*Metrics are based on a specific historical simulation setup and period,
and should not be interpreted as a guarantee of future outcomes.
"""
