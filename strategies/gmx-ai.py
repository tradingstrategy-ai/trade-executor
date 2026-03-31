"""Ichimoku trend-following strategy on GMX perpetual futures.

Dual long/short Ichimoku trend-following strategy trading
perpetual futures on GMX (Arbitrum).

The strategy goes long when the Ichimoku cloud signals a bullish trend flip,
confirmed by momentum indicators. It goes short when bearish trends reassert
themselves after failed price retests of key cloud levels.

Positions are evaluated on a 1-hour timeframe with signals derived from
4-hour price structure. Position sizes are tiered by market capitalisation.

Risk management:
- 1x leverage
- Up to 10 concurrent positions
- Dynamic stop losses based on market volatility (ATR)
- Trailing stops on short positions to lock in profits
- Market cap-weighted position sizing
- Stop losses placed on exchange for execution safety

Creates a GMX exchange account position that tracks the NAV of the vault's
GMX perpetuals positions. The Safe address is resolved at runtime from the
execution model's transaction builder.

Deploy, first simulate:

.. code-block:: shell

    source  scripts/set-latest-tag-yubi.sh     
    SIMULATE=true deploy/deploy-gmx-ai.sh
    SIMULATE=false deploy/deploy-gmx-ai.sh

Uppdate env/gmx-ai.env with the vault and adapter addresses from the deploy output, then update ~/secrets/gmx-ai.env with API and private keys.

Then:

.. code-block:: shell

    docker compose run gmx-ai init

    docker compose run gmx-ai lagoon-first-deposit --simulate --deposit-amount=25
    docker compose run gmx-ai lagoon-first-deposit --deposit-amount=25

Setup GMX approval:

.. code-block:: shell

    docker compose run gmx-ai console

Console copy-paste code:

.. code-block:: python

    from tradeexecutor.exchange_account.gmx import approve_gmx_trading ; approve_gmx_trading(vault, hot_wallet)    

"""

import datetime

from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import \
    CreateTradingUniverseInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, create_pair_universe_from_code)

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

CHAIN_ID = ChainId.arbitrum
USDC_ADDRESS = USDC_NATIVE_TOKEN[CHAIN_ID.value]


class Parameters:
    """Strategy parameters."""
    chain_id = CHAIN_ID
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create a trading universe with a GMX exchange account pair.

    Has USDC as reserve and a GMX exchange account pair.
    """
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )

    gmx_account_pair = create_gmx_exchange_account_pair(
        quote=usdc,
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [gmx_account_pair])

    gmx_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="gmx",
        address=gmx_account_pair.exchange_address,
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={gmx_exchange},
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
):
    """No indicators needed."""
    pass


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """Create the exchange account position on the first cycle, then do nothing.

    The position is created by calling ``open_exchange_account_position()``
    which creates a spoofed trade that never goes through routing/execution.
    """
    from decimal import Decimal

    from tradeexecutor.exchange_account.state import \
        open_exchange_account_position

    state = input.state
    timestamp = input.timestamp

    # Check if position already exists
    for pos in state.portfolio.open_positions.values():
        if pos.is_exchange_account():
            return []

    # First cycle: create exchange account position
    pair = next(input.strategy_universe.iterate_pairs())
    reserve = input.strategy_universe.reserve_assets[0]

    open_exchange_account_position(
        state=state,
        strategy_cycle_at=timestamp.to_pydatetime(),
        pair=pair,
        reserve_currency=reserve,
        reserve_amount=Decimal(0),
        notes="GMX exchange account position",
    )

    return []


tags = {StrategyTag.live, StrategyTag.beta}

name = "GMX AI"

short_description = "A directional multistrategy trading on GMX"

icon = ""

long_description = """
# GMX AI directional multistrategy

This strategy is currently in its ramp-up phase and is not yet fully operational. Performance and parameters are being actively tuned as the strategy matures.

Combines breakout and mean reversion trading approaches on GMX perpetual futures (Arbitrum). The strategy identifies directional opportunities by blending momentum-driven breakout signals with mean reversion entries, aiming to capture both trending moves and short-term price dislocations.

## View on GMX

[View strategy positions on GMX](https://app.gmx.io/#/accounts/0x7838A4E4ecD438c1BdD13b014675c7e877b8b490)

## Strategy features

- **Dual signal approach**: Combines breakout detection for trending markets with mean reversion trades during range-bound conditions
- **GMX perpetual futures**: Trades perpetual futures on GMX (Arbitrum) for capital-efficient directional exposure

## Assets and trading venues

- Trades on GMX perpetual futures on Arbitrum
- USDC as the reserve and settlement currency

## Risk management

- 1x leverage
- Dynamic stop losses based on market volatility (ATR)
- Market cap-weighted position sizing
- Stop losses placed on exchange for execution safety


**Past performance is not indicative of future results**.
"""
