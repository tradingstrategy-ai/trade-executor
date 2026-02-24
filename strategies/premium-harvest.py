"""Premium harvest vault strategy.

An options premium harvesting strategy on Derive.
Positions are managed manually on the Derive perpetual DEX,
while this strategy module handles universe setup, position tracking
and account value synchronisation.

Environment variables:
- DERIVE_SESSION_PRIVATE_KEY: Private key for the Derive session key (required, used for API authentication)
- DERIVE_OWNER_PRIVATE_KEY: Private key of the Ethereum wallet that owns the Derive account
  (optional if DERIVE_WALLET_ADDRESS is provided)
- DERIVE_WALLET_ADDRESS: Derive wallet address (optional, auto-derived from owner key if not provided)
"""

import datetime

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
trading_strategy_cycle = CycleDuration.cycle_1h
trade_routing = TradeRouting.ignore
reserve_currency = ReserveCurrency.usdc

CHAIN_ID = ChainId.ethereum


class Parameters:
    """Strategy parameters for premium harvest vault."""
    id = "premium-harvest-vault"
    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1h
    routing = TradeRouting.ignore
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe for Derive options premium harvesting.

    Sets up a single exchange account pair representing the Derive
    subaccount where options trades are executed manually.
    """

    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )

    derive_account = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    exchange_account_pair = TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="Derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": False,
        },
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [exchange_account_pair])

    mock_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="derive",
        address="0x0000000000000000000000000000000000000004",
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
    """No indicators needed for manual options trading."""
    pass


def decide_trades(
    timestamp: pd.Timestamp,
    universe: Universe,
    state: State,
    pricing_model: PricingModel,
    cycle_debug_data: dict,
) -> list[TradeExecution]:
    """No automated trades.

    Options positions are managed manually on the Derive exchange.
    The trade executor only tracks account value via sync.
    """
    return []
