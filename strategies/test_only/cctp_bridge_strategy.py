"""Test strategy for CCTP bridge position accounting.

A minimal strategy that creates a CCTP bridge pair in its universe
for testing correct-accounts CLI flow with bridge positions.

Uses two separate chains:

- Source chain (Arbitrum) holds reserve USDC
- Destination chain (Base) holds bridged USDC

Token addresses default to real mainnet USDC. Tests override them via
environment variables so that locally deployed tokens are used instead:

- ``TEST_USDC_SOURCE_ADDRESS``: reserve USDC token (source chain)
- ``TEST_USDC_DEST_ADDRESS``: bridged USDC token (destination chain)
"""

import datetime
import os

from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.token import USDC_NATIVE_TOKEN

from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.universe_model import UniverseOptions


trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

#: Source chain holds reserves
SOURCE_CHAIN_ID = ChainId.arbitrum
#: Destination chain holds bridged USDC
DEST_CHAIN_ID = ChainId.base


class Parameters:
    """Strategy parameters for CCTP bridge testing."""
    chain_id = ChainId.arbitrum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe with a CCTP bridge pair.

    Uses real mainnet USDC addresses by default.
    Tests override via ``TEST_USDC_SOURCE_ADDRESS`` / ``TEST_USDC_DEST_ADDRESS``.
    """

    # Allow tests to override with locally deployed token addresses
    usdc_source_address = os.environ.get(
        "TEST_USDC_SOURCE_ADDRESS",
        USDC_NATIVE_TOKEN[SOURCE_CHAIN_ID.value],
    )
    usdc_dest_address = os.environ.get(
        "TEST_USDC_DEST_ADDRESS",
        USDC_NATIVE_TOKEN[DEST_CHAIN_ID.value],
    )

    # Reserve token on source chain (Arbitrum)
    usdc_source = AssetIdentifier(
        chain_id=SOURCE_CHAIN_ID.value,
        address=usdc_source_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Bridged token on destination chain (Base)
    usdc_dest = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=usdc_dest_address,
        token_symbol="USDC",
        decimals=6,
    )

    # CCTP bridge pair: base=destination USDC, quote=source USDC
    # Pool and exchange addresses point to Circle's TokenMessengerV2
    # (same CREATE2 address on all EVM chains)
    cctp_bridge_pair = TradingPairIdentifier(
        base=usdc_dest,
        quote=usdc_source,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": DEST_CHAIN_ID.value,
        },
    )

    pair_universe = create_pair_universe_from_code(SOURCE_CHAIN_ID, [cctp_bridge_pair])

    cctp_exchange = Exchange(
        chain_id=SOURCE_CHAIN_ID,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={SOURCE_CHAIN_ID, DEST_CHAIN_ID},
        exchanges={cctp_exchange},
        pairs=pair_universe,
        candles=None,
        liquidity=None,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc_source],
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
    """Passive strategy — no trades."""
    return []
