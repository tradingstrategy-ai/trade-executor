"""Test strategy for CCTP bridge + ERC-4626 vault pair.

Extends the CCTP bridge test strategy with an Ostium vault pair on Arbitrum,
used to verify that ``translate_trading_universe_to_lagoon_config()`` correctly
collects vault addresses for guard whitelisting via ``whitelistERC4626()``.
"""

import datetime
import os
from decimal import Decimal

from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.token import USDC_NATIVE_TOKEN, WRAPPED_NATIVE_TOKEN
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS

from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
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

SOURCE_CHAIN_ID = ChainId.arbitrum
DEST_CHAIN_ID = ChainId.base

BASE_UNISWAP_V3_FACTORY = UNISWAP_V3_DEPLOYMENTS["base"]["factory"]
WETH_USDC_POOL = "0xd0b53d9277642d899df5c87a3966a349a798f224"

#: Ostium vault on Arbitrum
#: https://arbiscan.io/address/0x20D419a8e12C45f88fDA7c5760bb6923Cee27F98
OSTIUM_VAULT_ADDRESS = "0x20D419a8e12C45f88fDA7c5760bb6923Cee27F98"


class Parameters:
    chain_id = ChainId.arbitrum
    initial_cash = 10_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe with CCTP bridge + WETH/USDC on Base + Ostium vault on Arbitrum."""

    usdc_source_address = os.environ.get(
        "TEST_USDC_SOURCE_ADDRESS",
        USDC_NATIVE_TOKEN[SOURCE_CHAIN_ID.value],
    )
    usdc_dest_address = os.environ.get(
        "TEST_USDC_DEST_ADDRESS",
        USDC_NATIVE_TOKEN[DEST_CHAIN_ID.value],
    )
    weth_address = WRAPPED_NATIVE_TOKEN[DEST_CHAIN_ID.value]

    usdc_source = AssetIdentifier(
        chain_id=SOURCE_CHAIN_ID.value,
        address=usdc_source_address,
        token_symbol="USDC",
        decimals=6,
    )

    usdc_dest = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=usdc_dest_address,
        token_symbol="USDC",
        decimals=6,
    )

    weth = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=weth_address,
        token_symbol="WETH",
        decimals=18,
    )

    # Ostium vault share token on Arbitrum
    ostium_olp = AssetIdentifier(
        chain_id=SOURCE_CHAIN_ID.value,
        address=OSTIUM_VAULT_ADDRESS.lower(),
        token_symbol="oLP",
        decimals=18,
    )

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

    weth_usdc_pair = TradingPairIdentifier(
        base=weth,
        quote=usdc_dest,
        pool_address=WETH_USDC_POOL,
        exchange_address=BASE_UNISWAP_V3_FACTORY,
        internal_id=2,
        internal_exchange_id=2,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
        exchange_name="Uniswap v3 (Base)",
    )

    reverse_cctp_bridge_pair = TradingPairIdentifier(
        base=usdc_source,
        quote=usdc_dest,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=3,
        internal_exchange_id=3,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge (Base)",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": SOURCE_CHAIN_ID.value,
        },
    )

    # Ostium vault pair: deposit USDC, receive oLP
    ostium_vault_pair = TradingPairIdentifier(
        base=ostium_olp,
        quote=usdc_source,
        pool_address=OSTIUM_VAULT_ADDRESS.lower(),
        exchange_address=None,
        internal_id=4,
        internal_exchange_id=4,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="ostium",
        other_data={"vault_protocol": "ostium", "vault_features": ["ostium_like"]},
    )

    pair_universe = create_pair_universe_from_code(
        SOURCE_CHAIN_ID,
        [cctp_bridge_pair, weth_usdc_pair, reverse_cctp_bridge_pair, ostium_vault_pair],
    )

    cctp_exchange = Exchange(
        chain_id=SOURCE_CHAIN_ID,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    uniswap_v3_exchange = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="base",
        exchange_id=2,
        exchange_slug="uniswap-v3",
        address=BASE_UNISWAP_V3_FACTORY,
        exchange_type=ExchangeType.uniswap_v3,
        pair_count=1,
    )

    cctp_exchange_base = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="base",
        exchange_id=3,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    ostium_exchange = Exchange(
        chain_id=SOURCE_CHAIN_ID,
        chain_slug="arbitrum",
        exchange_id=4,
        exchange_slug="ostium",
        address=OSTIUM_VAULT_ADDRESS,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    exchange_universe = ExchangeUniverse({
        cctp_exchange.exchange_id: cctp_exchange,
        uniswap_v3_exchange.exchange_id: uniswap_v3_exchange,
        cctp_exchange_base.exchange_id: cctp_exchange_base,
        ostium_exchange.exchange_id: ostium_exchange,
    })
    pair_universe.exchange_universe = exchange_universe

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={SOURCE_CHAIN_ID, DEST_CHAIN_ID},
        exchanges={cctp_exchange, uniswap_v3_exchange, cctp_exchange_base, ostium_exchange},
        pairs=pair_universe,
        candles=None,
        liquidity=None,
        exchange_universe=exchange_universe,
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
    pass


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    return []
