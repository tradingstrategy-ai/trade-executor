"""Testnet strategy for cross-chain Lagoon vault with CCTP bridging + Uniswap v3.

A 5-cycle strategy exercising the full CCTP bridge round-trip on testnets:

- Cycle 1: Bridge 5 USDC from Arbitrum Sepolia -> Base Sepolia via CCTP
- Cycle 2: Swap bridged USDC -> WETH on Base Sepolia via Uniswap v3
- Cycle 3: Sell WETH -> USDC on Base Sepolia via Uniswap v3
- Cycle 4: Bridge ~3 USDC from Base Sepolia -> Arbitrum Sepolia via reverse CCTP
- Cycle 5: No-op (verify funds returned)

Uses testnet addresses (Arbitrum Sepolia + Base Sepolia).
WETH/USDC pool address must be provided via ``WETH_USDC_POOL_BASE_SEPOLIA``
environment variable since testnet pools are not well-known.
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

#: Source chain holds reserves
SOURCE_CHAIN_ID = ChainId.arbitrum_sepolia
#: Destination chain holds bridged USDC
DEST_CHAIN_ID = ChainId.base_sepolia

#: Uniswap v3 factory on Base Sepolia
BASE_SEPOLIA_UNISWAP_V3_FACTORY = UNISWAP_V3_DEPLOYMENTS["base_sepolia"]["factory"]

#: How much USDC to bridge
BRIDGE_AMOUNT = Decimal(5)

#: How much USDC to swap on Uniswap v3 (leave some for gas/fees)
SWAP_AMOUNT = Decimal(4)

#: How much USDC to bridge back from Base Sepolia to Arbitrum Sepolia
REVERSE_BRIDGE_AMOUNT = Decimal(3)


class Parameters:
    """Strategy parameters for testnet CCTP bridge + Uniswap v3 testing."""
    chain_id = ChainId.arbitrum_sepolia
    initial_cash = 10
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
    """Create a trading universe with CCTP bridge pair + WETH/USDC on Base Sepolia.

    Uses testnet addresses. WETH/USDC pool must be set via
    ``WETH_USDC_POOL_BASE_SEPOLIA`` environment variable.
    """

    usdc_source_address = USDC_NATIVE_TOKEN[SOURCE_CHAIN_ID.value]
    usdc_dest_address = USDC_NATIVE_TOKEN[DEST_CHAIN_ID.value]
    weth_address = WRAPPED_NATIVE_TOKEN[DEST_CHAIN_ID.value]

    # WETH/USDC pool on Base Sepolia (Uniswap v3, 0.05% fee tier)
    # Default: well-known pool at 0x94bfc...EeC0
    weth_usdc_pool = os.environ.get("WETH_USDC_POOL_BASE_SEPOLIA", "0x94bfc0574FF48E92cE43d495376C477B1d0EEeC0")

    # Reserve token on source chain (Arbitrum Sepolia)
    usdc_source = AssetIdentifier(
        chain_id=SOURCE_CHAIN_ID.value,
        address=usdc_source_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Bridged token on destination chain (Base Sepolia)
    usdc_dest = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=usdc_dest_address,
        token_symbol="USDC",
        decimals=6,
    )

    # WETH on Base Sepolia
    weth = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=weth_address,
        token_symbol="WETH",
        decimals=18,
    )

    # CCTP bridge pair: base=destination USDC, quote=source USDC
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

    # WETH/USDC on Base Sepolia Uniswap v3 (0.05% fee tier)
    # Token order: USDC (0x036C...) < WETH (0x4200...) so USDC=token0, WETH=token1
    # Since base=WETH != token0=USDC, we need reverse_token_order=True
    weth_usdc_pair = TradingPairIdentifier(
        base=weth,
        quote=usdc_dest,
        pool_address=weth_usdc_pool,
        exchange_address=BASE_SEPOLIA_UNISWAP_V3_FACTORY,
        internal_id=2,
        internal_exchange_id=2,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
        exchange_name="Uniswap v3 (Base Sepolia)",
        reverse_token_order=True,
    )

    # Reverse CCTP bridge pair: bridge USDC from Base Sepolia back to Arbitrum Sepolia
    # base=destination USDC (Arb Sepolia), quote=source USDC (Base Sepolia — where tokens are burned)
    reverse_cctp_bridge_pair = TradingPairIdentifier(
        base=usdc_source,
        quote=usdc_dest,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=3,
        internal_exchange_id=3,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge (Base Sepolia)",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": SOURCE_CHAIN_ID.value,
        },
    )

    pair_universe = create_pair_universe_from_code(
        SOURCE_CHAIN_ID, [cctp_bridge_pair, weth_usdc_pair, reverse_cctp_bridge_pair],
    )

    cctp_exchange = Exchange(
        chain_id=SOURCE_CHAIN_ID,
        chain_slug="arbitrum_sepolia",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    uniswap_v3_exchange = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="base_sepolia",
        exchange_id=2,
        exchange_slug="uniswap-v3",
        address=BASE_SEPOLIA_UNISWAP_V3_FACTORY,
        exchange_type=ExchangeType.uniswap_v3,
        pair_count=1,
    )

    # CCTP bridge exchange on Base Sepolia (for reverse bridge direction)
    cctp_exchange_base = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="base_sepolia",
        exchange_id=3,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    exchange_universe = ExchangeUniverse({
        cctp_exchange.exchange_id: cctp_exchange,
        uniswap_v3_exchange.exchange_id: uniswap_v3_exchange,
        cctp_exchange_base.exchange_id: cctp_exchange_base,
    })
    pair_universe.exchange_universe = exchange_universe

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={SOURCE_CHAIN_ID, DEST_CHAIN_ID},
        exchanges={cctp_exchange, uniswap_v3_exchange, cctp_exchange_base},
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
    """No indicators needed."""
    pass


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """5-cycle strategy: bridge -> buy -> sell -> bridge back -> no-op.

    - Cycle 1: Bridge USDC from Arbitrum Sepolia to Base Sepolia
    - Cycle 2: Buy WETH on Base Sepolia
    - Cycle 3: Sell WETH -> USDC on Base Sepolia
    - Cycle 4: Bridge USDC from Base Sepolia back to Arbitrum Sepolia
    - Cycle 5: No-op (verify funds returned)

    Uses position state (not cycle counters) to decide what to do, because
    ``state.cycle`` is never incremented in single-shot mode
    (``trade_immediately + max_cycles=1``).
    """
    position_manager = input.get_position_manager()
    state = input.state
    universe = input.strategy_universe

    all_positions = list(state.portfolio.open_positions.values()) + \
        list(state.portfolio.closed_positions.values())

    has_open_weth = any(
        p.pair.base.token_symbol == "WETH"
        for p in state.portfolio.open_positions.values()
    )
    has_closed_weth = any(
        p.pair.base.token_symbol == "WETH"
        for p in state.portfolio.closed_positions.values()
    )
    has_forward_bridge = any(
        p.pair.is_cctp_bridge() and p.pair.quote.chain_id == SOURCE_CHAIN_ID.value
        for p in all_positions
    )
    has_reverse_bridge = any(
        p.pair.is_cctp_bridge() and p.pair.quote.chain_id == DEST_CHAIN_ID.value
        for p in all_positions
    )

    # Cycle 1: Bridge USDC to Base Sepolia
    if not has_forward_bridge:
        pair = universe.get_pair_by_human_description(
            (SOURCE_CHAIN_ID, "cctp-bridge", "USDC", "USDC"),
        )
        return position_manager.open_spot(pair, value=BRIDGE_AMOUNT)

    # Cycle 2: Buy WETH on Base Sepolia
    if not has_open_weth and not has_closed_weth:
        pair = universe.get_pair_by_human_description(
            (DEST_CHAIN_ID, "uniswap-v3", "WETH", "USDC"),
        )
        return position_manager.open_spot(pair, value=SWAP_AMOUNT)

    # Cycle 3: Close WETH position (sell WETH -> USDC on Base Sepolia)
    if has_open_weth:
        weth_pos = next(
            p for p in state.portfolio.open_positions.values()
            if p.pair.base.token_symbol == "WETH"
        )
        return position_manager.close_position(weth_pos)

    # Cycle 4: Bridge USDC back from Base Sepolia to Arbitrum Sepolia
    if has_closed_weth and not has_reverse_bridge:
        pair = universe.get_pair_by_human_description(
            (DEST_CHAIN_ID, "cctp-bridge", "USDC", "USDC"),
        )
        return position_manager.open_spot(pair, value=REVERSE_BRIDGE_AMOUNT)

    # Cycle 5: No-op
    return []
