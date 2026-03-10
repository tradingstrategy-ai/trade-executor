"""Cross-chain strategy for Lagoon vault with CCTP bridging + Hypercore vault deposit.

A position-state-based strategy exercising the full cross-chain vault lifecycle:

- Step 1: Bridge USDC from Arbitrum → HyperEVM via CCTP
- Step 2: Deposit bridged USDC into Hypercore HLP vault on HyperEVM
- Step 3 (simulate only): Withdraw from Hypercore vault
- Step 4 (simulate only): Bridge USDC back HyperEVM → Arbitrum

Uses mainnet addresses (Arbitrum chain 42161 + HyperEVM chain 999).

Testnet is **not** supported because CCTP does not have a domain mapping
for HyperEVM testnet (chain 998). The Circle CCTP v2 attestation service
only supports HyperEVM mainnet (domain 19).

The ``decide_trades()`` function uses position state (not cycle counters)
to decide what to do, because ``state.cycle`` is never incremented in
single-shot mode (``trade_immediately + max_cycles=1``).
"""

import datetime
import os
from decimal import Decimal

from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.token import USDC_NATIVE_TOKEN

from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
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

#: Source chain holds reserves (Lagoon vault on Arbitrum)
SOURCE_CHAIN_ID = ChainId.arbitrum
#: Destination chain for Hypercore vault (HyperEVM mainnet)
DEST_CHAIN_ID = ChainId.hyperliquid

#: How much USDC to bridge from Arbitrum to HyperEVM
BRIDGE_AMOUNT = Decimal(os.environ.get("BRIDGE_AMOUNT", "7"))

#: How much USDC to deposit into Hypercore vault
VAULT_DEPOSIT_AMOUNT = Decimal(os.environ.get("VAULT_DEPOSIT_AMOUNT", "5"))

#: How much USDC to bridge back (simulate only)
REVERSE_BRIDGE_AMOUNT = Decimal(os.environ.get("REVERSE_BRIDGE_AMOUNT", "3"))


class Parameters:
    """Strategy parameters for cross-chain Hypercore vault testing."""
    chain_id = ChainId.arbitrum
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
    """Create a cross-chain trading universe with CCTP bridge + Hypercore vault.

    Pairs:
    1. Forward CCTP bridge: Arbitrum USDC → HyperEVM USDC
    2. Hypercore vault: deposit USDC into HLP vault on HyperEVM
    3. Reverse CCTP bridge: HyperEVM USDC → Arbitrum USDC (for withdrawal flow)
    """
    usdc_arb_address = USDC_NATIVE_TOKEN[SOURCE_CHAIN_ID.value]
    usdc_hyper_address = USDC_NATIVE_TOKEN[DEST_CHAIN_ID.value]

    # Reserve token on source chain (Arbitrum)
    usdc_arb = AssetIdentifier(
        chain_id=SOURCE_CHAIN_ID.value,
        address=usdc_arb_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Bridged token on destination chain (HyperEVM)
    usdc_hyper = AssetIdentifier(
        chain_id=DEST_CHAIN_ID.value,
        address=usdc_hyper_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Forward CCTP bridge pair: Arbitrum → HyperEVM
    forward_bridge_pair = TradingPairIdentifier(
        base=usdc_hyper,
        quote=usdc_arb,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge (Arb → HyperEVM)",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": DEST_CHAIN_ID.value,
        },
    )

    # Hypercore vault pair: deposit USDC into HLP vault on HyperEVM
    vault_address = HLP_VAULT_ADDRESS["mainnet"]
    hypercore_vault_pair = create_hypercore_vault_pair(
        quote=usdc_hyper,
        vault_address=vault_address,
        internal_id=2,
    )

    # Reverse CCTP bridge pair: HyperEVM → Arbitrum
    reverse_bridge_pair = TradingPairIdentifier(
        base=usdc_arb,
        quote=usdc_hyper,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=3,
        internal_exchange_id=3,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge (HyperEVM → Arb)",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": SOURCE_CHAIN_ID.value,
        },
    )

    pair_universe = create_pair_universe_from_code(
        SOURCE_CHAIN_ID,
        [forward_bridge_pair, hypercore_vault_pair, reverse_bridge_pair],
    )

    # CCTP bridge exchange on Arbitrum (for forward bridge)
    cctp_exchange_arb = Exchange(
        chain_id=SOURCE_CHAIN_ID,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    # Hypercore exchange on HyperEVM (vault deposits)
    hypercore_exchange = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="hyperliquid",
        exchange_id=2,
        exchange_slug="hypercore",
        address=hypercore_vault_pair.exchange_address,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=1,
    )

    # CCTP bridge exchange on HyperEVM (for reverse bridge)
    cctp_exchange_hyper = Exchange(
        chain_id=DEST_CHAIN_ID,
        chain_slug="hyperliquid",
        exchange_id=3,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    exchange_universe = ExchangeUniverse({
        cctp_exchange_arb.exchange_id: cctp_exchange_arb,
        hypercore_exchange.exchange_id: hypercore_exchange,
        cctp_exchange_hyper.exchange_id: cctp_exchange_hyper,
    })
    pair_universe.exchange_universe = exchange_universe

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={SOURCE_CHAIN_ID, DEST_CHAIN_ID},
        exchanges={cctp_exchange_arb, hypercore_exchange, cctp_exchange_hyper},
        pairs=pair_universe,
        candles=None,
        liquidity=None,
        exchange_universe=exchange_universe,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc_arb],
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
    """Position-state-based trade decisions.

    Uses open/closed position state to decide the next action:

    1. No bridge position → bridge USDC to HyperEVM
    2. Bridge exists, no vault position → deposit into Hypercore vault
    3. Both exist → no-op (mainnet) or withdraw + bridge back (simulate)
    """
    position_manager = input.get_position_manager()
    state = input.state

    all_positions = list(state.portfolio.open_positions.values()) + \
        list(state.portfolio.closed_positions.values())

    # Check for existing positions by type
    has_forward_bridge = any(
        p.pair.is_cctp_bridge() and p.pair.quote.chain_id == SOURCE_CHAIN_ID.value
        for p in all_positions
    )
    has_vault_open = any(
        p.is_vault() and p.pair.other_data.get("vault_protocol") == "hypercore"
        for p in state.portfolio.open_positions.values()
    )
    has_vault_closed = any(
        p.is_vault() and p.pair.other_data.get("vault_protocol") == "hypercore"
        for p in state.portfolio.closed_positions.values()
    )
    has_reverse_bridge = any(
        p.pair.is_cctp_bridge() and p.pair.quote.chain_id == DEST_CHAIN_ID.value
        for p in all_positions
    )

    universe = input.strategy_universe

    # Step 1: Bridge USDC from Arbitrum to HyperEVM
    if not has_forward_bridge:
        pair = universe.get_pair_by_human_description(
            (SOURCE_CHAIN_ID, "cctp-bridge", "USDC", "USDC"),
        )
        return position_manager.open_spot(pair, value=BRIDGE_AMOUNT)

    # Step 2: Deposit into Hypercore vault on HyperEVM
    if has_forward_bridge and not has_vault_open and not has_vault_closed:
        pair = universe.get_pair_by_human_description(
            (DEST_CHAIN_ID, "hypercore", "HYPERCORE-VAULT", "USDC"),
        )
        return position_manager.open_spot(pair, value=VAULT_DEPOSIT_AMOUNT)

    # Step 3 (simulate only): Withdraw from Hypercore vault
    if has_vault_open:
        vault_pos = next(
            p for p in state.portfolio.open_positions.values()
            if p.is_vault() and p.pair.other_data.get("vault_protocol") == "hypercore"
        )
        return position_manager.close_position(vault_pos)

    # Step 4 (simulate only): Bridge USDC back from HyperEVM to Arbitrum
    if has_vault_closed and not has_reverse_bridge:
        pair = universe.get_pair_by_human_description(
            (DEST_CHAIN_ID, "cctp-bridge", "USDC", "USDC"),
        )
        return position_manager.open_spot(pair, value=REVERSE_BRIDGE_AMOUNT)

    # No-op: all steps completed
    return []
