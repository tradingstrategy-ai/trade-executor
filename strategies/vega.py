"""Strategy for Derive vault on mainnet via CLI start command.

A strategy with a fixed universe of one Derive exchange account pair
targeting the mainnet Derive vault on Derive (Lyra) chain.

On the first cycle it opens the exchange account position;
on subsequent cycles it returns no trades (valuation happens via sync).

The Derive subaccount ID is discovered automatically from the Derive API
at universe creation time using ``DERIVE_OWNER_PRIVATE_KEY`` and
``DERIVE_SESSION_PRIVATE_KEY`` environment variables.

Uses ``TradeRouting.default`` with the real execution model so that
GenericRouting / EthereumPairConfigurator handle pricing and valuation.
"""

import datetime
import logging
import os
from decimal import Decimal

from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import (AssetIdentifier,
                                            TradingPairIdentifier,
                                            TradingPairKind)
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, create_pair_universe_from_code)
from tradeexecutor.strategy.universe_model import UniverseOptions

logger = logging.getLogger(__name__)

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

#: Derive (Lyra) mainnet
CHAIN_ID = ChainId.derive


class Parameters:
    chain_id = ChainId.derive
    initial_cash = 10_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


# USDC on Derive mainnet
USDC = AssetIdentifier(
    chain_id=CHAIN_ID.value,
    address="0x6879287835A86F50f784313dBEd5E5cCC5bb8481",
    token_symbol="USDC",
    decimals=6,
)

# Synthetic asset representing the Derive vault account value
DERIVE_ACCOUNT = AssetIdentifier(
    chain_id=CHAIN_ID.value,
    address="0x0000000000000000000000000000000000D371E0",
    token_symbol="DERIVE-ACCOUNT",
    decimals=6,
)


def create_trading_universe(
    ts: datetime.datetime,
    client: BaseClient,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create universe with a single Derive exchange account pair.

    Discovers the real subaccount ID from the Derive API so that
    sync and valuation target the correct account.
    """
    from tradeexecutor.exchange_account.derive import discover_derive_subaccount_id

    is_testnet = os.environ.get("DERIVE_NETWORK", "mainnet") == "testnet"
    subaccount_id = discover_derive_subaccount_id()

    # Encode subaccount ID into pool/exchange addresses for traceability
    subaccount_hex = hex(subaccount_id)

    exchange_account_pair = TradingPairIdentifier(
        base=DERIVE_ACCOUNT,
        quote=USDC,
        pool_address=subaccount_hex,
        exchange_address=subaccount_hex,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": subaccount_id,
            "exchange_is_testnet": is_testnet,
        },
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [exchange_account_pair])

    derive_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="derive",
        exchange_id=1,
        exchange_slug="derive",
        address=subaccount_hex,
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

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[USDC],
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
    """Ensure one exchange account position exists.

    On the first cycle, open the position via direct state injection.
    On subsequent cycles, do nothing (valuation is handled by sync).

    Always returns [] because exchange account trades are spoofed
    directly on the state and must never reach routing/execution.
    """
    state = input.state
    timestamp = input.timestamp

    # Check if position already exists
    for pos in state.portfolio.open_positions.values():
        if pos.pair.is_exchange_account():
            return []

    # Get the exchange account pair from the universe (has real subaccount ID)
    pair = None
    for p in input.strategy_universe.iterate_pairs():
        if p.is_exchange_account():
            pair = p
            break

    if pair is None:
        logger.error("No exchange account pair found in universe")
        return []

    # First cycle: create exchange account position
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=timestamp.to_pydatetime(),
        pair=pair,
        reserve_currency=USDC,
        reserve_amount=Decimal(1),
        notes="Initial Derive vault mainnet position",
    )

    return []
