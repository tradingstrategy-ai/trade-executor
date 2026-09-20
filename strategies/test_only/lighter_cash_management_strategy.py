"""Fixed-fork strategy used to exercise automatic Lighter cash management."""

import datetime
import os
from decimal import Decimal

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.lighter.session import create_lighter_session
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.lighter import (
    create_lighter_cash_management_transfer,
    create_lighter_exchange_account_pair,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
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

# The fixed-fork test replaces this public Lighter session with a sequencer mock.
LIGHTER_SESSION = create_lighter_session()
# The test strategy uses a stable public account index supplied by its environment.
LIGHTER_ACCOUNT_INDEX = int(os.environ.get("LIGHTER_ACCOUNT_INDEX", "125"))


class Parameters:
    """Parameters for the automatic cash-management test strategy."""

    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None
    lighter_cash_management = True
    lighter_safe_cash_buffer_usd = Decimal("20")
    lighter_free_collateral_buffer_usd = Decimal("0")
    lighter_min_transfer_usd = Decimal("1")
    lighter_withdrawal_timeout = 1800


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create one synthetic Ethereum Lighter exchange-account pair."""
    del input
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address=USDC_NATIVE_TOKEN[ChainId.ethereum.value],
        token_symbol="USDC",
        decimals=6,
    )
    lighter_pair = create_lighter_exchange_account_pair(
        quote=usdc,
        account_index=LIGHTER_ACCOUNT_INDEX,
    )
    pair_universe = create_pair_universe_from_code(ChainId.ethereum, [lighter_pair])
    exchange = Exchange(
        chain_id=ChainId.ethereum,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
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
    """Declare no market indicators for this custody-only test strategy."""
    del parameters, indicators, strategy_universe, execution_context


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Create the synthetic position, then return automatic custody trades."""
    pair = input.strategy_universe.get_single_pair()
    reserve_asset = input.strategy_universe.get_reserve_asset()
    position = input.state.portfolio.get_open_position_for_pair(pair)

    if position is None:
        # Position initialisation is a state operation; it is not a custody transfer.
        open_exchange_account_position(
            state=input.state,
            strategy_cycle_at=input.timestamp.to_pydatetime(),
            pair=pair,
            reserve_currency=reserve_asset,
            reserve_amount=Decimal(0),
            notes="Initialise Lighter exchange-account position",
        )
        return []

    if input.execution_context.mode.is_backtesting():
        return []

    position_manager = input.get_position_manager()
    safe_usdc = Decimal(str(position_manager.get_current_cash()))
    return _create_cash_management_trades(input, position, safe_usdc)


def _create_cash_management_trades(
    input: StrategyInput,
    position: TradingPosition,
    safe_usdc: Decimal,
) -> list[TradeExecution]:
    """Return one transfer using the latest treasury-synchronised Safe balance."""
    parameters = input.parameters
    transfer = create_lighter_cash_management_transfer(
        state=input.state,
        position=position,
        strategy_cycle_at=input.timestamp.to_pydatetime(),
        reserve_currency=input.strategy_universe.get_reserve_asset(),
        # PositionManager exposes the latest treasury sync; do not make a
        # direct on-chain balance call from the strategy reference code.
        safe_usdc=safe_usdc,
        safe_cash_buffer_usdc=Decimal(str(parameters.lighter_safe_cash_buffer_usd)),
        free_collateral_buffer_usdc=Decimal(str(parameters.lighter_free_collateral_buffer_usd)),
        minimum_transfer_usdc=Decimal(str(parameters.lighter_min_transfer_usd)),
        session=LIGHTER_SESSION,
    )
    return [transfer] if transfer is not None else []
