"""Fixed-fork strategy used to exercise automatic Lighter cash management."""

import datetime
from decimal import Decimal

from eth_defi.lighter.constants import LIGHTER_L1_CONTRACT
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.lighter import (
    create_lighter_exchange_account_pair,
)
from tradeexecutor.exchange_account.cash_manager import ExchangeCashManager, ExchangeCashSnapshot
from tradeexecutor.exchange_account.state import (
    create_exchange_account_transfer,
    open_exchange_account_position,
)
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

#: Synthetic public account index returned by the fixed-fork Lighter mock.
LIGHTER_ACCOUNT_INDEX = 126

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


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create one synthetic Ethereum Lighter exchange-account pair.

    :param input:
        Unused strategy-universe construction input.
    :return:
        Static universe containing the Lighter account and USDC reserve.
    """
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
    """Declare no market indicators for this custody-only test strategy.

    :param parameters:
        Unused strategy parameters.
    :param indicators:
        Empty indicator collection retained by the engine interface.
    :param strategy_universe:
        Static Lighter strategy universe.
    :param execution_context:
        Current execution context.
    """
    del parameters, indicators, strategy_universe, execution_context


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Create the synthetic position, then return automatic custody trades.

    :param input:
        Current strategy state, parameters, and treasury-synchronised reserve.
    :return:
        At most one transfer selected by the Lighter cash manager.
    """
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
    snapshot = ExchangeCashSnapshot(
        safe_usdc=Decimal(str(position_manager.get_current_cash())),
        exchange_available_usdc=Decimal(str(position_manager.get_exchange_account_available_balance(pair))),
        pending_deposits_usdc=Decimal(str(input.state.sync.treasury.pending_deposits or 0)),
        pending_redemptions_usdc=Decimal(str(input.state.sync.treasury.pending_redemptions or 0)),
    )
    manager = ExchangeCashManager(
        safe_cash_buffer_usdc=Decimal(str(input.parameters.lighter_safe_cash_buffer_usd)),
        free_collateral_buffer_usdc=Decimal(str(input.parameters.lighter_free_collateral_buffer_usd)),
        minimum_transfer_usdc=Decimal(str(input.parameters.lighter_min_transfer_usd)),
    )
    transfer_pending = any(
        trade.is_external_account_transfer_pending()
        for candidate in input.state.portfolio.get_open_and_frozen_positions()
        for trade in candidate.trades.values()
    )
    decision = manager.decide(snapshot, transfer_pending=transfer_pending)
    if not decision.should_transfer:
        return []
    return [create_exchange_account_transfer(
        state=input.state,
        position=position,
        strategy_cycle_at=input.timestamp.to_pydatetime(),
        reserve_currency=reserve_asset,
        amount=decision.amount_usdc,
        deposit=decision.direction == "deposit",
        notes="Automatic Lighter cash management",
        metadata={
            "direction": decision.direction,
            "protocol": "lighter",
            "policy": "exchange_cash_manager",
        },
    )]
