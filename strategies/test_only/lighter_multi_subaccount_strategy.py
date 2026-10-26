"""Strategy exercising Lighter master and sub-account cash management.

Used by the fixed-fork black-box test and the mainnet sub-account tutorial.
The Safe-owned Lighter master account is the treasury hub and each
sub-account is traded by an external process; this strategy only moves
custody capital between the Safe, the master and the sub-accounts, see
:py:func:`tradeexecutor.exchange_account.lighter_strategy.decide_lighter_cash_management`.

The module is loaded fresh by every Typer command, so the deployment and the
test can change these environment variables between ``start`` runs:

``LIGHTER_ACCOUNT_INDEXES``
    Comma-separated master account index followed by sub-account indices,
    from the public deployment report. Required.

``LIGHTER_SUB_ACCOUNT_WEIGHTS``
    Comma-separated target weights in sub-account order, for example
    ``0.5,0.5``. Empty means no sub-account rebalancing.

``LIGHTER_SAFE_CASH_BUFFER_USD``
    USDC kept in the Safe. Defaults to 20.

``LIGHTER_TRANSFER_FEE_USD``
    Fee budgeted on top of each master to sub-account transfer. Defaults to 0;
    Lighter documents these transfers as free.
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

from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pairs
from tradeexecutor.exchange_account.lighter_strategy import (
    decide_lighter_cash_management,
    parse_lighter_sub_account_weights,
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


class Parameters:
    """Parameters for the Lighter sub-account cash-management strategy."""

    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None
    lighter_cash_management = True
    lighter_safe_cash_buffer_usd = Decimal(os.environ.get("LIGHTER_SAFE_CASH_BUFFER_USD", "20"))
    lighter_free_collateral_buffer_usd = Decimal("0")
    lighter_min_transfer_usd = Decimal("1")
    lighter_sub_account_weights = parse_lighter_sub_account_weights(os.environ.get("LIGHTER_SUB_ACCOUNT_WEIGHTS", ""))
    lighter_rebalance_band_usd = Decimal("0")
    lighter_transfer_fee_usd = Decimal(os.environ.get("LIGHTER_TRANSFER_FEE_USD", "0"))


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create one exchange-account pair per Lighter account.

    :param input:
        Unused strategy-universe construction input.
    :return:
        Static universe with the master account, the sub-accounts and the USDC reserve.
    """
    del input
    raw = os.environ.get("LIGHTER_ACCOUNT_INDEXES", "")
    indices = [int(part) for part in raw.split(",") if part.strip()]
    if not indices:
        raise ValueError("Set LIGHTER_ACCOUNT_INDEXES to the master and sub-account indices from the deployment report")
    usdc = AssetIdentifier(
        chain_id=ChainId.ethereum.value,
        address=USDC_NATIVE_TOKEN[ChainId.ethereum.value],
        token_symbol="USDC",
        decimals=6,
    )
    pairs = create_lighter_exchange_account_pairs(usdc, master_account_index=indices[0], sub_account_indices=indices[1:])
    exchange = Exchange(
        chain_id=ChainId.ethereum,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="lighter",
        address=LIGHTER_L1_CONTRACT,
        exchange_type=ExchangeType.derive,
        pair_count=len(pairs),
    )
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.ethereum},
        exchanges={exchange},
        pairs=create_pair_universe_from_code(ChainId.ethereum, pairs),
        candles=None,
        liquidity=None,
    )
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """Declare no market indicators for this custody-only strategy.

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
    """Return this cycle's Lighter custody trades.

    :param input:
        Current strategy state, parameters and treasury-synchronised reserve.
    :return:
        Trades of at most one movement kind.
    """
    return decide_lighter_cash_management(input)
