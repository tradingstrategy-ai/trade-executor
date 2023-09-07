"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet, OutOfSimulatedBalance
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.execution_model import ExecutionModel, AutoClosingOrderUnsupported

logger = logging.getLogger(__name__)


class BacktestExecutionFailed(Exception):
    """Something went wrong in the backtest simulation."""


def fix_sell_token_amount(
        current_balance: Decimal,
        order_quantity: Decimal,
        epsilon=Decimal(10**-9)
) -> Tuple[Decimal, bool]:
    """Fix rounding errors that may cause wallet dust overflow.

    TODO: This should be handled other part of the system.

    :return:
        (new amount, was fixed) tuple
    """

    assert isinstance(current_balance, Decimal)
    assert isinstance(order_quantity, Decimal)
    assert order_quantity < 0

    # Not trying to sell more than we have
    if abs(order_quantity) <= current_balance:
        return order_quantity, False

    # We are trying to sell more we have
    diff = abs(current_balance + order_quantity)
    if diff <= epsilon:
        # Fix to be within the epsilon diff
        logger.warning("Fixing token sell amount to be within the epsilon. Wallet balance: %s, sell order quantity: %s, diff: %s",
                       current_balance,
                       order_quantity,
                       diff
                       )
        return -current_balance, True

    logger.warning("Trying to sell more than we have. Wallet balance: %s, sell order quantity: %s, diff: %s, epsilon: %s",
                   current_balance,
                   order_quantity,
                   diff,
                   epsilon
                   )
    return order_quantity, False


class BacktestExecutionModel(ExecutionModel):
    """Simulate trades against historical data."""

    def __init__(self,
                 wallet: SimulatedWallet,
                 max_slippage: float,
                 lp_fees: float=0.0030,
                 stop_loss_data_available=False,
                 ):
        self.wallet = wallet
        self.max_slippage = max_slippage
        self.lp_fees = lp_fees
        self.stop_loss_data_available = stop_loss_data_available

    def is_live_trading(self):
        return False

    def is_stop_loss_supported(self):
        return self.stop_loss_data_available

    def preflight_check(self):
        pass

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising backtest execution model")

    def simulate_spot(self, state: State, trade: TradeExecution) -> Tuple[Decimal, Decimal, bool]:
        """Spot market translation simulation with a simulated wallet.

        Check that the trade "executes" against the simulated wallet

        :param state:
            Backtester state

        :param trade:
            Trade to be executed

        :return:
            (ecuted_quantity, executed_reserve, sell_amount_epsilon_fix) tuple

        :raise OutOfSimulatedBalance:
            Wallet does not have enough tokens to do the trade
        """

        #
        base = trade.pair.base
        quote = trade.pair.quote
        reserve = trade.reserve_currency

        base_balance = self.wallet.get_balance(base.address)
        quote_balance = self.wallet.get_balance(quote.address)
        reserve_balance = self.wallet.get_balance(reserve.address)

        position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)

        sell_amount_epsilon_fix = False

        if trade.is_buy():
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity
        else:
            assert position and position.is_open(), f"Tried to execute sell on position that is not open: {trade}"
            executed_quantity, sell_amount_epsilon_fix = fix_sell_token_amount(base_balance, trade.planned_quantity)
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price))

        if trade.is_buy():
            self.wallet.update_balance(base.address, executed_quantity)
            self.wallet.update_balance(reserve.address, -executed_reserve)
        else:
            self.wallet.update_balance(base.address, executed_quantity)
            self.wallet.update_balance(reserve.address, executed_reserve)

        assert abs(executed_quantity) > 0, f"Expected executed_quantity for the trade to be above zero, got executed_quantity:{executed_quantity}, planned_quantity:{trade.planned_quantity}, trade is {trade}"

        return executed_quantity, executed_reserve, sell_amount_epsilon_fix

    def simulate_leverage(self, state: State, trade: TradeExecution):
        """Leverage simulation with a simulated wallet.

        Check that the trade "executes" against the simulated wallet

        TODO: currently doesn't support leverage long yet

        :param state:
            Backtester state

        :param trade:
            Trade to be executed

        :return:
            (ecuted_quantity, executed_reserve, sell_amount_epsilon_fix) tuple

        :raise OutOfSimulatedBalance:
            Wallet does not have enough tokens to do the trade
        """
        assert trade.is_short(), "Leverage long is not supported yet"

        borrowed_address = trade.pair.base.address
        collateral_address = trade.pair.quote.address
        reserve_address = trade.reserve_currency.address

        # position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)
        executed_reserve = trade.planned_reserve
        executed_quantity = trade.planned_quantity
        executed_collateral_consumption = trade.planned_collateral_consumption
        executed_collateral_allocation = trade.planned_collateral_allocation

        # Here is a mismatch between spot and leverage:
        # base.underlying token, or executed_quantity, never appears in the wallet
        # as we do loan based trading

        self.wallet.update_balance(reserve_address, -executed_reserve)

        # The leveraged tokens appear in the wallet

        # aToken amount is original deposit + any leverage we do

        self.wallet.update_balance(collateral_address, executed_collateral_consumption)
        self.wallet.update_balance(collateral_address, executed_reserve)

        # vToken amount us whatever quantity we execute
        if trade.is_short():
            self.wallet.update_balance(borrowed_address, -executed_quantity)
        else:
            self.wallet.update_balance(borrowed_address, executed_quantity)

        # move all leftover atoken to reserve when the position is closing
        # TODO: check if this is correct place to do this
        if self.wallet.get_balance(borrowed_address) == 0:
            remaining_collateral = self.wallet.get_balance(collateral_address)
            self.wallet.update_balance(reserve_address, remaining_collateral)
            self.wallet.update_balance(collateral_address, -remaining_collateral)

        assert abs(executed_quantity) > 0, f"Expected executed_quantity for the trade to be above zero, got executed_quantity:{executed_quantity}, planned_quantity:{trade.planned_quantity}, trade is {trade}"

        # for leverage short, we use collateral token as the reserve currency
        # so return executed_collateral_quantity here to correctly calculate the price
        return executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption

    def simulate_trade(self,
                       ts: datetime.datetime,
                       state: State,
                       idx: int,
                       trade: TradeExecution) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
        """Set backtesting trade state from planned to executed.
        
        Currently, always executes trades "perfectly" i.e. no different slipppage
        that was planned, etc.

        :param ts:
            Strategy cycle timestamp

        :param state:
            Current backtesting state

        :param idx:
            Index of the trade to be executed on this cycle

        :param trade:
            The actual trade

        :return:
            Executed quantity and executed reserve amounts
        """

        assert trade.get_status() == TradeStatus.started

        # In the backtesting simulation,
        # execution happens always perfectly
        # without any lag
        trade.started_at = trade.opened_at

        state.mark_broadcasted(ts, trade)

        executed_quantity = executed_reserve = sell_amount_epsilon_fix = Decimal(0)
        executed_collateral_allocation = executed_collateral_consumption = None

        try:
            if trade.is_spot() or trade.is_credit_supply():
                executed_quantity, executed_reserve, sell_amount_epsilon_fix = self.simulate_spot(state, trade)
            elif trade.is_leverage():
                executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption = self.simulate_leverage(state, trade)
            else:
                raise NotImplementedError(f"Does not know how to simulate: {trade}")

            trade.executed_loan_update = trade.planned_loan_update

        except OutOfSimulatedBalance as e:
            # Better error messages to helping out why backtesting failed

            position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)

            base = trade.pair.base
            quote = trade.pair.quote
            reserve = trade.reserve_currency

            base_balance = self.wallet.get_balance(base.address)
            quote_balance = self.wallet.get_balance(quote.address)
            reserve_balance = self.wallet.get_balance(reserve.address)

            if trade.is_buy():
                # Give a hint to the user
                extra_help_message = f"---\n" \
                                     f"Tip:" \
                                     f"This is a buy trade that failed.\n" \
                                     f"It means that the strategy had less cash to make purchases that it expected.\n" \
                                     f"It may happen during multiple rebalance operations, as the strategy model might not account properly the trading fees when\n" \
                                     f"it estimates the available cash in hand to make buys and sells for rebalancing operations.\n" \
                                     f"Try increasing the strategy cash buffer to see if it solves the problem.\n"
            else:
                extra_help_message = ""

            raise BacktestExecutionFailed(f"\n"
                f"  Trade #{idx} failed on strategy cycle {ts}\n"
                f"  Execution of trade {trade} failed.\n"
                f"  Pair: {trade.pair}.\n"
                f"  Trade type: {trade.trade_type.name}.\n"
                f"  Trade quantity: {trade.planned_quantity}, reserve: {trade.planned_reserve} {trade.reserve_currency}.\n"
                f"  Wallet base balance: {base_balance} {base.token_symbol} ({base.address}).\n"
                f"  Wallet quote balance: {quote_balance} {quote.token_symbol} ({quote.address}).\n"
                f"  Wallet reserve balance: {reserve_balance} {reserve.token_symbol} ({reserve.address}).\n"
                f"  Executed base amount: {executed_quantity} {base.token_symbol} ({base.address})\n"
                f"  Executed reserve amount: {executed_reserve} {reserve.token_symbol} ({reserve.address})\n"
                f"  Planned base amount: {trade.planned_quantity} {base.token_symbol} ({base.address})\n"
                f"  Planned reserve amount: {trade.planned_reserve} {reserve.token_symbol} ({reserve.address})\n"
                f"  Existing position quantity: {position and position.get_quantity() or '-'} {base.token_symbol}\n"
                f"  Sell amount epsilon fix applied: {sell_amount_epsilon_fix}.\n"
                f"  Out of balance: {e}\n"
                f"  {extra_help_message}\n"
            ) from e

        return executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption

    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: BacktestRoutingModel,
                       routing_state: BacktestRoutingState,
                       check_balances=False):
        """Execute the trades on a simulated environment.

        Calculates price impact based on historical data
        and fills the expected historical trade output.

        :param check_balances:
            Raise an error if we run out of balance to perform buys in some point.
        """
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_model, BacktestRoutingModel)
        assert isinstance(routing_state, BacktestRoutingState)

        state.start_execution_all(ts, trades, max_slippage=0)

        routing_model.setup_trades(
            routing_state,
            trades,
            check_balances=check_balances)

        # Check that backtest does not try to execute stop loss / take profit
        # trades when data is not available
        for t in trades:
            position = state.portfolio.open_positions.get(t.position_id)
            if position and position.has_automatic_close():
                # Check that we have stop loss data available
                # for backtesting
                if not self.is_stop_loss_supported():
                    raise AutoClosingOrderUnsupported("Trade was marked with stop loss/take profit even though backtesting trading universe does not have price feed for stop loss checks available. Remember to use the stop_loss_time_bucket parameter or equivalent when you create your trading universe to avoid this error.")

        for idx, trade in enumerate(trades):

            # 3. Simulate tx broadcast
            executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption = self.simulate_trade(ts, state, idx, trade)

            # TODO: Use colleteral values here

            # 4. execution is dummy operation where planned execution becomes actual execution
            # Assume we always get the same execution we planned
            if executed_quantity:
                if trade.is_short():
                    executed_price = trade.planned_price
                else:
                    executed_price = float(abs(executed_reserve / executed_quantity))

            else:
                executed_price = 0

            state.mark_trade_success(
                ts,
                trade,
                executed_price,
                executed_quantity,
                executed_reserve,
                lp_fees=trade.lp_fees_estimated,
                native_token_price=1,
                executed_collateral_allocation=executed_collateral_allocation,
                executed_collateral_consumption=executed_collateral_consumption,
            )

    def get_routing_state_details(self) -> dict:
        return {"wallet": self.wallet}

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        raise NotImplementedError()