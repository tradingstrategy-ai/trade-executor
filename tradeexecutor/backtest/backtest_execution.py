"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from tabulate import tabulate

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet, OutOfSimulatedBalance
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.types import Percent
from tradeexecutor.strategy.account_correction import calculate_total_assets
from tradeexecutor.strategy.execution_model import ExecutionModel, AutoClosingOrderUnsupported
from tradeexecutor.strategy.generic.generic_router import GenericRouting, GenericRoutingState
from tradeexecutor.strategy.interest import set_interest_checkpoint

logger = logging.getLogger(__name__)


class BacktestExecutionFailed(Exception):
    """Something went wrong in the backtest simulation."""


def fix_sell_token_amount(
        current_balance: Decimal,
        order_quantity: Decimal,
        epsilon=Decimal(10 ** -9)
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


class BacktestExecution(ExecutionModel):
    """Simulate trades against historical data."""

    def __init__(self,
                 wallet: SimulatedWallet,
                 max_slippage: Percent = 0.01,
                 lp_fees: Percent = 0.0030,
                 stop_loss_data_available=False,
                 ):
        self.wallet = wallet
        self.max_slippage = max_slippage
        self.lp_fees = lp_fees
        self.stop_loss_data_available = stop_loss_data_available

    def get_safe_latest_block(self):
        return None

    def get_balance_address(self):
        return None

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

        # More credit supply to its own function
        assert trade.is_spot() or trade.is_credit_supply(), f"simulate_spot(): received a trade that is not spot {trade}"
        # assert trade.pair.is_spot()

        #
        base = trade.pair.base
        # quote = trade.pair.quote
        reserve = trade.reserve_currency

        base_balance = self.wallet.get_balance(base.address)
        # quote_balance = self.wallet.get_balance(quote.address)
        # reserve_balance = self.wallet.get_balance(reserve.address)

        position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)

        sell_amount_epsilon_fix = False

        if trade.is_buy():
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity
        else:
            if not position or not position.is_open():
                logger.error("Selling closed position: %s, trade %s", position, trade)
                logger.error("Current positions")
                for p in state.portfolio.get_open_and_frozen_positions():
                    logger.error("Position %s", p)
                if position:
                    for t in position.trades:
                        logger.error("Position has earlier trade %s", t)
                    last_trade = position.trades[-1] if position.trades else None
                else:
                    last_trade = None
                raise AssertionError(f"Tried to execute sell on position {position} that is not open. This trade is {trade}, pair {trade.pair}, trade id: {trade.trade_id}, position id: {trade.position_id}, last trade was {last_trade}")
            executed_quantity, sell_amount_epsilon_fix = fix_sell_token_amount(base_balance, trade.planned_quantity)
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price))


        if trade.is_buy():
            # Will take also this path for credit supplies
            if trade.is_credit_supply():
                type = "credit supply"
            else:
                type = "spot buy"
            self.wallet.update_balance(base, executed_quantity, f"{type} trade #{trade.trade_id}")
            self.wallet.update_balance(reserve, -executed_reserve, f"{type} trade #{trade.trade_id}")
        else:
            if trade.is_credit_supply():
                type = "credit recall"
            else:
                type = "spot sell"
            self.wallet.update_balance(base, executed_quantity, f"{type} #{trade.trade_id}")
            self.wallet.update_balance(reserve, executed_reserve, f"{type} #{trade.trade_id}")

        assert abs(
            executed_quantity) > 0, f"Expected executed_quantity for the trade to be above zero, got executed_quantity:{executed_quantity}, planned_quantity:{trade.planned_quantity}, trade is {trade}"

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

        # TODO: Correctly use fix_sell_token_amount() here to work around dust issues

        borrowed_token = trade.pair.base
        collateral_token = trade.pair.quote
        reserve_token = trade.reserve_currency

        # position = state.portfolio.get_existing_open_position_by_trading_pair(trade.pair)
        executed_reserve = trade.planned_reserve
        executed_quantity = trade.planned_quantity
        executed_collateral_consumption = trade.planned_collateral_consumption
        executed_collateral_allocation = trade.planned_collateral_allocation

        assert isinstance(executed_reserve, Decimal)
        assert isinstance(executed_quantity, Decimal)
        assert isinstance(executed_collateral_consumption, Decimal)
        assert isinstance(executed_collateral_allocation, Decimal)

        logger.info("simulate_leverage(): wallet balances before updating for %s:\n%s", trade.get_short_label(), self.wallet.get_all_balances())

        # Here is a mismatch between spot and leverage:
        # base.underlying token, or executed_quantity, never appears in the wallet
        # as we do loan based trading

        self.wallet.update_balance(reserve_token, -executed_reserve, f"trade #{trade.trade_id} reserve updates")

        # The leveraged tokens appear in the wallet
        # aToken amount is original deposit + any leverage we do

        self.wallet.update_balance(collateral_token, executed_collateral_consumption, f"collateral consumption trade #{trade.trade_id}")
        self.wallet.update_balance(collateral_token, executed_reserve, f"reserves trade #{trade.trade_id}")

        # vToken amount us whatever quantity we execute.
        # When we short we gain more vToken (executed quantity), but executed quantity is negative for sell
        self.wallet.update_balance(borrowed_token, -executed_quantity, f"executed quantity trade #{trade.trade_id}")

        # <Close short #2
        #    0.3003021039165400376391259260 WETH at 1664.99 USD, broadcasted phase
        #    collateral consumption: -501.5045135406218656282035903 USDC, collateral allocation: -496.9954864593781343405713871 USDC
        #    reserve: 0
        #    >
        # remaining_collateral = self.wallet.get_balance(collateral_address)
        # import ipdb ; ipdb.set_trace()
        collateral_token_change = executed_collateral_allocation

        if collateral_token_change is not None:
            # Convert reserve to aToken
            self.wallet.update_balance(reserve_token, -collateral_token_change, f"Depositing/redeeming aToken for #{trade.trade_id}")

            # aToken appears in the wallet
            self.wallet.update_balance(collateral_token, collateral_token_change, f"Depositing/redeeming aToken for  #{trade.trade_id}")

        assert abs(
            executed_quantity) > 0, f"Expected executed_quantity for the trade to be above zero, got executed_quantity:{executed_quantity}, planned_quantity:{trade.planned_quantity}, trade is {trade}"

        logger.info("simulate_leverage(): wallet balances after updating for %s:\n%s", trade.get_short_label(), self.wallet.get_all_balances())

        # for leverage short, we use collateral token as the reserve currency
        # so return executed_collateral_quantity here to correctly calculate the price
        return executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption

    def simulate_trade(
        self,
        ts: datetime.datetime,
        state: State,
        idx: int,
        trade: TradeExecution
    ) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
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
                                          f"  Trade {idx + 1}. failed on strategy cycle {ts}\n"
                                          f"  Execution of trade failed:\n  {trade}\n"
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

    def execute_trades(
        self,
        ts: datetime.datetime,
        state: State,
        trades: List[TradeExecution],
        routing_model: BacktestRoutingModel,
        routing_state: BacktestRoutingState,
        check_balances=False,
        triggered=False,
    ):
        """Execute the trades on a simulated environment.

        Calculates price impact based on historical data
        and fills the expected historical trade output.

        :param check_balances:
            Raise an error if we run out of balance to perform buys in some point.
        """
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_model, (BacktestRoutingModel, GenericRouting))
        assert isinstance(routing_state, (BacktestRoutingState, GenericRoutingState))

        state.start_execution_all(ts, trades, max_slippage=0, triggered=triggered)

        routing_model.setup_trades(
            state,
            routing_state,
            trades,
            check_balances=check_balances
        )

        # Check that backtest does not try to execute stop loss / take profit
        # trades when data is not available
        for t in trades:

            assert not t.pair.is_cash(), f"Cannot do cash-cash trades. Got pair {t.pair}: {t}"

            position = state.portfolio.open_positions.get(t.position_id)
            if position and position.has_automatic_close():
                # Check that we have stop loss data available
                # for backtesting
                if not self.is_stop_loss_supported():
                    raise AutoClosingOrderUnsupported(
                        "Trade was marked with stop loss/take profit even though backtesting trading universe does not have price feed for stop loss checks available.\n"
                        "Remember to use the stop_loss_time_bucket parameter or equivalent when you create your trading universe to avoid this error."
                    )

        for idx, trade in enumerate(trades):

            # 3. Simulate tx broadcast
            try:
                executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption = self.simulate_trade(ts, state, idx, trade)
            except BacktestExecutionFailed as e:
                logger.info("Simulating %d. trade %s failed: %s", idx+1, trade.get_short_label(), e)
                raise BacktestExecutionFailed(f"Trade #{idx+1} out of {len(trades)} trades failed") from e

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

        # After all backtested trades have been executed and simulated wallet updated,
        # check that the simulated wallet and internal ledger still agree how rich we are
        all_assets = calculate_total_assets(state.portfolio)
        clean, asset_df = self.wallet.verify_balances(all_assets)
        if not clean:
            logger.error("Backtest sync issue")
            logger.error("All portfolio assets were")
            for a, v in all_assets.items():
                logger.error("Asset %s: %s", a, v)
            logger.error("Trades were")
            for t in trades:
                logger.error("Trade: %s", t)
            logger.error("Positions are")
            for p in state.portfolio.get_open_and_frozen_positions():
                logger.error("Position: %s", p)

            error_msg = f"Backtest simulated wallet and portfolio out of sync at {ts} after executing trades:\n{asset_df}"
            logger.error("Current chain status:\n%s", error_msg)

            raise RuntimeError(error_msg)

        # Set the check point interest balacnes for new positions
        set_interest_checkpoint(state, ts, None)

        # Print out trades and balances for diagnostics.
        # Extensive output. Very slow to create. Don't calculate/display if not absolutely necessary.
        if logger.getEffectiveLevel() >= logging.INFO:

            #
            # Output balances
            #

            trades = [
                {
                    "Trade": t.trade_id,
                    "Asset": t.pair.base.token_symbol,
                    "Type": t.trade_type.value,
                    "Executed price": t.executed_price,
                    "Executed value": t.get_value(),
                    "Executed qty": t.executed_quantity,
                }
                for t in trades
            ]

            if not trades:
                trades = [{"Trade": "None", "Asset": "No trades executed"}]

            table_msg = tabulate(
                trades,
                headers="keys",
                tablefmt="fancy_grid",
            )

            logger.info(
                "Trades at %s:\n%s",
                ts,
                table_msg,
            )

            #
            # Output assets
            #

            balances = [
                {"Asset": str(asset), "Balance": balance}
                for asset, balance in all_assets.items()
            ]

            if not balances:
                balances = [
                    {"Asset": "None", "Balance": "Wallet does not have any assets"}
                ]

            table_msg = tabulate(
                balances,
                headers="keys",
                tablefmt="fancy_grid",
            )
            logger.info(
                "Wallet balances at %s:\n%s",
                ts,
                table_msg,
            )

        logger.info("Finished backtest execution for %s", ts)

    def get_routing_state_details(self) -> dict:
        return {"wallet": self.wallet}

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        raise NotImplementedError()
