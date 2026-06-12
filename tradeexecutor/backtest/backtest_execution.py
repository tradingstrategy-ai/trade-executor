"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from tabulate import tabulate

from eth_defi.erc_4626.core import ERC4626Feature

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


#: Default simulated settlement delay for async (ERC-7540 / Ostium) vault
#: deposit and redeem requests in backtesting.
#:
#: Non-zero on purpose: any vault whose metadata carries async features
#: (``erc_7540_like``, ``lagoon_like``, ``ostium_like``) is automatically
#: backtested with the two-stage flow, and a realistic default delay makes the
#: behaviour change from older instant-settlement backtests explicit rather
#: than a silent one-cycle shift.
#:
#: Two days rather than one: with the common one-day decision cycle a one-day
#: delay settles every request exactly on the next cycle, so the pending
#: window never spans a full cycle and the queue dynamics stay invisible.
DEFAULT_VAULT_SETTLEMENT_DELAY = datetime.timedelta(days=2)

#: Hour of day (UTC, naive) when Ostium-style vaults settle in backtesting.
#:
#: Live Ostium settles in roughly daily epochs (``lastSettlementTs +
#: maxSettlementInterval`` on-chain); the backtest approximates this as
#: "the request becomes claimable the next day at this hour". Override the
#: schedule per vault with ``vault_settlement_delay_overrides``.
OSTIUM_BACKTEST_SETTLEMENT_HOUR = 18


class BacktestExecution(ExecutionModel):
    """Simulate trades against historical data."""

    def __init__(self,
                 wallet: SimulatedWallet,
                 max_slippage: Percent = 0.01,
                 lp_fees: Percent = 0.0030,
                 stop_loss_data_available=False,
                 vault_settlement_delay: datetime.timedelta = DEFAULT_VAULT_SETTLEMENT_DELAY,
                 vault_settlement_delay_overrides: dict[str, datetime.timedelta] | None = None,
                 ):
        self.wallet = wallet
        self.max_slippage = max_slippage
        self.lp_fees = lp_fees
        self.stop_loss_data_available = stop_loss_data_available

        #: Global default settlement delay for async (ERC-7540 / Ostium) vault
        #: deposit and redeem requests. Defaults to
        #: :py:data:`DEFAULT_VAULT_SETTLEMENT_DELAY` (two days). An explicit
        #: ``timedelta(0)`` means the request settles on the next cycle that
        #: runs the resolver (a one-cycle minimum, because the request is
        #: created after the resolver step within a tick). Ostium-style vaults
        #: ignore this and settle on their daily epoch hour unless a per-vault
        #: override is given — see :py:meth:`_get_settlement_due`.
        self.vault_settlement_delay = vault_settlement_delay

        #: Per-vault settlement delay overrides, keyed by lowercased vault address.
        #: Presence of a vault here also flags it as async even without vault features.
        self.vault_settlement_delay_overrides = {
            k.lower(): v for k, v in (vault_settlement_delay_overrides or {}).items()
        }

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

        # For satellite chain trades funded via a CCTP bridge, the on-chain
        # token spent is the pair's quote token (e.g. USDC on Base), not the
        # portfolio reserve currency (e.g. USDC on Arbitrum).
        bridge_position = state.portfolio.get_bridge_position_for_chain(trade.pair.chain_id)
        if bridge_position is not None:
            reserve = trade.pair.quote
        else:
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

    def simulate_bridge(self, state: State, trade: TradeExecution) -> Tuple[Decimal, Decimal]:
        """CCTP bridge simulation with a simulated wallet.

        Bridge trades transfer USDC 1:1 between chains. In the simulated
        wallet we debit the source chain USDC (reserve/quote) and credit the
        destination chain USDC (base).

        :param state:
            Backtester state

        :param trade:
            Trade to be executed (must be a CCTP bridge trade)

        :return:
            (executed_quantity, executed_reserve) tuple
        """
        assert trade.pair.is_cctp_bridge(), f"simulate_bridge(): received a non-bridge trade {trade}"

        base = trade.pair.base        # destination chain USDC
        reserve = trade.reserve_currency  # source chain USDC

        if trade.is_buy():
            # Bridge out: burn USDC on source, mint on destination
            executed_reserve = trade.planned_reserve
            executed_quantity = trade.planned_quantity
            self.wallet.update_balance(base, executed_quantity, f"bridge out trade #{trade.trade_id}")
            self.wallet.update_balance(reserve, -executed_reserve, f"bridge out trade #{trade.trade_id}")
        else:
            # Bridge back: burn USDC on destination, mint on source
            executed_quantity = trade.planned_quantity
            executed_reserve = abs(Decimal(trade.planned_quantity) * Decimal(trade.planned_price))
            self.wallet.update_balance(base, executed_quantity, f"bridge back trade #{trade.trade_id}")
            self.wallet.update_balance(reserve, executed_reserve, f"bridge back trade #{trade.trade_id}")

        return executed_quantity, executed_reserve

    #
    # Async (two-stage ERC-7540 / Ostium) vault deposit/redeem simulation
    #

    def _is_async_vault(self, pair) -> bool:
        """Does this vault pair use a two-stage (async) deposit/redeem flow in backtest?

        True if the pair has an explicit settlement-delay override, or its vault
        features mark it as ERC-7540 / Lagoon / Ostium style.
        """
        if not pair.is_vault():
            return False
        if pair.pool_address and pair.pool_address.lower() in self.vault_settlement_delay_overrides:
            return True
        return pair.is_async_vault()

    def _get_settlement_due(self, pair, ts: datetime.datetime) -> datetime.datetime:
        """When does an async vault request made at ``ts`` become claimable?

        Precedence:

        1. Per-vault override (``vault_settlement_delay_overrides``) — a fixed
           delay from the request time.
        2. Ostium-style vaults (``ostium_like`` feature) — the next day at
           :py:data:`OSTIUM_BACKTEST_SETTLEMENT_HOUR`, approximating Ostium's
           daily epoch settlement schedule.
        3. The global default delay (``vault_settlement_delay``).
        """
        if pair.pool_address:
            override = self.vault_settlement_delay_overrides.get(pair.pool_address.lower())
            if override is not None:
                return ts + override

        features = pair.get_vault_features() or set()
        if ERC4626Feature.ostium_like in features:
            # Ostium settles in daily epochs: the request becomes claimable
            # the following day at the epoch settlement hour.
            next_day = ts + datetime.timedelta(days=1)
            return next_day.replace(hour=OSTIUM_BACKTEST_SETTLEMENT_HOUR, minute=0, second=0, microsecond=0)

        return ts + self.vault_settlement_delay

    def simulate_async_vault_request(self, ts: datetime.datetime, state: State, trade: TradeExecution) -> None:
        """Stage 1 of a two-stage async vault deposit/redeem in backtest.

        Records the deposit/redeem request as pending settlement **without** touching
        the simulated wallet. For a deposit the state reserve ledger has already been
        debited by ``start_execution()``; for a redeem the shares stay in the position
        and wallet. The wallet share/reserve balances move only at simulated claim time
        in :py:meth:`resolve_pending_vault_settlements`, once the configured delay has
        elapsed.

        :param ts:
            Strategy cycle timestamp (request time).
        """
        assert trade.is_vault(), f"simulate_async_vault_request(): not a vault trade {trade}"
        settles_at = self._get_settlement_due(trade.pair, ts)
        trade.other_data["vault_settlement_estimated_at"] = settles_at.isoformat()
        # mark_vault_settlement_pending() sets vault_settlement_pending_at, vault_async_flow,
        # vault_chain_id and vault_direction. No protocol ticket data exists in backtest.
        state.mark_vault_settlement_pending(ts, trade, ticket_data={})
        logger.info(
            "Backtest async vault request: trade #%d (%s), settles at %s",
            trade.trade_id,
            "deposit" if trade.is_buy() else "redeem",
            settles_at,
        )

    def resolve_pending_vault_settlements(
        self,
        state: State,
        ts: datetime.datetime,
        pricing_model=None,
    ) -> List[TradeExecution]:
        """Stage 2 of a two-stage async vault deposit/redeem in backtest.

        Scans open and pending positions for trades sitting in
        ``vault_settlement_pending`` state and settles every one whose estimated
        settlement time has passed. Settlement updates the simulated wallet and
        marks the trade successful. Deposits realise the shares at the current
        simulated price; redeems realise the reserve at the current simulated
        price, so the settlement delay carries any vault NAV drift into P&L.

        :param ts:
            Strategy cycle timestamp (settlement-due cutoff).

        :param pricing_model:
            Pricing model used to value the settlement at the current simulated
            price. Falls back to the trade's planned price when not provided.

        :return:
            List of trades resolved this call.
        """
        from itertools import chain as ichain

        # Materialise the work-list before settling: mark_trade_success() on a
        # deposit can move a position from pending_positions to open_positions, so
        # iterating those dicts live while settling would skip work or error.
        pending_trades: List[TradeExecution] = []
        for position in ichain(state.portfolio.open_positions.values(), state.portfolio.pending_positions.values()):
            for trade in position.trades.values():
                if trade.get_status() == TradeStatus.vault_settlement_pending:
                    pending_trades.append(trade)

        resolved: List[TradeExecution] = []
        for trade in pending_trades:
            estimated = trade.other_data.get("vault_settlement_estimated_at")
            settles_at = datetime.datetime.fromisoformat(estimated) if estimated else trade.vault_settlement_pending_at
            if settles_at > ts:
                # Settlement delay has not elapsed yet — leave it pending.
                continue
            self._settle_async_vault_trade(state, trade, ts, pricing_model)
            resolved.append(trade)

        if resolved:
            logger.info("Backtest resolved %d pending vault settlement(s) at %s", len(resolved), ts)

        return resolved

    def _settle_async_vault_trade(
        self,
        state: State,
        trade: TradeExecution,
        ts: datetime.datetime,
        pricing_model,
    ) -> None:
        """Settle a single due async vault deposit/redeem against the simulated wallet."""
        pair = trade.pair
        reserve = trade.reserve_currency
        base = pair.base

        if trade.is_buy():
            # Deposit: realise shares at the current (settlement-time) price.
            if pricing_model is not None:
                settlement_price = pricing_model.get_buy_price(ts, pair, trade.planned_reserve).price
            else:
                settlement_price = float(trade.planned_price)
            executed_reserve = trade.planned_reserve
            # Decimal rounding drift between the state cash ledger and the
            # simulated wallet accumulates over many settlements; settle with
            # what the wallet holds when the difference is dust.
            reserve_balance = self.wallet.get_balance(reserve.address)
            if reserve_balance < executed_reserve <= reserve_balance + Decimal("0.000001"):
                executed_reserve = reserve_balance
            executed_quantity = executed_reserve / Decimal(str(settlement_price))
            # Wallet: receive shares, pay the committed reserve. The state cash
            # ledger was already debited at request time (start_execution); this
            # debits the simulated wallet so the two stay in sync post-settlement.
            self.wallet.update_balance(base, executed_quantity, f"vault deposit settle #{trade.trade_id}")
            self.wallet.update_balance(reserve, -executed_reserve, f"vault deposit settle #{trade.trade_id}")
        else:
            # Redeem: realise reserve at the current (settlement-time) price.
            if pricing_model is not None:
                settlement_price = pricing_model.get_sell_price(ts, pair, abs(trade.planned_quantity)).price
            else:
                settlement_price = float(trade.planned_price)
            base_balance = self.wallet.get_balance(base.address)
            executed_quantity, _ = fix_sell_token_amount(base_balance, trade.planned_quantity)
            executed_reserve = abs(executed_quantity) * Decimal(str(settlement_price))
            # Wallet: give up shares (executed_quantity is negative), receive reserve.
            self.wallet.update_balance(base, executed_quantity, f"vault redeem settle #{trade.trade_id}")
            self.wallet.update_balance(reserve, executed_reserve, f"vault redeem settle #{trade.trade_id}")

        executed_price = float(abs(executed_reserve / executed_quantity)) if executed_quantity else float(settlement_price)

        # Clear the pending marker before marking success (matches the live resolver).
        trade.vault_settlement_pending_at = None

        state.mark_trade_success(
            ts,
            trade,
            executed_price,
            executed_quantity,
            executed_reserve,
            lp_fees=0,
            native_token_price=1,
        )

        # mark_trade_success() does not refresh position.last_token_price, and the
        # runner revalues before this resolver runs, so revalue explicitly to keep
        # the just-opened (deposit) or surviving (partial redeem) position correct.
        position = state.portfolio.get_position_by_id(trade.position_id)
        if position is not None and position.is_open():
            position.revalue_base_asset(ts, float(settlement_price))

        logger.info(
            "Backtest async vault settled: trade #%d (%s), price %s, quantity %s, reserve %s",
            trade.trade_id,
            "deposit" if trade.is_buy() else "redeem",
            settlement_price,
            executed_quantity,
            executed_reserve,
        )

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
            if trade.is_vault() and self._is_async_vault(trade.pair):
                # Two-stage async vault: record the request as pending settlement
                # and return zeros. The trade is not marked successful here — the
                # resolver settles it on a later cycle once the delay elapses.
                self.simulate_async_vault_request(ts, state, trade)
                return Decimal(0), Decimal(0), None, None
            if trade.is_spot() or trade.is_credit_supply():
                executed_quantity, executed_reserve, sell_amount_epsilon_fix = self.simulate_spot(state, trade)
            elif trade.is_leverage():
                executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption = self.simulate_leverage(state, trade)
            elif trade.pair.is_cctp_bridge():
                executed_quantity, executed_reserve = self.simulate_bridge(state, trade)
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

    def _execute_trades_sequentially(
        self,
        ts: datetime.datetime,
        state: State,
        trades: list[TradeExecution],
        routing_model,
        routing_state,
        check_balances: bool,
        triggered: bool,
    ):
        """Execute trades one at a time for CCTP-dependent batches.

        Each trade is started, simulated, and settled before the next
        trade begins. This ensures bridge positions exist before
        satellite vault deposits try to allocate from them.

        The normal batch path calls ``start_execution_all()`` which
        allocates capital for every trade upfront. That fails when a
        satellite trade needs ``get_bridge_position_for_chain()`` to
        find a bridge position that has not been created yet.

        :param ts:
            Strategy cycle timestamp

        :param state:
            Current backtesting state

        :param trades:
            Trades to execute, already sorted by execution sort position

        :param routing_model:
            The routing model (calculates prices)

        :param routing_state:
            Routing state for the current execution

        :param check_balances:
            Whether to check wallet balances during setup

        :param triggered:
            True if this execution is from stop loss trigger checks
        """

        # Calculate prices for all trades upfront — this does not allocate capital
        routing_model.setup_trades(
            state,
            routing_state,
            trades,
            check_balances=check_balances,
        )

        # Sort trades for sequential execution.  The standard sort phases
        # (PR 1) place bridge-outs at +30M, between regular buys and vault
        # deposits.  But in a sequential backtest, bridge-out buys must
        # execute BEFORE any satellite trade that will allocate from the
        # resulting bridge position.  We use the standard sort as a base
        # but promote bridge-out buys to run right after all sells/redeems
        # complete (sort key 0) and demote bridge-backs to run last (after
        # vault deposits return capital).
        def _sequential_sort_key(t):
            base = t.get_execution_sort_position()
            if t.pair.is_cctp_bridge():
                if t.is_buy():
                    # Bridge-out buys: after all sells (negative range)
                    # but before any regular/vault buys (positive range)
                    return 0
                else:
                    # Bridge-back sells: after vault redeems but before
                    # bridge-outs.  Use the standard -30M position.
                    return base
            return base

        trades = sorted(trades, key=_sequential_sort_key)

        for idx, trade in enumerate(trades):
            # Start execution (allocate capital) for this trade only
            state.start_execution(ts, trade, triggered=triggered)

            # Simulate
            try:
                executed_quantity, executed_reserve, executed_collateral_allocation, executed_collateral_consumption = self.simulate_trade(ts, state, idx, trade)
            except BacktestExecutionFailed as e:
                logger.info("Simulating %d. trade %s failed: %s", idx + 1, trade.get_short_label(), e)
                raise BacktestExecutionFailed(f"Trade #{idx + 1} out of {len(trades)} trades failed") from e

            # Async vault deposit/redeem requests are now pending settlement — do not
            # mark them successful. resolve_pending_vault_settlements() settles them on
            # a later cycle once the configured delay has elapsed.
            if trade.get_status() == TradeStatus.vault_settlement_pending:
                continue

            # Settle immediately so the next trade can see the results
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

        # Verify wallet balances after all trades
        all_assets = calculate_total_assets(state.portfolio)
        clean, asset_df = self.wallet.verify_balances(all_assets)
        if not clean:
            logger.error("Backtest sync issue (sequential path)")
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

        # Set the check point interest balances for new positions
        set_interest_checkpoint(state, ts, None)

        logger.info("Finished sequential backtest execution for %s", ts)

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

        # When the batch contains CCTP bridge trades, we must execute each
        # trade sequentially (start -> simulate -> settle) so that bridge
        # positions exist before satellite trades try to allocate from them.
        has_bridge_trades = any(t.pair.is_cctp_bridge() for t in trades)

        if has_bridge_trades:
            self._execute_trades_sequentially(ts, state, trades, routing_model, routing_state, check_balances, triggered)
            return

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

            # Async vault deposit/redeem requests are now pending settlement — do not
            # mark them successful. resolve_pending_vault_settlements() settles them on
            # a later cycle once the configured delay has elapsed.
            if trade.get_status() == TradeStatus.vault_settlement_pending:
                continue

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
