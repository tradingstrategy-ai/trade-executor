"""Live pricing and valuation of Lighter public-pool positions.

Prices are USDC per raw pool share, read from current pool equity and raw
supply through :py:func:`eth_defi.lighter.execution.fetch_pool_state`. Position
quantity is raw shares, so ``value = quantity × share value``. Account equity
is never added to pool value: whether Lighter's account equity includes pool
holdings is unverified upstream, so the two are kept separate.

Deposit admission reads live pool status and operator-share capacity rather
than dataset metadata. See :py:mod:`tradeexecutor.ethereum.lighter.lighter_pool`.
"""

import datetime
import logging
from decimal import Decimal

from eth_defi.compat import native_datetime_utc_now
from eth_defi.lighter.execution import LighterPoolState, fetch_execution_account, fetch_pool_state
from eth_defi.lighter.session import LighterSession

from tradeexecutor.ethereum.lighter.lighter_pool import create_lighter_pool_session
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.redemption import (
    DepositBlockReason,
    DepositCheckResult,
    DepositCheckStage,
    RedemptionBlockReason,
    RedemptionCheckResult,
    RedemptionCheckStage,
)
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.valuation import ValuationModel


logger = logging.getLogger(__name__)


def get_lighter_pool_mint_capacity(pool: LighterPoolState) -> int:
    """Return how many raw shares can still be minted under the operator-share rule.

    The operator must hold at least ``min_operator_share_rate`` of supply
    after a mint, the same check as ``prepare_pool_amount()``.

    :param pool:
        Current pool observation.
    :return:
        Maximum additional raw shares, zero when the pool is full.
    """
    if pool.min_operator_share_rate <= 0:
        return 2**60
    maximum_supply = int(Decimal(pool.operator_shares) / pool.min_operator_share_rate)
    return max(maximum_supply - pool.total_shares, 0)


def check_lighter_pool_deposit(pool: LighterPoolState) -> tuple[DepositBlockReason | None, str | None]:
    """Classify whether a pool currently accepts new investment.

    :param pool:
        Current pool observation.
    :return:
        ``(None, None)`` when open, otherwise a reason code and message.
    """
    if pool.is_llp:
        return DepositBlockReason.deposit_request_unavailable, "LLP requires staked LIT, which this integration does not acquire"
    if pool.status != 0:
        return DepositBlockReason.vault_deposits_closed, f"Lighter pool status is {pool.status}"
    if get_lighter_pool_mint_capacity(pool) == 0:
        return DepositBlockReason.vault_max_deposit_zero, "Lighter pool is full: operator holds the minimum share of supply"
    return None, None


class LighterPoolPricing(PricingModel):
    """Live Lighter pool pricing from current pool state.

    :param session:
        Ethereum Lighter API session; created on first use if omitted.
    """

    def __init__(self, session: LighterSession | None = None):
        self._session = session

    def get_session(self) -> LighterSession:
        """Return the shared API session."""
        if self._session is None:
            self._session = create_lighter_pool_session()
        return self._session

    def fetch_pool(self, pair: TradingPairIdentifier) -> LighterPoolState:
        """Read one pool's current state.

        :param pair:
            Lighter pool pair.
        :return:
            Strict current observation.
        """
        assert pair.is_lighter_pool(), f"Not a Lighter pool: {pair}"
        return fetch_pool_state(self.get_session(), pair.get_lighter_pool_index())

    def _make_pricing(self, pair: TradingPairIdentifier, token_in: Decimal | None = None) -> TradePricing:
        pool = self.fetch_pool(pair)
        price = float(pool.share_value)
        return TradePricing(
            price=price,
            mid_price=price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=native_datetime_utc_now(),
            block_number=None,
            token_in=token_in,
        )

    def get_sell_price(self, ts: datetime.datetime, pair: TradingPairIdentifier, quantity: Decimal | None) -> TradePricing:
        return self._make_pricing(pair, token_in=quantity)

    def get_buy_price(self, ts: datetime.datetime, pair: TradingPairIdentifier, reserve: Decimal | None) -> TradePricing:
        return self._make_pricing(pair, token_in=reserve)

    def get_mid_price(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> float:
        return float(self.fetch_pool(pair).share_value)

    def get_pair_fee(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> float:
        return 0.0

    def get_usd_tvl(self, timestamp: datetime.datetime | None, pair: TradingPairIdentifier) -> float:
        return float(self.fetch_pool(pair).equity)

    def get_max_deposit(self, ts: datetime.datetime | None, pair: TradingPairIdentifier) -> Decimal | None:
        pool = self.fetch_pool(pair)
        return get_lighter_pool_mint_capacity(pool) * pool.share_value

    def can_deposit(self, ts: datetime.datetime | None, pair: TradingPairIdentifier) -> bool:
        return self.check_deposit(ts, pair).can_deposit

    def check_deposit(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier,
        *,
        stage: DepositCheckStage = DepositCheckStage.unknown,
    ) -> DepositCheckResult:
        """Check live pool status, LLP restriction and operator-share capacity."""
        pool = self.fetch_pool(pair)
        reason, message = check_lighter_pool_deposit(pool)
        return DepositCheckResult(
            timestamp=ts,
            stage=stage,
            can_deposit=reason is None,
            reason_code=reason,
            message=message,
            pair_ticker=pair.get_ticker(),
            vault_address=pair.pool_address,
        )

    def get_max_redemption(self, ts: datetime.datetime | None, pair: TradingPairIdentifier) -> Decimal | None:
        return self.fetch_pool(pair).available_balance

    def check_redemption(
        self,
        ts: datetime.datetime | None,
        pair: TradingPairIdentifier | None,
        *,
        stage: RedemptionCheckStage = RedemptionCheckStage.unknown,
        position: TradingPosition | None = None,
    ) -> RedemptionCheckResult:
        """A pool exit needs redeemable pool collateral; closed pools still allow exits.

        :param position:
            Unused; pool holdings are confirmed by the router before burning.
        """
        pool = self.fetch_pool(pair)
        can_redeem = pool.available_balance > 0
        return RedemptionCheckResult(
            timestamp=ts,
            stage=stage,
            can_redeem=can_redeem,
            reason_code=None if can_redeem else RedemptionBlockReason.vault_max_withdrawable_zero,
            message=None if can_redeem else "Lighter pool has no redeemable collateral",
            pair_ticker=pair.get_ticker(),
            vault_address=pair.pool_address,
            max_redemption=float(pool.available_balance),
        )

    def get_pair_for_id(self, internal_id: int):
        raise NotImplementedError("Lighter pool pricing does not look up pairs by id")


#: Sell phases after which the burned shares are collateral or a claim, not pool exposure.
RELEASED_SELL_PHASES = {"burned", "withdraw", "withdraw_submitted", "claim"}


def get_pending_pool_sell_data(position: TradingPosition) -> dict | None:
    """Return the lifecycle evidence of the position's pending pool exit, if any."""
    for trade in position.trades.values():
        if trade.is_sell() and trade.get_status() == TradeStatus.vault_settlement_pending:
            return trade.other_data.get("lighter_pool")
    return None


class LighterPoolValuator(ValuationModel):
    """Revalue a pool position at its net USDC value.

    Held shares are valued at the current pool share value minus the
    operator's estimated fee on positive profit, read from the account's
    holding when the account is known. During a pending exit whose burn has
    confirmed, the burned shares are valued at the collateral they released or
    the claimable withdrawal amount, because that value no longer follows the
    pool. The position quantity stays the booked share count until the trade
    completes, so the value is expressed as a per-share price.

    :param pricing_model:
        Pool pricing sharing the API session.
    :param account_index:
        Lighter account holding the shares; without it the fee is not netted.
    """

    def __init__(self, pricing_model: LighterPoolPricing, account_index: int | None = None):
        self.pricing_model = pricing_model
        self.account_index = account_index

    def get_net_value(self, position: TradingPosition, pool: LighterPoolState) -> Decimal:
        """Value the position's shares and any released exit collateral once.

        :param position:
            Pool position.
        :param pool:
            Current pool observation.
        :return:
            Net USDC value.
        """
        quantity = Decimal(position.get_quantity())
        held = quantity
        released = Decimal(0)
        pending = get_pending_pool_sell_data(position)
        if pending is not None and pending["phase"] in RELEASED_SELL_PHASES:
            burned = Decimal(pending["burned_shares"])
            held -= burned
            if "claimable_usdc" in pending:
                released = Decimal(pending["claimable_usdc"])
            elif "available_after_burn" in pending:
                released = Decimal(pending["available_after_burn"]) - Decimal(pending["collateral_before_burn"])
            else:
                # Burn confirmed, released collateral not yet observed: value
                # the burned shares at the pool price until the wait completes.
                released = burned * pool.share_value
        gross = held * pool.share_value
        fee = Decimal(0)
        if self.account_index is not None and held > 0:
            account, _ = fetch_execution_account(self.pricing_model.get_session(), self.account_index)
            rows = [row for row in account["shares"] if int(row["public_pool_index"]) == pool.pool_index and int(row["shares_amount"]) > 0]
            if rows:
                principal = Decimal(str(rows[0]["principal_amount"])) * held / Decimal(int(rows[0]["shares_amount"]))
                fee = max(gross - principal, Decimal(0)) * pool.operator_fee
        return released + gross - fee

    def __call__(self, ts: datetime.datetime, position: TradingPosition) -> ValuationUpdate:
        assert position.pair.is_lighter_pool(), f"Not a Lighter pool position: {position}"
        # Record wall-clock time: Lagoon NAV freshness compares against now.
        valued_at = native_datetime_utc_now()
        pool = self.pricing_model.fetch_pool(position.pair)
        old_price = position.last_token_price
        old_value = position.get_value()
        quantity = Decimal(position.get_quantity())
        new_price = float(self.get_net_value(position, pool) / quantity) if quantity else float(pool.share_value)
        new_value = position.revalue_base_asset(valued_at, new_price)
        position.last_pricing_at = valued_at
        evt = ValuationUpdate(
            created_at=valued_at,
            position_id=position.position_id,
            valued_at=valued_at,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=position.get_quantity(),
        )
        position.valuation_updates.append(evt)
        logger.info("Lighter pool position %s valued at %f USDC (%s raw shares)", position, new_value, position.get_quantity())
        return evt
