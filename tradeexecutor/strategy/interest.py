"""Functions to refresh accrued interest on credit positions."""
import datetime
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Tuple, Set, Literal, Dict
import logging

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import TradingPairKind, AssetIdentifier, AssetWithTrackedValue
from tradeexecutor.state.interest import PortfolioInterestTracker, Interest
from tradeexecutor.state.loan import LoanSide
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarPrice, Percent

logger = logging.getLogger(__name__)


@dataclass
class InterestDistributionEntry:
    """Map interest distribution across different trading positions.

    A helper class to help us to distribute accrued interest across different
    related trading positions.

    The lifetime of an instance is one sync_interests() run.
    """

    #: Which side we are on
    side: LoanSide

    #: The trading position containing this asset
    position: TradingPosition

    #: Can be either loan.collateral or loan.borrower
    tracker: AssetWithTrackedValue

    #: All weight are normalised to 0...1 based on loan token amount.
    weight: Percent = None

    @property
    def asset(self) -> AssetIdentifier:
        """Amount of tracked asset in tokens"""
        return self.tracker.asset

    @property
    def quantity(self) -> Decimal:
        """Amount of tracked asset in tokens"""
        return self.tracker.quantity

    def distribute_interest(self, asset_accrued_across_positions: Decimal):
        pass


@dataclass
class InterestDistributionOperation:
    """One interest update batch we do."""

    #: All interest bearing assets we have across positions
    assets: Set[AssetIdentifier]

    #: All entries we need to udpate
    entries: Dict[AssetIdentifier, InterestDistributionEntry]

    #: Portfolio totals of interest bearing assets
    totals: Dict[AssetIdentifier, Decimal]


def update_interest(
    state: State,
    position: TradingPosition,
    asset: AssetIdentifier,
    new_token_amount: Decimal,
    event_at: datetime.datetime,
    asset_price: USDollarPrice,
    block_number: int | None = None,
    tx_hash: int | None = None,
    log_index: int | None = None,
) -> BalanceUpdate:
    """Poke credit supply position to increase its interest amount.

    :param position:
        Trading position to update

    :param asset:
        The asset of which we update the events for.

        aToken for collateral, vToken for debt.

    :param new_token_amount:
        The new on-chain value of aToken/vToken tracking the loan.

    :param asset_price:
        The latest known price for the underlying asset.

        Needed to revalue dollar nominated loans.

    :param event_at:
        Block mined timestamp

    """

    assert asset is not None
    # assert position.pair.kind == TradingPairKind.credit_supply
    assert position.is_open() or position.is_frozen(), f"Cannot update interest for position {position.position_id}\n" \
                                                       f"Position details: {position}\n" \
                                                       f"Position closed at: {position.closed_at}\n" \
                                                       f"Interest event at: {event_at}"
    assert type(asset_price) == float, f"Got {asset_price.__class__}"
    assert isinstance(new_token_amount, Decimal), f"Got {new_token_amount.__class__}"

    loan = position.loan
    assert loan

    if asset == loan.collateral.asset:
        interest = loan.collateral_interest
    elif loan.borrowed and asset == position.loan.borrowed.asset:
        interest = loan.borrowed_interest
    else:
        raise AssertionError(f"Loan {loan} does not have asset {asset}\n"
                             f"We have\n"
                             f"- {loan.collateral.asset}\n"
                             f"- {loan.borrowed.asset if loan.borrowed else '<no borrow>'}")

    assert interest, f"Position does not have interest tracked set up on {asset.token_symbol}:\n" \
                     f"{position} \n" \
                     f"for asset {asset}"

    portfolio = state.portfolio

    event_id = portfolio.allocate_balance_update_id()

    # assert asset.underlying.is_stablecoin(), f"Credit supply is currently supported for stablecoin assets with 1:1 USD price assumption. Got: {asset}"

    previous_update_at = interest.last_event_at

    old_balance = interest.last_token_amount
    gained_interest = new_token_amount - old_balance
    usd_value = float(new_token_amount) * asset_price

    assert 0 < abs(gained_interest) < 999, f"Unlikely gained_interest: {gained_interest}, old quantity: {old_balance}, new quantity: {new_token_amount}"

    evt = BalanceUpdate(
        balance_update_id=event_id,
        position_type=BalanceUpdatePositionType.open_position,
        cause=BalanceUpdateCause.interest,
        asset=asset,
        block_mined_at=event_at,
        strategy_cycle_included_at=None,
        chain_id=asset.chain_id,
        old_balance=old_balance,
        usd_value=usd_value,
        quantity=gained_interest,
        owner_address=None,
        tx_hash=tx_hash,
        log_index=log_index,
        position_id=position.position_id,
        block_number=block_number,
        previous_update_at=previous_update_at,
    )

    position.add_balance_update_event(evt)

    # Update interest stats
    interest.last_accrued_interest = position.calculate_accrued_interest_quantity(asset)
    interest.last_updated_at = event_at
    interest.last_event_at = event_at
    interest.last_updated_block_number = block_number
    interest.last_token_amount = new_token_amount

    return evt


def update_leveraged_position_interest(
    state: State,
    position: TradingPosition,
    new_vtoken_amount: Decimal,
    new_token_amount: Decimal,
    event_at: datetime.datetime,
    vtoken_price: USDollarPrice,
    atoken_price: USDollarPrice = 1.0,
    block_number: int | None = None,
    tx_hash: int | None = None,
    log_index: int | None = None,
) -> Tuple[BalanceUpdate, BalanceUpdate]:
    """Updates accrued interest on lending protocol leveraged positions.

    Updates loan interest state for both collateral and debt.

    :param atoken_price:
        What is the current price of aToken.

        Needed to calculate dollar nominated amounts.

    :param vtoken_price:
        What is the current price of vToken

        Needed to calculate dollar nominated amounts.

    :return:
        Tuple (vToken update event, aToken update event)
    """

    assert position.is_leverage()

    pair = position.pair

    # vToken
    vevt = update_interest(
        state,
        position,
        pair.base,
        new_vtoken_amount,
        event_at,
        vtoken_price,
        block_number,
        tx_hash,
        log_index,
    )

    logger.info("Updated leveraged interest %s for %s", pair.base, vevt)

    # aToken
    aevt = update_interest(
        state,
        position,
        pair.quote,
        new_token_amount,
        event_at,
        atoken_price,
        block_number,
        tx_hash,
        log_index,
    )

    logger.info("Updated leveraged interest %s for %s", pair.quote, aevt)

    return (vevt, aevt)


def estimate_interest(
    start_at: datetime.datetime,
    end_at: datetime.datetime,
    start_quantity: Decimal,
    interest_rate: float,
    year = datetime.timedelta(days=360),
) -> Decimal:
    """Calculate new token amount, assuming fixed interest.

    :param interest_rate:

        Yearly interest relative to.

        1 = 0%.

        E.g. 1.02 for 2% yearly gained interest.

        Always positive.

    :param start_quantity:
        Tokens at the start of the period

    :param year:
        Year length.

        Default to the financial year or 360 days, 30 * 12 months,
        not calendar year.

    :return:
        Amount of token quantity with principal + interest after the period.
    """

    # 150x
    assert interest_rate >= 1

    assert end_at >= start_at
    duration = end_at - start_at
    multiplier = (end_at - start_at) / year
    return start_quantity * Decimal(interest_rate ** multiplier)


def prepare_interest_distribution(portfolio: Portfolio) -> InterestDistributionOperation:
    """Get all tokens in open positions that accrue interest.

    - We use this data to sync the accrued interest since
      the last cycle

    - See :py:func:`tradeexecutor.strategy.sync_model.SyncModel.sync_interests`

    :return:
        Interest bearing assets used in all open positions
    """

    assets = set()
    totals: Counter[AssetIdentifier, Decimal] = Counter()
    entries: Dict[AssetIdentifier, InterestDistributionEntry] = {}

    for p in portfolio.get_open_and_frozen_positions():
        for asset in (p.pair.base, p.pair.quote):

            if not asset.is_interest_accruing():
                # One side in spot-credit pair
                continue

            side, tracker = p.loan.get_tracked_asset(asset)

            assert side is not None, f"Got confused with asset {asset} on position {p}"

            entry = InterestDistributionEntry(
                side=side,
                position=p,
                tracker=tracker,
            )

            assert entry.quantity > 0, f"Zero-amount entry in the interest distribution: {p}: {tracker}"

            entries[asset] = entry
            assets.add(asset)

            totals[asset] += entry.quantity

    # Calculate distribution weights
    for entry in entries.values():
        entry.weight = entry.quantity / totals[entry.asset]

    return InterestDistributionOperation(
        assets,
        entries,
        totals,
    )


def initialise_tracking(
    tracker: PortfolioInterestTracker,
    interest_distribution: InterestDistributionOperation,
):
    """Start tracking any interest-based assets we do not track yet.

    If the asset was just added to the portfolio start with zero interest accrued.
    """

    for asset in interest_distribution.assets:
        if asset not in tracker.assets:
            tracker.assets[asset] = Interest(
                opening_amount=interest_distribution.totals[asset],
                last_token_amount=interest_distribution.totals[asset],
                last_updated_at=datetime.datetime.utcnow(),
                last_event_at=datetime.datetime.utcnow(),
                last_accrued_interest=Decimal(0),
                last_updated_block_number=None,
            )