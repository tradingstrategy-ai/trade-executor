"""Functions to refresh accrued interest on credit positions."""
import logging
import datetime
from collections import Counter
from decimal import Decimal
from typing import Tuple, Dict, List, Iterable

from eth_defi.aave_v3.rates import SECONDS_PER_YEAR_INT
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from tradingstrategy.utils.time import ZERO_TIMEDELTA

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.interest_distribution import InterestDistributionEntry, InterestDistributionOperation, AssetInterestData
from tradeexecutor.state.loan import LoanSide
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarPrice, Percent, BlockNumber
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.utils.accuracy import QUANTITY_EPSILON, INTEREST_EPSILON
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.ethereum.onchain_balance import fetch_address_balances

logger = logging.getLogger(__name__)


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
    max_interest_gain: Percent = 0.05,
) -> BalanceUpdate:
    """Poke leverage position to increase its interest amount.

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

    :param max_interest_gain:
        Safety threshold to check that any interest gains are below this value.

        Terminate execution if bad math detected.
    """
    assert asset is not None
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
        raise AssertionError(
            f"Loan {loan} does not have asset {asset}\n"
            f"We have\n"
            f"- {loan.collateral.asset}\n"
            f"- {loan.borrowed.asset if loan.borrowed else '<no borrow>'}"
        )

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

    gained_interest_percent = gained_interest / old_balance

    assert gained_interest_percent > 0, f"Negative interest for {asset}: {gained_interest} (diff {gained_interest_percent * 100:.2f}%), old quantity: {old_balance}, new quantity: {new_token_amount}"
    assert gained_interest_percent < max_interest_gain, f"Unlikely gained_interest for {asset}: {gained_interest} (diff {gained_interest_percent * 100:.2f}%, threshold {max_interest_gain * 100}%), old quantity: {old_balance}, new quantity: {new_token_amount}"

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
    max_interest_gain: Percent = 0.05,
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
        max_interest_gain=max_interest_gain,
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
        max_interest_gain=max_interest_gain,
    )

    logger.info("Updated leveraged interest %s for %s", pair.quote, aevt)

    return (vevt, aevt)


def estimate_interest(
    start_at: datetime.datetime,
    end_at: datetime.datetime,
    start_quantity: Decimal,
    interest_rate: float,
    year=datetime.timedelta(seconds=SECONDS_PER_YEAR_INT),
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
        Year length in Aave.

    :return:
        Amount of token quantity with principal + interest after the period.
    """

    # 150x
    assert interest_rate >= 1

    assert end_at >= start_at
    duration = end_at - start_at
    multiplier = (end_at - start_at) / year
    return start_quantity * Decimal(interest_rate ** multiplier)


def prepare_interest_distribution(
    start: datetime.datetime,
    end: datetime.datetime,
    portfolio: Portfolio,
    pricing_model: PricingModel
) -> InterestDistributionOperation:
    """Get all tokens in open positions that accrue interest.

    - We use this data to sync the accrued interest since
      the last cycle

    - See :py:func:`tradeexecutor.strategy.sync_model.SyncModel.sync_interests`

    :return:
        Interest bearing assets used in all open positions
    """

    assets = set()
    totals: Counter[AssetIdentifier, Decimal] = Counter()
    entries: List[InterestDistributionEntry] = []
    asset_interest_data: Dict[str, AssetInterestData] = {}
    position_count = 0

    timestamp = end

    for p in portfolio.get_open_and_frozen_positions():
        for asset in (p.pair.base, p.pair.quote):

            if not asset.is_interest_accruing():
                # One side in spot-credit pair
                continue

            side, tracker = p.loan.get_tracked_asset(asset)

            assert side is not None, f"Got confused with asset {asset} on position {p}"

            if side == LoanSide.collateral:
                # Currently supports stablecoin collateral only
                if not p.is_credit_supply():
                    assert not p.is_long(), f"Cannot handle position: {p}"
                underlying = asset.get_pricing_asset()
                assert underlying.is_stablecoin(), f"Asset is collateral but not stablecoin based: {asset}"
                price = 1.0
            else:
                price_structure = pricing_model.get_sell_price(
                    timestamp,
                    p.pair.get_pricing_pair(),
                    tracker.quantity,
                )
                price = price_structure.price

            entry = InterestDistributionEntry(
                side=side,
                position=p,
                tracker=tracker,
                price=price,
            )

            assert entry.quantity > 0, f"Zero-amount entry in the interest distribution: {p}: {tracker}"

            entries.append(entry)
            assets.add(asset)

            # Update totals
            asset_id = asset.get_identifier()
            asset_interest = asset_interest_data.get(asset_id, AssetInterestData())
            asset_interest.total += entry.quantity
            asset_interest_data[asset_id] = asset_interest

            position_count += 1

    # Calculate distribution weights
    for entry in entries:
        entry.weight = entry.quantity / asset_interest_data[entry.asset.get_identifier()].total

    logger.info("Preparing interest distribution with %d assets, %d positions, %d ledger entries", len(assets), position_count, len(entries))

    return InterestDistributionOperation(
        start,
        end,
        assets,
        asset_interest_data=asset_interest_data,
        entries=entries,
        effective_rate={},
    )


def distribute_to_entry(
    entry: InterestDistributionEntry,
    state: State,
    timestamp: datetime.datetime,
    block_number: BlockNumber,
    total_accrued: Decimal,
    max_interest_gain: Percent,
) -> BalanceUpdate:
    """Update interest one position, one side of loan."""

    position_accrued = total_accrued * entry.weight  # Calculate per-position portion of new tokens
    new_token_amount = entry.tracker.quantity + position_accrued
    assert entry.price is not None, f"Asset lacks updated price: {entry.asset}"
    assert new_token_amount > 0
    assert new_token_amount >= entry.tracker.quantity

    evt = update_interest(
        state,
        entry.position,
        entry.asset,
        new_token_amount=new_token_amount,
        event_at=timestamp,
        asset_price=entry.price,
        block_number=block_number,
        max_interest_gain=max_interest_gain,
    )
    return evt


def distribute_interest_for_assets(
    operation: InterestDistributionOperation,
    state: State,
    asset: AssetIdentifier,
    timestamp: datetime.datetime,
    block_number: BlockNumber | None,
    new_amount: Decimal,
    max_interest_gain: Percent,
) -> Iterable[BalanceUpdate]:
    """Distribute the accrued interest of an asset across all positions holding this asset.

     :return:
        An event
     """

    asset_total = operation.asset_interest_data[asset.get_identifier()].total

    # Either there has not be really any change over time (too fast refresh rate)
    # or this is unit test against mainnet fork where we cannot speed up the time.
    # In both cases we cannot generate any BalanceUpdate events,
    # because balance update can not be zero.
    # We also may encounter negative updates < epsilon due to the rounding
    # errors.
    interest_accrued = new_amount - asset_total
    logger.info("Interest accrued for %s: %s, %s, %s", asset, interest_accrued, new_amount, asset_total)
    if abs(interest_accrued) >= INTEREST_EPSILON:

        assert interest_accrued >= 0, f"Interest cannot go negative: {interest_accrued}, our epsilon is {INTEREST_EPSILON}"

        for entry in operation.entries:
            if entry.asset == asset:
                evt = distribute_to_entry(
                    entry,
                    state,
                    timestamp,
                    block_number,
                    interest_accrued,
                    max_interest_gain=max_interest_gain,
                )
                yield evt


def accrue_interest(
    state: State,
    on_chain_balances: Dict[AssetIdentifier, Decimal],
    interest_distribution: InterestDistributionOperation,
    block_timestamp: datetime.datetime,
    block_number: BlockNumber | None,
    max_interest_gain: Percent = 0.05,
    aave_financial_year=datetime.timedelta(seconds=SECONDS_PER_YEAR_INT),
) -> Iterable[BalanceUpdate]:
    """Update the internal ledger to match interest accrued on on-chain balances.

    - Read incoming on-chain balance updates

    - Distribute it to the trading positions based on our ``interest_distribution``

    - Set the interest sync checkpoint

    :param state:
        Strategy state.

    :param on_chain_balances:
        The current on-chain balances at ``block_number``.

    :param block_number:
        Last safe block read

    :param block_timestamp:
        The timestamp of ``block_number``.

    :param max_interest_gain:
        Abort if some asset has gained more interest than this threshold.

        A safety check to abort buggy code.

    :return:
        Balance update events applied to all positions.

    """


    if interest_distribution.duration == ZERO_TIMEDELTA:
        logger.info("accrue_interest(): Interest distribution duration zero, we probably got called twice in a row")
        return

    assert interest_distribution.duration > ZERO_TIMEDELTA, f"Tried to distribute interest for negative timespan {interest_distribution.start} - {interest_distribution.end}"

    block_number_str = f"{block_number,}" if block_number else "<no block>"
    logger.info(f"accrue_interest({block_timestamp}, {block_number_str})")

    part_of_year = interest_distribution.duration / aave_financial_year

    for asset, new_balance in on_chain_balances.items():

        # Track the effective interest for the asset
        asset_interest_data = interest_distribution.get_interest_data(asset)
        interest = float((new_balance - asset_interest_data.total) / asset_interest_data.total) / part_of_year
        asset_interest_data.effective_rate = interest

        logger.info("Effective interest is %f for the asset %s, %s, %s", interest, asset, new_balance, asset_interest_data.total)

        # We cannot generate interest events for zero updates,
        # as it breaks math
        if interest == 0:
            logger.warning(f"Effective interest is zero for the asset %s", asset)
            continue

        yield from distribute_interest_for_assets(
            interest_distribution,
            state,
            asset,
            block_timestamp,
            block_number,
            on_chain_balances[asset],
            max_interest_gain=max_interest_gain,
        )

    set_interest_checkpoint(state, block_timestamp, block_number, interest_distribution)

    #
    # events = []
    # for p in positions:
    #     if p.is_credit_supply():
    #         assert len(p.trades) <= 2, "This interest calculation does not support increase/reduce position"
    #
    #         new_amount = p.loan.collateral_interest.last_token_amount + accrued
    #
    #         # TODO: the collateral is stablecoin so this can be hardcode for now
    #         # but make sure to fetch it from somewhere later
    #         price = 1.0
    #
    #         evt = update_interest(
    #             state,
    #             p,
    #             p.pair.base,
    #             new_token_amount=new_amount,
    #             event_at=timestamp,
    #             asset_price=price,
    #         )
    #         events.append(evt)
    #
    #         # Make atokens magically appear in the simulated
    #         # backtest wallet. The amount must be updated, or
    #         # otherwise we get errors when closing the position.
    #         self.wallet.update_token_info(p.pair.base)
    #         self.wallet.update_balance(p.pair.base.address, accrued)
    #     elif p.is_leverage() and p.is_short():
    #         assert len(p.trades) <= 2, "This interest calculation does not support increase/reduce position"
    #
    #         accrued_collateral_interest = self.calculate_accrued_interest(
    #             universe,
    #             p,
    #             timestamp,
    #             "collateral",
    #         )
    #         accrued_borrow_interest = self.calculate_accrued_interest(
    #             universe,
    #             p,
    #             timestamp,
    #             "borrow",
    #         )
    #
    #         new_atoken_amount = p.loan.collateral_interest.last_token_amount + accrued_collateral_interest
    #         new_vtoken_amount = p.loan.borrowed_interest.last_token_amount + accrued_borrow_interest
    #
    #         atoken_price = 1.0
    #
    #         vtoken_price_structure = pricing_model.get_sell_price(
    #             timestamp,
    #             p.pair.get_pricing_pair(),
    #             p.loan.borrowed.quantity,
    #         )
    #         vtoken_price = vtoken_price_structure.price
    #
    #         vevt, aevt = update_leveraged_position_interest(
    #             state,
    #             p,
    #             new_vtoken_amount=new_vtoken_amount,
    #             new_token_amount=new_atoken_amount,
    #             vtoken_price=vtoken_price,
    #             atoken_price=atoken_price,
    #             event_at=timestamp,
    #         )
    #         events.append(vevt)
    #         events.append(aevt)

    # return events


def set_interest_checkpoint(
    state: State,
    timestamp: datetime.datetime,
    block_number: BlockNumber | None,
    distribution: InterestDistributionOperation | None = None,
):
    """Set the last updated at flag for rebase interest calcualtions at the internal state."""

    assert isinstance(timestamp, datetime.datetime)
    if block_number is not None:
        assert type(block_number) == int

    state.sync.interest.last_sync_at = timestamp
    state.sync.interest.last_sync_block = block_number

    # Always save the last distribution
    if distribution is not None:
        state.sync.interest.last_distribution = distribution

    block_number_str = f"{block_number,}" if block_number else "<no block>"
    logger.info(f"Interest check point set to {timestamp}, block: {block_number_str}")


def record_interest_rate(
    state: State,
    universe: TradingStrategyUniverse, 
    timestamp: datetime.datetime,
):
    """Record interest rate at the time opening position.

    - Currently support only credit supply positions
    - Sets `interest_rate_at_open` if not set yet
    """
    assert isinstance(universe, TradingStrategyUniverse)
    assert universe.has_lending_data()

    logger.info("Filling missing interest rate information")

    for p in state.portfolio.get_open_and_frozen_positions():
        if p.is_credit_supply():
            loan = p.loan

            last_interest_rate = universe.get_latest_supply_apr(
                timestamp=timestamp,
                tolerance=datetime.timedelta(days=7),
            ) / 100
            assert 0 < last_interest_rate < 1

            logger.info("Recording interest rate %f for %s at %s", last_interest_rate, p, timestamp)

            if not loan.collateral.interest_rate_at_open:
                loan.collateral.interest_rate_at_open = last_interest_rate

            loan.collateral.last_interest_rate = last_interest_rate


def sync_interests(
    *,
    web3,
    wallet_address: str,
    timestamp: datetime.datetime,
    state: State,
    universe: TradingStrategyUniverse,
    pricing_model: PricingModel,
) -> List[BalanceUpdate]:
    """Update position's interests on all tokens that receive interests

    - Credit supply positions: aToken
    - Short positions: aToken, vToken

    :param web3:
        Web3 connection to the active blockchain
    :param wallet_address:
        Hot wallet or vault address
    :param timestamp:
        Wall clock time
    :param state:
        Current strategy state
    :param universe:
        Trading universe that must include lending data
    :param pricing_model:
        Used to update asset price in loan
    """
    assert isinstance(timestamp, datetime.datetime), f"got {type(timestamp)}"
    if not universe.has_lending_data():
        # sync_interests() is not needed if the strategy isn't dealing with leverage
        return []

    previous_update_at = state.sync.interest.last_sync_at
    if not previous_update_at:
        # No interest based positions yet?
        logger.info(f"Interest sync checkpoint not set at {timestamp}, nothing to sync/cannot sync interest.")
        return []

    duration = timestamp - previous_update_at
    if duration <= ZERO_TIMEDELTA:
        logger.error(f"Sync time span must be positive: {previous_update_at} - {timestamp}")
        return []

    logger.info(
        "Starting interest distribution operation at: %s, previous update %s, syncing %s",
        timestamp,
        previous_update_at,
        duration,
    )

    record_interest_rate(state, universe, timestamp)

    interest_distribution = prepare_interest_distribution(
        state.sync.interest.last_sync_at,
        timestamp,
        state.portfolio,
        pricing_model
    )

    # Then sync interest back from the chain
    block_identifier = get_almost_latest_block_number(web3)
    balances = {}
    onchain_balances = fetch_address_balances(
        web3,
        wallet_address,
        interest_distribution.assets,
        filter_zero=True,
        block_number=block_identifier,
    )

    balances = {
        b.asset: b.amount
        for b in onchain_balances
    }

    # Then distribute gained interest (new atokens/vtokens) among positions
    events_iter = accrue_interest(state, balances, interest_distribution, timestamp, None)

    events = list(events_iter)

    return events
