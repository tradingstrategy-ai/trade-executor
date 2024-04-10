"""Apply accounting corrections on the strategy state.

- Read on-chain balances

- Compare them to the balances seen in the state

- Adjust statebalacnes to match chain based ones

- Generate the accounting events to reflect these changes

"""
import logging
import datetime
import enum
from _decimal import Decimal
from collections import Counter
from dataclasses import dataclass
from typing import List, Iterable, Collection, Tuple, Dict, Set

import pandas as pd
from web3 import Web3
from web3.types import BlockIdentifier

from eth_defi.provider.broken_provider import get_almost_latest_block_number
from eth_defi.token import fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_typing import HexAddress

from eth_defi.tx import AssetDelta
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.generic_position import GenericPosition
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.repair import close_position_with_empty_trade
from tradeexecutor.strategy.dust import DEFAULT_DUST_EPSILON, get_dust_epsilon_for_pair, get_dust_epsilon_for_asset
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.asset import get_relevant_assets, map_onchain_asset_to_position, build_expected_asset_map
from tradeexecutor.strategy.sync_model import SyncModel


logger = logging.getLogger(__name__)


#: The default % we allow the balance to drift before we consider it a mismatch.
#:
#: Set to 10 BPS
#:
RELATIVE_EPSILON = Decimal(10**-4)


class UnexpectedAccountingCorrectionIssue(Exception):
    """Something wrong in the token accounting we do not expect to be automatically correct."""


class AccountingCorrectionCause(enum.Enum):

    #: Do not know what caused the incorrect amount
    unknown_cause = "unknown_cause"

    #: aUSDC, etc.
    rebase = "rebase"


class AccountingCorrectionAborted(Exception):
    """User presses n"""


@dataclass
class AccountingBalanceCheck:
    """Accounting correction applied to a balance.

    Any irregular accounting correction will cause the position profit calcualtions
    and such to become invalid. Such positions should be separately market
    and not included in the profit calculations.
    """

    type: AccountingCorrectionCause

    #: Where is this token being stored
    #:
    #: Hot wallet address or Enzyme vault address
    holding_address: str

    #: Related on-chain asset
    asset: AssetIdentifier

    #: Related positions
    #:
    #: Set none if no open position was found
    #:
    positions: Set[GenericPosition] | None

    expected_amount: Decimal

    actual_amount: Decimal

    #: Dust epsilon
    dust_epsilon: Decimal

    #: Relative epsilon
    relative_epsilon: Decimal

    block_number: int | None

    timestamp: datetime.datetime | None

    #: Keep track of monetary value of corrections.
    #:
    #: An estimated value at the time of the correction creation.
    #:
    #: Negative for negative corrections
    #:
    #: `None` if the the tokens are for a new position and we do not have pricing information yet available,
    #: or if the position is not a spot position.
    #:
    usd_value: USDollarAmount | None

    #: Is this correction for reserve asset
    #:
    reserve_asset: bool

    #: Was there a balance mismatch that is larger than the epsilon
    #:
    mismatch: bool

    def __repr__(self):

        if self.position:
            position_name = self.position.get_human_readable_name()
        else:
            position_name = "unknown trading position"

        return f"<Accounting correction type {self.type.value} for {position_name} asset {self.asset.token_symbol}, expected {self.expected_amount}, actual {self.actual_amount} at {self.timestamp}>"

    @property
    def quantity(self):
        """How many tokens we corrected"""
        return self.actual_amount - self.expected_amount

    @property
    def position(self) -> GenericPosition | None:
        """Backwards compatibility.

        TODO: Remove code paths touching this
        """
        if len(self.positions) >= 1:
            return next(iter(self.positions))
        return None

    def has_extra_tokens(self) -> bool:
        """We have extra"""
        return self.quantity > 0

    def is_dusty(self) -> bool:
        """If there is a mismatch, is the mismatch within the dust tolerance,"""

        # Perfect accounting match
        if self.quantity == 0:
            return False

        # We have a mismatch, but is it larger
        # than the dust epsilon
        return not is_relative_mismatch(
            self.actual_amount,
            self.expected_amount,
            self.dust_epsilon,
            self.relative_epsilon,
        )

    def is_mismatch(self) -> bool:
        return self.mismatch


def is_relative_mismatch(
        actual_amount,
        expected_amount,
        relative_epsilon,
        dust_epsilon,
) -> bool:
    """Calculate if we are within the relative tolerance.

    Mismatch has two methods of ronding

    - Close to zero as absolute units (dust)

    - Relative % of the position size
    """

    # Accounting dust.
    # The position has been closed but we have left fractions of tokens on the account.
    # Cannot be compared with relative match.
    if abs(actual_amount) < dust_epsilon and abs(expected_amount) < dust_epsilon:
        return False

    # Avoid division by zero
    if actual_amount == 0 or expected_amount == 0:
        return actual_amount != expected_amount

    return abs((expected_amount - actual_amount) / actual_amount) > relative_epsilon


def calculate_total_assets(portfolio: Portfolio) -> Dict[AssetIdentifier, Decimal]:
    """Calculate total tokens the portfolio should held."""

    assets: Counter[AssetIdentifier, Decimal] = Counter()
    for p in portfolio.get_open_and_frozen_positions():
        for asset, quantity in p.get_held_assets():
            assets[asset] += quantity

    return assets


def calculate_account_corrections(
    pair_universe: PandasPairUniverse,
    reserve_assets: Collection[AssetIdentifier],
    state: State,
    sync_model: SyncModel,
    relative_epsilon=RELATIVE_EPSILON,
    all_balances=False,
    block_identifier: BlockIdentifier = None,
) -> Iterable[AccountingBalanceCheck]:
    """Figure out differences between our internal ledger (state) and on-chain balances.

    :param pair_universe:
        Needed to know what asses we are looking for

    :param reserve_assets:
        Needed to know what asses we are looking for

    :param state:
        The current state of the internal ledger

    :param sync_model:
        How ot access on-chain balances

    :param dust_epsilon:
        Minimum amount of token (abs quantity) before it is considered as a rounding error

    :param all_balances:
        If `True` iterate all balances even if there are no mismatch.

    :param block_identifier:
        Check at certain account height

    :raise UnexpectedAccountingCorrectionIssue:
        If we find on-chain tokens we do not know how to map any of our strategy positions

    :return:
        Difference in balances or all balances if `all_balances` is true.

        Yield one entry per token in positions.
    """

    assert isinstance(pair_universe, PandasPairUniverse)
    assert isinstance(state, State)
    assert len(state.portfolio.reserves) > 0, "No reserve positions. Did you run init for the strategy?"

    logger.info(
        "Scanning for account corrections, we have %d open positions, %d frozen positions",
        len(state.portfolio.open_positions),
        len(state.portfolio.frozen_positions),
    )

    if len(state.portfolio.frozen_positions) > 0:
        logger.warning("Be careful when doing check-accounts for frozen positions, as you should run repair first.")

    # assets = get_relevant_assets(pair_universe, reserve_assets, state)
    asset_to_position = build_expected_asset_map(state.portfolio, pair_universe=pair_universe)

    asset_balances = sync_model.fetch_onchain_balances(
        asset_to_position.keys(),
        filter_zero=False,
        block_identifier=block_identifier
    )
    asset_balances = list(asset_balances)

    logger.info("Found %d on-chain tokens", len(asset_balances))

    for ab in asset_balances:

        asset = ab.asset
        mapping = asset_to_position[asset]

        actual_amount = ab.amount
        expected_amount = mapping.quantity

        # position = map_onchain_asset_to_position(ab.asset, state)

        # if isinstance(position, TradingPosition):
        #    if position.is_closed():
        #        raise UnexpectedAccountingCorrectionIssue(f"Mapped found tokens to already closed position:\n"
        #                                                  f"{ab}\n"
        #                                                  f"{position}")

        # if isinstance(position, TradingPosition):
        #    # We might have balances tied up in frozen positions for the same pair
        #    for frozen_position in state.portfolio.frozen_positions.values():
        #        if frozen_position.pair == position.pair:
        #            expected_amount += frozen_position.get_quantity()

        diff = actual_amount - expected_amount

        reserve = mapping.is_for_reserve()

        if len(mapping.positions) == 0:
            # This asset does not have open our closed positions,
            # but is present in the trading universe
            position = None
            usd_value = None
            dust_epsilon = 0
        elif mapping.is_one_to_one_asset_to_position():
            position = mapping.get_only_position()

            if isinstance(position, TradingPosition):
                dust_epsilon = get_dust_epsilon_for_pair(position.pair)
            elif isinstance(position, ReservePosition):
                dust_epsilon = get_dust_epsilon_for_asset(position.asset)
            elif position is None:
                dust_epsilon = DEFAULT_DUST_EPSILON
            else:
                raise NotImplementedError(f"Could not figure out position: {position}")

            usd_value = position.calculate_quantity_usd_value(diff) if position else None
        else:
            # Loan based positions have multiple assets, both in base and quote.
            # We use some values from the first position (across multiple) to
            # estimate values.
            first_position = mapping.get_first_position()
            dust_epsilon = get_dust_epsilon_for_asset(asset)
            usd_value = None

        logger.debug("Correction check worth of %s worth of %f USD, actual amount %s, expected amount %s", ab.asset, usd_value or 0, actual_amount, expected_amount)

        mismatch = is_relative_mismatch(actual_amount, expected_amount, relative_epsilon, dust_epsilon)

        if mismatch or all_balances:
            yield AccountingBalanceCheck(
                AccountingCorrectionCause.unknown_cause,
                sync_model.get_token_storage_address(),
                ab.asset,
                mapping.positions,
                expected_amount,
                actual_amount,
                dust_epsilon,
                relative_epsilon,
                ab.block_number,
                ab.timestamp,
                usd_value,
                reserve,
                mismatch,
            )


def apply_accounting_correction(
        state: State,
        correction: AccountingBalanceCheck,
        strategy_cycle_included_at: datetime.datetime | None,
):
    """Update the state to reflect the true on-chain balances."""

    assert correction.type == AccountingCorrectionCause.unknown_cause, f"Not supported: {correction}"
    assert correction.timestamp

    frozen_count = len(state.portfolio.frozen_positions)
    if frozen_count > 0:
        raise AssertionError(f"We have {frozen_count} frozen positions. Run repair for these first before attempting an accounting correction.")

    portfolio = state.portfolio
    asset = correction.asset
    position = correction.position
    block_number = correction.block_number

    event_id = portfolio.next_balance_update_id
    portfolio.next_balance_update_id += 1

    if isinstance(position, TradingPosition):
        position_type = BalanceUpdatePositionType.open_position
        position_id = correction.position.position_id
        assert position.is_spot(), f"Correction not yet implemented for leveraged positions"
        logger.info("Correcting spot %s, asset %s, %f -> %f", position.get_human_readable_name(), asset, correction.expected_amount, correction.actual_amount)
        # assert position.is_open(), f"Cannot correct already closed positions, got {position}"
    elif isinstance(position, ReservePosition):
        position_type = BalanceUpdatePositionType.reserve
        position_id = None
        logger.info("Correcting reserve %s, asset %s, %f -> %f", position, asset, correction.expected_amount, correction.actual_amount)
    elif position is None:
        # Tokens were for a trading position, but no position was open.
        # Open a new position
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    notes = f"Accounting correction based on the actual on-chain balances.\n" \
        f"The internal ledger balance was  {correction.expected_amount} {asset.token_symbol}\n" \
        f"On-chain balance was {correction.actual_amount} {asset.token_symbol} at block {block_number or 0:,}\n" \
        f"Balance was updated {correction.quantity} {asset.token_symbol}\n"

    evt = BalanceUpdate(
        balance_update_id=event_id,
        position_type=position_type,
        cause=BalanceUpdateCause.correction,
        asset=correction.asset,
        block_mined_at=correction.timestamp,
        strategy_cycle_included_at=strategy_cycle_included_at,
        chain_id=asset.chain_id,
        old_balance=correction.actual_amount,
        usd_value=correction.usd_value,
        quantity=correction.quantity,
        owner_address=None,
        tx_hash=None,
        log_index=None,
        position_id=position_id,
        block_number=correction.block_number,
        notes=notes,
    )

    assert evt.balance_update_id not in position.balance_updates, f"Alreaddy written: {evt}"
    position.balance_updates[evt.balance_update_id] = evt

    ref = BalanceEventRef(
        balance_event_id=evt.balance_update_id,
        strategy_cycle_included_at=strategy_cycle_included_at,
        cause=evt.cause,
        position_type=position_type,
        position_id=evt.position_id,
        usd_value=evt.usd_value,
    )

    if isinstance(position, TradingPosition):
        # Balance_updates toggle is enough
        position.balance_updates[evt.balance_update_id] = evt

        # The position has gone to zero
        if position.can_be_closed():
            # In a lot of places we assume that a position with 1 trade cannot be closed
            # Make a 0-sized trade so that we know the position is closed
            t = close_position_with_empty_trade(portfolio, position)
            logger.info("Position %s closed with a trade %s", position, t)
            assert position.is_closed()
        else:
            assert position.get_quantity() > 0, \
                f"Spoit position should have positive quantity, got {position} with {position.get_quantity()}\n" \
                f"Accounting correction is: {correction}"

    elif isinstance(position, ReservePosition):
        # No fancy method to correct reserves
        position.quantity += correction.quantity
    else:
        raise NotImplementedError()

    # Bump our last updated date
    accounting = state.sync.accounting
    accounting.balance_update_refs.append(ref)
    accounting.last_updated_at = datetime.datetime.utcnow()
    accounting.last_block_scanned = evt.block_number

    return evt


def correct_accounts(
    state: State,
    corrections: List[AccountingBalanceCheck],
    strategy_cycle_included_at: datetime.datetime | None,
    tx_builder: TransactionBuilder,
    interactive=True,
    unknown_token_receiver: HexAddress | str | None = None,
    block_identifier: BlockIdentifier = None,
    block_timestamp: datetime.datetime = None,
) -> Iterable[BalanceUpdate]:
    """Apply the accounting corrections on the state (internal ledger).

    - Change values of the underlying positions

    - Create BalanceUpdate events and store them in the state

    - Create BalanceUpdateRefs and store them in the state

    .. note::

        You need to iterate the returend iterator to have any of the corrections applied.

    :return:
        Tuple (corrected anythimg, iterator of corrections).
    """

    if interactive:

        for c in corrections:
            print("Correction needed:", c)

        print(f"Any tokens that cannot be assigned to an open position will be send to {unknown_token_receiver}")
        confirmation = input("Attempt to repair [y/n]").lower()
        if confirmation != "y":
            raise AccountingCorrectionAborted()

    for correction in corrections:

        position = correction.position
        closed = False
        if isinstance(position, TradingPosition):
            if position.is_closed():
                closed = True

        # Could not map to open position,
        # but we do not have code to open new positions yet.
        # Just deal with it by transferring away.
        if position is None:
            logger.info("Asset transfer without position: %s", correction)
            transfer_away_assets_without_position(
                correction,
                unknown_token_receiver,
                tx_builder,
            )
        elif closed:
            logger.info("Asset transfer with closed position: %s", correction)
            # We have tokens on a closed position.
            # Likely we have a failure, we closed position internally,
            # but the selling trade failed to execute.
            # Alternatively we reopenend a position,
            # but the buying trade failed to execute.
            transfer_away_assets_without_position(
                correction,
                unknown_token_receiver,
                tx_builder,
            )
        else:
            logger.info("Internal state balance fix: %s", correction)
            # Change open position balance to match the on-chain balance
            yield apply_accounting_correction(state, correction, strategy_cycle_included_at)

    # Update last scanned block, so we do not rescan events we might have skipped
    if block_identifier is not None:
        state.sync.treasury.last_block_scanned = block_identifier
        if block_timestamp:
            state.sync.treasury.last_updated_at = block_timestamp

    else:
        logger.warning("Treasury sync block identifier missing")


def transfer_away_assets_without_position(
    correction: AccountingBalanceCheck,
    unknown_token_receiver: HexAddress | str,
    tx_builder: TransactionBuilder,
):
    """Transfer away non-reserve assets that cannot be mapped to an open position.

    TODO: Correct approach would be to open a new trading position
    directly in the correction, but it's complicated and we do not want to get there yet.

    :param correction:

    :param unknown_token_receiver:
    """

    position = correction.position

    if isinstance(position, TradingPosition):
        # Don't move tokens away from open position
        assert position.is_closed()
    else:
        # No closed or open position
        assert correction.position is None

    assert not correction.reserve_asset

    web3 = tx_builder.web3
    asset = correction.asset

    token = fetch_erc20_details(
        web3,
        asset.address,
    )

    tokens_to_transfer = correction.quantity
    tokens_to_transfer_raw = token.convert_to_raw(tokens_to_transfer)

    asset_delta = AssetDelta(
        Web3.to_checksum_address(token.address),
        -tokens_to_transfer_raw,
    )

    logger.info(f"Transfering %s %s to the clean up wallet %s as we could not map the token to any open position",
                correction.quantity,
                asset.token_symbol,
                unknown_token_receiver)

    args_bound_func = token.contract.functions.transfer(
        Web3.to_checksum_address(unknown_token_receiver),
        tokens_to_transfer_raw
    )

    blockchain_data = tx_builder.sign_transaction(
        token.contract,
        args_bound_func,
        gas_limit=250_000,
        asset_deltas=[asset_delta],
        notes="Accounting correction transaction, removing assets",
    )

    tx_hash = web3.eth.send_raw_transaction(blockchain_data.get_prepared_raw_transaction())
    logger.info("Broadcasted %s", tx_hash.hex())
    assert_transaction_success_with_explanation(web3, tx_hash)
    logger.info("Fix tx %s complete", tx_hash.hex())


def check_accounts(
    pair_universe: PandasPairUniverse,
    reserve_assets: Collection[AssetIdentifier],
    state: State,
    sync_model: SyncModel,
    block_identifier: BlockIdentifier = None,
) -> Tuple[bool, pd.DataFrame]:
    """Create a summary accounting corrections needed.

    Create a human-readable DataFrame of accounting inconsistencies.

    :param pair_universe:
        Trading pairs we have.

        Needed to get token addresses we read on-chain.

    :param reserve_assets:
        Cannot be deducted from pair universe.

    :param state:
        Current strategy state we check

    :param block_identifier:
        Check at certain block height

    :return:
        Tuple (accounts clean, accounting clean Dataframe that can be printed to the console)
    """

    # Any unbroadcasted trades need to be cleaned up first
    state.check_if_clean()

    if block_identifier is None:
        web3 = sync_model.web3
        block_identifier = get_almost_latest_block_number(web3)

    logger.info(f"Checking accounts at block {block_identifier:,}")

    clean = True
    corrections = calculate_account_corrections(
        pair_universe,
        reserve_assets,
        state,
        sync_model,
        relative_epsilon=RELATIVE_EPSILON,
        all_balances=True,
        block_identifier=block_identifier,
    )

    idx = []
    items = []
    for c in corrections:
        idx.append(c.asset.token_symbol)

        match c.position:
            case None:
                position_label = "No open position"
            case ReservePosition():
                position_label = "Reserves"
            case TradingPosition():
                position_label = c.position.pair.get_ticker()
            case _:
                raise NotImplementedError()

        dust = c.is_dusty()

        relative_diff = (c.actual_amount - c.expected_amount) / c.expected_amount

        items.append({
            "Address": c.asset.address,
            "Position": position_label,
            "Actual amount": c.actual_amount,
            "Expected amount": c.expected_amount,
            "Diff": c.quantity,
            "Dusty": "Y" if dust else "N",
            "Mismatch": f"{relative_diff * 100:.2f}%" if c.mismatch else "N",
            "Dust epsilon": c.dust_epsilon,
            "Relative epsilon": c.relative_epsilon,
        })

        if c.mismatch:
            clean = False

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return clean, df
