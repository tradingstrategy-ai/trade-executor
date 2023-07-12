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
from dataclasses import dataclass
from typing import List, Iterable, Collection

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.asset import get_relevant_assets, map_onchain_asset_to_position
from tradeexecutor.strategy.sync_model import SyncModel


logger = logging.getLogger(__name__)


#: The amount of token units that is considered "dust" or rounding error.
#:
DUST_EPSILON = Decimal(10**-5)


class UnexpectedAccountingCorrectionIssue(Exception):
    """Something wrong in the token accounting we do not expect to be automatically correct."""


class AccountingCorrectionType(enum.Enum):

    #: Do not know what caused the incorrect amount
    unknown = "unknown"

    #: aUSDC
    rebase = "rebase"


class AccountingCorrectionAborted(Exception):
    """User presses n"""


@dataclass
class AccountingCorrection:

    type: AccountingCorrectionType

    #: Related on-chain asset
    asset: AssetIdentifier

    #: Related position
    position: TradingPosition | ReservePosition

    expected_amount: Decimal

    actual_amount: Decimal

    block_number: int | None

    timestamp: datetime.datetime | None

    #: Keep track of monetary value of corrections.
    #:
    #: An estimated value at the time of the correction creation.
    #:
    #: Negative for negative corrections
    #:
    usd_value: USDollarAmount

    def __repr__(self):
        return f"<Accounting correction type {self.type.value} for {self.position}, expected {self.expected_amount}, actual {self.actual_amount} at {self.timestamp}>"

    @property
    def quantity(self):
        """How many tokens we corrected"""
        return self.actual_amount - self.expected_amount


def calculate_account_corrections(
        pair_universe: PandasPairUniverse,
        reserve_assets: Collection[AssetIdentifier],
        state: State,
        sync_model: SyncModel,
        epsilon=DUST_EPSILON,
) -> Iterable[AccountingCorrection]:
    """Figure out differences between our internal ledger (state) and on-chain balances.


    :raise UnexpectedAccountingCorrectionIssue:
        If we find on-chain tokens we do not know how to map any of our strategy positions


    """

    assert isinstance(pair_universe, PandasPairUniverse)
    assert isinstance(state, State)
    assert isinstance(sync_model, EnzymeVaultSyncModel), "Only EnzymeVaultSyncModel tested for now"
    assert len(state.portfolio.reserves) > 0, "No reserve positions. Did you run init for the strategy?"

    logger.info("Scanning for account corrections")

    assets = get_relevant_assets(pair_universe, reserve_assets, state)
    asset_balances = list(sync_model.fetch_onchain_balances(assets))

    logger.info("Found %d on-chain tokens", len(asset_balances))

    for ab in asset_balances:
        position = map_onchain_asset_to_position(ab.asset, state)

        if position is None:
            raise UnexpectedAccountingCorrectionIssue(f"Could not map the on-chain balance to any known position:\n{ab}")

        if isinstance(position, TradingPosition):
            if position.is_closed():
                raise UnexpectedAccountingCorrectionIssue(f"Mapped found tokens to already closed position:\n"
                                                          f"{ab}\n"
                                                          f"{position}")

        actual_amount = ab.amount
        expected_amount = position.get_quantity()
        diff = actual_amount - expected_amount

        usd_value = position.calculate_quantity_usd_value(diff)

        logger.info("Fix needed %s worth of %f USD", ab.asset, usd_value)

        if abs(actual_amount - expected_amount) > epsilon:
            yield AccountingCorrection(
                AccountingCorrectionType.unknown,
                ab.asset,
                position,
                expected_amount,
                actual_amount,
                ab.block_number,
                ab.timestamp,
                usd_value,
            )


def apply_accounting_correction(
        state: State,
        correction: AccountingCorrection,
        strategy_cycle_included_at: datetime.datetime | None,
):
    """Update the state to reflect the true on-chain balances."""

    assert correction.type == AccountingCorrectionType.unknown, f"Not supported: {correction}"
    assert correction.timestamp

    portfolio = state.portfolio
    asset = correction.asset
    position = correction.position
    block_number = correction.block_number

    event_id = portfolio.next_balance_update_id
    portfolio.next_balance_update_id += 1

    logger.info("Corrected %s", position)

    if isinstance(position, TradingPosition):
        position_type = BalanceUpdatePositionType.open_position
        position_id = correction.position.position_id
    elif isinstance(position, ReservePosition):
        position_type = BalanceUpdatePositionType.reserve
        position_id = None
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

        # TODO: Close position if the new balance is zero
        assert position.get_quantity() > 0, "Position closing logic missing"

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
        corrections: List[AccountingCorrection],
        strategy_cycle_included_at: datetime.datetime | None,
        interactive=True,
) -> Iterable[BalanceUpdate]:
    """Apply the accounting corrections on the state (internal ledger).

    - Change values of the underlying positions

    - Create BalanceUpdate events and store them in the state

    - Create BalanceUpdateRefs and store them in the state

    .. note::

        You need to iterate the returend iterator to have any of the corrections applied.

    :return:
        Iterator of corrections.
    """

    if interactive:

        for c in corrections:
            print("Correction needed:", c)

        confirmation = input("Attempt to repair [y/n]").lower()
        if confirmation != "y":
            raise AccountingCorrectionAborted()

    for correction in corrections:
        yield apply_accounting_correction(state, correction, strategy_cycle_included_at)
