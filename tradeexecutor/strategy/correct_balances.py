"""Apply accounting corrections on the strategy state.

- Read on-chain balances

- Compare them to the balances seen in the state

- Adjust statebalacnes to match chain based ones

- Generate the accounting events to reflect these changes

"""
import enum
from _decimal import Decimal
from dataclasses import dataclass
from typing import List, Iterable, Collection

from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import QUANTITY_EPSILON
from tradeexecutor.strategy.asset import get_relevant_assets, map_onchain_asset_to_position
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


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


@dataclass
class AccountingCorrection:

    type: AccountingCorrectionType

    position: TradingPosition | ReservePosition

    expected_amount: Decimal

    actual_amount: Decimal


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
    assert len(reserve_assets) > 0, "No reserve assets defined"

    assets = get_relevant_assets(pair_universe, reserve_assets, state)
    asset_balances = sync_model.fetch_onchain_balances(assets)

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

        if abs(actual_amount - expected_amount) > epsilon:
            yield AccountingCorrection(
                AccountingCorrectionType.unknown,
                position,
                expected_amount,
                actual_amount,
            )


def correct_balances(
        state: State,
        sync_model: SyncModel,
        corrections: List[AccountingCorrection],
        interactive=True,
):

    assets

