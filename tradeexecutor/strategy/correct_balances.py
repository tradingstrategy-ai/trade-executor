"""Apply accounting corrections on the strategy state.

- Read on-chain balances

- Compare them to the balances seen in the state

- Adjust statebalacnes to match chain based ones

- Generate the accounting events to reflect these changes

"""
from _decimal import Decimal
from dataclasses import dataclass
from typing import List, Iterable

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import QUANTITY_EPSILON
from tradeexecutor.strategy.asset import get_relevant_assets, map_onchain_asset_to_position
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class UnexpectedAccountingCorrectionIssue(Exception):
    """Something wrong in the token accounting we do not expect to be automatically correct."""


class AccountingCorrectionType:

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
        universe: TradingStrategyUniverse,
        state: State,
        sync_model: SyncModel,
        epsilon=QUANTITY_EPSILON,
) -> Iterable[AccountingCorrection]:
    """Figure out differences between our internal ledger (state) and on-chain balances.


    :raise UnexpectedAccountingCorrectionIssue:
        If we find on-chain tokens we do not know how to map any of our strategy positions


    """

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(state, State)
    assert isinstance(sync_model, EnzymeVaultSyncModel), "Only EnzymeVaultSyncModel tested for now"

    assets = get_relevant_assets(universe, state)
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

