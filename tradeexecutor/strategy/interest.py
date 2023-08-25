"""Functions to refresh accrued interest on credit positions."""
import datetime
from decimal import Decimal

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.identifier import TradingPairKind, AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State


def update_credit_supply_interest(
    state: State,
    position: TradingPosition,
    asset: AssetIdentifier,
    new_atoken_amount: Decimal,
    event_at: datetime.datetime,
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

    :param new_atoken_amount:
        The new on-chain value of aToken/vToken tracking the loan.

    :param event_at:
        Block mined timestamp

    """

    assert asset is not None
    assert position.pair.kind == TradingPairKind.credit_supply
    assert position.is_open() or position.is_frozen(), f"Cannot update interest for position {position}"

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

    assert asset.underlying.is_stablecoin(), f"Credit supply is currently supported for stablecoin assets with 1:1 USD price assumption. Got: {asset}"

    old_balance = interest.last_atoken_amount
    gained_interest = new_atoken_amount - old_balance
    usd_value = float(new_atoken_amount)

    assert 0 < gained_interest < 999, f"Unlikely gained_interest: {gained_interest}, old quantity: {old_balance}, new quantity: {new_atoken_amount}"

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
    )

    position.add_balance_update_event(evt)

    # Update interest stats
    interest.last_accrued_interest = position.calculate_accrued_interest_quantity()
    interest.last_updated_at = datetime.datetime.utcnow()
    interest.last_event_at = event_at
    interest.last_updated_block_number = block_number
    interest.last_atoken_amount = new_atoken_amount
    return evt
