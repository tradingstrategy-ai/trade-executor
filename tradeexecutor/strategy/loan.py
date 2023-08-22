import datetime

from tradeexecutor.state.identifier import AssetWithTrackedValue, AssetType
from tradeexecutor.state.loan import Loan
from tradeexecutor.state.trade import TradeExecution


def create_short_loan(
    trade: TradeExecution,
) -> Loan:
    """Create the loan data tracking for short position.

    Check that the informatio looks correct for a short position.
    """

    assert trade.is_leverage_short()

    pair = trade.pair

    assert pair.base.underlying, "Base token lacks underlying asset"
    assert pair.quote.underlying, "Quote token lacks underlying asset"

    assert pair.base.type == AssetType.borrowed, f"Trading pair base asset is not borrowed: {pair.base}"
    assert pair.quote.type == AssetType.collateral, f"Trading pair quote asset is not collateral: {pair.base}"

    assert trade.planned_quantity < 0, f"Short sell must be a sell with negative quantity, got: {trade.planned_quantity}"
    assert trade.planned_reserve > 0, f"Collateral must be positive: {trade.planned_reserve}"

    # vToken
    borrowed = AssetWithTrackedValue(
        asset=pair.base,
        last_usd_price=trade.planned_price,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=abs(trade.planned_quantity),
        created_strategy_cycle_at=trade.strategy_cycle_at,
    )

    # aToken
    collateral = AssetWithTrackedValue(
        asset=pair.quote,
        last_usd_price=trade.planned_price,
        last_pricing_at=datetime.datetime.utcnow(),
        quantity=trade.planned_reserve,
    )

    loan = Loan(
        pair=trade.pair,
        collateral=collateral,
        borrowed=borrowed,
    )

    return loan
