"""Profit and Loss (PnL) calculation helperes for trading positions."""
import dataclasses
import datetime

from tradeexecutor.state.types import USDollarPrice, USDollarAmount, Percent


@dataclasses.dataclass(frozen=True, slots=True)
class ProfitData:
    """Different profit values for a trading position."""
    profit_usd: USDollarAmount
    profit_pct: Percent
    profit_pct_annualised: Percent
    realised_usd: USDollarAmount
    unrealised_usd: USDollarAmount
    duration: datetime.timedelta


def calculate_pnl(
    position: "tradeexecutor.state.position.TradingPosition",
    end_at: datetime.datetime=None,
    mark_price: USDollarPrice=None,
) -> ProfitData:
    """Calculate the Profit and Loss (PnL) for a given trading position.

    - Use cumulative trading cost method
    - Calculate all profit variables in one pass
    - Works for realised and unrealised PnL

    See also :py:func:`tradeexecutor.visualisation.position.calculate_position_timeline`.

    :param end_at:
        For non-closed positions you need to provide the `end_at` timestamp until we calculate the PnL.
        
    :param mark_price:
        For non-closed positions you need to provide the `mark_price` to calculate the unrealised PnL.

        If not given, use ``TradingPosition.last_token_price``.

    :return:
        Profit in dollar and percentage, annualised percentage.
    """

    assert position.is_spot() or position.is_vault(), f"Test/fixc alculations for other position types: {position} - only spotty supported for now"

    cumulative_quantity = cumulative_cost = avg_price = 0
    realised_pnl_total = 0

    for trade in position.trades.values():
        delta = float(trade.executed_quantity)

        if delta > 0:
            # Buy: increase cost basis
            cumulative_cost += trade.get_value()
            cumulative_quantity += delta
        elif delta < 0:
            # Sell: reduce cost basis proportionally
            if cumulative_quantity > 0:
                avg_price = cumulative_cost / float(cumulative_quantity)
                cost_reduction = avg_price * abs(delta)
                cumulative_cost -= cost_reduction
                cumulative_quantity += delta  # delta is negative here

                realised_pnl = (trade.executed_price - avg_price) * abs(delta)
                realised_pnl_total += realised_pnl

        else:
            raise NotImplementedError(f"Got a trade withe executed quantity zero: {trade}")

    if position.is_closed():
        end_at = position.closed_at
    else:
        assert end_at, f"end_at must be provided for non-closed position: {position}"
        assert isinstance(end_at, datetime.datetime), f"end_at must be a datetime object: {end_at}"

    if not mark_price:
        mark_price = position.last_token_price

    unrealised_pnl = (mark_price * float(position.get_quantity())) - cumulative_cost

    duration = (end_at - position.opened_at)
    annualised_periods = datetime.timedelta(days=365) / duration
    profit_usd = realised_pnl_total + unrealised_pnl
    profit_pct = (profit_usd / cumulative_cost)
    profit_pct_annualised = (1 + profit_pct) ** annualised_periods - 1

    return ProfitData(
        profit_pct=profit_pct,
        profit_usd=profit_usd,
        profit_pct_annualised=profit_pct_annualised,
        realised_usd=realised_pnl_total,
        unrealised_usd=unrealised_pnl,
        duration=duration,
    )
