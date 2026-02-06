"""Profit and Loss (PnL) calculation helperes for trading positions."""
import dataclasses
import datetime
import math

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.types import USDollarPrice, USDollarAmount, Percent


@dataclasses.dataclass(frozen=True, slots=True)
class ProfitData:
    """Different profit values for a trading position."""
    profit_usd: USDollarAmount
    profit_pct: Percent
    profit_pct_annualised: Percent
    realised_usd: USDollarAmount | None
    unrealised_usd: USDollarAmount | None
    duration: datetime.timedelta

    def is_win(self) -> bool:
        return self.profit_usd > 0

    def is_loss(self) -> bool:
        return not self.is_win()


@dataclasses.dataclass(frozen=True, slots=True)
class SharePriceData:
    """Profit calculation using internal share price method.

    Inspired by ERC-4626 vault mechanics where:

    - Buys mint shares at current share price
    - Sells burn shares at current share price
    - PnL changes share price (total_assets changes, supply stays same between trades)

    The key insight is that price movements affect total_assets but NOT total_supply,
    so share price naturally reflects accumulated returns independent of capital flows.

    See :py:func:`calculate_share_price_pnl` for the calculation algorithm.
    """

    #: Starting share price (always 1.0)
    initial_share_price: float

    #: Current share price after all trades and price movements
    current_share_price: float

    #: Total shares outstanding
    total_supply: float

    #: Current USD value of the position (quantity * mark_price)
    total_assets: USDollarAmount

    #: Profit percentage: (current_share_price / initial_share_price) - 1
    profit_pct: Percent

    #: Profit in USD
    profit_usd: USDollarAmount

    #: Position duration
    duration: datetime.timedelta

    def is_win(self) -> bool:
        return self.profit_usd > 0

    def is_loss(self) -> bool:
        return not self.is_win()


def calculate_pnl(
    position: "tradeexecutor.state.position.TradingPosition",
    end_at: datetime.datetime=None,
    mark_price: USDollarPrice=None,
    epsilon: float=1e-6,
    max_annualised_profit: Percent = 100,  # 1000% annualised profit, or 100x
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

    assert position.is_spot() or position.is_vault(), f"Test/fix calculations for other position types: {position} - only spotty supported for now"

    cumulative_quantity = cumulative_cost = avg_price = 0
    realised_pnl_total = 0
    cumulative_value = 0

    for trade in position.trades.values():
        delta = float(trade.executed_quantity or 0)

        if delta > 0:
            # Buy: increase cost basis
            cumulative_cost += trade.get_value()
            cumulative_value += trade.get_value()
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

        elif not (trade.is_failed or trade.is_repaired or trade.is_repair_trade):
            # only raise error for 0 value if not a failed/repaired/repair trade
            raise NotImplementedError(f"Got a trade with executed quantity zero: {trade}")

    if abs(cumulative_cost) < epsilon:
        # Clean up 18 decimal madness
        cumulative_cost = 0.0

    if position.is_closed():
        end_at = position.closed_at
    else:
        assert end_at, f"end_at must be provided for non-closed position: {position}"
        assert isinstance(end_at, datetime.datetime), f"end_at must be a datetime object: {end_at}"

    if not mark_price:
        mark_price = position.last_token_price

    unrealised_pnl = (mark_price * float(position.get_quantity())) - cumulative_cost

    duration = (end_at - position.opened_at)
    assert duration > datetime.timedelta(0), f"Position {position} has a negative duration: {duration}, opened at {position.opened_at}, closed at {end_at}"
    annualised_periods = datetime.timedelta(days=365) / duration
    profit_usd = realised_pnl_total + unrealised_pnl
    profit_pct = (profit_usd / cumulative_value) if cumulative_value else 0

    try:
        profit_pct_annualised = (1 + profit_pct) ** annualised_periods - 1
    except OverflowError as e:
        # If we make a very short position (few hours) and make gains like 4%,
        # Python floating point will overflow when calculating annualised profit.
        profit_pct_annualised = math.copysign(max_annualised_profit, profit_pct)
    except Exception as e:
        raise RuntimeError(f"Failed to annualise profit_pct {profit_pct} for position {position} with duration {duration}, periods {annualised_periods}") from e

    return ProfitData(
        profit_pct=profit_pct,
        profit_usd=profit_usd,
        profit_pct_annualised=profit_pct_annualised,
        realised_usd=realised_pnl_total,
        unrealised_usd=unrealised_pnl,
        duration=duration,
    )



def calculate_share_price_pnl(
    position: "tradeexecutor.state.position.TradingPosition",
    end_at: datetime.datetime = None,
    mark_price: USDollarPrice = None,
    initial_share_price: float = 1.0,
) -> SharePriceData:
    """Calculate PnL using internal share price method.

    This method tracks an internal share price inspired by ERC-4626 vault mechanics:

    1. Share price starts at ``initial_share_price`` (default 1.0)
    2. Buys mint shares: ``shares = buy_value / share_price``
    3. Sells burn shares proportionally based on quantity sold
    4. Share price = total_assets / total_supply

    The key insight is that price movements (PnL) affect ``total_assets`` but NOT ``total_supply``,
    so share price naturally reflects accumulated returns independent of capital flows.

    Example:

    - Buy $100 of ETH at share price 1.0 → mint 100 shares, total_supply=100
    - ETH price increases 10% → total_assets=$110, share_price=1.1
    - Buy another $100 at share price 1.1 → mint 90.9 shares, total_supply=190.9
    - Profit is tracked via share price appreciation, not absolute dollars

    :param position:
        The trading position to calculate PnL for.

    :param end_at:
        For non-closed positions, the timestamp to calculate PnL until.

    :param mark_price:
        For non-closed positions, the price to value remaining assets.
        If not given, uses ``TradingPosition.last_token_price``.

    :param initial_share_price:
        Starting share price, default 1.0.

    :return:
        SharePriceData with current share price, total supply, total assets, and profit metrics.
    """

    assert position.is_spot() or position.is_vault(), \
        f"Share price calculation only supports spot/vault positions: {position}"

    current_share_price = initial_share_price
    total_supply = 0.0
    cumulative_quantity = 0.0
    total_invested = 0.0
    peak_total_supply = 0.0  # Track the peak supply for closed position calculations

    for trade in position.trades.values():
        delta = float(trade.executed_quantity or 0)
        trade_value = trade.get_value()

        if delta > 0:
            # Buy: mint shares at current share price
            shares_to_mint = trade_value / current_share_price
            total_supply += shares_to_mint
            cumulative_quantity += delta
            total_invested += trade_value
            peak_total_supply = max(peak_total_supply, total_supply)

            # After buying, share price stays same (we minted exactly enough shares)
            # share_price = total_assets / total_supply = (qty * price) / (value / share_price)
            # This is correct by construction

        elif delta < 0:
            # Sell: first update share price based on sell price, then burn proportional shares
            # The share price reflects the value we're getting out
            sell_quantity = abs(delta)

            if cumulative_quantity > 0:
                # Calculate proportion of position being sold
                proportion_sold = sell_quantity / cumulative_quantity

                # Update share price based on current value (at sell price)
                total_assets_before_sell = cumulative_quantity * trade.executed_price
                if total_supply > 0:
                    current_share_price = total_assets_before_sell / total_supply

                # Burn proportional shares
                shares_to_burn = total_supply * proportion_sold
                total_supply = max(0.0, total_supply - shares_to_burn)
                cumulative_quantity -= sell_quantity

        elif not (trade.is_failed or trade.is_repaired or trade.is_repair_trade):
            raise NotImplementedError(f"Got a trade with executed quantity zero: {trade}")

    # Determine end time and mark price
    if position.is_closed():
        end_at = position.closed_at
    else:
        assert end_at, f"end_at must be provided for non-closed position: {position}"
        assert isinstance(end_at, datetime.datetime), f"end_at must be a datetime object: {end_at}"

    if not mark_price:
        mark_price = position.last_token_price

    # Calculate final total_assets at current mark price
    final_quantity = float(position.get_quantity())
    total_assets = final_quantity * mark_price

    # Final share price calculation
    if total_supply > 0:
        current_share_price = total_assets / total_supply
    # else: keep the last calculated share price from the final sell

    # Calculate profit metrics
    profit_pct = (current_share_price / initial_share_price) - 1

    # Calculate profit in USD
    if total_supply > 0:
        # Open position: unrealised profit based on current value vs initial investment at current supply
        profit_usd = total_assets - (total_supply * initial_share_price)
    else:
        # Closed position: profit is what we got out minus what we put in
        total_sold = sum(t.get_value() for t in position.trades.values() if t.is_sell())
        profit_usd = total_sold - total_invested

    duration = end_at - position.opened_at
    assert duration > datetime.timedelta(0), \
        f"Position {position} has non-positive duration: {duration}"

    return SharePriceData(
        initial_share_price=initial_share_price,
        current_share_price=current_share_price,
        total_supply=total_supply,
        total_assets=total_assets,
        profit_pct=profit_pct,
        profit_usd=profit_usd,
        duration=duration,
    )


def calculate_pnl_generic(
    position: "tradeexecutor.state.position.TradingPosition",
    end_at: datetime.datetime=None,
    mark_price: USDollarPrice=None,
    epsilon: float=1e-6,
) -> ProfitData:
    """Handle different position types generically."""

    match position.pair.kind:
        case TradingPairKind.spot_market_hold | TradingPairKind.vault | TradingPairKind.freqtrade:
            return calculate_pnl(
                position=position,
                end_at=end_at,
                mark_price=mark_price,
                epsilon=epsilon,
            )
        case _:
            # Legacy path for other position types
            pnl_usd = position.get_total_profit_usd()
            pnl_pct = position.get_total_profit_percent(end_at=end_at)
            duration = position.get_duration(end_at=end_at)
            pnl_pct_annualised = position.calculate_total_profit_percent_annualised(end_at=end_at, calculation_method="legacy")
            return ProfitData(
                profit_usd=pnl_usd,
                profit_pct=pnl_pct,
                profit_pct_annualised=pnl_pct_annualised,
                realised_usd=None,  # Not applicable for non-spot/vault positions
                unrealised_usd=None,  # Not applicable for non-spot/vault positions
                duration=duration,
            )
