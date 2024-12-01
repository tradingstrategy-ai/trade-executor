"""Helpers for outputting strategy execution information to Python logging and Discord."""
import datetime
from io import StringIO
from typing import List, Iterable

from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount

#: See setup_discord_logging()
DISCORD_BREAK_CHAR = "…"


def format_trade(portfolio: Portfolio, trade: TradeExecution) -> List[str]:
    """Write a trade status line to logs.

    :return: List of log lines
    """
    pair = trade.pair
    if pair.info_url:
        link = pair.info_url
    else:
        link = ""

    if trade.is_buy():
        trade_type = "Buy"
    else:
        trade_type = "Sell"

    existing_position = portfolio.get_existing_open_position_by_trading_pair(trade.pair)
    if existing_position:
        # Quantity returns the total balance of unexecuted trades
        existing_balance = existing_position.get_quantity()
        amount = abs(trade.planned_quantity / existing_balance)
        existing_text = f", {amount*100:,.2f}% of existing position #{existing_position.position_id}"
    else:
        existing_text = ""
        existing_balance = 0

    if trade.price_structure:
        planned_price = trade.price_structure.price
    else:
        # Legacy unit tests
        planned_price = 0

    lines = [
        f"{trade_type:5} #{trade.trade_id} {pair.get_human_description()} v:${trade.get_planned_value():,.2f} p:${planned_price:,.4f}/{pair.base.token_symbol} q:{abs(trade.get_position_quantity())} {pair.base.token_symbol} {existing_text}",
    ]

    # Add existing balance
    if existing_position:
        lines += [
            f"Existing position: {existing_position}, with pre-trade balance: {existing_balance}"
        ]
    else:
        lines += [
            f"Opens a new position"
        ]

    if link:
        lines.append(f"Trading pair link: {link}")

    return lines


def format_position(
    position: TradingPosition,
    total_equity: USDollarAmount,
    up_symbol="🌲",
    down_symbol="🔻",
) -> List[str]:
    """Write a position status line to logs.

    Position can be open/closed.

    :return: List of log lines
    """
    symbol = up_symbol if position.get_total_profit_percent() >= 0 else down_symbol
    if position.pair.info_url:
        link = position.pair.info_url
    else:
        link = ""

    position_kind = position.pair.kind

    position_labels = {
        TradingPairKind.spot_market_hold: "spot",
        TradingPairKind.lending_protocol_short: "short",
        TradingPairKind.credit_supply: "credit",
    }

    position_label = position_labels.get(position_kind, "<unknown position type>")

    allocation = position.get_value() / total_equity

    if position.is_open():
        duration = position.get_duration()
    else:
        duration = datetime.datetime.utcnow() - position.opened_at

    lines = [
        f"{symbol} #{position.position_id} {position.pair.get_human_description()} {position_label} value: ${position.get_value():,.2f}, {allocation:.2f}% of portfolio",
        f"profit: {(position.get_total_profit_percent() * 100):.2f} % ({position.get_total_profit_usd():,.4f} USD)",
        f"duration: {duration}",
    ]

    if position.is_frozen():
        last_trade = "buy" if position.get_last_trade().is_buy() else "sell"
        lines.append(f"   last trade: {last_trade}, freeze reason: {position.get_freeze_reason()}")

    return lines


def output_positions(
    positions: Iterable[TradingPosition],
    total_equity: USDollarAmount,
    buf: StringIO,
    empty_message="No positions",
    break_after=4,
):
    """Write info on multiple trading positions formatted for Python logging system.

    :break_after:
        Insert Discord message break after this many positions to avoid
        chopped Discord messages.

    :return: A plain text string as a log message, suitable for Discord logging
    """

    assert type(total_equity) == float

    positions = list(positions)

    if len(positions) > 0:
        position: TradingPosition

        for idx, position in enumerate(positions):
            for line in format_position(position, total_equity):
                print("    " + line, file=buf)

            if (idx + 1) % break_after == 0:
                # Discord message break
                print(DISCORD_BREAK_CHAR, file=buf)
            else:
                # Line break
                print("", file=buf)

    else:
        print(f"    {empty_message}", file=buf)
    return buf.getvalue()


def output_trades(trades: List[TradeExecution], portfolio: Portfolio, buf: StringIO):
    """Write trades to the output logs."""
    for t in trades:
        for line in format_trade(portfolio, t):
            print("    " + line, file=buf)
        print("", file=buf)
