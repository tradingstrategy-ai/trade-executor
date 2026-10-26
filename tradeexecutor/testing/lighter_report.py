"""Human-readable reports of Lighter custody trades for tutorials and diagnostics."""

from decimal import Decimal
from typing import Any

from tabulate import tabulate

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag


def describe_lighter_trade(trade: TradeExecution) -> str:
    """Return a one-line human description of a Lighter custody trade.

    :param trade:
        Any trade on an exchange-account position.
    :return:
        What moved, how much and between which accounts.
    """
    account = trade.pair.get_exchange_account_id()
    amount = abs(trade.executed_quantity if trade.is_success() else trade.planned_quantity)
    if trade.is_internal_account_transfer():
        source = trade.other_data.get("funding_source_account_id")
        return f"Transfer {amount} USDC from Lighter account {source} to account {account}"
    if TradeFlag.external_account_transfer in (trade.flags or set()):
        if trade.is_buy():
            return f"Deposit {amount} USDC from the Safe to Lighter account {account}"
        return f"Withdraw {amount} USDC from Lighter account {account} to the Safe"
    if amount == 0:
        return f"Open the position of Lighter account {account}"
    return trade.get_human_description()


def get_lighter_trade_hashes(trade: TradeExecution) -> list[str]:
    """Return the Ethereum transaction hashes and Lighter identifiers of a trade.

    :param trade:
        Any trade on an exchange-account position.
    :return:
        ``label: hash`` lines, Ethereum transactions first.
    """
    lines = [f"ethereum: {tx.tx_hash}" for tx in trade.blockchain_transactions if tx.tx_hash]
    other_data = trade.other_data or {}
    for key, label in (
        ("lighter_deposit_tx_hash", "ethereum deposit"),
        ("lighter_withdrawal_request_id", "withdrawal request"),
        ("lighter_claim_tx_hash", "ethereum claim"),
        ("lighter_transfer_tx_hash", "lighter transfer"),
    ):
        if other_data.get(key):
            lines.append(f"{label}: {other_data[key]}")
    return lines


def build_trade_summary(state: State) -> list[dict[str, Any]]:
    """Return every exchange-account trade with its description, status and hashes.

    :param state:
        Executor state.
    :return:
        One row per trade, in trade id order.
    """
    rows = []
    for trade in sorted(state.portfolio.get_all_trades(), key=lambda trade: trade.trade_id):
        if not trade.pair.is_exchange_account():
            continue
        rows.append({
            "trade": trade.trade_id,
            "position": trade.pair.get_ticker(),
            "description": describe_lighter_trade(trade),
            "status": trade.get_status().value,
            "notes": trade.notes or "",
            "hashes": "\n".join(get_lighter_trade_hashes(trade)) or "-",
        })
    return rows


def print_trade_summary(state: State) -> list[dict[str, Any]]:
    """Print :py:func:`build_trade_summary` as a grid table and return its rows."""
    rows = build_trade_summary(state)
    print(tabulate(rows, headers="keys", tablefmt="grid"))
    return rows


def get_lighter_funding_fees(state: State) -> Decimal:
    """Return the fees Lighter charged on the executor's transfers between accounts."""
    return sum((Decimal(trade.other_data.get("funding_fee_usdc", "0")) for trade in state.portfolio.get_all_trades() if trade.is_internal_account_transfer()), Decimal(0))
