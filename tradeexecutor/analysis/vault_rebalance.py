"""Vault rebalance status display.

Console utilities for displaying vault allocation status.
"""
import datetime
from typing import Callable

import pandas as pd
from tabulate import tabulate

from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def get_vault_rebalance_status(
    state: State,
    strategy_universe: TradingStrategyUniverse,
) -> tuple[pd.DataFrame, USDollarAmount]:
    """Get vault allocation status as a DataFrame.

    Shows all available vaults in the universe with their current allocations
    from open positions.

    :param state:
        Current trading state

    :param strategy_universe:
        Trading universe containing vault pairs

    :return:
        Tuple of (DataFrame with vault status, current cash amount)

        DataFrame columns:
        - Vault: Vault name
        - Protocol: Vault protocol slug
        - Address: Vault contract address
        - Available: Whether vault is available in the universe
        - Position ID: Position ID if we have an open position
        - Value USD: Current position value in USD
        - Weight %: Percentage of total portfolio value
        - Shares: Number of vault shares held
        - 1M CAGR: One month CAGR if available from vault metadata
    """

    # Get current cash
    cash = state.portfolio.get_cash()

    # Get all vault pairs from universe
    vault_pairs = [p for p in strategy_universe.iterate_pairs() if p.is_vault()]

    # Get open vault positions
    vault_positions = {
        p.pair.pool_address.lower(): p
        for p in state.portfolio.open_positions.values()
        if p.pair.is_vault()
    }

    # Calculate total portfolio value for weight calculation
    total_portfolio_value = state.portfolio.calculate_total_equity()

    rows = []

    # First add positions for vaults in universe
    for pair in vault_pairs:
        address = pair.pool_address.lower() if pair.pool_address else None
        position = vault_positions.get(address) if address else None

        if position:
            value = position.get_value()
            shares = position.get_quantity()
            position_id = position.position_id
            weight = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        else:
            value = 0
            shares = 0
            position_id = None
            weight = 0

        # Get 1M CAGR from vault metadata if available
        one_month_cagr = None
        metadata = pair.get_vault_metadata()
        if metadata:
            one_month_cagr = metadata.one_month_cagr

        rows.append({
            "Vault": pair.get_vault_name() or pair.base.token_symbol,
            "Protocol": pair.get_vault_protocol() or "-",
            "Address": pair.pool_address or "-",
            "Available": True,
            "Position ID": position_id,
            "Value USD": value,
            "Weight %": weight,
            "Shares": shares,
            "1M CAGR": one_month_cagr,
        })

        # Remove from dict so we can track orphaned positions
        if address in vault_positions:
            del vault_positions[address]

    # Add any orphaned vault positions (positions for vaults not in current universe)
    for address, position in vault_positions.items():
        value = position.get_value()
        shares = position.get_quantity()
        weight = (value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0

        # Get 1M CAGR from vault metadata if available
        one_month_cagr = None
        metadata = position.pair.get_vault_metadata()
        if metadata:
            one_month_cagr = metadata.one_month_cagr

        rows.append({
            "Vault": position.pair.get_vault_name() or position.pair.base.token_symbol,
            "Protocol": position.pair.get_vault_protocol() or "-",
            "Address": address,
            "Available": False,  # Not in current universe
            "Position ID": position.position_id,
            "Value USD": value,
            "Weight %": weight,
            "Shares": shares,
            "1M CAGR": one_month_cagr,
        })

    df = pd.DataFrame(rows)

    # Sort by value USD descending (largest position first)
    if len(df) > 0:
        df = df.sort_values("Value USD", ascending=False)
        df = df.reset_index(drop=True)

    return df, cash


def print_vault_rebalance_status(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    printer: Callable = print,
) -> pd.DataFrame:
    """Print vault rebalance status to console.

    Displays:
    - Current cash balance
    - All available vaults in the universe
    - Current allocations in those vaults from open positions
    - Sorted by largest position first

    Example usage in console:

    .. code-block:: python

        from tradeexecutor.analysis.vault_rebalance import print_vault_rebalance_status

        df = print_vault_rebalance_status(state, strategy_universe)

    :param state:
        Current trading state

    :param strategy_universe:
        Trading universe containing vault pairs

    :param printer:
        Function to use for printing output.
        Defaults to print().
        Use logger.info for logging output.

    :return:
        DataFrame with vault status for further analysis
    """

    df, cash = get_vault_rebalance_status(state, strategy_universe)

    # Calculate totals
    total_vault_value = df["Value USD"].sum() if len(df) > 0 else 0
    total_portfolio_value = state.portfolio.calculate_total_equity()

    # Print header
    printer("=" * 80)
    printer("VAULT REBALANCE STATUS")
    printer("=" * 80)
    printer("")

    # Print cash info
    printer(f"Current cash:           ${cash:,.2f}")
    printer(f"Total vault value:      ${total_vault_value:,.2f}")
    printer(f"Total portfolio value:  ${total_portfolio_value:,.2f}")
    printer("")

    # Print vault table
    if len(df) > 0:
        # Format for display
        display_df = df.copy()
        display_df["Value USD"] = display_df["Value USD"].apply(lambda x: f"${x:,.2f}")
        display_df["Weight %"] = display_df["Weight %"].apply(lambda x: f"{x:.2f}%")
        display_df["Shares"] = display_df["Shares"].apply(
            lambda x: f"{float(x):,.4f}" if x else "-"
        )
        display_df["Position ID"] = display_df["Position ID"].apply(
            lambda x: str(x) if x is not None else "-"
        )
        display_df["Available"] = display_df["Available"].apply(
            lambda x: "Yes" if x else "No"
        )
        # Truncate address for display
        display_df["Address"] = display_df["Address"].apply(
            lambda x: f"{x[:10]}...{x[-8:]}" if x and len(x) > 20 else x
        )
        # Format 1M CAGR as percentage
        display_df["1M CAGR"] = display_df["1M CAGR"].apply(
            lambda x: f"{x * 100:.1f}%" if x is not None else "-"
        )

        printer("Vault Allocations (sorted by value, largest first):")
        printer("-" * 80)
        printer(tabulate(display_df, headers="keys", tablefmt="simple", showindex=False))
    else:
        printer("No vaults found in universe")

    printer("")
    printer("=" * 80)

    return df
