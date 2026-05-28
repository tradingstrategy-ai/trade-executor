"""Latest cycle cash, allocation, and redemption snapshot."""

import pandas as pd

from tradeexecutor.analysis.cycle_messages import extract_usd_value
from tradeexecutor.strategy.chart.definition import ChartInput


def latest_cycle_snapshot(input: ChartInput) -> pd.DataFrame:
    """Cash, allocation, and redemption breakdown for the most recent cycle.

    Parses the last strategy cycle message for equity, cash, redeemable
    capital, investable equity, and allocation figures.  Combines with
    alpha-model accepted-size data when available.
    """
    state = input.state

    message_map = state.visualisation.get_messages_tail(2)
    if not message_map:
        return pd.DataFrame(columns=["Metric", "Value"])

    latest_cycle_label = list(message_map.keys())[0]
    latest_message = list(message_map.values())[0]

    total_equity = extract_usd_value(latest_message, "Total equity")
    cash = extract_usd_value(latest_message, "Cash")
    redeemable_capital = extract_usd_value(latest_message, "Redeemable capital")
    pending_redemptions = extract_usd_value(latest_message, "Pending redemptions")
    investable = extract_usd_value(latest_message, "Investable equity")
    accepted_investable = extract_usd_value(latest_message, "Accepted investable equity")
    allocated_to_signals = extract_usd_value(latest_message, "Allocated to signals")
    discarded_lit_liquidity = extract_usd_value(latest_message, "Discarded allocation because of lack of lit liquidity")

    rows = [
        ("Final cycle", latest_cycle_label),
        ("Total equity USD", total_equity),
        ("Cash USD", cash),
        ("Cash % of equity", cash / total_equity if total_equity else float("nan")),
        ("Redeemable capital USD", redeemable_capital),
        ("Pending redemptions USD", pending_redemptions),
        ("Investable equity USD", investable),
        ("Accepted investable equity USD", accepted_investable),
        ("Accepted / investable %", accepted_investable / investable if investable else float("nan")),
        ("Allocated to signals USD", allocated_to_signals),
        ("Discarded because of lit liquidity USD", discarded_lit_liquidity),
        ("Unaccepted investable USD", investable - accepted_investable),
        ("Unaccepted / investable %", (investable - accepted_investable) / investable if investable else float("nan")),
    ]

    return pd.DataFrame(rows, columns=["Metric", "Value"])
