"""Parse structured data from strategy cycle visualisation messages.

Strategy ``decide_trades`` implementations write free-text reports into
``state.visualisation.messages``.  The helpers here extract typed values
back out so notebooks and charts can display them without duplicating
regex logic.
"""
import datetime
import re
from typing import Iterable

import pandas as pd

from tradeexecutor.state.state import State


def extract_usd_value(text: str, label: str) -> float:
    """Extract a ``label: 1,234.56 USD`` value from a cycle message.

    :return: The parsed float, or ``nan`` if the label is absent.
    """
    match = re.search(rf"{re.escape(label)}: ([0-9,]+(?:\.[0-9]+)?) USD", text)
    return float(match.group(1).replace(",", "")) if match else float("nan")


def extract_int_value(text: str, label: str) -> int:
    """Extract a ``label: 42`` integer from a cycle message.

    :return: The parsed int, or ``0`` if the label is absent.
    """
    match = re.search(rf"{re.escape(label)}: ([0-9]+)", text)
    return int(match.group(1)) if match else 0


def build_selection_diagnostics(state: State) -> pd.DataFrame:
    """Build a per-cycle DataFrame of signal selection counts.

    Parses ``Open/about to open positions``, ``Candidate signals created``,
    and ``Selected survivor signals`` from each cycle message.

    :return:
        DataFrame indexed by timestamp with columns: open_positions,
        candidate_signals_created, selected_survivor_signals.
    """
    rows = []
    for unix_ts, messages in sorted(state.visualisation.messages.items(), key=lambda item: item[0]):
        if not messages:
            continue
        message = "\n".join(messages)
        rows.append({
            "timestamp": datetime.datetime.utcfromtimestamp(unix_ts),
            "open_positions": extract_int_value(message, "Open/about to open positions"),
            "candidate_signals_created": extract_int_value(message, "Candidate signals created"),
            "selected_survivor_signals": extract_int_value(message, "Selected survivor signals"),
        })
    return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame(
        columns=["open_positions", "candidate_signals_created", "selected_survivor_signals"]
    )
