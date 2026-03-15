"""Shared helpers for vault position analytics and charts."""

from __future__ import annotations

from tradeexecutor.state.state import State


def get_vault_positions(state: State):
    """Return all vault positions from the portfolio."""
    return [p for p in state.portfolio.get_all_positions() if p.is_vault()]


def find_latest_position_for_pair(state: State, pair):
    """Find the latest portfolio position matching the given vault pair."""
    for position in reversed(list(state.portfolio.get_all_positions())):
        if position.pair == pair:
            return position
    return None
