"""Tests for HyperCore closed-entry audit tables."""

from types import SimpleNamespace

import pandas as pd

from tradeexecutor.analysis.hypercore_closed_entries import (
    analyse_hypercore_closed_entry_events,
    analyse_hypercore_closed_entry_summary,
)


class _Pair:
    def __init__(self, address: str):
        self.pool_address = address

    def is_hyperliquid_vault(self) -> bool:
        return True


class _Universe:
    def __init__(self):
        self.vault_state = pd.DataFrame(
            [
                {"pair_id": 1, "timestamp": pd.Timestamp("2026-04-11"), "deposits_open": False},
                {"pair_id": 1, "timestamp": pd.Timestamp("2026-04-13"), "deposits_open": True},
                {"pair_id": 1, "timestamp": pd.Timestamp("2026-04-15"), "deposits_open": False},
            ]
        )

    def get_pair_by_id(self, pair_id: int):
        return _Pair(f"0x{pair_id:040x}")


def _state():
    return SimpleNamespace(
        visualisation=SimpleNamespace(
            calculations={
                1775779200: {
                    "hypercore_closed_entry_events": [
                        {
                            "timestamp": "2026-04-11T00:00:00",
                            "status": "skipped",
                            "reason_code": "vault_deposits_closed",
                            "vault_name": "Closed One",
                            "pair_ticker": "CLOSED-USDC",
                            "vault_address": "0x0000000000000000000000000000000000000001",
                        },
                        {
                            "timestamp": "2026-04-11T00:00:00",
                            "status": "accepted",
                            "reason_code": None,
                            "vault_name": "Closed One",
                            "pair_ticker": "CLOSED-USDC",
                            "vault_address": "0x0000000000000000000000000000000000000001",
                        },
                    ]
                },
                1775952000: {
                    "hypercore_closed_entry_events": [
                        {
                            "timestamp": "2026-04-13T00:00:00",
                            "status": "skipped",
                            "reason_code": "vault_deposits_closed",
                            "vault_name": "Closed One",
                            "pair_ticker": "CLOSED-USDC",
                            "vault_address": "0x0000000000000000000000000000000000000001",
                        },
                    ]
                },
            }
        )
    )


def test_hypercore_closed_entry_tables():
    entries = analyse_hypercore_closed_entry_events(_state())
    assert list(entries.columns) == [
        "Vault name",
        "Skipped entries",
        "Accepted entries",
        "First skip",
        "Last skipped",
    ]
    assert entries.iloc[0]["Vault name"] == "Closed One"
    assert entries.iloc[0]["Skipped entries"] == 2
    assert entries.iloc[0]["Accepted entries"] == 1
    assert entries.iloc[0]["First skip"] == pd.Timestamp("2026-04-11")
    assert entries.iloc[0]["Last skipped"] == pd.Timestamp("2026-04-13")

    summary = analyse_hypercore_closed_entry_summary(_state(), _Universe())
    assert summary.set_index("Metric").loc["Proper closed data starting date", "Value"] == "2026-04-11"
    assert summary.set_index("Metric").loc["Total closed vaults", "Value"] == 1
    assert summary.set_index("Metric").loc["Total close periods", "Value"] == 2
    assert summary.set_index("Metric").loc["Total missed entries", "Value"] == 2
