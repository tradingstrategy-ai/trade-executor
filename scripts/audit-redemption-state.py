"""Audit blocked redemption diagnostics from a strategy state dump."""

import argparse
import datetime
import json
import sys
import urllib.request
from pathlib import Path

from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.analysis.redemption_audit import audit_redemption_state
from tradeexecutor.state.state import State


def _load_state(source: str) -> State:
    """Load state from a local JSON file or an HTTP endpoint."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return State.read_json_blob(json.dumps(payload))

    return State.read_json_file(Path(source))


def main() -> int:
    """Run the state audit and print a compact tabular report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Local state JSON path or HTTP(S) URL")
    parser.add_argument("--now", help="Naive UTC timestamp in ISO format", default=None)
    args = parser.parse_args()

    now = datetime.datetime.fromisoformat(args.now) if args.now else native_datetime_utc_now()
    state = _load_state(args.source)
    rows, mismatch_count = audit_redemption_state(state, now=now)

    print("pair_ticker\tvault_address\tstage\treason_code\trecorded_lockup_expired\tposition_lockup_expires_at\tuser_lockup_expires_at")
    for row in rows:
        print(
            "\t".join(
                [
                    row.pair_ticker or "",
                    row.vault_address or "",
                    row.stage or "",
                    row.reason_code or "",
                    "yes" if row.recorded_lockup_expired else "no",
                    row.position_recorded_lockup_expires_at.isoformat() if row.position_recorded_lockup_expires_at else "",
                    row.user_lockup_expires_at.isoformat() if row.user_lockup_expires_at else "",
                ]
            )
        )

    return 1 if mismatch_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
