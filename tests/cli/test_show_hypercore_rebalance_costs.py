"""CLI coverage for HyperCore rebalance cost reporting."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from tradeexecutor.analysis.hypercore_rebalance_cost import HypercoreRebalanceCostReport
from tradeexecutor.cli.commands import show_hypercore_rebalance_costs as cost_command
from tradeexecutor.cli.main import app


def test_show_hypercore_rebalance_costs_cli_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cost-report CLI renders its tables without changing the state file.

    1. Provide a minimal state and a completed report through command boundaries.
    2. Run the command with historical-price fetching disabled.
    3. Confirm all report sections render and the supplied state file is unchanged.
    """
    # 1. Provide a minimal state and a completed report through command boundaries.
    state_file = tmp_path / "hyper-ai.json"
    state_file.write_text("state remains untouched")
    state_digest = hashlib.sha256(state_file.read_bytes()).digest()
    state = SimpleNamespace(
        name="Hyper AI",
        portfolio=SimpleNamespace(get_all_trades=lambda: []),
    )
    report = HypercoreRebalanceCostReport(
        trades=pd.DataFrame(),
        rebalances=pd.DataFrame([{"cycle": "2026-07-01", "total_cost_usd": 1.0}]),
        summary=pd.DataFrame([{"eligible_rebalances": 1, "total_cost_usd": 1.0}]),
        ignored=pd.DataFrame([{"reason": "repair trade", "trades": 1}]),
    )
    # The command's report construction is mocked to isolate rendering and
    # read-only behaviour from trade serialisation, which is covered separately.
    monkeypatch.setattr(cost_command.State, "read_json_file", lambda _: state)
    monkeypatch.setattr(
        cost_command,
        "build_hypercore_rebalance_cost_report",
        lambda _, __: report,
    )

    # 2. Run the command with historical-price fetching disabled.
    result = CliRunner().invoke(
        app,
        [
            "show-hypercore-rebalance-costs",
            "--state-file",
            str(state_file),
            "--no-historical-prices",
        ],
    )

    # 3. Confirm all report sections render and the supplied state file is unchanged.
    if result.exception:
        raise result.exception
    assert result.exit_code == 0, result.stdout
    assert "HyperCore rebalance costs for Hyper AI" in result.stdout
    assert "Rebalances" in result.stdout
    assert "Strategy summary" in result.stdout
    assert "Ignored HyperCore history" in result.stdout
    assert hashlib.sha256(state_file.read_bytes()).digest() == state_digest
