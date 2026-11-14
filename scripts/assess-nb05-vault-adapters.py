"""Create an offline NB05 adapter inventory from licensed local research inputs.

Example (use the installed parent Poetry environment in a worktree)::

    poetry run python scripts/assess-nb05-vault-adapters.py \
        --notebook /path/to/source.ipynb --metadata /path/to/vault-metadata.json \
        --state /path/to/notebook-backtest-state.json \
        --eth-defi-repo /path/to/web3-ethereum-defi

Only the requested output file is written. No notebook code, network request,
transaction or strategy execution takes place.
"""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import typer

from tradeexecutor.analysis.vault_adapter_coverage import (
    build_vault_adapter_coverage,
    select_nb05_sources,
    summarise_coverage,
)
from tradeexecutor.testing.vault_adapter_checks import probe_nb05_adapter_boundaries


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_REPORTS = (
    "cross-chain-vault-test-2026-07-31-corrective.report.json",
    "cross-chain-vault-test-2026-07-31-hype-amendment.report.json",
    "cross-chain-vault-test-2026-07-31-permission-defaults.report.json",
)


def _git(repository: Path, *arguments: str) -> str:
    """Read an explicit repository without altering its checkout."""
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _read_input(path: Path) -> tuple[dict, dict]:
    """Read and fingerprint an input without publishing its contents or local path."""
    content = path.read_bytes()
    return json.loads(content), {
        "filename": path.name,
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main(
    notebook: Path = typer.Option(..., exists=True, dir_okay=False),
    metadata: Path = typer.Option(..., exists=True, dir_okay=False),
    state: Path | None = typer.Option(None, exists=True, dir_okay=False),
    eth_defi_repo: Path = typer.Option(
        REPOSITORY_ROOT / "deps/web3-ethereum-defi", exists=True, file_okay=False
    ),
    eth_defi_ref: str | None = typer.Option(
        None,
        help="Factory revision; defaults to this executor's pinned submodule commit.",
    ),
    expected_sources: int = typer.Option(347, min=1),
    output: Path = typer.Option(
        REPOSITORY_ROOT / "docs/reports/nb05-live-adapter-coverage.json"
    ),
) -> None:
    """Inventory every source and retain the provenance of all evidence."""
    notebook_data, notebook_identity = _read_input(notebook)
    metadata_data, metadata_identity = _read_input(metadata)
    state_data, state_identity = (
        _read_input(state) if state is not None else (None, None)
    )
    source_keys = select_nb05_sources(notebook_data, metadata_data)
    if len(source_keys) != expected_sources:
        raise typer.BadParameter(
            f"Expected {expected_sources} sources, reconstructed {len(source_keys)}"
        )
    if output.resolve() in {
        path.resolve() for path in (notebook, metadata, state) if path is not None
    }:
        raise typer.BadParameter("The report must not overwrite an input")

    pinned = _git(
        REPOSITORY_ROOT, "ls-tree", "HEAD", "deps/web3-ethereum-defi"
    ).split()[2]
    revision = _git(eth_defi_repo, "rev-parse", eth_defi_ref or pinned)
    # git show reads the selected revision even if the supplied checkout differs.
    classification = subprocess.check_output(
        [
            "git",
            "-C",
            str(eth_defi_repo),
            "show",
            f"{revision}:eth_defi/erc_4626/classification.py",
        ],
        text=True,
    )
    historical = []
    historical_identities = []
    for name in HISTORICAL_REPORTS:
        report, identity = _read_input(REPOSITORY_ROOT / "docs/reports" / name)
        historical.append((name, report))
        historical_identities.append(identity)
    rows = build_vault_adapter_coverage(
        source_keys,
        metadata_data,
        classification,
        state=state_data,
        historical_reports=historical,
    )
    command_source = (
        REPOSITORY_ROOT / "tradeexecutor/cli/commands/perform_test_trade.py"
    )
    test_trade_calls = [
        node
        for node in ast.walk(ast.parse(command_source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_test_trade"
    ]
    probes = probe_nb05_adapter_boundaries()
    probes["perform_test_trade_persistence"] = {
        "make_test_trade_calls": len(test_trade_calls),
        "calls_passing_sync_state_callback": sum(
            any(keyword.arg == "sync_state_callback" for keyword in call.keywords)
            for call in test_trade_calls
        ),
        "basis": "static direct keyword arguments; no CLI command executed",
    }
    report = {
        "schema_version": 1,
        "scope": "offline assessment; no new deployment lifecycle certification",
        "provenance": {
            "source_url": "https://github.com/tradingstrategy-ai/vault-notebooks/pull/3",
            "notebook": notebook_identity,
            "metadata": metadata_identity,
            "metadata_generated_at": metadata_data.get("generated_at"),
            "saved_state": state_identity,
            "trade_executor_base_commit": _git(REPOSITORY_ROOT, "rev-parse", "HEAD"),
            "trade_executor_worktree_dirty": bool(
                _git(REPOSITORY_ROOT, "status", "--porcelain")
            ),
            "pinned_eth_defi_commit": pinned,
            "classification_eth_defi_commit": revision,
            "classification_sha256": hashlib.sha256(
                classification.encode()
            ).hexdigest(),
            "assessment_source_sha256": {
                name: hashlib.sha256((REPOSITORY_ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/assess-nb05-vault-adapters.py",
                    "tradeexecutor/analysis/vault_adapter_coverage.py",
                    "tradeexecutor/testing/vault_adapter_checks.py",
                    "tradeexecutor/curator/curator.py",
                    "tradeexecutor/ethereum/lagoon/universe_config.py",
                    "tradeexecutor/ethereum/vault/vault_routing.py",
                    "tradeexecutor/state/identifier.py",
                    "tradeexecutor/cli/commands/perform_test_trade.py",
                )
            },
            "historical_reports_in_amendment_order": historical_identities,
        },
        "summary": summarise_coverage(rows),
        "read_only_diagnostic_probes": probes,
        "vaults": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
