"""Smoke-test the phase-aware capped-waterfall CCTP vault-universe backtest notebook.

``09-backtest-capped-waterfall-phase-aware.ipynb`` derives from the ``08`` base by swapping the
base ``AlphaModel`` for ``PhaseAwareAlphaModel``, adding a hub-chain synchronous USDC vault as a
``YieldManager`` queue venue, and modelling closed deposit/redemption windows for D2 (HYPE++) and
Ostium. The notebook self-verifies its acceptance criteria (Ostium + D2 present, daily rebalance,
park -> deposit-on-open events in the durable log, idle cash routed to the yield venue) in a
dedicated acceptance cell, so a criteria regression surfaces here as a failing notebook cell.

1. Copy the checked-in notebook to a temporary path so in-place execution does not touch the repo copy.
2. Execute the notebook with ``jupyter execute`` and keep stdout/stderr for crash diagnostics.
3. On failure, extract the crashing cell, exception, traceback, and source excerpt from the executed notebook.

This runs a full real-data cross-chain backtest and is correspondingly slow (tens of minutes); it
requires ``TRADING_STRATEGY_API_KEY`` in the environment (sourced from ``.local-test.env``).
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "strategies" / "test_only" / "09-backtest-capped-waterfall-phase-aware.ipynb"


def _source_excerpt(cell: dict, max_lines: int = 20) -> str:
    """Build a short source excerpt for a notebook cell."""
    source = "".join(cell.get("source", []))
    lines = source.splitlines()
    excerpt = "\n".join(lines[:max_lines]).strip()
    if len(lines) > max_lines:
        excerpt += "\n..."
    return excerpt or "<empty cell source>"


def _extract_notebook_failure_details(notebook_path: Path) -> str:
    """Extract the first notebook error output in an easy-to-fix format."""
    if not notebook_path.exists():
        return "Executed notebook was not created."

    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    for index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "error":
                continue

            exception_name = output.get("ename", "<unknown exception>")
            exception_value = output.get("evalue", "<no exception message>")
            traceback = "\n".join(output.get("traceback", [])) or "<no traceback captured>"
            source_excerpt = _source_excerpt(cell)
            return (
                f"Failing cell: {index}\n"
                f"Exception: {exception_name}: {exception_value}\n"
                f"Traceback:\n{traceback}\n"
                f"Cell source excerpt:\n{source_excerpt}"
            )

    return "Notebook execution failed, but no cell error output was captured in the executed notebook."


def _has_any_execution_output(notebook_path: Path) -> bool:
    """Check whether the executed notebook contains cell outputs."""
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    return any(cell.get("outputs") for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")


@pytest.mark.slow_test_group
def test_capped_waterfall_phase_aware_notebook(tmp_path: Path) -> None:
    """Verify the phase-aware capped-waterfall backtest notebook executes and self-verifies.

    1. Copy the notebook to a temporary location for in-place execution.
    2. Run the notebook via ``poetry run jupyter execute`` and fail with full crash diagnostics if it errors
       (its acceptance cell asserts the phase-aware criteria, so a regression there fails this test).
    3. Confirm the executed notebook contains outputs so the smoke test proves a real run happened.
    """
    notebook_path = tmp_path / NOTEBOOK_PATH.name

    # 1. Copy the notebook to a temporary location for in-place execution.
    shutil.copy(NOTEBOOK_PATH, notebook_path)

    # 2. Run the notebook via ``poetry run jupyter execute`` and fail with full crash diagnostics if it errors.
    result = subprocess.run(
        [
            "poetry",
            "run",
            "jupyter",
            "execute",
            str(notebook_path),
            "--inplace",
            "--timeout=3600",
        ],
        capture_output=True,
        text=True,
        timeout=5400,
    )

    if result.returncode != 0:
        notebook_failure_details = _extract_notebook_failure_details(notebook_path)
        raise AssertionError(
            "jupyter execute failed for the phase-aware capped-waterfall backtest notebook.\n"
            f"Return code: {result.returncode}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}\n"
            f"{notebook_failure_details}"
        )

    # 3. Confirm the executed notebook contains outputs so the smoke test proves a real run happened.
    assert _has_any_execution_output(notebook_path), "Expected the executed notebook to contain cell outputs"
