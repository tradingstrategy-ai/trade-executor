"""Smoke-test the Hyperliquid waterfall release-candidate notebook.

1. Copy the checked-in notebook to a temporary path so in-place execution does not touch the repo copy.
2. Execute the notebook with ``jupyter execute`` and keep stdout/stderr for crash diagnostics.
3. On failure, extract the crashing cell, exception, traceback, and source excerpt from the executed notebook.
"""

import json
import shutil
import subprocess
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "strategies" / "test_only" / "158-backtest-hyperliquid-waterfall-release-candidate.ipynb"


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


def test_hyperliquid_waterfall_release_candidate_notebook(tmp_path: Path) -> None:
    """Verify the release-candidate notebook executes and reports crashes clearly.

    1. Copy the release-candidate notebook to a temporary location for in-place execution.
    2. Run the notebook via ``poetry run jupyter execute`` and fail with full crash diagnostics if it errors.
    3. Confirm the executed notebook contains outputs so the smoke test proves a real run happened.
    """
    notebook_path = tmp_path / NOTEBOOK_PATH.name

    # 1. Copy the release-candidate notebook to a temporary location for in-place execution.
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
            "--timeout=900",
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode != 0:
        notebook_failure_details = _extract_notebook_failure_details(notebook_path)
        raise AssertionError(
            "jupyter execute failed for the Hyperliquid waterfall release-candidate notebook.\n"
            f"Return code: {result.returncode}\n"
            f"Stdout:\n{result.stdout}\n"
            f"Stderr:\n{result.stderr}\n"
            f"{notebook_failure_details}"
        )

    # 3. Confirm the executed notebook contains outputs so the smoke test proves a real run happened.
    assert _has_any_execution_output(notebook_path), "Expected the executed notebook to contain cell outputs"
