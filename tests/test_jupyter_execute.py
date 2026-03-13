"""Integration test: verify jupyter execute works with notebook ID detection and multiprocess support."""
import json
import subprocess
import shutil

import pytest


@pytest.fixture
def notebook_path(tmp_path):
    """Copy the test notebook to a temp dir so --inplace doesn't modify the repo."""
    src = "tests/test_jupyter_execute.ipynb"
    dst = tmp_path / "test_jupyter_execute.ipynb"
    shutil.copy(src, dst)
    return dst


def test_jupyter_execute_notebook(notebook_path):
    """Run a notebook via jupyter execute and verify notebook ID is resolved correctly."""
    result = subprocess.run(
        ["poetry", "run", "jupyter", "execute", str(notebook_path), "--inplace", "--timeout=60"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"jupyter execute failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    # Read the executed notebook and check cell outputs
    with open(notebook_path) as f:
        nb = json.load(f)

    def get_cell_stdout(cell):
        """Extract stdout text from a notebook cell's outputs."""
        parts = []
        for o in cell.get("outputs", []):
            if o["output_type"] == "stream":
                text = o.get("text", [])
                parts.extend(text if isinstance(text, list) else [text])
        return "".join(parts)

    # Cell 0: get_notebook_id() check
    cell0_text = get_cell_stdout(nb["cells"][0])
    assert "OK: notebook_id=test_jupyter_execute" in cell0_text, f"Notebook ID not resolved correctly. Cell output: {cell0_text}"

    # Cell 1: is_running_in_ipython() check
    cell1_text = get_cell_stdout(nb["cells"][1])
    assert "is_running_in_ipython=False" in cell1_text, f"IPython detection wrong. Cell output: {cell1_text}"
