"""Integration test: verify jupyter execute works with notebook ID detection and multiprocess support."""
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
    """Run a notebook via jupyter execute and check it completes without errors."""
    result = subprocess.run(
        ["poetry", "run", "jupyter", "execute", str(notebook_path), "--inplace", "--timeout=60"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"jupyter execute failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
