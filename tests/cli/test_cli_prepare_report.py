"""Test prepare-report CLI command."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.main import get_command

from tradeexecutor.cli.main import app


def test_cli_prepare_report(tmp_path):
    """prepare-report command injects iframe CSS/JS into an external HTML file."""

    input_html = tmp_path / "external-report.html"
    input_html.write_text("<html><head><title>Test</title></head><body><h1>Report</h1></body></html>")

    environment = {
        "EXECUTOR_ID": "test-strategy",
        "LOG_LEVEL": "disabled",
    }

    cli = get_command(app)

    # Run from tmp_path so state/ is created there
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with patch.dict("os.environ", environment, clear=True):
            with pytest.raises(SystemExit) as e:
                cli.main(args=["prepare-report", input_html.as_posix()])
            assert e.value.code == 0, f"Exit code: {e}"

        output_html = tmp_path / "state" / "test-strategy-backtest.html"
        assert output_html.exists(), f"Output file not created at {output_html}"

        html = output_html.read_text()
        assert 'id="trade-executor-css-inject"' in html
        assert 'id="trade-executor-js-inject"' in html
        assert "Dynamic iframe resizer loaded" in html
        assert "<h1>Report</h1>" in html
    finally:
        os.chdir(old_cwd)
