import logging
import os
from pathlib import Path

import pytest

from tradeexecutor.cli.log import setup_logging, setup_file_logging
from tradeexecutor.webhook.http_log import http_logger


@pytest.mark.skipif(os.environ.get("CI") is not None, reason="This test messes output on Github Actions.")
def test_http_log(tmpdir):
    """Inspect HTTP logging issues."""

    path = Path(tmpdir)

    root_logger = setup_logging()
    assert root_logger is not None

    file_path = path / "main.log"
    http_path = file_path.with_suffix(".http.log")

    setup_file_logging(file_path, http_logging=True)

    root_logger.warning("Test")
    http_logger.warning("HTTP warning")

    print("Writing to ", path)

    assert http_path.exists()

    content = open(http_path, "rt").read()
    assert "root" not in content
    assert "HTTP warning" in content

    root_logger.setLevel(logging.WARNING)  # Do not leak logging in parallel tests





