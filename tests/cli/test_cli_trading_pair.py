import os
from pathlib import Path

import pytest

from tradeexecutor.cli.main import app


@pytest.fixture(scope="session")
def unit_test_cache_path(persistent_test_cache_path):
    """Where unit tests  cache files.

    We have special path for CLI tests to make sure CLI tests
    always do fresh downloads.
    """
    return persistent_test_cache_path


@pytest.mark.slow_test_group
def test_cli_trading_pair(
    unit_test_cache_path: str,
    mocker,
    tmp_path,
):
    """trading-pair command works"""

    strategy_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "base-ath.py"))

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path.as_posix(),
        "CACHE_PATH": tmp_path.as_posix(),
        "LOG_LEVEL": "info",
        "UNIT_TESTING": "true",
        "MAX_DATA_DELAY_MINUTES": "99999",
        "TOKEN_ADDRESS": "0x6797b6244fa75f2e78cdffc3a4eb169332b730cc",  # EAI
    }

    mocker.patch.dict("os.environ", environment, clear=True)
    app(["trading-pair"], standalone_mode=False)


