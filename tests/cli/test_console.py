import os
import secrets

import pytest

from tradeexecutor.cli.commands.app import app

pytestmark = pytest.mark.skipifif(os.environ.get("JSON_RPC_ETHEREUM") is None, reason="Set JSON_RPC_ETHEREUM environment variable torun this test")


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") == "true", reason="This test seems to block Github CI for some reason")
def test_cli_console_2(
    logger,
    persistent_test_cache_path: str,
    mocker,
):
    """Check console CLI opens with a new style strategy with a complex universe and indicator setup."""

    strategy_path = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "ethereum-memecoin-vol-basket.py")

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": persistent_test_cache_path,
        "JSON_RPC_ETHEREUM": os.environ.get("JSON_RPC_ETHEREUM"),
        "PRIVATE_KEY": "0x" + secrets.token_hex(32),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    app(["console"], standalone_mode=False)
