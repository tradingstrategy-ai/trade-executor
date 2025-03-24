"""Command token-cache unit testst"""
import os.path
from contextlib import redirect_stdout
from io import StringIO



from tradeexecutor.cli.main import app

def test_cli_purge_token_cache(mocker, persistent_test_client):
    """token-cache command purging missing entries."""

    client = persistent_test_client

    environment = {
        "CACHE_PATH": client.transport.get_abs_cache_path(),
        "PURGE_TYPE": "missing_tokensniffer_data",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    f = StringIO()
    with redirect_stdout(f):
        app(["token-cache"], standalone_mode=False)

    assert "count" in f.getvalue()


