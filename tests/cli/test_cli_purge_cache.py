"""Command token-cache unit testst"""
import os.path
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


from tradeexecutor.cli.main import app

def test_cli_purge_token_cache(mocker, persistent_test_client):
    """token-cache command purging missing entries."""

    client = persistent_test_client

    environment = {
        "CACHE_PATH": client.transport.get_abs_cache_path(),
        "PURGE_TYPE": "missing_tokensniffer_data",
        "UNIT_TESTING": "true",
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    f = StringIO()
    with redirect_stdout(f):
        app(["token-cache"], standalone_mode=False)

    print(f.getvalue())

    import ipdb ; ipdb.set_trace()

    assert "Open positions" in f.getvalue()
    assert "No frozen positions" in f.getvalue()
    assert "Transactions by trade" in f.getvalue()

