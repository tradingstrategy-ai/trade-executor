"""Command show-positions with long output"""
import os.path
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


from tradeexecutor.cli.main import app

def test_cli_show_positions_long(mocker):
    """show-positions command work with long output of various output.
    """
    path = Path(os.path.dirname(__file__)) / "show-positions-long.json"

    environment = {
        "STATE_FILE": path.as_posix(),
    }

    mocker.patch.dict("os.environ", environment, clear=True)

    f = StringIO()
    with redirect_stdout(f):
        app(["show-positions"], standalone_mode=False)

    assert "Open positions" in f.getvalue()
    assert "No frozen positions" in f.getvalue()


