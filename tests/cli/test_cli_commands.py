"""Test additional main CLI commands."""

import os
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
from pyasn1_modules.rfc6031 import id_pskc_deviceBinding
from typer.main import get_command
from typer.testing import CliRunner

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

pytestmark = pytest.mark.skipif(
    os.environ.get("TRADING_STRATEGY_API_KEY") is None or os.environ.get("BNB_CHAIN_JSON_RPC") is None,
    reason="Set TRADING_STRATEGY_API_KEY and BNB_CHAIN_JSON_RPC environment variable to run this test module"
)


@pytest.fixture(scope="session")
def strategy_path():
    """We use pancake-eth-usd-sma.py as the strategy module input for these tests."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "pancake-eth-usd-sma.py")


@pytest.fixture(scope="session")
def unit_test_cache_path():
    """Where unit tests  cache files.

    We have special path for CLI tests to make sure CLI tests
    always do fresh downloads.
    """
    path = os.path.join(os.path.dirname(__file__), "/tmp/cli_tests")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_cli_check_wallet(
    logger,
    strategy_path: str,
    unit_test_cache_path: str,
    ):
    """check-wallet command works"""

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "JSON_RPC_BINANCE": os.environ.get("BNB_CHAIN_JSON_RPC"),
        # Random empty wallet
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "MIN_GAS_BALANCE": "0",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    runner = CliRunner()
    result = runner.invoke(app, "check-wallet", env=environment)

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("Runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


@pytest.mark.skip(reason="Feature disabled")
def test_cli_legacy_backtest_no_wrap(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """start backtest command works.

    Don't use Typer CLI wrapper, because it prevents using debuggeres.
    """

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "BACKTEST_CANDLE_TIME_FRAME_OVERRIDE": "1d",
        "BACKTEST_STOP_LOSS_TIME_FRAME_OVERRIDE": "1d",
        "BACKTEST_START": "2021-06-01",
        "BACKTEST_END": "2022-07-01",
        "ASSET_MANAGEMENT_MODE": "backtest",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])
        assert e.value.code == 0


def test_cli_legacy_backtest(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """start backtest command works"""

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "BACKTEST_CANDLE_TIME_FRAME_OVERRIDE": "1d",
        "BACKTEST_STOP_LOSS_TIME_FRAME_OVERRIDE": "1d",
        "BACKTEST_START": "2021-06-01",
        "BACKTEST_END": "2022-07-01",
        "ASSET_MANAGEMENT_MODE": "backtest",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }

    runner = CliRunner()
    result = runner.invoke(app, "start", env=environment)

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_cli_live_trading(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """Test live trading command.

    Do not do actual live trading - just
    check that we get everything up and running to the point where the main loop would start.

    To run:

    .. code-block:: shell

        pytest --log-cli-level=info -s -k test_cli_live_trading

    """

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "CYCLE_DURATION": "1d",
        "STOP_LOSS_CHECK_FREQUENCY": "1d",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "JSON_RPC_BINANCE": os.environ.get("BNB_CHAIN_JSON_RPC"),
        # Random empty wallet
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "HTTP_ENABLED": "true",

        # Make the applicaction terminate after the setup
        "MAX_CYCLES": "0",

        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    runner = CliRunner()

    try:
        result = runner.invoke(app, "start", env=environment)
    except OSError:
        # This cannot be caught, but pytest always
        # nags about some assync open file descriptor somewhere
        #
        #
        #   OSError: [Errno 9] Bad file descriptor
        # File "/opt/homebrew/Cellar/python@3.10/3.10.8/Frameworks/Python.framework/Versions/3.10/lib/python3.10/threading.py", line 953, in run
        #    self._target(*self._args, **self._kwargs)
        #  File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/webtest/http.py", line 87, in run
        #    self.asyncore.loop(.5, map=self._map)
        #  File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/waitress/wasyncore.py", line 245, in loop
        #    poll_fun(timeout, map)
        #  File "/Users/moo/Library/Caches/pypoetry/virtualenvs/trade-executor-8Oz1GdY1-py3.10/lib/python3.10/site-packages/waitress/wasyncore.py", line 172, in poll
        pass

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


def test_cli_version(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """version command works"""

    runner = CliRunner()
    result = runner.invoke(app, "version")
    assert result.exit_code == 0


@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_cli_console(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """console command works"""

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "JSON_RPC_BINANCE": os.environ.get("BNB_CHAIN_JSON_RPC"),
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    runner = CliRunner()
    result = runner.invoke(app, "console", env=environment)

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


@pytest.mark.slow_test_group
def test_cli_backtest(
        logger,
        unit_test_cache_path: str,
    ):
    """backtest command works.

    - Run backtest command

    - Check for the resulting files that should have been generated

    .. note ::

        This test is somewhat slow due to high number of charts generated.
        But no way to speed it up.
    """

    strategy_path = os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "backtest-cli-command.py")

    backtest_result = os.path.join(tempfile.mkdtemp(), 'test_cli_backtest.json')
    notebook_result = os.path.join(tempfile.mkdtemp(), 'test_cli_backtest.ipynb')
    html_result = os.path.join(tempfile.mkdtemp(), 'test_cli_backtest.html')

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "HTML_REPORT":  html_result,
        "NOTEBOOK_REPORT":  notebook_result,
        "STATE_FILE":  backtest_result,
    }

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["backtest"])

        assert e.value.code == 0, f"Exit code: {e}"

    # Check generated state file is good
    state = State.read_json_file(Path(backtest_result))
    assert len(state.portfolio.closed_positions) > 0

    # Check generated HTML file is good
    html = Path(html_result).open("rt").read()
    assert "/* trade-executor backtest report generator custom CSS */" in html

    # Check generated notebook is good
    with open(notebook_result, "rt") as inp:
        nb = nbformat.read(inp, as_version=4)
        assert len(nb.cells) > 0


def test_cli_show_positions(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """show-positions command works.

    Run against an empty state file.
    """

    path = Path(tempfile.mkdtemp()) / "test-cli-show-positions-state.json"
    state = State()
    with path.open("wt") as out:
        out.write(state.to_json_safe())

    environment = {
        "STATE_FILE": path.as_posix(),
    }

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            f = StringIO()
            with redirect_stdout(f):
                cli.main(args=["show-positions"])
        assert e.value.code == 0


def test_cli_export(
        logger,
        strategy_path: str,
        unit_test_cache_path: str,
        capsys,
    ):
    """export command does not crash"""

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
        "JSON_RPC_BINANCE": os.environ.get("BNB_CHAIN_JSON_RPC"),
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
    }

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["export"])
        assert e.value.code == 0

    captured = capsys.readouterr()
    assert "export LOG_LEVEL" in captured.out
    assert "STRATEGY_FILE" in captured.out


def test_cli_send_log_message(
    logger,
    strategy_path: str,
    capsys,
):
    """send-log-message command does not crash"""

    environment = {
        "LOG_LEVEL": "disabled",
        "STRATEGY_FILE": strategy_path,
    }

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["send-log-message"])
        assert e.value.code == 0
