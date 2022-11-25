"""Test additional main CLI commands."""

import os


import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.main import app


@pytest.fixture(scope="session")
def strategy_path():
    """We use pancake-eth-usd-sma.py as the strategy module input for these tests."""
    return os.path.join(os.path.dirname(__file__), "../strategies", "pancake-eth-usd-sma.py")


@pytest.fixture(scope="session")
def unit_test_cache_path():
    """Where unit tests  cache files.

    We have special path for CLI tests to make sure CLI tests
    always do fresh downloads.
    """
    path = os.path.join(os.path.dirname(__file__), "/tmp/cli_tests")
    os.makedirs(path, exist_ok=True)
    return path


def test_cli_check_universe(
        strategy_path: str,
        unit_test_cache_path: str,
    ):
    """check-universe command works"""

    environment = {
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "STRATEGY_FILE": strategy_path,
        "CACHE_PATH": unit_test_cache_path,
    }

    runner = CliRunner()
    result = runner.invoke(app, "check-universe", env=environment)

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_cli_check_wallet(
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
        "MINUMUM_GAS_BALANCE": "0",
    }

    runner = CliRunner()
    result = runner.invoke(app, "check-wallet", env=environment)

    if result.exception:
        raise result.exception

    # Dump any stdout to see why the command failed
    if result.exit_code != 0:
        print("runner failed")
        for line in result.stdout.split('\n'):
            print(line)
        raise AssertionError("runner launch failed")


def test_cli_backtest(
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
        "EXECUTION_TYPE": "backtest"
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
        "EXECUTION_TYPE": "uniswap_v2_hot_wallet",
        "JSON_RPC_BINANCE": os.environ.get("BNB_CHAIN_JSON_RPC"),
        # Random empty wallet
        "PRIVATE_KEY": "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83",
        "HTTP_ENABLED": "true",
        "GAS_PRICE_METHOD": "london",

        # Make the applicaction terminate after the setup
        "MAX_CYCLES": "0",
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
