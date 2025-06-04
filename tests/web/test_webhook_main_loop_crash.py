"""Webhook stays alive after the trade execution loop crashes."""
import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
import logging
from unittest.mock import patch

import flaky
import pytest
import requests
from eth_defi.utils import find_free_port
from hexbytes import HexBytes
from typer.main import get_command

from tradeexecutor.cli.main import app


logger = logging.getLogger(__name__)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "crash.py"))


@pytest.fixture()
def hot_wallet_private_key() -> HexBytes:
    """Generate a private key.

    Does not need to have balance.
    """
    return HexBytes(secrets.token_bytes(32))


@pytest.mark.slow_test_group
@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None, reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_main_loop_catch(
    strategy_path,
    hot_wallet_private_key
):
    """Same as below, but run inline and print out the traceback.

    Allows to debug the code without launching a hard-to-debug subprocess.
    """

    # Set up the configuration for the live trader
    env = {
        "STRATEGY_FILE": strategy_path.as_posix(),  # Pass crash test
        "PRIVATE_KEY": hot_wallet_private_key.hex(),
        "HTTP_ENABLED": "false",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "CACHE_PATH": "/tmp/main_loop_tests",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "HTTP_ENABLED": "false",
        "MIN_GAS_BALANCE": "0",
        "TRADE_IMMEDIATELY": "true",
        "JSON_RPC_BINANCE": os.environ["BNB_CHAIN_JSON_RPC"],
        "PATH": os.environ["PATH"],
        "HTTP_WAIT_GOOD_STARTUP_SECONDS": "0",
        "MAX_DATA_DELAY_MINUTES": str(10*60*24*365),  # 10 years or "disabled""
        "LOG_LEVEL": "disabled",
        "UNIT_TESTING": "true",
        "RUN_SINGLE_CYCLE": "true",  # Run only one cycle to crash immediately
        "SKIP_CRASH_SLEEP": "true",  # Skip the sleep in the crash strategy
    }

    cli = get_command(app)
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(Exception) as e:
            cli.main(args=["start"], standalone_mode=False)
        assert str(e.value) == "Boom", f"The received main loop exception was : {e}"


# Disabled on Github CI as flaky
@pytest.mark.skipif(os.environ.get("BNB_CHAIN_JSON_RPC") is None or os.environ.get("CI") == "true", reason="Set BNB_CHAIN_JSON_RPC environment variable to Binance Smart Chain node to run this test")
def test_main_loop_traceback_over_web(
    strategy_path,
    hot_wallet_private_key
):
    """Check that we can capture main loop crashes correctly.

    - Webhook server tells main loop is alive

    - Webhook server tells the  main loop has crashed
    """

    port = find_free_port(20_000, 40_000, 20)
    server = f"http://localhost:{port}"

    # Set up the configuration for the live trader
    env = {
        "STRATEGY_FILE": strategy_path.as_posix(),  # Pass crash test
        "PRIVATE_KEY": hot_wallet_private_key.hex(),
        "HTTP_ENABLED": "false",
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "CACHE_PATH": "/tmp/main_loop_tests",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "HTTP_PORT": f"{port}",
        "HTTP_ENABLED": "true",
        "MIN_GAS_BALANCE": "0",
        "TRADE_IMMEDIATELY": "true",
        "JSON_RPC_BINANCE": os.environ["BNB_CHAIN_JSON_RPC"],
        "PATH": os.environ["PATH"],
        "HTTP_WAIT_GOOD_STARTUP_SECONDS": "0",
        "MAX_DATA_DELAY_MINUTES": str(10*60*24*365),  # 10 years or "disabled"
        "UNIT_TESTING": "true",
    }

    proc = subprocess.Popen(
        [
            "trade-executor",
            "start"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        # First check the server comes up
        deadline = time.time() + 25  # TODO: Depends on Github CI?
        last_exception = None
        server_up = False
        resp = None
        while time.time() < deadline:
            try:
                resp = requests.get(f"{server}/status", timeout=0.1)
                logger.info("Checking server: %s", resp.status_code)
            except Exception as e:
                last_exception = e
                time.sleep(0.5)
                continue

            assert resp.status_code == 200, f"Got server reply: {resp.content}"
            server_up = True
            break

        logger.info("The web server is up")

        if not server_up:
            proc.terminate()
            proc.wait()
            pipes = proc.communicate()
            output = pipes[0].decode("utf-8")
            output += pipes[1].decode("utf-8")
            print("trade-executor output:")
            print(output)
            raise AssertionError(f"Webhook server did not come up, last exception was: {last_exception}")

        # Check that the first reply from /status webhook
        # is that loop is running (because it did not have time to crash yet)
        data = resp.json()
        assert "last_refreshed_at" in data
        assert data["completed_cycle"] is None
        assert data["exception"] is None
        assert data["executor_running"] is True

        logger.info("Waiting for the crash")

        # Now wait until the main loop crashes
        deadline = time.time() + 30
        got_right_exception = False
        while time.time() < deadline:
            try:
                resp = requests.get(f"{server}/status", timeout=1.0)
            except requests.exceptions.ReadTimeout:
                time.sleep(0.1)
                continue
            except requests.exceptions.ConnectionError:
                break

            assert resp.status_code == 200, f"Got server reply: {resp.content}"

            data = resp.json()

            exception = data.get("exception")
            if exception is not None:
                # Execution loop has crashed, webhook is serving exception status
                assert data["executor_running"] == False
                assert data["exception"]["exception_message"] == "Boom"
                got_right_exception = True
                break

            time.sleep(0.5)

        logger.info("Crash loop exited, got the right exception %s", got_right_exception)

        if not got_right_exception:
            # Execution loop did not crash
            logger.info("Did not get expected error")
            proc.terminate()
            proc.wait()
            pipes = proc.communicate()
            output = pipes[0].decode("utf-8")
            output += pipes[1].decode("utf-8")
            print("trade-executor output:")
            print(output)
            raise AssertionError(f"Execution loop did not crash when crash was expected, last reply: {data}")

    finally:
        logger.info("Terminating")
        proc.terminate()
        return_code = proc.wait()

        logger.info("Done, return code is %d", return_code)

        if proc.returncode != 0:
            logger.info("Reading output")
            pipes = proc.communicate()
            logger.info("Has pipes")
            output = pipes[0].decode("utf-8")
            output += pipes[1].decode("utf-8")
            # logger.info(f"trade-executor exited {proc.returncode}, output:")
            # logger.info(output)

        logger.info("Finished")
