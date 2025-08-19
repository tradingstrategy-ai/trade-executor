"""Test correct credit position where a redemption broke the accounting.

- Test only works with archive node
"""

import os.path
import secrets
import shutil
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest
from _pytest.fixtures import FixtureRequest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


CI = os.environ.get("CI") == "true"

pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=60855854,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        #anvil.close(log_level=logging.INFO)
        anvil.close()


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Because we modifty state file when fixing it, we need to make a working copy from the master copy."""
    p = Path(os.path.join(os.path.dirname(__file__), "credit-position-broken-redemption.json"))
    assert p.exists(), f"{p} missing"
    working_copy = tmp_path / "test-copy.state.json"
    shutil.copyfile(p, working_copy)
    return working_copy


@pytest.fixture()
def strategy_file() -> Path:
    p = Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "enzyme-polygon-eth-breakout.py"))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    persistent_test_client,
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Set up the env for CLI commands in this unit test."""

    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": "0x" + secrets.token_bytes(32).hex(),  # Not needed
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0xe59b7affB1AAf4fB063c8476199BE118D5B9955F",
        "VAULT_ADAPTER_ADDRESS": "0x1abc5e398775249622a561b2C14E8aeB7fD8e361",
        "VAULT_PAYMENT_FORWARDER_ADDRESS": "0x73b662A52C57C83dab75f1a9C209b5cE8beaB353",
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": "57311900",
        "TEST_VAR": "xxxx",
        "SKIP_SAVE": "false",  # Need to save between runs
        "SKIP_INTEREST": "true",  # This must be enabled so that correct-accounts do not crash in early intrest distribution phase
        "CACHE_PATH": str(persistent_test_client.transport.cache_path),  # Use unit test cache
        "RAISE_ON_UNCLEAN": "true",  # This is needed to detect unclean state
    }
    return environment


@pytest.mark.skipif(CI, reason="Github CI/Anvil crap")
def test_correct_accounts_redemption_on_ausdc(
    environment: dict,
    state_file: Path,
):
    """Fix aUSDC redemption breaking accounts.

    - There was a redemption on aUSDC position

    - The redemption event was not correctly handled (gone missing)

    - We need to correct accounts

    - We need to reset the loan to start interest tracking from zero

    """

    # Accounting is detect to be incorrect
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["check-accounts"], standalone_mode=False)

    # Fix issued
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["correct-accounts"], standalone_mode=False)

    # Check tracekd interest collateral amount is fixed.
    # This address the issue if state is not correctly written after the correct accounts.
    # This is for Typer/Click bug when SKIP_SAVE environment variable is incorrectly parsed
    # always as true.
    state = State.read_json_file(state_file)
    position = state.portfolio.get_position_by_id(21)  # 'Trading position #21 for aPolUSDC-USDC.e'
    loan = position.loan
    collateral = loan.collateral
    assert collateral.quantity == pytest.approx(Decimal(500.65))

    # See that the interest distribution now works
    # and is triggered at the start of correct-accounts command
    environment = environment.copy()
    del environment["SKIP_INTEREST"]
    # environment["LOG_LEVEL"] = "info"

    with mock.patch.dict('os.environ', environment, clear=True):
        with pytest.raises(SystemExit) as sys_exit:
            app(["correct-accounts"], standalone_mode=True)
        assert sys_exit.value.code == 0
