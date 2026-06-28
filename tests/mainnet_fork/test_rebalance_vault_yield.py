"""Test rebalance vault yield and get funds unstuck from Aave bug.
"""
import shutil
import os.path
import secrets
from pathlib import Path
from unittest import mock

import pytest
from pytest import FixtureRequest
from pytest_mock import MockerFixture

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.main import app
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.hex import hexbytes_to_hex_str
from tradingstrategy.client import Client


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_BASE") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def anvil(request: FixtureRequest) -> AnvilLaunch:
    """Do Ethereum mainnet fork from the damaged situation."""

    mainnet_rpc = os.environ["JSON_RPC_BASE"]

    anvil = launch_anvil(
        mainnet_rpc,
        fork_block_number=32092657,  # The timestamp on when the broken position was created
    )

    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    """Make a copy of the state file with the broken vault on a new test cycle"""
    template = Path(__file__).resolve().parent / "yield-manager-aave-bug.json"
    assert template.exists(), f"State dump missing: {template}"
    p = tmp_path / Path("vault.json")
    shutil.copy(template, p)
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" /  "test_only" / "base-ath-ipor-aave-bug-fast.py"
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    state_file: Path,
    strategy_file: Path,
    persistent_test_client: Client,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(secrets.token_bytes(32)),
        "JSON_RPC_BASE": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        # "LOG_LEVEL": "info",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_ADDRESS": "0x7d8Fab3E65e6C81ea2a940c050A7c70195d1504f",
        "VAULT_ADAPTER_ADDRESS": "0x6DCCA7f34EB8F1a519ae690E9A3101f705bB0393",
        "SKIP_SAVE": "true",
        "AUTO_APPROVE": "true",  # skip y/n prompt
        "CACHE_PATH": str(persistent_test_client.transport.cache_path),  # Use unit test cache
        "RAISE_ON_UNCLEAN": "true",
        "RUN_SINGLE_CYCLE": "true",  # Run only one cycle"
        "MIN_GAS_BALANCE": "0.0",   # Disable gas balance check
        "DISABLE_BROADCAST": "true",  # Disable wait_and_broadcast_multiple_nodes() broadcast as we do not have real private key
        # This test exercises rebalance transaction generation, not the live
        # data freshness alert.
        "MAX_DATA_DELAY_MINUTES": str(3 * 24 * 60),
    }
    return environment


@pytest.mark.slow_test_group
def test_rebalance_vault_yield(
    environment: dict,
    mocker: MockerFixture,
):
    """Run one cycle and generate a rebalance transaction for the vault yield.

    This mainnet-fork regression test verifies that the damaged vault-yield
    state can generate the next rebalance transactions without crashing.

    1. Patch the process environment with the forked RPC, copied state file and
       deterministic test settings.
    2. Disable remote market-data freshness checks.
    3. Run one `trade-executor start` cycle.
    4. Confirm the command completes, proving money can be withdrawn from Aave
       and redeployed to the vaults.
    """

    # 1. Patch the process environment with the forked RPC, copied state file and
    # deterministic test settings.
    mocker.patch.dict("os.environ", environment, clear=True)

    # 2. Disable remote market-data freshness checks.
    mocker.patch.object(TradingStrategyUniverseModel, "check_data_age", return_value=None)

    # 3. Run one `trade-executor start` cycle.
    # 4. Confirm the command completes, proving money can be withdrawn from Aave
    # and redeployed to the vaults.
    app(["start"], standalone_mode=False)
