"""Test GMX Lagoon vault lifecycle in simulated (Anvil fork) mode.

Wraps ``scripts/lagoon/manual-trade-executor-gmx.py`` to run the full
GMX lifecycle using an Anvil fork:

1. Deploy vault with GMX whitelisting (all markets)
2. Deposit USDC, settle
3. Run strategy cycle (creates exchange account position)
4. Verify vault deployment and exchange account position creation
5. Settle and redeem

GMX trading steps are skipped in simulated mode because keepers
cannot execute on an Anvil fork.

Requires ``JSON_RPC_ARBITRUM`` environment variable pointing at a
real Arbitrum mainnet RPC (used as Anvil fork source).
"""

import importlib.util
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM,
    reason="JSON_RPC_ARBITRUM environment variable required",
)


def _load_script_module():
    """Load the manual test script as a Python module."""
    script_path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "scripts" / "lagoon" / "manual-trade-executor-gmx.py"
    )
    spec = importlib.util.spec_from_file_location("gmx_test_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def strategy_file() -> Path:
    path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "strategies" / "test_only" / "minimal_gmx_strategy.py"
    )
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.fixture()
def simulation_setup():
    """Set up Anvil fork, deployer, and USDC using the script's helpers.

    Yields all objects needed for the test lifecycle,
    then tears down Anvil fork.
    """
    usdc_amount = Decimal("100")
    mod = _load_script_module()

    (
        web3, deployer, private_key,
        json_rpc_arbitrum, anvil_launches,
    ) = mod.setup_simulation(
        json_rpc_arbitrum=JSON_RPC_ARBITRUM,
        simulate=True,
        private_key=None,
        usdc_amount=usdc_amount,
    )

    usdc_token = mod.verify_deployer_balances(
        web3=web3,
        deployer=deployer,
        usdc_amount=usdc_amount,
    )

    try:
        yield {
            "mod": mod,
            "web3": web3,
            "deployer": deployer,
            "private_key": private_key,
            "json_rpc_arbitrum": json_rpc_arbitrum,
            "usdc_token": usdc_token,
            "usdc_amount": usdc_amount,
        }
    finally:
        for launch in anvil_launches:
            launch.close(log_level=logging.ERROR)


@pytest.mark.timeout(600)
def test_gmx_vault_lifecycle(simulation_setup, strategy_file):
    """Full GMX vault lifecycle: deploy, exchange account, verify."""
    s = simulation_setup
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    s["mod"]._run_test_lifecycle(
        simulate=True,
        web3=s["web3"],
        deployer=s["deployer"],
        usdc_token=s["usdc_token"],
        private_key=s["private_key"],
        json_rpc_arbitrum=s["json_rpc_arbitrum"],
        strategy_file=strategy_file,
        usdc_amount=s["usdc_amount"],
        trading_strategy_api_key=trading_strategy_api_key,
    )
