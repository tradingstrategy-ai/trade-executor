"""Test cross-chain Lagoon vault lifecycle in simulated (Anvil fork) mode.

Wraps ``scripts/lagoon/manual-trade-executor-multichain.py`` to run the full
multichain lifecycle using Anvil forks:

1. Deploy multichain vault (Arb Sepolia + Base Sepolia forks)
2. Deposit USDC, settle
3. Bridge USDC via CCTP (Arb -> Base, forged attestation)
4. Optionally swap on Uniswap v3
5. Bridge USDC back via CCTP (Base -> Arb, forged attestation)
6. Verify total equity equals deposited amount across all chains

Requires ``JSON_RPC_ARBITRUM_SEPOLIA`` and ``JSON_RPC_BASE_SEPOLIA``
environment variables pointing at real testnet RPCs (used as Anvil
fork sources).
"""

import importlib.util
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
JSON_RPC_BASE_SEPOLIA = os.environ.get("JSON_RPC_BASE_SEPOLIA")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM_SEPOLIA or not JSON_RPC_BASE_SEPOLIA,
    reason="JSON_RPC_ARBITRUM_SEPOLIA and JSON_RPC_BASE_SEPOLIA environment variables required",
)


def _load_script_module():
    """Load the manual test script as a Python module."""
    script_path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "scripts" / "lagoon" / "manual-trade-executor-multichain.py"
    )
    spec = importlib.util.spec_from_file_location("crosschain_test_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def strategy_file() -> Path:
    path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "strategies" / "test_only" / "lagoon_crosschain_manual_test.py"
    )
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.fixture()
def simulation_setup():
    """Set up Anvil forks, deployer, and CCTP attesters using the script's helpers.

    Yields all objects needed for the test lifecycle,
    then tears down Anvil forks.
    """
    usdc_amount = Decimal("5")
    mod = _load_script_module()

    (
        arb_web3, base_web3, deployer, private_key,
        json_rpc_arb_sepolia, json_rpc_base_sepolia,
        test_attesters, anvil_launches,
    ) = mod.setup_simulation(
        json_rpc_arb_sepolia=JSON_RPC_ARBITRUM_SEPOLIA,
        json_rpc_base_sepolia=JSON_RPC_BASE_SEPOLIA,
        simulate=True,
        private_key=None,
        usdc_amount=usdc_amount,
    )

    arb_usdc = mod.verify_deployer_balances(
        arb_web3=arb_web3,
        base_web3=base_web3,
        deployer=deployer,
        usdc_amount=usdc_amount,
    )

    try:
        yield {
            "mod": mod,
            "arb_web3": arb_web3,
            "base_web3": base_web3,
            "deployer": deployer,
            "private_key": private_key,
            "json_rpc_arb_sepolia": json_rpc_arb_sepolia,
            "json_rpc_base_sepolia": json_rpc_base_sepolia,
            "test_attesters": test_attesters,
            "arb_usdc": arb_usdc,
            "usdc_amount": usdc_amount,
        }
    finally:
        for launch in anvil_launches:
            launch.close(log_level=logging.ERROR)


@pytest.mark.timeout(300)
def test_cctp_bridge_round_trip(simulation_setup, strategy_file):
    """Full CCTP bridge round-trip: bridge forward, bridge back, verify equity."""
    s = simulation_setup
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    s["mod"]._run_test_lifecycle(
        simulate=True,
        test_attesters=s["test_attesters"],
        arb_web3=s["arb_web3"],
        base_web3=s["base_web3"],
        deployer=s["deployer"],
        arb_usdc=s["arb_usdc"],
        private_key=s["private_key"],
        json_rpc_arb_sepolia=s["json_rpc_arb_sepolia"],
        json_rpc_base_sepolia=s["json_rpc_base_sepolia"],
        strategy_file=strategy_file,
        usdc_amount=s["usdc_amount"],
        bridge_amount="3",
        swap_amount="0",
        reverse_bridge_amount="3",
        trading_strategy_api_key=trading_strategy_api_key,
        attestation_timeout=60,
    )
