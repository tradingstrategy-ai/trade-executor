"""Test cross-chain Lagoon vault lifecycle in simulated (Anvil fork) mode.

Wraps ``scripts/lagoon/manual-trade-executor-multichain.py`` to run the full
multichain lifecycle using Arbitrum and Base mainnet forks:

1. Deploy multichain vault (Arbitrum + Base forks)
2. Deposit USDC, settle
3. Bridge USDC via CCTP (Arbitrum -> Base, forged attestation)
4. Optionally swap on Uniswap v3
5. Bridge USDC back via CCTP (Base -> Arbitrum, forged attestation)
6. Verify total equity equals deposited amount across all chains

Requires ``JSON_RPC_ARBITRUM`` and ``JSON_RPC_BASE`` environment variables
pointing at real mainnet RPCs used as Anvil fork sources.
"""

import importlib.util
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not JSON_RPC_BASE,
    reason="JSON_RPC_ARBITRUM and JSON_RPC_BASE environment variables required",
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
    network_config = {
        "network": "mainnet",
        "is_testnet": False,
        "json_rpc_arbitrum": JSON_RPC_ARBITRUM,
        "json_rpc_base": JSON_RPC_BASE,
        "rpc_env_keys": ("JSON_RPC_ARBITRUM", "JSON_RPC_BASE"),
        "chain_ids": (42161, 8453),
        "chain_names": ("Arbitrum", "Base"),
        "chain_slugs": ("arbitrum", "base"),
        "strategy_file": None,
    }

    (
        arb_web3, base_web3, deployer, private_key,
        json_rpc_arbitrum, json_rpc_base,
        test_attesters, anvil_launches,
    ) = mod.setup_simulation(
        network_config=network_config,
        simulate=True,
        private_key=None,
        usdc_amount=usdc_amount,
    )

    arb_usdc = mod.verify_deployer_balances(
        network_config=network_config,
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
            "network_config": network_config,
            "json_rpc_arbitrum": json_rpc_arbitrum,
            "json_rpc_base": json_rpc_base,
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
        network_config=s["network_config"],
        private_key=s["private_key"],
        json_rpc_arbitrum=s["json_rpc_arbitrum"],
        json_rpc_base=s["json_rpc_base"],
        strategy_file=strategy_file,
        usdc_amount=s["usdc_amount"],
        bridge_amount="3",
        swap_amount="0",
        reverse_bridge_amount="3",
        trading_strategy_api_key=trading_strategy_api_key,
        attestation_timeout=60,
    )
