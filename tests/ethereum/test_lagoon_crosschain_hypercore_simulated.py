"""Test cross-chain Lagoon vault lifecycle with CCTP bridge + Hypercore vault (simulated).

Wraps ``scripts/lagoon/manual-trade-executor-crosschain-hypercore.py`` to run
the full multichain lifecycle using Anvil forks:

1. Deploy multichain vault (Arbitrum mainnet fork + HyperEVM mainnet fork)
2. Deposit USDC, settle
3. Bridge USDC via CCTP (Arb → HyperEVM, forged attestation)
4. Deposit into Hypercore HLP vault (batched multicall, mock contracts)
5. Verify vault position
6. Run correct-accounts
7. Withdraw from Hypercore vault
8. Bridge USDC back via CCTP (HyperEVM → Arb, forged attestation)

Requires ``JSON_RPC_ARBITRUM`` and ``JSON_RPC_HYPERLIQUID`` environment
variables pointing at real mainnet RPCs (used as Anvil fork sources).
"""

import importlib.util
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_HYPERLIQUID = os.environ.get("JSON_RPC_HYPERLIQUID")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not JSON_RPC_HYPERLIQUID,
    reason="JSON_RPC_ARBITRUM and JSON_RPC_HYPERLIQUID environment variables required",
)


def _load_script_module():
    """Load the manual test script as a Python module."""
    script_path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "scripts" / "lagoon" / "manual-trade-executor-crosschain-hypercore.py"
    )
    spec = importlib.util.spec_from_file_location("crosschain_hypercore_test_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def strategy_file() -> Path:
    path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "strategies" / "test_only" / "lagoon_crosschain_hypercore_manual_test.py"
    )
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.fixture()
def simulation_setup():
    """Set up Anvil forks, deployer, mock Hypercore contracts, and CCTP attesters.

    Yields all objects needed for the test lifecycle,
    then tears down Anvil forks.
    """
    usdc_amount = Decimal("10")
    mod = _load_script_module()

    (
        arb_web3, hyper_web3, deployer, private_key,
        json_rpc_arb, json_rpc_hyper,
        test_attesters, anvil_launches,
    ) = mod.setup_simulation(
        json_rpc_arb=JSON_RPC_ARBITRUM,
        json_rpc_hyper=JSON_RPC_HYPERLIQUID,
        simulate=True,
        private_key=None,
        usdc_amount=usdc_amount,
    )

    try:
        yield {
            "mod": mod,
            "arb_web3": arb_web3,
            "hyper_web3": hyper_web3,
            "deployer": deployer,
            "private_key": private_key,
            "json_rpc_arb": json_rpc_arb,
            "json_rpc_hyper": json_rpc_hyper,
            "test_attesters": test_attesters,
            "usdc_amount": usdc_amount,
        }
    finally:
        for launch in anvil_launches:
            launch.close(log_level=logging.ERROR)


@pytest.mark.timeout(300)
def test_crosschain_hypercore_vault(simulation_setup, strategy_file):
    """Full cross-chain lifecycle: bridge to HyperEVM, deposit into Hypercore vault, withdraw, bridge back."""
    s = simulation_setup
    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    s["mod"]._run_test_lifecycle(
        simulate=True,
        test_attesters=s["test_attesters"],
        arb_web3=s["arb_web3"],
        hyper_web3=s["hyper_web3"],
        deployer=s["deployer"],
        private_key=s["private_key"],
        json_rpc_arb=s["json_rpc_arb"],
        json_rpc_hyper=s["json_rpc_hyper"],
        strategy_file=strategy_file,
        usdc_amount=s["usdc_amount"],
        bridge_amount="7",
        vault_deposit_amount="5",
        trading_strategy_api_key=trading_strategy_api_key,
        attestation_timeout=60,
    )
