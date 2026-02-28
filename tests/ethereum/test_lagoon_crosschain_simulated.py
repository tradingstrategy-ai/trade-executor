"""Test cross-chain Lagoon vault lifecycle in simulated (Anvil fork) mode.

Wraps ``scripts/lagoon/manual-trade-executor-multichain.py`` to run the full
multichain lifecycle using Anvil forks:

1. Deploy multichain vault (Arb Sepolia + Base Sepolia forks)
2. Deposit USDC, settle
3. Bridge USDC via CCTP (Arb → Base, forged attestation)
4. Optionally swap on Uniswap v3
5. Bridge USDC back via CCTP (Base → Arb, forged attestation)
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

from eth_defi.cctp.testing import replace_attester_on_fork
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, set_balance, fund_erc20_on_anvil, AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
JSON_RPC_BASE_SEPOLIA = os.environ.get("JSON_RPC_BASE_SEPOLIA")

ARBITRUM_SEPOLIA_CHAIN_ID = 421614
BASE_SEPOLIA_CHAIN_ID = 84532

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM_SEPOLIA or not JSON_RPC_BASE_SEPOLIA,
    reason="JSON_RPC_ARBITRUM_SEPOLIA and JSON_RPC_BASE_SEPOLIA environment variables required",
)


@pytest.fixture()
def arb_anvil() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_ARBITRUM_SEPOLIA)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def base_anvil() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_BASE_SEPOLIA)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def arb_web3(arb_anvil):
    return create_multi_provider_web3(arb_anvil.json_rpc_url)


@pytest.fixture()
def base_web3(base_anvil):
    return create_multi_provider_web3(base_anvil.json_rpc_url)


@pytest.fixture()
def deployer(arb_web3, base_web3) -> HotWallet:
    wallet = HotWallet.create_for_testing(arb_web3, eth_amount=100)
    set_balance(base_web3, wallet.address, 100 * 10**18)
    return wallet


@pytest.fixture()
def test_attesters(arb_web3, base_web3):
    return {
        ARBITRUM_SEPOLIA_CHAIN_ID: replace_attester_on_fork(arb_web3),
        BASE_SEPOLIA_CHAIN_ID: replace_attester_on_fork(base_web3),
    }


@pytest.fixture()
def strategy_file() -> Path:
    path = (
        Path(__file__).resolve().parent / ".." / ".." /
        "strategies" / "test_only" / "lagoon_crosschain_manual_test.py"
    )
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.mark.timeout(300)
def test_cctp_bridge_round_trip(
    arb_anvil,
    base_anvil,
    arb_web3,
    base_web3,
    deployer,
    test_attesters,
    strategy_file,
):
    """Full CCTP bridge round-trip: bridge forward, bridge back, verify equity."""
    script_path = Path(__file__).resolve().parent / ".." / ".." / "scripts" / "lagoon" / "manual-trade-executor-multichain.py"
    spec = importlib.util.spec_from_file_location("crosschain_test_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    usdc_amount = Decimal("5")

    # Fund deployer with USDC on Arb Sepolia fork
    usdc_raw = int(usdc_amount * 10**6) * 10  # 10x headroom
    fund_erc20_on_anvil(
        arb_web3,
        USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID],
        deployer.address,
        usdc_raw,
    )

    arb_usdc = fetch_erc20_details(arb_web3, USDC_NATIVE_TOKEN[ARBITRUM_SEPOLIA_CHAIN_ID])
    private_key = "0x" + deployer.account.key.hex()

    trading_strategy_api_key = os.environ.get("TRADING_STRATEGY_API_KEY", "")

    mod._run_test_lifecycle(
        simulate=True,
        test_attesters=test_attesters,
        arb_web3=arb_web3,
        base_web3=base_web3,
        deployer=deployer,
        arb_usdc=arb_usdc,
        private_key=private_key,
        json_rpc_arb_sepolia=arb_anvil.json_rpc_url,
        json_rpc_base_sepolia=base_anvil.json_rpc_url,
        strategy_file=strategy_file,
        usdc_amount=usdc_amount,
        bridge_amount="3",
        swap_amount="0",
        reverse_bridge_amount="1",
        trading_strategy_api_key=trading_strategy_api_key,
        attestation_timeout=60,
    )
