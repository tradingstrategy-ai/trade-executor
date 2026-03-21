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
import json
import logging
import os
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.vault.base import VaultSpec

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")

pytestmark = pytest.mark.skip(reason="Takes too long - TODO create a fast variant of this")


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


@pytest.mark.timeout(600)
def test_gmx_collateral_approval(simulation_setup, strategy_file):
    """Verify approve_gmx_trading() succeeds after whitelistGMX() deployment.

    Deploys a Lagoon vault with GMX whitelisting (which calls
    ``whitelistGMX()`` → ``allowApprovalDestination(syntheticsRouter)``),
    then calls ``approve_gmx_trading()`` to ensure the Guard allows the
    USDC approval for the SyntheticsRouter.

    This is a regression test for the production failure where the Guard
    reverted with ``"Approve address not allowed"`` because the deployed
    contract version did not whitelist the SyntheticsRouter.
    """
    from tradeexecutor.exchange_account.gmx import approve_gmx_trading

    s = simulation_setup
    mod = s["mod"]
    web3 = s["web3"]
    deployer = s["deployer"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        vault_record_file = str(Path(tmp_dir) / "vault-record.txt")

        # Deploy vault with GMX whitelisting
        deploy_env = {
            "STRATEGY_FILE": str(strategy_file),
            "PRIVATE_KEY": "0x" + deployer.account.key.hex(),
            "JSON_RPC_ARBITRUM": s["json_rpc_arbitrum"],
            "VAULT_RECORD_FILE": vault_record_file,
            "FUND_NAME": "Test GMX Approval",
            "FUND_SYMBOL": "TAPPR",
            "ANY_ASSET": "true",
            "PERFORMANCE_FEE": "0",
            "MANAGEMENT_FEE": "0",
            "UNIT_TESTING": "true",
            "LOG_LEVEL": "warning",
        }
        mod.run_cli(["lagoon-deploy-vault"], deploy_env)

        # Read deployment record
        deployment_json = vault_record_file.replace(".txt", ".json")
        with open(deployment_json) as f:
            deployment_data = json.load(f)

        dep = deployment_data["deployments"]["arbitrum"]
        vault_address = dep["vault_address"]
        module_address = dep["module_address"]

        # Construct LagoonVault
        vault = LagoonVault(
            web3,
            VaultSpec(web3.eth.chain_id, vault_address),
            trading_strategy_module_address=module_address,
        )

        # Call approve_gmx_trading — this must not revert
        deployer.sync_nonce(web3)
        tx_hash = approve_gmx_trading(vault, deployer)
        assert tx_hash, "approve_gmx_trading() returned empty tx hash"

        # Verify on-chain allowance for SyntheticsRouter
        addresses = get_contract_addresses("arbitrum")
        usdc_token = s["usdc_token"]
        safe_address = dep["safe_address"]
        allowance = usdc_token.contract.functions.allowance(
            Web3.to_checksum_address(safe_address),
            Web3.to_checksum_address(addresses.syntheticsrouter),
        ).call()
        assert allowance == 2**256 - 1, f"Expected unlimited allowance, got {allowance}"
