"""Test CLI multichain Lagoon vault deployment with Anvil forks.

Tests two deployment modes:

1. External Anvil forks — forks created in fixtures, passed as JSON_RPC_xxx
2. SIMULATE=true — Web3Config creates its own Anvil forks internally

Uses the ``cctp_bridge_start_test.py`` strategy (Arbitrum + Base, CCTP bridge
+ Uniswap v3) to drive the multichain universe.

Requires ``JSON_RPC_ARBITRUM`` and ``JSON_RPC_BASE`` environment variables.
"""

import json
import logging
import os
from pathlib import Path

import pytest
from eth_account import Account
from typer.main import get_command
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.safe.deployment import fetch_safe_deployment
from eth_defi.token import USDC_WHALE

from tradeexecutor.cli.main import app
from tradeexecutor.utils.hex import hexbytes_to_hex_str

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not JSON_RPC_BASE,
    reason="JSON_RPC_ARBITRUM and JSON_RPC_BASE environment variables required",
)

#: Anvil default account #0 private key — deterministic deployer
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

#: Fixed salt nonce for deterministic Safe address across chains
SAFE_SALT_NONCE = 42


@pytest.fixture()
def strategy_file() -> Path:
    """CCTP bridge + Uniswap v3 multichain strategy."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "cctp_bridge_start_test.py"


@pytest.fixture()
def anvil_arbitrum() -> AnvilLaunch:
    launch = fork_network_anvil(
        JSON_RPC_ARBITRUM,
        unlocked_addresses=[USDC_WHALE[42161]],
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_base() -> AnvilLaunch:
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        unlocked_addresses=[USDC_WHALE[8453]],
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3_arbitrum(anvil_arbitrum) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_arbitrum.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 42161
    return web3


@pytest.fixture()
def web3_base(anvil_base) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_base.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 8453
    return web3


@pytest.fixture()
def deployer() -> Account:
    return Account.from_key(DEPLOYER_PRIVATE_KEY)


@pytest.mark.timeout(600)
def test_cli_lagoon_deploy_multichain_vault(
    web3_arbitrum,
    web3_base,
    anvil_arbitrum,
    anvil_base,
    deployer,
    strategy_file,
    mocker,
    tmp_path: Path,
):
    """Deploy multichain Lagoon vault via CLI using external Anvil forks.

    - Two Anvil forks created externally (Arbitrum + Base)
    - CLI receives Anvil RPC URLs as JSON_RPC_ARBITRUM / JSON_RPC_BASE
    - STRATEGY_FILE drives the multichain universe
    - Verifies deployment record: source chain + satellite chain
    """

    # Fund deployer with ETH on both forks
    for web3 in [web3_arbitrum, web3_base]:
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])

    multisig_owners = f"{deployer.address}"
    vault_record_file = tmp_path / "vault-record.txt"

    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "test_multichain_deploy",
        "NAME": "test_multichain_deploy",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ARBITRUM": anvil_arbitrum.json_rpc_url,
        "JSON_RPC_BASE": anvil_base.json_rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Multichain Test",
        "FUND_SYMBOL": "MCT",
        "MULTISIG_OWNERS": multisig_owners,
        "ANY_ASSET": "true",
        "SAFE_SALT_NONCE": str(SAFE_SALT_NONCE),
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)

    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # Read multichain deployment record
    deploy_record = json.load(vault_record_file.with_suffix(".json").open("rt"))
    assert deploy_record["multichain"] is True
    assert deploy_record["safe_salt_nonce"] == SAFE_SALT_NONCE
    assert len(deploy_record["deployments"]) == 2

    # Source chain (Arbitrum) has a vault
    arb_dep = deploy_record["deployments"]["arbitrum"]
    assert arb_dep["vault_address"] is not None
    assert arb_dep["is_satellite"] is False

    # Satellite chain (Base) has Safe + guard only
    base_dep = deploy_record["deployments"]["base"]
    assert base_dep["is_satellite"] is True

    # Same deterministic Safe address on both chains
    safe_address = arb_dep["safe_address"]
    assert safe_address == base_dep["safe_address"]

    # Verify Safe contract is deployed on both chains via web3
    for chain_name, web3 in [("arbitrum", web3_arbitrum), ("base", web3_base)]:
        code = web3.eth.get_code(Web3.to_checksum_address(safe_address))
        assert len(code) > 0, f"Safe not deployed on {chain_name} at {safe_address}"

        safe = fetch_safe_deployment(web3, safe_address)
        owners = safe.retrieve_owners()
        assert deployer.address in owners, f"Deployer not an owner on {chain_name}"
        assert safe.retrieve_threshold() == 1

        modules = safe.retrieve_modules()
        assert len(modules) >= 1, f"No guard module on {chain_name}"

    # Verify deployment report Markdown file was written
    md_path = vault_record_file.with_name("deployment-report.md")
    assert md_path.exists(), f"deployment-report.md not written at {md_path}"

    md_content = md_path.read_text()

    # Report contains deployment metadata
    assert "# Deployment report" in md_content
    assert safe_address in md_content

    # Report contains guard config tree for both chains
    assert "Arbitrum" in md_content
    assert "Base" in md_content

    # Report contains block explorer links (Arbitrum has arbiscan.io)
    assert "arbiscan.io/address/" in md_content

    # Report contains deployer address
    assert deployer.address in md_content

    # Sections rendered as bullet list tree
    assert "- **Senders" in md_content or "- **Any asset" in md_content


@pytest.mark.timeout(600)
def test_cli_lagoon_deploy_multichain_simulate(
    strategy_file,
    mocker,
    tmp_path: Path,
):
    """Deploy multichain Lagoon vault with SIMULATE=true.

    - No external Anvil forks — Web3Config creates them internally
    - Tests that multiple Anvil forks can be launched in a single CLI invocation
    - Original RPC URLs are passed; Web3Config forks each one
    - Vault record is not written in simulate mode, so we only verify
      the command completes without error
    """

    multisig_owners = Account.from_key(DEPLOYER_PRIVATE_KEY).address
    vault_record_file = tmp_path / "vault-record.txt"

    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "test_multichain_simulate",
        "NAME": "test_multichain_simulate",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ARBITRUM": JSON_RPC_ARBITRUM,
        "JSON_RPC_BASE": JSON_RPC_BASE,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "SIMULATE": "true",
        "LOG_LEVEL": "disabled",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Simulate Multichain",
        "FUND_SYMBOL": "SMC",
        "MULTISIG_OWNERS": multisig_owners,
        "ANY_ASSET": "true",
        "SAFE_SALT_NONCE": str(SAFE_SALT_NONCE),
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)

    # Should complete without error — previously crashed with
    # "Simulation can be used only with one chain"
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # Vault record is not written in simulate mode
    assert not vault_record_file.with_suffix(".json").exists()

    # Deployment report IS written even in simulate mode
    md_path = vault_record_file.with_name("deployment-report.md")
    assert md_path.exists(), "deployment-report.md should be written even in simulate mode"

    md_content = md_path.read_text()
    assert "# Deployment report" in md_content
    assert "- **Senders" in md_content or "- **Any asset" in md_content
