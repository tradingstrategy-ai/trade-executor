"""Test multichain deployment of the master-chain-multichain strategy.

Deploys a Lagoon vault across all chains used by the
``master-chain-multichain.py`` strategy (Ethereum, Base, Arbitrum,
Avalanche, HyperEVM, Monad) using Anvil forks, and verifies:

- All expected chains appear in the deployment record
- Arbitrum is the source chain (not a satellite)
- All other chains are satellites
- The same deterministic Safe address is shared across chains
- Guard modules exist on each chain

Hypercore native vaults (chain_id=9999) are mapped to HyperEVM (999)
for whitelisting — they are not a separate deployable chain.

Requires environment variables:

- ``JSON_RPC_ETHEREUM``
- ``JSON_RPC_BASE``
- ``JSON_RPC_ARBITRUM``
- ``JSON_RPC_AVALANCHE``
- ``JSON_RPC_HYPERLIQUID``
- ``JSON_RPC_MONAD``
- ``TRADING_STRATEGY_API_KEY``
"""

import json
import logging
import os
from pathlib import Path

import pytest
from eth_account import Account
from typer.main import get_command
from web3 import Web3

from eth_defi.hyperliquid.testing import setup_anvil_hypercore_mocks
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.safe.deployment import fetch_safe_deployment

from tradeexecutor.cli.main import app

logger = logging.getLogger(__name__)

JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_AVALANCHE = os.environ.get("JSON_RPC_AVALANCHE")
JSON_RPC_HYPERLIQUID = os.environ.get("JSON_RPC_HYPERLIQUID")
JSON_RPC_MONAD = os.environ.get("JSON_RPC_MONAD")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = pytest.mark.skipif(
    not all([
        JSON_RPC_ETHEREUM,
        JSON_RPC_BASE,
        JSON_RPC_ARBITRUM,
        JSON_RPC_AVALANCHE,
        JSON_RPC_HYPERLIQUID,
        JSON_RPC_MONAD,
        TRADING_STRATEGY_API_KEY,
    ]),
    reason="Requires JSON_RPC_ETHEREUM, JSON_RPC_BASE, JSON_RPC_ARBITRUM, "
           "JSON_RPC_AVALANCHE, JSON_RPC_HYPERLIQUID, JSON_RPC_MONAD, "
           "and TRADING_STRATEGY_API_KEY environment variables",
)

#: Anvil default account #0 private key
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

#: Fixed salt nonce for deterministic Safe address across chains
SAFE_SALT_NONCE = 42

#: Expected chains in the deployment (slug -> source RPC env var)
EXPECTED_CHAINS = {
    "ethereum": JSON_RPC_ETHEREUM,
    "base": JSON_RPC_BASE,
    "arbitrum": JSON_RPC_ARBITRUM,
    "avalanche": JSON_RPC_AVALANCHE,
    "hyperliquid": JSON_RPC_HYPERLIQUID,
    "monad": JSON_RPC_MONAD,
}


def _fork_chain(rpc_url: str, **kwargs) -> AnvilLaunch:
    return fork_network_anvil(rpc_url, **kwargs)


@pytest.fixture()
def strategy_file() -> Path:
    path = Path(__file__).resolve().parent / ".." / ".." / "strategies" / "master-chain-multichain.py"
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.fixture()
def anvil_ethereum() -> AnvilLaunch:
    launch = _fork_chain(JSON_RPC_ETHEREUM)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_base() -> AnvilLaunch:
    launch = _fork_chain(JSON_RPC_BASE)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_arbitrum() -> AnvilLaunch:
    launch = _fork_chain(JSON_RPC_ARBITRUM)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_avalanche() -> AnvilLaunch:
    launch = _fork_chain(JSON_RPC_AVALANCHE)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_hyperevm() -> AnvilLaunch:
    # HyperEVM needs higher gas limit for mock contract deployment
    launch = _fork_chain(JSON_RPC_HYPERLIQUID, gas_limit=30_000_000)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_monad() -> AnvilLaunch:
    launch = _fork_chain(JSON_RPC_MONAD)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def deployer() -> Account:
    return Account.from_key(DEPLOYER_PRIVATE_KEY)


@pytest.fixture()
def web3_hyperevm(anvil_hyperevm: AnvilLaunch, deployer: Account) -> Web3:
    """Create Web3 for HyperEVM and deploy mock Hypercore contracts."""
    web3 = create_multi_provider_web3(
        anvil_hyperevm.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    # Fund deployer with HYPE for gas
    web3.provider.make_request(
        "anvil_setBalance",
        [deployer.address, hex(1_000 * 10**18)],
    )
    # Deploy mock CoreWriter + CoreDepositWallet
    setup_anvil_hypercore_mocks(web3, deployer.address)
    return web3


@pytest.fixture()
def all_anvils(
    anvil_ethereum,
    anvil_base,
    anvil_arbitrum,
    anvil_avalanche,
    anvil_hyperevm,
    anvil_monad,
    web3_hyperevm,
) -> dict[str, AnvilLaunch]:
    """Map of chain slug → Anvil launch (ensures all forks are up)."""
    return {
        "ethereum": anvil_ethereum,
        "base": anvil_base,
        "arbitrum": anvil_arbitrum,
        "avalanche": anvil_avalanche,
        "hyperliquid": anvil_hyperevm,
        "monad": anvil_monad,
    }


@pytest.mark.timeout(600)
def test_deploy_multichain_vault(
    all_anvils: dict[str, AnvilLaunch],
    deployer: Account,
    strategy_file: Path,
    mocker,
    tmp_path: Path,
):
    """Deploy master-chain-multichain vault across all chains.

    Verifies:
    - Deployment completes without error
    - All expected chains appear in the deployment record
    - Arbitrum is the source chain (has vault_address, not a satellite)
    - Other chains are satellites
    - Same Safe address on all chains
    - Safe and guard modules deployed on each chain
    """

    # Fund deployer with ETH on all forks
    for slug, anvil in all_anvils.items():
        web3 = create_multi_provider_web3(
            anvil.json_rpc_url,
            default_http_timeout=(3, 250.0),
        )
        web3.provider.make_request(
            "anvil_setBalance",
            [deployer.address, hex(100 * 10**18)],
        )

    vault_record_file = tmp_path / "vault-record.txt"

    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "test_master_chain_multichain",
        "NAME": "test_master_chain_multichain",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ETHEREUM": all_anvils["ethereum"].json_rpc_url,
        "JSON_RPC_BASE": all_anvils["base"].json_rpc_url,
        "JSON_RPC_ARBITRUM": all_anvils["arbitrum"].json_rpc_url,
        "JSON_RPC_AVALANCHE": all_anvils["avalanche"].json_rpc_url,
        "JSON_RPC_HYPERLIQUID": all_anvils["hyperliquid"].json_rpc_url,
        "JSON_RPC_MONAD": all_anvils["monad"].json_rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Master Chain Multichain Test",
        "FUND_SYMBOL": "MCMT",
        "ANY_ASSET": "true",
        "SAFE_SALT_NONCE": str(SAFE_SALT_NONCE),
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
    }

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)

    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # Read multichain deployment record
    deploy_record = json.load(vault_record_file.with_suffix(".json").open("rt"))
    assert deploy_record["multichain"] is True
    assert deploy_record["safe_salt_nonce"] == SAFE_SALT_NONCE

    deployments = deploy_record["deployments"]
    deployed_chains = set(deployments.keys())

    # All expected chains should be present
    for chain_slug in EXPECTED_CHAINS:
        assert chain_slug in deployed_chains, (
            f"Chain {chain_slug} missing from deployment. "
            f"Deployed: {deployed_chains}"
        )

    # Arbitrum is the source chain (primary)
    arb_dep = deployments["arbitrum"]
    assert arb_dep["vault_address"] is not None
    assert arb_dep["is_satellite"] is False

    # All other chains are satellites
    for slug, dep in deployments.items():
        if slug == "arbitrum":
            continue
        assert dep["is_satellite"] is True, f"{slug} should be a satellite"

    # Same deterministic Safe address across all chains
    safe_address = arb_dep["safe_address"]
    for slug, dep in deployments.items():
        assert dep["safe_address"] == safe_address, (
            f"Safe address mismatch on {slug}: "
            f"{dep['safe_address']} != {safe_address}"
        )

    # Verify Safe contract and guard module on each chain
    for slug, anvil in all_anvils.items():
        if slug not in deployments:
            continue

        web3 = create_multi_provider_web3(
            anvil.json_rpc_url,
            default_http_timeout=(3, 250.0),
        )

        code = web3.eth.get_code(Web3.to_checksum_address(safe_address))
        assert len(code) > 0, f"Safe not deployed on {slug} at {safe_address}"

        safe = fetch_safe_deployment(web3, safe_address)
        owners = safe.retrieve_owners()
        assert deployer.address in owners, f"Deployer not an owner on {slug}"
        assert safe.retrieve_threshold() == 1

        modules = safe.retrieve_modules()
        assert len(modules) >= 1, f"No guard module on {slug}"

    # Verify deployment report Markdown file
    md_path = vault_record_file.with_name("deployment-report.md")
    assert md_path.exists()

    md_content = md_path.read_text()

    # All chains mentioned in report
    for chain_slug in EXPECTED_CHAINS:
        assert chain_slug in md_content.lower() or chain_slug.title() in md_content, \
            f"Chain {chain_slug} not found in deployment report"

    # Contains Safe address
    assert safe_address in md_content

    # Contains block explorer links for chains that have them
    assert "arbiscan.io/address/" in md_content
    assert "basescan.org/address/" in md_content
    assert "etherscan.io/address/" in md_content
