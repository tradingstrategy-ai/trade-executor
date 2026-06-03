"""Test translate_trading_universe_to_lagoon_config() with Anvil fork deployment.

Uses mainnet Anvil forks (Arbitrum + Base) to verify that:
1. The universe-to-config translator produces correct LagoonConfig objects
2. The generated configs deploy successfully via deploy_multichain_lagoon_vault()
3. ERC-4626 vault addresses are whitelisted on-chain even with any_asset enabled

Requires JSON_RPC_ARBITRUM and JSON_RPC_BASE environment variables.
"""

import datetime
import logging
import os
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonMultichainDeployment,
    deploy_multichain_lagoon_vault,
)
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonSatelliteVault, LagoonVault
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_WHALE

from tradeexecutor.ethereum.lagoon.universe_config import translate_trading_universe_to_lagoon_config
from tradeexecutor.strategy.strategy_module import read_strategy_module

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not JSON_RPC_BASE,
    reason="JSON_RPC_ARBITRUM and JSON_RPC_BASE environment variables required",
)

#: Anvil default account #0 private key — deterministic deployer address across chains
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"


@pytest.fixture()
def deployer() -> LocalAccount:
    return Account.from_key(DEPLOYER_PRIVATE_KEY)


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
def strategy_universe():
    """Load the existing mainnet CCTP bridge strategy and create its universe."""
    mod = read_strategy_module(Path("strategies/test_only/cctp_bridge_start_test.py"))
    universe = mod.create_trading_universe(
        ts=datetime.datetime.now(datetime.UTC),
        client=None,
        execution_context=None,
        universe_options=None,
    )
    return universe


@pytest.mark.timeout(600)
def test_translate_universe_to_lagoon_config_and_deploy(
    web3_arbitrum,
    web3_base,
    deployer,
    strategy_universe,
):
    """Deploy Lagoon vaults using configs generated from a strategy universe.

    Full integration test: translate universe -> generate configs -> deploy on Anvil forks.
    Verifies:
    - Source chain gets a full LagoonVault
    - Satellite chain gets a LagoonSatelliteVault
    - Same deterministic Safe address on both chains
    - CCTP whitelisting on both chains
    - Uniswap V3 whitelisting on Base
    """

    salt_nonce = 42

    # Fund deployer with ETH on both forks
    for web3 in [web3_arbitrum, web3_base]:
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])

    chain_web3 = {
        "arbitrum": web3_arbitrum,
        "base": web3_base,
    }

    configs = translate_trading_universe_to_lagoon_config(
        universe=strategy_universe,
        chain_web3=chain_web3,
        asset_manager=deployer.address,
        safe_owners=[deployer.address],
        safe_threshold=1,
        safe_salt_nonce=salt_nonce,
        fund_name="Universe Test Vault",
        fund_symbol="UTV",
    )

    # Verify configs before deployment
    assert len(configs) == 2
    assert configs["arbitrum"].satellite_chain is False
    assert configs["base"].satellite_chain is True
    assert configs["base"].uniswap_v3 is not None, "Base should have Uniswap v3 configured"
    assert configs["arbitrum"].cctp_deployment is not None
    assert configs["base"].cctp_deployment is not None

    # Deploy
    result = deploy_multichain_lagoon_vault(
        chain_web3=chain_web3,
        deployer=deployer,
        chain_configs=configs,
    )

    # Verify deployment result
    assert isinstance(result, LagoonMultichainDeployment)
    assert len(result.deployments) == 2

    # Source chain gets a full vault
    assert isinstance(result.deployments["arbitrum"].vault, LagoonVault)
    assert result.deployments["arbitrum"].is_satellite is False

    # Satellite chain gets Safe + guard only
    assert isinstance(result.deployments["base"].vault, LagoonSatelliteVault)
    assert result.deployments["base"].is_satellite is True

    # Same deterministic Safe address on both chains
    assert result.deployments["arbitrum"].safe_address == result.deployments["base"].safe_address

    # Verify whitelisted items on Arbitrum
    arb_kinds = {e.kind for e in result.deployments["arbitrum"].whitelisted_items}
    assert "CCTP" in arb_kinds
    assert "Sender" in arb_kinds
    assert "Receiver" in arb_kinds
    assert "Token" in arb_kinds
    assert "Vault settlement" in arb_kinds  # Source chain has vault

    # Verify whitelisted items on Base
    base_kinds = {e.kind for e in result.deployments["base"].whitelisted_items}
    assert "CCTP" in base_kinds
    assert "Uniswap V3 router" in base_kinds
    assert "Vault settlement" not in base_kinds  # Satellite has no vault


@pytest.fixture()
def vault_strategy_universe():
    """Load the CCTP bridge + Ostium vault strategy and create its universe."""
    mod = read_strategy_module(Path("strategies/test_only/cctp_bridge_with_vault_test.py"))
    universe = mod.create_trading_universe(
        ts=datetime.datetime.now(datetime.UTC),
        client=None,
        execution_context=None,
        universe_options=None,
    )
    return universe


@pytest.mark.timeout(600)
def test_erc4626_vault_whitelisted_on_chain(
    web3_arbitrum,
    web3_base,
    deployer,
    vault_strategy_universe,
):
    """ERC-4626 vault addresses are whitelisted on-chain even with any_asset enabled.

    The guard's ``anyAsset`` flag bypasses token-level checks but NOT target and
    approval destination checks. Vault pairs from the strategy universe must be
    explicitly whitelisted via ``whitelistERC4626()`` during deployment.

    1. Load a strategy universe that includes an Ostium vault pair on Arbitrum.
    2. Generate Lagoon configs with ``any_asset=True``.
    3. Deploy via ``deploy_multichain_lagoon_vault()`` on Anvil forks.
    4. Verify the deployment result includes ``"ERC-4626 vault"`` whitelist entries.
    5. Read the on-chain guard and verify ``isAllowedApprovalDestination()`` returns
       True for the Ostium vault address.
    """

    salt_nonce = 99

    # 1. Fund deployer with ETH on both forks
    for web3 in [web3_arbitrum, web3_base]:
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])

    chain_web3 = {
        "arbitrum": web3_arbitrum,
        "base": web3_base,
    }

    # 2. Generate configs with any_asset=True
    configs = translate_trading_universe_to_lagoon_config(
        universe=vault_strategy_universe,
        chain_web3=chain_web3,
        asset_manager=deployer.address,
        safe_owners=[deployer.address],
        safe_threshold=1,
        safe_salt_nonce=salt_nonce,
        fund_name="Vault Whitelist Test",
        fund_symbol="VWT",
        any_asset=True,
    )

    assert configs["arbitrum"].erc_4626_vaults is not None, "Ostium vault should be in Arbitrum config"
    assert len(configs["arbitrum"].erc_4626_vaults) == 1
    assert configs["base"].erc_4626_vaults is None, "Base should have no ERC-4626 vaults"

    # 3. Deploy
    result = deploy_multichain_lagoon_vault(
        chain_web3=chain_web3,
        deployer=deployer,
        chain_configs=configs,
    )

    assert isinstance(result, LagoonMultichainDeployment)

    # 4. Verify whitelist entries include ERC-4626 vault
    arb_kinds = {e.kind for e in result.deployments["arbitrum"].whitelisted_items}
    assert "ERC-4626 vault" in arb_kinds, (
        f"Ostium vault not whitelisted. Got kinds: {arb_kinds}"
    )

    arb_vault_entries = [
        e for e in result.deployments["arbitrum"].whitelisted_items
        if e.kind == "ERC-4626 vault"
    ]
    ostium_address = "0x20D419a8e12C45f88fDA7c5760bb6923Cee27F98"
    assert any(
        e.address.lower() == ostium_address.lower()
        for e in arb_vault_entries
    ), f"Ostium vault address not found in whitelist entries: {arb_vault_entries}"

    # 5. Verify on-chain guard state via the trading strategy module (guard contract)
    module = result.deployments["arbitrum"].trading_strategy_module
    assert module.functions.isAllowedApprovalDestination(
        Web3.to_checksum_address(ostium_address)
    ).call(), "Ostium vault should be allowed as approval destination on-chain"
