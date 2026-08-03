"""Test Hyperliquid vault whitelisting through ``lagoon-deploy-vault``.

The test deliberately uses the same fixed HyperEVM midnight fork as the
``web3-ethereum-defi`` guard tests. It exercises the complete strategy-file
deployment path: strategy universe construction, Lagoon config translation,
guard deployment, and per-vault whitelist transactions.
"""

import json
import logging
import os
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.signers.local import LocalAccount
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from eth_defi.abi import encode_function_call, get_deployed_contract
from eth_defi.hyperliquid.core_writer import CORE_WRITER_ADDRESS, encode_vault_deposit
from eth_defi.hyperliquid.testing import deploy_mock_core_writer
from eth_defi.provider.anvil import AnvilLaunch, fork_network_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.fork_blocks import HYPERLIQUID_MIDNIGHT_BLOCK

from tradeexecutor.cli.main import app
from tradeexecutor.strategy.execution_context import one_off_execution_context
from tradeexecutor.strategy.pandas_trader.create_universe_wrapper import (
    call_create_trading_universe,
)
from tradeexecutor.strategy.strategy_module import read_strategy_module


JSON_RPC_HYPERLIQUID = os.environ.get("JSON_RPC_HYPERLIQUID")

pytestmark = pytest.mark.skipif(
    not JSON_RPC_HYPERLIQUID,
    reason="JSON_RPC_HYPERLIQUID environment variable required",
)


#: Anvil default account #0 private key.
DEPLOYER_PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)


@pytest.fixture()
def deployer() -> LocalAccount:
    """Return the deterministic Anvil deployment account."""
    return Account.from_key(DEPLOYER_PRIVATE_KEY)


@pytest.fixture()
def anvil_hyperliquid() -> AnvilLaunch:
    """Fork HyperEVM at the canonical midnight block with deployment-sized blocks."""
    launch = fork_network_anvil(
        JSON_RPC_HYPERLIQUID,
        fork_block_number=HYPERLIQUID_MIDNIGHT_BLOCK,
        gas_limit=30_000_000,
    )
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def web3(anvil_hyperliquid: AnvilLaunch) -> Web3:
    """Connect to the deterministic HyperEVM Anvil fork."""
    web3 = create_multi_provider_web3(
        anvil_hyperliquid.json_rpc_url,
        default_http_timeout=(3, 500.0),
    )
    assert web3.eth.chain_id == 999
    return web3


@pytest.fixture()
def strategy_file() -> Path:
    """Return the self-contained Hyperliquid vault strategy used by the test."""
    return (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "test_only"
        / "minimal_hyperliquid_strategy.py"
    )


@pytest.mark.timeout(600)
def test_cli_whitelists_known_hyperliquid_vaults(
    anvil_hyperliquid: AnvilLaunch,
    deployer: LocalAccount,
    mocker: MockerFixture,
    strategy_file: Path,
    tmp_path: Path,
    web3: Web3,
) -> None:
    """Deploy a restrictive Hyperliquid guard from a strategy universe.

    1. Construct the strategy universe and collect its native vault addresses.
    2. Reject a settlement cooldown without its required settlement cap.
    3. Deploy through ``lagoon-deploy-vault`` with the explicit whitelist option.
    4. Check both broad guard flags remain disabled, every universe vault has a
       dedicated Hypercore whitelist entry, and the Lagoon settlement safety
       policy is applied.
    5. Validate a CoreWriter vault-transfer call for every whitelisted vault.
    """
    # 1. Construct the exact strategy universe used by the CLI deployment path.
    strategy_module = read_strategy_module(strategy_file)
    universe = call_create_trading_universe(
        strategy_module.create_trading_universe,
        client=None,
        universe_options=strategy_module.get_universe_options(),
        execution_context=one_off_execution_context,
    )
    known_vaults = {
        Web3.to_checksum_address(pair.pool_address)
        for pair in universe.iterate_pairs()
        if pair.is_hyperliquid_vault()
    }
    assert known_vaults

    web3.provider.make_request(
        "anvil_setBalance", [deployer.address, hex(100 * 10**18)]
    )
    vault_record_file = tmp_path / "hyperliquid-vault.txt"
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "test_known_hyperliquid_vaults",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_HYPERLIQUID": anvil_hyperliquid.json_rpc_url,
        "CHAIN_NAME": "hyperliquid",
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Known Hyperliquid vaults",
        "FUND_SYMBOL": "KHV",
        "MULTISIG_OWNERS": deployer.address,
        "ASSET_MANAGER": deployer.address,
        "SAFE_SALT_NONCE": "4242",
    }

    # 2. Reject a cooldown without the required settlement amount through Typer.
    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    with pytest.raises(
        AssertionError,
        match="--lagoon-settlement-cooldown requires --lagoon-max-settlement-amount",
    ):
        cli.main(
            args=["lagoon-deploy-vault", "--lagoon-settlement-cooldown", "3600"],
            standalone_mode=False,
        )

    # 3. Deploy the Lagoon vault and TradingStrategyModuleV0 through the CLI.
    cli.main(
        args=[
            "lagoon-deploy-vault",
            "--whitelist-known-hyperliquid-vaults",
            "--lagoon-max-settlement-amount",
            "250",
        ],
        standalone_mode=False,
    )

    deployment_record = vault_record_file.with_suffix(".json")
    deployment = json.loads(deployment_record.read_text())
    hyperliquid_deployment = deployment["deployments"]["hyperliquid"]
    module = get_deployed_contract(
        web3,
        "safe-integration/TradingStrategyModuleV0.json",
        hyperliquid_deployment["module_address"],
    )

    # 4. The options take the restrictive, per-vault and settlement-safety paths.
    assert hyperliquid_deployment["config"]["any_asset"] is False
    assert hyperliquid_deployment["config"]["any_hypercore_vault"] is False
    assert hyperliquid_deployment["config"]["max_settlement_amount"] == "250"
    assert hyperliquid_deployment["config"]["settlement_cooldown"] == 24 * 60 * 60
    assert {
        address.lower()
        for address in hyperliquid_deployment["config"]["hypercore_vaults"]
    } == {address.lower() for address in known_vaults}
    assert module.functions.anyAsset().call() is False
    assert module.functions.anyHypercoreVault().call() is False
    settlement_config = module.functions.getLagoonSettlementSafetyConfig(
        hyperliquid_deployment["vault_address"]
    ).call()
    assert settlement_config[0] is True
    assert settlement_config[1] is True
    assert settlement_config[4] == 250 * 10**6
    assert settlement_config[5] == 24 * 60 * 60

    whitelisted_vaults = {
        Web3.to_checksum_address(entry["address"])
        for entry in hyperliquid_deployment["whitelisted_items"]
        if entry["kind"] == "Hypercore vault"
    }
    assert whitelisted_vaults == known_vaults

    # 5. Every universe vault passes the module's CoreWriter action-2 validation.
    core_writer = deploy_mock_core_writer(web3)
    for vault_address in known_vaults:
        function_call = core_writer.functions.sendRawAction(
            encode_vault_deposit(vault_address, 5_000_000),
        )
        module.functions.validateCall(
            deployer.address,
            Web3.to_checksum_address(CORE_WRITER_ADDRESS),
            encode_function_call(function_call, function_call.arguments),
        ).call()
