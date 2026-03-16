"""lagoon-deploy-vault CLI command.

See :ref:`vault deployment` for the full documentation how to use this command.

Example how to manually test:

.. code-block:: shell

    export SIMULATE=true
    export FUND_NAME="Up only and then more"
    export FUND_SYMBOL="UP"
    export VAULT_RECORD_FILE="/tmp/sample-vault-deployment.json"
    export OWNER_ADDRESS="0x238B0435F69355e623d99363d58F7ba49C408491"

    #
    # Asset configuration
    #

    # USDC
    export DENOMINATION_ASSET="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    # Whitelisted tokens for Polygon: WETH, WMATIC
    export WHITELISTED_ASSETS="0x7ceb23fd6bc0add59e62ac25578270cff1b9f619 0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"

    #
    # Secret configuration
    #

    export JSON_RPC_POLYGON=
    export PRIVATE_KEY=
    # Is Polygonscan.com API key, passed to Forge
    export ETHERSCAN_API_KEY=

    #
    # Verifier configuration (optional)
    #

    # For Blockscout-based chains (e.g., Derive Chain):
    # export VERIFIER=blockscout
    # export VERIFIER_URL=https://explorer.derive.xyz/api

    #
    # Asset managers (optional, defaults to PRIVATE_KEY address)
    #

    # Ordered comma-separated list. The first address becomes the
    # primary asset manager and Lagoon valuation manager. Any later
    # addresses become secondary asset managers with guard sender
    # permissions only.
    #
    # export ASSET_MANAGER="0x..., 0x..."

    trade-executor lagoon-deploy-vault
"""

import datetime
import json
import os.path
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from eth_defi.aave_v3.constants import AAVE_V3_DEPLOYMENTS
from eth_defi.aave_v3.deployment import \
    fetch_deployment as fetch_aave_deployment
from eth_defi.abi import ONE_ADDRESS_STR
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.erc_4626.vault_protocol.lagoon.config import \
    get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    DEFAULT_MANAGEMENT_RATE, DEFAULT_PERFORMANCE_RATE,
    LagoonDeploymentParameters, deploy_automated_lagoon_vault,
    deploy_multichain_lagoon_vault)
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.constants import UNISWAP_V2_DEPLOYMENTS
from eth_defi.uniswap_v2.deployment import fetch_deployment
from eth_defi.uniswap_v3.constants import UNISWAP_V3_DEPLOYMENTS
from eth_defi.uniswap_v3.deployment import \
    fetch_deployment as fetch_deployment_uni_v3
from tradingstrategy.chain import ChainId
from typer import Option
from web3 import Web3

from tradeexecutor.cli.bootstrap import (create_web3_config, prepare_cache,
                                         prepare_token_cache)
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import choose_single_chain, create_hot_wallet
from tradeexecutor.cli.commands.shared_options import \
    parse_comma_separated_list
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.lagoon.deploy_report import (
    generate_multichain_deployment_report,
    print_deployment_report,
)
from tradeexecutor.ethereum.lagoon.preflight_report import log_deployment_preflight_report
from tradeexecutor.ethereum.lagoon.universe_config import \
    translate_trading_universe_to_lagoon_config
from tradeexecutor.ethereum.web3config import collect_rpc_kwargs
from tradeexecutor.monkeypatch.web3 import \
    construct_sign_and_send_raw_middleware
from tradeexecutor.strategy.execution_context import one_off_execution_context
from tradeexecutor.strategy.pandas_trader.create_universe_wrapper import \
    call_create_trading_universe
from tradeexecutor.strategy.strategy_module import read_strategy_module


def _normalize_multisig_owners(multisig_owners: list[str] | None, hot_wallet: HotWallet) -> list[str]:
    """Normalise owners list so single-chain and multichain paths behave identically."""
    if multisig_owners:
        return multisig_owners
    return [hot_wallet.address]


def _calculate_safe_threshold(multisig_owners: list[str]) -> int:
    """Lagoon deployment policy for Safe threshold."""
    return max(1, len(multisig_owners) - 1)


def _resolve_asset_managers(asset_manager_addresses: list[str] | str | None, hot_wallet: HotWallet) -> list[str]:
    """Resolve ordered asset manager addresses.

    Defaults to the deployer hot wallet when the CLI input is omitted or
    contains only whitespace.
    """
    if isinstance(asset_manager_addresses, str):
        raw_addresses = [asset_manager_addresses]
    elif asset_manager_addresses:
        raw_addresses = list(asset_manager_addresses)
    else:
        return [hot_wallet.address]

    resolved_addresses = [
        Web3.to_checksum_address(address.strip())
        for address in raw_addresses
        if address and address.strip()
    ]
    if resolved_addresses:
        return resolved_addresses
    return [hot_wallet.address]


def _write_file(path: Path, content: str) -> None:
    with open(path, "wt") as out:
        out.write(content)


def _write_json_file(path: Path, data: Any, *, indent: int | None = None) -> None:
    with open(path, "wt") as out:
        out.write(json.dumps(data, indent=indent))


def _write_deployment_artifacts(vault_record_file: Path | None, *, text_payload: str, json_payload: Any, simulate: bool, logger) -> None:
    """Write the shared human/machine readable deployment artifacts."""
    if not vault_record_file or simulate:
        logger.info("Skipping record file because of simulation")
        return

    _write_file(vault_record_file, text_payload)
    _write_json_file(vault_record_file.with_suffix(".json"), json_payload, indent=2 if isinstance(json_payload, dict) and json_payload.get("multichain") else None)
    logger.info("Wrote deployment record to %s", os.path.abspath(vault_record_file))


def _write_markdown_report(vault_record_file: Path | None, markdown_report: str, logger) -> None:
    """Write deployment Markdown report next to other artifacts."""
    if not vault_record_file:
        return

    md_path = vault_record_file.with_name("deployment-report.md")
    _write_file(md_path, markdown_report)
    logger.info("Wrote deployment report to %s", os.path.abspath(md_path))


def _serialise_simple_dataclass(value: Any) -> dict[str, Any]:
    """Serialise a dataclass containing only scalar-ish fields."""
    assert is_dataclass(value), f"Expected dataclass, got {type(value)}"
    return {
        field.name: _serialise_artifact_value(getattr(value, field.name))
        for field in fields(value)
    }


def _serialise_contract_address(contract: Any) -> str | None:
    if contract is None:
        return None
    return getattr(contract, "address", None)


def _serialise_erc_4626_vaults(vaults: list[Any] | None) -> list[dict[str, Any]]:
    if not vaults:
        return []

    return [
        {
            "address": getattr(vault, "vault_address", getattr(vault, "address", None)),
            "name": getattr(vault, "name", None),
            "symbol": getattr(vault, "symbol", None),
        }
        for vault in vaults
    ]


def _serialise_whitelist_entries(entries: tuple[Any, ...] | list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "kind": entry.kind,
            "name": entry.name,
            "address": entry.address,
        }
        for entry in entries
    ]


def _serialise_lagoon_config(config: Any) -> dict[str, Any]:
    """Capture all LagoonConfig fields in a JSON-safe form."""
    return {
        "parameters": _serialise_simple_dataclass(config.parameters),
        "safe_owners": [str(owner) for owner in config.safe_owners],
        "safe_threshold": config.safe_threshold,
        "asset_manager": str(config.asset_manager) if config.asset_manager else None,
        "asset_managers": [str(manager) for manager in config.asset_managers or []],
        "uniswap_v2": {
            "factory": _serialise_contract_address(config.uniswap_v2.factory),
            "router": _serialise_contract_address(config.uniswap_v2.router),
            "weth": _serialise_contract_address(config.uniswap_v2.weth),
            "init_code_hash": config.uniswap_v2.init_code_hash,
        } if config.uniswap_v2 else None,
        "uniswap_v3": {
            "factory": _serialise_contract_address(config.uniswap_v3.factory),
            "router": _serialise_contract_address(config.uniswap_v3.swap_router),
            "position_manager": _serialise_contract_address(config.uniswap_v3.position_manager),
            "quoter": _serialise_contract_address(config.uniswap_v3.quoter),
            "weth": _serialise_contract_address(config.uniswap_v3.weth),
            "quoter_v2": config.uniswap_v3.quoter_v2,
            "router_v2": config.uniswap_v3.router_v2,
        } if config.uniswap_v3 else None,
        "aave_v3": {
            "pool": _serialise_contract_address(config.aave_v3.pool),
            "data_provider": _serialise_contract_address(config.aave_v3.data_provider),
            "oracle": _serialise_contract_address(config.aave_v3.oracle),
            "ausdc": getattr(config.aave_v3.ausdc, "address", None),
        } if config.aave_v3 else None,
        "cowswap": config.cowswap,
        "velora": config.velora,
        "gmx_deployment": _serialise_simple_dataclass(config.gmx_deployment) if config.gmx_deployment else None,
        "cctp_deployment": _serialise_simple_dataclass(config.cctp_deployment) if config.cctp_deployment else None,
        "any_asset": config.any_asset,
        "etherscan_api_key": "<redacted>" if config.etherscan_api_key else None,
        "verifier": config.verifier,
        "verifier_url": config.verifier_url,
        "use_forge": config.use_forge,
        "between_contracts_delay_seconds": config.between_contracts_delay_seconds,
        "erc_4626_vaults": _serialise_erc_4626_vaults(config.erc_4626_vaults),
        "guard_only": config.guard_only,
        "existing_vault_address": config.existing_vault_address,
        "existing_safe_address": config.existing_safe_address,
        "vault_abi": config.vault_abi,
        "factory_contract": config.factory_contract,
        "from_the_scratch": config.from_the_scratch,
        "hypercore_vaults": [str(vault) for vault in config.hypercore_vaults or []],
        "assets": [str(asset) for asset in config.assets or []],
        "safe_salt_nonce": config.safe_salt_nonce,
        "safe_proxy_factory_address": config.safe_proxy_factory_address,
        "forge_cache_dir": str(config.forge_cache_dir) if config.forge_cache_dir else None,
        "deploy_retries": config.deploy_retries,
        "satellite_chain": config.satellite_chain,
    }


def _serialise_artifact_value(value: Any) -> Any:
    """Best-effort conversion for JSON deployment records."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialise_artifact_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialise_artifact_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return _serialise_simple_dataclass(value)
    if hasattr(value, "address"):
        return value.address
    return str(value)


def _format_multichain_text_section(title: str, values: dict[str, Any], indent: str = "  ") -> list[str]:
    lines = [title]
    for key, value in values.items():
        pretty_key = key.replace("_", " ").replace("-", " ").capitalize()
        if isinstance(value, dict):
            lines.append(f"{indent}{pretty_key}:")
            for child_key, child_value in value.items():
                child_pretty_key = child_key.replace("_", " ").replace("-", " ").capitalize()
                lines.append(f"{indent}  {child_pretty_key}: {child_value}")
        elif isinstance(value, list):
            if not value:
                lines.append(f"{indent}{pretty_key}: []")
            elif all(isinstance(item, dict) for item in value):
                lines.append(f"{indent}{pretty_key}:")
                for idx, item in enumerate(value, start=1):
                    lines.append(f"{indent}  - #{idx}")
                    for child_key, child_value in item.items():
                        child_pretty_key = child_key.replace("_", " ").replace("-", " ").capitalize()
                        lines.append(f"{indent}    {child_pretty_key}: {child_value}")
            else:
                lines.append(f"{indent}{pretty_key}: {', '.join(str(item) for item in value)}")
        else:
            lines.append(f"{indent}{pretty_key}: {value}")
    lines.append("")
    return lines


def _build_multichain_artifact_payload(
    result,
    safe_salt_nonce: int,
    chain_configs: dict[str, Any],
    guard_report: str,
) -> tuple[str, dict[str, Any]]:
    """Build the human-readable and JSON deployment payloads for multichain deploys."""
    deployment_data = {
        "multichain": True,
        "safe_salt_nonce": safe_salt_nonce,
        "deployments": {},
        "guard_report": guard_report,
    }
    lines: list[str] = [
        "Multichain Lagoon deployment",
        f"Safe salt nonce: {safe_salt_nonce}",
        f"Shared Safe: {result.safe_address}",
        "",
    ]
    for slug, dep in result.deployments.items():
        deployment_fields = dep.get_deployment_data()
        config_snapshot = _serialise_lagoon_config(chain_configs[slug])
        whitelist_entries = _serialise_whitelist_entries(dep.whitelisted_items)
        deployment_data["deployments"][slug] = {
            "vault_address": dep.vault.address if hasattr(dep.vault, "address") else None,
            "safe_address": dep.safe_address,
            "module_address": dep.trading_strategy_module.address if dep.trading_strategy_module else None,
            "asset_manager": dep.asset_manager,
            "asset_managers": list(dep.asset_managers),
            "valuation_manager": dep.valuation_manager,
            "is_satellite": dep.is_satellite,
            "deployment_data": deployment_fields,
            "whitelisted_items": whitelist_entries,
            "config": config_snapshot,
        }
        lines.append(f"Chain: {slug}")
        lines.extend(_format_multichain_text_section("  Deployment", deployment_fields, indent="    "))
        lines.extend(_format_multichain_text_section("  Lagoon config", config_snapshot, indent="    "))
        lines.extend(_format_multichain_text_section("  Guard whitelist", {"entries": whitelist_entries}, indent="    "))

    lines.append("Guard report")
    lines.append(guard_report)
    return "\n".join(lines), deployment_data


def _build_guard_migration_instructions(deploy_info, vault_adapter_address: str) -> dict[str, Any]:
    """Build manual Safe migration instructions for guard-only redeploys."""
    mods = deploy_info.safe.retrieve_modules()
    assert len(mods) == 1, f"Expected only one module enabled, got: {mods}"

    old_guard_address = deploy_info.old_trading_strategy_module.address if deploy_info.old_trading_strategy_module else vault_adapter_address
    safe_address = deploy_info.safe.address

    transactions = [
        {
            "step": 1,
            "target": safe_address,
            "function": "disableModule",
            "args": [ONE_ADDRESS_STR, old_guard_address],
            "call": f"{safe_address}.disableModule({ONE_ADDRESS_STR}, {old_guard_address})",
        },
        {
            "step": 2,
            "target": safe_address,
            "function": "enableModule",
            "args": [deploy_info.trading_strategy_module.address],
            "call": f"{safe_address}.enableModule({deploy_info.trading_strategy_module.address})",
        },
    ]

    return {
        "old_guard_address": old_guard_address,
        "new_guard_address": deploy_info.trading_strategy_module.address,
        "safe_address": safe_address,
        "vault_address": deploy_info.vault.address,
        "currently_enabled_modules": list(mods),
        "safe_transactions": transactions,
        "safe_abi": SAFE_ABI_STR,
    }


def _format_guard_migration_instructions(instructions: dict[str, Any]) -> str:
    """Format manual Safe migration instructions for persisted artefacts."""
    lines = [
        "Guard migration instructions",
        f"  Old guard address: {instructions['old_guard_address']}",
        f"  New guard address: {instructions['new_guard_address']}",
        f"  Safe address: {instructions['safe_address']}",
        f"  Vault address: {instructions['vault_address']}",
        f"  Currently enabled Safe modules: {instructions['currently_enabled_modules']}",
        "  Safe transactions needed:",
    ]

    for tx in instructions["safe_transactions"]:
        lines.append(f"  {tx['step']}. {tx['call']}")

    lines.append("  Safe ABI needed:")
    lines.append(instructions["safe_abi"])
    return "\n".join(lines)


def _augment_guard_only_artifacts(
    deploy_info,
    vault_adapter_address: str,
    *,
    text_payload: str,
    json_payload: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Attach Safe migration instructions to guard-only deploy artefacts."""
    instructions = _build_guard_migration_instructions(deploy_info, vault_adapter_address)
    augmented_json = dict(json_payload)
    augmented_json["Guard migration"] = instructions
    augmented_text = text_payload.rstrip() + "\n\n" + _format_guard_migration_instructions(instructions) + "\n"
    return augmented_text, augmented_json


def _log_guard_only_details(deploy_info, vault_adapter_address: str, logger) -> None:
    """Log manual guard replacement steps for guard-only mode."""
    instructions = _build_guard_migration_instructions(deploy_info, vault_adapter_address)
    logger.info("New guard deployed: %s", instructions["new_guard_address"])
    logger.info("Old guard address: %s", instructions["old_guard_address"])
    logger.info("Safe address: %s", instructions["safe_address"])
    logger.info("Vault address: %s", instructions["vault_address"])
    logger.info("Currently enabled Safe modules: %s", instructions["currently_enabled_modules"])
    logger.info("Safe transactions needed:")
    for tx in instructions["safe_transactions"]:
        logger.info("%d. %s", tx["step"], tx["call"])
    logger.info("Safe ABI needed: %s", instructions["safe_abi"])


def _confirm_deployment(*, simulate: bool, unit_testing: bool, verifier: str, etherscan_api_key: str | None, verifier_url: str | None, label: str = "vault") -> None:
    """Handle production deployment confirmation and verifier requirements."""
    if simulate or unit_testing:
        return

    if verifier == "etherscan" and not etherscan_api_key:
        raise RuntimeError("Etherscan API key needed for production deployments with etherscan verifier")
    if verifier == "blockscout" and not verifier_url:
        raise RuntimeError("Verifier URL needed for production deployments with blockscout verifier")

    confirm = input(f"Deploy {label}? [y/n] " if label != "vault" else "Ok [y/n]? ")
    if not confirm.lower().startswith("y"):
        print("Aborted")
        sys.exit(1)


@app.command()
def lagoon_deploy_vault(
    log_level: str = shared_options.log_level,
    json_rpc_binance: str | None = shared_options.json_rpc_binance,
    json_rpc_polygon: str | None = shared_options.json_rpc_polygon,
    json_rpc_avalanche: str | None = shared_options.json_rpc_avalanche,
    json_rpc_ethereum: str | None = shared_options.json_rpc_ethereum,
    json_rpc_base: str | None = shared_options.json_rpc_base,
    json_rpc_arbitrum: str | None = shared_options.json_rpc_arbitrum,
    json_rpc_anvil: str | None = shared_options.json_rpc_anvil,
    json_rpc_derive: str | None = shared_options.json_rpc_derive,
    json_rpc_arbitrum_sepolia: str | None = shared_options.json_rpc_arbitrum_sepolia,
    json_rpc_base_sepolia: str | None = shared_options.json_rpc_base_sepolia,
    json_rpc_hyperliquid: str | None = shared_options.json_rpc_hyperliquid,
    json_rpc_hyperliquid_testnet: str | None = shared_options.json_rpc_hyperliquid_testnet,
    json_rpc_monad: str | None = shared_options.json_rpc_monad,
    private_key: str = shared_options.private_key,

    # Vault options
    vault_record_file: Path = Option(..., envvar="VAULT_RECORD_FILE", help="Store vault data in this TXT file, paired with a JSON file."),
    fund_name: str | None = Option(None, envvar="FUND_NAME", help="On-chain name for the fund shares"),
    fund_symbol: str | None = Option(None, envvar="FUND_SYMBOL", help="On-chain token symbol for the fund shares"),
    denomination_asset: str | None = Option(None, envvar="DENOMINATION_ASSET", help="Stablecoin asset used for vault denomination"),
    multisig_owners: str | None = Option(None, callback=parse_comma_separated_list, envvar="MULTISIG_OWNERS", help="The list of acconts that are set to the cosigners of the Safe. The multisig threshold is number of cosigners - 1."),
    # terms_of_service_address: str | None = Option(None, envvar="TERMS_OF_SERVICE_ADDRESS", help="The address of the terms of service smart contract"),
    whitelisted_assets: str | None = Option(None, envvar="WHITELISTED_ASSETS", help="Space separarted list of ERC-20 addresses this vault can trade. Denomination asset does not need to be whitelisted separately."),
    any_asset: bool = Option(False, envvar="ANY_ASSET", help="Allow trading of any ERC-20 on Uniswap (unsecure)."),

    unit_testing: bool = shared_options.unit_testing,
    # production: bool = Option(False, envvar="PRODUCTION", help="Set production metadata flag true for the deployment."),
    simulate: bool = Option(False, envvar="SIMULATE", help="Simulate deployment using Anvil mainnet work, when doing manual deployment testing."),
    etherscan_api_key: str | None = Option(None, envvar="ETHERSCAN_API_KEY", help="Etherscan API key needed to verify contracts on a production deployment."),
    verifier: str = Option("etherscan", envvar="VERIFIER", help="Contract verifier to use: etherscan, blockscout, sourcify, oklink. Default: etherscan."),
    verifier_url: str | None = Option(None, envvar="VERIFIER_URL", help="Verifier API URL for Blockscout or custom verifiers (e.g., https://explorer.derive.xyz/api). Required when verifier=blockscout."),
    asset_manager_address: str | None = Option(None, callback=parse_comma_separated_list, envvar="ASSET_MANAGER", help="Ordered comma-separated list of vault asset manager addresses. If not provided, uses the deployer address (derived from PRIVATE_KEY). The first address becomes the Lagoon valuation manager; later addresses get guard sender permissions only."),
    one_delta: bool = Option(False, envvar="ONE_DELTA", help="Whitelist 1delta interaction with GuardV0 smart contract."),
    aave: bool = Option(False, envvar="AAVE", help="Whitelist Aave aUSDC deposits"),
    uniswap_v2: bool = Option(False, envvar="UNISWAP_V2", help="Whitelist Uniswap v2"),
    uniswap_v3: bool = Option(False, envvar="UNISWAP_V3", help="Whitelist Uniswap v3"),
    cowswap: bool = Option(False, envvar="COWSWAP", help="Whitelist CoW Swap"),
    erc_4626_vaults: str | None = Option(None, envvar="ERC_4626_VAULTS", help="Whitelist ERC-4626 vaults, a comma separated list of addresses"),
    verbose: bool = Option(False, envvar="VERBOSE", help="Extra verbosity with deploy commands"),
    performance_fee: int = Option(DEFAULT_PERFORMANCE_RATE, envvar="PERFORMANCE_FEE", help="Performance fee in BPS"),
    management_fee: int = Option(DEFAULT_MANAGEMENT_RATE, envvar="MANAGEMENT_FEE", help="Management fee in BPS"),
    guard_only: bool = Option(False, envvar="GUARD_ONLY", help="Deploys a new TradingStrategyModuleV0 guard with new settings. Lagoon multisig owners must then perform the transaction to enable this guard."),
    existing_vault_address: str | None = Option(None, envvar="EXISTING_VAULT_ADDRESS", help="When deploying a guard only, get the existing vault address."),
    existing_safe_address: str | None = Option(None, envvar="EXISTING_SAFE_ADDRESS", help="When deploying a guard only, get the existing safe address."),
    vault_adapter_address: str = shared_options.vault_adapter_address,
    cache_path: Path | None = shared_options.cache_path,
    strategy_file: Path | None = Option(None, envvar="STRATEGY_FILE", help="Strategy module for multichain deployment. When provided, uses translate_trading_universe_to_lagoon_config() to generate per-chain configs."),
    safe_salt_nonce: int | None = Option(None, envvar="SAFE_SALT_NONCE", help="CREATE2 salt nonce for deterministic Safe address across chains. Random if not given."),
    trading_strategy_api_key: str | None = shared_options.trading_strategy_api_key,
    hypersync_api_key: str | None = shared_options.hypersync_api_key,
    chain_name: str | None = shared_options.chain_name,
):
    """Deploy a Lagoon vault or modify the vault deployment.

    Deploys a new Lagoon vault, Safe and TradingStrategyModuleV0 guard for automated trading.

    When --strategy-file is provided, performs a multichain deployment using the
    strategy's trading universe to determine per-chain configurations (CCTP bridging,
    Uniswap v3 whitelisting, etc.).

    TODO: Heavily under development.
    """

    assert private_key, "PRIVATE_KEY not set"

    logger = setup_logging(log_level)

    # Prepare cache for token metadata storage
    # Use a fixed executor ID for this deployment command
    executor_id = "lagoon-deploy"
    cache_path = prepare_cache(executor_id, cache_path, unit_testing=unit_testing)
    token_cache = prepare_token_cache(cache_path, unit_testing=unit_testing)

    rpc_kwargs = collect_rpc_kwargs(
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_derive=json_rpc_derive,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
        json_rpc_monad=json_rpc_monad,
        chain_name=chain_name,
    )
    web3config = create_web3_config(
        **rpc_kwargs,
        simulate=simulate,
        mev_endpoint_disabled=True,
        simulate_http_timeout=(3.0, 90.0) if simulate else None,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("Vault deploy requires that you pass JSON-RPC connection to one of the networks")

    wallet_sync_web3 = next(iter(web3config.connections.values()))
    hot_wallet = create_hot_wallet(wallet_sync_web3, private_key)
    multisig_owners = _normalize_multisig_owners(multisig_owners, hot_wallet)
    asset_managers = _resolve_asset_managers(asset_manager_address, hot_wallet)
    assert not (strategy_file and denomination_asset), \
        f"Cannot use both --strategy-file and --denomination-asset. " \
        f"When --strategy-file is provided, the reserve asset is read from the strategy's create_trading_universe(). " \
        f"Remove --denomination-asset to use the strategy-file deployment path."

    # Strategy-file deployment path: use strategy file to generate per-chain configs
    # via translate_trading_universe_to_lagoon_config(). Handles both multichain
    # and single-chain strategies — protocol detection (GMX, CCTP, Uniswap v3, etc.)
    # is always driven by the strategy's trading universe.
    if strategy_file:
        _deploy_multichain(
            web3config=web3config,
            hot_wallet=hot_wallet,
            asset_managers=asset_managers,
            strategy_file=strategy_file,
            safe_salt_nonce=safe_salt_nonce,
            fund_name=fund_name,
            fund_symbol=fund_symbol,
            multisig_owners=multisig_owners,
            vault_record_file=vault_record_file,
            simulate=simulate,
            unit_testing=unit_testing,
            logger=logger,
            any_asset=any_asset,
            trading_strategy_api_key=trading_strategy_api_key,
            hypersync_api_key=hypersync_api_key,
            etherscan_api_key=etherscan_api_key,
            verifier=verifier,
            verifier_url=verifier_url,
            guard_only=guard_only,
            existing_vault_address=existing_vault_address,
            existing_safe_address=existing_safe_address,
        )
        web3config.close()
        logger.info("All ok.")
        return

    # Single-chain deployment path (original flow)
    choose_single_chain(web3config)

    web3 = web3config.get_default()
    chain_id = ChainId(web3.eth.chain_id)

    logger.info("Connected to chain %s", chain_id.name)

    hot_wallet.sync_nonce(web3)
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))

    assert not whitelisted_assets, "whitelisted_assets: Not implemented"
    whitelisted_asset_details = []

    # Check the chain is online
    logger.info(f"  Chain id is {web3.eth.chain_id:,}")
    logger.info(f"  Latest block is {web3.eth.block_number:,}")

    lagoon_chain_config = get_lagoon_chain_config(chain_id)

    if existing_vault_address:
        existing_vault = create_vault_instance(
            web3,
            existing_vault_address,
            {ERC4626Feature.lagoon_like},
        )
        logger.info("Deploying for existing vault %s", existing_vault.name)
        denomination_token = existing_vault.denomination_token
    else:
        denomination_token = fetch_erc20_details(
            web3,
            denomination_asset,
        )

    if simulate:
        logger.info("Simulation deployment")
    else:
        logger.info("Ready to deploy")

    chain_slug = chain_id.get_slug()
    log_deployment_preflight_report(
        hot_wallet=hot_wallet,
        chain_web3={chain_slug: web3},
        fund_name=fund_name,
        fund_symbol=fund_symbol,
        asset_managers=asset_managers,
        multisig_owners=multisig_owners,
        performance_fee=performance_fee,
        management_fee=management_fee,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        denomination_token=denomination_token,
        any_asset=any_asset,
        whitelisted_asset_details=whitelisted_asset_details,
        uniswap_v2=uniswap_v2,
        uniswap_v3=uniswap_v3,
        one_delta=one_delta,
        aave=aave,
        erc_4626_vaults=erc_4626_vaults,
        lagoon_chain_config=lagoon_chain_config,
        simulate=simulate,
        logger=logger,
    )

    _confirm_deployment(
        simulate=simulate,
        unit_testing=unit_testing,
        verifier=verifier,
        etherscan_api_key=etherscan_api_key,
        verifier_url=verifier_url,
    )

    # The first asset manager remains the Lagoon valuation manager.
    # Any later asset managers only receive guard sender permissions.
    parameters = LagoonDeploymentParameters(
        underlying=denomination_token.address,
        name=fund_name,
        symbol=fund_symbol,
        performanceRate=performance_fee,
        managementRate=management_fee,

    )

    chain_slug = chain_id.get_slug()

    if uniswap_v2:
        uniswap_v2_deployment = fetch_deployment(
            web3,
            factory_address=UNISWAP_V2_DEPLOYMENTS[chain_slug]["factory"],
            router_address=UNISWAP_V2_DEPLOYMENTS[chain_slug]["router"],
            init_code_hash=UNISWAP_V2_DEPLOYMENTS[chain_slug]["init_code_hash"],
        )
    else:
        uniswap_v2_deployment = None

    if uniswap_v3:
        chain_slug = chain_id.get_slug()
        deployment_data = UNISWAP_V3_DEPLOYMENTS[chain_slug]
        uniswap_v3_deployment= fetch_deployment_uni_v3(
            web3,
            factory_address=deployment_data["factory"],
            router_address=deployment_data["router"],
            position_manager_address=deployment_data["position_manager"],
            quoter_address=deployment_data["quoter"],
            quoter_v2=deployment_data.get("quoter_v2", False),
            router_v2=deployment_data.get("router_v2", False),
        )
    else:
        uniswap_v3_deployment = None

    if aave:
        chain_slug = chain_id.get_slug()
        deployment_data = AAVE_V3_DEPLOYMENTS[chain_slug]
        assert "ausdc" in deployment_data, f"No aUSDC configuration: {AAVE_V3_DEPLOYMENTS}"
        aave_v3_deployment = fetch_aave_deployment(
            web3,
            pool_address=deployment_data["pool"],
            data_provider_address=deployment_data["data_provider"],
            oracle_address=deployment_data["oracle"],
            ausdc_address=deployment_data["ausdc"],
        )
    else:
        aave_v3_deployment = None

    # Scanning ERC-4626 vaults on a startup for token details takes a long time
    # Token cache already prepared at the start of the command
    logger.info("Using token cache at %s", token_cache.filename)

    if erc_4626_vaults:
        erc_4626_vault_addresses = [Web3.to_checksum_address(a.strip()) for a in erc_4626_vaults.split(",")]
        erc_4626_vaults = []
        for addr in erc_4626_vault_addresses:
            logger.info("Resolving ERC-4626 vault at %s", addr)
            vault = cast(ERC4626Vault, create_vault_instance(web3, addr, token_cache=token_cache))
            assert vault.is_valid(), f"Invalid ERC-4626 vault at {addr}"
            logger.info("Preparing vault %s for whitelisting", vault.name)
            erc_4626_vaults.append(vault)

    # Capture block before deployment so the report can find guard config events
    # (deploy_info.block_number is set AFTER deployment, missing all events)
    pre_deploy_block = web3.eth.block_number

    deploy_info = deploy_automated_lagoon_vault(
        web3=web3,
        deployer=hot_wallet,
        asset_managers=asset_managers,
        parameters=parameters,
        safe_owners=multisig_owners,
        safe_threshold=_calculate_safe_threshold(multisig_owners),
        uniswap_v2=uniswap_v2_deployment,
        uniswap_v3=uniswap_v3_deployment,
        aave_v3=aave_v3_deployment,
        any_asset=any_asset,
        use_forge=True,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        guard_only=guard_only,
        existing_vault_address=existing_vault_address,
        existing_safe_address=existing_safe_address,
        erc_4626_vaults=erc_4626_vaults,
        factory_contract=lagoon_chain_config.factory_contract,
        from_the_scratch=lagoon_chain_config.from_the_scratch,
        cowswap=cowswap,
    )

    text_payload = deploy_info.pformat()
    json_payload = deploy_info.get_deployment_data()
    if guard_only:
        text_payload, json_payload = _augment_guard_only_artifacts(
            deploy_info,
            vault_adapter_address,
            text_payload=text_payload,
            json_payload=json_payload,
        )

    _write_deployment_artifacts(
        vault_record_file,
        text_payload=text_payload,
        json_payload=json_payload,
        simulate=simulate,
        logger=logger,
    )

    logger.info("Token cache %s contains %d entries", token_cache.filename, len(token_cache))

    if not guard_only:
        logger.info("Lagoon deployed:\n%s", deploy_info.pformat())
    else:
        _log_guard_only_details(deploy_info, vault_adapter_address, logger)


    # Print deployment guard configuration report
    _, markdown_report = print_deployment_report(
        safe_address=deploy_info.safe_address or deploy_info.safe.address,
        module_address=deploy_info.trading_strategy_module.address,
        web3=web3,
        hypersync_api_key=hypersync_api_key,
        simulate=simulate,
        from_block=pre_deploy_block,
    )

    _write_markdown_report(vault_record_file, markdown_report, logger)

    web3config.close()

    logger.info("All ok.")


def _deploy_multichain(
    web3config,
    hot_wallet: HotWallet,
    asset_managers: list[str],
    strategy_file: Path,
    safe_salt_nonce: int | None,
    fund_name: str | None,
    fund_symbol: str | None,
    multisig_owners: list[str] | None,
    vault_record_file: Path,
    simulate: bool,
    unit_testing: bool,
    logger,
    any_asset: bool = False,
    trading_strategy_api_key: str | None = None,
    hypersync_api_key: str | None = None,
    etherscan_api_key: str | None = None,
    verifier: str = "etherscan",
    verifier_url: str | None = None,
    guard_only: bool = False,
    existing_vault_address: str | None = None,
    existing_safe_address: str | None = None,
):
    """Deploy multichain Lagoon vault from a strategy file.

    Uses the strategy's trading universe to determine per-chain configurations.
    """

    if safe_salt_nonce is None:
        safe_salt_nonce = random.randint(1, 2**32)
        logger.info("Generated random safe_salt_nonce: %d", safe_salt_nonce)

    multisig_owners = _normalize_multisig_owners(multisig_owners, hot_wallet)
    safe_threshold = _calculate_safe_threshold(multisig_owners)

    # Create TradingStrategy client if API key is available
    # (needed by strategies that fetch exchange/pair data from the API)
    client = None
    if trading_strategy_api_key:
        from tradingstrategy.client import Client
        client = Client.create_live_client(api_key=trading_strategy_api_key)

    # Load strategy module and create trading universe
    mod = read_strategy_module(strategy_file)
    universe = call_create_trading_universe(
        mod.create_trading_universe,
        client=client,
        universe_options=mod.get_universe_options(),
        execution_context=one_off_execution_context,
    )

    # Build chain_web3 mapping: {chain_slug: web3}
    chain_web3 = {}
    for chain_id, web3 in web3config.connections.items():
        chain_web3[chain_id.get_slug()] = web3

    logger.info("Multichain deployment: chains=%s, strategy=%s", list(chain_web3.keys()), strategy_file)

    # Sync hot wallet nonce on each chain
    for slug, web3 in chain_web3.items():
        hot_wallet.sync_nonce(web3)
        web3.middleware_onion.add(construct_sign_and_send_raw_middleware(hot_wallet.account))
        logger.info("  Chain %s: block %d", slug, web3.eth.block_number)

    # Generate per-chain configs from the strategy universe
    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3=chain_web3,
        asset_managers=asset_managers,
        safe_owners=multisig_owners,
        safe_threshold=safe_threshold,
        safe_salt_nonce=safe_salt_nonce,
        fund_name=fund_name or "Strategy Vault",
        fund_symbol=fund_symbol or "CSV",
        any_asset=any_asset,
        guard_only=guard_only,
        existing_vault_address=existing_vault_address,
        existing_safe_address=existing_safe_address,
    )

    chain_word = "chain" if len(configs) == 1 else "chains"
    logger.info("Generated configs for %d %s:", len(configs), chain_word)

    log_deployment_preflight_report(
        hot_wallet=hot_wallet,
        chain_web3=chain_web3,
        fund_name=fund_name or "Strategy Vault",
        fund_symbol=fund_symbol or "CSV",
        asset_managers=asset_managers,
        multisig_owners=multisig_owners,
        performance_fee=configs[next(iter(configs))].parameters.performanceRate,
        management_fee=configs[next(iter(configs))].parameters.managementRate,
        etherscan_api_key=etherscan_api_key,
        verifier=verifier,
        verifier_url=verifier_url,
        chain_configs=configs,
        simulate=simulate,
        logger=logger,
    )

    _confirm_deployment(
        simulate=simulate,
        unit_testing=unit_testing,
        verifier=verifier,
        etherscan_api_key=etherscan_api_key,
        verifier_url=verifier_url,
        label="multichain vault" if len(configs) > 1 else "vault",
    )

    # Capture block before deployment so the report can find guard config events
    pre_deploy_blocks = {slug: w3.eth.block_number for slug, w3 in chain_web3.items()}

    # Deploy across all chains
    result = deploy_multichain_lagoon_vault(
        chain_web3=chain_web3,
        deployer=hot_wallet.account,
        chain_configs=configs,
    )

    logger.info("Deployment complete")
    logger.info("Safe address: %s", result.deployments[next(iter(result.deployments))].safe_address)

    for slug, dep in result.deployments.items():
        kind = "satellite" if dep.is_satellite else "source"
        logger.info("Lagoon deployed on %s (%s):\n%s", slug, kind, dep.pformat())

    # Build chain_id-keyed mappings for the multichain report
    chain_id_web3: dict[int, Web3] = {}
    from_block_by_chain_id: dict[int, int] = {}
    for slug, w3 in chain_web3.items():
        cid = w3.eth.chain_id
        chain_id_web3[cid] = w3
        from_block_by_chain_id[cid] = pre_deploy_blocks.get(slug, 0)

    # Generate multichain deployment report (single scan with follow_cctp=True)
    unicode_report, markdown_report = generate_multichain_deployment_report(
        safe_address=result.safe_address,
        chain_web3=chain_id_web3,
        deployment_result=result,
        hypersync_api_key=hypersync_api_key,
        simulate=simulate,
        from_block=from_block_by_chain_id,
    )

    text_payload, json_payload = _build_multichain_artifact_payload(
        result,
        safe_salt_nonce,
        configs,
        unicode_report,
    )
    _write_deployment_artifacts(
        vault_record_file,
        text_payload=text_payload,
        json_payload=json_payload,
        simulate=simulate,
        logger=logger,
    )

    _write_markdown_report(vault_record_file, markdown_report, logger)


SAFE_ABI_STR = """
[
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "module",
          "type": "address"
        }
      ],
      "name": "enableModule",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "address",
          "name": "prevModule",
          "type": "address"
        },
        {
          "internalType": "address",
          "name": "module",
          "type": "address"
        }
      ],
      "name": "disableModule",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
]
"""
