"""Inspect and simulate a manual Lagoon Safe settlement."""

import json
from pathlib import Path
from typing import Any

import typer
from hexbytes import HexBytes
from web3 import Web3

from eth_defi.abi import encode_function_call
from eth_defi.erc_4626.settlement_events import fetch_vault_settlement_logs
from eth_defi.provider.anvil import is_anvil
from tradeexecutor.cli.bootstrap import configure_default_chain, create_web3_config, prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import load_lagoon_vault, resolve_state_store
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.strategy.strategy_module import read_strategy_module


NO_PENDING_NAV = 2**256 - 1


def _build_settlement_call(vault, function_name: str, new_total_assets_raw: int | None) -> tuple[dict[str, Any], Any]:
    """Build the deployed Lagoon version's direct Safe settlement call."""
    abis = [item for item in vault.vault_contract.abi if item.get("type") == "function" and item.get("name") == function_name]
    if len(abis) != 1:
        raise RuntimeError(f"Expected one {function_name} ABI for {vault.address}, got {abis}")

    abi = abis[0]
    inputs = abi["inputs"]
    if len(inputs) == 0:
        return abi, getattr(vault.vault_contract.functions, function_name)()

    if new_total_assets_raw is None:
        raise RuntimeError(
            f"Vault {vault.address} needs _newTotalAssets for {function_name}, but no pending valuation was found on-chain. "
            "Run a normal Lagoon treasury sync to post a fresh valuation before preparing manual settlement."
        )
    assert len(inputs) == 1 and inputs[0]["type"] == "uint256", (
        f"Unexpected {function_name} ABI for {vault.address}: {abi}"
    )
    return abi, getattr(vault.vault_contract.functions, function_name)(new_total_assets_raw)


def _fetch_pending_new_total_assets(vault, deployment_block: int, block_number: int) -> tuple[int | None, str]:
    """Read the valuation waiting to be settled from contract state or events."""
    has_getter = any(
        item.get("type") == "function" and item.get("name") == "newTotalAssets" and not item["inputs"]
        for item in vault.vault_contract.abi
    )
    if has_getter:
        new_total_assets = vault.vault_contract.functions.newTotalAssets().call(block_identifier=block_number)
        value = None if new_total_assets == NO_PENDING_NAV else new_total_assets
        return value, "newTotalAssets()"

    new_nav_topic = Web3.keccak(text="NewTotalAssetsUpdated(uint256)")
    settled_nav_topic = Web3.keccak(text="TotalAssetsUpdated(uint256)")
    logs = fetch_vault_settlement_logs(
        web3=vault.web3,
        address=vault.address,
        topic0_list=[Web3.to_hex(new_nav_topic), Web3.to_hex(settled_nav_topic)],
        start_block=deployment_block,
        end_block=block_number,
        use_hypersync=False if is_anvil(vault.web3) else None,
    )
    pending_new_total_assets = None
    for log in sorted(logs, key=lambda item: (item["blockNumber"], item["logIndex"])):
        if HexBytes(log["topics"][0]) == new_nav_topic:
            value = int.from_bytes(HexBytes(log["data"]), byteorder="big")
            pending_new_total_assets = None if value == NO_PENDING_NAV else value
        else:
            pending_new_total_assets = None
    return pending_new_total_assets, "NewTotalAssetsUpdated/TotalAssetsUpdated events"


def inspect_manual_lagoon_settlement(vault, deployment_block: int) -> dict[str, Any]:
    """Read one on-chain Lagoon queue snapshot and simulate Safe settlement.

    The tool reads the pending ``newTotalAssets`` value from the deployed
    contract getter or its valuation events. It never calculates or posts a
    valuation.
    """
    web3: Web3 = vault.web3
    block_number = web3.eth.block_number
    denomination_token = vault.denomination_token
    flow_manager = vault.get_flow_manager()

    pending_deposit = flow_manager.fetch_pending_deposit(block_number)
    pending_redemption_shares = flow_manager.fetch_pending_redemption(block_number)
    onchain_nav = vault.fetch_nav(block_identifier=block_number)
    safe_address = Web3.to_checksum_address(vault.safe_address)
    safe_balance = denomination_token.fetch_balance_of(safe_address, block_identifier=block_number)

    new_total_assets_raw, new_total_assets_source = _fetch_pending_new_total_assets(vault, deployment_block, block_number)
    if pending_deposit > 0:
        function_name = "settleDeposit"
    elif pending_redemption_shares > 0:
        function_name = "settleRedeem"
    else:
        return {
            "chain_id": web3.eth.chain_id,
            "block_number": block_number,
            "vault": vault.address,
            "safe": safe_address,
            "pending_deposit": str(pending_deposit),
            "pending_redemption_shares": str(pending_redemption_shares),
            "settlement_required": False,
            "warnings": ["There are no pending deposits or redemptions, so no Safe settlement transaction is needed."],
        }

    abi, settle_call = _build_settlement_call(vault, function_name, new_total_assets_raw)
    inputs = abi["inputs"]
    contract_inputs_values = {inputs[0]["name"]: str(new_total_assets_raw)} if inputs else {}
    calldata = Web3.to_hex(encode_function_call(settle_call))
    safe_transaction_fields: dict[str, Any] = {
        "to": vault.address,
        "value": "0",
        "operation": 0,
        "contractMethod": {
            "inputs": inputs,
            "name": abi["name"],
            "payable": abi["stateMutability"] == "payable",
        },
        "contractInputsValues": contract_inputs_values,
        "data": calldata,
    }
    transaction: dict[str, Any] = {
        "from": safe_address,
        "to": vault.address,
        "data": calldata,
        "value": 0,
    }

    try:
        simulation: dict[str, Any] = {
            "succeeds": True,
            "estimated_gas": web3.eth.estimate_gas(transaction, block_identifier=block_number),
        }
    except Exception as e:
        simulation = {
            "succeeds": False,
            "result": "inconclusive: the estimate may have reverted or the RPC request may have failed",
            "error": str(e),
        }

    return {
        "chain_id": web3.eth.chain_id,
        "block_number": block_number,
        "vault": vault.address,
        "safe": safe_address,
        "denomination_token": {
            "address": denomination_token.address,
            "symbol": denomination_token.symbol,
            "decimals": denomination_token.decimals,
        },
        "onchain_nav": str(onchain_nav),
        "new_total_assets_raw": new_total_assets_raw,
        "new_total_assets_source": new_total_assets_source,
        "pending_deposit": str(pending_deposit),
        "pending_redemption_shares": str(pending_redemption_shares),
        "safe_balance": str(safe_balance),
        "settlement_required": True,
        "settlement_abi": abi,
        "gnosis_safe_transaction_fields": safe_transaction_fields,
        "target_call_simulation": simulation,
        "warnings": [
            "A successful estimate only means the direct vault target call did not revert at this block; it does not validate the NAV.",
            "Enter gnosis_safe_transaction_fields manually in Safe Transaction Builder; this output is not an importable Builder batch file.",
            "settleDeposit may settle pending redemptions when the Safe has enough assets at the new NAV; a redemption-only queue uses settleRedeem.",
            "Safe signatures, owner policy and guards are not simulated.",
        ],
    }


def format_manual_settlement_instructions(report: dict[str, Any]) -> str:
    """Format operator instructions for a direct Gnosis Safe transaction."""
    if not report["settlement_required"]:
        return (
            "No manual Lagoon settlement is needed.\n"
            f"Vault: {report['vault']}\n"
            f"Pending deposits: {report['pending_deposit']}\n"
            f"Pending redemption shares: {report['pending_redemption_shares']}"
        )

    transaction = report["gnosis_safe_transaction_fields"]
    method = transaction["contractMethod"]["name"]
    inputs = transaction["contractInputsValues"]
    simulation = report["target_call_simulation"]
    lines = [
        "Manual Lagoon settlement via Gnosis Safe",
        "",
        f"Queue at block: {report['block_number']}",
        f"Pending deposits: {report['pending_deposit']} {report['denomination_token']['symbol']}",
        f"Pending redemption shares: {report['pending_redemption_shares']}",
        f"Safe balance: {report['safe_balance']} {report['denomination_token']['symbol']}",
        "",
        "Set up the Safe transaction:",
        f"1. Open Safe {report['safe']} on chain {report['chain_id']}.",
        "2. Create a new Contract interaction transaction.",
        f"3. Set the target contract to {transaction['to']}.",
        "4. Set value to 0 and operation to Call (0).",
        f"5. Paste this ABI: {json.dumps([report['settlement_abi']], separators=(',', ':'))}",
        f"6. Select {method} and enter: {json.dumps(inputs, separators=(',', ':'))}.",
        f"7. Confirm the calldata is {transaction['data']}.",
        "8. Re-simulate in Safe, collect the required signatures, then execute.",
        "",
        f"Direct target-call simulation succeeds: {simulation['succeeds']}",
    ]
    if simulation.get("estimated_gas") is not None:
        lines.append(f"Estimated target-call gas: {simulation['estimated_gas']}")
    if simulation.get("error"):
        lines.append(f"Simulation error: {simulation['error']}")
    lines.extend((
        "",
        "This preflight does not validate the NAV, Safe signatures, owner policy or Safe guards.",
        "Do not execute if the Safe simulation differs from these values.",
    ))
    return "\n".join(lines)


@app.command()
@shared_options.with_json_rpc_options()
def lagoon_manual_settle(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Path | None = shared_options.state_file,
    vault_address: str | None = shared_options.vault_address,
    log_level: str = shared_options.log_level,
    rpc_kwargs: dict | None = None,
):
    """Print instructions for the required direct Safe settlement transaction.

    This is a read-only operational tool for a GuardV0-capped Lagoon queue.
    It does not post a valuation, create a Safe transaction, or broadcast one.
    It reads all configuration from the normal executor environment and state.
    """
    id = prepare_executor_id(id, strategy_file)
    setup_logging(log_level=log_level)
    mod = read_strategy_module(strategy_file)
    web3config = create_web3_config(**rpc_kwargs)
    if not web3config.has_any_connection():
        raise RuntimeError("Pass a JSON-RPC connection for the Lagoon vault chain")

    configure_default_chain(web3config, mod)
    try:
        state_path, store = resolve_state_store(id, state_file)
        assert not store.is_pristine(), f"Strategy state file does not exist: {state_path}"
        state = store.load()
        deployment = state.sync.deployment
        vault_address = vault_address or deployment.address
        assert vault_address, "Lagoon vault address is missing from VAULT_ADDRESS and strategy state"
        deployment_block = deployment.block_number
        assert deployment_block is not None, "Lagoon deployment block is missing from strategy state"
        vault = load_lagoon_vault(web3config.get_default(), vault_address)
        report = inspect_manual_lagoon_settlement(vault, deployment_block)
        typer.echo(format_manual_settlement_instructions(report))
    finally:
        web3config.close()

    if report["settlement_required"] and not report["target_call_simulation"]["succeeds"]:
        raise typer.Exit(code=1)
