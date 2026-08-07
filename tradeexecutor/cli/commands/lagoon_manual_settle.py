"""Inspect and simulate a manual Lagoon Safe settlement."""

import json
from typing import Any

import typer
from web3 import Web3

from eth_defi.abi import encode_function_call
from tradeexecutor.cli.bootstrap import create_web3_config
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import load_lagoon_vault
from tradeexecutor.cli.log import setup_logging


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
            f"Vault {vault.address} needs _newTotalAssets for {function_name}, but its ABI has no readable newTotalAssets() value. "
            "Pass --new-total-assets from the GuardV0 manual-settlement alert."
        )
    assert len(inputs) == 1 and inputs[0]["type"] == "uint256", (
        f"Unexpected {function_name} ABI for {vault.address}: {abi}"
    )
    return abi, getattr(vault.vault_contract.functions, function_name)(new_total_assets_raw)


def _fetch_pending_new_total_assets(vault, block_number: int) -> int | None:
    """Read the valuation waiting to be settled, if this Lagoon version exposes it."""
    has_getter = any(
        item.get("type") == "function" and item.get("name") == "newTotalAssets" and not item["inputs"]
        for item in vault.vault_contract.abi
    )
    if not has_getter:
        return None
    new_total_assets = vault.vault_contract.functions.newTotalAssets().call(block_identifier=block_number)
    return None if new_total_assets == NO_PENDING_NAV else new_total_assets


def inspect_manual_lagoon_settlement(vault, new_total_assets_raw: int | None = None) -> dict[str, Any]:
    """Read one on-chain Lagoon queue snapshot and simulate Safe settlement.

    The tool reads the pending ``newTotalAssets`` value where the deployed ABI
    exposes it. Otherwise the caller must supply the exact raw value from the
    executor's GuardV0 manual-settlement alert. It never calculates or posts a
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

    if new_total_assets_raw is not None:
        assert new_total_assets_raw >= 0, "new_total_assets_raw cannot be negative"
    pending_new_total_assets_raw = _fetch_pending_new_total_assets(vault, block_number)
    if new_total_assets_raw is None:
        new_total_assets_raw = pending_new_total_assets_raw
    elif pending_new_total_assets_raw is not None and new_total_assets_raw != pending_new_total_assets_raw:
        raise RuntimeError(
            f"--new-total-assets ({new_total_assets_raw}) does not match the on-chain newTotalAssets() "
            f"value ({pending_new_total_assets_raw}) at block {block_number}"
        )
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
        "pending_new_total_assets_raw": pending_new_total_assets_raw,
        "new_total_assets_raw": new_total_assets_raw,
        "new_total_assets_source": "on-chain newTotalAssets()" if pending_new_total_assets_raw is not None else "command input; copy the exact raw value from the GuardV0 manual-settlement alert",
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


@app.command()
@shared_options.with_json_rpc_options(include_chain_name=True)
def lagoon_manual_settle(
    vault_address: str = shared_options.required_option(shared_options.vault_address),
    new_total_assets: int | None = typer.Option(None, "--new-total-assets", envvar="NEW_TOTAL_ASSETS", help="Exact raw NAV if this Lagoon version does not expose newTotalAssets()"),
    log_level: str = shared_options.log_level,
    chain_name: str | None = shared_options.chain_name,
    rpc_kwargs: dict | None = None,
):
    """Display and safely simulate the required direct Safe settlement transaction.

    This is a read-only operational tool for a GuardV0-capped Lagoon queue.
    It does not post a valuation, create a Safe transaction, or broadcast one.
    The command reads ``newTotalAssets()`` when available; otherwise pass
    ``--new-total-assets`` from the GuardV0 manual-settlement alert. The
    estimate checks only the direct vault call with the Safe as sender.
    """
    setup_logging(log_level=log_level)
    # The shared decorator filters rpc_kwargs to chain_name before this command
    # is called, leaving a single connection when --chain-name is supplied.
    web3config = create_web3_config(**rpc_kwargs)
    if not web3config.has_any_connection():
        raise RuntimeError("Pass a JSON-RPC connection for the Lagoon vault chain")
    if len(web3config.connections) != 1:
        raise RuntimeError("Pass exactly one JSON-RPC connection or select one with --chain-name")

    web3config.choose_single_chain()
    try:
        vault = load_lagoon_vault(web3config.get_default(), vault_address)
        report = inspect_manual_lagoon_settlement(vault, new_total_assets)
        typer.echo(json.dumps(report, indent=2))
    finally:
        web3config.close()

    if report["settlement_required"] and not report["target_call_simulation"]["succeeds"]:
        raise typer.Exit(code=1)
