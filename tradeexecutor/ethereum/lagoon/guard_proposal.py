"""Prepare and submit an atomic Lagoon guard replacement through a Safe."""

from dataclasses import dataclass

from eth_account import Account
from hexbytes import HexBytes
from safe_eth.eth.ethereum_network import EthereumNetwork
from safe_eth.safe import Safe
from safe_eth.safe.api.transaction_service_api.transaction_service_api import TransactionServiceApi
from safe_eth.safe.multi_send import MultiSend, MultiSendOperation, MultiSendTx
from web3 import Web3

from eth_defi.abi import ONE_ADDRESS_STR
from eth_defi.safe.deployment import fetch_safe_deployment


GUARD_GOVERNANCE_ABI = [{
    "inputs": [],
    "name": "getGovernanceAddress",
    "outputs": [{"type": "address", "name": ""}],
    "stateMutability": "view",
    "type": "function",
}]


@dataclass(frozen=True, slots=True)
class GuardProposalContext:
    """Validated chain-specific Safe and batch contract."""

    web3: Web3
    safe: Safe
    multisend: MultiSend
    old_guard_address: str
    network: EthereumNetwork


def prepare_guard_proposal(
    web3: Web3,
    safe_address: str,
    private_key: str,
    *,
    expected_old_guard_address: str | None = None,
) -> GuardProposalContext:
    """Check proposal prerequisites before deploying a replacement guard."""
    safe = fetch_safe_deployment(web3, Web3.to_checksum_address(safe_address))
    version = safe.retrieve_version()
    if tuple(int(part) for part in version.split("+")[0].split(".")[:3]) < (1, 3, 0):
        raise ValueError(f"Safe {safe.address} version {version} does not support MultiSendCallOnly")
    modules = safe.retrieve_modules()
    if len(modules) != 1:
        raise ValueError(f"Safe {safe.address} must have exactly one enabled guard module, got {modules}")
    old_guard_address = Web3.to_checksum_address(modules[0])
    if expected_old_guard_address and old_guard_address != Web3.to_checksum_address(expected_old_guard_address):
        raise ValueError(f"Safe {safe.address} has guard {old_guard_address}, expected {expected_old_guard_address}")
    old_guard = web3.eth.contract(address=old_guard_address, abi=GUARD_GOVERNANCE_ABI)
    governance_address = old_guard.functions.getGovernanceAddress().call()
    if Web3.to_checksum_address(governance_address) != safe.address:
        raise ValueError(f"Module {old_guard_address} is not governed by Safe {safe.address}")
    proposer = Account.from_key(private_key).address
    if proposer.lower() not in {owner.lower() for owner in safe.retrieve_owners()}:
        raise ValueError(f"Deployer {proposer} is not an owner of Safe {safe.address}")
    network = EthereumNetwork(web3.eth.chain_id)
    if network not in TransactionServiceApi.NETWORK_SHORTNAME:
        raise ValueError(f"No Safe Transaction Service for chain {web3.eth.chain_id}")
    multisend = MultiSend(safe.ethereum_client, call_only=True)
    if not web3.eth.get_code(multisend.address):
        raise ValueError(f"MultiSendCallOnly is not deployed at {multisend.address} on chain {web3.eth.chain_id}")
    return GuardProposalContext(web3, safe, multisend, old_guard_address, network)


def build_guard_migration_transaction(context: GuardProposalContext, new_guard_address: str):
    """Build one Safe transaction that disables the old guard and enables the new one."""
    safe = context.safe
    new_guard_address = Web3.to_checksum_address(new_guard_address)
    if new_guard_address == context.old_guard_address:
        raise ValueError("Replacement guard must differ from the old guard")
    if not context.web3.eth.get_code(new_guard_address):
        raise ValueError(f"Replacement guard {new_guard_address} has no code")
    new_guard = context.web3.eth.contract(address=new_guard_address, abi=GUARD_GOVERNANCE_ABI)
    governance_address = new_guard.functions.getGovernanceAddress().call()
    if Web3.to_checksum_address(governance_address) != safe.address:
        raise ValueError(f"Replacement guard {new_guard_address} is not governed by Safe {safe.address}")
    modules = safe.retrieve_modules()
    if len(modules) != 1 or Web3.to_checksum_address(modules[0]) != context.old_guard_address:
        raise ValueError(f"Safe {safe.address} modules changed since guard proposal preflight: {modules}")
    calls = [
        MultiSendTx(
            MultiSendOperation.CALL,
            safe.address,
            0,
            HexBytes(safe.contract.functions.disableModule(ONE_ADDRESS_STR, context.old_guard_address)._encode_transaction_data()),
        ),
        MultiSendTx(
            MultiSendOperation.CALL,
            safe.address,
            0,
            HexBytes(safe.contract.functions.enableModule(new_guard_address)._encode_transaction_data()),
        ),
    ]
    nonce = safe.retrieve_nonce()
    return safe.build_multisig_tx(
        to=context.multisend.address,
        value=0,
        data=context.multisend.build_tx_data(calls),
        operation=MultiSendOperation.DELEGATE_CALL.value,
        safe_nonce=nonce,
    )


def describe_pending_guard_migration(context: GuardProposalContext, new_guard_address: str) -> dict:
    """Persist the unsigned transaction fields before contacting the Service."""
    safe_tx = build_guard_migration_transaction(context, new_guard_address)
    return _describe_transaction(context, safe_tx, status="pending")


def _describe_transaction(context: GuardProposalContext, safe_tx, *, status: str) -> dict:
    """Serialise public Safe fields for recovery and operator review."""
    return {
        "status": status,
        "to": context.multisend.address,
        "value": 0,
        "operation": MultiSendOperation.DELEGATE_CALL.value,
        "data": Web3.to_hex(safe_tx.data),
        "nonce": safe_tx.safe_nonce,
    }


def submit_guard_migration(
    context: GuardProposalContext,
    new_guard_address: str,
    private_key: str,
    *,
    safe_api_key: str | None = None,
    expected_proposal: dict | None = None,
) -> dict:
    """Submit or recognise the identical pending Safe migration proposal."""
    safe_tx = build_guard_migration_transaction(context, new_guard_address)
    if expected_proposal:
        actual = _describe_transaction(context, safe_tx, status="pending")
        for field in ("to", "value", "operation", "data", "nonce"):
            if field in expected_proposal and actual[field] != expected_proposal[field]:
                raise RuntimeError(f"Safe proposal {field} changed since deployment; inspect the Safe before retrying")
    safe_tx.sign(private_key)
    tx_hash = Web3.to_hex(safe_tx.safe_tx_hash)
    service = TransactionServiceApi(
        network=context.network,
        ethereum_client=context.safe.ethereum_client,
        api_key=safe_api_key,
    )
    pending = []
    while True:
        page = service.get_transactions(
            context.safe.address,
            executed=False,
            nonce=safe_tx.safe_nonce,
            limit=100,
            offset=len(pending),
        )
        pending.extend(page)
        if len(page) < 100:
            break
    if context.safe.retrieve_nonce() != safe_tx.safe_nonce:
        raise RuntimeError(f"Safe {context.safe.address} nonce changed while preparing the guard proposal")
    for transaction in pending:
        existing_hash = str(transaction.get("safeTxHash") or transaction.get("contractTransactionHash") or "")
        if existing_hash.lower() != tx_hash.lower():
            raise RuntimeError(f"Safe {context.safe.address} nonce {safe_tx.safe_nonce} already has a different pending proposal")
    if not pending:
        service.post_transaction(safe_tx)
    short_name = TransactionServiceApi.NETWORK_SHORTNAME[context.network]
    safe_address = context.safe.address
    return {
        **_describe_transaction(context, safe_tx, status="submitted"),
        "safe_tx_hash": tx_hash,
        "url": f"https://app.safe.global/transactions/tx?safe={short_name}:{safe_address}&id=multisig_{safe_address}_{tx_hash}",
    }
