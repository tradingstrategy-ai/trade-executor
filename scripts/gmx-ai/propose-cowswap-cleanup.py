"""Propose the one-off GMX AI stranded-token clean-up through its Safe.

This script is designed for the production Docker Compose container. It never
executes the 4-of-6 Safe transaction. ``--propose`` signs and submits the Safe
proposal using the existing asset-manager ``PRIVATE_KEY``. After the owners
execute the proposal, ``--post-orders`` submits the now-pre-signed orders to
CoW Protocol.

The live executor must remain stopped from proposal creation until both orders
have been posted and settled. From the production Compose project:

.. code-block:: shell

    docker compose stop gmx-ai
    docker compose run --entrypoint /bin/bash gmx-ai --
    test -s state/gmx-ai.json
    test -n "$SAFE_TRANSACTION_SERVICE_API_KEY"

    # Preview only: no Safe or CoW API writes.
    poetry run python scripts/gmx-ai/propose-cowswap-cleanup.py

    # Create and sign the 4-of-6 Safe proposal. Note the artefact path printed.
    poetry run python scripts/gmx-ai/propose-cowswap-cleanup.py \
        --propose --confirm-executor-stopped

    # After the Safe transaction has been executed by the owners:
    poetry run python scripts/gmx-ai/propose-cowswap-cleanup.py \
        --post-orders state/gmx-ai-cowswap-cleanup-safe-nonce-12.json \
        --confirm-executor-stopped

Do not restart the executor until the CoW explorer shows both orders settled
and the Safe's WBTC and WBNB balances are zero or economically negligible.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from eth_account import Account
from hexbytes import HexBytes
from safe_eth.safe import Safe
from safe_eth.safe.api.transaction_service_api.transaction_service_api import (
    TransactionServiceApi,
)
from safe_eth.safe.multi_send import MultiSend, MultiSendOperation, MultiSendTx
from safe_eth.safe.safe_tx import SafeTx
from web3 import Web3

from eth_defi.cow.api import get_cowswap_api
from eth_defi.cow.constants import COWSWAP_SETTLEMENT, COWSWAP_VAULT_RELAYER
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.safe.safe_compat import create_safe_ethereum_client
from eth_defi.token import fetch_erc20_details

from tradeexecutor.ethereum.cowswap.safe_proposal import (
    PreparedCowSwapOrder,
    calculate_order_uid,
    fetch_cowswap_quote,
    post_presigned_order,
    prepare_cowswap_order,
)


ARBITRUM_CHAIN_ID = 42161
SAFE_ADDRESS = Web3.to_checksum_address("0x7838A4E4ecD438c1BdD13b014675c7e877b8b490")
ASSET_MANAGER_ADDRESS = Web3.to_checksum_address(
    "0x350c2d78c06d4d6963eEB6cD44A5A038AAb41d3f"
)
USDC_ADDRESS = Web3.to_checksum_address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831")
MULTISEND_CALL_ONLY_ADDRESS = Web3.to_checksum_address(
    "0x9641d764fc13c8B624c04430C7356C1C7C8102e2"
)
DEFAULT_STATE_FILE = Path("state/gmx-ai.json")
DEFAULT_VALID_FOR_SECONDS = 24 * 60 * 60
DEFAULT_SLIPPAGE_BPS = 100

# Native ETH is deliberately excluded from this clean-up. The Safe uses ETH as
# its GMX execution-fee reserve; excess ETH will be analysed and handled in a
# separate, explicitly reviewed transaction.
ASSETS_TO_CASH_OUT = (
    ("WBTC", Web3.to_checksum_address("0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f")),
    ("WBNB", Web3.to_checksum_address("0xa9004A5421372E1D83fB1f85b0fc986c912f91f3")),
)


SETTLEMENT_ABI = [
    {
        "inputs": [
            {"name": "orderUid", "type": "bytes"},
            {"name": "signed", "type": "bool"},
        ],
        "name": "setPreSignature",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "orderUid", "type": "bytes"}],
        "name": "preSignature",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@dataclass(slots=True, frozen=True)
class PreparedSafeProposal:
    """Safe transaction and CoW orders produced from one balance snapshot."""

    safe: Safe
    safe_tx: SafeTx
    safe_tx_gas: int
    orders: tuple[PreparedCowSwapOrder, ...]
    balances: dict[str, str]


def parse_args() -> argparse.Namespace:
    """Parse production-operation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--propose",
        action="store_true",
        help="Sign and submit the Safe transaction proposal",
    )
    action.add_argument(
        "--post-orders",
        type=Path,
        metavar="ARTEFACT",
        help="Post orders after the Safe proposal was executed",
    )
    parser.add_argument(
        "--confirm-executor-stopped",
        action="store_true",
        help="Acknowledge that the production gmx-ai executor is stopped",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="Mounted authoritative production state file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Proposal artefact path; defaults to state/gmx-ai-cowswap-cleanup-safe-nonce-N.json",
    )
    parser.add_argument(
        "--valid-for-seconds",
        type=int,
        default=DEFAULT_VALID_FOR_SECONDS,
        help="CoW order validity window",
    )
    parser.add_argument(
        "--slippage-bps",
        type=int,
        default=DEFAULT_SLIPPAGE_BPS,
        help="Price protection applied below the verified quote",
    )
    return parser.parse_args()


def assert_production_preconditions(args: argparse.Namespace) -> None:
    """Verify the mounted production state and explicit operator acknowledgement."""
    if not args.state_file.is_file() or args.state_file.stat().st_size == 0:
        raise RuntimeError(
            f"Authoritative production state file is missing or empty: {args.state_file}"
        )
    if (args.propose or args.post_orders) and not args.confirm_executor_stopped:
        raise RuntimeError(
            "Stop the gmx-ai executor and pass --confirm-executor-stopped"
        )
    if args.propose and not os.environ.get("PRIVATE_KEY"):
        raise RuntimeError("PRIVATE_KEY is required to propose the Safe transaction")
    if args.propose and not os.environ.get("SAFE_TRANSACTION_SERVICE_API_KEY"):
        raise RuntimeError(
            "SAFE_TRANSACTION_SERVICE_API_KEY is required for a reliable production proposal"
        )
    if args.valid_for_seconds <= 0:
        raise ValueError("--valid-for-seconds must be positive")
    if not 0 <= args.slippage_bps < 10_000:
        raise ValueError("--slippage-bps must be between 0 and 9999")


def create_web3() -> Web3:
    """Create the production Arbitrum connection."""
    json_rpc = os.environ.get("JSON_RPC_ARBITRUM")
    if not json_rpc:
        raise RuntimeError("JSON_RPC_ARBITRUM is required")
    web3 = create_multi_provider_web3(json_rpc)
    if web3.eth.chain_id != ARBITRUM_CHAIN_ID:
        raise RuntimeError(
            f"Expected Arbitrum chain {ARBITRUM_CHAIN_ID}, got {web3.eth.chain_id}"
        )
    return web3


def prepare_safe_proposal(
    web3: Web3, *, valid_for_seconds: int, slippage_bps: int
) -> PreparedSafeProposal:
    """Fetch balances and quotes, then construct the approval and pre-sign batch."""
    ethereum_client = create_safe_ethereum_client(web3)
    safe = Safe(SAFE_ADDRESS, ethereum_client)
    safe_version = safe.get_version()
    if safe_version != "1.4.1":
        raise RuntimeError(f"Expected Safe version 1.4.1, got {safe_version}")
    owners = safe.retrieve_owners()
    threshold = safe.retrieve_threshold()
    if ASSET_MANAGER_ADDRESS not in owners:
        raise RuntimeError(f"Asset manager {ASSET_MANAGER_ADDRESS} is not a Safe owner")
    if threshold != 4 or len(owners) != 6:
        raise RuntimeError(f"Expected a 4-of-6 Safe, got {threshold}-of-{len(owners)}")

    settlement = web3.eth.contract(
        address=Web3.to_checksum_address(COWSWAP_SETTLEMENT), abi=SETTLEMENT_ABI
    )
    api_base_url = get_cowswap_api(ARBITRUM_CHAIN_ID)
    orders = []
    balances = {}
    calls = []

    for symbol, token_address in ASSETS_TO_CASH_OUT:
        token = fetch_erc20_details(web3, token_address)
        raw_balance = token.fetch_raw_balance_of(SAFE_ADDRESS)
        balances[symbol] = str(token.convert_to_decimals(raw_balance))
        if raw_balance == 0:
            continue
        quote = fetch_cowswap_quote(
            api_base_url=api_base_url,
            safe_address=SAFE_ADDRESS,
            sell_token=token_address,
            buy_token=USDC_ADDRESS,
            sell_amount_before_fee=raw_balance,
            valid_for_seconds=valid_for_seconds,
        )
        prepared_order = prepare_cowswap_order(
            symbol=symbol,
            balance=raw_balance,
            sell_token=token_address,
            buy_token=USDC_ADDRESS,
            quote_response=quote,
            chain_id=ARBITRUM_CHAIN_ID,
            settlement=COWSWAP_SETTLEMENT,
            owner=SAFE_ADDRESS,
            slippage_bps=slippage_bps,
        )
        orders.append(prepared_order)

        approval_data = HexBytes(
            token.contract.functions.approve(
                COWSWAP_VAULT_RELAYER, raw_balance
            )._encode_transaction_data()
        )
        presign_data = HexBytes(
            settlement.functions.setPreSignature(
                prepared_order.uid, True
            )._encode_transaction_data()
        )
        calls.extend(
            (
                MultiSendTx(MultiSendOperation.CALL, token.address, 0, approval_data),
                MultiSendTx(
                    MultiSendOperation.CALL,
                    Web3.to_checksum_address(COWSWAP_SETTLEMENT),
                    0,
                    presign_data,
                ),
            )
        )

    if not orders:
        raise RuntimeError("The Safe has no configured stranded assets to cash out")

    multisend = MultiSend(
        ethereum_client, address=MULTISEND_CALL_ONLY_ADDRESS, call_only=True
    )
    if not web3.eth.get_code(MULTISEND_CALL_ONLY_ADDRESS):
        raise RuntimeError(
            f"MultiSendCallOnly is not deployed at {MULTISEND_CALL_ONLY_ADDRESS}"
        )
    multisend_data = multisend.build_tx_data(calls)
    # Safe's SimulateTxAccessor cannot simulate this production Safe because it
    # has no fallback handler. A zero safeTxGas together with zero gasPrice is
    # the standard Safe proposal representation: execution forwards the gas
    # supplied by the final signer instead of imposing a signed internal cap.
    safe_tx_gas = 0
    safe_tx = safe.build_multisig_tx(
        to=MULTISEND_CALL_ONLY_ADDRESS,
        value=0,
        data=multisend_data,
        operation=MultiSendOperation.DELEGATE_CALL.value,
        safe_tx_gas=safe_tx_gas,
    )
    return PreparedSafeProposal(
        safe=safe,
        safe_tx=safe_tx,
        safe_tx_gas=safe_tx_gas,
        orders=tuple(orders),
        balances=balances,
    )


def build_artefact(proposal: PreparedSafeProposal, *, slippage_bps: int) -> dict:
    """Build the non-secret hand-off artefact for the post-execution phase."""
    return {
        "version": 1,
        "chain_id": ARBITRUM_CHAIN_ID,
        "safe": SAFE_ADDRESS,
        "safe_nonce": proposal.safe_tx.safe_nonce,
        "safe_tx_hash": Web3.to_hex(proposal.safe_tx.safe_tx_hash),
        "safe_tx_to": proposal.safe_tx.to,
        "safe_tx_operation": proposal.safe_tx.operation,
        "safe_tx_gas": proposal.safe_tx_gas,
        "safe_tx_data": Web3.to_hex(proposal.safe_tx.data),
        "slippage_bps": slippage_bps,
        "balances": proposal.balances,
        "orders": [order.to_json() for order in proposal.orders],
    }


def propose_safe_transaction(
    proposal: PreparedSafeProposal, *, private_key: str, safe_api_key: str
) -> str:
    """Sign and submit the proposal to the Safe Transaction Service."""
    proposer = Account.from_key(private_key)
    if proposer.address != ASSET_MANAGER_ADDRESS:
        raise RuntimeError(
            f"PRIVATE_KEY resolves to {proposer.address}, expected asset manager {ASSET_MANAGER_ADDRESS}"
        )
    if proposal.safe.retrieve_nonce() != proposal.safe_tx.safe_nonce:
        raise RuntimeError(
            f"Safe nonce changed after preparation: expected {proposal.safe_tx.safe_nonce}, "
            f"got {proposal.safe.retrieve_nonce()}"
        )

    proposal.safe_tx.sign(private_key)
    transaction_service = TransactionServiceApi(
        network=proposal.safe.ethereum_client.get_network(),
        ethereum_client=proposal.safe.ethereum_client,
        api_key=safe_api_key,
    )
    existing_transactions = transaction_service.get_transactions(
        SAFE_ADDRESS,
        executed=False,
        nonce=proposal.safe_tx.safe_nonce,
    )
    expected_hash = Web3.to_hex(proposal.safe_tx.safe_tx_hash).lower()
    for transaction in existing_transactions:
        existing_hash = str(
            transaction.get("safeTxHash")
            or transaction.get("contractTransactionHash")
            or ""
        ).lower()
        if existing_hash == expected_hash:
            return Web3.to_hex(proposal.safe_tx.safe_tx_hash)
    if existing_transactions:
        raise RuntimeError(
            f"Safe nonce {proposal.safe_tx.safe_nonce} already has a different pending "
            "proposal; inspect or remove it in the Safe UI before retrying"
        )
    transaction_service.post_transaction(proposal.safe_tx)
    return Web3.to_hex(proposal.safe_tx.safe_tx_hash)


def prepared_order_from_json(data: dict) -> PreparedCowSwapOrder:
    """Load one prepared order from the hand-off artefact."""
    prepared_order = PreparedCowSwapOrder(
        symbol=data["symbol"],
        balance=int(data["balance"]),
        quoted_buy_amount=int(data["quoted_buy_amount"]),
        order=data["order"],
        uid=HexBytes(data["uid"]),
    )
    expected_uid = calculate_order_uid(
        chain_id=ARBITRUM_CHAIN_ID,
        settlement=COWSWAP_SETTLEMENT,
        owner=SAFE_ADDRESS,
        order=prepared_order.order,
    )
    if prepared_order.uid != expected_uid:
        raise RuntimeError(
            f"Artefact order UID mismatch for {prepared_order.symbol}: "
            f"{Web3.to_hex(prepared_order.uid)} != {Web3.to_hex(expected_uid)}"
        )
    return prepared_order


def post_orders(web3: Web3, artefact_path: Path) -> None:
    """Verify Safe execution and submit its pre-signed orders to CoW."""
    artefact = json.loads(artefact_path.read_text())
    if (
        artefact["chain_id"] != ARBITRUM_CHAIN_ID
        or artefact["safe"].lower() != SAFE_ADDRESS.lower()
    ):
        raise RuntimeError(
            f"Artefact does not belong to the production GMX AI Safe: {artefact_path}"
        )
    settlement = web3.eth.contract(
        address=Web3.to_checksum_address(COWSWAP_SETTLEMENT), abi=SETTLEMENT_ABI
    )
    api_base_url = get_cowswap_api(ARBITRUM_CHAIN_ID)
    for order_data in artefact["orders"]:
        prepared_order = prepared_order_from_json(order_data)
        if int(prepared_order.order["validTo"]) <= int(time.time()):
            raise RuntimeError(
                f"CoW order for {prepared_order.symbol} expired at {prepared_order.order['validTo']}"
            )
        if settlement.functions.preSignature(prepared_order.uid).call() == 0:
            raise RuntimeError(
                f"Safe has not executed the pre-signature for {prepared_order.symbol}: {Web3.to_hex(prepared_order.uid)}"
            )
        posted_uid = post_presigned_order(
            api_base_url=api_base_url,
            owner=SAFE_ADDRESS,
            prepared_order=prepared_order,
        )
        print(
            f"Posted {prepared_order.symbol}: https://explorer.cow.fi/arb1/orders/{posted_uid}"
        )


def main() -> None:
    """Run preview, proposal, or post-execution order submission."""
    args = parse_args()
    assert_production_preconditions(args)
    web3 = create_web3()

    if args.post_orders:
        post_orders(web3, args.post_orders)
        return

    proposal = prepare_safe_proposal(
        web3,
        valid_for_seconds=args.valid_for_seconds,
        slippage_bps=args.slippage_bps,
    )
    artefact = build_artefact(proposal, slippage_bps=args.slippage_bps)
    print(json.dumps(artefact, indent=2))
    if not args.propose:
        print(
            "Preview only. Re-run with --propose --confirm-executor-stopped to submit to Safe."
        )
        return

    private_key = os.environ["PRIVATE_KEY"]
    safe_api_key = os.environ["SAFE_TRANSACTION_SERVICE_API_KEY"]
    output_path = args.output or Path(
        f"state/gmx-ai-cowswap-cleanup-safe-nonce-{proposal.safe_tx.safe_nonce}.json"
    )
    if output_path.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing proposal artefact: {output_path}"
        )
    safe_tx_hash = propose_safe_transaction(
        proposal,
        private_key=private_key,
        safe_api_key=safe_api_key,
    )
    # Persist only after the Transaction Service accepted the proposal. A failed
    # API call must not leave a stale artefact that blocks a fresh quote/retry.
    output_path.write_text(json.dumps(artefact, indent=2) + "\n")
    print(f"Safe proposal submitted: {safe_tx_hash}")
    print(f"Order artefact: {output_path}")
    print(
        "After 4-of-6 execution, run --post-orders with this artefact before restarting gmx-ai."
    )


if __name__ == "__main__":
    main()
