"""Tracking of blockchain transactions.

- Creating transactions and signing them from hot wallet

- Broadcasting and tracking transaction mining status

- Resolving succeed or failed transactions to succeed or failed trades

"""
import datetime
import enum
import textwrap
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple, List

from dataclasses_json import dataclass_json, config

from eth_defi.hotwallet import SignedTransactionWithNonce
from eth_defi.tx import decode_signed_transaction, AssetDelta
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json, decode_pickle_over_json
from tradeexecutor.state.types import JSONHexAddress, JSONHexBytes


def _clean_print_args(val: tuple):
    """Clean Solidity argument blobs for stdout printing"""
    if type(val) in (list, tuple):
        return list(_clean_print_args(x) for x in val)
    elif type(val) == bytes:
        return val.hex()
    else:
        return val


def solidity_arg_encoder(val: tuple | list) -> list:
    """JSON safe Solidity function argument encoder.

    - Support nested lists (Uniswap v3)

    - Fix big ints

    - Abort on floats
    """

    if type(val) not in (list, tuple):
        return val

    def _encode_solidity_value_json_safe(x):
        if type(x) == int:
            return str(x)
        if type(x) == bytes:
            return x.hex()
        if type(x) in (tuple, list):
            return solidity_arg_encoder(x)
        elif type(x) == float:
            # Smart contracts cannot have float arguents
            raise RuntimeError(f"Cannot encode float: {val}")
        return x

    return list(_encode_solidity_value_json_safe(x) for x in val)


class BlockchainTransactionType(enum.Enum):
    """What kind of transactions we generate."""

    #: Direct trade from a hot wallet
    #:
    hot_wallet = "hot_wallet"

    #: A trade tx through Enzyme's vault
    #:
    #: - Target address is the comptroller contract of the vault
    #:
    #: - The function is ComptrollerLib.callOnExtension()
    #:
    #: - The payload is integration manager call for generic adapter
    #:
    #: - The actual trade payload can be read from :py:attr:`BlockchainTransaction.details` field
    #:
    enzyme_vault = "enzyme_vault"

    #: By LagoonTransactionBuilder
    lagoon_vault = "lagoon_vault"

    #: Simulated transaction
    #:
    #: This transaction does not take place on EVM (real blockchain, unit test chain, etc).
    #: and any field like addresses or transaction hashes may not be real.
    simulated = "simulated"


@dataclass_json
@dataclass(frozen=True, slots=True)
class JSONAssetDelta:
    """JSON serialisable asset delta.

    Used for diagnostics purposes only.

    See :py:class:`eth_defi.tx.AssetDelta` for more information.
    """

    #: Address of ERC-20
    asset: str

    #: Integer as string serialisation.
    #:
    #: Because JSON cannot handle big ints.
    raw_amount: str

    @property
    def int_amount(self) -> int:
        """Convert token amount back to int."""
        return int(self.raw_amount)

    @staticmethod
    def from_asset_delta(source: AssetDelta) -> "JSONAssetDelta":
        return JSONAssetDelta(str(source.asset), str(source.raw_amount))


@dataclass_json
@dataclass(slots=True)
class BlockchainTransaction:
    """A stateful blockchain transaction.

    - The state tracks a transaction over its life cycle

    - Transactions are part of a larger logical operation (a trade)

    - Transactions can be resolved either to success or failed

    - Transaction information is easily exported to the frontend

    Transaction has (rough) four phases

    1. Preparation

    2. Signing

    3. Broadcast

    4. Confirmation

    .. note ::

        In the future, during the broadcasting phase, transactions can be re-signed.
        If the gas parameters are too low, a new transaction is generated
        with gas parameters changed and signed again.

    .. note ::

        A lot information is data structure is redundant and can be
        streamlined in the future.

    """

    #: What kidn of internal type of this transaction is
    #:
    type: BlockchainTransactionType = BlockchainTransactionType.hot_wallet

    #: Chain id from https://github.com/ethereum-lists/chains
    chain_id: Optional[int] = None

    #: TODO: Part of signed bytes. Create an accessor.
    from_address: Optional[str] = None

    #: Contract we called. Usually the Uniswap v2 router address.
    contract_address: Optional[JSONHexAddress] = None

    #: Function name we called
    #:
    #: This is Solidity function entry point from the transaction data payload
    #:
    #: Human-readable function name for debugging purposes.
    #:
    function_selector: Optional[str] = None

    #: Arguments we passed to the smart contract entry function.
    #:
    #: This is not JSON serialisable because
    #: individual arguments may contain values that are token amounts
    #: and thus outside the maximum int of JavaScript.
    #:
    transaction_args: Optional[Tuple[Any]] = field(
        default=None,
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        )
    )

    #: If this is a wrapped call, the underlying target of the wrapped call payload.
    #:
    wrapped_target: Optional[JSONHexAddress] = None

    #: Function name the vault calls.
    #:
    #: This is Solidity function entry point from the transaction data payload
    #:
    #:
    #: Human-readable function name for debugging purposes.
    #:
    wrapped_function_selector: Optional[str] = None

    #: Arguments that execute the actual trade.
    #:
    #: In the case of Enzyme's vaults, we need to store the underlying smart contract function call,
    #: so that we can analyse the slippage later on, because we need the swap function input args
    #: for the slippage analysis.
    #:
    wrapped_args: Optional[Tuple[Any]] = field(
        default=None,
        metadata=config(
            encoder=encode_pickle_over_json,
            decoder=decode_pickle_over_json,
        )
    )

    #: Blockchain bookkeeping
    #:
    #: Hex string, starts with 0x.
    #:
    tx_hash: Optional[JSONHexBytes] = None

    #: Blockchain bookkeeping
    nonce: Optional[int] = None

    #: Raw Ethereum transaction dict.
    #: Output from web3 buildTransaction()
    #:
    #: Example:
    #:
    #: `{'value': 0, 'maxFeePerGas': 1844540158, 'maxPriorityFeePerGas': 1000000000, 'chainId': 61, 'from': '0x6B49598B34B9c7FbF7C57306d0b0578676D55ffA', 'gas': 100000, 'to': '0xF2E246BB76DF876Cef8b38ae84130F4F55De395b', 'data': '0x095ea7b30000000000000000000000006d411e0a54382ed43f02410ce1c7a7c122afa6e1ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff', 'nonce': 0}`
    #:
    details: Optional[Dict] = None

    #: Raw bytes of the signed transaction
    #:
    #: Legacy. Use :py:attr:`signed_tx_object` instead.
    #:
    signed_bytes: Optional[JSONHexBytes] = None

    #: Pickled SignedTransactionWithNonce.
    #:
    #: This is a pickled binary of `SignedTransactionWithNonce`
    #: object, as hex. It is the latest signed tx object
    #: we broadcasted over the wire.
    #:
    #: This object may change if we have a broadcast failure (timeout)
    #: due to gas spike and we need to sign the tx again
    #: with different paramenters.
    #:
    #: See :py:class:`eth_defi.hotwallet.SignedTransactionWithNonce`
    #:
    signed_tx_object: Optional[JSONHexBytes] = None

    #: When this transaction was broadcasted
    broadcasted_at: Optional[datetime.datetime] = None

    #: Block timestamp when this tranasction was included in a block
    included_at: Optional[datetime.datetime] = None

    #: Block number when this transaction was included in a block
    block_number: Optional[int] = None

    #: Block has of the transaction where the executor saw the inclusion
    block_hash: Optional[JSONHexBytes] = None

    #: status from the tx receipt. True is success, false is revert.
    status: Optional[bool] = None

    #: Gas consumed by the tx
    realised_gas_units_consumed: Optional[int] = None

    #: Gas price for the tx in gwei
    realised_gas_price: Optional[int] = None

    #: The transaction revert reason if we manage to extract it
    revert_reason: Optional[str] = None

    #: Solidity stack trace of reverted transactions.
    #:
    #: Used in the unit testing environment with Anvil.
    #:
    #: See :py:mod:`eth_defi.trace`.
    stack_trace: Optional[str] = None

    #: List of assets this transaction touches
    #:
    #: Set in :py:class:`tradeexecutor.tx.TransactionBuilder`
    asset_deltas: List[JSONAssetDelta] = field(default_factory=list)

    #: Legacy compatibility field.
    #:
    #: "Somewhat" human-readable encoded Solidity args to be displayed in the frontend.
    #: Arguments cannot be decoded for programmatic use.
    #:
    #: Use :py:attr:`transaction_args` and :py:meth:`get_actual_function_input_args` instead.
    args: Optional[Tuple[Any]] = field(
        default=None,
        metadata=config(
            encoder=solidity_arg_encoder,
        )
    )

    #: Any other metadata associated with this transaction.
    #:
    #: Currently used for `vault_slippage_tolerance`.
    other: dict = field(default_factory=dict)

    #: Human readable notes on this transaction.
    #:
    #: - Used for diagnostics
    #:
    #: - E.g. the text line of the controlling trade that is causing this transaction,
    #:   with information about expected tokens, slippage, etc.
    #:
    #: - Newline separated
    #:
    notes: str | None = ""

    def __repr__(self):

        notes = self.notes or ""
        notes = "\n" + textwrap.indent(notes, prefix="      ")

        if self.status is True:
            return f"<Tx success \n" \
                   f"    from:{self.from_address}\n" \
                   f"    nonce:{self.nonce}\n" \
                   f"    to:{self.contract_address}\n" \
                   f"    func:{self.function_selector}\n" \
                   f"    args:{_clean_print_args(self.transaction_args)}\n" \
                   f"    wrapped args:{_clean_print_args(self.wrapped_args)}\n" \
                   f"    gas limit:{self.get_gas_limit():,}\n" \
                   f"    gas spent:{self.realised_gas_units_consumed:,}\n" \
                   f"    notes:{notes}\n" \
                   f"    >"
        elif self.status is False:
            return f"<Tx reverted \n" \
                   f"    from:{self.from_address}\n" \
                   f"    nonce:{self.nonce}\n" \
                   f"    to:{self.contract_address}\n" \
                   f"    func:{self.function_selector}\n" \
                   f"    args:{_clean_print_args(self.transaction_args)}\n" \
                   f"    wrapped args:{_clean_print_args(self.wrapped_args)}\n" \
                   f"    fail reason:{self.revert_reason}\n" \
                   f"    gas limit:{self.get_gas_limit():,}\n" \
                   f"    gas spent:{self.realised_gas_units_consumed:,}\n" \
                   f"    notes:{notes}\n" \
                   f"    >"
        else:
            return f"<Tx unresolved\n" \
                   f"    from:{self.from_address}\n" \
                   f"    nonce:{self.nonce}\n" \
                   f"    to:{self.contract_address}\n" \
                   f"    func:{self.function_selector}\n" \
                   f"    args:{_clean_print_args(self.transaction_args)}\n" \
                   f"    wrapped args:{_clean_print_args(self.wrapped_args)}\n" \
                   f"    notes:{notes}\n" \
                   f"    >"

    def get_transaction(self) -> SignedTransactionWithNonce | dict:
        """Return the transaction object as it would be in web3.py.

        Needed for :py:func:`analyse_trade_by_receipt`.
        This will reconstruct :py:class:`TypedTransaction` instance from the
        raw signed transaction bytes.
        The object will have a dict containing "data" field which we can then
        use for the trade analysis.
        """

        if self.signed_tx_object:
            return self.signed_tx_object

        # Legacy code path
        assert self.signed_bytes, "Not a signed transaction"
        return decode_signed_transaction(self.signed_bytes)

    def is_success(self) -> bool:
        """Transaction is success if it's succeed flag has been set."""
        return self.status

    def is_reverted(self) -> bool:
        """Transaction reverted."""
        return not self.status

    def set_target_information(
            self,
            chain_id: int,
            contract_address: str,
            function_selector: str,
            args: list,
            details: dict
    ):
        """Update the information on which transaction we are going to perform."""
        assert type(contract_address) == str
        assert type(function_selector) == str
        assert type(args) == list
        self.chain_id = chain_id
        self.contract_address = contract_address
        self.function_selector = function_selector
        self.transaction_args = args
        self.details = details

    def set_broadcast_information(self, nonce: int, tx_hash: str, signed_bytes: str):
        """Update the information we are going to use to broadcast the transaction."""
        assert type(nonce) == int
        assert type(tx_hash) == str
        assert type(signed_bytes) == str
        self.nonce = nonce
        self.tx_hash = tx_hash
        self.signed_bytes = signed_bytes

    def set_confirmation_information(self,
        ts: datetime.datetime,
        block_number: int,
        block_hash: str,
        realised_gas_units_consumed: int,
        realised_gas_price: int,
        status: bool,
        revert_reason: Optional[str] = None,
        stack_trace: Optional[str] = None,
        ):
        """Update the information we are going to use to broadcast the transaction."""
        assert isinstance(ts, datetime.datetime)
        assert type(block_number) == int
        assert type(block_hash) == str
        assert type(realised_gas_units_consumed) == int
        assert type(realised_gas_price) == int
        assert type(status) == bool
        self.included_at = ts
        self.block_number = block_number
        self.block_hash = block_hash
        self.realised_gas_price = realised_gas_price
        self.realised_gas_units_consumed = realised_gas_units_consumed
        self.status = status
        self.revert_reason = revert_reason
        self.stack_trace = stack_trace

    def get_planned_gas_price(self) -> int:
        """How much wei per gas unit we planned to spend on this transactions.

        Gets `maxFeePerGas` for EVM transction.

        :return:
            0 if unknown
        """
        assert self.details, "Details not set, cannot know the gas price"
        return self.details.get("maxFeePerGas", 0)

    def get_gas_limit(self) -> int:
        """Get the gas limit of the transaction.

        Gets `gas` for EVM transction.

        :return:
            0 if unknown
        """
        assert self.details, "Details not set, cannot know the gas price"
        return self.details.get("gas", 0)

    def get_actual_function_input_args(self) -> tuple:
        """Get the Solidity function input args this transaction was calling.

        - For any wrapped vault transaction this returns the real function that was called,
          instead of the proxy function.

        - Otherwise return the args in the transaction payload.
        """
        if self.wrapped_args is not None:
            return self.wrapped_args
        return self.transaction_args

    def get_tx_object(self) -> SignedTransactionWithNonce | None:
        """Get the raw transaction object.

        :return:
            Something that web3.py send_raw_transaction can accept
        """
        if not self.signed_tx_object:
            # Legacy
            return None

        return decode_pickle_over_json(self.signed_tx_object)

    def get_prepared_raw_transaction(self) -> bytes:
        """Get the bytes we can pass to web_ethSendRawTransction"""
        assert self.signed_bytes, "Signed payload missing"
        assert self.signed_bytes.startswith("0x")
        return bytes.fromhex(self.signed_bytes[2:])
