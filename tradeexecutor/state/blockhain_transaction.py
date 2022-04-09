import datetime
from dataclasses import dataclass
from typing import Optional, List, Any, Dict

from dataclasses_json import dataclass_json

from tradeexecutor.state.types import JSONHexAddress, JSONHexBytes


@dataclass_json
@dataclass
class BlockchainTransactionInfo:
    """Information about a blockchain level transaction associated with the trades and anything else.

    Transaction has four phases

    1. Preparation
    2. Signing
    3. Broadcast
    4. Confirmation
    """

    #: Chain id from https://github.com/ethereum-lists/chains
    chain_id: Optional[int] = None

    #: Contract we called. Usually the Uniswap v2 router address.
    contract_address: Optional[JSONHexAddress] = None

    #: Function we called
    function_selector: Optional[str] = None

    #: Arguments we passed to the smart contract function
    args: Optional[List[Any]] = None

    #: Blockchain bookkeeping
    tx_hash: Optional[JSONHexBytes] = None

    #: Blockchain bookkeeping
    nonce: Optional[int] = None

    #: Raw Ethereum transaction dict.
    #: Output from web3 buildTransaction()
    details: Optional[Dict] = None

    #: Raw bytes of the signed transaction
    signed_bytes: Optional[JSONHexBytes] = None

    included_at: Optional[datetime.datetime] = None
    block_number: Optional[int] = None
    block_hash: Optional[JSONHexBytes] = None

    #: status from the tx receipt. True is success, false is revert.
    status: Optional[bool] = None

    #: Gas consumed by the tx
    realised_gas_units_consumed: Optional[int] = None

    #: Gas price for the tx in gwei
    realised_gas_price: Optional[int] = None

    #: The transaction revert reason if we manage to extract it
    revert_reason: Optional[str] = None


    def set_target_information(self, chain_id: int, contract_address: str, function_selector: str, args: list, details: dict):
        """Update the information on which transaction we are going to perform."""
        assert type(contract_address) == str
        assert type(function_selector) == str
        assert type(args) == list
        self.chain_id = chain_id
        self.contract_address = contract_address
        self.function_selector = function_selector
        self.args = args
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
        status: bool):
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