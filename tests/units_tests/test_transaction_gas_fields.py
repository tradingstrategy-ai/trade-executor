"""Test that confirmed transactions store gas used and gas price the right way round.

Positional callers used to pass the price first, so every transaction on the main
execution path stored them swapped. Reports then showed a gas price as gas used.
"""

import datetime
from typing import Any

import pytest
from hexbytes import HexBytes
from pytest_mock import MockerFixture

import tradeexecutor.ethereum.tx as tx_module
from tradeexecutor.ethereum.execution import update_confirmation_status
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction

#: Gas used by the LIX deposit of 2026-11-05.
GAS_USED = 200_146
#: Effective gas price of that deposit, in wei.
GAS_PRICE = 4_652_987_018
TX_HASH = "0x" + "ab" * 32
TS = datetime.datetime(2026, 11, 5, 16, 18)


def create_receipt() -> dict[str, Any]:
    """Create a successful transaction receipt."""
    return {
        "blockNumber": 100,
        "blockHash": HexBytes("0x" + "cd" * 32),
        "gasUsed": GAS_USED,
        "effectiveGasPrice": GAS_PRICE,
        "status": 1,
    }


def test_execution_stores_gas_fields_correctly(mocker: MockerFixture) -> None:
    """The live execution path keeps gas used and gas price apart.

    1. Create a transaction and a receipt with a realistic gas price.
    2. Resolve it through the shared confirmation update.
    3. Verify each field holds its own value.
    """
    # 1. Create a transaction and a receipt with a realistic gas price.
    tx = BlockchainTransaction(tx_hash=TX_HASH, nonce=1, details={"gas": 900_000})
    trade = mocker.Mock()

    # 2. Resolve it through the shared confirmation update.
    update_confirmation_status(mocker.Mock(), TS, {TX_HASH: (trade, tx)}, {HexBytes(TX_HASH): create_receipt()})

    # 3. Verify each field holds its own value.
    assert tx.realised_gas_units_consumed == GAS_USED
    assert tx.realised_gas_price == GAS_PRICE


def test_transaction_builder_stores_gas_fields_correctly(mocker: MockerFixture) -> None:
    """The transaction builder's confirmation path keeps gas used and gas price apart.

    The broadcast is mocked, because the bug is in how the receipt is stored.

    1. Create a transaction and fake its broadcast.
    2. Broadcast it through the builder.
    3. Verify each field holds its own value.
    """
    # 1. Create a transaction and fake its broadcast.
    tx = BlockchainTransaction(tx_hash=TX_HASH, nonce=1)
    mocker.patch.object(tx_module.HotWalletTransactionBuilder, "serialise_to_broadcast_format")
    mocker.patch.object(tx_module, "broadcast_and_wait_transactions_to_complete", return_value={HexBytes(TX_HASH): create_receipt()})

    # 2. Broadcast it through the builder.
    TransactionBuilder.broadcast_and_wait_transactions_to_complete(mocker.Mock(), [tx])

    # 3. Verify each field holds its own value.
    assert tx.realised_gas_units_consumed == GAS_USED
    assert tx.realised_gas_price == GAS_PRICE


def test_confirmation_information_refuses_positional_arguments() -> None:
    """Keyword-only arguments stop the swap from coming back.

    1. Call the setter with positional arguments.
    2. Verify it refuses them.
    """
    # 1. Call the setter with positional arguments.
    tx = BlockchainTransaction(tx_hash=TX_HASH, nonce=1)

    # 2. Verify it refuses them.
    with pytest.raises(TypeError):
        tx.set_confirmation_information(TS, 100, "0x00", GAS_PRICE, GAS_USED, True)
