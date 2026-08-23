"""Unit tests for buffered HyperEVM transaction gas pricing."""

from unittest.mock import MagicMock, PropertyMock

import pytest
from eth_account import Account

from eth_defi.hotwallet import HotWallet

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_MAX_FEE_BASE_FEE_MULTIPLIER,
    HYPERCORE_MAX_PRIORITY_FEE_PER_GAS,
    HYPERCORE_MIN_MAX_FEE_PER_GAS,
    HypercoreVaultRouting,
    SettlementBroadcastError,
    calculate_hypercore_gas_price_suggestion,
)
from tradeexecutor.ethereum.web3config import DEFAULT_LONDON_CHAIN_IDS
from tradingstrategy.chain import ChainId


def test_calculate_and_apply_hypercore_gas_price_suggestion() -> None:
    """Test buffered HyperEVM pricing calculation and signing integration.

    1. Price a transaction during a quiet 0.5 gwei period.
    2. Check the historical 4 gwei value remains a minimum, not a ceiling.
    3. Price a transaction using the incident's observed 81.88 gwei base fee.
    4. Check the fee cap tracks congestion with the configured buffer.
    5. Sign through the routing helper using a mocked wallet and contract call.
    6. Check the computed EIP-1559 fields reach the signed transaction.
    """
    # 1. Price a transaction during a quiet 0.5 gwei period.
    quiet_base_fee = 500_000_000
    quiet = calculate_hypercore_gas_price_suggestion(quiet_base_fee)

    # 2. Check the historical 4 gwei value remains a minimum, not a ceiling.
    assert quiet.base_fee == quiet_base_fee
    assert quiet.max_priority_fee_per_gas == HYPERCORE_MAX_PRIORITY_FEE_PER_GAS
    assert quiet.max_fee_per_gas == HYPERCORE_MIN_MAX_FEE_PER_GAS
    assert ChainId.hyperliquid.value in DEFAULT_LONDON_CHAIN_IDS
    assert ChainId.hyperliquid_testnet.value in DEFAULT_LONDON_CHAIN_IDS

    # 3. Price a transaction using the incident's observed 81.88 gwei base fee.
    congested_base_fee = 81_880_000_000
    congested = calculate_hypercore_gas_price_suggestion(congested_base_fee)

    # 4. Check the fee cap tracks congestion with the configured buffer.
    assert congested.base_fee == congested_base_fee
    assert congested.max_fee_per_gas == (
        congested_base_fee * HYPERCORE_MAX_FEE_BASE_FEE_MULTIPLIER
    )
    assert congested.max_fee_per_gas > congested_base_fee

    # 5. Sign through the routing helper using a mocked wallet and contract call.
    # Mock the wallet and contract because this unit test checks transaction
    # plumbing without requiring a private key or live HyperEVM RPC endpoint.
    routing = object.__new__(HypercoreVaultRouting)
    routing.chain_id = 999
    routing.web3 = MagicMock()
    type(routing.web3.eth).gas_price = PropertyMock(return_value=congested_base_fee)
    routing.deployer = HotWallet(Account.create())
    routing.deployer.current_nonce = 7
    fn = MagicMock()
    fn.address = "0x" + "56" * 20
    fn.fn_name = "performCall"
    fn.build_transaction.side_effect = lambda params: {
        **params,
        "to": fn.address,
        "data": "0x",
        # Simulate Web3's historical legacy default. The routing must remove it
        # before asking the real eth-account signer to sign an EIP-1559 tx.
        "gasPrice": 1_000_000_000,
    }
    blockchain_tx = routing._sign_module_call(fn)

    # 6. Check the computed EIP-1559 fields reach the signed transaction.
    assert blockchain_tx.details["maxFeePerGas"] == congested.max_fee_per_gas
    assert (
        blockchain_tx.details["maxPriorityFeePerGas"]
        == HYPERCORE_MAX_PRIORITY_FEE_PER_GAS
    )
    assert "gasPrice" not in blockchain_tx.details
    assert blockchain_tx.signed_bytes.startswith("0x02")
    assert blockchain_tx.tx_hash.startswith("0x")
    assert blockchain_tx.nonce == 7
    assert blockchain_tx.signed_tx_object is not None
    assert blockchain_tx.details["function"] == fn.fn_name


def test_sign_module_call_preserves_preparation_failures() -> None:
    """Test pricing and signing failures are preserved before broadcast.

    1. Configure the HyperEVM gas-price RPC read to fail.
    2. Attempt to build and sign a module transaction.
    3. Check the failure uses the settlement error path and no transaction is signed.
    4. Configure the signer to fail after successful live pricing.
    5. Attempt to build and sign another module transaction.
    6. Check the signed transaction failure is also persisted as failed.
    """
    # 1. Configure the HyperEVM gas-price RPC read to fail.
    # Mock RPC, contract and signer boundaries because this unit test exercises
    # local failure persistence without external RPC or private-key access.
    routing = object.__new__(HypercoreVaultRouting)
    routing.chain_id = 999
    routing.web3 = MagicMock()
    type(routing.web3.eth).gas_price = PropertyMock(
        side_effect=ConnectionError("HyperEVM RPC unavailable")
    )
    routing.deployer = MagicMock()
    routing.deployer.address = "0x" + "12" * 20
    fn = MagicMock()
    fn.address = "0x" + "56" * 20
    fn.fn_name = "performCall"
    fn.build_transaction.side_effect = lambda params: dict(params)

    # 2. Attempt to build and sign a module transaction.
    # 3. Check the failure uses the settlement error path and no transaction is signed.
    with pytest.raises(
        SettlementBroadcastError, match="HyperEVM RPC unavailable"
    ) as exc_info:
        routing._sign_module_call(fn)
    routing.deployer.sign_transaction_with_new_nonce.assert_not_called()
    assert exc_info.value.tx.signed_bytes is None
    assert exc_info.value.tx.status is False
    assert exc_info.value.tx.details["function"] == fn.fn_name
    assert (
        exc_info.value.tx.revert_reason
        == "Transaction preparation failed: HyperEVM RPC unavailable"
    )

    # 4. Configure the signer to fail after successful live pricing.
    type(routing.web3.eth).gas_price = PropertyMock(return_value=81_880_000_000)
    signer_unavailable = ValueError("Signer unavailable")
    routing.deployer.sign_transaction_with_new_nonce.side_effect = signer_unavailable

    # 5. Attempt to build and sign another module transaction.
    # 6. Check the signed transaction failure is also persisted as failed.
    with pytest.raises(
        SettlementBroadcastError, match="Signer unavailable"
    ) as exc_info:
        routing._sign_module_call(fn)
    assert exc_info.value.tx.signed_bytes is None
    assert exc_info.value.tx.status is False
    assert exc_info.value.tx.details["function"] == fn.fn_name
    assert exc_info.value.tx.details["maxFeePerGas"] == 81_880_000_000 * 4
    assert (
        exc_info.value.tx.revert_reason
        == "Transaction preparation failed: Signer unavailable"
    )
