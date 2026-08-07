"""On-chain inspection coverage for the manual Lagoon Safe settlement command."""

import os
from decimal import Decimal

import pytest
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.abi import encode_function_call
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.commands.lagoon_manual_settle import _build_settlement_call, inspect_manual_lagoon_settlement


JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
pytestmark = pytest.mark.skipif(not JSON_RPC_BASE, reason="No JSON_RPC_BASE environment variable")


def test_inspect_manual_lagoon_settlement(
    web3: Web3,
    automated_lagoon_vault: LagoonAutomatedDeployment,
    base_usdc_token: TokenDetails,
    depositor: HexAddress,
    asset_manager: HotWallet,
) -> None:
    """Inspect a pending Lagoon deposit and simulate the Safe settlement.

    1. Create a pending investor deposit and post its valid fresh valuation.
    2. Inspect its live queue and construct the direct Safe settlement calldata.
    3. Verify gas estimation accepts the direct vault target call from the Safe.
    4. Verify the ABI encodes a non-zero raw NAV.
    """
    vault = automated_lagoon_vault.vault
    amount = Decimal(9)

    # 1. Create a pending investor deposit and post its valid fresh valuation.
    tx_hash = base_usdc_token.approve(vault.address, amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.request_deposit(depositor, base_usdc_token.convert_to_raw(amount)).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    valuation = Decimal(0)
    tx_hash = vault.post_new_valuation(valuation).transact({"from": asset_manager.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # 2. Inspect its live queue and construct the direct Safe settlement calldata.
    new_total_assets_raw = base_usdc_token.convert_to_raw(valuation)
    report = inspect_manual_lagoon_settlement(vault, new_total_assets_raw=new_total_assets_raw)

    # 3. Verify gas estimation accepts the direct vault target call from the Safe.
    assert Decimal(report["pending_deposit"]) == amount
    assert Decimal(report["pending_redemption_shares"]) == 0
    assert report["safe"] == vault.safe_address
    assert report["settlement_abi"]["name"] == "settleDeposit"
    assert report["pending_new_total_assets_raw"] is None
    assert report["new_total_assets_raw"] == new_total_assets_raw
    assert report["gnosis_safe_transaction_fields"]["to"] == vault.address
    assert report["gnosis_safe_transaction_fields"]["contractInputsValues"]["_newTotalAssets"] == str(new_total_assets_raw)
    assert report["target_call_simulation"]["succeeds"] is True, report["target_call_simulation"]

    # 4. Verify the ABI encodes a non-zero raw NAV.
    _, non_zero_settle_call = _build_settlement_call(vault, "settleDeposit", 1)
    assert Web3.to_hex(encode_function_call(non_zero_settle_call)).endswith("1".zfill(64))
