"""On-chain inspection coverage for the manual Lagoon Safe settlement command."""

import os
from decimal import Decimal
from types import SimpleNamespace

import pytest
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3

from eth_defi.abi import encode_function_call
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonAutomatedDeployment
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation

import tradeexecutor.cli.commands.lagoon_manual_settle as manual_settle
from tradeexecutor.cli.commands.lagoon_manual_settle import _build_settlement_call, format_manual_settlement_instructions, inspect_manual_lagoon_settlement


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
    5. Verify the printed instructions contain all Safe transaction inputs.
    """
    vault = automated_lagoon_vault.vault
    amount = Decimal(9)

    # 1. Create a pending investor deposit and post its valid fresh valuation.
    tx_hash = base_usdc_token.approve(vault.address, amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.request_deposit(depositor, base_usdc_token.convert_to_raw(amount)).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    valuation = Decimal(0)
    settlement_scan_start_block = web3.eth.block_number
    tx_hash = vault.post_new_valuation(valuation).transact({"from": asset_manager.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # 2. Inspect its live queue and construct the direct Safe settlement calldata.
    new_total_assets_raw = base_usdc_token.convert_to_raw(valuation)
    report = inspect_manual_lagoon_settlement(vault, settlement_scan_start_block)

    # 3. Verify gas estimation accepts the direct vault target call from the Safe.
    assert Decimal(report["pending_deposit"]) == amount
    assert Decimal(report["pending_redemption_shares"]) == 0
    assert report["safe"] == vault.safe_address
    assert report["settlement_abi"]["name"] == "settleDeposit"
    assert report["new_total_assets_raw"] == new_total_assets_raw
    assert report["new_total_assets_source"] == "NewTotalAssetsUpdated/TotalAssetsUpdated events"
    assert report["gnosis_safe_transaction_fields"]["to"] == vault.address
    assert report["gnosis_safe_transaction_fields"]["contractInputsValues"]["_newTotalAssets"] == str(new_total_assets_raw)
    assert report["target_call_simulation"]["succeeds"] is True, report["target_call_simulation"]

    # 4. Verify the ABI encodes a non-zero raw NAV.
    _, non_zero_settle_call = _build_settlement_call(vault, "settleDeposit", 1)
    assert Web3.to_hex(encode_function_call(non_zero_settle_call)).endswith("1".zfill(64))

    # 5. Verify the printed instructions contain all Safe transaction inputs.
    instructions = format_manual_settlement_instructions(report)
    assert f"Open Safe {vault.safe_address}" in instructions
    assert f"Set the target contract to {vault.address}" in instructions
    assert '"_newTotalAssets":"0"' in instructions
    assert report["gnosis_safe_transaction_fields"]["data"] in instructions


def test_fetch_pending_new_total_assets_from_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recover pending NAV safely when the Lagoon ABI has no getter.

    1. Mock the shared event reader because this test targets event ordering without an RPC scan.
    2. Recover a non-zero pending NAV from its event.
    3. Confirm a later settlement event and the Lagoon sentinel both clear it.
    """
    new_nav_topic = Web3.keccak(text="NewTotalAssetsUpdated(uint256)")
    settled_nav_topic = Web3.keccak(text="TotalAssetsUpdated(uint256)")
    vault = SimpleNamespace(
        address="0x0000000000000000000000000000000000000001",
        vault_contract=SimpleNamespace(abi=[]),
        web3=object(),
    )

    # 1. Mock the shared event reader because this test targets event ordering without an RPC scan.
    monkeypatch.setattr(manual_settle, "is_anvil", lambda _web3: False)

    def make_log(topic: HexBytes, value: int, block_number: int) -> dict:
        return {
            "topics": [topic],
            "data": HexBytes(value.to_bytes(32, byteorder="big")),
            "blockNumber": block_number,
            "logIndex": 0,
        }

    # 2. Recover a non-zero pending NAV from its event.
    monkeypatch.setattr(
        manual_settle,
        "fetch_vault_settlement_logs",
        lambda **_kwargs: [make_log(new_nav_topic, 123_456_789, 10)],
    )
    assert manual_settle._fetch_pending_new_total_assets(vault, 1, 10) == (
        123_456_789,
        "NewTotalAssetsUpdated/TotalAssetsUpdated events",
    )

    # 3. Confirm a later settlement event and the Lagoon sentinel both clear it.
    monkeypatch.setattr(
        manual_settle,
        "fetch_vault_settlement_logs",
        lambda **_kwargs: [
            make_log(new_nav_topic, 123_456_789, 10),
            make_log(settled_nav_topic, 123_456_789, 11),
        ],
    )
    assert manual_settle._fetch_pending_new_total_assets(vault, 1, 11)[0] is None
    monkeypatch.setattr(
        manual_settle,
        "fetch_vault_settlement_logs",
        lambda **_kwargs: [make_log(new_nav_topic, manual_settle.NO_PENDING_NAV, 12)],
    )
    assert manual_settle._fetch_pending_new_total_assets(vault, 1, 12)[0] is None
