"""Test P1: Equity snapshot failure aborts deposit to prevent false verification.

When the pre-phase2 equity snapshot API call fails, _settle_deposit() must
NOT proceed with phase 2. Without a baseline, the verification step cannot
distinguish pre-existing equity from a fresh deposit, risking a false pass.
"""

import datetime
import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch


VAULT_ADDR = "0xdfc24b077bc1425ad1dea75bcb6f8158e10df303"
SAFE_ADDR = "0xSAFE"


def _make_routing():
    from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting

    routing = object.__new__(HypercoreVaultRouting)
    routing.web3 = MagicMock()
    routing.lagoon_vault = MagicMock()
    routing.lagoon_vault.safe_address = SAFE_ADDR
    routing.deployer = MagicMock()
    routing.chain_id = 999
    routing.is_testnet = False
    routing.simulate = False
    routing.reserve_token_address = "0xusdc"
    routing._session = MagicMock()
    return routing


def _make_trade(planned_reserve=Decimal("50.0")):
    trade = MagicMock()
    trade.is_buy.return_value = True
    trade.is_vault.return_value = True
    trade.get_planned_reserve.return_value = planned_reserve
    trade.trade_id = 1
    trade.blockchain_transactions = [MagicMock(tx_hash="0xaa")]
    trade.other_data = {}
    trade.pair = MagicMock()
    trade.pair.pool_address = VAULT_ADDR
    trade.pair.other_data = {"vault_protocol": "hypercore"}
    return trade


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_snapshot_failure_aborts_deposit(
    mock_fetch_equity, mock_escrow, mock_block_ts, mock_report_failure,
    caplog,
):
    """If equity snapshot fails, deposit is aborted with stranded USDC recorded."""
    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)

    from hexbytes import HexBytes
    receipts = {HexBytes("0xaa"): {"status": 1, "blockNumber": 100}}

    # Escrow clears fine, but equity snapshot fails
    mock_escrow.return_value = None
    mock_fetch_equity.side_effect = Exception("Hyperliquid API 500")

    with caplog.at_level(logging.ERROR):
        routing._settle_deposit(
            routing.web3, state, trade, receipts,
            stop_on_execution_failure=False,
        )

    # Trade should be marked as failed
    mock_report_failure.assert_called_once()
    # Stranded USDC should be recorded
    assert trade.other_data.get("hypercore_stranded_usdc") is not None
    assert trade.other_data["hypercore_stranded_usdc"]["location"] == "hypercore_spot"
    # Phase 2 should NOT have been broadcast (no second tx appended)
    assert len(trade.blockchain_transactions) == 1  # only phase 1
    # Error message logged
    assert any("Cannot snapshot vault equity" in r.message for r in caplog.records)


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_snapshot_success_proceeds_with_phase2(
    mock_fetch_equity, mock_wait_confirm, mock_escrow, mock_block_ts,
    mock_report_failure,
):
    """When equity snapshot succeeds, phase 2 proceeds normally."""
    from eth_defi.hyperliquid.api import UserVaultEquity

    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    mock_block_ts.return_value = datetime.datetime(2025, 1, 1)

    from hexbytes import HexBytes
    receipts = {HexBytes("0xaa"): {"status": 1, "blockNumber": 100}}

    mock_escrow.return_value = None

    # Snapshot returns existing equity
    eq = UserVaultEquity(
        vault_address=VAULT_ADDR,
        equity=Decimal("100.0"),
        locked_until=datetime.datetime(2030, 1, 1),
    )
    mock_fetch_equity.return_value = eq

    # Phase 2 broadcast
    phase2_tx = MagicMock(tx_hash="0xbb")
    phase2_receipt = {"status": 1, "blockNumber": 101}

    # Deposit verification
    confirmed_eq = UserVaultEquity(
        vault_address=VAULT_ADDR,
        equity=Decimal("150.0"),
        locked_until=datetime.datetime(2030, 1, 1),
    )
    mock_wait_confirm.return_value = confirmed_eq

    with patch.object(routing, "_broadcast_phase2", return_value=(phase2_tx, phase2_receipt)):
        routing._settle_deposit(
            routing.web3, state, trade, receipts,
            stop_on_execution_failure=False,
        )

    # Trade should succeed
    state.mark_trade_success.assert_called_once()
    # report_failure should NOT have been called
    mock_report_failure.assert_not_called()
    # executed_amount should be the deposit delta (50), not total equity (150)
    call_kwargs = state.mark_trade_success.call_args
    assert call_kwargs[1]["executed_amount"] == Decimal("50.0") or call_kwargs[0][2] == Decimal("50.0")
