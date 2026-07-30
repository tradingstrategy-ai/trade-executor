"""Recovery-safe HyperCore vault deposit settlement tests.

These are mocked because CoreWriter receipts and HyperCore Info API state live
on different protocol layers and cannot be reproduced on a normal EVM testnet.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from eth_defi.hyperliquid.api import HypercoreDepositVerificationError, UserVaultEquity
from hexbytes import HexBytes
import pytest

from tradeexecutor.ethereum.vault.hypercore_routing import (
    HypercoreVaultRouting,
    HypercoreWithdrawalVerificationError,
)
from tradeexecutor.ethereum.execution import EthereumExecution


def _make_routing() -> HypercoreVaultRouting:
    """Create an isolated routing instance because live HyperCore is not deterministic."""
    routing = object.__new__(HypercoreVaultRouting)
    routing.web3 = MagicMock()
    routing.lagoon_vault = MagicMock()
    routing.lagoon_vault.safe_address = "0xSAFE"
    routing.deployer = MagicMock()
    routing.chain_id = 999
    routing.is_testnet = False
    routing.simulate = False
    routing.reserve_token_address = "0xusdc"
    routing._session = MagicMock()
    routing.allowed_intermediary_pairs = {}
    return routing


def _make_trade() -> MagicMock:
    """Create the minimal live buy trade required by deposit settlement."""
    trade = MagicMock()
    trade.is_buy.return_value = True
    trade.is_vault.return_value = True
    trade.get_planned_reserve.return_value = Decimal("48.884068")
    trade.trade_id = 1486
    trade.planned_quantity = Decimal("48.884068")
    trade.flags = set()
    trade.other_data = {
        "hypercore_phase1_spot_baseline_usdc": "0.217882",
        "hypercore_deposit_capital_at_risk": {
            "amount_raw": 48_884_068,
            "phase": "phase1_broadcast_pending",
        },
    }
    trade.blockchain_transactions = [MagicMock(tx_hash="0xaaa")]
    trade.pair = MagicMock()
    trade.pair.pool_address = "0xVAULT"
    return trade


def _make_equity(value: Decimal) -> UserVaultEquity:
    """Create an Info API-like vault equity response."""
    return UserVaultEquity(
        vault_address="0xVAULT",
        equity=value,
        locked_until=datetime.datetime(2030, 1, 1),
    )


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_deposit_confirmation_timeout_keeps_capital_at_risk_in_perp_or_vault(
    mock_fetch_equity: MagicMock,
    mock_wait_confirmation: MagicMock,
    mock_wait_escrow: MagicMock,
    mock_block_timestamp: MagicMock,
    mock_report_failure: MagicMock,
) -> None:
    """A successful receipt and timed-out confirmation must retain USDC conservatively.

    1. Reproduce trade #1486 with the 0.5 USDC perp baseline and successful spot-to-perp proof.
    2. Return a successful perp-to-vault receipt but time out vault equity confirmation.
    3. Verify retained allocation and the ambiguous perp-or-vault recovery location.
    """
    # 1. Reproduce the production balances with mocked CoreWriter and Info API calls.
    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    phase2_tx = MagicMock(tx_hash="0xbbb")
    phase3_tx = MagicMock(tx_hash="0xccc")
    receipts = {HexBytes("0xaaa"): {"status": 1, "blockNumber": 100}}
    mock_block_timestamp.return_value = datetime.datetime(2026, 7, 30)
    mock_fetch_equity.return_value = _make_equity(Decimal("9326.891623"))
    mock_wait_confirmation.side_effect = HypercoreDepositVerificationError("vault unchanged")

    # 2. Settle phase 2 then receive a successful phase-3 receipt without vault confirmation.
    with (
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("49.101950")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0.5")),
        patch.object(
            routing,
            "_wait_for_deposit_spot_to_perp_transfer",
            return_value=(Decimal("0.217882"), Decimal("49.384068")),
        ),
        patch.object(routing, "_broadcast_deposit_spot_to_perp", return_value=(phase2_tx, {"status": 1, "blockNumber": 101})),
        patch.object(routing, "_broadcast_deposit_perp_to_vault", return_value=(phase3_tx, {"status": 1, "blockNumber": 102})),
    ):
        routing._settle_deposit(routing.web3, state, trade, receipts, stop_on_execution_failure=False)

    # 3. A timeout cannot prove where an accepted vault action will finally settle.
    assert trade.other_data["retain_reserve_allocation_on_failure"] is True
    assert trade.other_data["hypercore_stranded_usdc"]["location"] == "hypercore_perp_or_vault"
    assert len(trade.blockchain_transactions) == 3
    mock_report_failure.assert_called_once()


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_deposit_does_not_send_vault_leg_without_spot_to_perp_proof(
    mock_fetch_equity: MagicMock,
    mock_wait_escrow: MagicMock,
    mock_block_timestamp: MagicMock,
    mock_report_failure: MagicMock,
) -> None:
    """A missing spot-to-perp balance movement must stop before vault transfer.

    1. Settle phase 1 and return a successful spot-to-perp EVM receipt.
    2. Make the dual-balance proof time out.
    3. Verify the vault leg is never broadcast and recovery remains ambiguous.
    """
    # 1. Set up the confirmed phase-1 and phase-2 receipt.
    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    phase2_tx = MagicMock(tx_hash="0xbbb")
    receipts = {HexBytes("0xaaa"): {"status": 1, "blockNumber": 100}}
    mock_block_timestamp.return_value = datetime.datetime(2026, 7, 30)
    mock_fetch_equity.return_value = _make_equity(Decimal("9326.891623"))

    # 2. Simulate the protocol no-op through the mocked balance verifier.
    with (
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("49.101950")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0.5")),
        patch.object(
            routing,
            "_wait_for_deposit_spot_to_perp_transfer",
            side_effect=HypercoreWithdrawalVerificationError("perp unchanged"),
        ),
        patch.object(routing, "_broadcast_deposit_spot_to_perp", return_value=(phase2_tx, {"status": 1, "blockNumber": 101})),
        patch.object(routing, "_broadcast_deposit_perp_to_vault") as mock_perp_to_vault,
    ):
        routing._settle_deposit(routing.web3, state, trade, receipts, stop_on_execution_failure=False)

    # 3. Do not create a second, potentially duplicate, vault action.
    mock_perp_to_vault.assert_not_called()
    assert trade.other_data["hypercore_stranded_usdc"]["location"] == "hypercore_spot_or_perp"
    mock_report_failure.assert_called_once()


def test_spot_to_perp_verifier_requires_both_balance_movements() -> None:
    """The deposit verifier requires coherent spot and perp changes.

    1. Create a mocked routing instance with post-transfer spot and perp balances.
    2. Run the dual-balance verifier for a 48.884068 USDC transfer.
    3. Verify it accepts only after both expected movements are visible.
    """
    # 1. The mocks model the Info API after a confirmed internal transfer.
    routing = _make_routing()

    # 2. Run the verifier against the exact production transfer amount.
    with (
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0.217882")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("49.384068")),
    ):
        spot, perp = routing._wait_for_deposit_spot_to_perp_transfer(
            spot_baseline=Decimal("49.101950"),
            perp_baseline=Decimal("0.5"),
            expected_increase_raw=48_884_068,
        )

    # 3. Both balances prove the same transfer rather than receipt status alone.
    assert spot == pytest.approx(Decimal("0.217882"))
    assert perp == pytest.approx(Decimal("49.384068"))


def test_spot_to_perp_verifier_rejects_partial_movement() -> None:
    """A partial transfer cannot authorise the full vault transfer.

    1. Model a 5 USDC deposit whose balances moved by only 4.90 USDC.
    2. Run the exact dual-balance verifier with no time left to poll.
    3. Verify it rejects the partial transfer before phase 3 can be created.
    """
    # 1. Use the minimum supported deposit to make the formerly permitted
    # 0.10 USDC shortfall materially visible.
    routing = _make_routing()

    # 2. A receipt alone must not make this incomplete move eligible for phase 3.
    with (
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("0.10")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("4.90")),
        pytest.raises(HypercoreWithdrawalVerificationError, match="Expected 5"),
    ):
        routing._wait_for_deposit_spot_to_perp_transfer(
            spot_baseline=Decimal("5.00"),
            perp_baseline=Decimal("0"),
            expected_increase_raw=5_000_000,
            timeout=0,
        )

    # 3. The raised error is the stop condition; phase 3 was never available.


@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity")
def test_successful_deposit_clears_at_risk_marker(
    mock_fetch_equity: MagicMock,
    mock_wait_confirmation: MagicMock,
    mock_wait_escrow: MagicMock,
    mock_block_timestamp: MagicMock,
) -> None:
    """A confirmed vault deposit must stop looking like stranded transit capital.

    1. Settle all three deposit phases with mocked protocol evidence.
    2. Confirm the expected vault-equity increase.
    3. Verify success clears both conservative failure-accounting markers.
    """
    # 1. Mock each external protocol layer because their receipt and Info API
    # states cannot be made deterministic on a normal EVM testnet.
    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    phase2_tx = MagicMock(tx_hash="0xbbb")
    phase3_tx = MagicMock(tx_hash="0xccc")
    receipts = {HexBytes("0xaaa"): {"status": 1, "blockNumber": 100}}
    mock_block_timestamp.return_value = datetime.datetime(2026, 7, 30)
    mock_fetch_equity.return_value = _make_equity(Decimal("9326.891623"))
    mock_wait_confirmation.return_value = _make_equity(Decimal("9375.775691"))

    # 2. Provide exact phase-2 proof and final vault confirmation.
    with (
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("49.101950")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("0.5")),
        patch.object(
            routing,
            "_wait_for_deposit_spot_to_perp_transfer",
            return_value=(Decimal("0.217882"), Decimal("49.384068")),
        ),
        patch.object(routing, "_broadcast_deposit_spot_to_perp", return_value=(phase2_tx, {"status": 1, "blockNumber": 101})),
        patch.object(routing, "_broadcast_deposit_perp_to_vault", return_value=(phase3_tx, {"status": 1, "blockNumber": 102})),
    ):
        routing._settle_deposit(routing.web3, state, trade, receipts, stop_on_execution_failure=False)

    # 3. Future account checks must see an ordinary successful position.
    assert "hypercore_deposit_capital_at_risk" not in trade.other_data
    assert "retain_reserve_allocation_on_failure" not in trade.other_data
    state.mark_trade_success.assert_called_once()


def test_execution_persists_at_risk_marker_before_broadcast() -> None:
    """The prepared at-risk marker is checkpointed before a node sees the transaction.

    1. Make routing setup create the phase-1 marker for a sequential trade.
    2. Capture the pre-broadcast state checkpoint and then enter broadcast.
    3. Verify the checkpoint already contains the marker, modelling crash restart.
    """
    # 1. Use mocks because this is the execution-order contract between the
    # generic executor and HyperCore routing, not a chain integration test.
    execution = object.__new__(EthereumExecution)
    execution.max_slippage = None
    execution._pre_broadcast_state_sync_callback = None
    trade = MagicMock()
    trade.trade_id = 1486
    trade.get_planned_value.return_value = 48.884068
    trade.pair.get_ticker.return_value = "Citadel-USDC"
    trade.is_failed.return_value = False
    trade.other_data = {}
    state = MagicMock()
    routing = MagicMock()
    checkpointed_markers: list[dict] = []

    def create_marker(*args, **kwargs) -> None:
        trade.other_data["hypercore_deposit_capital_at_risk"] = {"phase": "phase1_broadcast_pending"}

    def checkpoint() -> None:
        checkpointed_markers.append(dict(trade.other_data))

    def broadcast(*args, **kwargs) -> None:
        assert checkpointed_markers == [{"hypercore_deposit_capital_at_risk": {"phase": "phase1_broadcast_pending"}}]

    routing.setup_trades.side_effect = create_marker
    execution.set_pre_broadcast_state_sync_callback(checkpoint)
    execution._execute_trade_batch = broadcast

    # 2. Execute the sequential preparation-to-broadcast path.
    execution._execute_trades_sequentially(
        datetime.datetime(2026, 7, 30),
        state,
        [trade],
        routing,
        MagicMock(),
        check_balances=False,
        rebroadcast=False,
        triggered=False,
    )

    # 3. The persisted snapshot supplies the marker after a hard restart.
    assert checkpointed_markers == [{"hypercore_deposit_capital_at_risk": {"phase": "phase1_broadcast_pending"}}]


@patch("tradeexecutor.ethereum.vault.hypercore_routing.report_failure")
@patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp")
def test_phase1_revert_clears_at_risk_marker_for_normal_refund(
    mock_block_timestamp: MagicMock,
    mock_report_failure: MagicMock,
) -> None:
    """A definite phase-1 revert must restore ordinary failed-buy accounting.

    1. Create a deposit carrying the pre-broadcast at-risk marker.
    2. Return a reverted phase-1 receipt.
    3. Verify the marker is cleared before failed-trade processing.
    """
    # 1. The marker exists before the phase-1 receipt is known.
    routing = _make_routing()
    trade = _make_trade()
    state = MagicMock()
    mock_block_timestamp.return_value = datetime.datetime(2026, 7, 30)
    receipts = {HexBytes("0xaaa"): {"status": 0, "blockNumber": 100}}

    # 2. A reverted deposit transaction cannot have removed USDC from the Safe.
    routing._settle_deposit(routing.web3, state, trade, receipts, stop_on_execution_failure=False)

    # 3. The generic failed-buy path may now refund normally.
    assert "hypercore_deposit_capital_at_risk" not in trade.other_data
    assert "retain_reserve_allocation_on_failure" not in trade.other_data
    mock_report_failure.assert_called_once()
