"""Test P5: Stranded USDC marker on failed Hypercore deposits."""

from unittest.mock import MagicMock, patch

from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.state.repair import repair_trades


def test_mark_stranded_usdc_stores_info() -> None:
    """Stranded-deposit metadata must preserve specific recovery instructions.

    1. Create a minimal real routing object and failed deposit trade.
    2. Record USDC that is known to be in HyperCore spot.
    3. Verify the marker identifies the Safe, amount and recovery path.
    """
    # 1. Avoid a spec mock because this helper also calls routing recovery text.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock(safe_address="0xABC123")

    trade = MagicMock()
    trade.other_data = {}
    trade.trade_id = 42

    # 2. Record a known spot location and its exact raw USDC amount.
    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=50_000_000,
        location="hypercore_spot",
    )

    # 3. Operators need an unambiguous marker before they reconcile balances.
    info = trade.other_data["hypercore_stranded_usdc"]
    assert info["amount_raw"] == 50_000_000
    assert info["amount_human"] == "50"
    assert info["location"] == "hypercore_spot"
    assert info["safe_address"] == "0xABC123"
    assert "check-hypercore-user.py" in info["recovery"]
    trade.add_note.assert_called_once()
    assert "stranded" in trade.add_note.call_args[0][0].lower()


def test_mark_stranded_usdc_creates_other_data() -> None:
    """Stranded-deposit metadata must initialise an absent metadata dictionary.

    1. Create a minimal routing object and a trade without metadata.
    2. Record a deposit stranded in either spot or perp.
    3. Verify the helper creates the metadata dictionary and recovery marker.
    """
    # 1. Model an older trade object that has no auxiliary metadata yet.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock(safe_address="0xABC123")

    trade = MagicMock()
    trade.other_data = None
    trade.trade_id = 1

    # 2. Record the uncertain intermediate transfer location.
    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=10_000_000,
        location="hypercore_spot_or_perp",
    )

    # 3. The recovery marker is durable even without pre-existing metadata.
    assert trade.other_data is not None
    assert "hypercore_stranded_usdc" in trade.other_data


def test_phase1_at_risk_marker_is_written_before_broadcast() -> None:
    """Phase-1 deposit protection is persisted before an uncertain broadcast.

    1. Create a mocked live HyperCore routing and buy trade.
    2. Mark the phase-1 amount as at risk before broadcast.
    3. Verify generic failed-buy accounting will retain the allocation.
    """
    # 1. Use a mock because this marker must exist even if no RPC receipt arrives.
    routing = object.__new__(HypercoreVaultRouting)
    routing.lagoon_vault = MagicMock(safe_address="0xABC123")
    trade = MagicMock()
    trade.other_data = {}

    # 2. Persist the conservative phase-1 checkpoint.
    HypercoreVaultRouting._mark_deposit_capital_at_risk(
        routing,
        trade=trade,
        raw_amount=48_884_068,
    )

    # 3. A crash can no longer look like an unused reserve allocation.
    marker = trade.other_data["hypercore_deposit_capital_at_risk"]
    assert marker["amount_raw"] == 48_884_068
    assert marker["phase"] == "phase1_broadcast_pending"
    assert trade.other_data["retain_reserve_allocation_on_failure"] is True


def test_state_only_repair_defers_unreconciled_hypercore_deposit() -> None:
    """Repair must leave an unreconciled HyperCore deposit frozen without prompting.

    1. Create a failed trade carrying the HyperCore at-risk marker.
    2. Present it as the only frozen repair candidate to interactive repair.
    3. Verify repair creates no counter-trade, does not prompt or unfreeze, and
       leaves live reconciliation to correct-accounts.
    """
    # 1. This mock represents a crash before HyperCore receipt classification.
    state = MagicMock()
    position = MagicMock()
    position.position_id = 489
    state.portfolio.frozen_positions.values.return_value = [position]
    trade = MagicMock()
    trade.trade_id = 1486
    trade.position_id = 489
    trade.other_data = {"hypercore_deposit_capital_at_risk": {"amount_raw": 48_884_068}}

    # 2. State-only repair has no RPC/Info API evidence to resolve the location.
    with patch("tradeexecutor.state.repair.find_trades_to_be_repaired", return_value=[trade]), patch(
        "tradeexecutor.state.repair.unfreeze_position",
    ) as unfreeze_position, patch("builtins.input") as input_mock:
        # 3. It must defer rather than manufacture a counter-trade refund. This
        # lets repair unblock unrelated planned trades before correct-accounts
        # performs the Safe-level live recovery, without an unprompted state
        # change when this is the only repair candidate.
        result = repair_trades(state, attempt_repair=True, interactive=True)

    assert result.trades_needing_repair == [trade]
    assert result.new_trades == []
    assert result.unfrozen_positions == []
    input_mock.assert_not_called()
    unfreeze_position.assert_not_called()


def test_state_only_repair_repairs_unrelated_trade_while_deferring_hypercore_withdrawal() -> None:
    """Repair must clear independent failures without rewriting an ambiguous withdrawal.

    1. Model the production combination of one ordinary failed trade and one
       failed HyperCore withdrawal carrying its stranded-USDC marker.
    2. Run the state-only repair workflow without interactive confirmation.
    3. Verify only the ordinary trade receives a counter-trade and the
       HyperCore trade remains explicitly deferred with neutral diagnostics.

    Trades are mocked because this test isolates the command's selection rule;
    the real state-bookkeeping path is covered by the no-transaction regression
    in ``test_repair_trade_missing_tx.py``.
    """
    # 1. The protected trade models an ambiguous second-stage withdrawal,
    # while the ordinary trade models an independent accounting failure.
    state = MagicMock()
    frozen_position = MagicMock()
    frozen_position.position_id = 489
    state.portfolio.frozen_positions.values.return_value = [frozen_position]
    hypercore_trade = MagicMock()
    hypercore_trade.trade_id = 1605
    hypercore_trade.position_id = 489
    hypercore_trade.other_data = {
        "hypercore_stranded_usdc": {
            "amount_raw": 48_884_068,
            "location": "hypercore_perp_or_vault",
        },
    }
    ordinary_trade = MagicMock()
    ordinary_trade.trade_id = 4700
    ordinary_trade.position_id = 470
    ordinary_trade.other_data = {}
    counter_trade = MagicMock()

    # 2. Repair may create accounting entries only for the ordinary failure.
    with patch("tradeexecutor.state.repair.find_trades_to_be_repaired", return_value=[hypercore_trade, ordinary_trade]), patch(
        "tradeexecutor.state.repair.repair_trade",
        return_value=counter_trade,
    ) as repair_trade, patch("tradeexecutor.state.repair.logger.error") as logger_error, patch(
        "tradeexecutor.state.repair.logger.warning"
    ) as logger_warning:
        result = repair_trades(state, attempt_repair=True, interactive=False)

    # 3. The command completes, but never rewrites the withdrawal or calls it a deposit.
    repair_trade.assert_called_once_with(state.portfolio, ordinary_trade)
    assert result.trades_needing_repair == [hypercore_trade, ordinary_trade]
    assert result.new_trades == [counter_trade]
    message = logger_error.call_args.args[0]
    assert "HyperCore trade(s)" in message
    assert "deposit trade" not in message
    warning = logger_warning.call_args.args[0]
    assert "HyperCore trade" in warning
    assert "deposit" not in warning
