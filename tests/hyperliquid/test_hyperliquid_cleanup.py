"""Test Hyperliquid closed-position clean-up orchestration.

Verifies that:
1. A broken HyperAI failed-close state is loaded from a real fixture.
2. An open-in-state but effectively closed-in-reality vault is recognised as recoverable.
3. Mocked Hyperliquid and HyperEVM balances produce only stranded-money recovery actions.
4. The clean-up flow executes ``perp -> spot`` and ``spot -> EVM`` in order.
5. The state repair and accounting correction hooks run and the state is treated as saveable.
6. The spot -> EVM withdrawal leaves the configured HyperCore spot dust.
7. A small spot balance (below fee margin) raises a clear operator error.
"""

import shutil
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hexbytes import HexBytes

from tradeexecutor.ethereum.vault import hyperliquid_cleanup
from tradeexecutor.ethereum.vault import hypercore_transit_recovery
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore


def test_hyperliquid_cleanup_recovers_stranded_safe_balances(
    tmp_path: Path,
    monkeypatch,
):
    """Test the mocked HyperAI residual-vault-equity clean-up flow.

    1. Copy the broken HyperAI fixture state to a temporary location and force the position to look open in state.
    2. Mock Hyperliquid and HyperEVM balance reads so the live vault equity is only residual but USDC is stranded.
    3. Run the clean-up entrypoint and execute the mocked recovery actions.
    4. Verify the new residual-vault-equity classification and planned action set.
    5. Verify the execution order, repair/correction hooks, and final save path.
    """
    fixture_path = Path(__file__).parent / "state" / "hyperai-cleanup.json"
    state_file = tmp_path / "hyperai-cleanup.json"
    shutil.copy(fixture_path, state_file)

    state = State.read_json_file(state_file)

    # Step 1: Force the fixture into the console shape: open in state with no trusted failed-sell marker.
    position = state.portfolio.frozen_positions.pop(1)
    position.frozen_at = None
    position.unfrozen_at = None
    state.portfolio.open_positions[position.position_id] = position
    failed_trade = position.trades[2]
    failed_trade.failed_at = None
    failed_trade.executed_at = None
    state_file.write_text(state.to_json())

    store = JSONFileStore(state_file)
    safe_address = "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"
    vault_address = "0x07fd993f0fa3a185f7207adccd29f7a87404689d"

    live_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("1"),
        "spot_free": Decimal("1"),
        "perp_withdrawable": Decimal("6.99345"),
        "perp_account_value": Decimal("6.99345"),
        "vault_equity": Decimal("0.065975"),
        "perp_position_count": 0,
    }
    broadcast_order: list[str] = []
    captured_evm_usdc_amounts: list[int] = []

    reserve_token = MagicMock()
    reserve_token.fetch_balance_of.side_effect = lambda address: live_balances[
        "evm_usdc"
    ]
    reserve_token.convert_to_raw.side_effect = lambda amount: int(
        (Decimal(amount) * Decimal(10**6)).to_integral_value()
    )

    hot_wallet = MagicMock()

    def broadcast_side_effect(bound_func, gas_limit=hyperliquid_cleanup.SAFE_GAS_LIMIT):
        if isinstance(bound_func, tuple) and bound_func[0] == "perp_to_spot_fn":
            # Step 3: Simulate recovery from HyperCore perp back to spot.
            broadcast_order.append("perp_to_spot")
            amount = Decimal(bound_func[1]) / Decimal(10**6)
            live_balances["spot_total"] += amount
            live_balances["spot_free"] += amount
            live_balances["perp_withdrawable"] -= amount
            live_balances["perp_account_value"] -= amount
            return HexBytes("0x01")

        if isinstance(bound_func, tuple) and bound_func[0] == "spot_to_evm_fn":
            # Step 4: Simulate recovery from HyperCore spot back to HyperEVM.
            # Only the fee-adjusted amount arrives on EVM; the remainder stays
            # on spot to cover the bridge fee.
            broadcast_order.append("spot_to_evm")
            bridged_amount = Decimal(bound_func[1]) / Decimal(10**6)
            live_balances["evm_usdc"] += bridged_amount
            live_balances["spot_total"] -= bridged_amount
            live_balances["spot_free"] -= bridged_amount
            return HexBytes("0x02")

        raise AssertionError(f"Unexpected bound function: {bound_func}")

    hot_wallet.transact_and_broadcast_with_contract.side_effect = broadcast_side_effect

    lagoon_vault = SimpleNamespace(
        safe_address=safe_address,
        address="0x282cB588099844Dc93C0B7bd6701298666Ee76bE",
        trading_strategy_module_address="0xAf4e8d50dA5Aa49Eee8cf04fc4682d5c090902E7",
    )

    context = hyperliquid_cleanup.HyperliquidCleanupContext(
        state_file=state_file,
        strategy_file=Path("strategies/hyper-ai.py"),
        store=store,
        state=state,
        web3=MagicMock(),
        hot_wallet=hot_wallet,
        lagoon_vault=lagoon_vault,
        sync_model=MagicMock(),
        session=object(),
        reserve_token=reserve_token,
        trading_strategy_api_key="test-key",
        json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
        cache_path=None,
        unit_testing=False,
    )

    def fake_fetch_spot(_session, user: str):
        assert user == safe_address
        return SimpleNamespace(
            balances=[
                SimpleNamespace(
                    coin="USDC",
                    total=live_balances["spot_total"],
                    hold=live_balances["spot_total"] - live_balances["spot_free"],
                )
            ]
        )

    def fake_fetch_perp(_session, user: str):
        assert user == safe_address
        return SimpleNamespace(
            withdrawable=live_balances["perp_withdrawable"],
            margin_summary=SimpleNamespace(
                account_value=live_balances["perp_account_value"]
            ),
            asset_positions=[
                SimpleNamespace() for _ in range(live_balances["perp_position_count"])
            ],
        )

    def fake_fetch_vault_equities(_session, user: str):
        assert user == safe_address
        return [
            SimpleNamespace(
                vault_address=vault_address,
                equity=live_balances["vault_equity"],
            )
        ]

    repair_calls = []
    original_repair_trades = hyperliquid_cleanup.repair_trades

    def repair_wrapper(*args, **kwargs):
        repair_calls.append("repair")
        return original_repair_trades(*args, **kwargs)

    correct_calls = []

    def fake_correct_accounts(*args, **kwargs):
        correct_calls.append("correct")
        return iter([object()])

    monkeypatch.setattr(
        hyperliquid_cleanup, "load_cleanup_context", lambda **kwargs: context
    )
    monkeypatch.setattr(
        hypercore_transit_recovery, "fetch_spot_clearinghouse_state", fake_fetch_spot
    )
    monkeypatch.setattr(
        hypercore_transit_recovery, "fetch_perp_clearinghouse_state", fake_fetch_perp
    )
    monkeypatch.setattr(
        hyperliquid_cleanup, "fetch_user_vault_equities", fake_fetch_vault_equities
    )

    def fake_build_hypercore_transfer_usd_class_call(
        lagoon_vault_arg,
        hypercore_usdc_amount: int,
        to_perp: bool,
    ):
        assert lagoon_vault_arg is lagoon_vault
        assert isinstance(hypercore_usdc_amount, int)
        assert to_perp is False
        return ("perp_to_spot_fn", hypercore_usdc_amount)

    def fake_build_hypercore_send_asset_to_evm_call(
        lagoon_vault_arg,
        evm_usdc_amount: int,
    ):
        assert lagoon_vault_arg is lagoon_vault
        assert isinstance(evm_usdc_amount, int)
        captured_evm_usdc_amounts.append(evm_usdc_amount)
        return ("spot_to_evm_fn", evm_usdc_amount)

    monkeypatch.setattr(
        hypercore_transit_recovery,
        "build_hypercore_transfer_usd_class_call",
        fake_build_hypercore_transfer_usd_class_call,
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "build_hypercore_send_asset_to_evm_call",
        fake_build_hypercore_send_asset_to_evm_call,
    )
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "assert_transaction_success_with_explanation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(hyperliquid_cleanup, "repair_trades", repair_wrapper)
    monkeypatch.setattr(
        hyperliquid_cleanup,
        "_build_accounting_correction_context",
        lambda _context: hyperliquid_cleanup.HyperliquidAccountingContext(
            pair_universe=MagicMock(),
            reserve_assets=[MagicMock()],
            sync_model=SimpleNamespace(
                web3=SimpleNamespace(), create_transaction_builder=lambda: MagicMock()
            ),
            tx_builder=MagicMock(),
            strategy_universe=MagicMock(),
            pricing_model=MagicMock(),
        ),
    )
    monkeypatch.setattr(
        hyperliquid_cleanup,
        "calculate_account_corrections",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(hyperliquid_cleanup, "correct_accounts", fake_correct_accounts)
    monkeypatch.setattr(
        hyperliquid_cleanup, "check_state_internal_coherence", lambda state: None
    )
    monkeypatch.setattr(
        hyperliquid_cleanup, "check_accounts", lambda *args, **kwargs: (True, None)
    )
    monkeypatch.setattr(
        hyperliquid_cleanup, "get_almost_latest_block_number", lambda web3: 123456
    )

    # Step 2: Run the clean-up flow end to end with mocked live balances.
    report = hyperliquid_cleanup.run_hyperliquid_cleanup(
        state_file=state_file,
        strategy_file=Path("strategies/hyper-ai.py"),
        private_key="0x123",
        json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
        vault_address=lagoon_vault.address,
        vault_adapter_address=lagoon_vault.trading_strategy_module_address,
        trading_strategy_api_key="test-key",
        auto_approve=True,
        unit_testing=False,
    )

    # Step 3: Confirm the open-in-state residual-vault-equity case is classified as recoverable.
    assert len(report.comparison_rows) == 1
    assert (
        report.comparison_rows[0].classification
        == "open_in_state_residual_vault_equity_stranded"
    )

    # Step 4: Confirm the action planner only produced stranded-money recovery legs.
    assert [action.action_kind for action in report.planned_actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert "vault_to_perp" not in {
        action.action_kind for action in report.planned_actions
    }

    # Step 5: Confirm the mocked broadcasts executed in the expected order.
    assert broadcast_order == ["perp_to_spot", "spot_to_evm"]
    assert report.executed_action_kinds == ["perp_to_spot", "spot_to_evm"]

    # Step 6: Confirm repair, account correction, and final save completed.
    assert repair_calls == ["repair"]
    assert correct_calls == ["correct"]
    assert report.accounts_clean is True
    assert report.state_saved is True
    assert state_file.with_suffix(".backup-1.json").exists()

    # Step 7: Confirm the spot->EVM withdrawal leaves the configured spot dust.
    # Total spot after perp->spot recovery: 1 + (6.99345 - 0.50) = 7.49345 USDC.
    # The withdrawal should be 7.49345 - 0.50 = 6.99345 USDC
    # = 6_993_450 raw (6 decimals).
    assert len(captured_evm_usdc_amounts) == 1
    total_spot = Decimal("1") + Decimal("6.99345") - Decimal("0.50")
    expected_raw = int(
        (
            (total_spot - Decimal("0.50")) * Decimal(10**6)
        ).to_integral_value()
    )
    assert captured_evm_usdc_amounts[0] == expected_raw


def test_hyperliquid_cleanup_raises_on_dust_spot_balance(monkeypatch):
    """Spot balance at or below the bridge fee margin raises a clear operator error.

    1. Build a minimal HyperliquidCleanupContext with a dust-level spot balance.
    2. Call _execute_spot_to_evm directly with an amount equal to the spot balance.
    3. Assert that it raises RuntimeError instead of broadcasting a zero-amount withdrawal.

    We test _execute_spot_to_evm directly because the full cleanup flow would
    classify a dust spot balance as manual_review_required and never reach this code.
    """
    safe_address = "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"
    dust_balance = Decimal("0.005")

    # Step 1: Mock the live snapshot to return dust spot balance.
    def fake_fetch_spot(_session, user: str):
        return SimpleNamespace(
            balances=[SimpleNamespace(coin="USDC", total=dust_balance, hold=Decimal(0))]
        )

    def fake_fetch_perp(_session, user: str):
        return SimpleNamespace(
            withdrawable=Decimal(0),
            margin_summary=SimpleNamespace(account_value=Decimal(0)),
            asset_positions=[],
        )

    def fake_fetch_vault_equities(_session, user: str):
        return []

    monkeypatch.setattr(
        hypercore_transit_recovery, "fetch_spot_clearinghouse_state", fake_fetch_spot
    )
    monkeypatch.setattr(
        hypercore_transit_recovery, "fetch_perp_clearinghouse_state", fake_fetch_perp
    )

    reserve_token = MagicMock()
    reserve_token.fetch_balance_of.side_effect = lambda address: Decimal("1")

    lagoon_vault = SimpleNamespace(safe_address=safe_address)

    context = hyperliquid_cleanup.HyperliquidCleanupContext(
        state_file=Path("/dev/null"),
        strategy_file=Path("strategies/hyper-ai.py"),
        store=MagicMock(),
        state=MagicMock(),
        web3=MagicMock(),
        hot_wallet=MagicMock(),
        lagoon_vault=lagoon_vault,
        sync_model=MagicMock(),
        session=object(),
        reserve_token=reserve_token,
        trading_strategy_api_key="test-key",
        json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
        cache_path=None,
        unit_testing=False,
    )

    # Step 2: Call _execute_spot_to_evm with dust-level amount.
    # Step 3: Assert RuntimeError because spot is below fee margin.
    with pytest.raises(
        RuntimeError, match="too small to cover the HyperCore bridge fee margin"
    ):
        hyperliquid_cleanup._execute_spot_to_evm(context, dust_balance)


def test_hyperliquid_cleanup_allows_live_vault_rows_when_stranded_recovery_exists():
    """Mixed live-vault and stranded-balance rows should still plan Safe-side recovery.

    1. Build one genuinely live vault row and one closed-in-reality stranded row.
    2. Run the clean-up action planner for a Safe with stranded HyperCore perp USDC.
    3. Assert the planner ignores the live vault row and keeps the stranded recovery actions.
    """
    comparison_rows = [
        hyperliquid_cleanup.HyperliquidCleanupComparisonRow(
            position_id=34,
            state_status="frozen",
            vault_name="Still live vault",
            vault_address="0x0000000000000000000000000000000000000034",
            state_quantity=Decimal("591.373"),
            live_vault_equity=Decimal("0.47309"),
            reserve_quantity=Decimal("110.635"),
            spot_free_usdc=Decimal("0.009252"),
            perp_withdrawable=Decimal("4177.73"),
            perp_position_count=0,
            failed_trade_ids=[65],
            classification="live_vault_open_no_action",
        ),
        hyperliquid_cleanup.HyperliquidCleanupComparisonRow(
            position_id=33,
            state_status="frozen",
            vault_name="Closed in reality",
            vault_address="0x0000000000000000000000000000000000000033",
            state_quantity=Decimal("591.373"),
            live_vault_equity=Decimal("0.099999"),
            reserve_quantity=Decimal("110.635"),
            spot_free_usdc=Decimal("0.009252"),
            perp_withdrawable=Decimal("4177.73"),
            perp_position_count=0,
            failed_trade_ids=[64],
            classification="closed_in_reality_perp_stranded",
        ),
    ]
    live_snapshot = hyperliquid_cleanup.HyperliquidCleanupSnapshot(
        safe_address="0xB136581dFB3efA76Ae71293C1A70942f0726E8fD",
        evm_usdc_balance=Decimal("110.635"),
        spot_total_usdc=Decimal("0.009252"),
        spot_free_usdc=Decimal("0.009252"),
        perp_withdrawable=Decimal("4177.73"),
        perp_account_value=Decimal("4177.73"),
        perp_position_count=0,
        vault_equities={
            "0x0000000000000000000000000000000000000034": Decimal("0.47309"),
            "0x0000000000000000000000000000000000000033": Decimal("0.099999"),
        },
    )

    # Step 1: Build a mixed live-vault and stranded-balance comparison.
    # Step 2: Plan the clean-up actions from the Safe-level balances.
    actions = hyperliquid_cleanup._plan_cleanup_actions(comparison_rows, live_snapshot)

    # Step 3: Confirm we still recover the stranded Safe balances.
    assert [action.action_kind for action in actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert actions[0].amount == Decimal("4177.23")
    assert actions[1].amount == Decimal("4176.739252")


def test_wait_for_spot_free_balance_accepts_threshold_instead_of_exact_match(
    monkeypatch,
):
    """Cleanup spot wait should accept threshold arrival instead of exact final balance.

    1. Mock HyperCore spot balance slightly above the threshold implied by the expected increase.
    2. Wait for the cleanup helper to confirm the spot balance increase.
    3. Verify the helper accepts the observed balance without requiring an exact final match.
    """

    # Step 1: Mock HyperCore spot balance slightly above the threshold implied by the expected increase.
    monkeypatch.setattr(
        hypercore_transit_recovery,
        "fetch_spot_clearinghouse_state",
        lambda _session, user: SimpleNamespace(
            balances=[
                SimpleNamespace(
                    coin="USDC",
                    total=Decimal("11.497"),
                    hold=Decimal("0"),
                )
            ]
        ),
    )

    # Step 2: Wait for the cleanup helper to confirm the spot balance increase.
    result = hyperliquid_cleanup._wait_for_spot_free_balance(
        session=object(),
        user="0x123",
        baseline_balance=Decimal("10.0"),
        expected_increase=Decimal("1.5"),
        timeout=0.1,
        poll_interval=0.01,
    )

    # Step 3: Verify the helper accepts the observed balance without requiring an exact final match.
    assert result == Decimal("11.497")


def test_wait_for_evm_usdc_balance_accepts_threshold_instead_of_exact_match():
    """Cleanup EVM wait should accept threshold arrival instead of exact final balance.

    1. Mock a token balance slightly above the threshold implied by the expected bridged increase.
    2. Wait for the cleanup helper to confirm the EVM USDC balance increase.
    3. Verify the helper accepts the observed balance without requiring an exact final match.
    """
    token = MagicMock()

    # Step 1: Mock a token balance slightly above the threshold implied by the expected bridged increase.
    token.fetch_balance_of.return_value = Decimal("108.992")

    # Step 2: Wait for the cleanup helper to confirm the EVM USDC balance increase.
    result = hyperliquid_cleanup._wait_for_evm_usdc_balance(
        token=token,
        address="0x123",
        baseline_balance=Decimal("100.0"),
        expected_increase=Decimal("9.0"),
        timeout=0.1,
        poll_interval=0.01,
    )

    # Step 3: Verify the helper accepts the observed balance without requiring an exact final match.
    assert result == Decimal("108.992")
