"""Test Hyperliquid closed-position clean-up orchestration.

Verifies that:
1. A broken HyperAI failed-close state is loaded from a real fixture.
2. Mocked Hyperliquid and HyperEVM balances produce only stranded-money recovery actions.
3. The clean-up flow executes ``perp -> spot`` and ``spot -> EVM`` in order.
4. The state repair and accounting correction hooks run and the state is treated as saveable.
"""

import shutil
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from hexbytes import HexBytes

from tradeexecutor.ethereum.vault import hyperliquid_cleanup
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore


def test_hyperliquid_cleanup_recovers_stranded_safe_balances(
    tmp_path: Path,
    monkeypatch,
):
    """Test the mocked HyperAI failed-close clean-up flow.

    1. Copy the broken HyperAI fixture state to a temporary location.
    2. Mock Hyperliquid and HyperEVM balance reads so the vault is already empty but USDC is stranded.
    3. Run the clean-up entrypoint and execute the mocked recovery actions.
    4. Verify the action plan, execution order, repair/correction hooks, and final save path.
    """
    fixture_path = Path(__file__).parent / "state" / "hyperai-cleanup.json"
    state_file = tmp_path / "hyperai-cleanup.json"
    shutil.copy(fixture_path, state_file)

    state = State.read_json_file(state_file)
    store = JSONFileStore(state_file)
    safe_address = "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"
    vault_address = "0x07fd993f0fa3a185f7207adccd29f7a87404689d"

    live_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0.5"),
        "spot_free": Decimal("0.5"),
        "perp_withdrawable": Decimal("6.5"),
        "perp_account_value": Decimal("6.5"),
        "vault_equity": Decimal("0"),
    }
    broadcast_order: list[str] = []

    reserve_token = MagicMock()
    reserve_token.fetch_balance_of.side_effect = lambda address: live_balances["evm_usdc"]
    reserve_token.convert_to_raw.side_effect = lambda amount: int((Decimal(amount) * Decimal(10**6)).to_integral_value())

    hot_wallet = MagicMock()

    def broadcast_side_effect(bound_func, gas_limit=hyperliquid_cleanup.SAFE_GAS_LIMIT):
        if bound_func == "perp_to_spot_fn":
            # Step 3: Simulate recovery from HyperCore perp back to spot.
            broadcast_order.append("perp_to_spot")
            amount = live_balances["perp_withdrawable"]
            live_balances["spot_total"] += amount
            live_balances["spot_free"] += amount
            live_balances["perp_withdrawable"] = Decimal("0")
            live_balances["perp_account_value"] = Decimal("0")
            return HexBytes("0x01")

        if bound_func == "spot_to_evm_fn":
            # Step 4: Simulate recovery from HyperCore spot back to HyperEVM.
            broadcast_order.append("spot_to_evm")
            amount = live_balances["spot_free"]
            live_balances["evm_usdc"] += amount
            live_balances["spot_total"] -= amount
            live_balances["spot_free"] -= amount
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
            margin_summary=SimpleNamespace(account_value=live_balances["perp_account_value"]),
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

    monkeypatch.setattr(hyperliquid_cleanup, "load_cleanup_context", lambda **kwargs: context)
    monkeypatch.setattr(hyperliquid_cleanup, "fetch_spot_clearinghouse_state", fake_fetch_spot)
    monkeypatch.setattr(hyperliquid_cleanup, "fetch_perp_clearinghouse_state", fake_fetch_perp)
    monkeypatch.setattr(hyperliquid_cleanup, "fetch_user_vault_equities", fake_fetch_vault_equities)
    monkeypatch.setattr(hyperliquid_cleanup, "build_hypercore_transfer_usd_class_call", lambda *args, **kwargs: "perp_to_spot_fn")
    monkeypatch.setattr(hyperliquid_cleanup, "build_hypercore_send_asset_to_evm_call", lambda *args, **kwargs: "spot_to_evm_fn")
    monkeypatch.setattr(hyperliquid_cleanup, "assert_transaction_success_with_explanation", lambda *args, **kwargs: None)
    monkeypatch.setattr(hyperliquid_cleanup, "repair_trades", repair_wrapper)
    monkeypatch.setattr(
        hyperliquid_cleanup,
        "_build_accounting_correction_context",
        lambda _context: hyperliquid_cleanup.HyperliquidAccountingContext(
            pair_universe=MagicMock(),
            reserve_assets=[MagicMock()],
            sync_model=SimpleNamespace(web3=SimpleNamespace(), create_transaction_builder=lambda: MagicMock()),
            tx_builder=MagicMock(),
            strategy_universe=MagicMock(),
            pricing_model=MagicMock(),
        ),
    )
    monkeypatch.setattr(hyperliquid_cleanup, "calculate_account_corrections", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(hyperliquid_cleanup, "correct_accounts", fake_correct_accounts)
    monkeypatch.setattr(hyperliquid_cleanup, "check_state_internal_coherence", lambda state: None)
    monkeypatch.setattr(hyperliquid_cleanup, "check_accounts", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(hyperliquid_cleanup, "get_almost_latest_block_number", lambda web3: 123456)

    # Step 1: Run the clean-up flow end to end with mocked live balances.
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

    # Step 2: Confirm the action planner only produced stranded-money recovery legs.
    assert [action.action_kind for action in report.planned_actions] == [
        "perp_to_spot",
        "spot_to_evm",
    ]
    assert "vault_to_perp" not in {action.action_kind for action in report.planned_actions}

    # Step 3: Confirm the mocked broadcasts executed in the expected order.
    assert broadcast_order == ["perp_to_spot", "spot_to_evm"]
    assert report.executed_action_kinds == ["perp_to_spot", "spot_to_evm"]

    # Step 4: Confirm repair, account correction, and final save completed.
    assert repair_calls == ["repair"]
    assert correct_calls == ["correct"]
    assert report.accounts_clean is True
    assert report.state_saved is True
    assert state_file.with_suffix(".backup-1.json").exists()
