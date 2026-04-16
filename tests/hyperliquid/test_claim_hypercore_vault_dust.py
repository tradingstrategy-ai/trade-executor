"""Test claiming untracked Hypercore vault dust."""

import datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from eth_defi.hyperliquid.api import UserVaultEquity
from hexbytes import HexBytes
from tradingstrategy.chain import ChainId
from typer.testing import CliRunner
from web3 import Web3

from tradeexecutor.ethereum.vault import hypercore_dust_claim
from tradeexecutor.ethereum.vault.hypercore_dust_claim import (
    HYPERCORE_DUST_CLAIM_NOTE,
)
from tradeexecutor.ethereum.vault.hypercore_routing import (
    HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
    raw_to_usdc,
)
from tradeexecutor.ethereum.vault.hyperliquid_cleanup import HyperliquidCleanupContext
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType


SAFE_ADDRESS = "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"
VAULT_ADDRESS = "0x07fd993f0fa3a185f7207adccd29f7a87404689d"
OTHER_VAULT_ADDRESS = "0x1111111111111111111111111111111111111111"
NOW = datetime.datetime(2026, 4, 16, 12, 0, 0)


class MemoryStore:
    """Small state store for tests."""

    def __init__(self, state: State):
        self.state = state
        self.sync_count = 0
        self.path = Path("memory-state.json")

    def load(self) -> State:
        return self.state

    def sync(self, state: State):
        self.state = state
        self.sync_count += 1


def _make_reserve_asset() -> AssetIdentifier:
    """Create a Hypercore USDC reserve asset for tests."""
    return AssetIdentifier(
        chain_id=ChainId.hypercore.value,
        address="0x2222222222222222222222222222222222222222",
        token_symbol="USDC",
        decimals=6,
    )


def _make_state() -> State:
    """Create a state with initialised reserves."""
    state = State()
    state.portfolio.initialise_reserves(_make_reserve_asset(), reserve_token_price=1.0)
    return state


def _make_equity(
    vault_address: str = VAULT_ADDRESS,
    equity: Decimal = Decimal("10"),
    locked_until: datetime.datetime | None = None,
) -> UserVaultEquity:
    """Create a Hypercore vault equity row."""
    if locked_until is None:
        locked_until = NOW - datetime.timedelta(days=1)
    return UserVaultEquity(
        vault_address=vault_address,
        equity=equity,
        locked_until=locked_until,
    )


def _make_context(state: State, live_balances: dict) -> HyperliquidCleanupContext:
    """Create a mocked Hypercore dust claim context."""
    reserve_token = MagicMock()
    reserve_token.fetch_balance_of.side_effect = lambda address: live_balances[
        "evm_usdc"
    ]
    reserve_token.convert_to_raw.side_effect = lambda amount: int(
        Decimal(amount) * Decimal(10**6)
    )

    hot_wallet = MagicMock()
    lagoon_vault = SimpleNamespace(
        safe_address=SAFE_ADDRESS,
        address="0x282cB588099844Dc93C0B7bd6701298666Ee76bE",
        trading_strategy_module_address="0xAf4e8d50dA5Aa49Eee8cf04fc4682d5c090902E7",
    )
    return HyperliquidCleanupContext(
        state_file=Path("state.json"),
        strategy_file=Path("strategies/hyper-ai.py"),
        store=MemoryStore(state),
        state=state,
        web3=MagicMock(),
        hot_wallet=hot_wallet,
        lagoon_vault=lagoon_vault,
        sync_model=MagicMock(),
        session=object(),
        reserve_token=reserve_token,
        trading_strategy_api_key="",
        json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
        cache_path=None,
        unit_testing=True,
    )


def _patch_live_reads(
    monkeypatch,
    live_balances: dict,
    equities: list[UserVaultEquity],
    max_withdrawable: Decimal = Decimal("10"),
):
    """Patch Hyperliquid reads for a deterministic mocked account."""

    def fake_fetch_spot(_session, user: str):
        assert user == SAFE_ADDRESS
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
        assert user == SAFE_ADDRESS
        return SimpleNamespace(
            withdrawable=live_balances["perp_withdrawable"],
            asset_positions=[
                SimpleNamespace() for _ in range(live_balances["perp_position_count"])
            ],
        )

    class FakeVault:
        def __init__(self, session, vault_address: str):
            self.vault_address = vault_address

        def fetch_info(self, user: str):
            assert user == SAFE_ADDRESS
            return SimpleNamespace(
                name=f"Vault {self.vault_address[-4:]}",
                max_withdrawable=max_withdrawable,
            )

    monkeypatch.setattr(
        hypercore_dust_claim, "fetch_spot_clearinghouse_state", fake_fetch_spot
    )
    monkeypatch.setattr(
        hypercore_dust_claim, "fetch_perp_clearinghouse_state", fake_fetch_perp
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "fetch_user_vault_equities",
        lambda _session, user: equities,
    )
    monkeypatch.setattr(hypercore_dust_claim, "HyperliquidVault", FakeVault)


def _patch_execution(
    monkeypatch, live_balances: dict, context: HyperliquidCleanupContext
):
    """Patch CoreWriter calls and transaction receipts."""
    broadcast_order = []

    def fake_withdraw_from_vault_call(
        lagoon_vault, vault_address: str, hypercore_usdc_amount: int
    ):
        return ("vault_to_perp", hypercore_usdc_amount)

    def fake_transfer_usd_class_call(
        lagoon_vault, hypercore_usdc_amount: int, to_perp: bool
    ):
        assert to_perp is False
        return ("perp_to_spot", hypercore_usdc_amount)

    def fake_send_asset_to_evm_call(lagoon_vault, evm_usdc_amount: int):
        return ("spot_to_evm", evm_usdc_amount)

    def broadcast_side_effect(
        bound_func, gas_limit=hypercore_dust_claim.HYPERCORE_MULTICALL_GAS
    ):
        action, raw_amount = bound_func
        broadcast_order.append(action)
        amount = raw_to_usdc(raw_amount)
        if action == "vault_to_perp":
            live_balances["perp_withdrawable"] += amount
            return HexBytes("0x01")
        if action == "perp_to_spot":
            live_balances["perp_withdrawable"] -= amount
            live_balances["spot_total"] += amount
            live_balances["spot_free"] += amount
            return HexBytes("0x02")
        if action == "spot_to_evm":
            live_balances["spot_total"] = Decimal("0")
            live_balances["spot_free"] = Decimal("0")
            live_balances["evm_usdc"] += amount
            return HexBytes("0x03")
        raise AssertionError(f"Unexpected action {action}")

    monkeypatch.setattr(
        hypercore_dust_claim,
        "build_hypercore_withdraw_from_vault_call",
        fake_withdraw_from_vault_call,
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "build_hypercore_transfer_usd_class_call",
        fake_transfer_usd_class_call,
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "build_hypercore_send_asset_to_evm_call",
        fake_send_asset_to_evm_call,
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "assert_transaction_success_with_explanation",
        lambda *args, **kwargs: {"status": 1, "blockNumber": 123},
    )
    monkeypatch.setattr(
        hypercore_dust_claim, "get_block_timestamp", lambda web3, block_number: NOW
    )
    context.hot_wallet.transact_and_broadcast_with_contract.side_effect = (
        broadcast_side_effect
    )
    return broadcast_order


def test_claim_hypercore_vault_dust_persists_actual_delta_before_later_failure(
    monkeypatch,
):
    """Claim Hypercore dust and persist state before a later claim fails.

    1. Create two live claimable vault equities and a reserve-only state.
    2. Run the mocked claim flow, forcing the second vault to fail.
    3. Verify the first vault went through all phases and saved actual reserve delta accounting.
    """
    # 1. Create two live claimable vault equities and a reserve-only state.
    state = _make_state()
    live_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 0,
    }
    context = _make_context(state, live_balances)
    equities = [
        _make_equity(vault_address=VAULT_ADDRESS, equity=Decimal("10")),
        _make_equity(vault_address=OTHER_VAULT_ADDRESS, equity=Decimal("10")),
    ]
    _patch_live_reads(monkeypatch, live_balances, equities)
    broadcast_order = _patch_execution(monkeypatch, live_balances, context)
    monkeypatch.setattr(
        hypercore_dust_claim, "load_cleanup_context", lambda **kwargs: context
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "backup_state",
        lambda *args, **kwargs: (context.store, context.state),
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "_wait_for_evm_usdc_balance",
        lambda *args, **kwargs: Decimal("8.12"),
    )

    original_execute = hypercore_dust_claim.execute_hypercore_dust_claim

    def execute_once_then_fail(context_arg, candidate):
        if candidate.vault_address == Web3.to_checksum_address(OTHER_VAULT_ADDRESS):
            raise RuntimeError("second claim failed")
        return original_execute(context_arg, candidate)

    monkeypatch.setattr(
        hypercore_dust_claim, "execute_hypercore_dust_claim", execute_once_then_fail
    )

    # 2. Run the mocked claim flow, forcing the second vault to fail.
    with pytest.raises(RuntimeError, match="second claim failed"):
        hypercore_dust_claim.run_hypercore_dust_claim(
            state_file=Path("state.json"),
            strategy_file=Path("strategy.py"),
            private_key="0x123",
            json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
            vault_address="0x1",
            vault_adapter_address="0x2",
            auto_approve=True,
            max_claim_usdc=Decimal("25"),
            unit_testing=True,
        )

    # 3. Verify the first vault went through all phases and saved actual reserve delta accounting.
    reserve = state.portfolio.get_default_reserve_position()
    event = next(iter(reserve.balance_updates.values()))
    assert broadcast_order == ["vault_to_perp", "perp_to_spot", "spot_to_evm"]
    assert context.store.sync_count == 1
    assert reserve.quantity == Decimal("8.12")
    assert event.notes == HYPERCORE_DUST_CLAIM_NOTE
    assert event.quantity == Decimal("8.12")
    assert (
        state.sync.accounting.balance_update_refs[-1].balance_event_id
        == event.balance_update_id
    )


def test_claim_hypercore_vault_dust_safety_guards_and_cli_registration(
    monkeypatch,
):
    """Verify Hypercore dust claim safety guards and command registration.

    1. Verify active perp positions abort the command before broadcasting.
    2. Verify open-state, above-cap, below-floor, and re-read stale-state candidates are not executed.
    3. Verify the CLI command is exported and the Typer app can build the command tree.
    """
    # 1. Verify active perp positions abort the command before broadcasting.
    active_state = _make_state()
    active_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 1,
    }
    active_context = _make_context(active_state, active_balances)
    _patch_live_reads(monkeypatch, active_balances, [_make_equity()])
    monkeypatch.setattr(
        hypercore_dust_claim, "load_cleanup_context", lambda **kwargs: active_context
    )
    with pytest.raises(RuntimeError, match="active HyperCore perp positions"):
        hypercore_dust_claim.run_hypercore_dust_claim(
            state_file=Path("state.json"),
            strategy_file=Path("strategy.py"),
            private_key="0x123",
            json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
            vault_address="0x1",
            vault_adapter_address="0x2",
            auto_approve=True,
            max_claim_usdc=Decimal("25"),
            unit_testing=True,
        )
    assert (
        active_context.hot_wallet.transact_and_broadcast_with_contract.call_count == 0
    )

    # 2. Verify open-state, above-cap, below-floor, and re-read stale-state candidates are not executed.
    reserve_asset = _make_reserve_asset()

    open_state = _make_state()
    open_pair = create_hypercore_vault_pair(reserve_asset, vault_address=VAULT_ADDRESS)
    open_state.create_trade(
        strategy_cycle_at=NOW,
        pair=open_pair,
        quantity=None,
        reserve=Decimal("10"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    open_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 0,
    }
    open_context = _make_context(open_state, open_balances)
    _patch_live_reads(monkeypatch, open_balances, [_make_equity()])
    open_candidate = hypercore_dust_claim.discover_hypercore_dust_candidates(
        open_context, Decimal("25")
    )[0]
    assert open_candidate.status == "open_in_state"
    assert not open_candidate.is_claimable

    above_cap_state = _make_state()
    above_cap_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 0,
    }
    above_cap_context = _make_context(above_cap_state, above_cap_balances)
    _patch_live_reads(
        monkeypatch,
        above_cap_balances,
        [_make_equity(equity=Decimal("30"))],
        max_withdrawable=Decimal("30"),
    )
    monkeypatch.setattr(
        hypercore_dust_claim,
        "load_cleanup_context",
        lambda **kwargs: above_cap_context,
    )
    with pytest.raises(RuntimeError, match="auto-approve refused"):
        hypercore_dust_claim.run_hypercore_dust_claim(
            state_file=Path("state.json"),
            strategy_file=Path("strategy.py"),
            private_key="0x123",
            json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
            vault_address="0x1",
            vault_adapter_address="0x2",
            auto_approve=True,
            max_claim_usdc=Decimal("25"),
            unit_testing=True,
        )
    assert (
        above_cap_context.hot_wallet.transact_and_broadcast_with_contract.call_count
        == 0
    )

    below_floor_state = _make_state()
    below_floor_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 0,
    }
    below_floor_context = _make_context(below_floor_state, below_floor_balances)
    below_floor_equity = raw_to_usdc(HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW) + Decimal(
        "4.99"
    )
    _patch_live_reads(
        monkeypatch,
        below_floor_balances,
        [_make_equity(equity=below_floor_equity)],
        max_withdrawable=below_floor_equity,
    )
    below_floor_candidate = hypercore_dust_claim.discover_hypercore_dust_candidates(
        below_floor_context, Decimal("25")
    )[0]
    assert below_floor_candidate.status == "below_floor"
    assert below_floor_candidate.safe_raw_claim_amount < 5_000_000

    reread_state = _make_state()
    reread_balances = {
        "evm_usdc": Decimal("1"),
        "spot_total": Decimal("0"),
        "spot_free": Decimal("0"),
        "perp_withdrawable": Decimal("0"),
        "perp_position_count": 0,
    }
    reread_context = _make_context(reread_state, reread_balances)
    _patch_live_reads(monkeypatch, reread_balances, [_make_equity()])
    monkeypatch.setattr(
        hypercore_dust_claim, "load_cleanup_context", lambda **kwargs: reread_context
    )

    def add_position_on_reload():
        reread_pair = create_hypercore_vault_pair(
            reserve_asset, vault_address=VAULT_ADDRESS
        )
        if not reread_state.portfolio.open_positions:
            reread_state.create_trade(
                strategy_cycle_at=NOW,
                pair=reread_pair,
                quantity=None,
                reserve=Decimal("10"),
                assumed_price=1.0,
                trade_type=TradeType.rebalance,
                reserve_currency=reserve_asset,
                reserve_currency_price=1.0,
            )
        return reread_state

    reread_context.store.load = add_position_on_reload
    reread_report = hypercore_dust_claim.run_hypercore_dust_claim(
        state_file=Path("state.json"),
        strategy_file=Path("strategy.py"),
        private_key="0x123",
        json_rpc_hyperliquid="https://rpc.hyperliquid.xyz/evm",
        vault_address="0x1",
        vault_adapter_address="0x2",
        auto_approve=True,
        max_claim_usdc=Decimal("25"),
        unit_testing=True,
    )
    assert len(reread_report.executed_claims) == 0
    assert (
        reread_context.hot_wallet.transact_and_broadcast_with_contract.call_count == 0
    )

    # 3. Verify the CLI command is exported and the Typer app can build the command tree.
    from tradeexecutor.cli import main

    assert main.claim_hypercore_vault_dust in set(main.__all__)
    runner = CliRunner()
    assert runner.invoke(main.app, ["version"]).exit_code == 0
    claim_help = runner.invoke(main.app, ["claim-hypercore-vault-dust", "--help"])
    assert claim_help.exit_code == 0

    from typer.main import get_command

    click_command = get_command(main.app).commands["claim-hypercore-vault-dust"]
    max_claim_params = [
        param for param in click_command.params if param.name == "max_claim_usdc"
    ]
    assert len(max_claim_params) == 1
    assert max_claim_params[0].opts == ["--max-claim-usdc"]
    assert str(max_claim_params[0].type) == "STRING"
