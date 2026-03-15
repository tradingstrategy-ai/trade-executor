"""Focused regression tests for recent refactor helper seams."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from tradeexecutor.cli.commands.lagoon_deploy_vault import (
    _calculate_safe_threshold,
    _normalize_multisig_owners,
)
from tradeexecutor.cli.commands.lagoon_utils import (
    resolve_state_store,
    sync_reserve_balance_to_state,
)
from tradeexecutor.ethereum.lagoon.universe_config import (
    HYPEREVM_CHAIN_ID,
    normalise_deployment_chain_id,
    translate_trading_universe_to_lagoon_config,
)
from tradeexecutor.exchange_account import utils as exchange_account_utils
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State


class DummyPair:
    """Minimal pair object for helper-focused unit tests."""

    def __init__(
        self,
        *,
        protocol: str | None = None,
        base=None,
        quote=None,
        kind=None,
        other_data=None,
        exchange_account=False,
        vault=False,
    ):
        self._protocol = protocol
        self.base = base
        self.quote = quote
        self.kind = kind
        self.other_data = other_data or {}
        self._exchange_account = exchange_account
        self._vault = vault

    def get_exchange_account_protocol(self):
        return self._protocol

    def is_exchange_account(self):
        return self._exchange_account

    def is_vault(self):
        return self._vault


class DummyUniverse:
    """Minimal strategy universe for Lagoon translation unit tests."""

    def __init__(self, reserve_assets, pairs, exchanges=None):
        self.reserve_assets = reserve_assets
        self._pairs = pairs
        self.data_universe = SimpleNamespace(
            exchange_universe=SimpleNamespace(exchanges=exchanges or {}),
        )

    def iterate_pairs(self):
        return iter(self._pairs)


def test_sync_reserve_balance_to_state_updates_existing_reserves(tmp_path: Path):
    """Shared Lagoon reserve sync helper should update persisted reserve balances."""
    state_path, store = resolve_state_store("refactor-test", tmp_path / "state.json")
    state = State()
    reserve_asset = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    store.sync(state)

    updated_state, reserve_position = sync_reserve_balance_to_state(
        store,
        denomination_token=SimpleNamespace(symbol="USDC"),
        safe_balance=Decimal("123.45"),
    )

    assert state_path.name == "state.json"
    assert reserve_position.quantity == Decimal("123.45")
    assert updated_state.portfolio.get_default_reserve_position().quantity == Decimal("123.45")
    assert reserve_position.last_sync_at is not None


def test_multisig_normalisation_defaults_to_hot_wallet_address():
    """Shared multisig normalisation should cover the omitted-owner single-chain path."""
    hot_wallet = SimpleNamespace(address="0xabc")

    owners = _normalize_multisig_owners(None, hot_wallet)

    assert owners == ["0xabc"]
    assert _calculate_safe_threshold(owners) == 1
    assert _calculate_safe_threshold(["a", "b", "c"]) == 2


def test_translate_trading_universe_handles_hypercore_native_assets_when_any_asset_is_false():
    """Hypercore native assets should be normalised to HyperEVM without KeyError."""
    reserve_asset = SimpleNamespace(chain_id=42161)
    hypercore_asset = SimpleNamespace(
        chain_id=9999,
        address="0x0000000000000000000000000000000000000999",
    )
    usdc_asset = SimpleNamespace(
        chain_id=42161,
        address="0x0000000000000000000000000000000000000002",
    )
    pair = DummyPair(
        base=hypercore_asset,
        quote=usdc_asset,
        other_data={
            "vault_protocol": "hypercore",
            "hypercore_vault_address": "0x0000000000000000000000000000000000000abc",
        },
        vault=True,
    )
    universe = DummyUniverse([reserve_asset], [pair])

    configs = translate_trading_universe_to_lagoon_config(
        universe=universe,
        chain_web3={"arbitrum": object(), "hyperliquid": object()},
        asset_manager="0x0000000000000000000000000000000000000003",
        safe_owners=["0x0000000000000000000000000000000000000003"],
        safe_threshold=1,
        safe_salt_nonce=42,
        any_asset=False,
    )

    assert normalise_deployment_chain_id(9999) == HYPEREVM_CHAIN_ID
    assert "hyperliquid" in configs
    assert "0x0000000000000000000000000000000000000999" in configs["hyperliquid"].assets
    assert configs["hyperliquid"].hypercore_vaults == ["0x0000000000000000000000000000000000000abc"]


def test_exchange_account_value_dispatch_uses_protocol_registry(monkeypatch):
    """Unified exchange-account dispatch should route to protocol-specific valuators."""
    monkeypatch.setattr(
        exchange_account_utils,
        "_create_derive_protocol_value_func",
        lambda *args, **kwargs: (lambda pair: f"derive:{pair.get_exchange_account_protocol()}"),
    )
    monkeypatch.setattr(
        exchange_account_utils,
        "_create_ccxt_protocol_value_func",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        exchange_account_utils,
        "_create_gmx_protocol_value_func",
        lambda *args, **kwargs: (lambda pair: f"gmx:{pair.get_exchange_account_protocol()}"),
    )

    derive_pair = DummyPair(protocol="derive", exchange_account=True)
    gmx_pair = DummyPair(protocol="gmx", exchange_account=True)
    positions = [SimpleNamespace(pair=derive_pair), SimpleNamespace(pair=gmx_pair)]

    value_func = exchange_account_utils.create_exchange_account_value_func(
        positions=positions,
        derive_owner_private_key=None,
        derive_session_private_key=None,
        derive_wallet_address=None,
        derive_network=exchange_account_utils.DeriveNetwork.mainnet,
        ccxt_exchange_id=None,
        ccxt_options=None,
        ccxt_sandbox=False,
        logger=exchange_account_utils.logger,
        execution_model=object(),
    )

    assert value_func is not None
    assert value_func(derive_pair) == "derive:derive"
    assert value_func(gmx_pair) == "gmx:gmx"
