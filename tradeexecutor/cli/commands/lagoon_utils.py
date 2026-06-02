"""Shared helpers for Lagoon-oriented CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDiskCache
from web3 import Web3

from tradeexecutor.cli.bootstrap import configure_default_chain, create_state_store, create_web3_config
from tradeexecutor.ethereum.token import translate_token_details
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation
from tradeexecutor.utils.key import ensure_0x_prefixed_private_key


@dataclass(slots=True)
class LagoonCommandContext:
    """Shared execution context for simple single-chain Lagoon commands."""

    web3config: Any
    web3: Web3
    hot_wallet: HotWallet
    vault: LagoonVault


def create_single_chain_web3_config(*, mod: StrategyModuleInformation, simulate: bool = False, **rpc_kwargs):
    """Build a single-chain Web3 config from shared RPC kwargs.

    Uses the strategy module's ``get_default_chain_id()`` to select
    the correct chain when multiple JSON-RPC connections are configured.
    """
    web3config = create_web3_config(
        **rpc_kwargs,
        simulate=simulate,
    )

    if not web3config.has_any_connection():
        raise RuntimeError("This command requires that you pass a JSON-RPC connection to one of the networks")

    configure_default_chain(web3config, mod)
    return web3config


def create_hot_wallet(web3: Web3, private_key: str) -> HotWallet:
    """Create and nonce-sync the hot wallet used by Lagoon commands."""
    hot_wallet = HotWallet.from_private_key(ensure_0x_prefixed_private_key(private_key))
    hot_wallet.sync_nonce(web3)
    return hot_wallet


def load_lagoon_vault(
    web3: Web3,
    vault_address: str,
    token_cache: TokenDiskCache | None = None,
) -> LagoonVault:
    """Load a Lagoon vault instance from its address."""
    vault = create_vault_instance(
        web3,
        vault_address,
        features={ERC4626Feature.lagoon_like},
        default_block_identifier="latest",
        require_denomination_token=True,
        token_cache=token_cache,
    )
    assert isinstance(vault, LagoonVault), f"Not a Lagoon vault: {vault}"
    return vault


def create_lagoon_command_context(
    *,
    mod: StrategyModuleInformation,
    private_key: str,
    vault_address: str,
    token_cache: TokenDiskCache | None = None,
    simulate: bool = False,
    **rpc_kwargs,
) -> LagoonCommandContext:
    """Create the common single-chain context for simple Lagoon commands."""
    web3config = create_single_chain_web3_config(
        mod=mod,
        simulate=simulate,
        **rpc_kwargs,
    )
    web3 = web3config.get_default()
    hot_wallet = create_hot_wallet(web3, private_key)
    vault = load_lagoon_vault(web3, vault_address, token_cache=token_cache)
    return LagoonCommandContext(
        web3config=web3config,
        web3=web3,
        hot_wallet=hot_wallet,
        vault=vault,
    )


def resolve_state_store(id: str, state_file: Path | str | None, *, simulate: bool = False):
    """Resolve the state path and create the corresponding store."""
    resolved = Path(state_file) if state_file else Path(f"state/{id}.json")
    return resolved, create_state_store(resolved, simulate=simulate)


def ensure_state_store_exists(id: str, state_file: Path | str | None, *, simulate: bool = False):
    """Resolve a state store and assert it already exists."""
    resolved, store = resolve_state_store(id, state_file, simulate=simulate)
    assert not store.is_pristine(), (
        f"State file does not exist: {resolved}. "
        f"Run 'init' first to create the state file before continuing."
    )
    return resolved, store


def sync_reserve_balance_to_state(store, denomination_token, safe_balance):
    """Update the default reserve position to match the on-chain Safe balance."""
    state = store.load()
    ts = native_datetime_utc_now()
    if len(state.portfolio.reserves) == 0:
        reserve_asset = translate_token_details(denomination_token)
        state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)

    reserve_position = state.portfolio.get_default_reserve_position()
    reserve_position.quantity = safe_balance
    reserve_position.reserve_token_price = 1.0
    reserve_position.last_pricing_at = ts
    reserve_position.last_sync_at = ts
    store.sync(state)
    return state, reserve_position
