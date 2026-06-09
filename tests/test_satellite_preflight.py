"""Fast unit tests for satellite-module discovery and the cross-chain preflight.

These cover the deploy->runtime hand-off that stranded bridged USDC in
production, without needing forks or RPC:

- ``resolve_satellite_modules()`` resolves satellites from the env var or the
  deployment artifact, with the env var taking precedence.
- ``check_universe_contracts_resolve()`` only requires satellite modules for
  vault/Safe custody (Lagoon); hot-wallet custody (same EOA on every chain) must
  not be flagged for cross-chain trades.
"""

import json
from pathlib import Path

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import (
    resolve_satellite_modules,
    check_universe_contracts_resolve,
)


def test_resolve_satellite_modules_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Satellites resolve from the env var or the deployment artifact, env first.

    1. No source -> empty mapping
    2. Deployment artifact -> only the is_satellite entries, keyed by chain slug
    3. Missing artifact file -> empty mapping
    4. Single-chain artifact (no multichain flag) -> empty mapping
    5. SATELLITE_MODULES env var overrides the artifact
    """

    # 1. No source -> empty mapping
    monkeypatch.delenv("SATELLITE_MODULES", raising=False)
    assert resolve_satellite_modules(None) == {}

    # 2. Deployment artifact -> only the is_satellite entries
    artifact = tmp_path / "strategy.deployment.json"
    artifact.write_text(json.dumps({
        "multichain": True,
        "deployments": {
            "arbitrum": {"is_satellite": False, "module_address": "0xAAA"},  # primary, not a satellite
            "base": {"is_satellite": True, "module_address": "0xBBB"},
            "hyperliquid": {"is_satellite": True, "module_address": "0xCCC"},
        },
    }))
    assert resolve_satellite_modules(artifact) == {"base": "0xBBB", "hyperliquid": "0xCCC"}

    # 3. Missing artifact file -> empty mapping
    assert resolve_satellite_modules(tmp_path / "does-not-exist.json") == {}

    # 4. Single-chain artifact (no multichain flag) -> empty mapping
    single_chain = tmp_path / "single.json"
    single_chain.write_text(json.dumps({"vault_address": "0x1"}))
    assert resolve_satellite_modules(single_chain) == {}

    # 5. SATELLITE_MODULES env var overrides the artifact
    monkeypatch.setenv("SATELLITE_MODULES", json.dumps({"base": "0xENV"}))
    assert resolve_satellite_modules(artifact) == {"base": "0xENV"}


class _FakePair:
    """Minimal trading pair stub for the preflight check."""

    def __init__(self, chain_id: int, *, vault: bool = False, bridge: bool = False):
        self.chain_id = chain_id
        self._vault = vault
        self._bridge = bridge

    def is_cctp_bridge(self) -> bool:
        return self._bridge

    def is_vault(self) -> bool:
        return self._vault


class _FakeUniverse:
    """Minimal universe stub exposing only what the preflight check reads."""

    def __init__(self, pairs: list, primary_chain: ChainId):
        self._pairs = pairs
        self._primary_chain = primary_chain

    def iterate_pairs(self):
        return self._pairs

    def get_primary_chain(self) -> ChainId:
        return self._primary_chain


class _FakeTxBuilder:
    def __init__(self, chain_id: int, vault=None):
        self.chain_id = chain_id
        if vault is not None:
            self.vault = vault


class _FakeExecutionModel:
    def __init__(self, tx_builder: _FakeTxBuilder, satellite_vaults: dict):
        self.tx_builder = tx_builder
        self.satellite_vaults = satellite_vaults


class _FakeWeb3Config:
    def __init__(self):
        # Real (non-test) chains so the check does not early-return for Anvil.
        self.connections = {ChainId.arbitrum: object(), ChainId.base: object()}


def test_check_universe_contracts_resolve_custody_gating():
    """The satellite requirement applies to vault custody only, not hot wallets.

    Replicates a cross-chain universe (Arbitrum primary + Base satellite, both
    non-vault pairs so no on-chain code lookups happen) and asserts:

    1. Hot-wallet custody (tx_builder without a vault) does NOT raise — the same
       EOA works on every chain, so no satellite module is required.
    2. Vault/Safe custody (tx_builder with a vault) with no satellite configured
       for the Base chain MUST raise the fail-fast guard.
    """

    web3config = _FakeWeb3Config()
    universe = _FakeUniverse(
        pairs=[_FakePair(ChainId.arbitrum.value), _FakePair(ChainId.base.value)],
        primary_chain=ChainId.arbitrum,
    )

    # 1. Hot-wallet custody -> no satellite required -> must not raise
    hot_wallet_model = _FakeExecutionModel(
        tx_builder=_FakeTxBuilder(ChainId.arbitrum.value),  # no vault attribute
        satellite_vaults={},
    )
    check_universe_contracts_resolve(web3config, universe, hot_wallet_model)

    # 2. Vault/Safe custody with no satellite for Base -> must raise
    lagoon_model = _FakeExecutionModel(
        tx_builder=_FakeTxBuilder(ChainId.arbitrum.value, vault=object()),
        satellite_vaults={},
    )
    with pytest.raises(RuntimeError, match="satellite"):
        check_universe_contracts_resolve(web3config, universe, lagoon_model)
