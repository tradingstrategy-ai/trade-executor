"""Fast unit tests for satellite-module discovery and the cross-chain preflight.

These cover the deploy->runtime hand-off that stranded bridged USDC in
production, without needing forks or RPC:

- ``resolve_satellite_modules()`` resolves satellites from the env var or the
  deployment artifact, with the env var taking precedence.
- ``check_universe_contracts_resolve()`` only requires satellite modules for
  vault/Safe custody (Lagoon); hot-wallet custody (same EOA on every chain) must
  not be flagged for cross-chain trades.
"""

import datetime
import json
from pathlib import Path

import pytest
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import (
    resolve_satellite_modules,
    resolve_deployment_file,
    update_strategy_file_deployment_info,
    check_universe_contracts_resolve,
)
from tradeexecutor.state.state import State


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


def test_resolve_deployment_file():
    """resolve_deployment_file points at the ``{id}.deployment.json`` sibling of the state file.

    Every CLI command that builds an execution model must pass this path into
    ``create_execution_and_sync_model`` so satellite vaults are populated; a command
    that omits it (the production ``trade-ui`` bug that left cross-chain Lagoon trades
    unrouteable with "No satellite vault configured for chain ...") regresses silently.
    This locks the path derivation now shared by all commands.

    1. id only -> default ``state/{id}.deployment.json``
    2. explicit None state_file -> same default
    3. default state_file path -> sibling deployment artifact
    4. custom state_file location -> deployment artifact next to it
    """

    # 1. id only -> default state/{id}.deployment.json
    assert resolve_deployment_file("master-vault-v2") == Path("state/master-vault-v2.deployment.json")

    # 2. explicit None state_file -> same default
    assert resolve_deployment_file("master-vault-v2", None) == Path("state/master-vault-v2.deployment.json")

    # 3. default state_file path -> sibling deployment artifact
    assert resolve_deployment_file("master-vault-v2", "state/master-vault-v2.json") == Path("state/master-vault-v2.deployment.json")

    # 4. custom state_file location -> deployment artifact next to it
    assert resolve_deployment_file("foo", "/custom/dir/foo.json") == Path("/custom/dir/foo.deployment.json")


def test_update_strategy_file_deployment_info_detects_changes(tmp_path: Path):
    """Strategy-file deployment info is persisted and changed artifacts append audit entries.

    1. Ignore missing and legacy single-chain deployment artifacts.
    2. Store initial multichain deployment data in ``state.deployment_info``.
    3. Re-read unchanged deployment data without adding another modified entry.
    4. Change deployment data and verify a new modified entry is appended.
    5. Round-trip through state JSON and verify deployment info survives serialisation.
    """

    state = State()
    missing_artifact = tmp_path / "missing.deployment.json"

    # 1. Ignore missing and legacy single-chain deployment artifacts.
    assert update_strategy_file_deployment_info(state, missing_artifact) is False
    assert state.deployment_info is None

    legacy_artifact = tmp_path / "legacy.deployment.json"
    legacy_artifact.write_text(json.dumps({"vault_address": "0xVault"}))
    assert update_strategy_file_deployment_info(state, legacy_artifact) is False
    assert state.deployment_info is None

    # 2. Store initial multichain deployment data in state.deployment_info.
    artifact = tmp_path / "strategy.deployment.json"
    deployment_payload = {
        "multichain": True,
        "deployment_mode": "full deployment",
        "safe_salt_nonce": 42,
        "guard_report": "large generated report should not be stored in state",
        "deployments": {
            "arbitrum": {
                "vault_address": "0xVault",
                "safe_address": "0xSafe",
                "module_address": "0xModule",
                "asset_manager": "0xAssetManager",
                "valuation_manager": "0xValuationManager",
                "is_satellite": False,
                "deployment_data": {
                    "Safe": "0xSafe",
                    "Vault": "0xVault",
                    "Trading strategy module": "0xModule",
                },
                "config": {
                    "uniswap_v3": {
                        "router": "0xRouter",
                        "position_manager": "0xPositionManager",
                    },
                },
                "whitelisted_items": [
                    {
                        "kind": "Sender",
                        "name": "trade-executor",
                        "address": "0xAssetManager",
                    },
                ],
            },
        },
    }
    artifact.write_text(json.dumps(deployment_payload))
    assert update_strategy_file_deployment_info(state, artifact) is True
    assert state.deployment_info is not None
    assert state.deployment_info.source == "strategy_file"
    assert state.deployment_info.deployment_file == str(artifact)
    assert state.deployment_info.data == {
        "multichain": True,
        "deployment_mode": "full deployment",
        "safe_salt_nonce": 42,
        "deployments": deployment_payload["deployments"],
    }
    assert "guard_report" not in state.deployment_info.data
    assert len(state.deployment_info.modified) == 1
    assert isinstance(state.deployment_info.modified[0].modified_at, datetime.datetime)
    assert state.deployment_info.modified[0].change_summary_message == f"Initialised strategy-file deployment information from {artifact}"

    # 3. Re-read unchanged deployment data without adding another modified entry.
    assert update_strategy_file_deployment_info(state, artifact) is False
    assert len(state.deployment_info.modified) == 1

    # 4. Change deployment data and verify a new modified entry is appended.
    deployment_payload["deployments"]["base"] = {
        "vault_address": None,
        "safe_address": "0xSafe",
        "module_address": "0xBaseModule",
        "asset_manager": "0xAssetManager",
        "valuation_manager": "0xValuationManager",
        "is_satellite": True,
    }
    artifact.write_text(json.dumps(deployment_payload))
    assert update_strategy_file_deployment_info(state, artifact) is True
    assert state.deployment_info.data["deployments"]["base"]["module_address"] == "0xBaseModule"
    assert len(state.deployment_info.modified) == 2
    assert isinstance(state.deployment_info.modified[1].modified_at, datetime.datetime)
    assert state.deployment_info.modified[1].change_summary_message == f"Updated strategy-file deployment information from {artifact}"

    # 5. Round-trip through state JSON and verify deployment info survives serialisation.
    restored_state = State.from_json(state.to_json())
    assert restored_state.deployment_info is not None
    assert restored_state.deployment_info.source == "strategy_file"
    assert restored_state.deployment_info.data["deployments"]["base"]["module_address"] == "0xBaseModule"
    assert len(restored_state.deployment_info.modified) == 2


class _FakePair:
    """Minimal trading pair stub for the preflight check."""

    def __init__(self, chain_id: int, *, vault: bool = False, bridge: bool = False, hypercore: bool = False):
        self.chain_id = chain_id
        self._vault = vault or hypercore
        self._bridge = bridge
        self._hypercore = hypercore

    def is_cctp_bridge(self) -> bool:
        return self._bridge

    def is_vault(self) -> bool:
        return self._vault

    def is_hyperliquid_vault(self) -> bool:
        return self._hypercore


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


def test_check_universe_contracts_resolve_skips_hypercore_vaults():
    """HyperCore vault pairs are skipped — no satellite or EVM code requirement.

    HyperCore (Hyperliquid) vault pairs are kind=vault but use the synthetic chain
    id 9999 and are not EVM contracts. The preflight must not require a satellite
    for chain 9999 nor try to read on-chain code for them, so a Lagoon strategy
    that trades a HyperCore vault must not be blocked.

    1. Build a Lagoon-custody universe with the primary chain plus a HyperCore
       vault pair (synthetic chain 9999, no EVM code)
    2. Assert the preflight does not raise even though no satellite is configured
    """

    # 1. Lagoon custody, HyperCore vault pair on synthetic chain 9999, no satellites
    web3config = _FakeWeb3Config()
    universe = _FakeUniverse(
        pairs=[
            _FakePair(ChainId.arbitrum.value),
            _FakePair(ChainId.hypercore.value, hypercore=True),
        ],
        primary_chain=ChainId.arbitrum,
    )
    lagoon_model = _FakeExecutionModel(
        tx_builder=_FakeTxBuilder(ChainId.arbitrum.value, vault=object()),
        satellite_vaults={},
    )

    # 2. HyperCore vault is skipped -> no satellite required, no get_code() -> no raise
    check_universe_contracts_resolve(web3config, universe, lagoon_model)
