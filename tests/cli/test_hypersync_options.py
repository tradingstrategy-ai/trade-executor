"""HyperSync configuration through registered commands and settlement recovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from typer.testing import CliRunner

import tradeexecutor.cli.commands.check_accounts as check_module
import tradeexecutor.cli.commands.lagoon_settle as settle_module
import tradeexecutor.ethereum.lagoon.vault as lagoon_module
import tradeexecutor.ethereum.vault.settlement_retry as retry_module
import tradeexecutor.cli.commands.lagoon_manual_settle as manual_module
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.state import State


@pytest.mark.parametrize("args, enabled, expected", [
    ([], "true", "environment-key"),
    (["--hypersync-api-key", "explicit-key"], "true", "explicit-key"),
    (["--no-use-hypersync"], "true", None),
    ([], "false", None),
])
def test_account_check_forwards_hypersync_to_reconciliation(mocker: MockerFixture, tmp_path: Path, args: list[str], enabled: str, expected: str | None) -> None:
    """Read-only reconciliation receives the CLI selection without signing keys.

    1. Mock chain/bootstrap reads because this test targets CLI credential wiring.
    2. Invoke the registered account check with environment and CLI overrides.
    3. Verify both sync setup and reconciliation receive the effective key.
    """
    # 1. Mock chain/bootstrap reads because this test targets CLI credential wiring.
    mocker.patch.object(check_module, "setup_logging")
    config = mocker.patch.object(check_module, "create_web3_config").return_value
    config.get_default.return_value = SimpleNamespace(eth=SimpleNamespace(chain_id=1, block_number=100))
    mocker.patch.object(check_module, "read_strategy_module")
    mocker.patch.object(check_module, "configure_default_chain")
    sync = mocker.patch.object(check_module, "create_sync_model")
    sync.return_value.get_key_address.return_value = None
    store = mocker.patch.object(check_module, "create_state_store").return_value
    store.is_pristine.return_value = False
    store.load.return_value = State()
    mocker.patch.object(check_module, "configure_account_check")
    mocker.patch.object(check_module, "load_lighter_signers", return_value={})
    reconcile = mocker.patch.object(check_module, "reconcile_verified_lighter_transfers", side_effect=RuntimeError("reached-reconciliation"))

    # 2. Invoke the registered account check with environment and CLI overrides.
    result = CliRunner().invoke(app, ["check-accounts", "--id", "hypersync-test", "--strategy-file", str(tmp_path / "strategy.py"), *args], env={
        "HYPERSYNC_API_KEY": "environment-key", "USE_HYPERSYNC": enabled,
        "PRIVATE_KEY": None, "JSON_RPC_ETHEREUM": None,
    })

    # 3. Verify both sync setup and reconciliation receive the effective key.
    assert isinstance(result.exception, RuntimeError), result.output
    assert str(result.exception) == "reached-reconciliation"
    assert sync.call_args.kwargs["hypersync_api_key"] == expected
    assert reconcile.call_args.kwargs["hypersync_api_key"] == expected
    assert reconcile.call_args.kwargs["signers"] == {}


def test_settle_cli_forwards_key_to_shared_execution_bootstrap(mocker: MockerFixture, tmp_path: Path) -> None:
    """The registered settlement command passes an explicit credential to bootstrap.

    1. Mock strategy/cache/provider setup to avoid external work in a wiring test.
    2. Invoke Lagoon settlement with an explicit key overriding the environment.
    3. Verify shared execution and sync construction receives that key.
    """
    # 1. Mock strategy/cache/provider setup to avoid external work in a wiring test.
    mocker.patch.object(settle_module, "setup_logging")
    mocker.patch.object(settle_module, "read_strategy_module")
    mocker.patch.object(settle_module, "prepare_cache_and_token_cache", return_value=(tmp_path, None))
    mocker.patch.object(settle_module, "create_web3_config")
    mocker.patch.object(settle_module, "configure_default_chain")
    bootstrap = mocker.patch.object(settle_module, "create_execution_and_sync_model", side_effect=RuntimeError("reached-bootstrap"))

    # 2. Invoke Lagoon settlement with an explicit key overriding the environment.
    result = CliRunner().invoke(app, ["lagoon-settle", "--id", "hypersync-test", "--strategy-file", str(tmp_path / "strategy.py"), "--hypersync-api-key", "explicit-key"], env={"HYPERSYNC_API_KEY": "environment-key", "USE_HYPERSYNC": "true"})

    # 3. Verify shared execution and sync construction receives that key.
    assert str(result.exception) == "reached-bootstrap", result.output
    assert bootstrap.call_args.kwargs["hypersync_api_key"] == "explicit-key"


def test_lagoon_indexer_failure_retries_rpc_and_preserves_read_errors(mocker: MockerFixture) -> None:
    """Incomplete indexer evidence is retried with RPC, whose errors must propagate.

    1. Construct the reader portion of a sync model and mock provider responses.
    2. Verify indexer failure produces exactly one full-range RPC retry.
    3. Verify RPC failure propagates and Anvil never selects the indexer.
    """
    # 1. Construct the reader portion of a sync model and mock provider responses.
    model = object.__new__(LagoonVaultSyncModel)
    model.vault = SimpleNamespace()
    model.anvil = False
    model.hypersync_api_key = "test-key"
    read = mocker.patch.object(lagoon_module, "fetch_lagoon_settlements", side_effect=[RuntimeError("indexer incomplete"), []])

    # 2. Verify indexer failure produces exactly one full-range RPC retry.
    assert model._fetch_lagoon_settlements(10, 20) == []
    assert [c.kwargs["use_hypersync"] for c in read.call_args_list] == [True, False]
    assert read.call_args_list[0].kwargs["hypersync_api_key"] == "test-key"
    assert all(c.kwargs["start_block"] == 10 and c.kwargs["end_block"] == 20 for c in read.call_args_list)

    # 3. Verify RPC failure propagates and Anvil never selects the indexer.
    read.side_effect = [RuntimeError("indexer incomplete"), RuntimeError("RPC failed")]
    with pytest.raises(RuntimeError, match="RPC failed"):
        model._fetch_lagoon_settlements(10, 20)
    model.anvil = True
    read.reset_mock(side_effect=True)
    read.return_value = []
    assert model._fetch_lagoon_settlements(10, 20) == []
    assert read.call_args.kwargs["use_hypersync"] is False
    read.assert_called_once()


def test_unconfirmed_probe_and_manual_nav_use_rpc_when_required(mocker: MockerFixture) -> None:
    """Recent probes bypass the indexer and manual NAV retries its failures.

    1. Mock chain reads to isolate recent-settlement selection.
    2. Verify a configured key cannot select HyperSync for the unconfirmed probe.
    3. Verify manual NAV inspection retries an indexer failure with RPC.
    """
    # 1. Mock chain reads to isolate recent-settlement selection.
    model = object.__new__(LagoonVaultSyncModel)
    model.vault = SimpleNamespace(web3=SimpleNamespace(eth=SimpleNamespace(block_number=22)))
    model.anvil = False
    model.hypersync_api_key = "test-key"
    read = mocker.patch.object(lagoon_module, "fetch_lagoon_settlements", return_value=[])
    # 2. Verify a configured key cannot select HyperSync for the unconfirmed probe.
    model._probe_unconfirmed_lagoon_settlements(safe_sync_block=20, recorded_hashes=set())
    assert read.call_args.kwargs["use_hypersync"] is False
    assert read.call_args.kwargs["start_block"] == 21
    # 3. Verify manual NAV inspection retries an indexer failure with RPC.
    vault = SimpleNamespace(web3=object(), address="0x" + "11" * 20, vault_contract=SimpleNamespace(abi=[]))
    mocker.patch.object(manual_module, "is_anvil", return_value=False)
    nav_read = mocker.patch.object(manual_module, "fetch_vault_settlement_logs", side_effect=[RuntimeError("indexer failed"), []])
    assert manual_module._fetch_pending_new_total_assets(vault, 10, 20, hypersync_api_key="test-key")[0] is None
    assert [c.kwargs["use_hypersync"] for c in nav_read.call_args_list] == [True, False]


def test_pool_recovery_forwards_execution_credential(mocker: MockerFixture) -> None:
    """Recovery's separately constructed router receives the read credential.

    1. Mock the router and builder so recovery cannot broadcast or query accounts.
    2. Resume a pool trade with an execution model holding a HyperSync key.
    3. Verify construction forwards the key and retains the unresolved trade.
    """
    # 1. Mock the router and builder so recovery cannot broadcast or query accounts.
    factory = mocker.patch.object(retry_module, "LighterPoolRouting")
    factory.return_value.resume_pending_trade.return_value = False
    mocker.patch.object(retry_module, "_get_chain_aware_tx_builder")
    execution = SimpleNamespace(web3=object(), hypersync_api_key="test-key")
    # 2. Resume a pool trade with an execution model holding a HyperSync key.
    resolved = []
    retry_module._resolve_lighter_pool_trade(State(), object(), execution, None, resolved)
    # 3. Verify construction forwards the key and retains the unresolved trade.
    assert factory.call_args.kwargs["hypersync_api_key"] == "test-key"
    assert resolved == []
