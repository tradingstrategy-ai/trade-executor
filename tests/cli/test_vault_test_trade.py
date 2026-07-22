"""Unit tests for the standalone vault-test-trade command helpers."""

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from eth_defi.vault.base import VaultSpec
from requests.exceptions import ReadTimeout
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.commands.lagoon_deploy_vault import (
    _write_state_sibling_deployment_artifact,
)
from tradeexecutor.cli import vault_test_trade_tui as vault_test_trade_tui_module
from tradeexecutor.cli import (
    vault_test_trade_simulation as vault_test_trade_simulation_module,
)
from tradeexecutor.cli.vault_test_trade import (
    create_vault_test_diagnostic_pair,
    filter_rpc_kwargs_for_vault_specs,
    get_vault_trade_position,
    load_lagoon_deployment,
    parse_vault_ids,
)
from tradeexecutor.cli.vault_test_trade_tui import (
    VaultChoice,
    VaultSearchScreen,
    VaultTestTradeApp,
)
from tradeexecutor.cli.vault_test_trade_simulation import (
    SimulatedVaultAttemptTimeout,
    SimulatedVaultRuntime,
    is_simulated_infrastructure_failure,
    queue_simulated_infrastructure_retry,
    raise_simulated_vault_attempt_timeout,
    take_simulated_snapshots,
)
from tradeexecutor.cli.vault_test_trade_runner import (
    record_attempt_result,
    should_leave_deposit_open,
)
from tradeexecutor.ethereum import web3config as web3config_module
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State


class VaultSearchHarness(App):
    """Minimal app used to exercise the real vault typeahead widget."""

    def __init__(self, screen: VaultSearchScreen):
        super().__init__()
        self.vault_search = screen

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self.vault_search)


def test_parse_vault_ids_keeps_order_and_rejects_duplicates():
    """Explicit vault ids preserve operator order and fail safely on duplicates.

    1. Parse two distinct chain-address ids in the supplied order.
    2. Verify their canonical ids retain that order.
    3. Submit the same id twice and verify parsing refuses the ambiguous batch.
    """
    first = "1-0x0000000000000000000000000000000000000001"
    second = "42161-0x0000000000000000000000000000000000000002"

    # 1. Parse two distinct chain-address ids in the supplied order.
    parsed = parse_vault_ids(f"{first}, {second}")

    # 2. Verify their canonical ids retain that order.
    assert [item.as_string_id() for item in parsed] == [first, second]

    # 3. Submit the same id twice and verify parsing refuses the ambiguous batch.
    with pytest.raises(ValueError, match="duplicate vault id"):
        parse_vault_ids(f"{first},{first}")


def test_simulated_vault_rpc_filter_keeps_only_selected_chains() -> None:
    """Simulated deployment forks only the explicitly selected vault chains.

    1. Configure Ethereum, Base, Arbitrum and Hyperliquid RPC values.
    2. Select vaults on Ethereum, Arbitrum and Hyperliquid only.
    3. Verify Base is removed and the Hyperliquid slug override is retained.
    """
    # 1. Configure Ethereum, Base, Arbitrum and Hyperliquid RPC values.
    rpc_kwargs = {
        "json_rpc_ethereum": "ethereum-rpc",
        "json_rpc_base": "base-rpc",
        "json_rpc_arbitrum": "arbitrum-rpc",
        "json_rpc_hyperliquid": "hyperliquid-rpc",
    }

    # 2. Select vaults on Ethereum, Arbitrum and Hyperliquid only.
    specs = [
        VaultSpec(ChainId.ethereum.value, "0x0000000000000000000000000000000000000001"),
        VaultSpec(ChainId.arbitrum.value, "0x0000000000000000000000000000000000000002"),
        VaultSpec(
            ChainId.hyperliquid.value, "0x0000000000000000000000000000000000000003"
        ),
    ]
    filtered = filter_rpc_kwargs_for_vault_specs(rpc_kwargs, specs)

    # 3. Verify Base is removed and the Hyperliquid slug override is retained.
    assert filtered == {
        "json_rpc_ethereum": "ethereum-rpc",
        "json_rpc_base": None,
        "json_rpc_arbitrum": "arbitrum-rpc",
        "json_rpc_hyperliquid": "hyperliquid-rpc",
    }


def test_load_lagoon_deployment_reads_source_and_satellite_modules(tmp_path: Path):
    """A state-sibling Lagoon artefact supplies all executor topology.

    1. Write a source and satellite deployment artefact.
    2. Load it through the standalone command helper.
    3. Verify source chain, vault and satellite module addresses are retained.
    """
    # 1. Write a source and satellite deployment artefact.
    deployment_file = tmp_path / "vault-test.deployment.json"
    deployment_file.write_text(
        json.dumps(
            {
                "deployments": {
                    "base": {
                        "vault_address": "0x0000000000000000000000000000000000000001",
                        "module_address": "0x0000000000000000000000000000000000000002",
                        "is_satellite": False,
                    },
                    "arbitrum": {
                        "module_address": "0x0000000000000000000000000000000000000003",
                        "is_satellite": True,
                    },
                },
            }
        )
    )

    # 2. Load it through the standalone command helper.
    deployment = load_lagoon_deployment(deployment_file)

    # 3. Verify source chain, vault and satellite module addresses are retained.
    assert deployment.primary_chain_id.value == 8453
    assert deployment.vault_address == "0x0000000000000000000000000000000000000001"
    assert (
        deployment.satellite_modules[42161]
        == "0x0000000000000000000000000000000000000003"
    )


def test_single_chain_deployment_writes_a_runtime_deployment_artifact(tmp_path: Path):
    """A standalone Lagoon deployment emits the runtime artefact required by vault tests.

    1. Write a single-chain deployment record beside an explicit executor state file.
    2. Load the generated sibling artefact through the vault-test deployment reader.
    3. Verify it contains the source vault and module from the original record.
    4. Verify a later simulated deployment cannot overwrite the live artefact.
    """
    state_file = tmp_path / "state" / "vault-tester.json"
    record = {
        "Vault": "0x0000000000000000000000000000000000000001",
        "Trading strategy module": "0x0000000000000000000000000000000000000002",
    }

    # 1. Write a single-chain deployment record beside an explicit executor state file.
    _write_state_sibling_deployment_artifact(
        None,
        record,
        simulate=False,
        logger=logging.getLogger(__name__),
        executor_id="vault-tester",
        state_file=state_file,
        primary_chain_id=ChainId.base,
    )

    # 2. Load the generated sibling artefact through the vault-test deployment reader.
    deployment = load_lagoon_deployment(
        tmp_path / "state" / "vault-tester.deployment.json"
    )

    # 3. Verify it contains the source vault and module from the original record.
    assert deployment.primary_chain_id == ChainId.base
    assert deployment.module_address == record["Trading strategy module"]

    # 4. Verify a later simulated deployment cannot overwrite the live artefact.
    simulated_record = dict(record)
    simulated_record["Vault"] = "0x0000000000000000000000000000000000000099"
    _write_state_sibling_deployment_artifact(
        None,
        simulated_record,
        simulate=True,
        logger=logging.getLogger(__name__),
        executor_id="vault-tester",
        state_file=state_file,
        primary_chain_id=ChainId.base,
    )
    unchanged = load_lagoon_deployment(
        tmp_path / "state" / "vault-tester.deployment.json"
    )
    assert unchanged.vault_address == record["Vault"]


def test_deposit_round_trip_gating() -> None:
    """Only safe automatic instant deposits continue directly to redemption.

    1. Check an automatic instant deposit with redemption available.
    2. Check async, redemption-unavailable and manual deposits.
    3. Verify a redemption operation is never treated as a deposit-only action.
    """
    # 1. Check an automatic instant deposit with redemption available.
    assert (
        should_leave_deposit_open(
            operation="deposit", is_async=False, redemption_available=True, manual=False
        )
        is False
    )

    # 2. Check async, redemption-unavailable and manual deposits.
    assert (
        should_leave_deposit_open(
            operation="deposit", is_async=True, redemption_available=True, manual=False
        )
        is True
    )
    assert (
        should_leave_deposit_open(
            operation="deposit",
            is_async=False,
            redemption_available=False,
            manual=False,
        )
        is True
    )
    assert (
        should_leave_deposit_open(
            operation="deposit", is_async=False, redemption_available=True, manual=True
        )
        is True
    )

    # 3. Verify a redemption operation is never treated as a deposit-only action.
    assert (
        should_leave_deposit_open(
            operation="redeem", is_async=True, redemption_available=False, manual=True
        )
        is False
    )


def test_adapter_failure_can_be_recorded_as_a_normal_position() -> None:
    """An unsupported vault still produces a persistent diagnostic position.

    1. Create a state reserve and a placeholder pair for an unavailable adapter.
    2. Record the adapter failure through the normal attempt-state path.
    3. Verify the closed position is identifiable and JSON serialisable.
    """
    # 1. Create a state reserve and a placeholder pair for an unavailable adapter.
    state = State()
    reserve_asset = AssetIdentifier(
        ChainId.base.value,
        "0x0000000000000000000000000000000000000010",
        "USDC",
        6,
    )
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    spec = VaultSpec(
        ChainId.arbitrum.value,
        "0x0000000000000000000000000000000000000020",
    )
    pair = create_vault_test_diagnostic_pair(spec, reserve_asset)

    # 2. Record the adapter failure through the normal attempt-state path.
    position = record_attempt_result(
        state,
        pair,
        spec,
        simulated=False,
        result="failed",
        detail="adapter unavailable",
    )

    # 3. Verify the closed position is identifiable and JSON serialisable.
    assert position.is_closed()
    assert position.pair.pool_address == spec.vault_address
    assert position.other_data["vault_test_attempt"]["result"] == "failed"
    payload = state.to_json_safe()
    assert (
        json.loads(payload)["portfolio"]["closed_positions"][str(position.position_id)][
            "simulated"
        ]
        is False
    )
    restored_position = State.from_json(payload).portfolio.get_position_by_id(
        position.position_id
    )
    assert restored_position.simulated is False


def test_simulated_vault_attempt_timeout_is_recordable() -> None:
    """A stuck simulated adapter is interrupted with a normal catchable error.

    1. Invoke the signal handler used by the per-vault wall-clock guard.
    2. Catch its dedicated control-flow exception at the CLI boundary.
    3. Verify the diagnostic identifies the bounded simulated attempt.
    """
    # 1. Invoke the signal handler used by the per-vault wall-clock guard.
    with pytest.raises(SimulatedVaultAttemptTimeout) as exc_info:
        raise_simulated_vault_attempt_timeout(None, None)

    # 2. Catch its dedicated control-flow exception at the CLI boundary.
    assert isinstance(exc_info.value, SimulatedVaultAttemptTimeout)
    assert not isinstance(exc_info.value, Exception)

    # 3. Verify the diagnostic identifies the bounded simulated attempt.
    assert "Simulated vault attempt exceeded" in str(exc_info.value)


def test_simulated_infrastructure_failure_queues_only_one_clean_rerun() -> None:
    """Transport failures rerun once on a new Anvil while adapter errors remain terminal.

    1. Classify a local RPC timeout and an ordinary adapter failure.
    2. Queue the affected vault after its first infrastructure failure.
    3. Verify a second infrastructure failure is not queued indefinitely.
    """
    spec = VaultSpec(ChainId.base.value, "0x0000000000000000000000000000000000000001")
    pending_specs = deque()
    restart_counts = defaultdict(int)

    # 1. Classify a local RPC timeout and an ordinary adapter failure.
    assert (
        is_simulated_infrastructure_failure(ReadTimeout("localhost Anvil timed out"))
        is True
    )
    assert (
        is_simulated_infrastructure_failure(
            RuntimeError("execution reverted: deposit closed")
        )
        is False
    )

    # 2. Queue the affected vault after its first infrastructure failure.
    assert (
        queue_simulated_infrastructure_retry(spec, pending_specs, restart_counts)
        is True
    )
    assert list(pending_specs) == [spec]

    # 3. Verify a second infrastructure failure is not queued indefinitely.
    pending_specs.clear()
    assert (
        queue_simulated_infrastructure_retry(spec, pending_specs, restart_counts)
        is False
    )
    assert list(pending_specs) == []


def test_simulated_infrastructure_failure_ignores_implicit_exception_context() -> None:
    """An adapter failure raised during RPC handling must remain an adapter failure.

    1. Raise an RPC timeout and handle it locally.
    2. Raise an unrelated adapter error from that handler without explicit chaining.
    3. Verify the implicit context does not request a fresh Anvil generation.
    """
    # 1. Raise an RPC timeout and handle it locally.
    caught_error = None
    try:
        raise ReadTimeout("localhost Anvil timed out")
    except ReadTimeout:
        # 2. Raise an unrelated adapter error from that handler without explicit chaining.
        try:
            raise RuntimeError("execution reverted: unsupported adapter")
        except RuntimeError as adapter_error:
            assert isinstance(adapter_error.__context__, ReadTimeout)
            caught_error = adapter_error

    # 3. Verify the implicit context does not request a fresh Anvil generation.
    assert caught_error is not None
    assert is_simulated_infrastructure_failure(caught_error) is False


def test_simulated_snapshots_only_touch_attempt_chains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multichain attempt snapshots only its source and selected vault chains.

    1. Configure source, target and unrelated simulated chain connections.
    2. Take snapshots for an Arbitrum vault with Base as the source.
    3. Verify the unrelated Ethereum Anvil was not snapshotted.
    """
    web3config = MagicMock()
    connections = {
        ChainId.base: object(),
        ChainId.arbitrum: object(),
        ChainId.ethereum: object(),
    }
    web3config.get_connection.side_effect = connections.get
    deployment = MagicMock()
    deployment.primary_chain_id = ChainId.base
    spec = VaultSpec(
        ChainId.arbitrum.value, "0x0000000000000000000000000000000000000001"
    )
    snapshotted_connections = []

    # 1. Configure source, target and unrelated simulated chain connections.
    def snapshot(web3, method, args=None):
        snapshotted_connections.append(web3)
        assert method == "evm_snapshot"
        return hex(len(snapshotted_connections))

    monkeypatch.setattr(
        vault_test_trade_simulation_module, "make_anvil_custom_rpc_request", snapshot
    )

    # 2. Take snapshots for an Arbitrum vault with Base as the source.
    snapshots = take_simulated_snapshots(web3config, deployment, spec)

    # 3. Verify the unrelated Ethereum Anvil was not snapshotted.
    assert set(snapshots) == {ChainId.base, ChainId.arbitrum}
    assert set(snapshotted_connections) == {
        connections[ChainId.base],
        connections[ChainId.arbitrum],
    }
    assert connections[ChainId.ethereum] not in snapshotted_connections


def test_simulated_runtime_close_uses_bounded_hard_shutdown() -> None:
    """Discarding a simulation generation bounds process shutdown and removes its artefact.

    1. Construct a disposable runtime with mocked Web3 and temporary-directory owners.
    2. Close the runtime after an infrastructure failure.
    3. Verify all Anvils receive the short shutdown budget and the artefact is removed.
    """
    web3config = MagicMock()
    temporary_deployment_dir = MagicMock()
    runtime = SimulatedVaultRuntime(
        generation=3,
        web3config=web3config,
        deployment=MagicMock(),
        deployment_file=Path("/tmp/generation-3.deployment.json"),
        execution_model=MagicMock(),
        sync_model=MagicMock(),
        reserve_asset=MagicMock(),
        temporary_deployment_dir=temporary_deployment_dir,
    )

    # 1. Construct a disposable runtime with mocked Web3 and temporary-directory owners.
    assert runtime.generation == 3

    # 2. Close the runtime after an infrastructure failure.
    runtime.close()

    # 3. Verify all Anvils receive the short shutdown budget and the artefact is removed.
    web3config.close.assert_called_once_with(log_level=logging.ERROR, block_timeout=5)
    temporary_deployment_dir.cleanup.assert_called_once_with()


def test_simulated_web3_never_retries_failed_local_anvil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulated Web3 connection replaces a failed Anvil instead of retrying localhost.

    1. Replace Anvil launch and local Web3 construction with controlled test doubles.
    2. Create a simulated Base connection through the production factory.
    3. Verify localhost retries are disabled and upstream failover has a bounded budget.
    """
    anvil = MagicMock()
    anvil.json_rpc_url = "http://localhost:23456"
    web3 = MagicMock()
    web3.eth.chain_id = ChainId.base.value
    captured = {}

    # 1. Replace Anvil launch and local Web3 construction with controlled test doubles.
    def launch_anvil(configuration_line, **kwargs):
        captured["configuration_line"] = configuration_line
        captured["launch_kwargs"] = kwargs
        return anvil

    def create_multi_provider_web3(configuration_line, **kwargs):
        captured["local_url"] = configuration_line
        captured["web3_kwargs"] = kwargs
        return web3

    monkeypatch.setattr(web3config_module, "launch_anvil", launch_anvil)
    monkeypatch.setattr(
        web3config_module, "create_multi_provider_web3", create_multi_provider_web3
    )
    setup_custom_log_levels()

    # 2. Create a simulated Base connection through the production factory.
    result = Web3Config.create_web3(
        "https://base-a.example https://base-b.example",
        simulate=True,
        chain_id=ChainId.base,
    )

    # 3. Verify localhost retries are disabled and upstream failover has a bounded budget.
    assert result is web3
    assert captured["local_url"] == anvil.json_rpc_url
    assert captured["web3_kwargs"]["retries"] == 0
    assert captured["web3_kwargs"]["default_http_timeout"] == (3.0, 40.0)
    proxy_config = captured["launch_kwargs"]["proxy_multiple_upstream"]
    assert proxy_config.timeout == 10.0
    assert proxy_config.retries == 3


def test_real_position_lookup_does_not_relabel_simulated_history() -> None:
    """Real lifecycle updates select only real target-vault positions.

    1. Model an older real position and a newer closed simulated position for one vault.
    2. Query the normal latest trade position and the real-only position.
    3. Verify real execution can never select and relabel the simulated record.
    """
    # 1. Model an older real position and a newer closed simulated position for one vault.
    spec = VaultSpec(ChainId.base.value, "0x0000000000000000000000000000000000000001")
    real_position = MagicMock()
    real_position.position_id = 1
    real_position.pair.chain_id = spec.chain_id
    real_position.pair.pool_address = spec.vault_address
    real_position.trades = {1: MagicMock()}
    real_position.simulated = False
    simulated_position = MagicMock()
    simulated_position.position_id = 2
    simulated_position.pair.chain_id = spec.chain_id
    simulated_position.pair.pool_address = spec.vault_address
    simulated_position.trades = {2: MagicMock()}
    simulated_position.simulated = True
    state = MagicMock()
    state.portfolio.get_all_positions.return_value = [real_position, simulated_position]

    # 2. Query the normal latest trade position and the real-only position.
    latest = get_vault_trade_position(state, spec)
    latest_real = get_vault_trade_position(state, spec, simulated=False)

    # 3. Verify real execution can never select and relabel the simulated record.
    assert latest is simulated_position
    assert latest_real is real_position


@pytest.mark.anyio
async def test_vault_typeahead_filters_downloaded_vaults():
    """The manual new-deposit dialogue filters the complete downloaded vault list.

    1. Mount the real Textual vault search screen with two downloadable vaults.
    2. Type a partial vault name in its search input.
    3. Verify the table leaves only the matching vault available for selection.
    """
    alpha = VaultChoice(
        VaultSpec(1, "0x0000000000000000000000000000000000000001"),
        "Alpha vault",
        "ethereum",
        "Alpha",
    )
    beta = VaultChoice(
        VaultSpec(42161, "0x0000000000000000000000000000000000000002"),
        "Beta vault",
        "arbitrum",
        "Beta",
    )
    app = VaultSearchHarness(VaultSearchScreen([alpha, beta]))

    async with app.run_test() as pilot:
        # 1. Mount the real Textual vault search screen with two downloadable vaults.
        await pilot.pause()
        search_input = app.vault_search.query_one("#vault-search-input", Input)

        # 2. Type a partial vault name in its search input.
        search_input.value = "beta"
        await pilot.pause()

        # 3. Verify the table leaves only the matching vault available for selection.
        table = app.vault_search.query_one("#vault-search-table", DataTable)
        assert table.row_count == 1


@pytest.mark.anyio
async def test_vault_main_table_enter_selects_redemption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enter on a deposited vault emits a manual redemption action.

    1. Mount the real main table with one deposited vault.
    2. Press Enter while its DataTable row is focused.
    3. Verify the selected action requests redemption for that vault.
    """
    choice = VaultChoice(
        VaultSpec(8453, "0x0000000000000000000000000000000000000001"),
        "Deposited vault",
        "base",
        "Lagoon",
    )
    position = MagicMock()
    position.position_id = 1
    position.simulated = False
    position.is_open.return_value = True
    monkeypatch.setattr(
        vault_test_trade_tui_module,
        "get_latest_vault_position",
        lambda state, vault_spec: position,
    )
    monkeypatch.setattr(
        vault_test_trade_tui_module,
        "get_vault_trade_position",
        lambda state, vault_spec, open_only=False: position,
    )
    monkeypatch.setattr(
        vault_test_trade_tui_module,
        "get_vault_test_status",
        lambda position: "deposited",
    )
    app = VaultTestTradeApp(choices=[choice], state=MagicMock())

    async with app.run_test() as pilot:
        # 1. Mount the real main table with one deposited vault.
        await pilot.pause()

        # 2. Press Enter while its DataTable row is focused.
        await pilot.press("enter")
        await pilot.pause()

    # 3. Verify the selected action requests redemption for that vault.
    assert app.selected_action is not None
    assert app.selected_action.action == "redeem"
    assert app.selected_action.vault_spec == choice.vault_spec


def test_vault_main_table_retains_historical_vault_missing_from_download() -> None:
    """A previously tested vault remains visible after universe metadata changes.

    1. Build state containing a historical vault-test attempt.
    2. Construct the TUI with an empty freshly downloaded choice list.
    3. Verify the historical vault is still present in the main table model.
    """
    # 1. Build state containing a historical vault-test attempt.
    position = MagicMock()
    position.other_data = {
        "vault_test_attempt": {
            "vault_id": "8453-0x0000000000000000000000000000000000000001",
        },
    }
    position.pair.other_data = {"vault_protocol": "lagoon"}
    position.pair.exchange_name = "Historical vault"
    state = MagicMock()
    state.portfolio.get_all_positions.return_value = [position]

    # 2. Construct the TUI with an empty freshly downloaded choice list.
    app = VaultTestTradeApp(choices=[], state=state)

    # 3. Verify the historical vault is still present in the main table model.
    assert len(app.choices) == 1
    assert app.choices[0].name == "Historical vault"
    assert app.choices[0].protocol == "lagoon"
