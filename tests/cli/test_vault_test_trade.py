"""Unit tests for the standalone vault-test-trade command helpers."""

import datetime
import json
import logging
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
from eth_defi.middleware import ProbablyNodeHasNoBlock
from eth_defi.testing.fork_blocks import MIDNIGHT_BLOCKS
from eth_defi.vault.base import VaultSpec
from eth_defi.vault.deposit_redeem import (
    UnsupportedVaultSimulation,
    VaultDepositManagerCapability,
    VaultFlowUnavailable,
    WhitelistingRequired,
)
from hexbytes import HexBytes
from requests.exceptions import ReadTimeout
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultDepositStatus
from web3.exceptions import BadFunctionCallOutput

from tradeexecutor.cli.commands.lagoon_deploy_vault import (
    _write_state_sibling_deployment_artifact,
)
from tradeexecutor.cli.commands.vault_test_trade import _validate_vault_test_options
from tradeexecutor.cli.vault_trade import tui as tui_module
from tradeexecutor.cli.vault_trade import (
    runner as runner_module,
    simulation as simulation_module,
)
from tradeexecutor.cli.vault_trade.core import (
    filter_rpc_kwargs_for_vault_specs,
    load_lagoon_deployment,
    parse_vault_ids,
)
from tradeexecutor.cli.vault_trade.state import (
    capture_vault_test_error,
    classify_vault_test_failure,
    create_vault_test_diagnostic_pair,
    export_vault_test_report,
    get_vault_test_status,
    get_vault_trade_position,
    record_attempt_result,
    stamp_position_vault_test_attempt,
)
from tradeexecutor.cli.vault_trade.tui import (
    VaultChoice,
    VaultSearchScreen,
    VaultTestTradeApp,
)
from tradeexecutor.cli.vault_trade.setup import VaultTestRuntime, load_vault_test_state
from tradeexecutor.cli.vault_trade.simulation import (
    SimulatedVaultAttemptTimeout,
    SimulatedVaultRuntime,
    get_shared_simulation_fork_blocks,
    is_simulated_infrastructure_failure,
    queue_simulated_infrastructure_retry,
    raise_simulated_vault_attempt_timeout,
    rotate_simulated_rpc_upstreams,
    take_simulated_snapshots,
)
from tradeexecutor.cli.vault_trade.runner import (
    VaultAttemptContext,
    VaultTestBatchRunner,
    apply_vault_simulation_options,
    get_adapter_unsupported_detail,
    get_bridge_conflict,
    get_deposit_closed_detail,
    get_incorrect_deposit_status_reporting,
    get_incorrect_whitelisting_detail,
    get_redemption_unavailable_detail,
    get_unknown_deposit_permission_detail,
    has_async_vault_lifecycle,
    get_latest_attempt_vault_operation,
    get_whitelisting_needed_detail,
    normalise_vault_flow_failure,
    resolve_redemption_available,
    SimulatedSuccessOutcome,
    should_leave_deposit_open,
    validate_simulated_closed_deposit,
)
from tradeexecutor.cli.testtrade import BridgeProceedsUnavailable
from tradeexecutor.ethereum import web3config as web3config_module
from tradeexecutor.ethereum.vault import vault_routing
from tradeexecutor.ethereum.vault.vault_routing import (
    IncompatibleDepositAsset,
    convert_vault_flow_analysis,
    get_async_vault_request_transactions,
    reconcile_vault_redemption_amount,
    resolve_multi_asset_deposit_asset,
)
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.execution_model import AssetManagementMode


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


def test_async_vault_lifecycle_uses_manager_capability_metadata() -> None:
    """Manager flow metadata supplements static pair-kind async detection.

    1. Model real string capabilities for mixed and synchronous-only lifecycles.
    2. Model absent capability metadata and an intrinsically asynchronous pair.
    3. Verify every async source is recognised without changing safe fallbacks.

    Mocks cover metadata combinations without constructing chain-backed adapters.
    """
    # 1. Model real string capabilities for mixed and synchronous-only lifecycles.
    synchronous_pair = MagicMock()
    synchronous_pair.is_async_vault.return_value = False
    mixed_vault = MagicMock()
    mixed_vault.get_deposit_manager_capability.return_value = (
        VaultDepositManagerCapability(
            can_deposit=True,
            can_redeem=True,
            deposit_flow="synchronous",
            redemption_flow="asynchronous",
        )
    )
    synchronous_vault = MagicMock()
    synchronous_vault.get_deposit_manager_capability.return_value = (
        VaultDepositManagerCapability(
            can_deposit=True,
            can_redeem=True,
            deposit_flow="synchronous",
            redemption_flow="synchronous",
        )
    )

    # 2. Model absent capability metadata and an intrinsically asynchronous pair.
    missing_capability_vault = object()
    null_capability_vault = MagicMock()
    null_capability_vault.get_deposit_manager_capability.return_value = None
    asynchronous_pair = MagicMock()
    asynchronous_pair.is_async_vault.return_value = True

    # 3. Verify every async source is recognised without changing safe fallbacks.
    assert has_async_vault_lifecycle(synchronous_pair, mixed_vault) is True
    assert has_async_vault_lifecycle(synchronous_pair, synchronous_vault) is False
    assert (
        has_async_vault_lifecycle(synchronous_pair, missing_capability_vault) is False
    )
    assert has_async_vault_lifecycle(synchronous_pair, null_capability_vault) is False
    assert (
        has_async_vault_lifecycle(asynchronous_pair, missing_capability_vault) is True
    )


def test_shared_bridge_position_blocks_unrelated_vault() -> None:
    """One vault cannot consume another vault's per-chain CCTP lane.

    1. Model an in-transit bridge owned by the first vault.
    2. Verify both the owner and a second vault are blocked while it is in transit.
    3. Settle the transfer and verify only the owning vault may consume its capital.
    """
    first = VaultSpec(
        ChainId.arbitrum.value,
        "0x0000000000000000000000000000000000000001",
    )
    second = VaultSpec(
        ChainId.arbitrum.value,
        "0x0000000000000000000000000000000000000002",
    )
    trade = MagicMock()
    trade.get_status.return_value = TradeStatus.cctp_in_transit
    bridge_position = MagicMock()
    bridge_position.trades = {1: trade}
    bridge_position.other_data = {
        "vault_test_attempt": {
            "vault_id": first.as_string_id(),
            "phase": "bridge_out_pending",
        },
    }

    # 1. Model an in-transit bridge owned by the first vault.
    assert trade.get_status() == TradeStatus.cctp_in_transit

    # 2. Verify both the owner and a second vault are blocked while it is in transit.
    assert "still in transit" in get_bridge_conflict(bridge_position, first)
    assert "still in transit" in get_bridge_conflict(bridge_position, second)

    # 3. Settle the transfer and verify only the owning vault may consume its capital.
    trade.get_status.return_value = TradeStatus.success
    assert get_bridge_conflict(bridge_position, first) is None
    assert first.as_string_id() in get_bridge_conflict(bridge_position, second)


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


def test_whitelisted_vault_permission_gap_is_reported_before_execution() -> None:
    """A known vault allow-list denial gets a dedicated report outcome.

    1. Model an eth-defi vault adapter that requires a whitelisted depositor.
    2. Resolve the executor Safe address through the pricing route.
    3. Verify the report helper classifies only the denied account.
    """
    # 1. Model an eth-defi vault adapter that requires a whitelisted depositor.
    vault = MagicMock()
    vault.is_whitelisted_deposit.return_value = True
    vault.is_account_whitelisted.return_value = False

    # 2. Resolve the executor Safe address through the pricing route.
    pair = MagicMock()
    route = MagicMock()
    route.get_vault.return_value = vault
    route.get_owner_address.return_value = "0x0000000000000000000000000000000000000001"
    pricing_model = MagicMock()
    pricing_model.route.return_value = route
    attempt = MagicMock(
        pair=pair,
        pricing_model=pricing_model,
        executable_vault=vault,
    )

    # 3. Verify the report helper classifies only the denied account.
    detail = get_whitelisting_needed_detail(attempt)
    assert detail == (
        "Vault requires whitelisting for executor Safe "
        "0x0000000000000000000000000000000000000001"
    )
    vault.is_account_whitelisted.return_value = True
    assert get_whitelisting_needed_detail(attempt) is None
    vault.is_whitelisted_deposit.return_value = False
    assert get_whitelisting_needed_detail(attempt) is None


def test_whitelisting_needed_status_round_trips_as_report_outcome() -> None:
    """The new whitelisting outcome remains stable in state and reports.

    1. Record a whitelisting-needed diagnostic attempt.
    2. Serialise and reload the normal state JSON.
    3. Verify display status and report output preserve the exact condition.
    """
    # 1. Record a whitelisting-needed diagnostic attempt.
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
    position = record_attempt_result(
        state,
        create_vault_test_diagnostic_pair(spec, reserve_asset),
        spec,
        simulated=True,
        result="whitelisting-needed",
        detail="Vault requires whitelisting for executor Safe 0x1",
        outcome_data={"executor_safe": "0x1"},
    )

    # 2. Serialise and reload the normal state JSON.
    restored = State.from_json(state.to_json_safe())
    restored_position = restored.portfolio.get_position_by_id(position.position_id)

    # 3. Verify display status and report output preserve the exact condition.
    assert get_vault_test_status(restored_position) == "whitelisting-needed"
    report = export_vault_test_report(
        restored,
        [{"vault id": spec.as_string_id(), "status": "whitelisting-needed"}],
    )
    assert report["results"][0]["attempt"]["result"] == "whitelisting-needed"
    assert report["results"][0]["attempt"]["outcome_data"] == {"executor_safe": "0x1"}


def test_async_request_only_status_round_trips_as_report_outcome() -> None:
    """A request-only simulation is distinct from unsupported settlement.

    1. Record an async request-only simulated attempt.
    2. Serialise and reload the normal state JSON.
    3. Verify the raw result and display status remain distinct and readable.
    """
    # 1. Record an async request-only simulated attempt.
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
    position = record_attempt_result(
        state,
        create_vault_test_diagnostic_pair(spec, reserve_asset),
        spec,
        simulated=True,
        result="async_request_only",
        detail="Async deposit request completed; full lifecycle was not requested",
    )

    # 2. Serialise and reload the normal state JSON.
    restored = State.from_json(state.to_json_safe())
    restored_position = restored.portfolio.get_position_by_id(position.position_id)

    # 3. Verify the raw result and display status remain distinct and readable.
    assert get_vault_test_status(restored_position) == "async request only"
    assert (
        restored_position.other_data["vault_test_attempt"]["result"]
        == "async_request_only"
    )


def test_vault_test_failure_persists_redacted_traceback_and_revert_evidence() -> None:
    """Vault-test failures preserve reporter-ready diagnostics in the state.

    1. Create a failed transaction containing its receipt block and Anvil trace.
    2. Capture an exception that includes a credential-bearing JSON-RPC URL.
    3. Persist the diagnostic result and verify state serialisation retains the
       traceback, blocks and revert evidence while redacting the URL.
    """
    # 1. Create a failed transaction containing its receipt block and Anvil trace.
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
    transaction = MagicMock(
        chain_id=ChainId.arbitrum.value,
        tx_hash="0xdeadbeef",
        contract_address=spec.vault_address,
        function_selector="deposit",
        wrapped_target=None,
        wrapped_function_selector=None,
        nonce=7,
        block_number=123_456,
        block_hash="0xblock",
        status=False,
        revert_reason="custom error 0x12345678",
        stack_trace="revert: VaultNotOpen()",
    )
    trade = MagicMock(trade_id=42, blockchain_transactions=[transaction])
    failed_position = MagicMock(position_id=3, trades={42: trade})
    state.portfolio.get_all_positions = MagicMock(return_value=[failed_position])

    # 2. Capture an exception that includes a credential-bearing JSON-RPC URL.
    web3 = MagicMock()
    web3.eth.block_number = 654_321
    web3config = MagicMock(connections={ChainId.arbitrum: web3})
    try:
        raise RuntimeError("RPC wss://rpc.example.test/secret-key rejected the call")
    except RuntimeError as error:
        diagnostics = capture_vault_test_error(
            error,
            state=state,
            original_trade_ids=set(),
            web3config=web3config,
            phase="execute",
        )

    # 3. Persist the diagnostics and verify external consumers can read them safely.
    pair = create_vault_test_diagnostic_pair(spec, reserve_asset)
    position = record_attempt_result(
        state,
        pair,
        spec,
        simulated=True,
        result="failed",
        detail="deposit failed",
        error=diagnostics,
    )
    payload = json.loads(state.to_json_safe())
    error_payload = payload["portfolio"]["closed_positions"][str(position.position_id)][
        "other_data"
    ]["vault_test_attempt"]["error"]

    assert error_payload["phase"] == "execute"
    assert (
        error_payload["chain_blocks"][str(ChainId.arbitrum.value)]["block_number"]
        == 654_321
    )
    assert error_payload["transactions"] == [
        {
            "position_id": 3,
            "trade_id": 42,
            "chain_id": ChainId.arbitrum.value,
            "tx_hash": "0xdeadbeef",
            "contract_address": spec.vault_address,
            "function_selector": "deposit",
            "wrapped_target": None,
            "wrapped_function_selector": None,
            "nonce": 7,
            "block_number": 123_456,
            "block_hash": "0xblock",
            "status": False,
            "revert_reason": "custom error 0x12345678",
            "stack_trace": "revert: VaultNotOpen()",
        }
    ]
    assert "secret-key" not in error_payload["traceback"]
    assert "<redacted-url>" in error_payload["traceback"]


def test_vault_test_report_retains_provenance_and_legacy_result_values() -> None:
    """Reports expose authoritative diagnostics without mutating legacy state.

    1. Create a diagnostic attempt with reproducible provenance.
    2. Mark its raw result as a future/unknown legacy value and serialise it.
    3. Build the compact report and verify display normalisation never overwrites
       the raw persisted result.
    """
    # 1. Create a diagnostic attempt with reproducible provenance.
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
    position = record_attempt_result(
        state,
        create_vault_test_diagnostic_pair(spec, reserve_asset),
        spec,
        simulated=True,
        result="transaction_reverted",
        attempt_id="attempt-1",
        operation="deposit",
        provenance={"fork_blocks": {"42161": 123_456}},
    )

    # 2. Mark its raw result as a future/unknown legacy value and serialise it.
    attempt = position.other_data["vault_test_attempt"]
    attempt["result"] = "future_executor_result"
    restored = State.from_json(state.to_json_safe())
    restored_position = restored.portfolio.get_position_by_id(position.position_id)

    # 3. Build the report and verify display normalisation never overwrites state.
    assert get_vault_test_status(restored_position) == "legacy result"
    report = export_vault_test_report(
        restored,
        [{"vault id": spec.as_string_id(), "status": "legacy result"}],
    )
    assert report["results"][0]["attempt"]["result"] == "future_executor_result"
    assert report["results"][0]["attempt"]["provenance"] == {
        "fork_blocks": {"42161": 123_456}
    }


def test_vault_failure_classifier_uses_transaction_evidence() -> None:
    """Failure status is determined by lifecycle evidence rather than text.

    1. Classify a preflight exception with no transaction evidence.
    2. Classify reverted and broadcast receipts during execution.
    3. Verify unsigned call context and no-evidence execution classifications.
    """
    # 1. Classify a preflight exception with no transaction evidence.
    assert (
        classify_vault_test_failure(phase="preflight", error_data={})
        == "preflight_failed"
    )

    # 2. Classify reverted and broadcast receipts during execution.
    assert (
        classify_vault_test_failure(
            phase="execute",
            error_data={"transactions": [{"status": False}]},
        )
        == "transaction_reverted"
    )
    assert (
        classify_vault_test_failure(
            phase="execute",
            error_data={"transactions": [{"status": True}]},
        )
        == "execution_failed"
    )
    assert (
        classify_vault_test_failure(
            phase="execute",
            error_data={"transactions": [{"tx_hash": "0xdeadbeef"}]},
        )
        == "broadcast_failed"
    )

    # 3. Verify unsigned call context and no-evidence execution classifications.
    assert (
        classify_vault_test_failure(
            phase="execute",
            error_data={"call_context": [{"function_selector": "deposit"}]},
        )
        == "gas_estimation_reverted"
    )
    assert (
        classify_vault_test_failure(phase="execute", error_data={})
        == "execution_failed"
    )


def test_vault_error_call_context_contains_replayable_unsigned_calldata() -> None:
    """An estimate failure report is sufficient for eth-defi to replay the call.

    1. Model an unsigned vault transaction that stopped before a receipt.
    2. Capture the failure diagnostics from the attempted trade.
    3. Verify the report contains target, sender, gas and full unsigned calldata.
    """
    # 1. Model an unsigned vault transaction that stopped before a receipt.
    transaction = MagicMock(
        chain_id=ChainId.base.value,
        from_address="0x0000000000000000000000000000000000000001",
        contract_address="0x0000000000000000000000000000000000000002",
        function_selector="deposit",
        wrapped_target=None,
        wrapped_function_selector=None,
        nonce=7,
        block_number=None,
    )
    transaction.details = {
        "data": "0xd0e30db0",
        "value": 123,
        "gas": 456_789,
        "maxFeePerGas": 10,
        "maxPriorityFeePerGas": 2,
    }
    trade = MagicMock(trade_id=42, blockchain_transactions=[transaction])
    position = MagicMock(position_id=3, trades={42: trade})
    state = MagicMock()
    state.portfolio.get_all_positions.return_value = [position]

    # 2. Capture the failure diagnostics from the attempted trade.
    try:
        raise RuntimeError("estimate reverted")
    except RuntimeError as error:
        diagnostics = capture_vault_test_error(
            error,
            state=state,
            original_trade_ids=set(),
            web3config=None,
            phase="execute",
        )

    # 3. Verify the report contains target, sender, gas and full unsigned calldata.
    assert diagnostics["call_context"] == [
        {
            "position_id": 3,
            "trade_id": 42,
            "chain_id": ChainId.base.value,
            "sender": "0x0000000000000000000000000000000000000001",
            "target": "0x0000000000000000000000000000000000000002",
            "function_selector": "deposit",
            "wrapped_target": None,
            "wrapped_function_selector": None,
            "value": "123",
            "gas": 456_789,
            "gas_price": None,
            "max_fee_per_gas": 10,
            "max_priority_fee_per_gas": 2,
            "nonce": 7,
            "calldata": "0xd0e30db0",
            "calldata_hash": "5cd92c6d850367a4db763ab4a4c33567ade46ebfddfdd73cd31d130db24c6b0f",
        }
    ]


def test_vault_attempt_stamp_replaces_previous_attempt_identity() -> None:
    """A resumed vault lifecycle retains the identity of its latest operation.

    1. Persist a deposit diagnostic with an initial attempt id.
    2. Stamp the same position as a later redemption attempt.
    3. Verify the reporter-visible metadata identifies the latter attempt.
    """
    # 1. Persist a deposit diagnostic with an initial attempt id.
    state = State()
    reserve_asset = AssetIdentifier(
        ChainId.base.value,
        "0x0000000000000000000000000000000000000010",
        "USDC",
        6,
    )
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    spec = VaultSpec(
        ChainId.base.value,
        "0x0000000000000000000000000000000000000020",
    )
    position = record_attempt_result(
        state,
        create_vault_test_diagnostic_pair(spec, reserve_asset),
        spec,
        simulated=False,
        result="failed",
        attempt_id="deposit-attempt",
        operation="deposit",
    )

    # 2. Stamp the same position as a later redemption attempt.
    stamp_position_vault_test_attempt(
        position,
        spec,
        simulated=False,
        phase="redemption_requested",
        attempt_id="redeem-attempt",
        operation="redeem",
    )

    # 3. Verify the reporter-visible metadata identifies the latter attempt.
    attempt = position.other_data["vault_test_attempt"]
    assert attempt["attempt_id"] == "redeem-attempt"
    assert attempt["operation"] == "redeem"


def test_async_anvil_settlement_option_requires_simulated_mode() -> None:
    """Async Anvil settlement cannot accidentally run against a live vault.

    1. Validate an opt-in simulated Anvil invocation.
    2. Request the option in real mode.
    3. Verify validation rejects the unsafe combination before any RPC is opened.
    """
    # 1. Validate an opt-in simulated Anvil invocation.
    _validate_vault_test_options(
        auto_simulated=True,
        auto_real=False,
        rerun=False,
        settle_async_on_anvil=True,
        asset_management_mode=AssetManagementMode.lagoon,
    )

    # 2. Request the option in real mode.
    with pytest.raises(RuntimeError, match="requires --auto-simulated"):
        # 3. Verify validation rejects the unsafe combination before any RPC is opened.
        _validate_vault_test_options(
            auto_simulated=False,
            auto_real=True,
            rerun=False,
            settle_async_on_anvil=True,
            asset_management_mode=AssetManagementMode.lagoon,
        )


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


def test_shared_simulation_fork_blocks_skip_chains_without_archive_history() -> None:
    """Vault batches share canonical cacheable fork blocks but leave Base and Monad at live tips.

    1. Request one Ethereum, one Base and one Monad vault.
    2. Resolve the matrix's shared cached fork blocks.
    3. Verify only Ethereum receives eth-defi's canonical historical height.
    """
    specs = [
        VaultSpec(ChainId.ethereum.value, "0x0000000000000000000000000000000000000000"),
        VaultSpec(ChainId.base.value, "0x0000000000000000000000000000000000000001"),
        VaultSpec(ChainId.monad.value, "0x0000000000000000000000000000000000000002"),
    ]

    # 1. + 2. Resolve the shared fork map.
    fork_blocks = get_shared_simulation_fork_blocks(specs)

    # 3. Base and Monad must capture a live tip; Ethereum uses a shared cacheable block.
    assert fork_blocks == {ChainId.ethereum: MIDNIGHT_BLOCKS[ChainId.ethereum.value]}


def test_simulated_rpc_retry_rotates_upstreams() -> None:
    """A replacement generation begins with a different archive provider.

    1. Configure a simulated chain with ordered primary and fallback endpoints.
    2. Rotate the order after an infrastructure failure.
    3. Verify the former fallback is first and non-RPC inputs are unchanged.
    """
    rpc_kwargs = {
        "json_rpc_base": "https://base-primary.example https://base-fallback.example",
        "json_rpc_ethereum": "https://ethereum-only.example",
        "unit_testing": False,
    }

    # 1. + 2. Rotate the configured archive-provider order.
    rotate_simulated_rpc_upstreams(rpc_kwargs)

    # 3. The retry starts from the alternate provider without touching other values.
    assert rpc_kwargs == {
        "json_rpc_base": "https://base-fallback.example https://base-primary.example",
        "json_rpc_ethereum": "https://ethereum-only.example",
        "unit_testing": False,
    }


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


def test_simulated_infrastructure_failure_detects_wrapped_empty_anvil_response() -> None:
    """An empty Anvil response wrapped by Web3 still replaces the fork.

    1. Model eth-defi's missing-block exception during an ``eth_call``.
    2. Wrap it as Web3 does when ABI decoding receives no return bytes.
    3. Verify the known wrapper chain remains eligible for one clean rerun.
    """
    # 1. + 2. Model Web3's decode wrapper while retaining eth-defi's context.
    caught_error = None
    try:
        raise ProbablyNodeHasNoBlock("Anvil returned an empty eth_call response")
    except ProbablyNodeHasNoBlock:
        try:
            raise ValueError("ABI decoder received no data")
        except ValueError as decode_error:
            try:
                raise BadFunctionCallOutput("Could not call contract") from decode_error
            except BadFunctionCallOutput as wrapped_error:
                caught_error = wrapped_error

    # 3. The specific Web3 wrapper must retain the infrastructure classification.
    assert caught_error is not None
    assert is_simulated_infrastructure_failure(caught_error) is True


def test_repeated_infrastructure_failure_does_not_restart_next_vault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted infrastructure retry records the failure without recycling again.

    1. Construct a simulated runner whose vault already used its one retry.
    2. Record another infrastructure failure using isolated persistence doubles.
    3. Verify the next unrelated vault does not inherit an unnecessary restart.
    """
    spec = VaultSpec(ChainId.base.value, "0x0000000000000000000000000000000000000001")
    runner = VaultTestBatchRunner(
        runtime=MagicMock(reserve_asset=MagicMock(), web3config=MagicMock()),
        client=MagicMock(),
        vault_universe=MagicMock(),
        state=MagicMock(),
        store=MagicMock(),
        vault_specs=[],
        amount=Decimal("1"),
        max_slippage=0.005,
        auto_simulated=True,
        rerun=False,
    )
    runner.infrastructure_restart_counts[spec.as_string_id()] = 1

    # 1. Isolate terminal-result persistence from the recovery decision.
    monkeypatch.setattr(
        runner_module, "capture_vault_test_error", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        runner_module, "record_attempt_result", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(VaultTestBatchRunner, "_append_result", lambda *args: None)

    # 2. Record the repeated failure.
    runner._handle_infrastructure_failure(
        ReadTimeout("localhost Anvil timed out"),
        spec,
        MagicMock(),
        MagicMock(),
        None,
        original_trade_ids=set(),
        phase="execute",
    )

    # 3. The terminal result must not request a new generation.
    assert runner.restart_requested is None
    runner.store.sync.assert_called_once_with(runner.state)


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

    monkeypatch.setattr(simulation_module, "make_anvil_custom_rpc_request", snapshot)

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
        fork_block_number=49_153_000,
    )

    # 3. Verify localhost retries are disabled and upstream failover has a bounded budget.
    assert result is web3
    assert captured["local_url"] == anvil.json_rpc_url
    assert captured["web3_kwargs"]["retries"] == 0
    assert captured["web3_kwargs"]["default_http_timeout"] == (3.0, 40.0)
    assert captured["launch_kwargs"]["proxy_multiple_upstream"] is True
    assert captured["launch_kwargs"]["fork_block_number"] == 49_153_000


def test_simulated_runtime_replacement_reuses_first_fork_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anvil replacements retain fixed fork heights for cache reuse.

    1. Model a first simulation generation that captured fixed chain heights.
    2. Replace the failed generation through the normal runtime owner.
    3. Verify the replacement receives those heights instead of forking new tips.
    """
    old_runtime = MagicMock()
    old_runtime.generation = 1
    old_runtime.pinned_fork_blocks = {
        ChainId.base: 49_153_000,
        ChainId.arbitrum: 488_000_000,
    }
    reserve_asset = MagicMock()
    replacement = MagicMock()
    replacement.reserve_asset = reserve_asset
    replacement.web3config = MagicMock()
    replacement.deployment = MagicMock()
    replacement.execution_model = MagicMock()
    replacement.sync_model = MagicMock()
    captured = {}

    # 1. Prepare a captured first-generation fork map.
    runtime = VaultTestRuntime(
        web3config=MagicMock(),
        deployment=MagicMock(),
        execution_model=MagicMock(),
        sync_model=MagicMock(),
        reserve_asset=reserve_asset,
        simulated_runtime=old_runtime,
        simulated_runtime_kwargs={
            "vault_specs": [],
            "rpc_kwargs": {
                "json_rpc_base": "https://base-primary.example https://base-fallback.example",
            },
        },
    )

    def start_replacement(**kwargs):
        captured.update(kwargs)
        return replacement

    monkeypatch.setattr(
        "tradeexecutor.cli.vault_trade.setup.start_simulated_vault_runtime_with_replacement",
        start_replacement,
    )

    # 2. Replace the failed generation.
    runtime.replace_simulation(ReadTimeout("localhost Anvil timed out"))

    # 3. Verify replacements pin the first generation's heights.
    old_runtime.close.assert_called_once_with()
    assert captured["generation"] == 2
    assert captured["pinned_fork_blocks"] == old_runtime.pinned_fork_blocks
    assert captured["rpc_kwargs"]["json_rpc_base"] == (
        "https://base-fallback.example https://base-primary.example"
    )


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


def test_failure_attachment_ignores_positions_from_previous_attempts() -> None:
    """A new failed attempt must not overwrite its predecessor's result.

    1. Model a successful real vault position created by an earlier attempt.
    2. Invoke the failure-attachment path with that position in the baseline.
    3. Verify the earlier attempt's success metadata remains unchanged.
    """
    # 1. Model a successful real vault position created by an earlier attempt.
    spec = VaultSpec(ChainId.base.value, "0x0000000000000000000000000000000000000001")
    old_trade = MagicMock(trade_id=1)
    old_position = MagicMock(position_id=1, simulated=False, trades={1: old_trade})
    old_position.pair.chain_id = spec.chain_id
    old_position.pair.pool_address = spec.vault_address
    old_position.other_data = {"vault_test_attempt": {"result": "success"}}
    state = MagicMock()
    state.portfolio.get_all_positions.return_value = [old_position]
    state.portfolio.get_all_trades.return_value = [old_trade]

    # 2. Invoke the failure-attachment path with that position in the baseline.
    runner = object.__new__(VaultTestBatchRunner)
    runner.auto_simulated = False
    runner.state = state
    runner.current_attempt = VaultAttemptContext(
        attempt_id="attempt-2",
        original_position_ids={1},
        original_trade_ids={1},
        provenance={},
        phase="preflight",
    )
    attached_position_id = runner._attach_failure_to_attempt_position(
        spec=spec,
        error_state=state,
        result="preflight_failed",
        detail="RPC unavailable",
        error_data={},
        outcome_data=None,
        previous=old_position,
    )

    # 3. Verify the earlier attempt's success metadata remains unchanged.
    assert attached_position_id is None
    assert old_position.other_data["vault_test_attempt"] == {"result": "success"}


def test_vault_flow_failures_have_typed_report_outcomes() -> None:
    """Typed eth-defi flow failures retain their protocol capacity context.

    1. Create a redemption failure with a requested amount above available capacity.
    2. Normalise the error through the vault-test reporting helper.
    3. Verify the result and JSON-safe capacity evidence are explicit.
    """
    # 1. Create a redemption failure with a requested amount above available capacity.
    error = VaultFlowUnavailable(
        "Only part of the redemption is currently available at https://rpc.example/api-key",
        protocol="csigma",
        direction="redeem",
        phase="preflight",
        requested_raw_amount=200,
        available_raw_amount=100,
    )
    error.minimum_raw_amount = 10
    error.next_open = datetime.datetime(2026, 7, 24, 12, 30)

    # 2. Normalise the error through the vault-test reporting helper.
    result, detail, outcome_data = normalise_vault_flow_failure(error)

    # 3. Verify the result and JSON-safe capacity evidence are explicit.
    assert result == "redemption_capacity_limited"
    assert (
        detail == "Only part of the redemption is currently available at <redacted-url>"
    )
    assert outcome_data == {
        "protocol": "csigma",
        "direction": "redeem",
        "phase": "preflight",
        "decoded_error": None,
        "preflight_result": None,
        "requested_raw_amount": "200",
        "available_raw_amount": "100",
        "minimum_raw_amount": "10",
        "next_open": "2026-07-24T12:30:00",
    }


def test_simulated_vault_detects_incorrect_json_whitelist_status() -> None:
    """A simulated fork exposes stale vault JSON deposit permissions.

    1. Model matching downloaded and onchain whitelist policies.
    2. Change the onchain policy while retaining the downloaded status.
    3. Verify the mismatch is a structured terminal diagnostic and unknown stays non-blocking.
    """
    # 1. Model matching downloaded and onchain whitelist policies.
    matching_attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(deposit_permission="whitelisted"),
        ),
        executable_vault=SimpleNamespace(
            is_whitelisted_deposit=lambda: True,
        ),
    )
    assert get_incorrect_whitelisting_detail(matching_attempt) is None

    # 2. Change the onchain policy while retaining the downloaded status.
    mismatched_attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(deposit_permission="whitelisted"),
        ),
        executable_vault=SimpleNamespace(
            is_whitelisted_deposit=lambda: False,
        ),
    )
    detail, outcome_data = get_incorrect_whitelisting_detail(mismatched_attempt)

    # 3. Verify the mismatch is structured and unknown remains non-blocking.
    assert "does not match" in detail
    assert outcome_data == {
        "json_deposit_permission": "whitelisted",
        "onchain_deposit_permission": "permissionless",
    }
    unknown_attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(deposit_permission="unknown"),
        ),
        executable_vault=SimpleNamespace(
            is_whitelisted_deposit=lambda: False,
        ),
    )
    assert get_incorrect_whitelisting_detail(unknown_attempt) is None


def test_simulated_vault_records_incorrect_json_whitelist_status() -> None:
    """The simulated deposit runner persists a whitelist metadata disagreement.

    1. Prepare an automatic simulated deposit with disagreeing JSON and fork policies.
    2. Process the attempt without allowing it to construct a deposit transaction.
    3. Verify the dedicated terminal status and both policy values are recorded.
    """
    # 1. Prepare an automatic simulated deposit with disagreeing JSON and fork policies.
    runner = object.__new__(VaultTestBatchRunner)
    runner.auto_simulated = True
    runner.deposit_asset = None
    runner.current_attempt = SimpleNamespace()
    runner.state = SimpleNamespace(
        portfolio=SimpleNamespace(get_all_trades=lambda: []),
    )
    attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(deposit_permission="permissionless"),
        ),
        executable_vault=SimpleNamespace(
            is_whitelisted_deposit=lambda: True,
        ),
        pair=MagicMock(),
        spec=MagicMock(),
    )

    # 2. Process the attempt without allowing it to construct a deposit transaction.
    with (
        patch.object(VaultTestBatchRunner, "_choose_operation", return_value="deposit"),
        patch.object(VaultTestBatchRunner, "_record_terminal_result") as record_result,
    ):
        stop_batch = runner._process_attempt(attempt, MagicMock())

    # 3. Verify the dedicated terminal status and both policy values are recorded.
    assert stop_batch is False
    record_result.assert_called_once()
    assert record_result.call_args.kwargs == {
        "result": "whitelisted-incorrectly",
        "detail": "Vault JSON deposit permission 'permissionless' does not match "
        "simulated onchain permission 'whitelisted'",
        "outcome_data": {
            "json_deposit_permission": "permissionless",
            "onchain_deposit_permission": "whitelisted",
        },
    }


def test_simulated_vault_uses_explicit_json_deposit_status() -> None:
    """Only an explicit recognised JSON deposit status creates mismatch evidence.

    1. Compare explicit open JSON status with a closed onchain observation.
    2. Verify status provenance and the legacy reason are retained in the report.
    3. Confirm missing, unknown and future JSON values remain non-blocking.
    """
    # 1. Compare explicit open JSON status with a closed onchain observation.
    observed_at = datetime.datetime(2026, 7, 30, 7, 0)
    attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(
                deposit_status=VaultDepositStatus.open,
                deposit_closed_reason=None,
                deposit_status_source="eth-defi",
                deposit_status_observed_at=observed_at,
                deposit_status_observed_block=23_456_789,
            ),
        ),
    )
    mismatch = get_incorrect_deposit_status_reporting(attempt, "closed")

    # 2. Verify status provenance and the legacy reason are retained in the report.
    assert mismatch == {
        "vault_json_deposit_status": "open",
        "vault_json_deposit_closed_reason": None,
        "onchain_deposit_status": "closed",
        "vault_json_deposit_status_source": "eth-defi",
        "vault_json_deposit_status_observed_at": "2026-07-30T07:00:00",
        "vault_json_deposit_status_observed_block": 23_456_789,
    }

    # 3. Confirm missing, unknown and future JSON values remain non-blocking.
    for status in (None, VaultDepositStatus.unknown, "maintenance"):
        attempt.vault.metadata.deposit_status = status
        assert get_incorrect_deposit_status_reporting(attempt, "closed") is None


def test_simulated_closed_deposit_records_guard_validation_evidence() -> None:
    """A typed closure becomes a non-broadcast GuardV0 success.

    1. Raise a typed deposit closure from the manager's normal request.
    2. Return eth-defi evidence for the matching validation-only request.
    3. Verify trade-executor preserves both closure and GuardV0 call evidence.
    """
    # 1. Raise a typed deposit closure from the manager's normal request.
    owner = "0x0000000000000000000000000000000000000001"
    vault_address = "0x0000000000000000000000000000000000000002"
    guard_address = "0x0000000000000000000000000000000000000003"
    asset_manager = "0x0000000000000000000000000000000000000004"
    closure = VaultFlowUnavailable(
        "Global deposit cap reached",
        protocol="yearn",
        vault_address=vault_address,
        caller=owner,
        direction="deposit",
        phase="preflight",
        preflight_result="deposit_closed",
        requested_raw_amount=1_000_000,
        available_raw_amount=0,
    )
    manager = MagicMock()
    manager.create_deposit_request.side_effect = closure
    validation_request = object()
    manager.create_deposit_request_for_guard_validation.return_value = (
        validation_request
    )
    executable_vault = MagicMock(
        denomination_token=MagicMock(convert_to_raw=MagicMock(return_value=1_000_000)),
    )
    executable_vault.get_deposit_manager.return_value = manager

    route = MagicMock()
    route.get_owner_address.return_value = owner
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
    )
    guard = SimpleNamespace(address=guard_address)
    execution_vault = SimpleNamespace(
        safe_address=owner,
        trading_strategy_module=guard,
    )
    tx_builder = MagicMock(vault=execution_vault)
    tx_builder.get_gas_wallet_address.return_value = asset_manager
    runtime = SimpleNamespace(
        deployment=SimpleNamespace(primary_chain_id=ChainId.ethereum),
        execution_model=SimpleNamespace(
            tx_builder=tx_builder,
            satellite_vaults={},
        ),
    )
    evidence = SimpleNamespace(
        vault_address=vault_address,
        owner=owner,
        raw_amount=1_000_000,
        preflight_result="deposit_closed",
        closure_reason=closure.reason,
        calls=(
            SimpleNamespace(
                target=vault_address,
                selector=HexBytes("0x6e553f65"),
                calldata="0x6e553f65",
            ),
        ),
    )

    # 2. Mock eth-defi's static Guard call to isolate executor evidence serialisation.
    with patch.object(
        runner_module,
        "validate_closed_deposit_request_with_guard",
        return_value=evidence,
    ) as validate_guard:
        result = validate_simulated_closed_deposit(
            attempt,
            runtime,
            Decimal("1"),
        )

    # 3. Verify trade-executor preserves both closure and GuardV0 call evidence.
    detail, outcome_data = result
    assert detail == "Global deposit cap reached"
    assert outcome_data["preflight_result"] == "deposit_closed"
    assert outcome_data["requested_raw_amount"] == "1000000"
    assert outcome_data["available_raw_amount"] == "0"
    assert outcome_data["deposit_executed"] is False
    assert outcome_data["guard_validation"] == {
        "mode": "closed_deposit_guard_v0",
        "guard_address": guard_address,
        "asset_manager": asset_manager,
        "vault_address": vault_address,
        "owner": owner,
        "raw_amount": "1000000",
        "preflight_result": "deposit_closed",
        "closure_reason": "Global deposit cap reached",
        "calls": [
            {
                "target": vault_address,
                "selector": "6e553f65",
                "calldata": "0x6e553f65",
            },
        ],
    }
    validate_guard.assert_called_once_with(
        validation_request,
        closure,
        guard,
        asset_manager,
    )
    manager.create_deposit_request.assert_called_once_with(
        owner=owner,
        raw_amount=1_000_000,
        check_enough_token=True,
    )


def test_simulated_closed_deposit_ignores_satellite_prefunding() -> None:
    """Satellite preflight waits for the normal CCTP funding lifecycle.

    1. Prepare an open satellite vault whose Safe has not received USDC yet.
    2. Run the closed-deposit preflight before the normal simulated trade.
    3. Verify only the temporary token-balance check is disabled.
    """
    # 1. Prepare an open satellite vault whose Safe has not received USDC yet.
    owner = "0x0000000000000000000000000000000000000001"
    manager = MagicMock()
    executable_vault = MagicMock(
        denomination_token=MagicMock(convert_to_raw=MagicMock(return_value=1_000_000)),
    )
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = owner
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.base.value),
        executable_vault=executable_vault,
    )
    runtime = SimpleNamespace(
        deployment=SimpleNamespace(primary_chain_id=ChainId.ethereum),
    )

    # 2. Run the closed-deposit preflight before the normal simulated trade.
    result = validate_simulated_closed_deposit(
        attempt,
        runtime,
        Decimal("1"),
    )

    # 3. Verify only the temporary token-balance check is disabled.
    assert result is None
    manager.create_deposit_request.assert_called_once_with(
        owner=owner,
        raw_amount=1_000_000,
        check_enough_token=False,
    )


def test_simulated_closed_deposit_requires_typed_closure() -> None:
    """An untyped manager refusal cannot enter GuardV0-only validation.

    1. Raise an ordinary deposit preflight failure without a closed result.
    2. Attempt the closed-deposit simulation helper.
    3. Verify the original refusal propagates and no bypass request is built.
    """
    # 1. Raise an ordinary deposit preflight failure without a closed result.
    refusal = VaultFlowUnavailable(
        "Deposit cannot be prepared",
        protocol="example",
        direction="deposit",
        phase="preflight",
    )
    manager = MagicMock()
    manager.create_deposit_request.side_effect = refusal
    executable_vault = MagicMock(
        denomination_token=MagicMock(convert_to_raw=MagicMock(return_value=1)),
    )
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = (
        "0x0000000000000000000000000000000000000001"
    )
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
    )

    # 2. Attempt the closed-deposit simulation helper.
    with pytest.raises(VaultFlowUnavailable) as exc_info:
        validate_simulated_closed_deposit(
            attempt,
            MagicMock(),
            Decimal("1"),
        )

    # 3. Verify the original refusal propagates and no bypass request is built.
    assert exc_info.value is refusal
    manager.create_deposit_request_for_guard_validation.assert_not_called()


def test_simulated_closed_deposit_discloses_public_eligibility_funding() -> None:
    """The closure probe discloses public eligibility funding on Anvil.

    1. Raise the D2-style typed minimum result seen before simulated Safe funding.
    2. Run the early closed-deposit probe with the production 1,001 USDC amount.
    3. Verify the intervention is recorded before the funded retry succeeds.
    """
    # 1. Raise the D2-style typed minimum result seen before simulated Safe funding.
    refusal = VaultFlowUnavailable(
        "Public USDC eligibility minimum not met",
        protocol="D2 Finance",
        asset_address="0x0000000000000000000000000000000000000002",
        direction="deposit",
        phase="preflight",
        decoded_error="InsufficientEligibilityBalance",
        preflight_result="below_minimum",
        available_raw_amount=0,
        minimum_raw_amount=1_000_001,
    )
    manager = MagicMock()
    manager.create_deposit_request.side_effect = [refusal, MagicMock()]
    denomination_token = MagicMock(
        address="0x0000000000000000000000000000000000000002",
        convert_to_raw=MagicMock(return_value=1_001_000_000),
    )
    executable_vault = MagicMock(
        denomination_token=denomination_token,
    )
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = (
        "0x0000000000000000000000000000000000000001"
    )
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
        interventions=[],
    )
    runtime = MagicMock()
    chain_web3 = MagicMock()
    runtime.web3config.get_connection.return_value = chain_web3

    # 2. Run the early closed-deposit probe with the production 1,001 USDC amount.
    with (
        patch.object(runner_module, "fund_erc20_on_anvil") as fund_eligibility,
        patch.object(runner_module, "is_anvil", return_value=True),
    ):
        result = validate_simulated_closed_deposit(
            attempt,
            runtime,
            Decimal("1001"),
        )

    # 3. Verify the intervention is recorded before the funded retry succeeds.
    assert result is None
    fund_eligibility.assert_called_once_with(
        chain_web3,
        refusal.asset_address,
        route.get_owner_address.return_value,
        1_001_000_000,
    )
    assert manager.create_deposit_request.call_count == 2
    manager.create_deposit_request_for_guard_validation.assert_not_called()
    assert attempt.interventions == [
        {
            "kind": "eligibility_asset_funded",
            "token": refusal.asset_address,
            "target": route.get_owner_address.return_value,
            "raw_amount": "1001000000",
            "original_reason": str(refusal),
            "original_preflight_result": "below_minimum",
        }
    ]


def test_simulated_closed_deposit_defers_unfunded_typed_minimum() -> None:
    """A non-eligibility minimum belongs to the normal funded lifecycle.

    1. Raise an Accountable-style typed minimum from the early closure probe.
    2. Run the probe without a D2 eligibility-asset intervention.
    3. Confirm it returns without retrying or constructing Guard calldata.
    """
    # 1. Raise an Accountable-style typed minimum from the early closure probe.
    refusal = VaultFlowUnavailable(
        "Deposit below protocol minimum",
        protocol="Accountable",
        direction="deposit",
        phase="preflight",
        decoded_error="InsufficientAmount",
        preflight_result="below_minimum",
        minimum_raw_amount=1_000_000_000,
    )
    manager = MagicMock()
    manager.create_deposit_request.side_effect = refusal
    executable_vault = MagicMock(
        denomination_token=MagicMock(convert_to_raw=MagicMock(return_value=1_001_000_000)),
    )
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = "0x0000000000000000000000000000000000000001"
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
        interventions=[],
    )

    # 2. Run the probe without a D2 eligibility-asset intervention.
    result = validate_simulated_closed_deposit(
        attempt,
        MagicMock(),
        Decimal("1001"),
    )

    # 3. Confirm it returns without retrying or constructing Guard calldata.
    assert result is None
    manager.create_deposit_request.assert_called_once()
    manager.create_deposit_request_for_guard_validation.assert_not_called()
    assert attempt.interventions == []


def test_simulated_closed_deposit_is_persisted_as_terminal_success() -> None:
    """The automatic runner records the combined closed-vault outcome.

    1. Prepare an automatic deposit attempt with successful Guard evidence.
    2. Process the attempt without entering the normal trade lifecycle.
    3. Verify the combined outcome is persisted with its evidence.
    """
    # 1. Prepare an automatic deposit attempt with successful Guard evidence.
    runner = object.__new__(VaultTestBatchRunner)
    runner.auto_simulated = True
    runner.deposit_asset = None
    runner.current_attempt = SimpleNamespace()
    runner.runtime = MagicMock()
    runner.amount = Decimal("1")
    runner.state = SimpleNamespace(
        portfolio=SimpleNamespace(get_all_trades=lambda: []),
    )
    attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(
                deposit_status=VaultDepositStatus.closed,
                deposit_closed_reason="Global deposit cap reached",
            ),
        ),
        executable_vault=MagicMock(),
        pair=MagicMock(),
        spec=MagicMock(),
    )
    evidence = {
        "deposit_executed": False,
        "guard_validation": {"mode": "closed_deposit_guard_v0"},
    }

    # 2. Mock routing helpers so this test isolates terminal-result persistence.
    with (
        patch.object(VaultTestBatchRunner, "_choose_operation", return_value="deposit"),
        patch.object(VaultTestBatchRunner, "_record_terminal_result") as record_result,
        patch.object(
            runner_module,
            "get_incorrect_whitelisting_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "get_whitelisting_needed_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "validate_simulated_closed_deposit",
            return_value=("Global deposit cap reached", evidence),
        ),
    ):
        stop_batch = runner._process_attempt(attempt, MagicMock())

    # 3. Verify the combined outcome is persisted with its evidence.
    assert stop_batch is False
    record_result.assert_called_once_with(
        attempt,
        ANY,
        result="success-deposit-closed",
        detail="Global deposit cap reached",
        outcome_data=evidence,
    )


def test_simulated_closed_deposit_open_json_gets_directional_result() -> None:
    """The runner distinguishes a live closure missing from vault JSON.

    1. Prepare an automatic deposit attempt whose JSON reports deposits open.
    2. Process typed live closure evidence through GuardV0 validation.
    3. Verify the incorrect-reporting outcome includes both status observations.
    """
    # 1. Prepare an automatic deposit attempt whose JSON reports deposits open.
    runner = object.__new__(VaultTestBatchRunner)
    runner.auto_simulated = True
    runner.deposit_asset = None
    runner.current_attempt = SimpleNamespace()
    runner.runtime = MagicMock()
    runner.amount = Decimal("1")
    runner.state = SimpleNamespace(
        portfolio=SimpleNamespace(get_all_trades=lambda: []),
    )
    attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(
                deposit_status=VaultDepositStatus.open,
                deposit_closed_reason=None,
            ),
        ),
        executable_vault=MagicMock(),
        pair=MagicMock(),
        spec=MagicMock(),
    )
    evidence = {
        "deposit_executed": False,
        "guard_validation": {"mode": "closed_deposit_guard_v0"},
    }

    # 2. Mock routing helpers so this test isolates directional result persistence.
    with (
        patch.object(VaultTestBatchRunner, "_choose_operation", return_value="deposit"),
        patch.object(VaultTestBatchRunner, "_record_terminal_result") as record_result,
        patch.object(
            runner_module,
            "get_incorrect_whitelisting_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "get_whitelisting_needed_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "validate_simulated_closed_deposit",
            return_value=("Global deposit cap reached", evidence),
        ),
    ):
        stop_batch = runner._process_attempt(attempt, MagicMock())

    # 3. Verify the incorrect-reporting outcome includes both status observations.
    assert stop_batch is False
    record_result.assert_called_once_with(
        attempt,
        ANY,
        result="success-deposit-closed-incorrectly-reported-open",
        detail="Global deposit cap reached",
        outcome_data={
            **evidence,
            "vault_json_deposit_status": "open",
            "vault_json_deposit_closed_reason": None,
            "onchain_deposit_status": "closed",
        },
    )


def test_simulated_open_deposit_closed_json_is_executed_and_persisted() -> None:
    """The runner tests an onchain-open deposit despite stale closed vault JSON.

    1. Prepare an automatic deposit whose JSON reports a closure.
    2. Accept the live preflight and execute the simulated deposit.
    3. Verify the successful interaction receives the directional mismatch label.
    """
    # 1. Prepare an automatic deposit whose JSON reports a closure.
    runner = object.__new__(VaultTestBatchRunner)
    runner.auto_simulated = True
    runner.deposit_asset = None
    runner.current_attempt = SimpleNamespace()
    runner.runtime = MagicMock()
    runner.amount = Decimal("1")
    runner.state = SimpleNamespace(
        portfolio=SimpleNamespace(
            get_all_trades=lambda: [],
            get_default_reserve_position=lambda: SimpleNamespace(get_value=lambda: 1),
        ),
    )
    pair = MagicMock()
    pair.is_async_vault.return_value = False
    pricing_model = MagicMock()
    pricing_model.can_deposit.return_value = False
    attempt = SimpleNamespace(
        vault=SimpleNamespace(
            metadata=SimpleNamespace(
                deposit_status=VaultDepositStatus.closed,
                deposit_closed_reason="Scanner reported deposits closed",
            ),
        ),
        executable_vault=MagicMock(),
        pair=pair,
        pricing_model=pricing_model,
        spec=MagicMock(),
        previous=None,
        bridge_position=None,
    )

    # 2. Mock execution dependencies so this test isolates stale-JSON gate bypass.
    with (
        patch.object(VaultTestBatchRunner, "_choose_operation", return_value="deposit"),
        patch.object(VaultTestBatchRunner, "_execute_simulated") as execute_simulated,
        patch.object(VaultTestBatchRunner, "_append_result"),
        patch.object(
            runner_module,
            "get_adapter_unsupported_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "get_incorrect_whitelisting_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "get_whitelisting_needed_detail",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "validate_simulated_closed_deposit",
            return_value=None,
        ),
        patch.object(
            runner_module,
            "resolve_redemption_available",
            return_value=True,
        ),
    ):
        stop_batch = runner._process_attempt(attempt, MagicMock())

    # 3. Verify the successful interaction receives the directional mismatch label.
    assert stop_batch is False
    pricing_model.can_deposit.assert_not_called()
    execute_simulated.assert_called_once_with(
        attempt,
        "deposit",
        True,
        ANY,
        simulated_success=ANY,
    )
    simulated_success = execute_simulated.call_args.kwargs["simulated_success"]
    assert simulated_success == SimulatedSuccessOutcome(
        result="simulated-success-deposit-open-incorrectly-reported-closed",
        detail=(
            "Vault JSON reports deposits closed, but simulated "
            "onchain preflight accepted the deposit"
        ),
        outcome_data={
            "vault_json_deposit_status": "closed",
            "vault_json_deposit_closed_reason": "Scanner reported deposits closed",
            "onchain_deposit_status": "open",
        },
    )


def test_decoded_vault_errors_map_to_typed_results() -> None:
    """eth-defi #1374 decoded custom errors become distinct typed results.

    1. Map a missing deposit whitelist to whitelisting-needed.
    2. Map each redemption custom error to its stable current-state result.
    3. Map a below-minimum deposit refusal to below_minimum.
    """
    # 1. Map a missing deposit whitelist to whitelisting-needed.
    whitelist_error = WhitelistingRequired(
        "Depositor not whitelisted on chain 1 vault 0xabc for 0xdef",
        protocol="lagoon",
        direction="deposit",
        phase="preflight",
    )
    result, _detail, outcome_data = normalise_vault_flow_failure(whitelist_error)
    assert result == "whitelisting-needed"
    assert outcome_data["direction"] == "deposit"

    # 2. Map each redemption custom error to its stable current-state result.
    cases = {
        "EndOfEpoch": "redemption_window_closed",
        "WithdrawalsArePaused": "redemption_paused",
        "WithdrawalPending": "redemption_capacity_limited",
        "ExceededMaxRedeem": "redemption_capacity_limited",
        "AddressNotAllowed": "whitelisting-needed",
    }
    for decoded_error, expected in cases.items():
        error = VaultFlowUnavailable(
            f"redeem refused: {decoded_error}",
            protocol="gains",
            direction="redeem",
            phase="preflight",
            decoded_error=decoded_error,
        )
        result, _detail, outcome_data = normalise_vault_flow_failure(error)
        assert result == expected, f"{decoded_error} -> {result}"
        assert outcome_data["decoded_error"] == decoded_error

    # 3. Map a below-minimum deposit refusal to below_minimum.
    minimum_error = VaultFlowUnavailable(
        "deposit below minimum",
        protocol="accountable",
        direction="deposit",
        phase="preflight",
        decoded_error="InsufficientAmount",
        minimum_raw_amount=1000,
    )
    result, _detail, outcome_data = normalise_vault_flow_failure(minimum_error)
    assert result == "below_minimum"
    assert outcome_data["minimum_raw_amount"] == "1000"

    for preflight_result in (
        "redemption_closed",
        "redemption_liquidity_unavailable",
        "redemption_zero_payout",
        "redemption_not_yet_matured",
    ):
        error = VaultFlowUnavailable(
            "typed live redemption state",
            protocol="test",
            direction="redeem",
            phase="preflight",
            preflight_result=preflight_result,
        )
        result, _detail, _outcome_data = normalise_vault_flow_failure(error)
        assert result == preflight_result


def test_preflight_result_is_copied_verbatim() -> None:
    """The authoritative eth-defi preflight_result maps regardless of decoded_error.

    1. Simulate an adapter that sets preflight_result plus a decoded_error name
       the executor heuristic does not enumerate (e.g. InsufficientShares).
    2. Verify the executor copies preflight_result verbatim as the result.
    3. Verify an unrecognised preflight_result falls back to the decoded_error map.
    """

    # 1. Adapter sets preflight_result + a decoded_error name outside the heuristic.
    error = VaultFlowUnavailable(
        "redeem refused",
        protocol="ember",
        direction="redeem",
        phase="preflight",
    )
    error.preflight_result = "below_minimum"
    error.decoded_error = "InsufficientShares"  # not in the decoded-error heuristic

    # 2. Verify the executor copies preflight_result verbatim as the result.
    result, _detail, outcome_data = normalise_vault_flow_failure(error)
    assert result == "below_minimum"
    assert outcome_data["preflight_result"] == "below_minimum"
    assert outcome_data["decoded_error"] == "InsufficientShares"

    # 3. Verify an unrecognised preflight_result falls back to the decoded_error map.
    fallback = VaultFlowUnavailable(
        "redeem refused",
        protocol="gains",
        direction="redeem",
        phase="preflight",
        decoded_error="EndOfEpoch",
    )
    fallback.preflight_result = "not_a_known_result"
    result, _detail, _outcome_data = normalise_vault_flow_failure(fallback)
    assert result == "redemption_window_closed"


def test_live_capability_outranks_stale_pair_redemption_flag() -> None:
    """A stale pair snapshot must not suppress an adapter-supported redemption.

    ``TradingPairIdentifier.can_redeem()`` is a data-pipeline snapshot, so a
    stale ``False`` previously skipped the redemption leg entirely and reported
    a bare ``redemption_unavailable`` (Plutus Hedge), hiding the adapter's own
    typed async result.

    1. Resolve availability when the pair says no but the adapter says yes.
    2. Resolve availability when the adapter says no.
    3. Fall back to the pair flag when the adapter publishes no capability.
    """

    class _Pair:
        def __init__(self, flag: bool):
            self._flag = flag

        def can_redeem(self) -> bool:
            return self._flag

    # 1. Resolve availability when the pair says no but the adapter says yes.
    supported = SimpleNamespace(can_redeem=True, redemption_unsupported_reason=None)
    vault = SimpleNamespace(get_deposit_manager_capability=lambda: supported)
    assert resolve_redemption_available(_Pair(False), vault) is True

    # 2. Resolve availability when the adapter says no.
    refused = SimpleNamespace(
        can_redeem=False,
        redemption_unsupported_reason="vault_is_wind_down_only",
    )
    refused_vault = SimpleNamespace(get_deposit_manager_capability=lambda: refused)
    assert resolve_redemption_available(_Pair(True), refused_vault) is False

    # 3. Fall back to the pair flag when the adapter publishes no capability.
    assert resolve_redemption_available(_Pair(True), SimpleNamespace()) is True
    assert resolve_redemption_available(_Pair(False), SimpleNamespace()) is False


def test_redemption_unavailable_always_has_a_reason() -> None:
    """A skipped redemption must never be recorded without a reason.

    1. Use the adapter's published reason when it has one.
    2. State explicitly that no reason was published when it has none.
    """

    # 1. Use the adapter's published reason when it has one.
    capability = SimpleNamespace(
        can_redeem=False,
        redemption_unsupported_reason="vault_is_wind_down_only",
    )
    vault = SimpleNamespace(get_deposit_manager_capability=lambda: capability)
    detail = get_redemption_unavailable_detail(vault)
    assert "vault_is_wind_down_only" in detail

    # 2. State explicitly that no reason was published when it has none.
    fallback = get_redemption_unavailable_detail(SimpleNamespace())
    assert fallback
    assert "no reason" in fallback


def test_unknown_deposit_hook_fails_closed() -> None:
    """An unrecognised gate is not treated as permissionless.

    1. Create an adapter whose policy inspection is explicitly unknown.
    2. Classify it through the vault-test permission preflight.
    3. Confirm the result explains why simulation must fail closed.
    """
    # 1. Create an adapter whose policy inspection is explicitly unknown.
    vault = SimpleNamespace(
        is_whitelisted_deposit=MagicMock(
            side_effect=NotImplementedError("custom EVK hook")
        )
    )
    attempt = SimpleNamespace(executable_vault=vault)

    # 2. Classify it through the vault-test permission preflight.
    detail = get_unknown_deposit_permission_detail(attempt)

    # 3. Confirm the result explains why simulation must fail closed.
    assert detail is not None
    assert "custom EVK hook" in detail


def test_incompatible_deposit_asset_lists_supported_and_selected() -> None:
    """A multi-asset vault whitelist mismatch reports its own failure mode.

    1. Resolve a deposit asset for a multi-asset vault that excludes our asset.
    2. Verify the raised error names both the supported assets and our asset.
    3. Verify the reporting helper maps it to the incompatible_deposit_asset result.
    """

    # A minimal fake multi-asset manager whose whitelist excludes USDC.
    class _Token:
        def __init__(self, symbol: str, address: str):
            self.symbol = symbol
            self.address = address

    class _MultiAssetManager:
        def fetch_accepted_assets(self) -> list[_Token]:
            return [
                _Token("USDT", "0xdAC17F958D2ee523a2206206994597C13D831ec7"),
                _Token("PYUSD", "0x6c3ea9036406852006290770BEdFcAbA0e23A0e8"),
            ]

    # 1. Resolve a deposit asset for a multi-asset vault that excludes our asset.
    try:
        resolve_multi_asset_deposit_asset(_MultiAssetManager(), 1)
    except IncompatibleDepositAsset as error:
        raised = error
    else:
        raised = None

    # 2. Verify the raised error names both the supported assets and our asset.
    assert raised is not None
    message = str(raised)
    assert "USDT" in message and "PYUSD" in message
    assert raised.selected_asset is not None
    assert raised.selected_asset in message  # our attempted asset is shown

    # 3. Verify the reporting helper maps it to the incompatible_deposit_asset result.
    result, _detail, outcome_data = normalise_vault_flow_failure(raised)
    assert result == "incompatible_deposit_asset"
    assert outcome_data["selected_asset"] == raised.selected_asset
    assert {
        "symbol": "USDT",
        "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    } in (outcome_data["accepted_assets"])


def test_closed_deposit_probe_resolves_multi_asset_before_manager_preflight() -> None:
    """The early closure probe must preserve an unsupported deposit-asset result.

    1. Prepare a multi-asset manager which accepts USDT but not default USDC.
    2. Run the simulated closed-deposit probe for an Ethereum vault.
    3. Confirm asset incompatibility is raised before request construction.
    """
    # 1. Prepare a multi-asset manager which accepts USDT but not default USDC.
    accepted_token = SimpleNamespace(
        symbol="USDT",
        address="0xdAC17F958D2ee523a2206206994597C13D831ec7",
    )

    class _MultiAssetManager:
        def __init__(self) -> None:
            self.create_deposit_request = MagicMock()

        def fetch_accepted_assets(self) -> tuple[SimpleNamespace, ...]:
            return (accepted_token,)

    manager = _MultiAssetManager()
    executable_vault = MagicMock(
        denomination_token=MagicMock(convert_to_raw=MagicMock(return_value=1_001_000_000)),
    )
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = (
        "0x0000000000000000000000000000000000000001"
    )
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
    )

    # 2. Run the simulated closed-deposit probe for an Ethereum vault.
    with pytest.raises(IncompatibleDepositAsset) as exc_info:
        validate_simulated_closed_deposit(
            attempt,
            MagicMock(),
            Decimal("1001"),
        )

    # 3. Confirm asset incompatibility is raised before request construction.
    assert exc_info.value.selected_asset.lower() == (
        "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".lower()
    )
    assert exc_info.value.accepted_assets == [("USDT", accepted_token.address)]
    manager.create_deposit_request.assert_not_called()


def test_closed_deposit_probe_uses_selected_asset_decimals() -> None:
    """Convert a multi-asset probe amount with the selected token's decimals.

    1. Prepare an 18-decimal accepted token beside a 6-decimal denomination token.
    2. Run the probe with an explicit accepted-asset override.
    3. Confirm request construction receives the selected token's raw amount.
    """
    # 1. Prepare an 18-decimal accepted token beside a 6-decimal denomination token.
    accepted_token = SimpleNamespace(
        symbol="RLUSD",
        address="0x8292Bb45bf1Ee4d140127049757C2E0fF06317eD",
        convert_to_raw=MagicMock(return_value=1_001 * 10**18),
    )
    manager = SimpleNamespace(
        fetch_accepted_assets=MagicMock(return_value=(accepted_token,)),
        create_deposit_request=MagicMock(),
    )
    denomination_token = MagicMock(
        convert_to_raw=MagicMock(return_value=1_001 * 10**6),
    )
    executable_vault = MagicMock(denomination_token=denomination_token)
    executable_vault.get_deposit_manager.return_value = manager
    route = MagicMock()
    route.get_owner_address.return_value = "0x0000000000000000000000000000000000000001"
    attempt = SimpleNamespace(
        pricing_model=MagicMock(route=MagicMock(return_value=route)),
        pair=MagicMock(),
        spec=SimpleNamespace(chain_id=ChainId.ethereum.value),
        executable_vault=executable_vault,
        interventions=[],
    )
    runtime = MagicMock()
    runtime.deployment.primary_chain_id.value = ChainId.ethereum.value

    # 2. Run the probe with an explicit accepted-asset override.
    result = validate_simulated_closed_deposit(
        attempt,
        runtime,
        Decimal("1001"),
        deposit_asset=accepted_token.address,
    )

    # 3. Confirm request construction receives the selected token's raw amount.
    assert result is None
    manager.create_deposit_request.assert_called_once_with(
        owner=route.get_owner_address.return_value,
        raw_amount=1_001 * 10**18,
        check_enough_token=True,
        accepted_asset=accepted_token.address,
    )
    accepted_token.convert_to_raw.assert_called_once_with(Decimal("1001"))
    denomination_token.convert_to_raw.assert_not_called()


def test_vault_simulation_options_reach_lazy_configurator() -> None:
    """Vault-test options are retained before the vault route is materialised.

    1. Create a generic routing shell whose configurator cache is empty.
    2. Apply an accepted-asset override and Anvil intervention mode.
    3. Verify the lazy configurator retains both defaults for route creation.
    """
    # 1. Create a generic routing shell whose configurator cache is empty.
    configurator = SimpleNamespace(configs={})
    routing_model = SimpleNamespace(pair_configurator=configurator)

    # 2. Apply an accepted-asset override and Anvil intervention mode.
    apply_vault_simulation_options(
        routing_model,
        deposit_asset="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        simulate_redemption_with_liquidity=True,
    )

    # 3. Verify the lazy configurator retains both defaults for route creation.
    assert configurator.vault_deposit_asset_override == "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    assert configurator.vault_simulate_redemption_with_liquidity is True


def test_unsupported_vault_simulation_has_typed_report_outcome() -> None:
    """Unsupported manager settlement becomes an actionable terminal result.

    1. Create the typed eth-defi simulation capability failure.
    2. Normalise it through the vault-test reporting helper.
    3. Verify the report result remains distinct from execution failures.
    4. Verify the structured false-capability context is retained verbatim.
    """
    # 1. Create the typed eth-defi simulation capability failure.
    error = UnsupportedVaultSimulation(
        "AccountableDepositManager does not advertise Anvil settlement"
    )

    # 2. Normalise it through the vault-test reporting helper.
    result, detail, outcome_data = normalise_vault_flow_failure(error)

    # 3. Verify the report result remains distinct from execution failures.
    assert result == "simulation_unsupported_async"
    assert detail == "AccountableDepositManager does not advertise Anvil settlement"

    # 4. Verify the structured false-capability context is retained verbatim.
    contextual = UnsupportedVaultSimulation(
        "Ember cannot settle on this fork",
        unsupported_reason="operator_role_required",
        protocol="ember",
        vault_address="0x9be9294722f8AAd37b11a9792Be2C782182caFA2",
        direction="redeem",
    )
    _result, _detail, contextual_data = normalise_vault_flow_failure(contextual)
    assert contextual_data["unsupported_reason"] == "operator_role_required"
    assert contextual_data["protocol"] == "ember"
    assert contextual_data["direction"] == "redeem"


def test_failure_operation_uses_latest_attempt_vault_trade() -> None:
    """A redemption failure is not labelled as the outer deposit attempt.

    1. Model an old vault trade and new deposit/redemption trades.
    2. Add a newer non-vault bridge trade that must not own the operation.
    3. Verify the latest new vault trade identifies the redemption phase.

    Mocks isolate trade ordering from portfolio serialisation.
    """
    # 1. Model an old vault trade and new deposit/redemption trades.
    old_trade = MagicMock(trade_id=1)
    old_trade.is_vault.return_value = True
    deposit_trade = MagicMock(trade_id=2)
    deposit_trade.is_vault.return_value = True
    deposit_trade.is_buy.return_value = True
    redemption_trade = MagicMock(trade_id=3)
    redemption_trade.is_vault.return_value = True
    redemption_trade.is_buy.return_value = False

    # 2. Add a newer non-vault bridge trade that must not own the operation.
    bridge_trade = MagicMock(trade_id=4)
    bridge_trade.is_vault.return_value = False
    state = MagicMock()
    state.portfolio.get_all_trades.return_value = [
        old_trade,
        deposit_trade,
        redemption_trade,
        bridge_trade,
    ]

    # 3. Verify the latest new vault trade identifies the redemption phase.
    assert get_latest_attempt_vault_operation(state, {1}) == "redeem"


def test_bridge_proceeds_failure_has_typed_report_outcome() -> None:
    """Unavailable satellite redemption proceeds stay distinguishable in reports.

    1. Create the bridge reconciliation failure.
    2. Normalise it through the vault-test reporting helper.
    3. Verify the report records the actionable bridge outcome.
    """
    # 1. Create the bridge reconciliation failure.
    error = BridgeProceedsUnavailable("bridge_proceeds_unavailable: no new USDC")

    # 2. Normalise it through the vault-test reporting helper.
    result, detail, outcome_data = normalise_vault_flow_failure(error)

    # 3. Verify the report records the actionable bridge outcome.
    assert result == "bridge_proceeds_unavailable"
    assert detail == "bridge_proceeds_unavailable: no new USDC"
    assert outcome_data == {}


def test_adapter_capability_gap_is_reported_before_execution() -> None:
    """Unsupported eth-defi adapters become terminal diagnostics before routing.

    1. Model a vault whose manager explicitly declines redemption support.
    2. Check the batch preflight helper for a redemption operation.
    3. Verify the result contains the directional capability record.
    """
    # 1. Model a vault whose manager explicitly declines redemption support.
    capability = MagicMock()
    capability.can_deposit = True
    capability.can_redeem = False
    capability.as_dict.return_value = {"can_deposit": True, "can_redeem": False}
    vault = MagicMock()
    vault.get_deposit_manager_capability.return_value = capability
    attempt = MagicMock(vault=vault, executable_vault=vault)

    # 2. Check the batch preflight helper for a redemption operation.
    detail, outcome_data = get_adapter_unsupported_detail(attempt, "redeem")

    # 3. Verify the result contains the directional capability record.
    assert detail == "eth-defi adapter does not support redeem for this vault"
    assert outcome_data == {
        "operation": "redeem",
        "capability": {"can_deposit": True, "can_redeem": False},
    }


def test_unknown_adapter_capability_continues_to_execution() -> None:
    """Unknown adapter support does not become an unsupported result.

    1. Model a vault whose eth-defi adapter has no published capability.
    2. Run the directional preflight helper for a deposit.
    3. Verify the helper leaves the normal execution path available.
    """
    # 1. Model a vault whose eth-defi adapter has no published capability.
    vault = MagicMock()
    vault.get_deposit_manager_capability.return_value = None
    attempt = MagicMock(vault=vault, executable_vault=vault)

    # 2. Run the directional preflight helper for a deposit.
    result = get_adapter_unsupported_detail(attempt, "deposit")

    # 3. Verify the helper leaves the normal execution path available.
    assert result is None


def test_adapter_without_capability_api_continues_to_execution() -> None:
    """Adapters without the optional eth-defi capability API retain their flow.

    1. Model a legacy vault adapter without capability metadata.
    2. Run the directional preflight helper for a deposit.
    3. Verify the helper leaves the normal execution path available.
    """
    # 1. Model a legacy vault adapter without capability metadata.
    vault = object()
    attempt = MagicMock(vault=vault, executable_vault=vault)

    # 2. Run the directional preflight helper for a deposit.
    result = get_adapter_unsupported_detail(attempt, "deposit")

    # 3. Verify the helper leaves the normal execution path available.
    assert result is None


def test_adapter_without_deposit_closure_api_continues_to_execution() -> None:
    """Legacy adapters without a closure probe continue to their normal flow.

    1. Model a legacy vault adapter without deposit-closure metadata.
    2. Read its closure detail through the batch preflight helper.
    3. Verify the helper leaves the normal execution path available.
    """
    # 1. Model a legacy vault adapter without deposit-closure metadata.
    vault = object()
    attempt = MagicMock(vault=vault, executable_vault=vault)

    # 2. Read its closure detail through the batch preflight helper.
    result = get_deposit_closed_detail(attempt)

    # 3. Verify the helper leaves the normal execution path available.
    assert result is None


def test_vault_flow_analysis_conversion_preserves_trade_signs() -> None:
    """Shared receipt conversion uses the same signs for sync and async settlement.

    1. Model a successful manager analysis with reserve and share quantities.
    2. Convert it once as a deposit and once as a redemption.
    3. Verify reserve, share sign and price follow the executor trade contract.
    """
    # 1. Model a successful manager analysis with reserve and share quantities.
    analysis = MagicMock(
        denomination_amount=Decimal("10"),
        share_count=Decimal("8"),
    )

    # 2. Convert it once as a deposit and once as a redemption.
    deposit = convert_vault_flow_analysis(analysis, direction="deposit")
    redemption = convert_vault_flow_analysis(analysis, direction="redeem")

    # 3. Verify reserve, share sign and price follow the executor trade contract.
    assert deposit == (Decimal("10"), Decimal("8"), Decimal("1.25"))
    assert redemption == (Decimal("10"), Decimal("-8"), Decimal("1.25"))


def test_vault_redemption_amount_reconciles_only_small_share_shortfalls() -> None:
    """Vault redemption uses the available balance only for tolerable shortfalls.

    1. Reconcile exact, surplus and small-shortfall on-chain share balances.
    2. Verify a small shortfall caps the redemption while other balances retain the plan.
    3. Verify a material accounting shortfall remains an error.
    """
    # 1. Reconcile exact, surplus and small-shortfall on-chain share balances.
    exact = reconcile_vault_redemption_amount(
        Decimal("1"),
        Decimal("1"),
        epsilon=0.025,
    )
    surplus = reconcile_vault_redemption_amount(
        Decimal("1"),
        Decimal("1.01"),
        epsilon=0.025,
    )
    small_shortfall = reconcile_vault_redemption_amount(
        Decimal("1"),
        Decimal("0.99"),
        epsilon=0.025,
    )

    # 2. Verify a small shortfall caps the redemption while other balances retain the plan.
    assert exact == Decimal("1")
    assert surplus == Decimal("1")
    assert small_shortfall == Decimal("0.99")

    # 3. Verify a material accounting shortfall remains an error.
    with pytest.raises(AssertionError, match="large relative shortfall"):
        reconcile_vault_redemption_amount(
            Decimal("1"),
            Decimal("0.9"),
            epsilon=0.025,
        )


def test_async_vault_request_transaction_roles_ignore_selectors() -> None:
    """Async settlement selects persisted request roles instead of function names.

    1. Create an approval and two arbitrarily named manager request transactions.
    2. Select the manager request transaction set for settlement.
    3. Verify approval exclusion and request ordering do not depend on selectors.
    """
    # 1. Create an approval and two arbitrarily named manager request transactions.
    approval = BlockchainTransaction(
        function_selector="approve",
        other={
            "vault_transaction_role": "vault_approval",
        },
    )
    second_request = BlockchainTransaction(
        function_selector="makeWithdrawRequest",
        other={
            "vault_transaction_role": "vault_request",
            "vault_request_ordinal": 1,
        },
    )
    first_request = BlockchainTransaction(
        function_selector="redeemShares",
        other={
            "vault_transaction_role": "vault_request",
            "vault_request_ordinal": 0,
        },
    )
    trade = MagicMock(
        blockchain_transactions=[approval, second_request, first_request],
        other_data={},
    )

    # 2. Select the manager request transaction set for settlement.
    selected = get_async_vault_request_transactions(
        trade,
        request_function_count=2,
    )

    # 3. Verify approval exclusion and request ordering do not depend on selectors.
    assert selected == [first_request, second_request]


def test_async_vault_request_transaction_legacy_upgrade() -> None:
    """Legacy pending requests gain durable roles before their settlement retry.

    1. Create a legacy approval plus manager request transaction set without roles.
    2. Select the final manager calls using the rebuilt request function count.
    3. Verify role metadata and durable transaction indices are persisted.
    """
    # 1. Create a legacy approval plus manager request transaction set without roles.
    approval = BlockchainTransaction(function_selector="approve")
    first_request = BlockchainTransaction(function_selector="redeemShares")
    second_request = BlockchainTransaction(function_selector="makeWithdrawRequest")
    trade = MagicMock(
        trade_id=42,
        blockchain_transactions=[approval, first_request, second_request],
        other_data={},
    )

    # 2. Select the final manager calls using the rebuilt request function count.
    selected = get_async_vault_request_transactions(
        trade,
        request_function_count=2,
    )

    # 3. Verify role metadata and durable transaction indices are persisted.
    assert selected == [first_request, second_request]
    assert trade.other_data["vault_request_transaction_indices"] == [1, 2]
    assert first_request.other["vault_request_ordinal"] == 0
    assert second_request.other["vault_request_ordinal"] == 1


def test_async_settlement_parses_only_manager_request_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async settlement gives manager parsing only persisted request calls.

    1. Model an approval and an arbitrarily named async manager request.
    2. Settle the trade with successful receipts for both transactions.
    3. Verify the manager sees only the request hash and no global selector is used.
    """
    # 1. Model an approval and an arbitrarily named async manager request.
    approval_hash = HexBytes("0x" + "01" * 32)
    request_hash = HexBytes("0x" + "02" * 32)
    approval = BlockchainTransaction(
        tx_hash=approval_hash,
        function_selector="approve",
        other={"vault_transaction_role": "vault_approval"},
    )
    request = BlockchainTransaction(
        tx_hash=request_hash,
        function_selector="redeemShares",
        other={
            "vault_transaction_role": "vault_request",
            "vault_request_ordinal": 0,
        },
    )
    trade = MagicMock()
    trade.trade_id = 12
    trade.other_data = {
        "vault_async_flow": True,
        "vault_direction": "deposit",
        "vault_owner_address": "0x0000000000000000000000000000000000000001",
        "vault_raw_amount": "1000000",
    }
    trade.blockchain_transactions = [approval, request]
    trade.is_buy.return_value = True
    manager = MagicMock()
    manager_request = MagicMock(funcs=[MagicMock()])
    manager.create_deposit_request.return_value = manager_request
    manager.serialize_deposit_ticket.return_value = {"ticket": "request"}
    vault = MagicMock()
    vault.get_deposit_manager.return_value = manager
    routing = vault_routing.VaultRouting("0x0000000000000000000000000000000000000001")
    state = MagicMock()

    # 2. Settle the trade with successful receipts for both transactions.
    monkeypatch.setattr(vault_routing, "get_vault_for_pair", lambda *_, **__: vault)
    monkeypatch.setattr(vault_routing, "get_block_timestamp", lambda *_: None)
    routing.settle_trade(
        MagicMock(),
        state,
        trade,
        {
            approval_hash: {"status": 1, "blockNumber": 100},
            request_hash: {"status": 1, "blockNumber": 101},
        },
    )

    # 3. Verify the manager sees only the request hash and no global selector is used.
    manager_request.parse_deposit_transaction.assert_called_once_with([request_hash])
    state.mark_vault_settlement_pending.assert_called_once()


def test_reverted_synchronous_vault_trade_skips_receipt_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status-zero receipt is reported before a manager tries event parsing.

    1. Build a synchronous vault trade with a reverted receipt.
    2. Settle it with a manager whose analyser would fail if called.
    3. Verify normal trade failure reporting occurs without manager analysis.
    """
    # 1. Build a synchronous vault trade with a reverted receipt.
    routing = vault_routing.VaultRouting("0x0000000000000000000000000000000000000001")
    tx_hash = HexBytes("0x" + "01" * 32)
    swap_transaction = MagicMock(tx_hash=tx_hash)
    trade = MagicMock()
    trade.other_data = {}
    trade.is_buy.return_value = True
    vault = MagicMock()
    report_failure = MagicMock()
    receipt = {"status": 0, "blockNumber": 123}

    # 2. Settle it with a manager whose analyser would fail if called.
    monkeypatch.setattr(vault_routing, "get_vault_for_pair", lambda *_, **__: vault)
    monkeypatch.setattr(
        vault_routing,
        "get_swap_transactions",
        lambda _: swap_transaction,
    )
    monkeypatch.setattr(vault_routing, "get_block_timestamp", lambda *_: None)
    monkeypatch.setattr(vault_routing, "report_failure", report_failure)
    routing.settle_trade(
        MagicMock(),
        MagicMock(),
        trade,
        {tx_hash: receipt},
    )

    # 3. Verify normal trade failure reporting occurs without manager analysis.
    report_failure.assert_called_once()
    vault.get_deposit_manager.assert_not_called()


def test_vault_test_report_reads_the_latest_run_record() -> None:
    """The exported report must use the current run provenance.

    1. Store an initial vault-test run and a later unrelated state cycle.
    2. Export a report and verify the initial run remains discoverable.
    3. Replace the run record in the newest cycle and verify the report updates.
    """
    # 1. Store an initial vault-test run and a later unrelated state cycle.
    state = State()
    state.other_data.save(0, "vault_test_run", {"run_started_at": "first"})
    state.other_data.save(1, "unrelated", True)

    # 2. Export a report and verify the initial run remains discoverable.
    report = export_vault_test_report(state, [])
    assert report["run"] == {"run_started_at": "first"}

    # 3. Replace the run record in the newest cycle and verify the report updates.
    state.other_data.save(1, "vault_test_run", {"run_started_at": "second"})
    assert export_vault_test_report(state, [])["run"] == {"run_started_at": "second"}


def test_vault_test_state_refreshes_run_provenance(tmp_path: Path) -> None:
    """Reloading tester state must record the current command provenance.

    1. Write an existing tester state with stale run provenance.
    2. Load it through the command state helper with a new runtime provenance.
    3. Verify the in-memory and persisted state contain the new run record.
    """
    # 1. Write an existing tester state with stale run provenance.
    state_file = tmp_path / "vault-test-state.json"
    state = State()
    state.other_data.save(0, "vault_test_run", {"run_started_at": "old"})
    state.write_json_file(state_file)

    # 2. Load it through the command state helper with a new runtime provenance.
    runtime = MagicMock()
    runtime.get_provenance.return_value = {"run_started_at": "new"}
    loaded, _ = load_vault_test_state(
        state_file=state_file,
        state_name="vault-test",
        runtime=runtime,
    )

    # 3. Verify the in-memory and persisted state contain the new run record.
    assert loaded.other_data.load_latest("vault_test_run") == {"run_started_at": "new"}
    assert State.read_json_file(state_file).other_data.load_latest(
        "vault_test_run"
    ) == {"run_started_at": "new"}


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
        tui_module,
        "get_latest_vault_position",
        lambda state, vault_spec: position,
    )
    monkeypatch.setattr(
        tui_module,
        "get_vault_trade_position",
        lambda state, vault_spec, open_only=False: position,
    )
    monkeypatch.setattr(
        tui_module,
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
