"""Unit tests for the standalone vault-test-trade command helpers."""

import json
import logging
from collections import defaultdict, deque
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from eth_defi.vault.base import VaultSpec
from eth_defi.vault.deposit_redeem import VaultFlowUnavailable
from hexbytes import HexBytes
from requests.exceptions import ReadTimeout
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Input
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.commands.lagoon_deploy_vault import (
    _write_state_sibling_deployment_artifact,
)
from tradeexecutor.cli.commands.vault_test_trade import _validate_vault_test_options
from tradeexecutor.cli.vault_trade import tui as tui_module
from tradeexecutor.cli.vault_trade import (
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
from tradeexecutor.cli.vault_trade.setup import load_vault_test_state
from tradeexecutor.cli.vault_trade.simulation import (
    SimulatedVaultAttemptTimeout,
    SimulatedVaultRuntime,
    is_simulated_infrastructure_failure,
    queue_simulated_infrastructure_retry,
    raise_simulated_vault_attempt_timeout,
    take_simulated_snapshots,
)
from tradeexecutor.cli.vault_trade.runner import (
    VaultAttemptContext,
    VaultTestBatchRunner,
    get_adapter_unsupported_detail,
    get_bridge_conflict,
    get_deposit_closed_detail,
    get_whitelisting_needed_detail,
    normalise_vault_flow_failure,
    should_leave_deposit_open,
)
from tradeexecutor.ethereum import web3config as web3config_module
from tradeexecutor.ethereum.vault import vault_routing
from tradeexecutor.ethereum.vault.vault_routing import convert_vault_flow_analysis
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
    attempt = MagicMock(pair=pair, pricing_model=pricing_model)

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
    assert report["results"][0]["attempt"]["outcome_data"] == {
        "executor_safe": "0x1"
    }


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
        simulation_module, "make_anvil_custom_rpc_request", snapshot
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

    # 2. Normalise the error through the vault-test reporting helper.
    result, detail, outcome_data = normalise_vault_flow_failure(error)

    # 3. Verify the result and JSON-safe capacity evidence are explicit.
    assert result == "redemption_capacity_limited"
    assert detail == "Only part of the redemption is currently available at <redacted-url>"
    assert outcome_data == {
        "protocol": "csigma",
        "direction": "redeem",
        "phase": "preflight",
        "requested_raw_amount": "200",
        "available_raw_amount": "100",
    }


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
    attempt = MagicMock(vault=vault)

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
    attempt = MagicMock(vault=vault)

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
    attempt = MagicMock(vault=object())

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
    attempt = MagicMock(vault=object())

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


def test_reverted_synchronous_vault_trade_skips_receipt_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status-zero receipt is reported before a manager tries event parsing.

    1. Build a synchronous vault trade with a reverted receipt.
    2. Settle it with a manager whose analyser would fail if called.
    3. Verify normal trade failure reporting occurs without manager analysis.
    """
    # 1. Build a synchronous vault trade with a reverted receipt.
    routing = vault_routing.VaultRouting(
        "0x0000000000000000000000000000000000000001"
    )
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
    assert export_vault_test_report(state, [])["run"] == {
        "run_started_at": "second"
    }


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
    assert loaded.other_data.load_latest("vault_test_run") == {
        "run_started_at": "new"
    }
    assert State.read_json_file(state_file).other_data.load_latest("vault_test_run") == {
        "run_started_at": "new"
    }


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
