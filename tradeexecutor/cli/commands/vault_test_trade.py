"""Typer entry point for testing external vault deposits and redemptions."""

from decimal import Decimal
from pathlib import Path

from tabulate import tabulate
from typer import Option

from tradeexecutor.cli.bootstrap import prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.cli.vault_trade.core import parse_vault_ids
from tradeexecutor.cli.vault_trade.runner import VaultTestBatchRunner
from tradeexecutor.cli.vault_trade.state import write_vault_test_report
from tradeexecutor.cli.vault_trade.setup import (
    choose_manual_vault_action,
    create_vault_test_runtime,
    load_vault_test_data,
    load_vault_test_state,
    resolve_pending_real_actions,
)
from tradeexecutor.strategy.execution_model import AssetManagementMode


def _validate_vault_test_options(
    *,
    auto_simulated: bool,
    auto_real: bool,
    rerun: bool,
    settle_async_on_anvil: bool,
    asset_management_mode: AssetManagementMode | None,
) -> None:
    """Reject option combinations before downloading data or opening RPCs."""

    if auto_simulated and auto_real:
        raise RuntimeError("--auto-simulated and --auto-real are mutually exclusive")
    if rerun and not (auto_simulated or auto_real):
        raise RuntimeError("--rerun requires --auto-simulated or --auto-real")
    if settle_async_on_anvil and not auto_simulated:
        raise RuntimeError("--settle-async-on-anvil requires --auto-simulated")
    if asset_management_mode != AssetManagementMode.lagoon:
        raise RuntimeError("vault-test-trade requires ASSET_MANAGEMENT_MODE=lagoon")


@app.command()
@shared_options.with_json_rpc_options()
def vault_test_trade(
    id: str = Option(
        ...,
        "--id",
        envvar="EXECUTOR_ID",
        help="Dedicated Lagoon test executor id.",
    ),
    name: str | None = shared_options.name,
    trading_strategy_api_key: str | None = shared_options.trading_strategy_api_key,
    state_file: Path | None = shared_options.state_file,
    cache_path: Path | None = shared_options.cache_path,
    log_level: str | None = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str | None = shared_options.private_key,
    asset_management_mode: AssetManagementMode
    | None = shared_options.asset_management_mode,
    min_gas_balance: float | None = shared_options.min_gas_balance,
    max_slippage: float | None = shared_options.max_slippage,
    confirmation_block_count: int = shared_options.confirmation_block_count,
    confirmation_timeout: int = shared_options.confirmation_timeout,
    unit_testing: bool = shared_options.unit_testing,
    amount: float = Option(1.0, "--amount", envvar="AMOUNT", min=0.0000001),
    vault_id: str | None = Option(
        None,
        "--vault-id",
        envvar="VAULT_ID",
        help="Comma-separated chain-address vault ids.",
    ),
    auto_simulated: bool = Option(
        False,
        "--auto-simulated",
        help="Run fork-only simulations for VAULT_ID.",
    ),
    auto_real: bool = Option(
        False,
        "--auto-real",
        help="Run real deposits/redemptions for VAULT_ID.",
    ),
    rerun: bool = Option(
        False,
        "--rerun",
        help="Create a new attempt after a terminal result.",
    ),
    settle_async_on_anvil: bool = Option(
        False,
        "--settle-async-on-anvil",
        help="Force supported async vault settlements on simulated Anvil forks.",
    ),
    report_json: Path | None = Option(
        None,
        "--report-json",
        help="Write a machine-readable report containing the selected vault results.",
    ),
) -> None:
    """Test selected external vaults using one Lagoon executor.

    Automatic modes process the explicitly supplied ``VAULT_ID`` values in
    order and print one result table.  Without an automatic mode the command
    opens the Textual interface, where the operator selects one new deposit or
    one existing deposit to redeem.

    Unlike ``perform-test-trade``, this command has no strategy file.  The
    runner constructs a minimal universe from downloaded vault metadata for
    each individual action.
    """

    _validate_vault_test_options(
        auto_simulated=auto_simulated,
        auto_real=auto_real,
        rerun=rerun,
        settle_async_on_anvil=settle_async_on_anvil,
        asset_management_mode=asset_management_mode,
    )
    assert asset_management_mode == AssetManagementMode.lagoon

    # Resolve normal executor paths and download the complete metadata universe
    # before opening any chain resources.
    executor_id = prepare_executor_id(id, None)
    setup_logging(log_level=log_level)
    resolved_state_file = state_file or Path(f"state/{executor_id}.json")
    vault_specs = parse_vault_ids(vault_id) if auto_simulated or auto_real else []
    test_amount = Decimal(str(amount))
    slippage_tolerance = max_slippage or 0.005
    data = load_vault_test_data(
        executor_id=executor_id,
        cache_path=cache_path,
        trading_strategy_api_key=trading_strategy_api_key,
        unit_testing=unit_testing,
    )

    # Runtime creation chooses either the state-sibling Lagoon deployment or a
    # complete ephemeral multichain Lagoon deployment on Anvil forks.
    runtime = create_vault_test_runtime(
        executor_id=executor_id,
        state_file=resolved_state_file,
        rpc_kwargs=rpc_kwargs or {},
        private_key=private_key,
        asset_management_mode=asset_management_mode,
        min_gas_balance=min_gas_balance,
        max_slippage=slippage_tolerance,
        confirmation_block_count=confirmation_block_count,
        confirmation_timeout=confirmation_timeout,
        unit_testing=unit_testing,
        amount=test_amount,
        vault_specs=vault_specs,
        data=data,
        auto_simulated=auto_simulated,
    )

    try:
        state, store = load_vault_test_state(
            state_file=resolved_state_file,
            state_name=name or executor_id,
            runtime=runtime,
        )

        # Real invocations first advance CCTP attestations and asynchronous vault
        # tickets left pending by previous command runs.
        if not auto_simulated:
            resolve_pending_real_actions(runtime=runtime, state=state, store=store)

        manual_action = None
        if not (auto_simulated or auto_real):
            manual_action = choose_manual_vault_action(
                vault_universe=data.vault_universe,
                state=state,
            )
            if manual_action is None:
                return
            vault_specs = [manual_action.vault_spec]

        # The runner owns only the sequential lifecycle state machine.  Bootstrap
        # and final resource cleanup remain visible at this command boundary.
        runner = VaultTestBatchRunner(
            runtime=runtime,
            client=data.client,
            vault_universe=data.vault_universe,
            state=state,
            store=store,
            vault_specs=vault_specs,
            amount=test_amount,
            max_slippage=slippage_tolerance,
            auto_simulated=auto_simulated,
            rerun=rerun,
            settle_async_on_anvil=settle_async_on_anvil,
            manual_action=manual_action,
        )
        rows = runner.run()
        print(tabulate(rows, headers="keys", tablefmt="rounded_outline"))
        if report_json:
            write_vault_test_report(report_json, state, rows)
    finally:
        runtime.close()
