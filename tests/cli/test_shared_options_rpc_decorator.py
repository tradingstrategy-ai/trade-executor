"""Tests for shared JSON-RPC Typer option injection."""

import inspect

import typer.main

from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.main import app


def _get_registered_callback(name: str):
    """Look up a registered Typer callback by its Python function name."""
    for command in app.registered_commands:
        if command.callback.__name__ == name:
            return command.callback
    raise AssertionError(f"Could not find registered command callback {name}")


def _get_typer_param_names(callback) -> list[str]:
    """Get Typer-registered parameter names in CLI order."""
    params, _convertors, _ctx_param_name = typer.main.get_params_convertors_ctx_param_name_from_function(callback)
    return [param.name for param in params]


def test_with_json_rpc_options_collects_rpc_kwargs_and_hides_placeholder() -> None:
    """Test the decorator collects filtered RPC kwargs and hides the placeholder.

    1. Decorate a small function with shared JSON-RPC options and chain filtering enabled.
    2. Call the decorated function with two RPC inputs and a selected chain name.
    3. Confirm the function receives filtered `rpc_kwargs` while the public signature exposes only CLI options.
    """
    captured: dict[str, object] = {}

    @shared_options.with_json_rpc_options(include_chain_name=True)
    def decorated_command(
        label: str = "demo",
        chain_name: str | None = None,
        rpc_kwargs: dict | None = None,
    ) -> dict[str, object]:
        # 1. Decorate a small function with shared JSON-RPC options and chain filtering enabled.
        captured["label"] = label
        captured["chain_name"] = chain_name
        captured["rpc_kwargs"] = rpc_kwargs
        return captured

    # 2. Call the decorated function with two RPC inputs and a selected chain name.
    result = decorated_command(
        label="ok",
        chain_name="base",
        json_rpc_base="https://base-rpc.invalid",
        json_rpc_arbitrum="https://arbitrum-rpc.invalid",
    )

    # 3. Confirm the function receives filtered `rpc_kwargs` while the public signature exposes only CLI options.
    assert result["label"] == "ok"
    assert result["chain_name"] == "base"
    assert result["rpc_kwargs"]["json_rpc_base"] == "https://base-rpc.invalid"
    assert result["rpc_kwargs"]["json_rpc_arbitrum"] is None

    signature = inspect.signature(decorated_command)
    signature_rpc_names = [name for name in signature.parameters if name.startswith("json_rpc_")]
    assert signature_rpc_names == list(shared_options.JSON_RPC_OPTION_NAMES)
    assert "rpc_kwargs" not in signature.parameters


def test_registered_commands_expose_expected_rpc_option_order() -> None:
    """Test representative commands share the same default RPC option ordering.

    1. Load representative command callbacks for the default, console, Lagoon, and deploy variants.
    2. Read the Typer-registered parameter names from each callback.
    3. Confirm each command exposes the same all-inclusive JSON-RPC order and does not leak `rpc_kwargs`.
    """
    expected_rpc_names = list(shared_options.JSON_RPC_OPTION_NAMES)
    command_names = [
        "check_universe",
        "start",
        "lagoon_first_deposit",
        "lagoon_deploy_vault",
    ]

    # 1. Load representative command callbacks for the default, console, Lagoon, and deploy variants.
    for command_name in command_names:
        callback = _get_registered_callback(command_name)

        # 2. Read the Typer-registered parameter names from each callback.
        typer_param_names = _get_typer_param_names(callback)
        actual_rpc_names = [name for name in typer_param_names if name.startswith("json_rpc_")]

        # 3. Confirm each command exposes the same all-inclusive JSON-RPC order and does not leak `rpc_kwargs`.
        assert actual_rpc_names == expected_rpc_names
        assert "rpc_kwargs" not in typer_param_names


def test_lagoon_deploy_vault_keeps_chain_name_after_rpc_options() -> None:
    """Test Lagoon deploy keeps `chain_name` as a separate CLI option after RPC inputs.

    1. Load the registered Lagoon deploy callback.
    2. Read its Typer parameter order.
    3. Confirm `chain_name` remains available and comes after the injected JSON-RPC options.
    """
    # 1. Load the registered Lagoon deploy callback.
    callback = _get_registered_callback("lagoon_deploy_vault")

    # 2. Read its Typer parameter order.
    typer_param_names = _get_typer_param_names(callback)
    chain_name_index = typer_param_names.index("chain_name")
    monad_index = typer_param_names.index("json_rpc_monad")

    # 3. Confirm `chain_name` remains available and comes after the injected JSON-RPC options.
    assert chain_name_index > monad_index
    assert "chain_name" in inspect.signature(callback).parameters
