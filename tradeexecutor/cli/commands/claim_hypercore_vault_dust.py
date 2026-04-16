"""Claim untracked Hypercore vault dust back to reserves."""

from decimal import Decimal
from pathlib import Path

from typer import Option

from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id
from ...ethereum.vault.hypercore_dust_claim import (
    DEFAULT_MAX_CLAIM_USDC,
    run_hypercore_dust_claim,
)


@app.command()
@shared_options.with_json_rpc_options()
def claim_hypercore_vault_dust(
    id: str = shared_options.id,
    strategy_file: Path = shared_options.strategy_file,
    state_file: Path | None = shared_options.state_file,
    private_key: str | None = shared_options.private_key,
    vault_address: str | None = shared_options.vault_address,
    vault_adapter_address: str | None = shared_options.vault_adapter_address,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Path | None = shared_options.cache_path,
    log_level: str = shared_options.log_level,
    unit_testing: bool = shared_options.unit_testing,
    rpc_kwargs: dict | None = None,
    auto_approve: bool = Option(
        False,
        envvar="AUTO_APPROVE",
        help="Approve Hypercore vault dust claims without asking for confirmation.",
    ),
    max_claim_usdc: str = Option(
        str(DEFAULT_MAX_CLAIM_USDC),
        envvar="MAX_CLAIM_USDC",
        help="Maximum per-vault Hypercore dust claim amount in USDC.",
    ),
):
    """Claim live Hypercore vault dust that is not represented in state."""
    id = prepare_executor_id(id, strategy_file)
    if state_file is None:
        state_file = Path(f"state/{id}.json")

    assert private_key, (
        "PRIVATE_KEY is needed to broadcast Hypercore dust claim transactions"
    )
    assert vault_address, "VAULT_ADDRESS is needed to resolve the Lagoon vault"
    assert vault_adapter_address, (
        "VAULT_ADAPTER_ADDRESS is needed to resolve the Lagoon trading strategy module"
    )

    json_rpc_hyperliquid = rpc_kwargs.get("json_rpc_hyperliquid")
    network = "mainnet"
    if not json_rpc_hyperliquid:
        json_rpc_hyperliquid = rpc_kwargs.get("json_rpc_hyperliquid_testnet")
        network = "testnet"
    assert json_rpc_hyperliquid, (
        "JSON_RPC_HYPERLIQUID or JSON_RPC_HYPERLIQUID_TESTNET is needed"
    )

    max_claim_usdc_decimal = Decimal(max_claim_usdc)

    run_hypercore_dust_claim(
        state_file=state_file,
        strategy_file=strategy_file,
        private_key=private_key,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        trading_strategy_api_key=trading_strategy_api_key or "",
        network=network,
        auto_approve=auto_approve,
        max_claim_usdc=max_claim_usdc_decimal,
        cache_path=cache_path,
        unit_testing=unit_testing,
        log_level=log_level or "info",
    )
