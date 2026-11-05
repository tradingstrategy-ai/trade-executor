"""Clean up Hyperliquid closed vault positions with stranded Safe balances.

Run from the deployment's stopped Docker Compose service, following
``docs/docker.md``. Confirm the state mount and use the script shipped in the
deployed image. Compose supplies the environment; do not source local test
settings there. Confirm the repair backup and account validation before
restarting the executor. Inspect the proposed actions before confirming;
``AUTO_APPROVE`` bypasses that confirmation.

The console command can also run the helper below; its session already holds
the state lock.

Console usage
-------------

.. code-block:: python

    import os
    from pathlib import Path
    from tradeexecutor.ethereum.vault.hyperliquid_cleanup import run_hyperliquid_cleanup

    run_hyperliquid_cleanup(
        state_file=Path(store.path),
        strategy_file=Path(os.environ["STRATEGY_FILE"]),
        private_key=os.environ["PRIVATE_KEY"],
        json_rpc_hyperliquid=os.environ["JSON_RPC_HYPERLIQUID"],
        vault_address=vault.address,
        vault_adapter_address=vault.trading_strategy_module_address,
        trading_strategy_api_key=os.environ["TRADING_STRATEGY_API_KEY"],
        network=os.environ.get("NETWORK", "mainnet"),
        auto_approve=False,
    )
"""

import os
from pathlib import Path

import click

from tradeexecutor.cli.bootstrap import locked_state_file
from tradeexecutor.ethereum.vault.hyperliquid_cleanup import run_hyperliquid_cleanup


def main() -> None:
    """Run the Hyperliquid clean-up flow from environment variables."""
    with locked_state_file(os.environ["STATE_FILE"]) as state_file:
        run_hyperliquid_cleanup(
            state_file=state_file,
            strategy_file=Path(os.environ["STRATEGY_FILE"]),
            private_key=os.environ["PRIVATE_KEY"],
            json_rpc_hyperliquid=os.environ["JSON_RPC_HYPERLIQUID"],
            vault_address=os.environ["VAULT_ADDRESS"],
            vault_adapter_address=os.environ["VAULT_ADAPTER_ADDRESS"],
            trading_strategy_api_key=os.environ.get("TRADING_STRATEGY_API_KEY", ""),
            network=os.environ.get("NETWORK", "mainnet"),
            auto_approve=os.environ.get("AUTO_APPROVE", "").lower() in {"1", "true", "yes"},
            cache_path=Path(os.environ["CACHE_PATH"]) if os.environ.get("CACHE_PATH") else None,
            unit_testing=os.environ.get("UNIT_TESTING", "").lower() in {"1", "true", "yes"},
            log_level=os.environ.get("LOG_LEVEL", "info"),
        )


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code)
