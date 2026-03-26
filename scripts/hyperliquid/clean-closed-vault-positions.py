"""Clean up Hyperliquid closed vault positions with stranded Safe balances.

Use the console command.

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

from tradeexecutor.ethereum.vault.hyperliquid_cleanup import \
    run_hyperliquid_cleanup_from_environment


def main() -> None:
    """Run the Hyperliquid clean-up flow from environment variables."""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
