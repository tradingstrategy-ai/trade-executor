"""Clean up Hyperliquid closed vault positions with stranded Safe balances.

This script is only for the known Hypercore failure mode where:

1. A vault close was attempted
2. The withdrawal trade failed in state
3. In reality the capital already left the vault
4. The USDC is stranded on HyperCore ``perp`` and/or ``spot``
5. The stranded USDC now needs to be returned to the Safe on HyperEVM
6. The strategy state then needs repair and account correction

The implementation lives in :mod:`tradeexecutor.ethereum.vault.hyperliquid_cleanup`.
This wrapper stays intentionally minimal so the same functionality can also be
called from the trade-executor console.

Shell usage
-----------

.. code-block:: bash

    source .local-test.env && poetry run python scripts/hyperliquid/clean-closed-vault-positions.py

Console usage
-------------

.. code-block:: python

    from tradeexecutor.ethereum.vault.hyperliquid_cleanup import run_hyperliquid_cleanup_from_environment
    run_hyperliquid_cleanup_from_environment()
"""

from tradeexecutor.ethereum.vault.hyperliquid_cleanup import (
    run_hyperliquid_cleanup_from_environment,
)


def main() -> None:
    """Run the Hyperliquid clean-up flow from environment variables."""
    run_hyperliquid_cleanup_from_environment()


if __name__ == "__main__":
    main()
