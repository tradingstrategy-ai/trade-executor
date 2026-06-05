"""Test trade-ui vault deposit status for xchain-master-vault ERC-4626 pairs.

Before this fix, non-Hyperliquid vault pairs always showed Deposits: Yes
because ``TradingPairIdentifier.can_deposit()`` only checked
``deposit_closed_reason`` for Hyperliquid vaults, and the TUI never
consulted the live pricing model.
"""

import datetime

from tradeexecutor.cli.trade_ui_tui import (
    _format_deposits_open,
    _get_deposit_status,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)


def _make_erc4626_vault_pair(
    *,
    chain_id: int = 8453,
    vault_address: str = "0x0000000000000000000000000000000000000001",
    vault_name: str = "Test USDC Vault",
    protocol_slug: str = "lagoon-finance",
    deposit_closed_reason: str | None = None,
    internal_id: int = 1,
) -> TradingPairIdentifier:
    """Create a vault pair resembling xchain-master-vault ERC-4626 vaults."""
    base = AssetIdentifier(
        chain_id=chain_id,
        address=vault_address,
        token_symbol="vUSDC",
        decimals=18,
    )
    quote = AssetIdentifier(
        chain_id=chain_id,
        address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=vault_address,
        exchange_address="0x0000000000000000000000000000000000000000",
        internal_id=internal_id,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name=protocol_slug,
        other_data={
            "vault_protocol": "erc4626",
            "vault_name": vault_name,
            "deposit_closed_reason": deposit_closed_reason,
        },
    )


def test_tui_erc4626_deposit_status_respects_closed_reason() -> None:
    """ERC-4626 vaults with deposit_closed_reason show "No" in the TUI.

    This is the regression test for the core bug: before the fix,
    ``can_deposit()`` always returned True for non-Hyperliquid vaults,
    so the TUI showed Deposits: Yes even for capped vaults.

    1. Create two ERC-4626 vault pairs mimicking the xchain-master-vault
       multi-chain universe: one open on Base, one closed on Ethereum.
    2. Resolve deposit status through the TUI helpers (static fallback path).
    3. Verify the open vault renders "Yes" and the closed vault renders "No".
    """
    # 1. Create two ERC-4626 vault pairs: one open, one closed.
    open_vault = _make_erc4626_vault_pair(
        chain_id=8453, vault_name="Open Base Vault",
        deposit_closed_reason=None, internal_id=1,
    )
    closed_vault = _make_erc4626_vault_pair(
        chain_id=1, vault_name="Capped ETH Vault",
        deposit_closed_reason="Max deposit cap reached", internal_id=2,
    )

    # 2. Resolve deposit status through the TUI helpers (static fallback path).
    open_status = _get_deposit_status(None, open_vault)
    closed_status = _get_deposit_status(None, closed_vault)

    # 3. Verify the open vault renders "Yes" and the closed vault renders "No".
    assert open_status is True
    assert closed_status is False
    assert _format_deposits_open(open_vault, open_status).plain == "Yes"
    assert _format_deposits_open(closed_vault, closed_status).plain == "No"
