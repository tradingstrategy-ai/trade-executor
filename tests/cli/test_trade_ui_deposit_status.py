"""Test trade-ui vault deposit status rendering."""

import datetime

from tradeexecutor.cli.trade_ui_tui import _format_deposits_open, _get_deposit_status
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind


def _make_vault_pair(deposit_closed_reason: str | None = None) -> TradingPairIdentifier:
    base = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="vUSDC",
        decimals=18,
    )
    quote = AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=base.address,
        exchange_address=base.address,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="vault",
        other_data={
            "vault_protocol": "erc4626",
            "deposit_closed_reason": deposit_closed_reason,
        },
    )


def test_vault_pair_static_deposit_status_uses_closed_reason() -> None:
    """Check ERC-4626 vault metadata can close deposits in the TUI fallback path.

    1. Create a non-Hyperliquid vault pair with a deposit closed reason.
    2. Ask the pair for its static deposit availability.
    3. Render the TUI deposit status without a pricing model result.
    4. Verify both paths report deposits as closed.
    """
    # 1. Create a non-Hyperliquid vault pair with a deposit closed reason.
    pair = _make_vault_pair("Max deposit cap reached")

    # 2. Ask the pair for its static deposit availability.
    can_deposit = pair.can_deposit()

    # 3. Render the TUI deposit status without a pricing model result.
    rendered = _format_deposits_open(pair)

    # 4. Verify both paths report deposits as closed.
    assert can_deposit is False
    assert rendered.plain == "No"


def test_trade_ui_deposit_status_uses_live_pricing_model() -> None:
    """Check trade-ui honours live pricing model deposit gates.

    1. Create a vault pair whose static metadata says deposits are open.
    2. Create a pricing model that reports deposits as closed.
    3. Resolve and render the deposit status through the TUI helpers.
    4. Verify the live closed status wins over static metadata.
    """

    class ClosedPricingModel:
        def can_deposit(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> bool:
            return False

    # 1. Create a vault pair whose static metadata says deposits are open.
    pair = _make_vault_pair()

    # 2. Create a pricing model that reports deposits as closed.
    pricing_model = ClosedPricingModel()

    # 3. Resolve and render the deposit status through the TUI helpers.
    status = _get_deposit_status(pricing_model, pair, datetime.datetime(2026, 6, 3))
    rendered = _format_deposits_open(pair, status)

    # 4. Verify the live closed status wins over static metadata.
    assert status is False
    assert rendered.plain == "No"


def test_trade_ui_deposit_status_shows_unknown_when_live_check_fails() -> None:
    """Check trade-ui does not claim deposits are open after a failed live check.

    1. Create a vault pair whose static metadata has no closed reason.
    2. Create a pricing model whose live deposit check fails.
    3. Resolve and render the deposit status through the TUI helpers.
    4. Verify the UI renders unknown instead of a false open status.
    """

    class FailingPricingModel:
        def can_deposit(self, ts: datetime.datetime, pair: TradingPairIdentifier) -> bool:
            raise RuntimeError("RPC unavailable")

    # 1. Create a vault pair whose static metadata has no closed reason.
    pair = _make_vault_pair()

    # 2. Create a pricing model whose live deposit check fails.
    pricing_model = FailingPricingModel()

    # 3. Resolve and render the deposit status through the TUI helpers.
    status = _get_deposit_status(pricing_model, pair, datetime.datetime(2026, 6, 3))
    rendered = _format_deposits_open(pair, status)

    # 4. Verify the UI renders unknown instead of a false open status.
    assert status is None
    assert rendered.plain == "?"
