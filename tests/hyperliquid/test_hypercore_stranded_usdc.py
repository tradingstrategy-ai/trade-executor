"""Test P5: Stranded USDC marker on failed Hypercore deposits."""

from decimal import Decimal
from unittest.mock import MagicMock

from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting


def test_mark_stranded_usdc_stores_info():
    """_mark_stranded_usdc records recovery info in trade.other_data."""
    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC123"

    trade = MagicMock()
    trade.other_data = {}
    trade.trade_id = 42

    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=50_000_000,
        location="hypercore_spot",
    )

    info = trade.other_data["hypercore_stranded_usdc"]
    assert info["amount_raw"] == 50_000_000
    assert info["amount_human"] == "50"
    assert info["location"] == "hypercore_spot"
    assert info["safe_address"] == "0xABC123"
    assert "check-hypercore-user.py" in info["recovery"]
    trade.add_note.assert_called_once()
    assert "stranded" in trade.add_note.call_args[0][0].lower()


def test_mark_stranded_usdc_creates_other_data():
    """Works even if trade.other_data is None."""
    routing = MagicMock(spec=HypercoreVaultRouting)
    routing.safe_address = "0xABC123"

    trade = MagicMock()
    trade.other_data = None
    trade.trade_id = 1

    HypercoreVaultRouting._mark_stranded_usdc(
        routing,
        trade=trade,
        raw_amount=10_000_000,
        location="hypercore_spot_or_perp",
    )

    assert trade.other_data is not None
    assert "hypercore_stranded_usdc" in trade.other_data
