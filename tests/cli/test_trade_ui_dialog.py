"""Test the trade-ui TUI dialog submit behaviour.

Verifies that:
1. "Sell all" mode bypasses amount validation and dismisses immediately.
2. "Buy only" mode validates the amount input before dismissing.
3. Cancel button dismisses with None.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from tradeexecutor.cli.trade_ui_tui import TradeDialog


def test_sell_all_bypasses_amount_validation():
    """Sell all dismisses immediately without reading the amount input.

    The backend always closes the full position, so the dialog should not
    block on amount validation when "Sell all" is selected.

    1. Create a TradeDialog configured for close mode.
    2. Simulate the widget state (radio index 2, disabled amount input).
    3. Call _submit() directly.
    4. Verify dismiss is called with ("close_all", position_value).
    """
    dialog = TradeDialog(
        pair_symbol="pmalt-USDC",
        reserve_symbol="USDC",
        default_mode="close",
        position_value=Decimal("7.128756"),
    )

    dismissed_with = []
    dialog.dismiss = lambda result=None: dismissed_with.append(result)

    # Mock the RadioSet to report index 2 (Sell all)
    mock_radio = MagicMock()
    mock_radio.pressed_index = 2
    dialog.query_one = lambda selector, *args: mock_radio if "radio" in selector else MagicMock()

    dialog._submit()

    assert len(dismissed_with) == 1
    trade_mode, amount = dismissed_with[0]
    assert trade_mode == "close_all"
    assert amount == Decimal("7.128756")


def test_sell_all_works_with_no_position_value():
    """Sell all falls back to Decimal("0") when position_value is None.

    1. Create a TradeDialog without position_value.
    2. Select "Sell all" mode.
    3. Call _submit() — should dismiss with Decimal("0").
    """
    dialog = TradeDialog(
        pair_symbol="pmalt-USDC",
        reserve_symbol="USDC",
        default_mode="close",
        position_value=None,
    )

    dismissed_with = []
    dialog.dismiss = lambda result=None: dismissed_with.append(result)

    mock_radio = MagicMock()
    mock_radio.pressed_index = 2
    dialog.query_one = lambda selector, *args: mock_radio if "radio" in selector else MagicMock()

    dialog._submit()

    assert len(dismissed_with) == 1
    trade_mode, amount = dismissed_with[0]
    assert trade_mode == "close_all"
    assert amount == Decimal("0")


def test_buy_mode_rejects_invalid_amount():
    """Buy mode shows an error when the amount is invalid.

    1. Create a TradeDialog in buy mode.
    2. Set the amount input to "0" (invalid).
    3. Call _submit() — should not dismiss.
    """
    dialog = TradeDialog(
        pair_symbol="pmalt-USDC",
        reserve_symbol="USDC",
        default_mode="open",
        min_amount=Decimal("7"),
    )

    dismissed_with = []
    dialog.dismiss = lambda result=None: dismissed_with.append(result)

    mock_radio = MagicMock()
    mock_radio.pressed_index = 1  # Buy only

    mock_input = MagicMock()
    mock_input.value = "0"

    mock_error_label = MagicMock()

    def query_one(selector, *args):
        if "radio" in selector:
            return mock_radio
        if "amount" in selector:
            return mock_input
        if "error" in selector:
            return mock_error_label
        return MagicMock()

    dialog.query_one = query_one

    dialog._submit()

    # Should NOT have dismissed — amount is invalid
    assert len(dismissed_with) == 0
    mock_error_label.update.assert_called_once()


def test_buy_mode_accepts_valid_amount():
    """Buy mode dismisses with the entered amount when valid.

    1. Create a TradeDialog in buy mode.
    2. Set a valid amount.
    3. Call _submit() — should dismiss with ("open", amount).
    """
    dialog = TradeDialog(
        pair_symbol="pmalt-USDC",
        reserve_symbol="USDC",
        default_mode="open",
    )

    dismissed_with = []
    dialog.dismiss = lambda result=None: dismissed_with.append(result)

    mock_radio = MagicMock()
    mock_radio.pressed_index = 1  # Buy only

    mock_input = MagicMock()
    mock_input.value = "10"

    def query_one(selector, *args):
        if "radio" in selector:
            return mock_radio
        if "amount" in selector:
            return mock_input
        return MagicMock()

    dialog.query_one = query_one

    dialog._submit()

    assert len(dismissed_with) == 1
    trade_mode, amount = dismissed_with[0]
    assert trade_mode == "open"
    assert amount == Decimal("10")


def test_buy_mode_rejects_below_minimum():
    """Buy mode shows an error when the amount is below the minimum.

    1. Create a TradeDialog with min_amount=7.
    2. Set amount to "5" (below minimum).
    3. Call _submit() — should show error, not dismiss.
    """
    dialog = TradeDialog(
        pair_symbol="pmalt-USDC",
        reserve_symbol="USDC",
        default_mode="open",
        min_amount=Decimal("7"),
    )

    dismissed_with = []
    dialog.dismiss = lambda result=None: dismissed_with.append(result)

    mock_radio = MagicMock()
    mock_radio.pressed_index = 1  # Buy only

    mock_input = MagicMock()
    mock_input.value = "5"

    mock_error_label = MagicMock()

    def query_one(selector, *args):
        if "radio" in selector:
            return mock_radio
        if "amount" in selector:
            return mock_input
        if "error" in selector:
            return mock_error_label
        return MagicMock()

    dialog.query_one = query_one

    dialog._submit()

    assert len(dismissed_with) == 0
    mock_error_label.update.assert_called_once()
    assert "Minimum" in mock_error_label.update.call_args[0][0]
