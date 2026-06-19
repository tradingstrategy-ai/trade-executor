"""Test the trade-ui TUI dialog submit behaviour.

Verifies that:
1. "Sell all" mode bypasses amount validation and dismisses immediately.
2. "Buy only" mode validates the amount input before dismissing.
3. Cancel button dismisses with None.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, RadioSet

from tradeexecutor.cli.trade_ui_tui import TradeDialog


class TradeDialogHarness(App):
    """Minimal Textual app for exercising TradeDialog with a real pilot."""

    def __init__(self, dialog: TradeDialog):
        super().__init__()
        self.dialog = dialog
        self.dismissed = False
        self.dismissed_with: tuple[str, Decimal] | None = None

    def compose(self) -> ComposeResult:
        yield from ()

    def on_mount(self) -> None:
        self.push_screen(self.dialog, self.on_dialog_dismissed)

    def on_dialog_dismissed(self, result: tuple[str, Decimal] | None) -> None:
        self.dismissed = True
        self.dismissed_with = result


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


@pytest.mark.anyio
async def test_sell_all_selection_works_through_textual_radio_event():
    """Sell all can be selected through the real Textual radio event path.

    This covers the production failure where arrow-key navigation highlighted
    ``Sell all`` without changing the actual pressed radio value.

    1. Mount the TradeDialog in a real Textual test app.
    2. Use keyboard navigation to select the Sell all radio button.
    3. Execute the dialog and verify close_all is returned.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="open",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog in a real Textual test app.
        await pilot.pause()
        radio_set = dialog.query_one("#mode-radio", RadioSet)
        assert app.focused is radio_set

        # 2. Use keyboard navigation to select the Sell all radio button.
        await pilot.press("down", "down")
        await pilot.pause()
        assert radio_set.pressed_index == 2
        amount_input = dialog.query_one("#amount-input", Input)
        assert amount_input.disabled is True

        # 3. Execute the dialog and verify close_all is returned.
        dialog.query_one("#execute-btn", Button).press()
        await pilot.pause()

    assert app.dismissed_with == ("close_all", Decimal("3.000083"))


@pytest.mark.anyio
async def test_close_mode_stays_selected_after_dialog_mount():
    """Close mode remains the selected radio button when the dialog opens.

    1. Mount the TradeDialog with close as the default mode.
    2. Verify Sell all is still the pressed radio button.
    3. Verify the ignored amount input starts disabled.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="close",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog with close as the default mode.
        await pilot.pause()
        radio_set = dialog.query_one("#mode-radio", RadioSet)
        amount_input = dialog.query_one("#amount-input", Input)

        # 2. Verify Sell all is still the pressed radio button.
        assert radio_set.pressed_index == 2

        # 3. Verify the ignored amount input starts disabled.
        assert amount_input.disabled is True


@pytest.mark.anyio
async def test_close_mode_enter_keeps_sell_all_selected_after_mount():
    """Enter keeps the default Sell all selection when the dialog opens.

    1. Mount the TradeDialog with close as the default mode.
    2. Press Enter immediately on the focused radio set.
    3. Verify Sell all remains selected.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="close",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog with close as the default mode.
        await pilot.pause()
        radio_set = dialog.query_one("#mode-radio", RadioSet)
        assert app.focused is radio_set

        # 2. Press Enter immediately on the focused radio set.
        await pilot.press("enter")
        await pilot.pause()

        # 3. Verify Sell all remains selected.
        assert radio_set.pressed_index == 2


@pytest.mark.anyio
async def test_trade_mode_keyboard_navigation_does_not_wrap():
    """Trade mode keyboard navigation stops at the first and last modes.

    1. Mount the TradeDialog in close mode.
    2. Press Down/Right and verify Sell all remains selected.
    3. Move to the first mode and verify Up/Left keep it selected.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="close",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog in close mode.
        await pilot.pause()
        radio_set = dialog.query_one("#mode-radio", RadioSet)

        # 2. Press Down/Right and verify Sell all remains selected.
        await pilot.press("down", "right")
        await pilot.pause()
        assert radio_set.pressed_index == 2

        # 3. Move to the first mode and verify Up/Left keep it selected.
        await pilot.press("up", "up", "left")
        await pilot.pause()
        assert radio_set.pressed_index == 0


@pytest.mark.anyio
async def test_trade_dialog_buttons_are_visible_and_keyboard_reachable():
    """Trade dialog has visible OK and Cancel buttons reachable by Tab.

    1. Mount the TradeDialog in a real Textual test app.
    2. Verify Cancel and OK buttons are rendered and focusable.
    3. Navigate with Tab until both buttons receive focus.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="open",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog in a real Textual test app.
        await pilot.pause()
        cancel_button = dialog.query_one("#cancel-btn", Button)
        ok_button = dialog.query_one("#execute-btn", Button)

        # 2. Verify Cancel and OK buttons are rendered and focusable.
        assert str(cancel_button.label) == "Cancel"
        assert str(ok_button.label) == "OK"
        assert cancel_button.display is True
        assert ok_button.display is True
        assert cancel_button.can_focus is True
        assert ok_button.can_focus is True
        assert 0 <= cancel_button.region.y < app.size.height
        assert 0 <= ok_button.region.y < app.size.height

        # 3. Navigate with Tab until both buttons receive focus.
        seen_focus = set()
        for _ in range(8):
            focused = app.focused
            if focused is cancel_button:
                seen_focus.add("cancel")
            if focused is ok_button:
                seen_focus.add("ok")
            await pilot.press("tab")
            await pilot.pause()

        assert seen_focus == {"cancel", "ok"}


@pytest.mark.anyio
async def test_trade_dialog_cancel_button_dismisses_without_result():
    """Cancel button dismisses the trade dialog without choosing a trade.

    1. Mount the TradeDialog in a real Textual test app.
    2. Press the Cancel button.
    3. Verify the dismiss callback receives ``None``.
    """
    dialog = TradeDialog(
        pair_symbol="Ostium Liquidity Pool Vault",
        reserve_symbol="USDC",
        default_mode="open",
        position_value=Decimal("3.000083"),
    )
    app = TradeDialogHarness(dialog)

    async with app.run_test() as pilot:
        # 1. Mount the TradeDialog in a real Textual test app.
        await pilot.pause()

        # 2. Press the Cancel button.
        dialog.query_one("#cancel-btn", Button).press()
        await pilot.pause()

    # 3. Verify the dismiss callback receives ``None``.
    assert app.dismissed is True
    assert app.dismissed_with is None


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
