"""Check for double positions."""
from tradeexecutor.state.state import State


def check_double_position(state: State, printer=print, crash=False) -> bool:
    """Check that we do not have multiple positions open for the same trading pair.

    :param printer:
        Replace with logger.error() for live execution

    :param crash:
        Safety crash - do not allow continue beyond this point if we detect double positions.

    :return:
        True if there are double positions
    """
    # Warn about pairs appearing twice in the portfolio
    double_positions = False
    pairs = {p.pair for p in state.portfolio.get_open_and_frozen_positions()}
    for pair in pairs:

        positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair == pair]

        about_to_close = [p for p in positions if p.is_about_to_close()]
        not_about_to_close = [p for p in positions if not p.is_about_to_close()]

        if len(not_about_to_close) >= 2:
            printer(f"Warning: pair {pair} has multiple open positions: {len(positions)}")

            for p in positions:
                printer(f"Position {p}, quantity {p.get_quantity(planned=False)}, planned quantity: {p.get_quantity(planned=True)}")

            double_positions = True

            if crash:
                raise AssertionError(f"Double positions detected for pair {pair} - crashing for safety reasons.\nPositions: {positions}\nDiagnose what is causing the double position creation and manually clean up with close-position CLI command.\nSee logs for more details.")

    return double_positions

