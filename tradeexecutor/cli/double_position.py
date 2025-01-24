"""Check for double positions."""
from tradeexecutor.state.state import State


def check_double_position(state: State, printer=print) -> bool:
    """Check that we do not have multiple positions open for the same trading pair.

    :return:
        True if there are double positions
    """
    # Warn about pairs appearing twice in the portfolio
    double_positions = False
    pairs = {p.pair for p in state.portfolio.get_open_and_frozen_positions()}
    for pair in pairs:
        positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair == pair]
        if len(positions) >= 2:
            printer(f"Warning: pair {pair} has multiple open positions: {len(positions)}")
            for p in positions:
                printer(p)
            double_positions = True

    return double_positions

