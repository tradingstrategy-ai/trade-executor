"""show-valuation command.

"""

from pathlib import Path
from typing import Optional

from tabulate import tabulate

from .app import app
from .shared_options import PositionType
from ..bootstrap import prepare_executor_id, create_state_store
from ...analysis.position import display_position_valuations
from ...state.state import State
from . import shared_options


@app.command()
def show_valuation(
    id: str = shared_options.id,
    state_file: Optional[Path] = shared_options.state_file,
    strategy_file: Optional[Path] = shared_options.optional_strategy_file,
    position_type: Optional[PositionType] = shared_options.position_type,
):
    """Show the valuation of the current portfolio.

    - Last valued
    - Value of individual positions
    - Warn if there are multiple positions for a same trading pair
    """

    if not state_file:
        # Guess id from the strategy file
        id = prepare_executor_id(id, strategy_file)
        assert id, "Executor id must be given if not absolute state file path is given"
        state_file = Path(f"state/{id}.json")

    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State file does not exists: {state_file}"

    state = State.read_json_file(state_file)

    print(f"Displaying valuation for state {state.name}")
    print(f"State last updated: {state.last_updated_at}")

    print(f"Portfolio last value: {state.portfolio.get_total_equity()} USD")
    print(f"Valued at: {state.sync.treasury.last_updated_at}")

    print("Open positions")
    df = display_position_valuations(state.portfolio.open_positions.values())
    # https://pypi.org/project/tabulate/
    # https://stackoverflow.com/a/31885295/315168
    if len(df) > 0:
        print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
    else:
        print("No open positions")
    print()

    if position_type in (PositionType.all, PositionType.open_and_frozen):
        print("Frozen positions")
        df = display_position_valuations(state.portfolio.frozen_positions.values())
        if len(df) > 0:
            print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
        else:
            print("No frozen positions")
        print()

    if position_type == PositionType.all:
        print("Closed positions")
        df = display_position_valuations(state.portfolio.closed_positions.values())
        print(tabulate(df, headers='keys', tablefmt='rounded_outline'))

    # Warn about pairs appearing twice in the portfolio
    pairs = {p.pair for p in state.portfolio.get_open_and_frozen_positions()}
    for pair in pairs:
        positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair == pair]
        if len(positions) >= 2:
            print(f"Warning: pair {pair} has multiple open positions: {len(positions)}")
            for p in positions:
                print(p)


