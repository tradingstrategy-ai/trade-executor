"""show-positions command.

"""
import enum
from pathlib import Path
from typing import Optional

from IPython.core.display_functions import display
from tabulate import tabulate
from typer import Option

from .app import app
from ..bootstrap import prepare_executor_id, create_state_store
from ...analysis.position import display_positions
from ...state.state import State
from . import shared_options


class PositionType(enum.Enum):
    """What output we want from show-positions command."""

    #: Show only open positions
    open = "open"

    #: Show only open positions
    open_and_frozen = "open_and_frozen"

    #: Show all positions
    all = "all"


@app.command()
def show_positions(
    id: str = shared_options.id,
    state_file: Optional[Path] = shared_options.state_file,
    strategy_file: Optional[Path] = shared_options.optional_strategy_file,
    position_type: PositionType = Option(PositionType.open_and_frozen.value, envvar="POSITION_TYPE", help="Which position types to display")
):
    """Display trading positions from a state file.

    - Dumps all open and historical positions from the state file
      for debug inspection

    - This command does not read any live chain state, but merely
      dumps the existing state file positions to the console.
    """

    if not state_file:
        # Guess id from the strategy file
        id = prepare_executor_id(id, strategy_file)
        assert id, "Executor id must be given if not absolute state file path is given"
        state_file = Path(f"state/{id}.json")

    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State file does not exists: {state_file}"

    state = State.read_json_file(state_file)

    print(f"Displaying positions and trades for state {state.name}")
    print(f"State last updated: {state.last_updated_at}")
    print(f"Position flags: F=frozen, R=contains repairs, UE=unexpected trades, SL=stop loss triggered")

    print("Open positions")
    df = display_positions(state.portfolio.open_positions.values())
    # https://pypi.org/project/tabulate/
    # https://stackoverflow.com/a/31885295/315168
    if len(df) > 0:
        print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
    else:
        print("No open positions")
    print()

    if position_type in (PositionType.all, PositionType.open_and_frozen):
        print("Frozen positions")
        df = display_positions(state.portfolio.frozen_positions.values())
        if len(df) > 0:
            print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
        else:
            print("No frozen positions")
        print()

    if position_type == PositionType.all:
        print("Closed positions")
        df = display_positions(state.portfolio.closed_positions.values())
        print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
