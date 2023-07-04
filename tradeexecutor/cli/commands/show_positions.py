"""show-positions command.

"""

from pathlib import Path
from typing import Optional

from IPython.core.display_functions import display
from tabulate import tabulate

from .app import app
from ..bootstrap import prepare_executor_id, create_state_store
from ...analysis.position import display_positions
from ...state.state import State
from . import shared_options


@app.command()
def show_positions(
    id: str = shared_options.id,
    state_file: Optional[Path] = shared_options.state_file,
):
    """Display trading positions from a state file.

    - Dumps all open and historical positions from the state file
      for debug inspection

    - This command does not read any live chain state, but merely
      dumps the existing state file positions to the console.
    """

    if not state_file:
        assert id, "Executor id must be given if not absolute state file path is given"
        state_file = f"state/{id}.json"

    store = create_state_store(Path(state_file))
    assert not store.is_pristine(), f"State file does not exists: {state_file}"

    state = State.read_json_file(state_file)

    print("Open positions")
    df = display_positions(state.portfolio.open_positions.values())
    # https://pypi.org/project/tabulate/
    # https://stackoverflow.com/a/31885295/315168
    print(tabulate(df, headers='keys', tablefmt='rounded_outline'))

    print()
    print("Frozen positions")
    df = display_positions(state.portfolio.frozen_positions.values())
    print(tabulate(df, headers='keys', tablefmt='rounded_outline'))

    print()
    print("Closed positions")
    df = display_positions(state.portfolio.closed_positions.values())
    print(tabulate(df, headers='keys', tablefmt='rounded_outline'))
