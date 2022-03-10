import abc
import enum
from pathlib import Path

from tradeexecutor.state.state import State


class StateStoreModel(enum.Enum):
    """How the algorithm execution state is stored."""

    file = "file"

    on_chain = "on_chain"


class StateStore(abc.ABC):
    """Backend to manage the trade exeuction persistent state."""

    @abc.abstractmethod
    def load(self) -> State:
        """Load the state from the storage."""

    @abc.abstractmethod
    def sync(self, state: State):
        """Save the state to the storage."""

    @abc.abstractmethod
    def create(self) -> State:
        """Create a new state storage."""


class JSONFileStore(StateStore):

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> State:
        with open(self.path, "rt") as inp:
            return State.from_json(inp)

    def sync(self, state: State):
        with open(self.path, "wt") as out:
            txt = state.to_json()
            out.write(txt)

    def create(self) -> State:
        return State()

