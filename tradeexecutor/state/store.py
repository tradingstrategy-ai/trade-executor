import abc
import enum
from pathlib import Path
import logging

from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)


class StateStoreModel(enum.Enum):
    """How the algorithm execution state is stored."""

    file = "file"

    on_chain = "on_chain"


class StateStore(abc.ABC):
    """Backend to manage the trade exeuction persistent state."""

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """State has not been written yet."""

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
    """Store the state of the executor as a JSON file.

    - Read by strategy on a startup
    - Read by webhook when asked over the API
    """

    def __init__(self, path: Path):
        self.path = path

    def is_empty(self) -> bool:
        return not self.path.exists()

    def load(self) -> State:
        logger.info("Loaded state from %s", self.path)
        with open(self.path, "rt") as inp:
            return State.from_json(inp.read())

    def sync(self, state: State):
        with open(self.path, "wt") as out:
            txt = state.to_json()
            out.write(txt)
            logger.info("Saved state to %s, total %d chars", self.path, len(txt))

    def create(self) -> State:
        logger.info("Created new state for %s", self.path)
        return State()

