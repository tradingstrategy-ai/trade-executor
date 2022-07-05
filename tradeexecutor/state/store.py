"""State serialisation to disk and JavaScript clients."""
import abc
import enum
import os
import shutil
import tempfile
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
        """Write new JSON state dump using Linux atomic file replacement."""
        dirname, basename = os.path.split(self.path)
        temp = tempfile.NamedTemporaryFile(mode='wt', delete=False, dir=dirname)
        with open(temp.name, "wt") as out:
            txt = state.to_json()
            out.write(txt)
            logger.info("Saved state to %s, total %d chars", self.path, len(txt))
        temp.close()
        shutil.move(temp.name, self.path)

    def create(self) -> State:
        logger.info("Created new state for %s", self.path)
        return State()


class NoneStore(StateStore):
    """Store that is not persistent.

    Used in unit tests. Seed with initial state.
    """

    def __init__(self, state: State):
        self.created = False
        self.state = state

    def is_empty(self) -> bool:
        return False

    def load(self) -> State:
        return self.state

    def sync(self, state: State):
        """Do not persist anything."""
        pass

    def create(self) -> State:
        raise NotImplementedError()

