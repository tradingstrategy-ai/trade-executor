"""State serialisation to disk and JavaScript clients."""
import abc
import enum
import json
import os
import shutil
import tempfile
from pathlib import Path
import logging
from pprint import pprint
from typing import Union, Optional

from dataclasses_json.core import _ExtendedEncoder

from tradeexecutor.state.state import State
from tradeexecutor.state.validator import validate_nested_state_dict

logger = logging.getLogger(__name__)


class StateStoreModel(enum.Enum):
    """How the algorithm execution state is stored."""

    file = "file"

    on_chain = "on_chain"


class StateStore(abc.ABC):
    """Backend to manage the trade exeuction persistent state."""

    @abc.abstractmethod
    def is_pristine(self) -> bool:
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

    def __init__(self, path: Union[Path, str]):
        assert path
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path

    def is_pristine(self) -> bool:
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

            # Insert special validation logic here to have
            # friendly error messages for the JSON serialisation errors
            data = state.to_dict(encode_json=False)
            validate_nested_state_dict(data)

            try:
                txt = json.dumps(data, cls=_ExtendedEncoder)
            except TypeError as e:
                # add some helpful debug info.
                # The usual cause of state serialisation failure is having
                # non-JSON objects in the state
                logger.error("State serialisation failed: %s", e)
                pprint(state.to_dict())
                raise

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

    def __init__(self, state: Optional[State]=None):
        self.created = False

        if not state:
            state = State()

        self.state = state

    def is_pristine(self) -> bool:
        return False

    def load(self) -> State:
        return self.state

    def sync(self, state: State):
        """Do not persist anything."""
        pass

    def create(self) -> State:
        raise NotImplementedError("This should not be called for NoneStore.\n"
                                  "Backtest have explicit state set for them at the start that should not be cleared.")

