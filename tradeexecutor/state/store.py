"""State serialisation to disk and JavaScript clients."""
import abc
import datetime
import enum
import json
import os
import shutil
import tempfile
from pathlib import Path
import logging
from pprint import pprint
from typing import Union, Optional, Callable

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
    def create(self, name: str) -> State:
        """Create a new state storage.

        :param name:
            Name of the strategy this State belongs to
        """
        state = State()
        state.name = name
        return state


class JSONFileStore(StateStore):
    """Store the state of the executor as a JSON file.

    - Read by strategy on a startup

    - Read by webhook when asked over the API
    """

    def __init__(self, path: Union[Path, str], on_save: Callable=None):
        """

        :param path:
            Path to the JSON file

        :param on_save:
            Save hook. Used by `RunState.read_only_state_copy`
        """
        assert path
        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.on_save = on_save

    def __repr__(self):
        path = os.path.abspath(self.path)
        return f"<JSON file at {path}>"

    def is_pristine(self) -> bool:
        return not self.path.exists()

    def load(self) -> State:
        logger.info("Loaded state from %s", self.path)
        return State.read_json_file(self.path)

    def sync(self, state: State):
        """Write new JSON state dump using Linux atomic filereplacement."""
        dirname, basename = os.path.split(self.path)
        # Prepare for an atomic replacement
        temp = tempfile.NamedTemporaryFile(mode='wt', delete=False, dir=dirname)
        with open(temp.name, "wt") as out:

            state.last_updated_at = datetime.datetime.utcnow()

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
            written = len(txt)
            logger.info(f"Saved state to %s, total {written:,} chars", self.path)
        temp.close()
        shutil.move(temp.name, self.path)

        if self.on_save:
            self.on_save(state)

    def create(self, name: str) -> State:
        logger.info("Created new state for the strategy %s at %s", name, os.path.realpath(self.path))
        return super().create(name)


class SimulateStore(JSONFileStore):
    """Store backend used in trade simulations.

    - Never persist this store, as the generated txs and their hashes exist
      only in the local simulated chain memory
    """

    def sync(self, state: State):
        """No-op - never write a file."""

        if self.on_save:
            self.on_save(state)


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
        state.last_updated_at = datetime.datetime.utcnow()

    def create(self) -> State:
        raise NotImplementedError("This should not be called for NoneStore.\n"
                                  "Backtest have explicit state set for them at the start that should not be cleared.")

