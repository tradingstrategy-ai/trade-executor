import logging
import os
import shutil
from pathlib import Path
from typing import Tuple, cast

from tradeexecutor.cli.bootstrap import create_state_store
from tradeexecutor.state.state import State
from tradeexecutor.state.store import StateStore, JSONFileStore

logger = logging.getLogger(__name__)


def backup_state(
        executor_id: str,
        state_file: str | Path | None,
        backup_suffix="reinit-backup"
) -> JSONFileStore:
    """Backup the current state file and then read it."""

    if not state_file:
        state_file = f"state/{executor_id}.json"

    state_file = Path(state_file)
    store = create_state_store(state_file)

    store = cast(JSONFileStore, store)

    assert not store.is_pristine(), f"State does not exists yet: {state_file}"

    # Make a backup
    # https://stackoverflow.com/a/47528275/315168
    backup_file = None
    for i in range(1, 20):  # Try 20 different iterateive backup filenames
        backup_file = state_file.with_suffix(f".{backup_suffix}-{i}.json")
        if os.path.exists(backup_file):
            continue

        shutil.copy(state_file, backup_file)
        break
    else:
        raise RuntimeError(f"Could not create backup {backup_file}")

    logger.info("Old state backed up as %s", backup_file)

    return store
