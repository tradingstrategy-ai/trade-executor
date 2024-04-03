"""Normalise folder locations across different Jupyter run-time environment."""
import inspect
import os
from pathlib import Path


def get_strategies_folder() -> Path:
    """Get 'strategies' folder in the source tree.

    - This is a massive hack, because some things in Jupyter are very  broken

    - Depends on how you run the notebook, there is no standard
    """

    # inside_ipython = any(frame for frame in inspect.stack() if frame.function == "start_ipython")

    current_dir = Path(os.getcwd())

    while current_dir.name != "trade-executor":
        if current_dir.parent != current_dir:
            current_dir = current_dir.parent
        else:
            raise RuntimeError(f"Could not find trade-executor repo root in {os.getcwd()}")

    return current_dir / "strategies"
