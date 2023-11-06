"""Helper for default strategies distributed with the package itself.

- Used in testing, demos and example code
"""
import os
from pathlib import Path


def get_default_strategies_path() -> Path:
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "strategies"))
    return Path(raw_path)
