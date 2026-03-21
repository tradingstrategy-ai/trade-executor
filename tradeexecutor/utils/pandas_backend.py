"""Helpers to switch Pandas to Arrow-backed dtypes where supported."""

import functools
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_ARROW_DTYPE_READERS = (
    "read_csv",
    "read_json",
    "read_parquet",
)

_configured = False


def _wrap_reader_with_arrow_backend(reader):
    """Force Arrow dtype backend for Pandas IO helpers that support it."""

    @functools.wraps(reader)
    def wrapped(*args, **kwargs):
        kwargs.setdefault("dtype_backend", "pyarrow")
        return reader(*args, **kwargs)

    return wrapped


def configure_pandas_arrow_backend() -> None:
    """Enable Arrow-backed string and IO dtypes for this process."""
    global _configured

    if _configured:
        return

    pd.options.mode.string_storage = "pyarrow"
    pd.options.future.infer_string = True

    for reader_name in _ARROW_DTYPE_READERS:
        reader = getattr(pd, reader_name, None)
        if reader is None:
            continue
        setattr(pd, reader_name, _wrap_reader_with_arrow_backend(reader))

    _configured = True
    logger.info("Enabled Pandas Arrow backend experiment")
