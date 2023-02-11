"""Python module loading helpers."""

import importlib
import os.path
import sys
import types
from pathlib import Path


def import_python_source_file(fname: Path) -> types.ModuleType:
    """Import a Python source file and return the loaded module.

    `See the original StackOverflow answer <https://stackoverflow.com/a/41595552/315168>`_.

    :param fname:
        The full path to the source file.
        It may container characters like `.` or `-`.

    :return:
         The imported module

    :raise:
         ImportError: If the file cannot be imported (e.g, if it's not a `.py` file or if
             it does not exist).
         Exception: Any exception that is raised while executing the module (e.g.,
             :exc:`SyntaxError).  These are errors made by the author of the module!
     """

    modname = os.path.basename(fname)

    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e

    return module
    

def extract_module_members(module: types.ModuleType) -> dict:
    """Get all variables in a module.

    See https://stackoverflow.com/a/5103466/315168
    """
    return module.__dict__.copy()
