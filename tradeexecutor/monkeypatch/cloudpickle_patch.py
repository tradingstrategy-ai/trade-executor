"""Enable using multiprocess in Jupyter Notebooks.

See https://stackoverflow.com/questions/77453594/parallelising-functions-using-multiprocessing-in-jupyter-notebook

- Monkey patches the standard library `pickle` implementation with one that fully pickles
  functions, instead of passing them by a dotted name reference, between multiprocesses
"""

import sys
from multiprocessing import Pool
from multiprocessing.reduction import ForkingPickler
from types import FunctionType
import cloudpickle

assert sys.version_info >= (3, 8), 'python3.8 or greater required to use reducer_override'

def reducer_override(obj):
    if type(obj) is FunctionType:
        return (cloudpickle.loads, (cloudpickle.dumps(obj),))
    else:
        return NotImplemented

# Monkeypatch our function reducer into the pickler for multiprocessing.
# Without this line, the main block will not work on windows or macOS.
# Alternatively, moving the definition of foo outside of the if statement
# would make the main block work on windows or macOS (when run from
# the command line).
ForkingPickler.reducer_override = staticmethod(reducer_override)