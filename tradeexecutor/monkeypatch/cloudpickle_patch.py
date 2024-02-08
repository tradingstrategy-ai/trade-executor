"""Make it possible to work.

See https://stackoverflow.com/questions/77453594/parallelising-functions-using-multiprocessing-in-jupyter-notebook
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
# Alterntively, moving the defintionn of foo outside of the if statement
# would make the main block work on windows or macOS (when run from
# the command line).
ForkingPickler.reducer_override = staticmethod(reducer_override)

if __name__ == '__main__':
    def foo(x, y):
        return x * y

    with Pool() as pool:
        res = pool.apply(foo, (10, 3))

    print(res)
    assert res == 30
