"""Python cProf utilities"""

import cProfile
import pstats
from contextlib import contextmanager

@contextmanager
def profiled(count=30):
    """Run a Python snippet and show the spent time.

    - A crude profiling for notebooks / scripts

    Example:

    .. code-block:: python

        # Usage
        with profiled():
            # Your code section to profile goes here
            for i in range(1000000):
                _ = i ** 2
    """
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative').print_stats(30)

