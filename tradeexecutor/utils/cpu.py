"""Run-time environment CPU queries."""
import inspect
import multiprocessing


def get_safe_max_workers_count(os_reserved_cpus=2) -> int:
    """Get safe number of worker processes.

    - A helper method to allocate CPUs for a grid search without grinding
      your computer to a halt

    - Handle the special case of Docker

    :param os_reserved_cpus:
        Leave this number of CPUs for the host operating systems and other applications.

    :return:
        The number of worker processes that it safe to launch without overloading the machine,
        assuming there is no other load.

        Some environments like Docker do not expose the CPU count.
        In this case we return 1.
    """
    cpus = multiprocessing.cpu_count() - os_reserved_cpus

    if cpus <= 0:
        return 1

    return cpus


def is_running_in_ipython() -> bool:
    """Detect if we are running inside the IPython CLI (not a Jupyter kernel).

    - ``ipython notebook.ipynb`` executes cells directly in its own process,
      where ``start_ipython`` is on the call stack. Spawning child processes
      via Loky/joblib fails because IPython's dynamic ``__main__`` module
      cannot be properly resolved in spawned worker processes.

    - ``jupyter execute notebook.ipynb`` launches an IPython **kernel as a
      separate subprocess** (via ``IPKernelApp``, not ``start_ipython``),
      so multiprocessing works normally — Loky can spawn workers and
      cloudpickle handles function serialisation across processes.

    - This function checks for ``start_ipython`` in the call stack to
      distinguish the two cases.

    - When IPython CLI is detected, callers should fall back to
      ``max_workers=1`` and log a warning suggesting ``jupyter execute``.
    """
    return any(frame for frame in inspect.stack() if frame.function == "start_ipython")