"""Run-time environment CPU queries."""
import multiprocessing


def get_safe_max_workers_count(os_reserved_cpus=2) -> int:
    """Get safe number of worker processes.

    - A helper method to allocate CPUs for a grid search without grinding
      your computer to a halt

    :param os_reserved_cpus:
        Leave this number of CPUs for the host operating systems and other applications.
    """
    cpus = multiprocessing.cpu_count() - os_reserved_cpus
    assert cpus > 0
    return cpus