"""Run-time environment CPU queries."""
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