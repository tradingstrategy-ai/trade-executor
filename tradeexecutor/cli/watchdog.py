"""Watch out for internal oracle hang conditions.

Suitable for multithread and multiprocess watching.
"""
import enum
import logging
import os
import signal
import time
from multiprocessing import Manager
from multiprocessing.managers import DictProxy
from threading import Thread
from typing import List, TypeAlias, Optional


logger = logging.getLogger()


#: Watchdog is a multprocess safe dict for now
WatchdogRegistry: TypeAlias = DictProxy | dict


#: Internal multiprocess manager co-ordinator
#:
#: We need to lazy init because of Python VM bootstrap order
_manager: Optional[Manager] = None


class WatchedWorkerDidNotReport(Exception):
    """Raised when a watched process/thread/loop fails to report in time."""


class WorkerNotRegistered(Exception):
    """Tried to get ping from a worker that is not yet registered."""


class WatchdogMode(enum.Enum):
    """How does the watchdog communicate with its tasks."""

    #: Thread based.
    #:
    #: The application does not need to communicate with child processe.
    thread_based = "thread_based"

    #: Process based.
    #:
    #: The application has child processes and needs to set up
    #: multiprocess communicatons.
    #:
    #: :py:class:`DictProxy` is used to communicate task
    #: liveness and it works across process boundaries.
    #:
    #: For the caveats see,
    #: https://stackoverflow.com/a/75385991/315168 as this may
    #: cause zombie processes.
    process_based = "process_based"


def create_watchdog_registry(mode: WatchdogMode) -> WatchdogRegistry:
    """Create new multiprocess co-ordation structure.

    - Call in the master process

    - Registry must be passed to threads/processes setting up their own loops

    :return:
        Multiprocess communication safe dict
    """

    global _manager

    if mode == WatchdogMode.process_based:
        if _manager is None:
            _manager = Manager()
        return _manager.dict()
    else:
        # For thread-based co-ordinate we can use a normal Python dict
        # that is thread safe
        return dict()


def register_worker(watchdog_registry: WatchdogRegistry, name: str, timeout_seconds: float):
    """Register a worker/main loop.

    Call before your process begins.

    :param name:
        Name of this process/thread/loop

    :param timeout_seconds:
        How often this
    """

    assert timeout_seconds > 0

    # For each process we create timeout and last ping entries
    watchdog_registry[f"{name}:timeout"] = timeout_seconds

    # Initialise the value
    mark_alive(watchdog_registry, name)


def mark_alive(watchdog_registry: WatchdogRegistry, name: str, time_: Optional[float] = None):
    """Mark the worker being alive.

    Call for each registered worker at the end of its
    duty cycle.
    """

    if not time_:
        time_ = time.time()

    watchdog_registry[f"{name}:last_ping"] = time_


def get_last_ping(watchdog_registry: WatchdogRegistry, name: str) -> float:
    """Get the UNIX timestamp when a process pinged last time.

    :raise WorkerNotRegistered:
        If querying an unknown worker.
    """

    timeout = watchdog_registry.get(f"{name}:timeout")
    if timeout is None:
        raise WorkerNotRegistered(f"No watchdog worker: {name}")

    val = watchdog_registry.get(f"{name}:last_ping")
    return val


def get_timeout(watchdog_registry: WatchdogRegistry, name: str) -> float:
    """Get the UNIX timestamp when a process pinged last time."""
    return watchdog_registry[f"{name}:timeout"]


def get_watched_workers(watchdog_registry: WatchdogRegistry) -> List[str]:
    """Get list of processes that are on the watchdog list."""

    workers = []

    for key in watchdog_registry:
        if ":timeout" in key:
            name, timeout = key.split(":")
            workers.append(name)

    return workers


def check_hung_workers(watchdog_registry: WatchdogRegistry):
    """Check that all workers have reported in time.

    - Call in the master process regularly.

    :raise WatchedProcessDidNotReport:
        If any of the watched processes has failed to report in time.
    """
    workers = get_watched_workers(watchdog_registry)
    for w in workers:
        last_ping = get_last_ping(watchdog_registry, w)
        assert last_ping is not None, f"No ping data for worker: {w}"
        timeout = get_timeout(watchdog_registry, w)
        since_ping = time.time() - last_ping
        if since_ping > timeout:
            raise WatchedWorkerDidNotReport(f"Watched worker {w} did not report back in time. Threshold seconds {timeout}, but it has been {since_ping} seconds.")


def start_background_watchdog(watchdog_registry: WatchdogRegistry):
    """Call in the main thread.

    - Starts the watchdog background thread that will watch over all workers in all processes

    - Kill the main process if any of the workers seem to be hung
    """
    def _run():

        last_report = 0

        while True:

            # Ping logs we are still alive
            if time.time() - last_report > 1800:
                logger.info("Watchdog background thread running")
                last_report = time.time()

            try:
                check_hung_workers(watchdog_registry)
            except Exception as e:
                logger.critical("Watchdog detected a process has hung: %s Shutting down.", e)
                time.sleep(10)  # Give some time to Discord logger to send the message before crashing down
                suicide()

            time.sleep(5)

    t = Thread(target=_run, daemon=True)
    t.start()
    logger.info("Watchdog thread started")


def suicide():
    """Hard kill Python application despite multiple threads.

    faulthandler will cause a thread dump,
    so you can examine the hung sate.

    https://stackoverflow.com/a/7099229/315168
    """
    os.kill(os.getpid(), signal.SIGINT)
