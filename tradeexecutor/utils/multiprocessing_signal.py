"""Helpers for multiprocessing signal handling."""

import signal


def suppress_worker_sigint_tracebacks() -> None:
    """Suppress notebook interrupt noise in child worker processes.

    In notebook-driven backtests and research runs we have observed child
    ``ProcessPoolExecutor`` workers printing ``KeyboardInterrupt`` tracebacks
    during otherwise successful parent shutdown. Our 2026-03-19 NB153 rerun
    investigation showed the notebook completed and saved results correctly,
    while idle worker processes still emitted noisy shutdown tracebacks.

    Python's multiprocessing documentation notes that child ``SIGINT``
    behaviour can be altered with :py:func:`signal.signal`. We ignore
    ``SIGINT`` in worker children and let the parent notebook process own the
    interrupt and teardown path, so genuine task failures still surface
    through futures instead of being drowned in worker shutdown noise.
    """

    signal.signal(signal.SIGINT, signal.SIG_IGN)
