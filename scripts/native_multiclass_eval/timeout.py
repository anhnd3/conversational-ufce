"""Per-query wall-clock timeout shared by the experiment runner and adapters."""

import signal
from contextlib import contextmanager


class QueryTimeout(Exception):
    pass


@contextmanager
def query_timeout(seconds):
    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handler(_signum, _frame):
        raise QueryTimeout()

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

