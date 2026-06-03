"""Bounded shutdown helper for process entry points."""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_STOP_TIMEOUT = 5.0


def stop_with_timeout(
    stop: Callable[[], None],
    *,
    timeout: float = DEFAULT_STOP_TIMEOUT,
    force_exit: Callable[[int], None],
) -> None:
    """Invoke ``stop`` with a bounded wait, forcing exit if it hangs."""

    worker = threading.Thread(target=stop, name="app-stop", daemon=True)
    worker.start()
    worker.join(timeout=timeout)

    if worker.is_alive():
        logger.error("Shutdown did not complete within %.1fs; forcing exit.", timeout)
        force_exit(1)
