"""
Transactions - Simple Atomic State Management

Clean context managers for safe state updates.
"""

__all__ = ["atomic", "Atomic"]

from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Optional
import copy
import logging

logger = logging.getLogger(__name__)


@contextmanager
def atomic(obj: Any, *, deep: bool = False):
    """
    Context manager for atomic state changes.

    Automatically rolls back on exception.

    Args:
        obj: Object to manage (must have __dict__ or be dict-like)
        deep: If True, perform deep copy (safer but slower)

    Usage:
        with atomic(printer.state):
            printer.state.temperature = 200
            printer.state.status = "printing"
            # Automatically committed on success
            # Automatically rolled back on exception
    """
    # Take snapshot
    if hasattr(obj, "__dict__"):
        snapshot = copy.deepcopy(obj.__dict__) if deep else obj.__dict__.copy()
    elif isinstance(obj, dict):
        snapshot = copy.deepcopy(obj) if deep else obj.copy()
    else:
        raise TypeError(f"Cannot create transaction for {type(obj)}")

    try:
        yield obj
        # Success - changes committed
    except Exception:
        # Error - rollback
        if hasattr(obj, "__dict__"):
            obj.__dict__.clear()
            obj.__dict__.update(snapshot)
        elif isinstance(obj, dict):
            obj.clear()
            obj.update(snapshot)
        raise


@asynccontextmanager
async def atomic_async(obj: Any, *, deep: bool = False):
    """
    Async version of atomic context manager.

    Usage:
        async with atomic_async(printer.state):
            printer.state.temperature = 200
            await validate_state(printer.state)
    """
    if hasattr(obj, "__dict__"):
        snapshot = copy.deepcopy(obj.__dict__) if deep else obj.__dict__.copy()
    elif isinstance(obj, dict):
        snapshot = copy.deepcopy(obj) if deep else obj.copy()
    else:
        raise TypeError(f"Cannot create transaction for {type(obj)}")

    try:
        yield obj
    except Exception:
        if hasattr(obj, "__dict__"):
            obj.__dict__.clear()
            obj.__dict__.update(snapshot)
        elif isinstance(obj, dict):
            obj.clear()
            obj.update(snapshot)
        raise


class Atomic:
    """
    Mixin to add transaction support to classes.

    Usage:
        class PrinterState(Atomic):
            def __init__(self):
                self.temperature = 0.0
                self.status = "idle"

        state = PrinterState()

        with state.atomic():
            state.temperature = 200
            state.status = "printing"
    """

    def atomic(self, *, deep: bool = False):
        """Create atomic context for this object"""
        return atomic(self, deep=deep)

    async def atomic_async(self, *, deep: bool = False):
        """Create async atomic context for this object"""
        return atomic_async(self, deep=deep)
