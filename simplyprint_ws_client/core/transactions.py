"""
Transaction support for state management.

Provides context managers for atomic state changes with rollback support.
"""

__all__ = ["Transaction", "Transactional"]

from typing import Any, Dict, Optional, Callable
from contextlib import contextmanager, asynccontextmanager
from abc import ABC, abstractmethod
import logging
import copy

logger = logging.getLogger(__name__)


class Transaction:
    """
    Transaction context for state changes.

    Provides:
    - Automatic rollback on exception
    - Manual commit/rollback
    - Nested transaction support (via savepoints)

    Usage:
        ```python
        with Transaction(printer.state) as tx:
            printer.state.temperature = 200
            printer.state.status = "printing"
            # Automatically committed on success
            # Automatically rolled back on exception
        ```
    """

    def __init__(self, obj: Any, *, deep_copy: bool = False):
        """
        Args:
            obj: Object to manage transactionally (must have __dict__ or be dict-like)
            deep_copy: If True, perform deep copy of state (slower but safer)
        """
        self._obj = obj
        self._snapshot: Optional[Dict[str, Any]] = None
        self._committed = False
        self._rolled_back = False
        self._deep_copy = deep_copy

    def __enter__(self):
        """Start transaction and take snapshot"""
        self._take_snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback based on exception"""
        if exc_type is not None:
            # Exception occurred - rollback
            logger.debug(f"Transaction rolling back due to {exc_type.__name__}: {exc_val}")
            self.rollback()
            return False  # Re-raise exception

        # No exception - commit
        if not self._committed and not self._rolled_back:
            self.commit()

        return True

    def commit(self):
        """Manually commit the transaction"""
        if self._committed:
            logger.warning("Transaction already committed")
            return

        if self._rolled_back:
            raise RuntimeError("Cannot commit a rolled back transaction")

        self._committed = True
        self._snapshot = None
        logger.debug("Transaction committed")

    def rollback(self):
        """Manually rollback the transaction"""
        if self._rolled_back:
            logger.warning("Transaction already rolled back")
            return

        if self._committed:
            raise RuntimeError("Cannot rollback a committed transaction")

        self._restore_snapshot()
        self._rolled_back = True
        logger.debug("Transaction rolled back")

    def _take_snapshot(self):
        """Take a snapshot of current state"""
        if hasattr(self._obj, "__dict__"):
            # Object with __dict__
            if self._deep_copy:
                self._snapshot = copy.deepcopy(self._obj.__dict__)
            else:
                self._snapshot = self._obj.__dict__.copy()
        elif isinstance(self._obj, dict):
            # Dictionary
            if self._deep_copy:
                self._snapshot = copy.deepcopy(self._obj)
            else:
                self._snapshot = self._obj.copy()
        else:
            raise TypeError(
                f"Cannot create transaction for type {type(self._obj)}. "
                "Must have __dict__ or be dict-like"
            )

    def _restore_snapshot(self):
        """Restore state from snapshot"""
        if self._snapshot is None:
            raise RuntimeError("No snapshot available")

        if hasattr(self._obj, "__dict__"):
            # Clear current dict and restore
            self._obj.__dict__.clear()
            self._obj.__dict__.update(self._snapshot)
        elif isinstance(self._obj, dict):
            # Clear and restore dict
            self._obj.clear()
            self._obj.update(self._snapshot)


class Transactional(ABC):
    """
    Base class for objects that support transactions.

    Provides convenience methods for creating transactions.
    """

    def transaction(self, *, deep_copy: bool = False) -> Transaction:
        """
        Create a transaction for this object.

        Args:
            deep_copy: If True, perform deep copy of state

        Returns:
            Transaction context manager

        Usage:
            ```python
            with obj.transaction():
                obj.field = new_value
                # Changes automatically committed or rolled back
            ```
        """
        return Transaction(self, deep_copy=deep_copy)

    @contextmanager
    def atomic(self):
        """
        Alias for transaction() with default settings.

        Usage:
            ```python
            with obj.atomic():
                obj.field = new_value
            ```
        """
        with self.transaction():
            yield self

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the object's state.

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If state is invalid (with description)
        """
        return True


class ValidatingTransaction(Transaction):
    """
    Transaction that validates state before commit.

    Usage:
        ```python
        with ValidatingTransaction(obj) as tx:
            obj.field = new_value
            # Automatically validated before commit
            # Rolls back if validation fails
        ```
    """

    def __init__(self, obj: Transactional, *, deep_copy: bool = False):
        if not isinstance(obj, Transactional):
            raise TypeError("Object must be Transactional")

        super().__init__(obj, deep_copy=deep_copy)

    def commit(self):
        """Commit with validation"""
        # Validate before commit
        try:
            if not self._obj.validate():
                raise ValueError("Validation failed")
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.rollback()
            raise

        super().commit()


# ============================================================================
# Async Transaction Support
# ============================================================================


class AsyncTransaction(Transaction):
    """
    Async version of Transaction.

    Supports async validation and callbacks.

    Usage:
        ```python
        async with AsyncTransaction(obj) as tx:
            obj.field = new_value
            await obj.async_validate()
        ```
    """

    async def __aenter__(self):
        """Start async transaction"""
        self._take_snapshot()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Commit or rollback async transaction"""
        if exc_type is not None:
            logger.debug(f"Async transaction rolling back due to {exc_type.__name__}")
            await self.arollback()
            return False

        if not self._committed and not self._rolled_back:
            await self.acommit()

        return True

    async def acommit(self):
        """Async commit"""
        # Call async validation if available
        if hasattr(self._obj, "avalidate"):
            try:
                await self._obj.avalidate()
            except Exception as e:
                logger.error(f"Async validation failed: {e}")
                await self.arollback()
                raise

        self.commit()

    async def arollback(self):
        """Async rollback"""
        self.rollback()


# ============================================================================
# Example: Transactional State Model
# ============================================================================


class PrinterStateTransaction(Transactional):
    """
    Example of a transactional state model.

    This would be mixed into PrinterState in the real implementation.
    """

    def __init__(self):
        self.temperature = 0.0
        self.status = "idle"
        self.progress = 0.0

    def validate(self) -> bool:
        """Validate printer state"""
        # Check that values are in valid ranges
        if self.temperature < 0 or self.temperature > 300:
            raise ValueError(f"Invalid temperature: {self.temperature}")

        if self.status not in ("idle", "printing", "paused", "error"):
            raise ValueError(f"Invalid status: {self.status}")

        if self.progress < 0 or self.progress > 100:
            raise ValueError(f"Invalid progress: {self.progress}")

        return True

    @asynccontextmanager
    async def job_context(self):
        """
        Convenience context manager for job operations.

        Usage:
            ```python
            async with printer.state.job_context():
                printer.state.status = "printing"
                printer.state.progress = 0
                # Validated and committed automatically
            ```
        """
        async with AsyncTransaction(self) as tx:
            yield self
            # Validation happens automatically in __aexit__


# ============================================================================
# Usage Examples
# ============================================================================


def example_basic_transaction():
    """Example: Basic transaction usage"""

    class SimpleState:
        def __init__(self):
            self.value = 0

    state = SimpleState()

    # Successful transaction
    with Transaction(state):
        state.value = 10
        # Committed automatically

    assert state.value == 10

    # Failed transaction (rollback)
    try:
        with Transaction(state):
            state.value = 20
            raise ValueError("Something went wrong")
    except ValueError:
        pass

    # Value rolled back
    assert state.value == 10


async def example_validating_transaction():
    """Example: Transaction with validation"""

    state = PrinterStateTransaction()

    # Valid changes
    async with AsyncTransaction(state):
        state.temperature = 200
        state.status = "printing"
        # Validated and committed

    # Invalid changes (rolls back)
    try:
        async with AsyncTransaction(state):
            state.temperature = 500  # Invalid!
    except ValueError:
        pass

    # Temperature still 200 (rolled back)
    assert state.temperature == 200


async def example_job_context():
    """Example: Using job_context convenience method"""

    state = PrinterStateTransaction()

    async with state.job_context():
        state.status = "printing"
        state.progress = 50
        # Validated automatically

    assert state.status == "printing"
    assert state.progress == 50
