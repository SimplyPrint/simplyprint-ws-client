"""
SPC Core Future - Clean, Modern API for Printer Integrations

A complete rewrite with:
- Cleaner, more conventional naming
- Better separation of concerns
- Isolated, runnable examples
- Type-safe patterns
"""

__version__ = "2.0.0-alpha"

# Core exports with clean naming
from .di import Container, Injected, Service, inject
from .events import MessageBus, handles, emits, Event
from .scheduling import Scheduled, interval, on_startup, on_shutdown
from .protocols import Protocol, Client, Printer
from .transactions import atomic, Atomic

__all__ = [
    # Dependency Injection
    "Container",
    "Injected",
    "Service",
    "inject",
    # Event System
    "MessageBus",
    "handles",
    "emits",
    "Event",
    # Scheduling
    "Scheduled",
    "interval",
    "on_startup",
    "on_shutdown",
    # Core Abstractions
    "Protocol",
    "Client",
    "Printer",
    # Transactions
    "atomic",
    "Atomic",
]
