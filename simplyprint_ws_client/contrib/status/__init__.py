"""Producer-agnostic service status: publish a section, serve a snapshot.

Periodic tasks publish a :class:`StatusEntry` per *section* into a
:class:`StatusRegistry`; a single status endpoint reads
:meth:`StatusRegistry.snapshot` and rolls the sections up. The registry knows no
section by name -- a new producer (health, OTA, discovery, ...) is just another
``publish`` call. See :mod:`simplyprint_ws_client.contrib.tasks` for the
producers (tasks) that feed it.
"""

from simplyprint_ws_client.contrib.status.registry import (
    StatusEntry,
    StatusRegistry,
    StatusState,
)

__all__ = [
    "StatusEntry",
    "StatusRegistry",
    "StatusState",
]
