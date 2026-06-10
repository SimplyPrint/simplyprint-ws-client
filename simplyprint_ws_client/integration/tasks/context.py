"""Runtime handle a task receives when it runs: :class:`TaskContext`.

A :class:`~simplyprint_ws_client.integration.tasks.spec.TaskSpec` is pure
declaration; when the scheduler (or an on-demand trigger) runs the task it calls
``spec.fn(ctx)`` with a ``TaskContext``. The context is how a task reaches the
wired runtime -- chiefly the
:class:`~simplyprint_ws_client.contrib.status.registry.StatusRegistry` it
publishes results into -- without reaching for a global. This is what keeps
producers (health, OTA, discovery, ...) decoupled: they are handed what they
need rather than importing a service locator.
"""

from __future__ import annotations

from dataclasses import dataclass

from simplyprint_ws_client.contrib.status.registry import StatusRegistry


@dataclass(frozen=True)
class TaskContext:
    """What a task body is handed on each run.

    ``key`` distinguishes per-entity runs of the same task: a service-wide
    scheduled run carries the default empty key, while an on-demand "re-check
    *this* printer" run carries that printer's id, so single-flight coalescing
    and status sections can be scoped per entity.
    """

    status: StatusRegistry
    """The status registry this run publishes its result into."""

    key: str = ""
    """Entity scope for this run ("" = the service-wide run; else e.g. a printer
    id)."""

    # NOTE: more wiring (a cancellation token, the triggering app context, a
    # logger bound to the task name) lands here as the scheduler internals are
    # filled in. Additive only -- the (status, key) contract above is stable.
