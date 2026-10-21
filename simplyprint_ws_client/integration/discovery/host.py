"""Harness-owned host for the always-on multicast discovery listeners.

Runs every backend's :meth:`MulticastDiscoveryBackend.run` coroutine concurrently
on ONE daemon thread + ONE asyncio loop, instead of a thread per brand. An
internal supervisor relaunches a backend that stops or crashes -- on the same
loop, so each backend's stop ``asyncio.Event`` stays bound to a single loop for
the whole process lifetime (re-awaiting it on a fresh loop would raise). All the
socket setup and the active-search cadence live inside ``run()``; this host only
owns *where* those coroutines execute and their restart-on-death.

Backends can be added and removed while the host runs
(:meth:`DiscoveryServiceHost.add_backend` / :meth:`~DiscoveryServiceHost.remove_backend`);
both hop onto the host loop, which is the only place the task map is touched.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Coroutine, Dict, Iterable, List, Optional, Protocol

from simplyprint_ws_client.common.asyncio.offload import install_default_executor


class DiscoveryBackendSpec(Protocol):
    """The slice of a backend's spec the host reads: its registry key."""

    @property
    def brand(self) -> str: ...


class DiscoveryBackend(Protocol):
    """What the host requires of a multicast listener backend.

    Structural: any object with a branded ``spec``, an ``async run()`` listener
    coroutine, an idempotent ``stop()``, and a ``clear()`` that re-arms the stop
    signal before a relaunch satisfies it (e.g. ``MulticastDiscoveryBackend``,
    ``MDNSDiscoveryBackend``).
    """

    @property
    def spec(self) -> DiscoveryBackendSpec: ...

    def run(self) -> Coroutine[Any, Any, None]: ...

    def stop(self) -> None: ...

    def clear(self) -> None: ...


class DiscoveryServiceHost:
    """One daemon thread + one loop hosting every multicast listener coroutine."""

    def __init__(
        self,
        backends: Iterable[DiscoveryBackend],
        *,
        restart_interval: float = 5.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._backends: List[DiscoveryBackend] = list(backends)
        self._restart_interval = restart_interval
        self.logger = logger or logging.getLogger("discovery")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._host_stop: Optional[asyncio.Event] = None
        self._tasks: Dict[str, asyncio.Task] = {}
        self._started = False
        # The backend list is copy-on-write under this lock: the loop iterates
        # whichever list it read while another thread adds or removes one.
        self._lock = threading.Lock()

        seen: set[str] = set()
        duplicates: set[str] = set()
        for backend in self._backends:
            key = backend.spec.brand
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            raise ValueError(
                "duplicate discovery backend key(s): " + ", ".join(sorted(duplicates))
            )

    def start(self) -> None:
        """Spin up the host loop + thread and launch every backend (idempotent)."""
        if self._started:
            return
        self._started = True
        self._ready.clear()
        self._loop = asyncio.new_event_loop()
        install_default_executor(
            self._loop,
            workers=2,
            thread_name_prefix="sp-discovery",
        )
        self._thread = threading.Thread(
            target=self._run_loop, name="discovery-host", daemon=True
        )
        self._thread.start()
        self._ready.wait()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()
            self._loop = None
            self._host_stop = None
            self._tasks.clear()

    async def _serve(self) -> None:
        self._host_stop = asyncio.Event()
        for backend in self._backends:
            self._tasks[backend.spec.brand] = asyncio.create_task(backend.run())
        # Sockets bind asynchronously inside run(); callers poll readiness rather
        # than assume a bound socket once start() returns.
        self._ready.set()
        try:
            while not self._host_stop.is_set():
                try:
                    await asyncio.wait_for(
                        self._host_stop.wait(), timeout=self._restart_interval
                    )
                except asyncio.TimeoutError:
                    self._restart_dead()
        finally:
            for backend in self._backends:
                backend.stop()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    def _restart_dead(self) -> None:
        """Relaunch, on this loop, any backend whose coroutine stopped or crashed."""
        for backend in self._backends:
            task = self._tasks.get(backend.spec.brand)
            if task is None or not task.done():
                continue
            if not task.cancelled() and task.exception() is not None:
                self.logger.error(
                    "discovery backend %s crashed - restarting",
                    backend.spec.brand,
                    exc_info=task.exception(),
                )
            else:
                self.logger.warning(
                    "discovery backend %s stopped - restarting", backend.spec.brand
                )
            backend.clear()
            self._tasks[backend.spec.brand] = asyncio.create_task(backend.run())

    def shutdown(self) -> None:
        """Stop every backend and the host loop, then join the thread (idempotent)."""
        if not self._started:
            return
        self._started = False
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._signal_stop)
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise TimeoutError("discovery host did not stop within 10 seconds")
            self._thread = None

    def add_backend(self, backend: DiscoveryBackend) -> None:
        """Host ``backend`` too; it is launched at once if the host is running."""
        brand = backend.spec.brand
        with self._lock:
            if any(existing.spec.brand == brand for existing in self._backends):
                raise ValueError("duplicate discovery backend key(s): " + brand)
            self._backends = self._backends + [backend]
            loop = self._loop
        if loop is None:
            return  # not started: _serve launches it with the rest
        try:
            loop.call_soon_threadsafe(self._launch, backend)
        except RuntimeError:
            pass  # the loop closed under us; a restart launches it

    def _launch(self, backend: DiscoveryBackend) -> None:
        brand = backend.spec.brand
        if self._host_stop is None or self._host_stop.is_set():
            return
        if brand in self._tasks or backend not in self._backends:
            return  # _serve already launched it, or it was removed meanwhile
        self._tasks[brand] = asyncio.create_task(backend.run())

    def remove_backend(
        self, backend: DiscoveryBackend
    ) -> Optional["concurrent.futures.Future[None]"]:
        """Stop and forget ``backend``.

        Returns a future that resolves once its listener has wound down on the
        host loop, or ``None`` when there was nothing running to wait for.
        """
        with self._lock:
            if not any(existing is backend for existing in self._backends):
                return None
            self._backends = [b for b in self._backends if b is not backend]
            loop = self._loop
        if loop is None or loop.is_closed():
            backend.stop()
            return None
        try:
            return asyncio.run_coroutine_threadsafe(self._retire(backend), loop)
        except RuntimeError:  # the loop closed under us
            backend.stop()
            return None

    async def _retire(self, backend: DiscoveryBackend) -> None:
        backend.stop()
        brand = backend.spec.brand
        task = self._tasks.get(brand)
        if task is None:
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._tasks.get(brand) is task:
            del self._tasks[brand]

    def backend_keys(self) -> List[str]:
        """The keys of the backends this host runs (or will run)."""
        return [backend.spec.brand for backend in self._backends]

    def _signal_stop(self) -> None:
        if self._host_stop is not None:
            self._host_stop.set()
