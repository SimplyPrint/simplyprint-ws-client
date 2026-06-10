"""The polling driver: request/response devices get the same driver lifecycle.

A :class:`DevicePoller` runs the client's ``poll_device()`` on an interval, on
the client's loop, and derives the connected/disconnected edges from poll
outcomes — the loop an HTTP-polling integration used to hand-roll across
``init``/``tick`` overrides (ensure-session, cadence, exception backoff,
offline-after-silence), as one supervised lifecycle with policy knobs.

Rules per cycle:

* the poll returns      -> sign of life; first success (or first after a
                           disconnected stretch) fires the connected edge.
* raises DeviceAuthError-> the single-flight credential refresh runs (and the
                           cycle backs off like a failure).
* raises anything else  -> log at debug, sleep ``failure_backoff``.
* no success for ``offline_after`` seconds -> the disconnected edge fires once;
  polling continues (the device may come back).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from simplyprint_ws_client.integration.driver import DeviceAuthError, DeviceDriver

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.client import PrinterClient

__all__ = ["DevicePoller"]


class DevicePoller(DeviceDriver):
    """Drives ``poll_device()`` on an interval and owns the edge bookkeeping."""

    def __init__(
        self,
        client: "PrinterClient",
        *,
        interval: float,
        poll: Optional[Callable[[], Awaitable[None]]] = None,
        offline_after: float = 300.0,
        failure_backoff: float = 10.0,
        name: str = "device",
    ) -> None:
        super().__init__(client, name=name)
        self.interval = interval
        self.offline_after = offline_after
        self.failure_backoff = failure_backoff
        self._poll = poll
        self._task: Optional[asyncio.Task] = None

    @property
    def connected(self) -> bool:
        return self.is_connected is True

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = self.client.event_loop
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            self.client.logger.debug(
                "cannot start %s poller yet: no running event loop", self.name
            )
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._task = loop.create_task(self._run())
        else:
            loop.call_soon_threadsafe(self._start_on_loop, loop)

    def _start_on_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run())

    def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()

    async def _run(self) -> None:
        while True:
            failed = False
            try:
                await (self._poll or self.client.poll_device)()
            except asyncio.CancelledError:
                raise
            except DeviceAuthError:
                failed = True
                self.request_credential_refresh()
            except Exception:  # noqa: BLE001 -- supervised: a bad poll backs off
                failed = True
                self.client.logger.debug("%s poll failed", self.name, exc_info=True)
            else:
                self.last_message_at = time.monotonic()
                if self.is_connected is not True:
                    self.is_connected = True
                    await self.client.on_device_connected(self)

            if (
                self.is_connected is True
                and self.last_message_at is not None
                and time.monotonic() - self.last_message_at >= self.offline_after
            ):
                self.is_connected = False
                await self.client.on_device_disconnected(self, reason=None)

            await asyncio.sleep(self.failure_backoff if failed else self.interval)
