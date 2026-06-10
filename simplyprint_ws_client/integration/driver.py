"""The device-side half of one printer client: how the printer is reached.

A :class:`DeviceDriver` owns one lifecycle (``start`` -> ``ensure_started`` ->
``suspend`` -> ``stop``), liveness (``is_connected`` tri-state,
``last_message_at``), and the single-flight credential-refresh choreography every
re-authenticating device needs. Concrete drivers — the pooled links in
:mod:`~simplyprint_ws_client.integration.link` and the request/response
:class:`~simplyprint_ws_client.integration.poller.DevicePoller` — deliver device
edges by calling their client's ``on_device_connected`` /
``on_device_disconnected`` / ``on_device_message`` hooks, already on the client's
loop, so a brand never writes thread-hop or re-emit plumbing again.

The base :class:`~simplyprint_ws_client.integration.client.PrinterClient` owns
WHEN drivers run: it starts every declared driver in ``init``, sweeps
``ensure_started`` each tick (a driver whose config wasn't ready yet retries for
free), suspends on ``halt``, and stops on ``teardown``.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from simplyprint_ws_client.integration.client import PrinterClient

__all__ = ["DeviceAuthError", "DeviceDriver"]


class DeviceAuthError(Exception):
    """The device rejected our credentials (session/token expired).

    Raise it from ``poll_device()`` (or anywhere a driver surfaces it) to trigger
    the single-flight ``refresh_device_credentials`` -> restart choreography.
    """


class DeviceDriver(ABC):
    """One client's supervised attachment to its physical device.

    Lifecycle contract (the base printer client drives it):

    * :meth:`start` — idempotent and tolerant: a device whose config is not ready
      yet (no host, no credentials) logs and returns; the per-tick
      :meth:`ensure_started` sweep retries for free.
    * :meth:`suspend` — the client left scheduling temporarily (``halt``).
    * :meth:`stop` — final teardown. Idempotent.
    * :meth:`restart` — stop + start; re-resolves URLs/sessions, which is why
      link parameters are callables.

    Liveness: ``is_connected`` is tri-state (``None`` = never connected yet) and
    ``last_message_at`` is a monotonic timestamp of the last inbound sign of
    life — both fed by the concrete driver.

    Credential refresh: :meth:`request_credential_refresh` is single-flight; it
    awaits the client's ``refresh_device_credentials(driver)`` on the client's
    loop and restarts the driver on ``True``.
    """

    def __init__(self, client: "PrinterClient", *, name: str = "device") -> None:
        self.client = client
        self.name = name
        self.is_connected: Optional[bool] = None
        self.last_message_at: Optional[float] = None
        self._refresh_lock = threading.Lock()
        self._refreshing = False

    @abstractmethod
    def start(self) -> None:
        """Begin reaching the device. Idempotent; 'not ready yet' is tolerated."""

    def ensure_started(self) -> None:
        """Cheap per-tick retry; the default just calls the idempotent start."""
        self.start()

    def suspend(self) -> None:
        """The client is temporarily out of scheduling. Default: full stop."""
        self.stop()

    @abstractmethod
    def stop(self) -> None:
        """Tear the attachment down. Idempotent."""

    def restart(self) -> None:
        """Stop and start again, re-resolving URLs/sessions."""
        self.stop()
        self.start()

    def request_credential_refresh(self) -> None:
        """Run the client's credential refresh exactly once, then restart.

        Safe to call from any thread and repeatedly: concurrent requests
        coalesce into the one in flight (the anycubic-style lock+flag, owned
        here once).
        """
        with self._refresh_lock:
            if self._refreshing:
                return
            self._refreshing = True
        self.client.submit_to_loop(self._run_credential_refresh())

    async def _run_credential_refresh(self) -> None:
        refreshed = False
        try:
            refreshed = await self.client.refresh_device_credentials(self)
        except Exception:  # noqa: BLE001 -- a failed refresh must not kill the loop
            self.client.logger.warning(
                "device credential refresh failed", exc_info=True
            )
        finally:
            with self._refresh_lock:
                self._refreshing = False
        if refreshed:
            self.client.logger.info(
                "device credentials refreshed; restarting %s", self.name
            )
            self.restart()
