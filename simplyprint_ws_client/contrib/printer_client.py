"""The headline base for authoring a printer integration, as one business object.

A printer client's job is to boil a device's model data into a
:class:`~simplyprint_ws_client.core.state.PrinterState`. The *machinery* around
that -- running the connection-component lifecycle, ticking host telemetry +
ping, wiring the device connection's events, reducing a device-mapped status
through the shared cancel/pause/download holds and job-start/finish edges, and
resolving the camera URI -- is identical across every brand; only the
device-data extraction differs.

This used to be ~80 near-identical lines re-implemented in every integration's
``printer.py``. It now lives here as :class:`PrinterClient`; a brand subclass
supplies only what is genuinely device-specific via the hooks at the bottom of
the class:

* ``_resolve_camera_uri``   -- the device's current camera URL (or None)
* ``_on_job_start`` / ``_on_job_finish`` / ``_on_job_progress``
                            -- capture/classify the device's job fields
* ``_start_connection`` / ``_stop_connection`` / ``_connection_components``
                            -- arm and tear down the device connection component
* ``_connection_event_bus`` / ``_connection_event_bindings``
                            -- declare which device events drive update/connect/
                               disconnect (skipped by HTTP-polling devices)
* ``_tick_progress``        -- drive a client-side progress shim each tick

The subtle part is :meth:`apply_status`: the guard -> edge -> apply pipeline is
owned here whole, because the *transition semantics* are identical across
devices; only the device-state mapping and the per-edge field capture differ,
and those are delegated to the hooks. A subclass maps its device status to a
:class:`PrinterStatus` then calls :meth:`apply_status` -- it never re-implements
the holds or the start/finish detection.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
)

from ..core.client import DefaultClient
from ..core.config import PrinterConfig
from ..core.state import PrinterStatus
from ..shared.camera.mixin import ClientCameraMixin
from ..shared.hardware.physical_machine import PhysicalMachine

if TYPE_CHECKING:
    from yarl import URL

    from ..events.event_bus import EventBus

TConfig = TypeVar("TConfig", bound=PrinterConfig)

#: One device-event-class -> base/brand-handler binding for the connection
#: component's own event bus. A flat list of these expresses both single-update
#: devices and devices with several update events without the base assuming a
#: shape.
ConnectionEventBinding = Tuple[type, Callable[..., Any]]

#: Host usage is read at most this often, shared across every client, so the
#: delta-based ``psutil.cpu_percent`` isn't reset by every client every tick.
_HOST_USAGE_MIN_INTERVAL = 5.0
_host_usage_snapshot: dict = {}
_host_usage_read_at: float = 0.0


def _host_usage() -> dict:
    """Return a process-wide, throttled snapshot of host CPU/memory usage."""
    global _host_usage_snapshot, _host_usage_read_at
    now = time.monotonic()
    if (
        not _host_usage_snapshot
        or now - _host_usage_read_at >= _HOST_USAGE_MIN_INTERVAL
    ):
        _host_usage_snapshot = PhysicalMachine.get_usage()
        _host_usage_read_at = now
    return _host_usage_snapshot


class PrinterClient(
    DefaultClient[TConfig], ClientCameraMixin[TConfig], Generic[TConfig]
):
    """Common base for device printer clients.

    Subclasses still own ``__init__`` (they construct their connection
    component, do device info population, and -- for push devices -- call
    :meth:`_wire_connection_events` after the component exists) and the
    device-model -> :class:`PrinterState` mapping. Everything else (lifecycle,
    status application, camera resolution, host telemetry) is owned here and
    parameterised through the hooks below.
    """

    #: Camera mixin tuning consumed by :meth:`_init_camera`. Subclasses override
    #: the class attribute when their camera wants a different cache window.
    camera_pause_timeout: int = 10
    camera_max_cache_age: timedelta = timedelta(seconds=1)

    # -- lifecycle template --------------------------------------------------
    # init/halt are called once per halt + initially; tick every scheduling
    # slice; teardown once at final cleanup (see Client docstring).

    async def init(self) -> None:
        """Arm the device connection component."""
        await self._start_connection()

    async def tick(self, _delta) -> None:
        """Host housekeeping every slice: progress shim, ambient sensor, host
        telemetry and the SimplyPrint heartbeat ping (each interval-gated)."""
        self._tick_progress()
        self.printer.ambient_temperature.tick(self.printer)
        self.update_host_telemetry()
        await self.send_ping()

    async def halt(self) -> None:
        """Temporarily out of scheduling. No-op by default; HTTP-polling devices
        override to cancel their driver task(s)."""

    async def teardown(self) -> None:
        """Final cleanup: tear the device connection down."""
        await self._stop_connection()

    # -- connection-component lifecycle hooks --------------------------------

    async def _start_connection(self) -> None:
        """Establish/arm the device connection. Push devices start MQTT/WS here.
        No-op default for devices that start in ``__init__``."""

    async def _stop_connection(self) -> None:
        """Tear the connection down. Default stops every component returned by
        :meth:`_connection_components`; override for devices that need extra
        teardown commands first."""
        for component in self._connection_components():
            component.stop()

    def _connection_components(self) -> Iterable[Any]:
        """The connection component(s) to ``stop()`` on teardown. Override to
        return your device client(s)."""
        return ()

    def _tick_progress(self) -> None:
        """Drive a client-side progress shim each tick. Default no-op; devices
        with a fake-progress shim override to tick it."""

    # -- connection event wiring (push devices) ------------------------------

    def _connection_event_bus(self) -> Optional["EventBus"]:
        """The connection component's own event bus, if it has one. Push devices
        return e.g. ``self.device_client.event_bus``; HTTP-polling and
        single-callback devices return ``None`` so wiring is skipped."""
        return None

    def _connection_event_bindings(self) -> Iterable[ConnectionEventBinding]:
        """Declare which device events drive update/connect/disconnect. Bind
        connect/disconnect to :meth:`on_connected_to_printer` /
        :meth:`on_disconnected_from_printer` (or a subclass override). Only
        consulted when :meth:`_connection_event_bus` is not ``None``."""
        return ()

    def _wire_connection_events(self) -> None:
        """Apply :meth:`_connection_event_bindings` to the component event bus.
        Subclasses call this from ``__init__`` once the component exists."""
        bus = self._connection_event_bus()
        if bus is None:
            return
        for event, handler in self._connection_event_bindings():
            bus.on(event, handler)

    # -- connect/disconnect defaults -----------------------------------------

    def on_connected_to_printer(self, *_args) -> None:
        """The device connection came up: mark active and (re)resolve the
        camera. Subclasses override to add device startup commands."""
        self.active = True
        self.logger.info("Connected to printer")
        self.update_camera_uri()

    def on_disconnected_from_printer(self, *_args) -> None:
        """The device connection went away: mark inactive and drop the camera."""
        self.active = False
        self.logger.info("Disconnected from printer")
        self.camera_uri = None

    # -- status translation (reads/uses self.printer; no device specifics) ---

    @staticmethod
    def _guard_cancelling(
        current: Optional[PrinterStatus], new: PrinterStatus
    ) -> PrinterStatus:
        if current == PrinterStatus.CANCELLING and new == PrinterStatus.PRINTING:
            return PrinterStatus.CANCELLING
        return new

    @staticmethod
    def _guard_pausing(
        current: Optional[PrinterStatus], new: PrinterStatus
    ) -> PrinterStatus:
        if current == PrinterStatus.PAUSING and new == PrinterStatus.PRINTING:
            return PrinterStatus.PAUSING
        return new

    @staticmethod
    def _guard_downloading(new: PrinterStatus, is_downloading: bool) -> PrinterStatus:
        if new == PrinterStatus.OPERATIONAL and is_downloading:
            return PrinterStatus.DOWNLOADING
        return new

    def hold_status_on_cancel(self, new_status: PrinterStatus) -> PrinterStatus:
        """Keep ``CANCELLING`` until the firmware confirms a non-printing state."""
        return self._guard_cancelling(self.printer.status, new_status)

    def hold_status_on_pause(self, new_status: PrinterStatus) -> PrinterStatus:
        """Keep ``PAUSING`` until the firmware acknowledges the pause."""
        return self._guard_pausing(self.printer.status, new_status)

    def hold_status_while_downloading(
        self, new_status: PrinterStatus, downloading: bool
    ) -> PrinterStatus:
        """Hold ``DOWNLOADING`` so an idle firmware state doesn't flap to
        OPERATIONAL while a print is still being prepared client-side."""
        return self._guard_downloading(new_status, downloading)

    def is_job_start(self, new_status: PrinterStatus) -> bool:
        """True on the edge from a non-printing state into PRINTING/PAUSED.

        (A printer may start straight into PAUSED if an error is detected at once.)
        """
        return (
            new_status in (PrinterStatus.PRINTING, PrinterStatus.PAUSED)
            and self.printer.status is not None
            and not self.printer.is_printing()
        )

    def is_job_finish(self, new_status: PrinterStatus) -> bool:
        """True on the edge from a printing state back to OPERATIONAL."""
        return (
            new_status == PrinterStatus.OPERATIONAL
            and self.printer.status is not None
            and self.printer.is_printing()
        )

    # -- status application --------------------------------------------------

    def apply_status(
        self,
        new_status: PrinterStatus,
        *,
        downloading: bool = False,
        guard_pause: bool = False,
        apply: bool = True,
    ) -> PrinterStatus:
        """Run the canonical guard -> edge -> apply pipeline shared by all devices.

        ``new_status`` is the device-state mapping (step 1, subclass-owned).
        Then: hold ``CANCELLING`` (always), hold ``PAUSING`` (when ``guard_pause``;
        some devices don't), hold ``DOWNLOADING`` (when ``downloading``); dispatch
        the job-start / job-finish / in-progress edge to the subclass hook; and
        apply the status unless ``apply`` is ``False`` (a device with a "state
        unknown" sentinel passes ``apply=False`` and the edges still run --
        matching the existing behaviour where edges fire but the assignment is
        suppressed).

        Returns the guarded status.
        """
        new_status = self.hold_status_on_cancel(new_status)
        if guard_pause:
            new_status = self.hold_status_on_pause(new_status)
        new_status = self.hold_status_while_downloading(new_status, downloading)

        if self.is_job_start(new_status):
            self.printer.job_info.started = True
            self._on_job_start(new_status)
        elif self.is_job_finish(new_status):
            self._on_job_finish(new_status)
        elif new_status == PrinterStatus.PRINTING:
            self._on_job_progress(new_status)

        if apply:
            self.printer.status = new_status
        return new_status

    def _on_job_start(self, new_status: PrinterStatus) -> None:
        """Capture device job-start fields (filename/progress/layer/time, reprint
        detection). ``job_info.started`` is already set by :meth:`apply_status`."""

    def _on_job_finish(self, new_status: PrinterStatus) -> None:
        """Classify the finish: set exactly one of ``job_info.finished`` /
        ``.cancelled`` / ``.failed`` from the device's terminal state."""

    def _on_job_progress(self, new_status: PrinterStatus) -> None:
        """Update in-progress job fields (progress/layer/time). Default no-op."""

    # -- camera --------------------------------------------------------------

    def _init_camera(self, **kwargs) -> None:
        """Initialise the camera mixin with the device-tuned cache constants."""
        self.initialize_camera_mixin(
            pause_timeout=self.camera_pause_timeout,
            max_cache_age=self.camera_max_cache_age,
            **kwargs,
        )

    def _resolve_camera_uri(self) -> Optional["URL"]:
        """Return the current camera URI, or ``None`` if unavailable. Device hook
        (e.g. a custom URL from config, else the connection component's probe)."""
        return None

    def update_camera_uri(self) -> None:
        """Resolve the device camera URI and hand it to the mixin, logging the
        outcome (and redacting any password) the way every integration did by
        hand."""
        camera_uri = self._resolve_camera_uri()

        if not camera_uri:
            self.logger.debug("No camera URI available")
            return

        if camera_uri.password:
            redacted_uri = camera_uri.with_password("*" * len(camera_uri.password))
        else:
            redacted_uri = camera_uri

        try:
            self.camera_uri = camera_uri

            # NB: if/elif rather than match — the library supports Python 3.9.
            if self.camera_status == "new":
                self.logger.debug("Set camera URI to %s", redacted_uri)
            elif self.camera_status == "err":
                self.logger.warning("Failed to set camera URI to %s", redacted_uri)

        except Exception as e:
            self.logger.warning(
                "Failed to set camera URI to %s", redacted_uri, exc_info=e
            )

    # -- host telemetry ------------------------------------------------------

    def update_host_telemetry(self) -> None:
        """Populate the host CPU/memory sensors from the machine running the client."""
        usage = _host_usage()
        self.printer.cpu_info.usage = usage.get("usage")
        self.printer.cpu_info.temp = usage.get("temp")
        self.printer.cpu_info.memory = usage.get("memory")
