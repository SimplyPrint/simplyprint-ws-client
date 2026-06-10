"""The headline base for authoring a printer integration, as one business object.

A printer client's job is to boil a device's model data into a
:class:`~simplyprint_ws_client.cloud.state.PrinterState`. The *machinery* around
that -- driving the device-driver lifecycle, ticking host telemetry + ping,
hopping device edges onto the client loop, reducing a device-mapped status
through the shared cancel/pause/download holds and job-start/finish edges, and
resolving the camera URI -- is identical across every brand; only the
device-data extraction differs.

This used to be ~80 near-identical lines re-implemented in every integration's
``printer.py``. It now lives here as :class:`PrinterClient`; a brand subclass
supplies only what is genuinely device-specific via the hooks at the bottom of
the class:

* ``device_drivers``        -- declare how the device is reached (links/poller)
* ``on_device_connected`` / ``on_device_disconnected`` / ``on_device_message``
                            -- the device edges, delivered on the client loop
* ``poll_device`` / ``refresh_device_credentials``
                            -- the polling cycle and the re-auth seam
* ``_resolve_camera_uri``   -- the device's current camera URL (or None)
* ``on_job_start`` / ``on_job_finish`` / ``on_job_progress``
                            -- capture/classify the device's job fields (each
                               receives a :class:`JobEdge` carrying ``raw``)
* ``_stop_connection``      -- extra device teardown after drivers stop
* ``_tick_progress``        -- drive a client-side progress shim each tick

The subtle part is :meth:`apply_status`: the guard -> edge -> apply pipeline is
owned here whole, because the *transition semantics* are identical across
devices; only the device-state mapping and the per-edge field capture differ,
and those are delegated to the hooks. A subclass maps its device status to a
:class:`PrinterStatus` then calls :meth:`apply_status` -- it never re-implements
the holds or the start/finish detection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Coroutine,
    Generic,
    Iterable,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

from simplyprint_ws_client.cloud.client import ClientConfigChangedEvent
from simplyprint_ws_client.cloud.config import PrinterConfig
from simplyprint_ws_client.cloud.state import PrinterStatus
from simplyprint_ws_client.cloud.protocol.messages import (
    CancelDemandData,
    GcodeDemandData,
    PauseDemandData,
    PluginInstallDemandData,
    ResumeDemandData,
    SkipObjectsDemandData,
    SystemRestartDemandData,
    SystemShutdownDemandData,
    TerminalDemandData,
)
from simplyprint_ws_client.device.camera.mixin import ClientCameraMixin
from simplyprint_ws_client.common.hardware.physical_machine import PhysicalMachine

if TYPE_CHECKING:
    import logging

    from yarl import URL

    from simplyprint_ws_client.device.discovery.device import DiscoveredDevice
    from simplyprint_ws_client.integration.driver import DeviceDriver

TConfig = TypeVar("TConfig", bound=PrinterConfig)

#: Host usage is read at most this often, shared across every client, so the
#: delta-based ``psutil.cpu_percent`` isn't reset by every client every tick.
_HOST_USAGE_MIN_INTERVAL = 5.0
_host_usage_snapshot: dict = {}
_host_usage_read_at: float = 0.0


async def _host_usage() -> dict:
    """Return a process-wide, throttled snapshot of host CPU/memory usage.

    The refresh reads sysfs/proc via ``psutil`` and is offloaded to a worker
    thread so it never blocks the loop; a fresh-enough snapshot is returned
    immediately without a thread hop.
    """
    global _host_usage_snapshot, _host_usage_read_at
    now = time.monotonic()
    if (
        not _host_usage_snapshot
        or now - _host_usage_read_at >= _HOST_USAGE_MIN_INTERVAL
    ):
        _host_usage_snapshot = await asyncio.to_thread(PhysicalMachine.get_usage)
        _host_usage_read_at = now
    return _host_usage_snapshot


@dataclass(frozen=True)
class JobEdge:
    """One status transition handed to the job-edge hooks.

    ``raw`` is whatever brand payload the ``apply_status`` caller passed along --
    the device's print dict, raw state enum, ... -- so a hook never has to read
    smuggled instance attributes to see what produced the edge.
    """

    new_status: PrinterStatus
    previous_status: Optional[PrinterStatus]
    raw: object = None


class AppUpdater(Protocol):
    """The connector's self-update entry point, wired once by the integration.

    Updating the connector is an app-level concern, not a brand one, so the
    library handles the plugin-install demand generically and delegates the
    actual update here. The integration supplies an implementation (and sets
    :attr:`PrinterClient.app_updater`); brands never see it.
    """

    #: The connector's plugin name in the SimplyPrint demand. A demand naming a
    #: different plugin is ignored.
    plugin_name: str

    async def run(self, logger: "logging.Logger") -> None:
        """Perform (or decline) the connector self-update, logging the outcome."""
        ...


class PrinterClient(ClientCameraMixin[TConfig], Generic[TConfig]):
    """Common base for device printer clients.

    Subclasses own ``__init__`` (device info population, device-client
    construction) and the device-model -> :class:`PrinterState` mapping; they
    declare how the device is reached via :meth:`device_drivers`. Everything
    else (lifecycle, status application, camera resolution, host telemetry) is
    owned here and parameterised through the hooks below.
    """

    #: Camera mixin tuning consumed by :meth:`_init_camera`. Subclasses override
    #: the class attribute when their camera wants a different cache window.
    camera_pause_timeout: int = 10
    camera_max_cache_age: timedelta = timedelta(seconds=1)

    #: Whether the PAUSING hold applies: some firmwares keep reporting PRINTING
    #: until a pause lands, so the brand opts in at the class level (this used to
    #: be a per-call ``guard_pause=`` flag repeated at every call site).
    hold_pausing: ClassVar[bool] = False

    #: Connector self-updater, wired once by the integration (``None`` = updates
    #: not wired, so a plugin-install demand is a no-op). See :class:`AppUpdater`.
    app_updater: ClassVar[Optional[AppUpdater]] = None

    # init/halt are called once per halt + initially; tick every scheduling
    # slice; teardown once at final cleanup (see Client docstring).

    async def init(self) -> None:
        """Arm the device drivers."""
        for driver in self._device_drivers():
            driver.start()

    async def tick(self, _delta) -> None:
        """Host housekeeping every slice: progress shim, ambient sensor, host
        telemetry, the SimplyPrint heartbeat ping (each interval-gated), and the
        driver ensure-started sweep (a driver whose config wasn't ready at init
        retries here for free)."""
        self._tick_progress()
        self.printer.ambient_temperature.tick(self.printer)
        await self.update_host_telemetry()
        await self.send_ping()
        for driver in self._device_drivers():
            driver.ensure_started()

    async def halt(self) -> None:
        """Temporarily out of scheduling: suspend every driver."""
        for driver in self._device_drivers():
            driver.suspend()

    async def teardown(self) -> None:
        """Final cleanup: stop the drivers, then any extra device teardown."""
        for driver in self._device_drivers():
            driver.stop()
        await self._stop_connection()

    # -- device drivers: how this client reaches its physical printer --

    def device_drivers(self) -> Iterable["DeviceDriver"]:
        """Declare how this client reaches its device: zero or more drivers
        (:class:`~simplyprint_ws_client.integration.link.WsDeviceLink` /
        :class:`~simplyprint_ws_client.integration.link.MqttDeviceLink` /
        :class:`~simplyprint_ws_client.integration.poller.DevicePoller`).
        Called once; the base owns when they start/suspend/stop."""
        return ()

    def _device_drivers(self) -> Tuple["DeviceDriver", ...]:
        drivers = getattr(self, "_device_drivers_cache", None)
        if drivers is None:
            drivers = tuple(self.device_drivers())
            self._device_drivers_cache = drivers
        return drivers

    async def on_device_connected(self, driver: "DeviceDriver") -> None:
        """A driver reached the device: mark active and (re)resolve the camera.
        Override to add device startup commands (call ``await super()...``)."""
        self.active = True
        self.logger.info("Connected to printer")
        self.update_camera_uri()

    async def on_device_disconnected(
        self, driver: "DeviceDriver", reason: Optional[object] = None
    ) -> None:
        """A driver lost the device: mark inactive and drop the camera."""
        self.active = False
        self.logger.info("Disconnected from printer")
        self.clear_camera_uri()

    async def on_device_message(self, message: object, driver: "DeviceDriver") -> None:
        """One inbound device message (a link's frame payload / an MqttMessage).
        Push brands override; the default ignores it."""

    async def poll_device(self) -> None:
        """One poll cycle for request/response devices, driven by a
        :class:`~simplyprint_ws_client.integration.poller.DevicePoller`."""
        raise NotImplementedError

    async def refresh_device_credentials(self, driver: "DeviceDriver") -> bool:
        """Re-mint expired device credentials (update the config) and return
        ``True`` to have the driver restart with them. Default: cannot refresh."""
        return False

    # -- the demand surface (typed, discoverable; autowired by name) ---------
    # Override what your device supports; each default is a no-op (logged at
    # debug). An override registers exactly once -- autowire resolves one
    # attribute per name through the MRO -- and may use any tolerated arity.

    def _unhandled_demand(self, name: str) -> None:
        self.logger.debug("demand %s received but not implemented", name)

    async def on_pause(self, data: PauseDemandData) -> None:
        """SimplyPrint asks the printer to pause the running job."""
        self._unhandled_demand("pause")

    async def on_resume(self, data: ResumeDemandData) -> None:
        """SimplyPrint asks the printer to resume a paused job."""
        self._unhandled_demand("resume")

    async def on_cancel(self, data: CancelDemandData) -> None:
        """SimplyPrint asks the printer to cancel the running job."""
        self._unhandled_demand("cancel")

    async def on_gcode(self, data: GcodeDemandData) -> None:
        """SimplyPrint sends gcode lines (``data.list``) to run on the device."""
        self._unhandled_demand("gcode")

    async def on_terminal(self, data: TerminalDemandData) -> None:
        """SimplyPrint toggles terminal/console streaming for this printer."""
        self._unhandled_demand("terminal")

    async def on_skip_objects(self, data: SkipObjectsDemandData) -> None:
        """SimplyPrint asks the printer to skip printing named objects."""
        self._unhandled_demand("skip_objects")

    async def on_system_restart(self, data: SystemRestartDemandData) -> None:
        """Restart the machine this client runs on. Opt-in (the old
        PhysicalClient auto-restarted): override and call
        ``PhysicalMachine.restart()`` when this client IS the host."""
        self._unhandled_demand("system_restart")

    async def on_system_shutdown(self, data: SystemShutdownDemandData) -> None:
        """Shut down the machine this client runs on. Opt-in, like
        :meth:`on_system_restart` (``PhysicalMachine.shutdown()``)."""
        self._unhandled_demand("system_shutdown")

    async def _stop_connection(self) -> None:
        """Extra device teardown after the drivers stop (close an HTTP session,
        send a goodbye). Default no-op."""

    def submit_to_loop(
        self, coro: Coroutine[Any, Any, Any]
    ) -> "concurrent.futures.Future":
        """Schedule ``coro`` on this client's event loop from another thread.

        The one sanctioned way for a device callback running on a connection or
        worker thread to fire async work on the client's loop -- a thin, named
        wrapper over ``run_coroutine_threadsafe`` so brands don't reach across
        threads themselves. Returns the future so a caller can wait if needed.
        """
        return asyncio.run_coroutine_threadsafe(coro, self.event_loop)

    def _tick_progress(self) -> None:
        """Drive a client-side progress shim each tick. Default no-op; devices
        with a fake-progress shim override to tick it."""

    def apply_discovered(self, device: "DiscoveredDevice") -> bool:
        """A device re-announced itself on the LAN: if it is *this* printer, let the
        brand refresh the config's mutable network facts (its address, ...).

        The single re-discovery seam shared by every brand, replacing each brand's
        bespoke "did my printer change IP" handler. Correlation is by stable
        hardware identity (the same rule the add-time reconciler uses), so a
        printer that moved to a new DHCP address is followed rather than orphaned;
        a device with no stable identity can't be correlated across an IP change,
        so it is ignored. When the brand reports a change, the config-changed event
        fires (which may reconnect the client). Returns whether anything changed.
        """
        if not self._is_same_device(device):
            return False
        if not self._apply_discovered(device):
            return False
        self.event_bus.emit_sync(ClientConfigChangedEvent)
        return True

    def _is_same_device(self, device: "DiscoveredDevice") -> bool:
        """True when ``device`` is this client's printer, by hardware identity."""
        from simplyprint_ws_client.device.discovery.reconcile import (
            config_hardware_id,
            device_hardware_id,
        )

        hardware_id = device_hardware_id(device.host, device.serial, device.extra)
        return hardware_id is not None and hardware_id == config_hardware_id(
            self.config
        )

    def _apply_discovered(self, device: "DiscoveredDevice") -> bool:
        """Brand hook: update this config's mutable network fields from a
        re-announced ``device`` (already confirmed to be this printer). Return
        whether anything changed. Default no-op for devices without passive
        re-discovery; override to follow the device's ``host``/address."""
        return False

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

    def apply_status(
        self,
        new_status: PrinterStatus,
        *,
        raw: object = None,
        downloading: bool = False,
        apply: bool = True,
    ) -> PrinterStatus:
        """Run the canonical guard -> edge -> apply pipeline shared by all devices.

        ``new_status`` is the device-state mapping (step 1, subclass-owned);
        ``raw`` is the brand payload that produced it, carried to the edge hooks
        on the :class:`JobEdge`. Then: hold ``CANCELLING`` (always), hold
        ``PAUSING`` (when the class opts in via :attr:`hold_pausing`), hold
        ``DOWNLOADING`` (when ``downloading``); dispatch the job-start /
        job-finish / in-progress edge to the subclass hook; and apply the status
        unless ``apply`` is ``False`` (a device with a "state unknown" sentinel
        passes ``apply=False`` and the edges still run -- matching the existing
        behaviour where edges fire but the assignment is suppressed).

        Returns the guarded status.
        """
        new_status = self.hold_status_on_cancel(new_status)
        if self.hold_pausing:
            new_status = self.hold_status_on_pause(new_status)
        new_status = self.hold_status_while_downloading(new_status, downloading)

        edge = JobEdge(new_status, self.printer.status, raw)
        if self.is_job_start(new_status):
            self.printer.job_info.started = True
            self.on_job_start(edge)
        elif self.is_job_finish(new_status):
            self.on_job_finish(edge)
        elif new_status == PrinterStatus.PRINTING:
            self.on_job_progress(edge)

        if apply:
            self.printer.status = new_status
        return new_status

    def on_job_start(self, edge: JobEdge) -> None:
        """Capture device job-start fields (filename/progress/layer/time, reprint
        detection) from ``edge.raw``. ``job_info.started`` is already set by
        :meth:`apply_status`."""

    def on_job_finish(self, edge: JobEdge) -> None:
        """Classify the finish from ``edge.raw``: set exactly one of
        ``job_info.finished`` / ``.cancelled`` / ``.failed``."""

    def on_job_progress(self, edge: JobEdge) -> None:
        """Update in-progress job fields (progress/layer/time). Default no-op."""

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

    def clear_camera_uri(self) -> None:
        try:
            self.camera_uri = None
        except Exception as e:
            self.logger.warning("Failed to clear camera URI", exc_info=e)

    async def update_host_telemetry(self) -> None:
        """Populate the host CPU/memory sensors from the machine running the client."""
        usage = await _host_usage()
        self.printer.cpu_info.usage = usage.get("usage")
        self.printer.cpu_info.temp = usage.get("temp")
        self.printer.cpu_info.memory = usage.get("memory")

    async def on_plugin_install(self, event: PluginInstallDemandData) -> None:
        """Update the connector when SimplyPrint asks the brand to.

        Updating the connector is brand-agnostic, so the library owns the demand
        and delegates to :attr:`app_updater` (wired once by the integration);
        no brand reimplements this.
        """
        # Non-mutating read: two concurrent dispatches must not race on the
        # shared list (a pop() raced and could IndexError on the second).
        if not event.plugins:
            return
        plugin = event.plugins[0]

        updater = self.app_updater
        if updater is None:
            self.logger.debug(
                "Plugin install demand received, but no app updater is wired."
            )
            return

        if plugin.get("type") != "install" or plugin.get("name") != updater.plugin_name:
            self.logger.warning(
                "Plugin install demand received for %s, but it is not supported.",
                plugin,
            )
            return

        await updater.run(self.logger)
