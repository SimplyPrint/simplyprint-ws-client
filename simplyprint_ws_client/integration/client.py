"""The headline base for authoring a printer integration, as one business object.

A printer client's job is to boil a device's model data into a
:class:`~simplyprint_ws_client.core.state.PrinterState`. The *machinery* around
that -- driving the device-driver lifecycle, ticking host telemetry + ping,
hopping device edges onto the client loop, reducing a device-mapped status
through the shared cancel/pause/download holds and job-start/finish edges, and
resolving the camera URI -- is identical across every brand; only the
device-data extraction differs.

This used to be ~80 near-identical lines re-implemented in every integration's
``printer.py``. It now lives here as :class:`PrinterClient`; a brand constructor
attaches its concrete device drivers, then supplies only what is genuinely
device-specific via the hooks at the bottom of the class:

* ``on_device_connected`` / ``on_device_disconnected`` / ``on_device_message``
                            -- the device edges, delivered on the client loop
* ``poll_device`` / ``refresh_device_credentials``
                            -- the polling cycle and the re-auth seam
* ``resolve_camera_uri``    -- the device's current camera URL (or None)
* ``on_job_start`` / ``on_job_finish`` / ``on_job_progress``
                            -- capture/classify the device's job fields (each
                               receives a :class:`JobEdge` carrying ``raw``)
* ``stop_connection``       -- extra device teardown after drivers stop
* ``tick_progress``         -- drive a client-side progress shim each tick

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
    Optional,
    Protocol,
    TypeVar,
)

from typing_extensions import TypeVar as TypeVarWithDefault

from simplyprint_ws_client.core.client import Client, ClientConfigChangedEvent
from simplyprint_ws_client.core.client_context import ClientContext
from simplyprint_ws_client.core.config import PrinterConfig
from simplyprint_ws_client.core.state import FileProgressStateEnum, PrinterStatus
from simplyprint_ws_client.core.protocol.messages import (
    ApiRestartDemandData,
    CancelDemandData,
    ConnectedMsg,
    FileDemandData,
    FileProgressMsg,
    GcodeDemandData,
    PauseDemandData,
    PeripheralActionDemandData,
    PluginInstallDemandData,
    PrinterSettingsMsg,
    ResumeDemandData,
    ResolveNotificationDemandData,
    SendLogsDemandData,
    SetMaterialDataDemandData,
    SkipObjectsDemandData,
    StartPrintDemandData,
    StreamOffDemandData,
    StreamOnDemandData,
    SystemRestartDemandData,
    SystemShutdownDemandData,
    TerminalDemandData,
    WebcamSnapshotDemandData,
    WebcamTestDemandData,
)
from simplyprint_ws_client.core.protocol.models import DemandMsgType, ServerMsgType
from simplyprint_ws_client.integration.camera.controller import CameraController

if TYPE_CHECKING:
    import logging

    from yarl import URL

    from simplyprint_ws_client.integration.discovery.device import DiscoveredDevice
    from simplyprint_ws_client.integration.drivers import DeviceDriver
    from simplyprint_ws_client.integration.transfer import FileTransfer

TConfig = TypeVar("TConfig", bound=PrinterConfig)
#: The brand payload a :class:`JobEdge` carries; defaults to ``object`` so an
#: unparameterized ``JobEdge`` keeps working for existing subscribers.
TRaw = TypeVarWithDefault("TRaw", default=object)


@dataclass(frozen=True)
class JobEdge(Generic[TRaw]):
    """One status transition handed to the job-edge hooks.

    ``raw`` is whatever brand payload the ``apply_status`` caller passed along --
    the device's print dict, raw state enum, ... -- so a hook never has to read
    smuggled instance attributes to see what produced the edge. A brand that
    always passes one payload shape may annotate its hooks ``JobEdge[ThatShape]``.
    """

    new_status: PrinterStatus
    previous_status: Optional[PrinterStatus]
    raw: TRaw = None  # type: ignore[assignment]


class AppUpdater(Protocol):
    """The connector's self-update entry point, wired once by the integration.

    Updating the connector is an app-level concern, not a brand one, so the
    library handles the plugin-install demand generically and delegates the
    actual update here. The host injects one implementation through
    :class:`ClientContext`; brands never see it.
    """

    #: The connector's plugin name in the SimplyPrint demand. A demand naming a
    #: different plugin is ignored.
    plugin_name: str

    async def run(self, logger: "logging.Logger") -> None:
        """Perform (or decline) the connector self-update, logging the outcome."""
        ...


class OwnedResource(Protocol):
    """An async resource retained and closed by a printer client."""

    async def close(self) -> None:
        """Release all work owned by this resource."""
        ...


TOwnedResource = TypeVar("TOwnedResource", bound=OwnedResource)
TDeviceDriver = TypeVar("TDeviceDriver", bound="DeviceDriver")


class PrinterClient(Client[TConfig], Generic[TConfig]):
    """Common base for device printer clients.

    Subclasses own ``__init__`` (device info population, device-client
    construction) and the device-model -> :class:`PrinterState` mapping; they
    attach each concrete link or poller during construction. Everything else
    (lifecycle, status application, camera resolution, host telemetry) is owned
    here and parameterised through the hooks below.
    """

    #: Camera-controller tuning consumed during construction. Subclasses override
    #: the class attribute when their camera wants a different cache window.
    #: How long a continuous camera worker may sit unpolled before it pauses.
    #: Must comfortably exceed the cloud's stream-demand cadence (~15s between
    #: webcam_snapshot demands): at 10s the worker paused between every two
    #: demands and each frame paid a full worker respawn + camera connect
    #: (1-2s on RTSP) instead of ~0ms from the hot stream. ``stream_off``
    #: still pauses immediately; this is only the lost-demand safety net.
    camera_pause_timeout: int = 60
    camera_max_cache_age: timedelta = timedelta(seconds=1)

    #: Whether the PAUSING hold applies: some firmwares keep reporting PRINTING
    #: until a pause lands, so the brand opts in at the class level (this used to
    #: be a per-call ``guard_pause=`` flag repeated at every call site).
    hold_pausing: ClassVar[bool] = False

    #: Most devices lose their camera with their control link. Brands whose
    #: camera remains independently reachable opt out without replacing the
    #: shared disconnect/status projection.
    clear_camera_on_unreachable: ClassVar[bool] = True

    #: Maximum time a true link loss may retain a status that protects active
    #: physical work. Idle devices still project OFFLINE immediately.
    device_loss_grace: ClassVar[float] = 300.0

    # ``active`` is a pure allocation-policy knob ("represent this printer on
    # the SimplyPrint connection"), true for the client's whole membership.
    # Device liveness is NOT expressed by toggling it -- an unreachable device
    # is reported through ``printer.status`` (OFFLINE) while the printer stays
    # allocated, so SimplyPrint keeps the printer's context (setup codes,
    # logs, notifications) and the connection doesn't churn on a flaky wire.
    # The scheduler runs init and tick (and therefore the drivers) regardless
    # of this flag, because the drivers are what *detect* liveness.

    # init runs once at scheduler entry; tick every scheduling slice (active or
    # not); halt on SimplyPrint deallocation; teardown once at final cleanup
    # (see the Client lifecycle docstrings). Drivers run from init to teardown.

    def __init__(
        self,
        config: TConfig,
        *,
        context: ClientContext,
    ) -> None:
        super().__init__(config, context=context)
        self.app_updater = context.app_updater
        self.host_telemetry = context.host_telemetry
        self.camera = CameraController(
            printer=self.printer,
            logger=self.logger,
            event_loop_provider=self,
            send_stream=self.send,
            context=context,
            pause_timeout=self.camera_pause_timeout,
            max_cache_age=self.camera_max_cache_age,
        )
        self._owned_resources: list[OwnedResource] = []
        self._drivers: list["DeviceDriver"] = []
        self.file_transfer: Optional["FileTransfer"] = None
        self._register_printer_handlers()

    def _register_printer_handlers(self) -> None:
        """Install the author-facing printer routes in one visible table."""
        on = self.event_bus.on
        on(ServerMsgType.CONNECTED, self.on_connected)
        on(ServerMsgType.PRINTER_SETTINGS, self.on_printer_settings)
        on(DemandMsgType.FILE, self.on_file)
        on(DemandMsgType.START_PRINT, self.on_start_print)
        on(DemandMsgType.PAUSE, self.on_pause)
        on(DemandMsgType.RESUME, self.on_resume)
        on(DemandMsgType.CANCEL, self.on_cancel)
        on(DemandMsgType.GCODE, self.on_gcode)
        on(DemandMsgType.TERMINAL, self.on_terminal)
        on(DemandMsgType.SKIP_OBJECTS, self.on_skip_objects)
        on(DemandMsgType.SYSTEM_RESTART, self.on_system_restart)
        on(DemandMsgType.SYSTEM_SHUTDOWN, self.on_system_shutdown)
        on(DemandMsgType.API_RESTART, self.on_api_restart)
        on(DemandMsgType.PLUGIN_INSTALL, self.on_plugin_install)
        on(DemandMsgType.SEND_LOGS, self.on_send_logs)
        on(DemandMsgType.PERIPHERAL_ACTION, self.on_peripheral_action)
        on(DemandMsgType.SET_MATERIAL_DATA, self.on_set_material_data)
        on(DemandMsgType.RESOLVE_NOTIFICATION, self.on_resolve_notification)
        on(DemandMsgType.STREAM_ON, self.on_stream_on)
        on(DemandMsgType.STREAM_OFF, self.on_stream_off)
        on(DemandMsgType.TEST_WEBCAM, self.on_test_webcam)
        on(DemandMsgType.WEBCAM_SNAPSHOT, self.on_webcam_snapshot)

    def own(self, resource: TOwnedResource) -> TOwnedResource:
        """Retain an async resource for reverse-order teardown."""
        self._owned_resources.append(resource)
        return resource

    def attach_file_transfer(self, transfer: "FileTransfer") -> None:
        """Install the one FILE/START lifecycle for this printer."""
        if self.file_transfer is not None:
            raise RuntimeError("a file transfer is already attached")
        self.file_transfer = self.own(transfer)

    def attach_driver(self, driver: TDeviceDriver) -> TDeviceDriver:
        """Retain one concrete device link or poller for this client."""
        self._drivers.append(driver)
        return driver

    @property
    def drivers(self) -> tuple["DeviceDriver", ...]:
        """The drivers attached by the concrete printer constructor."""
        return tuple(self._drivers)

    async def report_job_error(self, job_id: int, message: str) -> None:
        """Report a superseded FILE operation without exposing ``send``.

        The transfer controller depends on this narrow callback instead of the
        client's message construction, dispatch flags, or outbound API.
        """
        await self.send(
            FileProgressMsg(
                data={
                    "state": FileProgressStateEnum.ERROR,
                    "job_id": job_id,
                    "message": message,
                }
            ),
            skip_dispatch=True,
        )

    async def init(self) -> None:
        """Arm the device drivers and keep them tracking the config."""
        self.event_bus.on(ClientConfigChangedEvent, self._on_config_changed_base)
        for driver in self._drivers:
            driver.start()

    def _on_config_changed_base(self) -> None:
        """The config changed (web edit, re-discovery, refreshed credentials):
        re-resolve the camera and let every driver decide whether its endpoint
        moved (URL-diff -> restart). Hops to the client loop, because config
        changes are emitted from web/worker threads too."""
        if not self.event_loop_is_running():
            return
        self.submit_to_loop(self._apply_config_change())

    async def _apply_config_change(self) -> None:
        self.update_camera_uri()
        for driver in self._drivers:
            driver.ensure_current()

    async def tick(self, _delta) -> None:
        """Host housekeeping every slice: progress shim, ambient sensor, host
        telemetry, the SimplyPrint heartbeat ping (interval-gated, only while
        added), and the driver ensure-started sweep (a driver whose config
        wasn't ready at init retries here for free)."""
        await self.project_device_reachability()
        self.tick_progress()
        self.printer.ambient_temperature.tick(self.printer)
        await self.update_host_telemetry()
        if self.is_added():
            await self.send_ping()
        for driver in self._drivers:
            driver.ensure_started()

    async def teardown(self) -> None:
        """Final cleanup: close owned work, then stop device-side resources."""
        while self._owned_resources:
            resource = self._owned_resources.pop()
            try:
                await resource.close()
            except Exception:  # noqa: BLE001 -- close every retained resource
                self.logger.warning("owned resource close failed", exc_info=True)
        await super().teardown()
        for driver in self._drivers:
            await driver.close()
        await self.camera.close()
        await self.stop_connection()

    async def on_device_connected(self, driver: "DeviceDriver") -> None:
        """A driver reached the device: ensure allocation and (re)resolve the
        camera. Override to add device startup commands (call
        ``await super()...``). The OFFLINE status is *not* cleared here -- the
        first real device report maps it through ``apply_status``."""
        self.active = True
        self.logger.info("Connected to printer")
        self.update_camera_uri()

    async def on_device_disconnected(
        self, driver: "DeviceDriver", reason: Optional[object] = None
    ) -> None:
        """Handle the true link edge immediately, without inferring job state.

        The driver separately projects an unprotected loss (or a protected
        loss whose bounded deadline expires) through
        :meth:`on_device_unreachable`.
        """
        self.logger.info(
            "Disconnected from printer%s", f" ({reason})" if reason else ""
        )

    def link_loss_is_protected(self) -> bool:
        """Whether an immediate OFFLINE would contradict active work."""
        return (
            self.printer.is_printing()
            or self.printer.status == PrinterStatus.DOWNLOADING
            or self.printer.file_progress.state == FileProgressStateEnum.DOWNLOADING
        )

    async def project_device_reachability(self, now: Optional[float] = None) -> bool:
        """Project all driver sessions into one printer liveness outcome.

        Drivers own observations; this client owns status policy. Any live
        driver keeps the printer reachable. When all observed paths are down,
        active print/transfer state is held until the most recent down edge's
        fixed grace expires. Returns whether OFFLINE was applied now.
        """
        from simplyprint_ws_client.integration.drivers import DeviceReachability

        sessions = tuple((driver, driver.session) for driver in self._drivers)
        if not sessions or any(
            session.reachability is DeviceReachability.UP for _, session in sessions
        ):
            return False
        if any(
            session.reachability is DeviceReachability.NEVER_SEEN
            for _, session in sessions
        ):
            return False
        down = tuple(
            (driver, session)
            for driver, session in sessions
            if session.reachability is DeviceReachability.DOWN
        )
        if not down:
            return False

        protected = self.link_loss_is_protected()
        if (
            self.printer.file_progress.state == FileProgressStateEnum.DOWNLOADING
            and not self.printer.is_printing()
        ):
            self.printer.status = PrinterStatus.DOWNLOADING
        latest_down = max(session.observed_at for _, session in down)
        if protected and (time.monotonic() if now is None else now) < (
            latest_down + self.device_loss_grace
        ):
            return False
        if self.printer.status == PrinterStatus.OFFLINE:
            return False
        self.printer.status = PrinterStatus.OFFLINE
        if self.clear_camera_on_unreachable:
            self.clear_camera_uri()
        return True

    async def on_device_message(self, message: object, driver: "DeviceDriver") -> None:
        """One inbound device message (a link's frame payload / an MqttMessage).
        Push brands override; the default ignores it."""

    async def poll_device(self) -> None:
        """One poll cycle for request/response devices, driven by a
        :class:`~simplyprint_ws_client.integration.drivers.DevicePoller`."""
        raise NotImplementedError

    async def refresh_device_credentials(self, driver: "DeviceDriver") -> bool:
        """Re-mint expired device credentials (update the config) and return
        ``True`` to have the driver restart with them. Default: cannot refresh."""
        return False

    # Override what your device supports. Each route above binds one public,
    # typed method exactly once; names and annotations never select events.

    def _unhandled_demand(self, name: str) -> None:
        self.logger.debug("demand %s received but not implemented", name)

    async def on_connected(self, _msg: ConnectedMsg) -> None:
        """SimplyPrint accepted this printer connection."""

    async def on_printer_settings(self, _msg: PrinterSettingsMsg) -> None:
        """React after the core has applied cloud-side printer settings."""

    async def on_stream_on(self, data: Optional[StreamOnDemandData] = None) -> None:
        """Start the composed camera stream.

        Brands may override to prepare device-side camera power, then call
        ``await super().on_stream_on(data)``.
        """
        await self.camera.stream_on(data)

    async def on_stream_off(self, data: Optional[StreamOffDemandData] = None) -> None:
        """Pause the composed camera stream and discard stream credit."""
        await self.camera.stream_off(data)

    async def on_test_webcam(self, data: Optional[WebcamTestDemandData] = None) -> None:
        """Capture one camera frame for the webcam test demand."""
        await self.camera.test_webcam(data)

    async def on_webcam_snapshot(
        self, data: Optional[WebcamSnapshotDemandData] = None
    ) -> None:
        """Admit one stream-frame credit or identified snapshot request."""
        await self.camera.snapshot(data)

    async def on_api_restart(self, _data: ApiRestartDemandData) -> None:
        self._unhandled_demand("api_restart")

    async def on_send_logs(self, _data: SendLogsDemandData) -> None:
        self._unhandled_demand("send_logs")

    async def on_peripheral_action(self, _data: PeripheralActionDemandData) -> None:
        self._unhandled_demand("peripheral_action")

    async def on_set_material_data(self, _data: SetMaterialDataDemandData) -> None:
        """Apply device-side material changes after core state is updated."""

    async def on_resolve_notification(
        self, _data: ResolveNotificationDemandData
    ) -> None:
        """React after the core resolves/responds to a notification."""

    async def on_file(self, data: FileDemandData) -> None:
        """Submit a file to the attached transfer lifecycle."""
        if self.file_transfer is None:
            self._unhandled_demand("file")
            return
        self.file_transfer.submit(data)

    async def on_start_print(
        self, _data: Optional[StartPrintDemandData] = None
    ) -> None:
        """Start the file staged by a prior FILE demand."""
        if self.file_transfer is None:
            self._unhandled_demand("start_print")
            return
        self.file_transfer.start_staged()

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

    async def stop_connection(self) -> None:
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

    def tick_progress(self) -> None:
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
        if not self.update_from_discovery(device):
            return False
        self.event_bus.emit_sync(ClientConfigChangedEvent)
        return True

    def _is_same_device(self, device: "DiscoveredDevice") -> bool:
        """True when ``device`` is this client's printer, by hardware identity."""
        from simplyprint_ws_client.integration.discovery.reconcile import (
            DeviceReconciler,
        )

        return DeviceReconciler.same_device(
            self.config, device, allow_address_match=False
        )

    def update_from_discovery(self, device: "DiscoveredDevice") -> bool:
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

    def resolve_camera_uri(self) -> Optional["URL"]:
        """Return the device's current camera URI, or ``None`` if unavailable.
        Device hook (the brand's probe); a user-set ``custom_webcam_url`` wins
        before this is even consulted (see :meth:`update_camera_uri`)."""
        return None

    def update_camera_uri(self) -> None:
        """Resolve the camera URI and hand it to the controller, logging the
        outcome (and redacting any password) the way every integration did by
        hand.

        A user-supplied ``custom_webcam_url`` on the config takes precedence
        over the brand's own camera resolution -- every printer supports a
        custom webcam, with zero brand code.
        """
        custom = self.config.custom_webcam_url
        if custom:
            from yarl import URL as _URL

            camera_uri = _URL(custom)
        else:
            camera_uri = self.resolve_camera_uri()

        if not camera_uri:
            self.logger.debug("No camera URI available")
            self.clear_camera_uri()
            return

        if camera_uri.password:
            redacted_uri = camera_uri.with_password("*" * len(camera_uri.password))
        else:
            redacted_uri = camera_uri

        try:
            if self.camera.set_uri(camera_uri):
                self.logger.debug("Set camera URI to %s", redacted_uri)
            else:
                self.logger.warning("Failed to set camera URI to %s", redacted_uri)

        except Exception as e:
            self.logger.warning(
                "Failed to set camera URI to %s", redacted_uri, exc_info=e
            )

    def clear_camera_uri(self) -> None:
        try:
            self.camera.set_uri(None)
        except Exception as e:
            self.logger.warning("Failed to clear camera URI", exc_info=e)

    async def update_host_telemetry(self) -> None:
        """Populate the host CPU/memory sensors from the machine running the client."""
        if self.host_telemetry is None:
            return
        usage = await self.host_telemetry()
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
