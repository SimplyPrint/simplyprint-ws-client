"""Brand-neutral network facts for discovery, onboarding, and diagnostics.

The scanner owns cheap, reusable facts such as "is TCP port 80 open on this
host?" Brand modules own protocol-specific fingerprints and response parsing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from simplyprint_ws_client._compat import StrEnum
from typing import Awaitable, Callable, Mapping, Optional

from simplyprint_ws_client.integration.discovery.device import JsonValue
from simplyprint_ws_client.integration.discovery.spec import NetworkServiceSpec

DiagnosticDetail = Mapping[str, JsonValue]


class DiagnosticReason(StrEnum):
    """Why a host diagnostic landed where it did (stable, machine-readable).

    A ``str`` subclass, so members compare and serialize byte-identically to the
    plain reason strings they replace.
    """

    REQUIRED_SERVICE_CLOSED = "required_service_closed"
    REQUIRED_SERVICES_OPEN = "required_services_open"
    PROBE_ERROR = "probe_error"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    MATCHED = "matched"


@dataclass(frozen=True)
class PortCheckResult:
    host: str
    service_id: str
    transport: str
    port: int
    open: bool
    required: bool
    label: Optional[str] = None
    purposes: tuple[str, ...] = ()
    error: Optional[str] = None
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class DiagnosticCheckResult:
    """One machine-readable diagnostic check row."""

    id: str
    status: str
    code: str
    label: str
    message: str
    subject: Optional[str] = None
    detail: DiagnosticDetail = field(default_factory=dict)


@dataclass(frozen=True)
class HostProbeContext:
    """Cached network facts for one host, handed to an optional brand probe."""

    host: str
    services: Mapping[str, PortCheckResult]

    def service(self, service_id: str) -> Optional[PortCheckResult]:
        return self.services.get(service_id)

    def service_open(self, service_id: str) -> bool:
        result = self.service(service_id)
        return bool(result and result.open)

    @property
    def required_services_open(self) -> bool:
        required = [result for result in self.services.values() if result.required]
        return all(result.open for result in required)


@dataclass(frozen=True)
class HostDiagnostic:
    """A user/debug-facing explanation of what happened for one host."""

    brand: str
    host: str
    status: str
    services: tuple[PortCheckResult, ...]
    checks: tuple[DiagnosticCheckResult, ...]
    matched: Optional[bool]
    reason: str
    message: str

    @property
    def required_services_open(self) -> bool:
        required = [result for result in self.services if result.required]
        return all(result.open for result in required)


def service_diagnostic_check(result: PortCheckResult) -> DiagnosticCheckResult:
    """Convert a raw service check into a stable diagnostic check row."""
    label = result.label or f"{result.transport.upper()} {result.port}"
    subject = f"{result.transport}:{result.port}"

    if result.open:
        status = "ok"
        code = "service_open"
        message = f"{label} is reachable"
    elif result.required:
        status = "error"
        code = DiagnosticReason.REQUIRED_SERVICE_CLOSED
        message = f"{label} is not reachable"
    else:
        status = "warning"
        code = "optional_service_closed"
        message = f"Optional {label} is not reachable"

    if result.error:
        code = (
            "service_check_error" if result.required else "optional_service_check_error"
        )
        message = f"{label} check failed: {result.error}"

    return DiagnosticCheckResult(
        id=f"service:{result.service_id}",
        status=status,
        code=code,
        label=label,
        message=message,
        subject=subject,
        detail={
            "host": result.host,
            "service_id": result.service_id,
            "transport": result.transport,
            "port": result.port,
            "required": result.required,
            "purposes": list(result.purposes),
            "open": result.open,
            "elapsed_ms": result.elapsed_ms,
            **({"error": result.error} if result.error else {}),
        },
    )


def diagnostic_status(checks: tuple[DiagnosticCheckResult, ...]) -> str:
    """Roll check rows up to one coarse status."""
    statuses = {check.status for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    if "unknown" in statuses:
        return "unknown"
    return "ok"


PortChecker = Callable[[str, int, float], Awaitable[bool]]


async def tcp_port_open(host: str, port: int, timeout: float) -> bool:
    """Return true when a TCP connection to ``host:port`` can be opened."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout
        )
    except (OSError, asyncio.TimeoutError):
        return False

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True


class NetworkScanContext:
    """Per-scan cache for reusable network checks.

    Concurrent brand scans share this object so the same ``host:port`` check is
    issued once, even when multiple brands care about that service.
    """

    def __init__(self, port_checker: PortChecker = tcp_port_open) -> None:
        self._port_checker = port_checker
        self._lock = asyncio.Lock()
        self._port_results: dict[
            tuple[str, int], tuple[bool, Optional[str], float]
        ] = {}
        self._port_tasks: dict[
            tuple[str, int], asyncio.Task[tuple[bool, Optional[str], float]]
        ] = {}

    async def check_service(
        self, host: str, service: NetworkServiceSpec, timeout: float
    ) -> PortCheckResult:
        open_, error, elapsed_ms = await self._check_port(host, service.port, timeout)
        return PortCheckResult(
            host=host,
            service_id=service.id,
            transport=service.transport,
            port=service.port,
            open=open_,
            required=service.required,
            label=service.label,
            purposes=service.purposes,
            error=error,
            elapsed_ms=elapsed_ms,
        )

    async def host_context(
        self, host: str, services: tuple[NetworkServiceSpec, ...], timeout: float
    ) -> HostProbeContext:
        results = await asyncio.gather(
            *(self.check_service(host, service, timeout) for service in services)
        )
        return HostProbeContext(
            host=host,
            services={result.service_id: result for result in results},
        )

    async def _check_port(
        self, host: str, port: int, timeout: float
    ) -> tuple[bool, Optional[str], float]:
        key = (host, port)
        async with self._lock:
            if key in self._port_results:
                return self._port_results[key]

            task = self._port_tasks.get(key)
            if task is None:
                task = asyncio.create_task(self._run_port_check(host, port, timeout))
                self._port_tasks[key] = task

        result = await task

        async with self._lock:
            self._port_results[key] = result
            if self._port_tasks.get(key) is task:
                self._port_tasks.pop(key, None)

        return result

    async def _run_port_check(
        self, host: str, port: int, timeout: float
    ) -> tuple[bool, Optional[str], float]:
        started = time.monotonic()
        try:
            open_ = await self._port_checker(host, port, timeout)
            error = None
        except Exception as exc:
            open_ = False
            error = str(exc) or exc.__class__.__name__

        elapsed_ms = (time.monotonic() - started) * 1000
        return open_, error, elapsed_ms
