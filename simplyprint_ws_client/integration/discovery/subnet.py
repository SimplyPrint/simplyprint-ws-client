"""Generic bounded-concurrency subnet scanner.

Runs one brand's :class:`SubnetScanSpec` over the local subnet(s): enumerate
hosts, optionally gate each on a cheap TCP port check, then run the brand probe
on the survivors -- all under an ``asyncio.Semaphore`` so the scan never opens
more than ``spec.concurrency`` connections at once (replacing the per-brand
unbounded ``gather`` of ~254 tasks and the 256-thread pool). On-demand only.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from simplyprint_ws_client.common.asyncio.bounded_dispatch import map_concurrently
from simplyprint_ws_client.integration.discovery import netif
from simplyprint_ws_client.integration.discovery.network import (
    DiagnosticCheckResult,
    DiagnosticReason,
    HostDiagnostic,
    HostProbeContext,
    NetworkScanContext,
    diagnostic_status,
    service_diagnostic_check,
)
from simplyprint_ws_client.integration.discovery.spec import SubnetScanSpec


class SubnetScanBackend:
    def __init__(
        self, spec: SubnetScanSpec, network_context: Optional[NetworkScanContext] = None
    ) -> None:
        self.spec = spec
        self.logger = logging.getLogger("discovery")
        self.network_context = network_context or NetworkScanContext()

    async def _host_context(self, host: str) -> HostProbeContext:
        return await self.network_context.host_context(
            host, self.spec.services, self.spec.gate_timeout
        )

    async def _probe(self, host: str, context: HostProbeContext, timeout: float):
        if self.spec.context_probe is not None:
            return await asyncio.wait_for(self.spec.context_probe(context), timeout)
        return await asyncio.wait_for(self.spec.probe(host), timeout)

    def _closed_port_message(self, host: str, context: HostProbeContext) -> str:
        closed = [
            result
            for result in context.services.values()
            if result.required and not result.open
        ]
        if not closed:
            return f"{host} did not match {self.spec.brand}"

        labels = [
            result.label or f"{result.transport}:{result.port}" for result in closed
        ]
        return (
            f"{host} is not answering on the expected "
            f"{self.spec.brand} service port(s): {', '.join(labels)}"
        )

    def _service_checks(self, services: tuple) -> tuple[DiagnosticCheckResult, ...]:
        return tuple(service_diagnostic_check(service) for service in services)

    def _diagnostic(
        self,
        host: str,
        services: tuple,
        checks: tuple[DiagnosticCheckResult, ...],
        matched: bool | None,
        reason: str,
        message: str,
    ) -> HostDiagnostic:
        return HostDiagnostic(
            brand=self.spec.brand,
            host=host,
            status=diagnostic_status(checks),
            services=services,
            checks=checks,
            matched=matched,
            reason=reason,
            message=message,
        )

    async def diagnose(
        self, host: str, timeout: float = 5.0, *, run_probe: bool = True
    ) -> HostDiagnostic:
        """Check one host and return diagnostics suitable for onboarding/debug UI."""
        context = await self._host_context(host)
        services = tuple(context.services.values())
        service_checks = self._service_checks(services)

        if not context.required_services_open:
            return self._diagnostic(
                host,
                services,
                service_checks,
                False,
                DiagnosticReason.REQUIRED_SERVICE_CLOSED,
                self._closed_port_message(host, context),
            )

        if not run_probe:
            return self._diagnostic(
                host,
                services,
                service_checks,
                None,
                DiagnosticReason.REQUIRED_SERVICES_OPEN,
                f"{host} is answering on the expected {self.spec.brand} port(s)",
            )

        try:
            record = await self._probe(host, context, timeout)
        except asyncio.TimeoutError:
            probe_check = DiagnosticCheckResult(
                id="probe:fingerprint",
                status="error",
                code=DiagnosticReason.PROBE_ERROR,
                label="Printer fingerprint",
                message="Fingerprint probe timed out",
                subject=host,
                detail={"brand": self.spec.brand, "timeout": timeout},
            )
            return self._diagnostic(
                host,
                services,
                (*service_checks, probe_check),
                False,
                DiagnosticReason.PROBE_ERROR,
                f"{host} answered on the expected port(s), but the probe timed out",
            )
        except Exception:
            self.logger.debug(
                "discovery diagnostic probe failed for %s at %s",
                self.spec.brand,
                host,
                exc_info=True,
            )
            probe_check = DiagnosticCheckResult(
                id="probe:fingerprint",
                status="error",
                code=DiagnosticReason.PROBE_ERROR,
                label="Printer fingerprint",
                message="Fingerprint probe failed",
                subject=host,
                detail={"brand": self.spec.brand},
            )
            return self._diagnostic(
                host,
                services,
                (*service_checks, probe_check),
                False,
                DiagnosticReason.PROBE_ERROR,
                f"{host} answered on the expected port(s), but the probe failed",
            )

        if record is None:
            probe_check = DiagnosticCheckResult(
                id="probe:fingerprint",
                status="error",
                code=DiagnosticReason.FINGERPRINT_MISMATCH,
                label="Printer fingerprint",
                message=f"Response did not match {self.spec.brand}",
                subject=host,
                detail={"brand": self.spec.brand},
            )
            return self._diagnostic(
                host,
                services,
                (*service_checks, probe_check),
                False,
                DiagnosticReason.FINGERPRINT_MISMATCH,
                f"{host} answered, but did not look like {self.spec.brand}",
            )

        probe_check = DiagnosticCheckResult(
            id="probe:fingerprint",
            status="ok",
            code="fingerprint_matched",
            label="Printer fingerprint",
            message=f"Response matched {self.spec.brand}",
            subject=host,
            detail={"brand": self.spec.brand},
        )
        return self._diagnostic(
            host,
            services,
            (*service_checks, probe_check),
            True,
            DiagnosticReason.MATCHED,
            f"{host} matched {self.spec.brand}",
        )

    async def scan(
        self, timeout: float = 5.0, hosts: Optional[List[str]] = None
    ) -> list:
        """Probe every candidate host (bounded), return deduplicated records."""
        if hosts is None:
            hosts = netif.scan_hosts()

        async def probe_one(host: str):
            context = await self._host_context(host)
            if not context.required_services_open:
                return None
            try:
                return await self._probe(host, context, timeout)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                self.logger.debug(
                    "discovery probe timed out for %s at %s after %.1fs",
                    self.spec.brand,
                    host,
                    timeout,
                )
                return None
            except Exception:
                self.logger.debug(
                    "discovery probe failed for %s at %s",
                    self.spec.brand,
                    host,
                    exc_info=True,
                )
                return None

        results = await map_concurrently(
            probe_one,
            hosts,
            concurrency=self.spec.concurrency,
        )

        records: dict = {}
        for record in results:
            if record is not None:
                records[self.spec.key(record)] = record

        return list(records.values())
