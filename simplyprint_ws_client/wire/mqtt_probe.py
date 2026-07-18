"""One-shot MQTT diagnostics isolated from persistent application connections."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import yarl

from simplyprint_ws_client._compat import StrEnum
from simplyprint_ws_client.common.asyncio.event_loop_provider import EventLoopProvider
from simplyprint_ws_client.wire.errors import ErrorCode
from simplyprint_ws_client.wire.events import Connected, Disconnected, MessageReceived
from simplyprint_ws_client.wire.messages import MqttMessage
from simplyprint_ws_client.wire.mqtt import connect, pool_for
from simplyprint_ws_client.wire.options import ConnectionOptions, WireKeepalive
from simplyprint_ws_client.wire.policy import RetryPolicy
from simplyprint_ws_client.wire.pools import PoolRegistry
from simplyprint_ws_client.wire.transport import MqttTransport


class MqttProbeOutcome(StrEnum):
    VERIFIED = "verified"
    AUTH_FAILED = "auth_failed"
    CONNECT_FAILED = "connect_failed"
    CONNECT_TIMEOUT = "connect_timeout"
    PUBLISH_FAILED = "publish_failed"
    REPORT_TIMEOUT = "report_timeout"


@dataclass(frozen=True)
class MqttProbeResult:
    outcome: MqttProbeOutcome
    report: Optional[MqttMessage] = None
    messages_seen: int = 0
    reason: Optional[str] = None
    reason_code: Optional[ErrorCode] = None


_AUTH_FAILURE_REASON_CODES = frozenset((134, 135, 140))


async def probe_mqtt(
    url: Union[str, yarl.URL],
    *,
    connect_timeout: float,
    report_timeout: float,
    require_report: bool,
    initial_messages: Sequence[MqttMessage] = (),
) -> MqttProbeResult:
    """Connect, optionally publish, then await the first subscribed report.

    An optional report can capture identity; the lease always closes here.
    """

    provider = EventLoopProvider(asyncio.get_running_loop())
    keepalive = WireKeepalive(interval=20)
    registry: PoolRegistry[MqttTransport] = PoolRegistry()
    pool = pool_for(registry, provider, keepalive)
    options = ConnectionOptions(
        provider=provider,
        retry=RetryPolicy(),
        wire_keepalive=keepalive,
    )
    parsed_url = yarl.URL(url) if isinstance(url, str) else url
    try:
        lease = connect(parsed_url, pool=pool, options=options)
    except Exception as error:  # noqa: BLE001 -- a diagnostic returns wire failure
        await registry.close()
        return MqttProbeResult(MqttProbeOutcome.CONNECT_FAILED, reason=str(error))

    lifecycle: asyncio.Future[Union[Connected, Disconnected]] = (
        asyncio.get_running_loop().create_future()
    )
    report: asyncio.Future[MqttMessage] = asyncio.get_running_loop().create_future()

    def on_lifecycle(event: Union[Connected, Disconnected]) -> None:
        if not lifecycle.done():
            lifecycle.set_result(event)

    def on_message(event: MessageReceived) -> None:
        if isinstance(event.message, MqttMessage) and not report.done():
            report.set_result(event.message)

    lease.event_bus.on(Connected, on_lifecycle)
    lease.event_bus.on(Disconnected, on_lifecycle)
    lease.event_bus.on(MessageReceived, on_message)
    try:
        if lease.connected and not lifecycle.done():
            lifecycle.set_result(Connected(lease.generation))
        try:
            edge = await asyncio.wait_for(lifecycle, connect_timeout)
        except asyncio.TimeoutError:
            return MqttProbeResult(MqttProbeOutcome.CONNECT_TIMEOUT)

        if isinstance(edge, Disconnected):
            reason = str(edge.code) if edge.code is not None else None
            reason_code = edge.code.code if edge.code is not None else None
            outcome = (
                MqttProbeOutcome.AUTH_FAILED
                if reason_code in _AUTH_FAILURE_REASON_CODES
                else MqttProbeOutcome.CONNECT_FAILED
            )
            return MqttProbeResult(outcome, reason=reason, reason_code=reason_code)

        for message in initial_messages:
            try:
                await lease.send(message)
            except Exception as error:  # noqa: BLE001 -- preserve probe outcome
                return MqttProbeResult(
                    MqttProbeOutcome.PUBLISH_FAILED, reason=str(error)
                )

        try:
            received = await asyncio.wait_for(report, report_timeout)
        except asyncio.TimeoutError:
            outcome = (
                MqttProbeOutcome.REPORT_TIMEOUT
                if require_report
                else MqttProbeOutcome.VERIFIED
            )
            return MqttProbeResult(outcome)
        return MqttProbeResult(MqttProbeOutcome.VERIFIED, received, 1)
    finally:
        lease.event_bus.off(Connected, on_lifecycle)
        lease.event_bus.off(Disconnected, on_lifecycle)
        lease.event_bus.off(MessageReceived, on_message)
        await lease.close()
        await registry.close()
