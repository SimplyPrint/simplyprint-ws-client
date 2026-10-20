"""One MQTT connection's bytes on the event loop: TCP, optional TLS, packets.

:class:`MqttStream` is an ``asyncio.Protocol`` over a plain TCP transport.
It does TLS itself with :class:`ssl.SSLObject` over memory BIOs (the public
``ssl`` API, the same on every supported Python), and splits MQTT packets
straight out of the bytes it receives.

Why not ``asyncio.open_connection(ssl=...)``: asyncio's TLS layer allocates a
fixed 256 KiB read buffer for every connection, which dominates the memory of
a host running hundreds of printer links. Here every buffer grows only to what
the connection actually carries.

Everything runs on one loop; nothing is shared with another thread. Reading
pauses the socket when unread packets pile up, so a slow consumer pushes back
on the broker instead of buffering without bound.
"""

from __future__ import annotations

import asyncio
import collections
import ssl
from typing import Callable, Deque, List, Optional

from simplyprint_ws_client.wire import mqtt_codec as codec

__all__ = ["MqttStream", "ConnectionClosed"]

#: Largest plaintext one TLS record carries.
TLS_READ = 16 * 1024
#: Unread packet bytes at which reading pauses (and half of it, resumes).
READ_HIGH_WATER = 1024 * 1024


class ConnectionClosed(ConnectionError):
    """The broker closed the connection (or it was lost)."""


class MqttStream(asyncio.Protocol):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        ssl_context: Optional[ssl.SSLContext] = None,
        server_hostname: Optional[str] = None,
        on_control: Optional[Callable[[codec.Packet], None]] = None,
    ) -> None:
        self._loop = loop
        #: Receives every control packet (all but PUBLISH and PUBREL, which stay
        #: in wire order) as soon as it arrives, so acknowledgements never wait
        #: behind a slow message consumer.
        self._on_control = on_control
        self._context = ssl_context
        self._server_hostname = server_hostname
        # Any transport with the asyncio.Transport interface (uvloop's doesn't
        # subclass it).
        self._transport: Optional[asyncio.Transport] = None
        self._tls: Optional[ssl.SSLObject] = None
        self._incoming: Optional[ssl.MemoryBIO] = None
        self._outgoing: Optional[ssl.MemoryBIO] = None
        self._handshaken = False
        #: Plaintext held back while TLS waits on the broker (renegotiation).
        self._backlog: List[bytes] = []
        self._buffer = bytearray()
        self._packets: Deque[codec.Packet] = collections.deque()
        self._queued = 0
        self._reading_paused = False
        self._reader: Optional["asyncio.Future[None]"] = None
        self._writing_paused = False
        self._drainers: List["asyncio.Future[None]"] = []
        self._error: Optional[BaseException] = None
        self._closed = False
        self._ready: "asyncio.Future[None]" = loop.create_future()
        self._lost: "asyncio.Future[None]" = loop.create_future()

    @classmethod
    async def open(
        cls,
        host: str,
        port: int,
        *,
        ssl_context: Optional[ssl.SSLContext] = None,
        server_hostname: Optional[str] = None,
        on_control: Optional[Callable[[codec.Packet], None]] = None,
    ) -> "MqttStream":
        """Connect (and finish the TLS handshake). Raises on failure."""
        loop = asyncio.get_running_loop()
        transport, connection = await loop.create_connection(
            lambda: cls(loop, ssl_context, server_hostname, on_control), host, port
        )
        if connection._transport is None:  # connection_made failed
            transport.abort()
        try:
            await asyncio.shield(connection._ready)
        except BaseException:
            connection.abort()
            raise
        return connection

    # -- reading ---------------------------------------------------------------

    async def read_packet(self) -> codec.Packet:
        """The next packet; packets received before a close are delivered first."""
        while not self._packets:
            if self._error is not None:
                raise self._error
            reader = self._reader = self._loop.create_future()
            try:
                await reader
            finally:
                self._reader = None
        packet = self._packets.popleft()
        self._queued -= len(packet.body)
        if self._reading_paused and self._queued < READ_HIGH_WATER // 2:
            self._reading_paused = False
            if self._transport is not None and not self._transport.is_closing():
                self._transport.resume_reading()
        return packet

    # -- writing ---------------------------------------------------------------

    def write(self, data: bytes) -> None:
        """Queue ``data`` (whole packets) on the socket."""
        if self._error is not None:
            raise self._error
        transport = self._transport
        if transport is None or transport.is_closing():
            raise ConnectionClosed("MQTT connection closed")
        if self._tls is None:
            transport.write(data)
            return
        if self._backlog:
            self._backlog.append(data)
            return
        self._encrypt(data)

    async def drain(self) -> None:
        """Wait while the socket's send buffer is full."""
        if self._error is not None:
            raise self._error
        if not self._writing_paused:
            return
        drainer = self._loop.create_future()
        self._drainers.append(drainer)
        await drainer

    def is_closing(self) -> bool:
        return self._closed or self._transport is None or self._transport.is_closing()

    def close(self) -> None:
        """Close gracefully: TLS close_notify, then the socket."""
        if self._closed:
            return
        self._closed = True
        if self._tls is not None and self._handshaken:
            try:
                self._tls.unwrap()
            except (ssl.SSLError, ValueError):
                pass
            self._flush()
        if self._transport is not None:
            self._transport.close()

    def abort(self) -> None:
        self._closed = True
        if self._transport is not None:
            self._transport.abort()

    async def wait_closed(self) -> None:
        await asyncio.shield(self._lost)

    # -- asyncio.Protocol --------------------------------------------------------

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport  # type: ignore[assignment]
        if self._context is None:
            self._ready.set_result(None)
            return
        self._incoming, self._outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        try:
            self._tls = self._context.wrap_bio(
                self._incoming,
                self._outgoing,
                server_side=False,
                server_hostname=self._server_hostname,
            )
        except (ssl.SSLError, ValueError) as error:
            self._fail(error)
            transport.abort()  # type: ignore[attr-defined]
            return
        self._handshake()

    def data_received(self, data: bytes) -> None:
        if self._tls is None:
            self._buffer += data
        else:
            assert self._incoming is not None
            self._incoming.write(data)
            if not self._handshaken and not self._handshake():
                return
            if not self._decrypt():
                return
        self._split()

    def eof_received(self) -> bool:
        self._fail(ConnectionClosed("MQTT connection closed by the broker"))
        return False

    def connection_lost(self, exc: Optional[Exception]) -> None:
        self._fail(
            ConnectionClosed(f"MQTT connection lost: {exc}")
            if exc is not None
            else ConnectionClosed("MQTT connection closed")
        )
        if not self._lost.done():
            self._lost.set_result(None)

    def pause_writing(self) -> None:
        self._writing_paused = True

    def resume_writing(self) -> None:
        self._writing_paused = False
        self._wake_drainers(None)

    # -- TLS ---------------------------------------------------------------------

    def _handshake(self) -> bool:
        assert self._tls is not None
        try:
            self._tls.do_handshake()
        except ssl.SSLWantReadError:
            self._flush()
            return False
        except (ssl.SSLError, ssl.CertificateError) as error:
            self._flush()
            self._fail(error)
            if self._transport is not None:
                self._transport.close()
            return False
        self._flush()
        self._handshaken = True
        if not self._ready.done():
            self._ready.set_result(None)
        return True

    def _decrypt(self) -> bool:
        assert self._tls is not None
        while True:
            try:
                chunk = self._tls.read(TLS_READ)
            except ssl.SSLWantReadError:
                break
            except ssl.SSLZeroReturnError:
                chunk = b""
            except ssl.SSLError as error:
                self._fail(error)
                if self._transport is not None:
                    self._transport.close()
                return False
            if not chunk:
                # close_notify from the broker (a read with no data raises
                # SSLWantReadError, so empty means the session ended).
                self._flush()
                self._split()
                self._fail(ConnectionClosed("MQTT connection closed by the broker"))
                return False
            self._buffer += chunk
        if self._backlog:  # TLS may now accept what it refused while renegotiating
            backlog, self._backlog = self._backlog, []
            for data in backlog:
                if self._backlog:
                    self._backlog.append(data)
                else:
                    self._encrypt(data)
        self._flush()
        return True

    def _encrypt(self, data: bytes) -> None:
        assert self._tls is not None
        view = memoryview(data)
        while view:
            try:
                written = self._tls.write(view)
            except ssl.SSLWantReadError:
                self._backlog.append(bytes(view))
                break
            view = view[written:]
        self._flush()

    def _flush(self) -> None:
        outgoing, transport = self._outgoing, self._transport
        if outgoing is not None and outgoing.pending and transport is not None:
            data = outgoing.read()
            if not transport.is_closing():
                transport.write(data)

    # -- internals -------------------------------------------------------------------

    def _split(self) -> None:
        offset, buffer = 0, self._buffer
        try:
            while True:
                packet, used = codec.split_packet(buffer, offset)
                if packet is None:
                    break
                offset += used
                if packet.type not in codec.IN_ORDER and self._on_control is not None:
                    self._on_control(packet)
                    continue
                self._packets.append(packet)
                self._queued += len(packet.body)
        except Exception as error:  # noqa: BLE001 -- malformed, or refused by on_control
            self._fail(error)
            if self._transport is not None:
                self._transport.close()
        if offset:
            del buffer[:offset]
        if self._packets:
            self._wake_reader()
        if (
            not self._reading_paused
            and self._queued >= READ_HIGH_WATER
            and self._transport is not None
        ):
            self._reading_paused = True
            self._transport.pause_reading()

    def _fail(self, error: BaseException) -> None:
        if self._error is None:
            self._error = error
        if not self._ready.done():
            self._ready.set_exception(error)
            self._ready.exception()  # retrieved: open() re-raises its own copy
        self._wake_reader()
        self._wake_drainers(error)

    def _wake_reader(self) -> None:
        reader = self._reader
        if reader is not None and not reader.done():
            reader.set_result(None)

    def _wake_drainers(self, error: Optional[BaseException]) -> None:
        drainers, self._drainers = self._drainers, []
        for drainer in drainers:
            if drainer.done():
                continue
            if error is None:
                drainer.set_result(None)
            else:
                drainer.set_exception(error)
