"""Loopback servers that misbehave below MQTT, for shutdown/timeout tests.

:class:`SilentServer` accepts TCP and then never says a word: no CONNACK, no
TLS handshake -- a printer that is blackholed or hung mid-boot. With
``stall_after_connack`` it answers CONNECT and then stops *reading*, so the
client's outbound bytes pile up in its socket buffer (a printer whose MQTT
task wedged), until :attr:`release` is set. Either way it records accepts,
when each connection ends, and the bytes it read from it.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from simplyprint_ws_client.wire import mqtt_codec as codec


class SilentServer:
    def __init__(self, *, stall_after_connack: bool = False) -> None:
        self.stall_after_connack = stall_after_connack
        self.server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.accepted = 0
        self.closed = 0
        #: Bytes each connection sent before it ended (after CONNECT, if stalled).
        self.received: List[bytes] = []
        self.writers: List[asyncio.StreamWriter] = []
        self.release = asyncio.Event()
        self.changed = asyncio.Event()

    async def start(self) -> "SilentServer":
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    def url(self, scheme: str = "mqtt", query: str = "") -> str:
        return f"{scheme}://bblp:secret@127.0.0.1:{self.port}/{query}"

    async def stop(self) -> None:
        self.release.set()
        for writer in self.writers:
            writer.transport.abort()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()

    async def wait_for(self, predicate, timeout: float = 5.0) -> None:
        async def wait() -> None:
            while not predicate():
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(wait(), timeout)

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.accepted += 1
        self.writers.append(writer)
        self.changed.set()
        data = bytearray()
        try:
            if self.stall_after_connack:
                await codec.read_packet(reader)
                writer.write(codec._packet(codec.CONNACK, 0, bytes((0, 0))))
                await writer.drain()
                await self.release.wait()  # read nothing until released
            while True:
                chunk = await reader.read(1 << 20)
                if not chunk:
                    break
                data += chunk
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.received.append(bytes(data))
            self.closed += 1
            self.changed.set()
            writer.close()
