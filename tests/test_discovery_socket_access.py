import errno
from types import SimpleNamespace

import pytest

from simplyprint_ws_client.events import EventBus
from simplyprint_ws_client.integration.discovery.multicast_base import (
    MulticastListenerBase,
)


class _Socket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _DeniedListener(MulticastListenerBase):
    def __init__(self, error: OSError) -> None:
        super().__init__(
            SimpleNamespace(
                brand="bambu",
                event_type=object,
                group="239.255.255.250",
                multicast_ttl=None,
                port=2021,
            ),
            EventBus(),
        )
        self.error = error
        self.socket = _Socket()
        self.parked = False

    def _make_socket(self) -> _Socket:
        return self.socket

    def protocol_factory(self):
        raise AssertionError("socket binding should fail first")

    async def bind_socket(self, _sock) -> None:
        raise self.error

    def join_group(self, _sock) -> None:
        raise AssertionError("socket binding should fail first")

    async def run_transport(self, _transport) -> None:
        raise AssertionError("socket binding should fail first")

    async def wait(self, timeout=None) -> None:
        assert timeout is None
        self.parked = True


@pytest.mark.asyncio
async def test_socket_access_denied_disables_only_that_listener(caplog):
    listener = _DeniedListener(PermissionError(errno.EACCES, "denied"))

    await listener.run()

    assert listener.parked is True
    assert listener.socket.closed is True
    assert "backend bambu disabled" in caplog.text


@pytest.mark.asyncio
async def test_unrelated_socket_error_still_fails_listener():
    listener = _DeniedListener(OSError(errno.EINVAL, "invalid"))

    with pytest.raises(OSError, match="invalid"):
        await listener.run()

    assert listener.parked is False
    assert listener.socket.closed is True
