from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Union,
    Iterator,
    Iterable,
    AsyncIterable,
    AsyncIterator,
    Coroutine,
    ClassVar,
    Optional,
)

from yarl import URL

if TYPE_CHECKING:
    from simplyprint_ws_client.common.worker.context import ExecutionContext

# Typically JPEG bytes.
FrameT = Union[bytes, bytearray, memoryview]


class CameraProtocolException(Exception): ...


class CameraProtocolConnectionError(CameraProtocolException, ConnectionError):
    """Raise when the connection to the camera fails."""

    ...


class CameraProtocolInvalidState(CameraProtocolException):
    """Raise when the camera state needs to be destroyed and recreated."""

    ...


class CameraProtocolPollingMode(Enum):
    """Camera protocol polling mode"""

    CONTINUOUS = auto()
    """Must keep polling"""

    ON_DEMAND = auto()
    """Snapshot based"""


class BaseCameraProtocol(ABC, Iterable[FrameT], AsyncIterable[FrameT]):
    polling_mode: ClassVar[CameraProtocolPollingMode] = (
        CameraProtocolPollingMode.ON_DEMAND
    )
    """Camera polling mode"""
    is_async: ClassVar[bool] = False
    """Is the camera protocol async?"""

    execution_context: ClassVar[Optional["ExecutionContext"]] = None
    """Explicit override for where this protocol runs. When ``None`` the pool
    routes it: an async protocol runs INLINE on the consumer loop (no process, no
    pickle); a sync protocol runs in the shared worker PROCESSes (a thread each,
    CPU-isolated from the loop). Set ``PROCESS`` to move a CPU-heavy async camera
    off the loop too (it then runs on a worker's event loop), or ``THREAD`` to
    give a camera its own thread in this process."""

    uri: URL
    """Configuration URI for the camera protocol, and the only input we have access to."""

    def __init__(self, uri: URL) -> None:
        self.uri = uri

    @staticmethod
    @abstractmethod
    def test(uri: URL) -> Union[bool, Coroutine[None, None, bool]]:
        """Is the configuration valid for this protocol?"""
        ...

    @abstractmethod
    def read(
        self,
    ) -> Union[Iterator[FrameT], Coroutine[None, None, AsyncIterator[FrameT]]]:
        """Read frames from the camera, blocking."""
        ...

    def __iter__(self):
        return self.read()

    def __aiter__(self):
        return self.read()
