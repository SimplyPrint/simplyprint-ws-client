from typing import Union, NamedTuple, Optional

from .base import BaseCameraProtocol


class CreateCamera(NamedTuple):
    id: int
    protocol: BaseCameraProtocol
    pause_timeout: Optional[int] = None


class PollCamera(NamedTuple):
    id: int


class StartCamera(NamedTuple):
    id: int


class StopCamera(NamedTuple):
    id: int


class DeleteCamera(NamedTuple):
    id: int


Request = Union[
    CreateCamera,
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
]
