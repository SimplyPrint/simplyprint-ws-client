from typing import Union, NamedTuple


class PollCamera(NamedTuple):
    id: int


class StartCamera(NamedTuple):
    id: int


class StopCamera(NamedTuple):
    id: int


class DeleteCamera(NamedTuple):
    id: int


Request = Union[
    PollCamera,
    StartCamera,
    StopCamera,
    DeleteCamera,
]
