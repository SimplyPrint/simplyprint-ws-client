from __future__ import annotations

from typing import NamedTuple


class WsParams(NamedTuple):
    """Hashable identity of a WebSocket endpoint."""

    url: str

    def __str__(self) -> str:
        return self.url
