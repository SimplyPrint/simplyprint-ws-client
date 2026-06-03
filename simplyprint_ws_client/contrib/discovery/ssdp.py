"""Brand-neutral SSDP request parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SSDPRequest:
    method: str
    uri: str
    version: str
    headers: Dict[str, str]


class SSDPRequestParser:
    @staticmethod
    def _parse_headers(lines: List[str]) -> Dict[str, str]:
        headers = {}
        for line in lines:
            if ":" not in line:
                break

            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        return headers

    @classmethod
    def parse(cls, data: bytes) -> Optional[SSDPRequest]:
        try:
            text = data.decode()
        except UnicodeDecodeError:
            return None

        if not text:
            return None

        lines = text.splitlines()
        first = lines[0].strip().split()

        if len(first) != 3:
            return None

        method, uri, version = first

        return SSDPRequest(method, uri, version, cls._parse_headers(lines[1:]))
