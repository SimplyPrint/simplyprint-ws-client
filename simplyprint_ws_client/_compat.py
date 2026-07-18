"""Explicit imports for standard-library features absent on older Python."""

import sys

__all__ = ["StrEnum"]

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # the `strenum` backport is a dependency on Python < 3.11
    from strenum import StrEnum  # noqa: F401
