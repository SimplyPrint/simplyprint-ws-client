"""Version-compatibility imports.

Import the names from here instead of relying on :mod:`_polyfill`'s historic
stdlib monkeypatching - an explicit import survives static analysis and works
no matter which module is imported first.
"""

import sys

__all__ = ["StrEnum"]

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # the `strenum` backport is a dependency on Python < 3.11
    from strenum import StrEnum  # noqa: F401
