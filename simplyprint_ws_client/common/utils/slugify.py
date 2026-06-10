__all__ = ["slugify"]

import re


def slugify(name: str) -> str:
    # Slugify the log name
    name = re.sub(r"[^\w\s.-]", "", name)
    name = re.sub(r"[\s_-]+", "-", name)
    name = re.sub(r"^-+|-+$", "", name)
    return name
