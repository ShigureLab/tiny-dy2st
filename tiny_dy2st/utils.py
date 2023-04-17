from __future__ import annotations

NAMES_MAP: dict[str, int] = {}


def unique_name(name: str = "var") -> str:
    if name not in NAMES_MAP:
        NAMES_MAP[name] = 0
    else:
        NAMES_MAP[name] += 1
    return f"{name}_{NAMES_MAP[name]}"
