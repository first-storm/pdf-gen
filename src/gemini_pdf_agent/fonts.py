from __future__ import annotations

import subprocess
from typing import Iterable


def fontconfig_families() -> list[str]:
    try:
        output = subprocess.check_output(
            ["fc-list", "-f", "%{family}\n"], text=True
        )
    except Exception:
        return []

    families: set[str] = set()
    for line in output.splitlines():
        for family in line.split(","):
            name = family.strip()
            if name:
                families.add(name)
    return sorted(families)
