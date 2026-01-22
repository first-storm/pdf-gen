from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .utils import ensure_dir


def _config_path() -> Path:
    override = os.getenv("GEMINI_PDF_AGENT_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".config" / "gemini-pdf-agent" / "config.json"


def load_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_config(config: dict[str, Any]) -> None:
    path = _config_path()
    ensure_dir(path.parent)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
