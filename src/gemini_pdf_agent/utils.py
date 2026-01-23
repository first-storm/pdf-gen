from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def safe_extract_json(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from text.

    Raises ValueError if no valid JSON object is found.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    decoder = json.JSONDecoder()

    if text.startswith("{"):
        try:
            candidate, _ = decoder.raw_decode(text)
        except json.JSONDecodeError:
            candidate = None
        else:
            if isinstance(candidate, dict):
                return candidate

    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            break
        try:
            candidate, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start += 1
            continue
        if isinstance(candidate, dict):
            return candidate
        start += 1

    raise ValueError("No JSON object found")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _default_font_stack() -> str:
    return "\"Noto Serif CJK SC\", \"Source Han Serif SC\", \"Songti SC\", \"SimSun\", serif"


def _font_family_from_path(path: Path) -> str:
    return path.stem


def parse_font_file_entry(entry: str) -> tuple[str, Path]:
    if "::" in entry:
        family, path_str = entry.split("::", 1)
        return family.strip(), Path(path_str).expanduser().resolve()
    path = Path(entry).expanduser().resolve()
    return _font_family_from_path(path), path


def build_font_css(
    cjk_font_path: str | None,
    allowed_fonts: Iterable[str] | None = None,
    font_files: Iterable[str] | None = None,
) -> str:
    allowed = [font.strip() for font in (allowed_fonts or []) if font.strip()]
    font_entries = list(font_files or [])

    if cjk_font_path:
        font_entries.insert(0, f"CustomCJKSerif::{cjk_font_path}")
        if not allowed:
            allowed.append("CustomCJKSerif")

    font_faces: list[str] = []
    for entry in font_entries:
        family, font_path = parse_font_file_entry(entry)
        font_uri = font_path.as_uri()
        font_faces.append(
            "@font-face {\n"
            f"  font-family: \"{family}\";\n"
            f"  src: url('{font_uri}');\n"
            "  font-weight: normal;\n"
            "  font-style: normal;\n"
            "}"
        )
        if family not in allowed:
            allowed.append(family)

    if allowed:
        family_stack = ", ".join(f"\"{name}\"" for name in allowed)
    else:
        family_stack = _default_font_stack()

    css_parts = font_faces + [f"body {{ font-family: {family_stack}; }}"]
    return "\n".join(css_parts)


def assemble_html(html_body: str, css: str) -> str:
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <style>\n"
        f"{css}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"{html_body}\n"
        "</body>\n"
        "</html>\n"
    )


def apply_css_update(existing: str, update: str | None, mode: str | None) -> str:
    if update is None:
        return existing
    if mode == "replace":
        return update
    addition = update.strip()
    if not addition:
        return existing
    existing_trimmed = existing.rstrip()
    if not existing_trimmed:
        return addition
    return f"{existing_trimmed}\n\n{addition}"


def load_base_css() -> str:
    base_css_path = Path(__file__).parent / "assets" / "base.css"
    return read_text(base_css_path)
