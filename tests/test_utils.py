import pytest

from gemini_pdf_agent.utils import build_font_css, safe_extract_json


def test_safe_extract_json_plain():
    data = safe_extract_json('{"a": 1}')
    assert data["a"] == 1


def test_safe_extract_json_wrapped():
    data = safe_extract_json("prefix {\"b\": 2} suffix")
    assert data["b"] == 2


def test_safe_extract_json_code_block():
    data = safe_extract_json("```json\n{\"c\": 3}\n```")
    assert data["c"] == 3


def test_safe_extract_json_skips_invalid_braces():
    data = safe_extract_json("prefix body { color: red; } {\"d\": 4}")
    assert data["d"] == 4


def test_safe_extract_json_invalid():
    with pytest.raises(ValueError):
        safe_extract_json("no json here")


def test_build_font_css_without_font():
    css = build_font_css(None)
    assert "@font-face" not in css
    assert "serif" in css


def test_build_font_css_with_font(tmp_path):
    font_path = tmp_path / "font.otf"
    font_path.write_bytes(b"dummy")
    css = build_font_css(str(font_path))
    assert "@font-face" in css
    assert "CustomCJKSerif" in css
    assert "file://" in css


def test_build_font_css_with_allowed_fonts():
    css = build_font_css(None, allowed_fonts=["Test Serif"])
    assert "Test Serif" in css
    assert "serif" not in css.split("font-family:")[1]
