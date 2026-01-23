from gemini_pdf_agent.server import _extract_extra_css


def test_extract_extra_css_with_prefix() -> None:
    base_css = "base"
    font_css = "font"
    extra_css = ".a { color: red; }"
    combined = "\n".join([base_css, font_css, extra_css])
    assert _extract_extra_css(combined, base_css, font_css) == extra_css


def test_extract_extra_css_strips_blank_line() -> None:
    base_css = "base"
    font_css = "font"
    combined = "base\nfont\n\n.extra { margin: 0; }"
    assert _extract_extra_css(combined, base_css, font_css) == ".extra { margin: 0; }"


def test_extract_extra_css_without_prefix() -> None:
    base_css = "base"
    font_css = "font"
    saved = "body { margin: 0; }"
    assert _extract_extra_css(saved, base_css, font_css) == saved


def test_extract_extra_css_empty() -> None:
    assert _extract_extra_css("", "base", "font") == ""


def test_extract_extra_css_with_empty_font_css() -> None:
    base_css = "base"
    font_css = ""
    combined = "base\n.extra { padding: 0; }"
    assert _extract_extra_css(combined, base_css, font_css) == ".extra { padding: 0; }"


def test_extract_extra_css_with_empty_base_css() -> None:
    base_css = ""
    font_css = "font"
    combined = "\nfont\n.extra { line-height: 1; }"
    assert _extract_extra_css(combined, base_css, font_css) == ".extra { line-height: 1; }"
