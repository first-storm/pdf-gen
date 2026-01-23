import pytest

from gemini_pdf_agent.gemini import Delimiters, GeminiClient


def _dummy_client() -> GeminiClient:
    return GeminiClient.__new__(GeminiClient)


def test_parse_draft_strips_code_fence() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        f"{delims.html}\n"
        "```html\n"
        "<p>Hello</p>\n"
        "```\n"
        f"{delims.css}\n"
        "```css\n"
        "body { color: red; }\n"
        "```\n"
        f"{delims.end}"
    )
    parsed = client._parse_draft_response(text, delims)
    assert parsed["html_body"] == "<p>Hello</p>"
    assert parsed["extra_css"] == "body { color: red; }"


def test_parse_draft_out_of_order_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = f"{delims.css}\nbody{{}}\n{delims.html}\n<p>Hi</p>\n{delims.end}"
    with pytest.raises(ValueError, match="Delimiters out of order"):
        client._parse_draft_response(text, delims)


def test_parse_draft_missing_delimiters_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    with pytest.raises(ValueError, match="Missing delimiters"):
        client._parse_draft_response("no delimiters here", delims)


def test_parse_draft_json_fallback_empty_html_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = "{\"html_body\": \"\", \"extra_css\": \"body{}\"}"
    with pytest.raises(ValueError, match="Empty html_body in JSON fallback"):
        client._parse_draft_response(text, delims)


def test_parse_review_strips_code_fence() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        f"{delims.meta}\n"
        "{\"done\": false, \"issues\": [], \"changes\": []}\n"
        f"{delims.html}\n"
        "```html\n"
        "<div>OK</div>\n"
        "```\n"
        f"{delims.css}\n"
        "```css\n"
        "p { margin: 0; }\n"
        "```\n"
        f"{delims.end}"
    )
    parsed = client._parse_review_response(text, delims)
    assert parsed["done"] is False
    assert parsed["html_body"] == "<div>OK</div>"
    assert parsed["css"] == "p { margin: 0; }"


def test_parse_review_done_true_with_changes_forces_false() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        f"{delims.meta}\n"
        "{\"done\": true, \"issues\": [], \"changes\": []}\n"
        f"{delims.css}\n"
        "p { margin: 0; }\n"
        f"{delims.end}"
    )
    parsed = client._parse_review_response(text, delims)
    assert parsed["done"] is False


def test_parse_review_html_before_meta_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        f"{delims.html}\n"
        "<p>Hi</p>\n"
        f"{delims.meta}\n"
        "{\"done\": false, \"issues\": [], \"changes\": []}\n"
        f"{delims.end}"
    )
    with pytest.raises(ValueError, match="HTML section appears before META"):
        client._parse_review_response(text, delims)


def test_parse_review_css_before_html_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        f"{delims.meta}\n"
        "{\"done\": false, \"issues\": [], \"changes\": []}\n"
        f"{delims.css}\n"
        "p { margin: 0; }\n"
        f"{delims.html}\n"
        "<p>Hi</p>\n"
        f"{delims.end}"
    )
    with pytest.raises(ValueError, match="CSS section appears before HTML"):
        client._parse_review_response(text, delims)


def test_parse_review_invalid_meta_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = f"{delims.meta}\nnot-json\n{delims.end}"
    with pytest.raises(ValueError, match="Invalid META JSON"):
        client._parse_review_response(text, delims)


def test_parse_review_missing_end_raises() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = f"{delims.meta}\nno-json-here\n"
    with pytest.raises(ValueError, match="Missing delimiters"):
        client._parse_review_response(text, delims)


def test_parse_review_json_fallback_strips_code_fence() -> None:
    client = _dummy_client()
    delims = Delimiters(
        observations="===OBS===",
        meta="===META===",
        html="===HTML===",
        css="===CSS===",
        end="===END===",
    )
    text = (
        "{"
        "\"done\": false, "
        "\"issues\": [], "
        "\"changes\": [], "
        "\"html_body\": \"```html\\n<span>Hi</span>\\n```\", "
        "\"css\": \"```css\\nspan { color: blue; }\\n```\""
        "}"
    )
    parsed = client._parse_review_response(text, delims)
    assert parsed["html_body"] == "<span>Hi</span>"
    assert parsed["css"] == "span { color: blue; }"


def test_infer_api_mode() -> None:
    client = _dummy_client()
    assert client._infer_api_mode(None, "https://api.openai.com/v1", None) == "openai"
    assert client._infer_api_mode(None, "https://myopenai.azure.com", None) == "openai"
    assert (
        client._infer_api_mode(None, "https://generativelanguage.googleapis.com/v1beta", None)
        == "gemini"
    )
    assert client._infer_api_mode(None, None, "gpt-4o") == "openai"
    assert client._infer_api_mode(None, None, "gemini-1.5-pro") == "gemini"
    assert client._infer_api_mode(None, "https://example.com/v1", None) == "openai"
