from gemini_pdf_agent.render import _should_allow_request


def test_should_allow_request_local_schemes() -> None:
    assert _should_allow_request("file:///tmp/test.png", "image", False) is True
    assert _should_allow_request("data:image/png;base64,abc", "image", False) is True
    assert _should_allow_request("blob:https://example.com/123", "image", False) is True
    assert _should_allow_request("about:blank", "document", False) is True


def test_should_block_external_when_disabled() -> None:
    assert _should_allow_request("https://example.com/a.png", "image", False) is False
    assert _should_allow_request("http://example.com/style.css", "stylesheet", False) is False


def test_should_allow_only_external_images() -> None:
    assert _should_allow_request("https://example.com/a.png", "image", True) is True
    assert _should_allow_request("http://example.com/a.png", "image", True) is True
    assert _should_allow_request("HTTP://example.com/a.png", "image", True) is True
    assert _should_allow_request("https://example.com/app.js", "script", True) is False


def test_should_block_unknown_scheme() -> None:
    assert _should_allow_request("ftp://example.com/a.png", "image", True) is False
