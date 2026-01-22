from gemini_pdf_agent.utils import assemble_html


def test_assemble_html():
    html = assemble_html("<h1>Title</h1>", "body { color: red; }")
    assert "<h1>Title</h1>" in html
    assert "color: red" in html
    assert "<!DOCTYPE html>" in html
