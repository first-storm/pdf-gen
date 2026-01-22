from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def render_html_to_pdf(
    html_path: Path,
    pdf_path: Path,
    backend: str = "playwright",
) -> None:
    if backend == "playwright":
        _render_playwright(html_path, pdf_path)
        return
    if backend == "weasyprint":
        _render_weasyprint(html_path, pdf_path)
        return
    raise ValueError(f"Unsupported backend: {backend}")


def _render_playwright(html_path: Path, pdf_path: Path) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "Playwright is not installed. Run: pip install playwright"
        ) from exc

    html_uri = html_path.resolve().as_uri()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_uri, wait_until="load")
            page.pdf(
                path=str(pdf_path),
                print_background=True,
                prefer_css_page_size=True,
            )
            browser.close()
    except Exception as exc:  # pragma: no cover - runtime path
        message = (
            "Failed to render with Playwright. Ensure Chromium is installed: "
            "python -m playwright install chromium"
        )
        raise RuntimeError(message) from exc


def _render_weasyprint(html_path: Path, pdf_path: Path) -> None:
    try:
        from weasyprint import HTML
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "WeasyPrint is not installed. Run: pip install .[weasyprint]"
        ) from exc

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
