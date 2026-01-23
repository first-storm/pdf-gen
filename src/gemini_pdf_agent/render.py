from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _should_allow_request(url: str, resource_type: str, allow_external_images: bool) -> bool:
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme in {"file", "data", "blob"} or url.startswith("about:"):
        return True
    if allow_external_images and scheme in {"http", "https"} and resource_type == "image":
        return True
    return False


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
    allow_network = os.getenv("GEMINI_PDF_AGENT_ALLOW_NETWORK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    allow_external_images = os.getenv("GEMINI_PDF_AGENT_ALLOW_EXTERNAL_IMAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    allow_js = os.getenv("GEMINI_PDF_AGENT_ALLOW_JS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(java_script_enabled=allow_js)
            if not allow_network:
                def route_handler(route, request) -> None:
                    if _should_allow_request(
                        request.url,
                        request.resource_type,
                        allow_external_images,
                    ):
                        route.continue_()
                        return
                    route.abort()

                context.route("**/*", route_handler)
            page = context.new_page()
            page.goto(html_uri, wait_until="load")
            page.pdf(
                path=str(pdf_path),
                print_background=True,
                prefer_css_page_size=True,
            )
            context.close()
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
