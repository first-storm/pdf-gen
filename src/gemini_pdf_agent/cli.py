from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .gemini import GeminiClient
from .pdf_inspect import pdf_to_pngs
from .render import render_html_to_pdf
from .regression import regression_compare, save_report
from .utils import (
    assemble_html,
    build_font_css,
    ensure_dir,
    load_base_css,
    read_text,
)
from .config import load_config, save_config
from .fonts import fontconfig_families

logger = logging.getLogger(__name__)


def read_interactive_prompt() -> str:
    print("Enter prompt text, end with EOF (Ctrl-D/Ctrl-Z):", file=sys.stderr)
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def prompt_continue_after_done() -> tuple[bool, str]:
    if not sys.stdin.isatty():
        return True, ""
    print("Model requested to stop. End iteration? [Y/n]", file=sys.stderr)
    answer = input().strip().lower()
    if answer in ("", "y", "yes"):
        return True, ""
    print("Add extra prompt (single line, empty to stop):", file=sys.stderr)
    extra = input().strip()
    if not extra:
        return True, ""
    return False, extra

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini PDF agent")
    parser.add_argument("--prompt", help="Prompt text file (UTF-8)")
    parser.add_argument("--prompt-text", help="Prompt text provided directly")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive prompt input")
    parser.add_argument("--out", help="Output PDF path")
    parser.add_argument("--workdir", help="Working directory")
    parser.add_argument("--model", help="Gemini model")
    parser.add_argument("--iterations", type=int, help="Iteration count")
    parser.add_argument("--backend", default="playwright", choices=["playwright", "weasyprint"])
    parser.add_argument("--zoom", type=float, default=2.0, help="PNG render zoom")
    parser.add_argument("--cjk-font", help="Path to CJK serif font file")
    parser.add_argument("--baseline", help="Baseline directory")
    parser.add_argument("--init-baseline", action="store_true", help="Initialize baseline")
    parser.add_argument("--diff-threshold", type=float, default=0.005)
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument(
        "--api-mode",
        choices=["gemini", "openai"],
        help="API mode: gemini (default) or openai (/v1/chat/completions)",
    )
    parser.add_argument("--temperature", type=float, help="Sampling temperature for OpenAI mode")
    parser.add_argument(
        "--reasoning-effort",
        help="Reasoning effort for OpenAI mode (e.g. low, medium, high)",
    )
    parser.add_argument("--api-key", help="API key (stored in config if --save-config)")
    parser.add_argument("--save-config", action="store_true", help="Save defaults to config")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def copy_pages(src_dir: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    for path in src_dir.glob("page_*.png"):
        shutil.copy2(path, dest_dir / path.name)


def main() -> None:
    setup_logging()
    args = parse_args()

    config = load_config()
    model = args.model or config.get("model") or "gemini-2.0-flash"
    base_url = args.base_url or config.get("base_url")
    api_mode = args.api_mode or config.get("api_mode")
    iterations_value = args.iterations or config.get("iterations") or 2
    temperature = args.temperature if args.temperature is not None else config.get("temperature")
    reasoning_effort = args.reasoning_effort or config.get("reasoning_effort")
    api_key = args.api_key or config.get("api_key")
    allowed_fonts = config.get("allowed_fonts") or []
    font_files = config.get("font_files") or []
    use_fontconfig = bool(config.get("use_fontconfig"))
    if use_fontconfig:
        allowed_fonts = list(allowed_fonts) + fontconfig_families()
    if api_key and not os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = api_key

    if args.save_config:
        save_config(
            {
                "model": model,
                "base_url": base_url,
                "api_mode": api_mode,
                "iterations": iterations_value,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "api_key": api_key,
                "allowed_fonts": allowed_fonts,
                "font_files": font_files,
                "use_fontconfig": use_fontconfig,
            }
        )

    if not args.out:
        if args.save_config:
            logger.info("Config saved.")
            return
        if args.interactive or args.prompt_text:
            args.out = "result.pdf"
            logger.info("No --out provided; defaulting to %s", args.out)
        else:
            raise ValueError("Provide --out when generating PDF")

    if args.interactive and not args.prompt_text and not args.prompt:
        prompt_text = read_interactive_prompt()
    elif args.prompt_text:
        prompt_text = args.prompt_text
    elif args.prompt:
        prompt_path = Path(args.prompt)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        prompt_text = read_text(prompt_path)
    else:
        raise ValueError("Provide --prompt, --prompt-text, or use --interactive")

    iterations = max(int(iterations_value), 1)
    workdir = Path(args.workdir) if args.workdir else Path(
        "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    ensure_dir(workdir)

    out_path = Path(args.out)
    base_css = load_base_css()
    font_css = build_font_css(
        args.cjk_font,
        allowed_fonts=allowed_fonts,
        font_files=font_files,
    )

    client = GeminiClient(
        model=model,
        base_url=base_url,
        allowed_fonts=allowed_fonts,
        api_mode=api_mode,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )

    html_body = ""
    extra_css = ""
    combined_css = ""
    page_bytes: list[bytes] = []
    last_report: dict | None = None
    last_pdf_path: Path | None = None

    for i in range(iterations):
        logger.info("Iteration %s/%s", i + 1, iterations)

        if i == 0:
            draft = client.generate_draft(prompt_text)
            html_body = str(draft.get("html_body", ""))
            extra_css = str(draft.get("extra_css", ""))
        else:
            review = client.review_and_revise(prompt_text, html_body, combined_css, page_bytes)
            done = bool(review.get("done"))
            prev_html_body = html_body
            prev_extra_css = extra_css
            html_update = review.get("html_body")
            if html_update is not None and "html_body" in review:
                html_body = str(html_update)
            css_update = review.get("css")
            if css_update is not None and "css" in review:
                extra_css = str(css_update)
            issues = review.get("issues") or []
            changes = review.get("changes") or []
            has_edits = bool(changes) or html_body != prev_html_body or extra_css != prev_extra_css
            if done and (has_edits or issues):
                done = False
            if issues:
                logger.info("Issues: %s", issues)
            if changes:
                logger.info("Changes: %s", changes)
            if done and last_pdf_path is not None:
                stop, extra_prompt = prompt_continue_after_done()
                if stop:
                    logger.info("Model requested early stop; using previous output.")
                    shutil.copy2(last_pdf_path, out_path)
                    logger.info("Output PDF saved to %s", out_path)
                    return
                prompt_text = f"{prompt_text}\n\nAdditional request:\n{extra_prompt}"
                logger.info("Continuing with additional user request.")

        combined_css = "\n".join([base_css, font_css, extra_css])
        html = assemble_html(html_body, combined_css)

        html_path = workdir / f"draft_{i:02d}.html"
        pdf_path = workdir / f"draft_{i:02d}.pdf"
        pages_dir = workdir / f"pages_{i:02d}"
        html_path.write_text(html, encoding="utf-8")

        render_html_to_pdf(html_path, pdf_path, backend=args.backend)
        page_paths = pdf_to_pngs(pdf_path, pages_dir, zoom=args.zoom)
        page_bytes = [path.read_bytes() for path in page_paths]
        last_pdf_path = pdf_path

        if args.init_baseline:
            baseline_dir = Path(args.baseline) if args.baseline else workdir / "baseline"
            copy_pages(pages_dir, baseline_dir)
            shutil.copy2(pdf_path, out_path)
            logger.info("Baseline initialized at %s", baseline_dir)
            logger.info("Output PDF saved to %s", out_path)
            return

        if args.baseline:
            diff_dir = workdir / "diff"
            report = regression_compare(
                baseline_dir=Path(args.baseline),
                pages_dir=pages_dir,
                diff_dir=diff_dir,
                threshold=args.diff_threshold,
            )
            last_report = report
            save_report(report, workdir / "regression_report.json")
            logger.info("Regression passed: %s", report.get("passed"))
            if report.get("passed"):
                shutil.copy2(pdf_path, out_path)
                logger.info("Output PDF saved to %s", out_path)
                return

    shutil.copy2(pdf_path, out_path)
    logger.info("Output PDF saved to %s", out_path)
    if last_report is not None:
        save_report(last_report, workdir / "regression_report.json")


if __name__ == "__main__":
    main()
