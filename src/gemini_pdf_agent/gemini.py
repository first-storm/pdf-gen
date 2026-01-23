from __future__ import annotations

import base64
import json
import logging
import os
import time
from typing import Any, Iterable, cast
from urllib import error as url_error
from urllib import request as url_request

from google import genai
from google.genai import types

from .utils import safe_extract_json

logger = logging.getLogger(__name__)

_DELIM_OBSERVATIONS = "===OBSERVATIONS==="
_DELIM_META = "===META==="
_DELIM_HTML = "===HTML==="
_DELIM_CSS = "===CSS==="
_DELIM_END = "===END==="


class OpenAIRequestError(RuntimeError):
    def __init__(self, code: int, reason: str, body: str) -> None:
        message = f"OpenAI endpoint error: {code} {reason} {body}"
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.body = body


class GeminiClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        allowed_fonts: list[str] | None = None,
        api_mode: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._api_mode = self._infer_api_mode(api_mode, base_url)
        self._base_url = base_url
        resolved_key = api_key or self._resolve_api_key()
        if not resolved_key:
            raise RuntimeError(self._missing_key_message())

        self._client = None
        if self._api_mode == "gemini":
            http_options: types.HttpOptionsDict | None = None
            if base_url:
                http_options = cast(types.HttpOptionsDict, {"base_url": base_url})
            self._client = genai.Client(api_key=resolved_key, http_options=http_options)
        self._model = model
        self._allowed_fonts = [font for font in (allowed_fonts or []) if font]
        self._api_key = resolved_key
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._openai_retries = self._resolve_openai_retries()

    def _resolve_openai_retries(self) -> int:
        raw = os.getenv("GEMINI_PDF_AGENT_OPENAI_RETRIES", "2")
        try:
            return max(0, int(raw))
        except ValueError:
            return 2

    def _infer_api_mode(self, api_mode: str | None, base_url: str | None) -> str:
        if api_mode:
            return api_mode
        if base_url and "/v1/chat/completions" in base_url:
            return "openai"
        return "gemini"

    def _resolve_api_key(self) -> str | None:
        if self._api_mode == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        return os.getenv("GEMINI_API_KEY")

    def _missing_key_message(self) -> str:
        if self._api_mode == "openai":
            return "OPENAI_API_KEY is not set (fallback GEMINI_API_KEY also missing)"
        return "GEMINI_API_KEY is not set"

    def _font_hint(self) -> str:
        if not self._allowed_fonts:
            return ""
        fonts = ", ".join(self._allowed_fonts)
        return f"Allowed font families: {fonts}. Use only these in CSS."

    def _draft_output_format(self) -> str:
        return (
            "Output format (use exact delimiters; no markdown or extra text):\n"
            f"{_DELIM_HTML}\n"
            "[HTML body only]\n"
            f"{_DELIM_CSS}\n"
            "[Extra CSS only]\n"
            f"{_DELIM_END}"
        )

    def _review_output_format(self) -> str:
        return (
            "Output format (use exact delimiters; no markdown or extra text):\n"
            f"{_DELIM_OBSERVATIONS}\n"
            "- Brief, visible observations only\n"
            f"{_DELIM_META}\n"
            "{\"done\": true, \"issues\": [], \"changes\": []}\n"
            f"{_DELIM_HTML}\n"
            "[Full HTML body if changed; omit section if unchanged]\n"
            f"{_DELIM_CSS}\n"
            "[Full CSS if changed; omit section if unchanged]\n"
            f"{_DELIM_END}"
        )

    def _generate_text(self, parts: list[types.Part]) -> str:
        if self._api_mode == "openai":
            return self._openai_generate_text(parts, use_response_format=False)
        contents = [types.Content(role="user", parts=parts)]
        config: types.GenerateContentConfig | None = None
        if self._temperature is not None:
            config = types.GenerateContentConfig(temperature=self._temperature)
        if self._client is None:
            raise RuntimeError("Gemini client is not initialized")
        kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": cast(types.ContentListUnionDict, contents),
        }
        if config is not None:
            kwargs["config"] = config
        response = self._client.models.generate_content(**kwargs)
        return response.text or ""

    def _coerce_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(item).strip() for item in value if str(item).strip()]
            return items
        if isinstance(value, str):
            item = value.strip()
            return [item] if item else []
        return [str(value).strip()]

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        if isinstance(value, (int, float)):
            return value != 0
        return bool(value)

    def _parse_json_fallback(self, text: str) -> dict[str, Any] | None:
        try:
            candidate = safe_extract_json(text)
        except ValueError:
            return None
        if not isinstance(candidate, dict):
            return None
        return candidate

    def _parse_draft_response(self, text: str) -> dict[str, Any]:
        html_start = text.find(_DELIM_HTML)
        css_start = text.find(_DELIM_CSS)
        end_start = text.find(_DELIM_END)

        if html_start == -1 or css_start == -1 or end_start == -1:
            fallback = self._parse_json_fallback(text)
            if fallback:
                html_body = str(fallback.get("html_body", ""))
                extra_css = str(fallback.get("extra_css", fallback.get("css", "")))
                if not html_body:
                    raise ValueError("Empty html_body in JSON fallback")
                return {"html_body": html_body, "extra_css": extra_css}
            missing = []
            if html_start == -1:
                missing.append(_DELIM_HTML)
            if css_start == -1:
                missing.append(_DELIM_CSS)
            if end_start == -1:
                missing.append(_DELIM_END)
            raise ValueError(f"Missing delimiters: {', '.join(missing)}")

        if not (html_start < css_start < end_start):
            raise ValueError("Delimiters out of order; expected HTML -> CSS -> END.")

        html_body = text[html_start + len(_DELIM_HTML) : css_start].strip()
        extra_css = text[css_start + len(_DELIM_CSS) : end_start].strip()
        if not html_body:
            raise ValueError("Empty HTML section")
        return {"html_body": html_body, "extra_css": extra_css}

    def _parse_review_response(self, text: str) -> dict[str, Any]:
        meta_start = text.find(_DELIM_META)
        end_start = text.find(_DELIM_END)

        if meta_start == -1 or end_start == -1 or meta_start > end_start:
            fallback = self._parse_json_fallback(text)
            if fallback:
                result: dict[str, Any] = {
                    "done": self._coerce_bool(fallback.get("done")),
                    "issues": self._coerce_list(fallback.get("issues")),
                    "changes": self._coerce_list(fallback.get("changes")),
                }
                if "html_body" in fallback:
                    result["html_body"] = str(fallback.get("html_body", ""))
                if "css" in fallback:
                    result["css"] = str(fallback.get("css", ""))
                if result["done"] and (
                    result["issues"]
                    or result["changes"]
                    or "html_body" in result
                    or "css" in result
                ):
                    result["done"] = False
                return result
            missing = []
            if meta_start == -1:
                missing.append(_DELIM_META)
            if end_start == -1:
                missing.append(_DELIM_END)
            raise ValueError(f"Missing delimiters: {', '.join(missing)}")

        html_start = text.find(_DELIM_HTML)
        css_start = text.find(_DELIM_CSS)
        if html_start != -1 and html_start > end_start:
            html_start = -1
        if css_start != -1 and css_start > end_start:
            css_start = -1

        if html_start != -1 and html_start < meta_start:
            raise ValueError("HTML section appears before META")
        if css_start != -1 and css_start < meta_start:
            raise ValueError("CSS section appears before META")
        if html_start != -1 and css_start != -1 and css_start < html_start:
            raise ValueError("CSS section appears before HTML")

        meta_end_candidates = [end_start]
        if html_start != -1:
            meta_end_candidates.append(html_start)
        if css_start != -1:
            meta_end_candidates.append(css_start)
        meta_end = min(pos for pos in meta_end_candidates if pos > meta_start)
        meta_block = text[meta_start + len(_DELIM_META) : meta_end].strip()
        if not meta_block:
            raise ValueError("Empty META section")
        try:
            meta = safe_extract_json(meta_block)
        except ValueError as exc:
            detail = ""
            try:
                json.loads(meta_block)
            except json.JSONDecodeError as json_exc:
                detail = f" (line {json_exc.lineno} column {json_exc.colno})"
            raise ValueError(f"Invalid META JSON{detail}") from exc

        result = {
            "done": self._coerce_bool(meta.get("done")),
            "issues": self._coerce_list(meta.get("issues")),
            "changes": self._coerce_list(meta.get("changes")),
        }

        if html_start != -1:
            html_end_candidates = [end_start]
            if css_start != -1:
                html_end_candidates.append(css_start)
            html_end = min(pos for pos in html_end_candidates if pos > html_start)
            html_body = text[html_start + len(_DELIM_HTML) : html_end].strip()
            if not html_body:
                raise ValueError("Empty HTML section")
            result["html_body"] = html_body

        if css_start != -1:
            css_body = text[css_start + len(_DELIM_CSS) : end_start].strip()
            result["css"] = css_body

        if result["done"] and (
            result["issues"]
            or result["changes"]
            or "html_body" in result
            or "css" in result
        ):
            result["done"] = False
        return result

    def _repair_output(
        self,
        parts: list[types.Part],
        bad_text: str,
        error_message: str,
        output_kind: str,
        repair_attempts: int = 2,
    ) -> dict[str, Any]:
        if output_kind == "draft":
            format_hint = self._draft_output_format()
        else:
            format_hint = self._review_output_format()
        repair_prompt = (
            "The previous response did not follow the required delimiter format. "
            f"Error: {error_message}. "
            "Re-emit the response using ONLY the required format. "
            "Do not wrap in markdown or add extra text.\n"
            f"{format_hint}"
        )
        last_error: Exception | None = None
        current_bad = bad_text
        for _ in range(max(1, repair_attempts)):
            repair_parts = list(parts) + [
                types.Part.from_text(text=repair_prompt),
                types.Part.from_text(text="Invalid response:\n" + current_bad),
            ]
            text = self._generate_text(repair_parts)
            try:
                if output_kind == "draft":
                    return self._parse_draft_response(text)
                return self._parse_review_response(text)
            except ValueError as exc:
                last_error = exc
                current_bad = text
                continue
        if last_error:
            raise last_error
        raise ValueError("Invalid response format")

    def _openai_endpoint(self) -> str:
        if not self._base_url:
            return "https://api.openai.com/v1/chat/completions"
        if "/v1/chat/completions" in self._base_url:
            return self._base_url
        base = self._base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _openai_content_from_part(self, part: types.Part) -> dict[str, Any] | None:
        text = getattr(part, "text", None)
        if text:
            return {"type": "text", "text": text}
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            mime_type = getattr(inline_data, "mime_type", "image/png")
            b64 = base64.b64encode(inline_data.data).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        return None

    def _openai_response_format(self) -> dict[str, Any] | None:
        raw = os.getenv("GEMINI_PDF_AGENT_OPENAI_RESPONSE_FORMAT", "").strip().lower()
        if raw in {"0", "false", "off", "none"}:
            return None
        return {"type": "json_object"}

    def _openai_response_format_unsupported(self, error: OpenAIRequestError) -> bool:
        if error.code != 400:
            return False
        body = error.body.lower()
        return "response_format" in body or "response format" in body

    def _openai_request(self, payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        request = url_request.Request(
            self._openai_endpoint(),
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        max_attempts = max(1, 1 + self._openai_retries)
        response_data = ""
        for attempt in range(1, max_attempts + 1):
            try:
                with url_request.urlopen(request, timeout=420) as response:
                    response_data = response.read().decode("utf-8")
                return response_data
            except url_error.HTTPError as exc:
                retryable = exc.code in {429, 500, 502, 503, 504}
                if retryable and attempt < max_attempts:
                    try:
                        exc.read()
                    except Exception:
                        pass
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = 2 ** (attempt - 1)
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "OpenAI endpoint error %s; retrying in %.1fs (attempt %s/%s).",
                        exc.code,
                        delay,
                        attempt,
                        max_attempts,
                    )
                    time.sleep(delay)
                    continue
                body = exc.read().decode("utf-8", errors="ignore")
                raise OpenAIRequestError(exc.code, exc.reason, body) from exc
        return response_data

    def _openai_generate_text(
        self,
        parts: list[types.Part],
        use_response_format: bool = True,
    ) -> str:
        contents: list[dict[str, Any]] = []
        for part in parts:
            content = self._openai_content_from_part(part)
            if content:
                contents.append(content)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": contents}],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        response_format = None
        if use_response_format:
            response_format = self._openai_response_format()
            if response_format:
                payload["response_format"] = response_format

        try:
            response_data = self._openai_request(payload)
        except OpenAIRequestError as exc:
            if response_format and self._openai_response_format_unsupported(exc):
                logger.warning("OpenAI endpoint rejected response_format; retrying without it.")
                payload.pop("response_format", None)
                response_data = self._openai_request(payload)
            else:
                raise RuntimeError(str(exc)) from exc

        raw = json.loads(response_data)
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            content = "".join(text_parts)
        return str(content)

    def generate_draft(
        self,
        prompt_text: str,
        image_context: str | None = None,
        images: Iterable[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        prompt = (
            "You are a layout agent for generating HTML/CSS optimized for PDF printing.\n"
            "Goal: Create an HTML body and extra CSS that satisfy the requirement.\n"
            "Constraints:\n"
            "- Return only the HTML body (no <html>, <head>, or <body> wrapper).\n"
            "- Return only additional CSS; base styles already exist.\n"
            "- Use CSS paged media rules; adjust @page size/margins if needed and avoid"
            " awkward page breaks with break-inside: avoid for key blocks.\n"
            "- Assume A4 unless the requirement specifies otherwise.\n"
            "- Output raw HTML/CSS without JSON escaping, markdown, or extra commentary.\n"
            "- Do not include placeholder text from the format example.\n"
            f"{self._font_hint()}\n"
            f"{self._draft_output_format()}"
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="Requirement:\n" + prompt_text),
        ]
        if image_context:
            parts.append(types.Part.from_text(text=image_context))
        if images:
            for data, mime_type in images:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type or "image/png"))
        text = self._generate_text(parts)
        try:
            return self._parse_draft_response(text)
        except ValueError as exc:
            logger.warning("Invalid draft format from model; requesting repair: %s", exc)
            return self._repair_output(
                parts,
                text,
                str(exc),
                output_kind="draft",
            )

    def review_and_revise(
        self,
        prompt_text: str,
        html_body: str,
        css: str,
        page_pngs: Iterable[bytes],
        image_context: str | None = None,
        images: Iterable[tuple[bytes, str]] | None = None,
    ) -> dict[str, Any]:
        page_list = list(page_pngs)
        page_count = len(page_list)
        prompt = (
            "Review the layout using the requirement, current HTML/CSS, and page images.\n"
            "Focus on obvious, visible issues (missing content, overflow, clipping, overlap, "
            "incorrect ordering, or major spacing problems). If uncertain, say so in observations.\n"
            "If you make any edits, set done=false.\n"
            "If no changes are needed, set done=true and omit HTML/CSS sections entirely.\n"
            "If only one section changes, include only that section; provide full replacements for"
            " any section you include.\n"
            "The current CSS shown below includes base styles and fonts; only return extra CSS"
            " overrides, not the full combined CSS.\n"
            "Use CSS paged media rules; adjust @page size/margins if needed and avoid awkward page"
            " breaks with break-inside: avoid for key blocks.\n"
            "Assume A4 unless the requirement specifies otherwise.\n"
            "META rules: issues/changes must be arrays of strings; when done=true they must be empty.\n"
            "Output raw HTML/CSS without JSON escaping, markdown, or extra commentary.\n"
            "Do not include placeholder text from the format example.\n"
            f"{self._font_hint()}\n"
            f"{self._review_output_format()}"
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="Requirement:\n" + prompt_text),
            types.Part.from_text(text="Current HTML:\n" + html_body),
            types.Part.from_text(text="Current CSS:\n" + css),
            types.Part.from_text(text=f"Current page count: {page_count}."),
        ]
        if image_context:
            parts.append(types.Part.from_text(text=image_context))
        if images:
            for data, mime_type in images:
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type or "image/png"))
        for image_bytes in page_list:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        text = self._generate_text(parts)
        try:
            return self._parse_review_response(text)
        except ValueError as exc:
            logger.warning("Invalid review format from model; requesting repair: %s", exc)
            return self._repair_output(
                parts,
                text,
                str(exc),
                output_kind="review",
            )
