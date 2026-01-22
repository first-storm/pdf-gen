from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Iterable
from urllib import error as url_error
from urllib import request as url_request

from google import genai
from google.genai import types

from .utils import safe_extract_json

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        allowed_fonts: list[str] | None = None,
        api_mode: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self._api_mode = self._infer_api_mode(api_mode, base_url)
        self._base_url = base_url
        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError(self._missing_key_message())

        self._client = None
        if self._api_mode == "gemini":
            http_options: dict[str, Any] | None = None
            if base_url:
                http_options = {"base_url": base_url}
            self._client = genai.Client(api_key=api_key, http_options=http_options)
        self._model = model
        self._allowed_fonts = [font for font in (allowed_fonts or []) if font]
        self._api_key = api_key
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort

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

    def _generate(self, parts: list[types.Part]) -> dict[str, Any]:
        if self._api_mode == "openai":
            return self._generate_openai(parts)
        contents = [types.Content(role="user", parts=parts)]
        config = types.GenerateContentConfig(response_mime_type="application/json")
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        return safe_extract_json(text)

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

    def _generate_openai(self, parts: list[types.Part]) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for part in parts:
            content = self._openai_content_from_part(part)
            if content:
                contents.append(content)

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": contents}],
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
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
        try:
            with url_request.urlopen(request, timeout=120) as response:
                response_data = response.read().decode("utf-8")
        except url_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI endpoint error: {exc.code} {exc.reason} {body}") from exc

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
        return safe_extract_json(str(content))

    def generate_draft(self, prompt_text: str) -> dict[str, Any]:
        prompt = (
            "You are a layout agent. Generate printable Chinese HTML/CSS. "
            "Return JSON only with keys: html_body, extra_css. "
            + self._font_hint()
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="Requirement:\n" + prompt_text),
        ]
        return self._generate(parts)

    def review_and_revise(
        self,
        prompt_text: str,
        html_body: str,
        css: str,
        page_pngs: Iterable[bytes],
    ) -> dict[str, Any]:
        prompt = (
            "Review the layout based on the requirement and page images. "
            "Return JSON only with keys: html_body, css, issues, changes. "
            + self._font_hint()
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_text(text="Requirement:\n" + prompt_text),
            types.Part.from_text(text="Current HTML:\n" + html_body),
            types.Part.from_text(text="Current CSS:\n" + css),
        ]
        for image_bytes in page_pngs:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        return self._generate(parts)
