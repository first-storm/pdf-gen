from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import mimetypes
import os
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import load_config
from .fonts import fontconfig_families
from .gemini import GeminiClient
from .pdf_inspect import pdf_to_pngs
from .render import render_html_to_pdf
from .utils import assemble_html, build_font_css, ensure_dir, load_base_css

logger = logging.getLogger(__name__)

CORS_ORIGIN_REGEX = os.getenv("GEMINI_PDF_AGENT_CORS_ORIGIN_REGEX")

app = FastAPI(title="gemini-pdf-agent")

if CORS_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

RESULTS: dict[str, Path] = {}
STATES: dict[str, Path] = {}
STOP_REQUESTS: dict[str, bool] = {}
WORKDIR_ROOT = Path(os.getenv("GEMINI_PDF_AGENT_WORKDIR", "server_runs"))
STATE_DIR = WORKDIR_ROOT / "state"
_CLEANUP_STARTED = False


@dataclass
class StorageConfig:
    provider: str
    bucket: str
    region: str | None
    endpoint_url: str | None
    access_key_id: str | None
    secret_access_key: str | None
    prefix: str
    public_url_base: str | None


@dataclass
class RenderConfig:
    prompt: str
    model: str
    base_url: str | None
    api_mode: str | None
    api_key: str | None
    iterations: int
    backend: str
    zoom: float
    temperature: float | None
    reasoning_effort: str | None
    allowed_fonts: list[str]
    font_files: list[str]
    use_fontconfig: bool
    cjk_font: str | None
    return_pdf: str
    workdir: Path
    job_id: str
    ttl_seconds: int
    storage: StorageConfig | None
    images: list["ImageAsset"]
    max_image_bytes: int | None
    max_image_count: int | None
    resume_from: str | None = None


@dataclass
class ImageAsset:
    name: str
    filename: str
    mime_type: str
    local_path: Path
    s3_key: str | None = None
    s3_url: str | None = None


def _format_sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_uploaded_fonts(
    uploads: Iterable[Any],
    families: list[str] | None,
    target_dir: Path,
) -> list[str]:
    entries: list[str] = []
    ensure_dir(target_dir)
    for idx, upload in enumerate(uploads):
        filename = Path(getattr(upload, "filename", None) or f"font_{idx}").name
        dest = target_dir / filename
        with dest.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        family = ""
        if families and idx < len(families):
            family = families[idx].strip()
        if not family:
            family = dest.stem
        entries.append(f"{family}::{dest}")
        try:
            upload.file.close()
        except Exception:
            pass
    return entries


def _save_uploaded_images(
    uploads: Iterable[Any],
    names: list[str] | None,
    target_dir: Path,
    max_bytes: int | None = None,
) -> list[ImageAsset]:
    images: list[ImageAsset] = []
    ensure_dir(target_dir)
    for idx, upload in enumerate(uploads):
        original_name = Path(getattr(upload, "filename", None) or f"image_{idx}").name
        if not original_name:
            original_name = f"image_{idx}"
        filename = f"{idx:02d}_{original_name}"
        dest = target_dir / filename
        with dest.open("wb") as output:
            shutil.copyfileobj(upload.file, output)
        if max_bytes and dest.stat().st_size > max_bytes:
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            raise ValueError(f"Image exceeds max size: {dest.name}")
        mime_type = getattr(upload, "content_type", None) or mimetypes.guess_type(dest.name)[0]
        mime_type = mime_type or "application/octet-stream"
        if not mime_type.startswith("image/"):
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            raise ValueError(f"Invalid image content type: {mime_type}")
        name = ""
        if names and idx < len(names):
            name = str(names[idx]).strip()
        if not name:
            name = Path(original_name).stem
        images.append(
            ImageAsset(
                name=name,
                filename=filename,
                mime_type=mime_type,
                local_path=dest,
            )
        )
        try:
            upload.file.close()
        except Exception:
            pass
    return images


def _state_path(workdir: Path) -> Path:
    return STATE_DIR / f"{workdir.name}.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_time(value: datetime) -> str:
    return value.isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _save_state(
    config: RenderConfig,
    html_body: str,
    combined_css: str,
    pages_dir: Path,
    iteration: int,
    storage_info: dict[str, Any] | None,
) -> None:
    created_at = _now_utc()
    expires_at = created_at + timedelta(seconds=config.ttl_seconds)
    payload = {
        "job_id": config.job_id,
        "prompt": config.prompt,
        "html_body": html_body,
        "css": combined_css,
        "pages_dir": str(pages_dir),
        "iteration": iteration,
        "workdir": str(config.workdir),
        "created_at": _serialize_time(created_at),
        "expires_at": _serialize_time(expires_at),
        "ttl_seconds": config.ttl_seconds,
        "config": {
            "model": config.model,
            "base_url": config.base_url,
            "api_mode": config.api_mode,
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "allowed_fonts": config.allowed_fonts,
            "font_files": config.font_files,
            "use_fontconfig": config.use_fontconfig,
            "cjk_font": config.cjk_font,
            "backend": config.backend,
            "zoom": config.zoom,
            "max_image_bytes": config.max_image_bytes,
            "max_image_count": config.max_image_count,
        },
    }
    if config.images:
        payload["images"] = _serialize_images(config.images)
    if storage_info:
        payload["storage"] = storage_info
    path = _state_path(config.workdir)
    ensure_dir(path.parent)
    existing = STATES.get(config.job_id)
    if existing and existing != path and existing.exists():
        try:
            existing.unlink()
        except FileNotFoundError:
            pass
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    STATES[config.job_id] = path


def _serialize_images(images: list[ImageAsset]) -> list[dict[str, Any]]:
    return [
        {
            "name": image.name,
            "filename": image.filename,
            "mime_type": image.mime_type,
            "local_path": str(image.local_path),
            "s3_key": image.s3_key,
            "s3_url": image.s3_url,
        }
        for image in images
    ]


def _deserialize_images(state: dict[str, Any]) -> list[ImageAsset]:
    raw_list = state.get("images") or []
    images: list[ImageAsset] = []
    if not isinstance(raw_list, list):
        return images
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        local_path = item.get("local_path")
        images.append(
            ImageAsset(
                name=str(item.get("name") or ""),
                filename=str(item.get("filename") or ""),
                mime_type=str(item.get("mime_type") or "image/png"),
                local_path=Path(str(local_path)) if local_path else Path(),
                s3_key=str(item.get("s3_key")) if item.get("s3_key") else None,
                s3_url=str(item.get("s3_url")) if item.get("s3_url") else None,
            )
        )
    return images


def _image_context(images: list[ImageAsset]) -> str:
    if not images:
        return ""
    lines = [
        "You may embed the following user-provided images in HTML using <img src=\"...\">.",
        "Use the provided local_src values exactly (available at render time).",
        "Images:",
    ]
    for image in images:
        local_src = image.local_path.resolve().as_uri()
        entry = f"- name: {image.name}; local_src: {local_src}"
        if image.s3_url:
            entry += f"; remote_src: {image.s3_url}"
        lines.append(entry)
    return "\n".join(lines)


def _image_parts(images: list[ImageAsset]) -> list[tuple[bytes, str]]:
    parts: list[tuple[bytes, str]] = []
    for image in images:
        if not image.local_path.exists() or not image.local_path.is_file():
            continue
        parts.append((image.local_path.read_bytes(), image.mime_type))
    return parts


def _prepare_images(config: RenderConfig) -> list[ImageAsset]:
    if not config.images:
        return []
    images_dir = WORKDIR_ROOT / "images" / config.job_id
    ensure_dir(images_dir)
    prepared: list[ImageAsset] = []
    for image in config.images:
        if not image.filename:
            image.filename = image.local_path.name if image.local_path else ""
        if not image.filename:
            image.filename = f"{len(prepared):02d}_image"
        dest = images_dir / image.filename
        if image.local_path and image.local_path.exists() and image.local_path.is_file():
            if image.local_path.resolve() != dest.resolve():
                shutil.copy2(image.local_path, dest)
        elif config.storage and image.s3_key:
            client = _s3_client(config.storage)
            client.download_file(config.storage.bucket, image.s3_key, str(dest))
        else:
            raise RuntimeError(f"Image file missing: {image.filename}")
        if config.max_image_bytes and dest.stat().st_size > config.max_image_bytes:
            raise RuntimeError(f"Image exceeds max size: {dest.name}")
        prepared.append(
            ImageAsset(
                name=image.name or Path(image.filename).stem,
                filename=image.filename,
                mime_type=image.mime_type or "image/png",
                local_path=dest,
                s3_key=image.s3_key,
                s3_url=image.s3_url,
            )
        )
    _cleanup_uploaded_images(config.job_id)
    config.images = prepared
    return prepared


def _storage_from_payload(payload: dict[str, Any], config: dict[str, Any]) -> StorageConfig | None:
    storage_payload = config.get("storage") or {}
    if payload.get("storage") is not None:
        logger.info("Ignoring client-provided storage config")
    if not isinstance(storage_payload, dict):
        return None
    if storage_payload.get("enabled") is False:
        return None
    bucket = storage_payload.get("bucket")
    if not bucket:
        return None
    provider = storage_payload.get("provider") or "s3"
    endpoint_url = storage_payload.get("endpoint_url") or os.getenv("S3_ENDPOINT_URL")
    region = storage_payload.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    access_key_id = storage_payload.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_access_key = storage_payload.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    prefix = storage_payload.get("prefix") or "gemini-pdf-agent"
    public_url_base = storage_payload.get("public_url_base")
    return StorageConfig(
        provider=str(provider),
        bucket=str(bucket),
        region=str(region) if region else None,
        endpoint_url=str(endpoint_url) if endpoint_url else None,
        access_key_id=str(access_key_id) if access_key_id else None,
        secret_access_key=str(secret_access_key) if secret_access_key else None,
        prefix=str(prefix),
        public_url_base=str(public_url_base) if public_url_base else None,
    )


def _s3_client(storage: StorageConfig):
    import boto3

    return boto3.client(
        "s3",
        region_name=storage.region,
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.access_key_id,
        aws_secret_access_key=storage.secret_access_key,
    )


def _s3_upload_file(client, bucket: str, key: str, path: Path) -> None:
    client.upload_file(str(path), bucket, key)


def _s3_delete_prefix(client, bucket: str, prefix: str) -> None:
    continuation = True
    token: str | None = None
    while continuation:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        contents = response.get("Contents") or []
        if contents:
            client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": item["Key"]} for item in contents]},
            )
        continuation = bool(response.get("IsTruncated"))
        token = response.get("NextContinuationToken")


def _cleanup_workdir(workdir: Path) -> None:
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)


def _cleanup_uploaded_fonts(job_id: str) -> None:
    path = WORKDIR_ROOT / "uploaded_fonts" / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_uploaded_images(job_id: str) -> None:
    path = WORKDIR_ROOT / "uploaded_images" / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_persistent_images(job_id: str) -> None:
    path = WORKDIR_ROOT / "images" / job_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _should_stop(job_id: str) -> bool:
    return job_id in STOP_REQUESTS


def _consume_stop(job_id: str) -> bool:
    return bool(STOP_REQUESTS.pop(job_id, False))


def _cleanup_job(job_id: str, workdir: Path | None, storage: StorageConfig | None) -> None:
    if storage:
        try:
            storage_info = _build_storage_info(storage, job_id)
            client = _s3_client(storage)
            _s3_delete_prefix(client, storage.bucket, storage_info["job_prefix"])
        except Exception:
            logger.exception("Failed to delete storage objects for job %s", job_id)
    if workdir:
        _cleanup_workdir(workdir)
        state_path = _state_path(workdir)
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
    _cleanup_uploaded_fonts(job_id)
    _cleanup_uploaded_images(job_id)
    _cleanup_persistent_images(job_id)
    RESULTS.pop(job_id, None)
    STATES.pop(job_id, None)


def _load_state_for_job(job_id: str) -> tuple[dict[str, Any], Path] | None:
    state_path = STATES.get(job_id)
    if not state_path:
        candidates = list(STATE_DIR.glob("*.json"))
        for candidate in candidates:
            try:
                state = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if state.get("job_id") == job_id:
                STATES[job_id] = candidate
                return state, candidate
        return None
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text(encoding="utf-8")), state_path


def _cleanup_expired_jobs_loop() -> None:
    while True:
        try:
            for state_path in STATE_DIR.glob("*.json"):
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                expires_at = state.get("expires_at")
                if not expires_at:
                    continue
                if _now_utc() < _parse_time(str(expires_at)):
                    continue
                job_id = state.get("job_id")
                storage = state.get("storage") or {}
                if storage and storage.get("bucket") and storage.get("job_prefix"):
                    try:
                        storage_config = StorageConfig(
                            provider=str(storage.get("provider") or "s3"),
                            bucket=str(storage.get("bucket")),
                            region=str(storage.get("region")) if storage.get("region") else None,
                            endpoint_url=str(storage.get("endpoint_url")) if storage.get("endpoint_url") else None,
                            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                            prefix=str(storage.get("prefix") or ""),
                            public_url_base=str(storage.get("public_url_base"))
                            if storage.get("public_url_base")
                            else None,
                        )
                        client = _s3_client(storage_config)
                        _s3_delete_prefix(client, storage_config.bucket, str(storage.get("job_prefix")))
                    except Exception:
                        logger.exception("Failed to delete S3 objects for job %s", job_id)
                workdir_value = state.get("workdir")
                if workdir_value:
                    shutil.rmtree(Path(workdir_value), ignore_errors=True)
                _cleanup_uploaded_fonts(str(job_id))
                _cleanup_uploaded_images(str(job_id))
                _cleanup_persistent_images(str(job_id))
                try:
                    state_path.unlink()
                except FileNotFoundError:
                    pass
                if job_id in RESULTS:
                    RESULTS.pop(job_id, None)
                if job_id in STATES:
                    STATES.pop(job_id, None)
        except Exception:
            logger.exception("Cleanup loop failed")
        time.sleep(60)


def _build_storage_info(storage: StorageConfig, job_id: str) -> dict[str, Any]:
    prefix = storage.prefix.strip("/")
    if prefix:
        job_prefix = f"{prefix}/{job_id}/"
    else:
        job_prefix = f"{job_id}/"
    result_key = f"{job_prefix}result.pdf"
    public_url = _build_public_url(storage, result_key)
    return {
        "provider": storage.provider,
        "bucket": storage.bucket,
        "region": storage.region,
        "endpoint_url": storage.endpoint_url,
        "prefix": storage.prefix,
        "job_prefix": job_prefix,
        "result_key": result_key,
        "state_key": f"{job_prefix}state.json",
        "public_url_base": storage.public_url_base,
        "result_url": public_url,
    }


def _build_public_url(storage: StorageConfig, key: str) -> str | None:
    if not storage.public_url_base:
        return None
    base = storage.public_url_base.rstrip("/")
    return f"{base}/{key}"


def _upload_and_cleanup(
    config: RenderConfig,
    final_pdf: Path,
    last_pages_dir: Path | None,
    images: list[ImageAsset],
    html_body: str,
    combined_css: str,
    iteration: int,
    event_queue: queue.Queue[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    storage_info: dict[str, Any] = {}
    if config.storage:
        storage_info = _build_storage_info(config.storage, config.job_id)

    if last_pages_dir is None:
        raise RuntimeError("No pages available for state save")
    _save_state(config, html_body, combined_css, last_pages_dir, iteration, storage_info)
    state_path = _state_path(config.workdir)

    if config.storage:
        client = _s3_client(config.storage)
        _s3_upload_file(client, config.storage.bucket, storage_info["result_key"], final_pdf)
        event_queue.put(
            (
                "storage",
                {
                    "status": "uploaded",
                    "bucket": config.storage.bucket,
                    "result_key": storage_info["result_key"],
                    "result_url": storage_info.get("result_url"),
                },
            )
        )
        _cleanup_workdir(config.workdir)
    else:
        event_queue.put(("storage", {"status": "skipped"}))

    return storage_info


def _resolve_config(
    payload: dict[str, Any],
    font_entries: list[str],
    images: list[ImageAsset],
) -> RenderConfig:
    config = load_config()
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    model = payload.get("model") or config.get("model") or "gemini-2.0-flash"
    base_url = payload.get("base_url") or config.get("base_url")
    api_mode = payload.get("api_mode") or config.get("api_mode")
    api_key = payload.get("api_key") or config.get("api_key")
    iterations = int(payload.get("iterations") or config.get("iterations") or 2)
    backend = payload.get("backend") or config.get("backend") or "playwright"
    zoom = payload.get("zoom")
    if zoom is None:
        zoom = config.get("zoom") or 2.0
    zoom = float(zoom)
    temperature = payload.get("temperature")
    if temperature is None:
        temperature = config.get("temperature")
    reasoning_effort = payload.get("reasoning_effort") or config.get("reasoning_effort")
    allowed_fonts = list(payload.get("allowed_fonts") or config.get("allowed_fonts") or [])
    font_files = list(payload.get("font_files") or config.get("font_files") or [])
    font_files.extend(font_entries)
    use_fontconfig = payload.get("use_fontconfig")
    if use_fontconfig is None:
        use_fontconfig = config.get("use_fontconfig")
    if use_fontconfig is None:
        use_fontconfig = True
    use_fontconfig = bool(use_fontconfig)
    if use_fontconfig:
        allowed_fonts = list(allowed_fonts) + fontconfig_families()
    cjk_font = payload.get("cjk_font") or config.get("cjk_font")
    return_pdf = payload.get("return_pdf") or "url"
    ttl_seconds = payload.get("ttl_seconds") or config.get("ttl_seconds") or 600
    if "max_image_bytes" in payload:
        max_image_bytes = payload.get("max_image_bytes")
    else:
        max_image_bytes = config.get("max_image_bytes")
    if max_image_bytes is None:
        max_image_bytes = 5_000_000
    max_image_bytes = int(max_image_bytes)
    if max_image_bytes <= 0:
        max_image_bytes = None
    if "max_image_count" in payload:
        max_image_count = payload.get("max_image_count")
    else:
        max_image_count = config.get("max_image_count")
    if max_image_count is None:
        max_image_count = 10
    max_image_count = int(max_image_count)
    if max_image_count <= 0:
        max_image_count = None
    storage = _storage_from_payload(payload, config)
    workdir_value = payload.get("workdir")
    job_id = payload.get("job_id") or uuid.uuid4().hex[:12]
    if max_image_count is not None and len(images) > max_image_count:
        raise ValueError("Too many images uploaded")
    if workdir_value:
        workdir = Path(workdir_value)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = WORKDIR_ROOT / f"run_{timestamp}_{job_id}"
    return RenderConfig(
        prompt=prompt,
        model=str(model),
        base_url=str(base_url) if base_url else None,
        api_mode=str(api_mode) if api_mode else None,
        api_key=str(api_key) if api_key else None,
        iterations=max(iterations, 1),
        backend=str(backend),
        zoom=zoom,
        temperature=float(temperature) if temperature is not None else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort else None,
        allowed_fonts=[str(item) for item in allowed_fonts],
        font_files=[str(item) for item in font_files],
        use_fontconfig=use_fontconfig,
        cjk_font=str(cjk_font) if cjk_font else None,
        return_pdf=str(return_pdf),
        workdir=workdir,
        job_id=str(job_id),
        ttl_seconds=int(ttl_seconds),
        storage=storage,
        images=images,
        max_image_bytes=max_image_bytes,
        max_image_count=max_image_count,
    )


def _resolve_continue_config(
    payload: dict[str, Any],
    state: dict[str, Any],
    font_entries: list[str],
    images: list[ImageAsset],
) -> RenderConfig:
    base_config = state.get("config") or {}
    merged = dict(base_config)
    merged.update(payload)
    merged["job_id"] = payload.get("job_id") or state.get("job_id") or uuid.uuid4().hex[:12]
    merged["prompt"] = payload.get("prompt_append") or payload.get("prompt") or state.get("prompt") or ""
    config = _resolve_config(merged, font_entries, images)
    if config.storage is None:
        storage = state.get("storage") or {}
        if storage.get("bucket"):
            config.storage = StorageConfig(
                provider=str(storage.get("provider") or "s3"),
                bucket=str(storage.get("bucket")),
                region=str(storage.get("region")) if storage.get("region") else None,
                endpoint_url=str(storage.get("endpoint_url")) if storage.get("endpoint_url") else None,
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                prefix=str(storage.get("prefix") or ""),
                public_url_base=str(storage.get("public_url_base"))
                if storage.get("public_url_base")
                else None,
            )
    config.resume_from = payload.get("resume_from") or str(state.get("job_id") or "")
    return config


def _render_worker(config: RenderConfig, event_queue: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
    try:
        ensure_dir(config.workdir)
        event_queue.put(
            (
                "start",
                {
                    "job_id": config.job_id,
                    "workdir": str(config.workdir),
                },
            )
        )

        base_css = load_base_css()
        font_css = build_font_css(
            config.cjk_font,
            allowed_fonts=config.allowed_fonts,
            font_files=config.font_files,
        )
        client = GeminiClient(
            model=config.model,
            base_url=config.base_url,
            allowed_fonts=config.allowed_fonts,
            api_mode=config.api_mode,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            api_key=config.api_key,
        )
        prepared_images = _prepare_images(config)
        image_context = _image_context(prepared_images)
        image_parts = _image_parts(prepared_images)

        html_body = ""
        extra_css = ""
        combined_css = ""
        page_bytes: list[bytes] = []
        last_pdf_path: Path | None = None
        output_pdf_path: Path | None = None
        last_pages_dir: Path | None = None

        i = 0
        for i in range(config.iterations):
            if _should_stop(config.job_id):
                cleanup = _consume_stop(config.job_id)
                if cleanup:
                    _cleanup_job(config.job_id, config.workdir, config.storage)
                event_queue.put(("stopped", {"job_id": config.job_id, "cleanup": cleanup}))
                return
            event_queue.put(
                (
                    "iteration_start",
                    {"iteration": i + 1, "iterations": config.iterations},
                )
            )

            if i == 0:
                draft = client.generate_draft(
                    config.prompt,
                    image_context=image_context or None,
                    images=image_parts,
                )
                html_body = str(draft.get("html_body", ""))
                extra_css = str(draft.get("extra_css", ""))
            else:
                review = client.review_and_revise(
                    config.prompt,
                    html_body,
                    combined_css,
                    page_bytes,
                    image_context=image_context or None,
                    images=image_parts,
                )
                done = bool(review.get("done"))
                prev_html_body = html_body
                prev_extra_css = extra_css
                html_body = str(review.get("html_body", html_body))
                extra_css = str(review.get("css", extra_css))
                issues = review.get("issues")
                changes = review.get("changes")
                has_edits = bool(changes) or html_body != prev_html_body or extra_css != prev_extra_css
                if done and (has_edits or issues):
                    # Require a clean follow-up before honoring done.
                    done = False
                if issues:
                    event_queue.put(("issues", {"iteration": i + 1, "issues": issues}))
                if changes:
                    event_queue.put(("changes", {"iteration": i + 1, "changes": changes}))
                if done and last_pdf_path is not None:
                    event_queue.put(
                        (
                            "early_stop",
                            {"iteration": i + 1, "reason": "model_done"},
                        )
                    )
                    output_pdf_path = last_pdf_path
                    break

            combined_css = "\n".join([base_css, font_css, extra_css])
            html = assemble_html(html_body, combined_css)
            html_path = config.workdir / f"draft_{i:02d}.html"
            pdf_path = config.workdir / f"draft_{i:02d}.pdf"
            pages_dir = config.workdir / f"pages_{i:02d}"
            html_path.write_text(html, encoding="utf-8")

            if _should_stop(config.job_id):
                cleanup = _consume_stop(config.job_id)
                if cleanup:
                    _cleanup_job(config.job_id, config.workdir, config.storage)
                event_queue.put(("stopped", {"job_id": config.job_id, "cleanup": cleanup}))
                return

            render_html_to_pdf(html_path, pdf_path, backend=config.backend)
            page_paths = pdf_to_pngs(pdf_path, pages_dir, zoom=config.zoom)
            page_bytes = [path.read_bytes() for path in page_paths]
            last_pdf_path = pdf_path
            output_pdf_path = pdf_path
            last_pages_dir = pages_dir

            event_queue.put(
                (
                    "rendered",
                    {
                        "iteration": i + 1,
                        "pdf_path": str(pdf_path),
                        "page_count": len(page_paths),
                    },
                )
            )

        if output_pdf_path is None and last_pdf_path is not None:
            output_pdf_path = last_pdf_path

        if output_pdf_path is None:
            raise RuntimeError("No PDF was generated.")

        final_pdf = config.workdir / "result.pdf"
        shutil.copy2(output_pdf_path, final_pdf)
        pdf_base64 = None
        if config.return_pdf == "base64":
            pdf_base64 = base64.b64encode(final_pdf.read_bytes()).decode("ascii")
        RESULTS[config.job_id] = final_pdf
        storage_info = _upload_and_cleanup(
            config,
            final_pdf,
            last_pages_dir,
            prepared_images,
            html_body,
            combined_css,
            i + 1,
            event_queue,
        )

        result_payload: dict[str, Any] = {
            "job_id": config.job_id,
            "pdf_path": str(final_pdf),
            "download_url": f"/v1/results/{config.job_id}",
        }
        if storage_info:
            result_payload["storage"] = storage_info
            if storage_info.get("result_url"):
                result_payload["download_url"] = storage_info["result_url"]
        if pdf_base64 is not None:
            result_payload["pdf_base64"] = pdf_base64
        STOP_REQUESTS.pop(config.job_id, None)
        event_queue.put(("done", result_payload))
    except Exception as exc:
        logger.exception("Render failed")
        STOP_REQUESTS.pop(config.job_id, None)
        event_queue.put(("error", {"message": str(exc)}))


def _continue_worker(
    config: RenderConfig,
    state: dict[str, Any],
    event_queue: queue.Queue[tuple[str, dict[str, Any]]],
) -> None:
    try:
        ensure_dir(config.workdir)
        event_queue.put(
            (
                "start",
                {
                    "job_id": config.job_id,
                    "workdir": str(config.workdir),
                    "resume_from": config.resume_from,
                },
            )
        )

        html_body = str(state.get("html_body", ""))
        combined_css = str(state.get("css", ""))
        prompt_base = str(state.get("prompt", ""))
        prompt_append = str(config.prompt)
        if prompt_append and prompt_append != prompt_base:
            prompt_text = f"{prompt_base}\n\nAdditional request:\n{prompt_append}"
        else:
            prompt_text = prompt_base
        config.prompt = prompt_text

        pages_dir = Path(str(state.get("pages_dir", "")))
        page_paths: list[Path] = []
        if pages_dir.exists():
            page_paths = sorted(pages_dir.glob("page_*.png"))
        if not page_paths:
            if not html_body or not combined_css:
                raise RuntimeError("Saved pages are missing for continuation")
            try:
                resume_html = assemble_html(html_body, combined_css)
                resume_html_path = config.workdir / "resume.html"
                resume_pdf_path = config.workdir / "resume.pdf"
                resume_pages_dir = config.workdir / "resume_pages"
                resume_html_path.write_text(resume_html, encoding="utf-8")
                render_html_to_pdf(resume_html_path, resume_pdf_path, backend=config.backend)
                page_paths = pdf_to_pngs(resume_pdf_path, resume_pages_dir, zoom=config.zoom)
                pages_dir = resume_pages_dir
            except Exception as exc:
                raise RuntimeError("Failed to rebuild pages for continuation") from exc
            if not page_paths:
                raise RuntimeError("Failed to rebuild pages for continuation")
        page_bytes = [path.read_bytes() for path in page_paths]
        base_css = load_base_css()
        font_css = build_font_css(
            config.cjk_font,
            allowed_fonts=config.allowed_fonts,
            font_files=config.font_files,
        )
        client = GeminiClient(
            model=config.model,
            base_url=config.base_url,
            allowed_fonts=config.allowed_fonts,
            api_mode=config.api_mode,
            temperature=config.temperature,
            reasoning_effort=config.reasoning_effort,
            api_key=config.api_key,
        )
        prepared_images = _prepare_images(config)
        image_context = _image_context(prepared_images)
        image_parts = _image_parts(prepared_images)

        output_pdf_path: Path | None = None
        last_pages_dir: Path | None = None
        extra_css = ""
        i = 0
        for i in range(config.iterations):
            if _should_stop(config.job_id):
                cleanup = _consume_stop(config.job_id)
                if cleanup:
                    _cleanup_job(config.job_id, config.workdir, config.storage)
                event_queue.put(("stopped", {"job_id": config.job_id, "cleanup": cleanup}))
                return
            event_queue.put(
                (
                    "iteration_start",
                    {"iteration": i + 1, "iterations": config.iterations},
                )
            )

            review = client.review_and_revise(
                prompt_text,
                html_body,
                combined_css,
                page_bytes,
                image_context=image_context or None,
                images=image_parts,
            )
            done = bool(review.get("done"))
            prev_html_body = html_body
            prev_extra_css = extra_css
            html_body = str(review.get("html_body", html_body))
            extra_css = str(review.get("css", extra_css))
            issues = review.get("issues")
            changes = review.get("changes")
            has_edits = bool(changes) or html_body != prev_html_body or extra_css != prev_extra_css
            if done and (has_edits or issues):
                # Require a clean follow-up before honoring done.
                done = False
            if issues:
                event_queue.put(("issues", {"iteration": i + 1, "issues": issues}))
            if changes:
                event_queue.put(("changes", {"iteration": i + 1, "changes": changes}))

            combined_css = "\n".join([base_css, font_css, extra_css])
            html = assemble_html(html_body, combined_css)
            html_path = config.workdir / f"draft_{i:02d}.html"
            pdf_path = config.workdir / f"draft_{i:02d}.pdf"
            pages_dir = config.workdir / f"pages_{i:02d}"
            html_path.write_text(html, encoding="utf-8")

            if _should_stop(config.job_id):
                cleanup = _consume_stop(config.job_id)
                if cleanup:
                    _cleanup_job(config.job_id, config.workdir, config.storage)
                event_queue.put(("stopped", {"job_id": config.job_id, "cleanup": cleanup}))
                return

            render_html_to_pdf(html_path, pdf_path, backend=config.backend)
            page_paths = pdf_to_pngs(pdf_path, pages_dir, zoom=config.zoom)
            page_bytes = [path.read_bytes() for path in page_paths]
            output_pdf_path = pdf_path
            last_pages_dir = pages_dir

            event_queue.put(
                (
                    "rendered",
                    {
                        "iteration": i + 1,
                        "pdf_path": str(pdf_path),
                        "page_count": len(page_paths),
                    },
                )
            )

            if done:
                event_queue.put(
                    (
                        "early_stop",
                        {"iteration": i + 1, "reason": "model_done"},
                    )
                )
                break

        if output_pdf_path is None:
            raise RuntimeError("No PDF was generated.")

        final_pdf = config.workdir / "result.pdf"
        shutil.copy2(output_pdf_path, final_pdf)
        pdf_base64 = None
        if config.return_pdf == "base64":
            pdf_base64 = base64.b64encode(final_pdf.read_bytes()).decode("ascii")
        RESULTS[config.job_id] = final_pdf
        storage_info = _upload_and_cleanup(
            config,
            final_pdf,
            last_pages_dir,
            prepared_images,
            html_body,
            combined_css,
            i + 1,
            event_queue,
        )

        result_payload: dict[str, Any] = {
            "job_id": config.job_id,
            "pdf_path": str(final_pdf),
            "download_url": f"/v1/results/{config.job_id}",
        }
        if storage_info:
            result_payload["storage"] = storage_info
            if storage_info.get("result_url"):
                result_payload["download_url"] = storage_info["result_url"]
        if pdf_base64 is not None:
            result_payload["pdf_base64"] = pdf_base64
        STOP_REQUESTS.pop(config.job_id, None)
        event_queue.put(("done", result_payload))
    except Exception as exc:
        logger.exception("Continue failed")
        STOP_REQUESTS.pop(config.job_id, None)
        event_queue.put(("error", {"message": str(exc)}))


@app.get("/v1/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/v1/fonts")
def list_fonts() -> JSONResponse:
    return JSONResponse({"fontconfig": fontconfig_families()})


@app.get("/v1/results/{job_id}")
def download_result(job_id: str) -> FileResponse:
    path = RESULTS.get(job_id)
    if not path or not path.exists():
        loaded = _load_state_for_job(job_id)
        if not loaded:
            raise HTTPException(status_code=404, detail="result not found")
        state, _ = loaded
        storage = state.get("storage") or {}
        if not storage.get("bucket") or not storage.get("result_key"):
            raise HTTPException(status_code=404, detail="result not found")
        storage_config = StorageConfig(
            provider=str(storage.get("provider") or "s3"),
            bucket=str(storage.get("bucket")),
            region=str(storage.get("region")) if storage.get("region") else None,
            endpoint_url=str(storage.get("endpoint_url")) if storage.get("endpoint_url") else None,
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            prefix=str(storage.get("prefix") or ""),
            public_url_base=str(storage.get("public_url_base")) if storage.get("public_url_base") else None,
        )
        client = _s3_client(storage_config)
        response = client.get_object(Bucket=storage_config.bucket, Key=str(storage.get("result_key")))
        return StreamingResponse(
            response["Body"].iter_chunks(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={job_id}.pdf"},
        )
    return FileResponse(path, media_type="application/pdf", filename=f"{job_id}.pdf")


@app.post("/v1/stop")
async def stop_render(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    cleanup = bool(payload.get("cleanup", False))
    STOP_REQUESTS[str(job_id)] = cleanup
    return JSONResponse({"status": "stopping", "job_id": str(job_id), "cleanup": cleanup})


@app.post("/v1/cleanup")
async def cleanup_job(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc
    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    loaded = _load_state_for_job(str(job_id))
    workdir: Path | None = None
    storage: StorageConfig | None = None
    if loaded:
        state, _ = loaded
        workdir_value = state.get("workdir")
        if workdir_value:
            workdir = Path(str(workdir_value))
        storage_state = state.get("storage") or {}
        if storage_state.get("bucket"):
            storage = StorageConfig(
                provider=str(storage_state.get("provider") or "s3"),
                bucket=str(storage_state.get("bucket")),
                region=str(storage_state.get("region")) if storage_state.get("region") else None,
                endpoint_url=str(storage_state.get("endpoint_url")) if storage_state.get("endpoint_url") else None,
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                prefix=str(storage_state.get("prefix") or ""),
                public_url_base=str(storage_state.get("public_url_base"))
                if storage_state.get("public_url_base")
                else None,
            )
    _cleanup_job(str(job_id), workdir, storage)
    return JSONResponse({"status": "cleaned", "job_id": str(job_id)})


@app.post("/v1/render")
async def render(request: Request) -> StreamingResponse:
    content_type = request.headers.get("content-type", "")
    payload: dict[str, Any]
    font_uploads: list[Any] = []
    image_uploads: list[Any] = []
    if "multipart/form-data" in content_type:
        form = await request.form()
        payload_raw = form.get("payload")
        if not payload_raw:
            raise HTTPException(status_code=400, detail="payload is required in form data")
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="payload must be JSON") from exc
        font_uploads = list(form.getlist("fonts"))
        image_uploads = list(form.getlist("images"))
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    job_id = payload.get("job_id") or uuid.uuid4().hex[:12]
    payload["job_id"] = job_id

    config_defaults = load_config()
    if "max_image_bytes" in payload:
        max_image_bytes = payload.get("max_image_bytes")
    else:
        max_image_bytes = config_defaults.get("max_image_bytes")
    if max_image_bytes is None:
        max_image_bytes = 5_000_000
    max_image_bytes = int(max_image_bytes)
    if max_image_bytes <= 0:
        max_image_bytes = None
    if "max_image_count" in payload:
        max_image_count = payload.get("max_image_count")
    else:
        max_image_count = config_defaults.get("max_image_count")
    if max_image_count is None:
        max_image_count = 10
    max_image_count = int(max_image_count)
    if max_image_count <= 0:
        max_image_count = None

    font_families = payload.get("font_families") or []
    font_entries = []
    if font_uploads:
        font_entries = _save_uploaded_fonts(
            font_uploads, list(font_families), WORKDIR_ROOT / "uploaded_fonts" / job_id
        )

    image_names = payload.get("image_names") or []
    images: list[ImageAsset] = []
    if image_uploads:
        try:
            if max_image_count is not None and len(image_uploads) > max_image_count:
                raise HTTPException(status_code=400, detail="Too many images uploaded")
            images = _save_uploaded_images(
                image_uploads,
                list(image_names),
                WORKDIR_ROOT / "uploaded_images" / job_id,
                max_bytes=max_image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        config = _resolve_config(payload, font_entries, images)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    thread = threading.Thread(
        target=_render_worker,
        args=(config, event_queue),
        daemon=True,
    )
    thread.start()

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event, data = await asyncio.to_thread(event_queue.get)
            yield _format_sse(event, data)
            if event in {"done", "error", "stopped"}:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/continue")
async def continue_render(request: Request) -> StreamingResponse:
    content_type = request.headers.get("content-type", "")
    payload: dict[str, Any]
    font_uploads: list[Any] = []
    image_uploads: list[Any] = []
    if "multipart/form-data" in content_type:
        form = await request.form()
        payload_raw = form.get("payload")
        if not payload_raw:
            raise HTTPException(status_code=400, detail="payload is required in form data")
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="payload must be JSON") from exc
        font_uploads = list(form.getlist("fonts"))
        image_uploads = list(form.getlist("images"))
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    job_id = payload.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required")
    loaded = _load_state_for_job(job_id)
    if not loaded:
        raise HTTPException(status_code=404, detail="state not found for job_id")
    state, _ = loaded

    config_defaults = load_config()
    if "max_image_bytes" in payload:
        max_image_bytes = payload.get("max_image_bytes")
    else:
        max_image_bytes = config_defaults.get("max_image_bytes")
    if max_image_bytes is None:
        max_image_bytes = state.get("config", {}).get("max_image_bytes")
    if max_image_bytes is None:
        max_image_bytes = 5_000_000
    max_image_bytes = int(max_image_bytes)
    if max_image_bytes <= 0:
        max_image_bytes = None
    if "max_image_count" in payload:
        max_image_count = payload.get("max_image_count")
    else:
        max_image_count = config_defaults.get("max_image_count")
    if max_image_count is None:
        max_image_count = state.get("config", {}).get("max_image_count")
    if max_image_count is None:
        max_image_count = 10
    max_image_count = int(max_image_count)
    if max_image_count <= 0:
        max_image_count = None

    font_families = payload.get("font_families") or []
    font_entries = []
    if font_uploads:
        font_entries = _save_uploaded_fonts(
            font_uploads, list(font_families), WORKDIR_ROOT / "uploaded_fonts" / job_id
        )

    image_names = payload.get("image_names") or []
    images = _deserialize_images(state)
    if max_image_count is not None and len(images) > max_image_count:
        raise HTTPException(status_code=400, detail="Too many images in state")
    if image_uploads:
        try:
            if max_image_count is not None and len(images) + len(image_uploads) > max_image_count:
                raise HTTPException(status_code=400, detail="Too many images uploaded")
            uploaded_images = _save_uploaded_images(
                image_uploads,
                list(image_names),
                WORKDIR_ROOT / "uploaded_images" / job_id,
                max_bytes=max_image_bytes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        images.extend(uploaded_images)

    try:
        config = _resolve_continue_config(payload, state, font_entries, images)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    thread = threading.Thread(
        target=_continue_worker,
        args=(config, state, event_queue),
        daemon=True,
    )
    thread.start()

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            event, data = await asyncio.to_thread(event_queue.get)
            yield _format_sse(event, data)
            if event in {"done", "error", "stopped"}:
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


@app.on_event("startup")
def _startup() -> None:
    global _CLEANUP_STARTED
    ensure_dir(WORKDIR_ROOT)
    ensure_dir(STATE_DIR)
    if not _CLEANUP_STARTED:
        thread = threading.Thread(target=_cleanup_expired_jobs_loop, daemon=True)
        thread.start()
        _CLEANUP_STARTED = True


def main() -> None:
    parser = argparse.ArgumentParser(description="gemini-pdf-agent HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--config", help="Path to config.json")
    args = parser.parse_args()

    _setup_logging()
    ensure_dir(WORKDIR_ROOT)
    if args.config:
        os.environ["GEMINI_PDF_AGENT_CONFIG"] = args.config

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError("Uvicorn is required. Run: pip install uvicorn") from exc

    uvicorn.run(
        "gemini_pdf_agent.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
