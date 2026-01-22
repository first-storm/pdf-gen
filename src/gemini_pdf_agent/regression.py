from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def image_diff_score(image_a: Path, image_b: Path) -> float:
    img_a = Image.open(image_a).convert("RGB")
    img_b = Image.open(image_b).convert("RGB")

    if img_a.size != img_b.size:
        return 1.0

    arr_a = np.asarray(img_a, dtype=np.float32) / 255.0
    arr_b = np.asarray(img_b, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(arr_a - arr_b)))


def regression_compare(
    baseline_dir: Path,
    pages_dir: Path,
    diff_dir: Path,
    threshold: float = 0.005,
) -> dict:
    ensure_dir(diff_dir)

    report: dict[str, object] = {
        "pages": {},
        "worst_page": None,
        "worst_score": 0.0,
        "passed": True,
        "missing": [],
    }

    page_paths = sorted(pages_dir.glob("page_*.png"))
    if not page_paths:
        report["passed"] = False
        return report

    for page_path in page_paths:
        baseline_path = baseline_dir / page_path.name
        if not baseline_path.exists():
            report["missing"].append(page_path.name)
            report["passed"] = False
            continue

        score = image_diff_score(baseline_path, page_path)
        report["pages"][page_path.name] = score
        if score > report["worst_score"]:
            report["worst_score"] = score
            report["worst_page"] = page_path.name
        if score > threshold:
            report["passed"] = False

        diff_image = ImageChops.difference(
            Image.open(baseline_path).convert("RGB"),
            Image.open(page_path).convert("RGB"),
        )
        diff_path = diff_dir / f"diff_{page_path.name}"
        diff_image.save(diff_path)

    return report


def save_report(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
