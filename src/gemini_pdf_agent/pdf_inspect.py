from __future__ import annotations

import logging
from pathlib import Path

import fitz

from .utils import ensure_dir

logger = logging.getLogger(__name__)


def pdf_to_pngs(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> list[Path]:
    ensure_dir(out_dir)
    doc = fitz.open(str(pdf_path))
    paths: list[Path] = []
    matrix = fitz.Matrix(zoom, zoom)

    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        filename = f"page_{idx:03d}.png"
        out_path = out_dir / filename
        pix.save(str(out_path))
        paths.append(out_path)

    return paths
