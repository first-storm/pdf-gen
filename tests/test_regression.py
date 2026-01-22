from pathlib import Path

import numpy as np
from PIL import Image

from gemini_pdf_agent.regression import image_diff_score, regression_compare


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    data = np.zeros((10, 10, 3), dtype=np.uint8)
    data[:, :] = color
    Image.fromarray(data, "RGB").save(path)


def test_image_diff_score_identical(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_image(a, (0, 0, 0))
    _write_image(b, (0, 0, 0))
    assert image_diff_score(a, b) == 0.0


def test_image_diff_score_different(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_image(a, (0, 0, 0))
    _write_image(b, (255, 255, 255))
    assert image_diff_score(a, b) > 0.5


def test_regression_compare_missing_baseline(tmp_path):
    baseline = tmp_path / "baseline"
    pages = tmp_path / "pages"
    diff = tmp_path / "diff"
    baseline.mkdir()
    pages.mkdir()
    _write_image(pages / "page_001.png", (0, 0, 0))

    report = regression_compare(baseline, pages, diff, threshold=0.005)
    assert report["passed"] is False
    assert report["missing"]
