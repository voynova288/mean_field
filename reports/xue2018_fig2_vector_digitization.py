from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PAPER_FIGURE = Path(
    "/data/home/ziyuzhu/Mean_Field/tmp/pdfs/xue2018_nematic/"
    "arxiv_source/unpacked/path.pdf"
)
PAPER_PDF = Path(
    "/data/home/ziyuzhu/Mean_Field/tmp/pdfs/xue2018_nematic/arxiv_1710.00410.pdf"
)
OUTPUT = Path(__file__).resolve().parent / "data/xue2018_fig2_digitized_markers.json"
COLORS = {
    "black": "fill:rgb(0%,0%,0%)",
    "red": "fill:rgb(100%,0%,0%)",
    "blue": "fill:rgb(0%,0%,100%)",
}
# Major ticks transformed into PDF page coordinates.  The zero tick is shared
# by both vertical axes; the next major tick is 0.5 in the corresponding units.
PHI_Y_ZERO = 412.429687413251
PHI_Y_HALF = 333.554687310687
GAP_Y_ZERO = 412.429687413251
GAP_Y_HALF = 349.136718962395


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_center(path_data: str) -> tuple[float, float, float, float]:
    numbers = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path_data)]
    if len(numbers) < 4:
        raise ValueError("marker path has too few coordinates")
    x_values = numbers[0::2]
    y_values = numbers[1::2]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return 0.5 * (x_min + x_max), 0.5 * (y_min + y_max), x_max - x_min, y_max - y_min


def extract_markers(svg_path: Path, color: str) -> list[tuple[float, float, float, float]]:
    root = ET.parse(svg_path).getroot()
    rows = []
    for element in root.iter():
        if not element.tag.endswith("path") or color not in element.attrib.get("style", ""):
            continue
        center = path_center(element.attrib.get("d", ""))
        x_value, y_value, _width, _height = center
        if 105.0 < x_value < 750.0 and 45.0 < y_value < 420.0:
            rows.append(center)
    rows.sort(key=lambda row: row[0])
    if len(rows) != 62:
        raise ValueError(f"expected 62 vector markers, found {len(rows)}")
    return rows


def calibrated_rows(
    markers: list[tuple[float, float, float, float]],
    *,
    y_zero: float,
    y_half: float,
) -> list[dict[str, object]]:
    scale = 0.5 / (y_zero - y_half)
    return [
        {
            "sequence": index,
            "x_pdf_pt": x_value,
            "y_pdf_pt": y_value,
            "value": (y_zero - y_value) * scale,
            "bbox_pdf_pt": [width, height],
            "sampling": "official_figure_vector_path_bbox_center",
        }
        for index, (x_value, y_value, width, height) in enumerate(markers, 1)
    ]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xue2018-vector-") as directory:
        svg_path = Path(directory) / "path.svg"
        subprocess.run(
            ["pdftocairo", "-svg", str(PAPER_FIGURE), str(svg_path)],
            check=True,
        )
        markers = {
            name: extract_markers(svg_path, color)
            for name, color in COLORS.items()
        }
    series = {
        "black": calibrated_rows(markers["black"], y_zero=PHI_Y_ZERO, y_half=PHI_Y_HALF),
        "red": calibrated_rows(markers["red"], y_zero=GAP_Y_ZERO, y_half=GAP_Y_HALF),
        "blue": calibrated_rows(markers["blue"], y_zero=GAP_Y_ZERO, y_half=GAP_Y_HALF),
    }
    payload = {
        "schema": "xue2018-fig2-vector-marker-digitization-v3",
        "authority": "official_paper_figure_vector_digitization_comparison_only",
        "source_figure": str(PAPER_FIGURE),
        "source_figure_sha256": sha256(PAPER_FIGURE),
        "paper_pdf_sha256": sha256(PAPER_PDF),
        "extraction": {
            "command": "pdftocairo -svg path.pdf path.svg",
            "marker_center": "bounding-box center of each filled vector marker path",
            "plot_bounds_pdf_pt": {"x": [105.0, 750.0], "y": [45.0, 420.0]},
            "marker_counts": {name: len(rows) for name, rows in markers.items()},
        },
        "calibration": {
            "phi_y_zero_pdf_pt": PHI_Y_ZERO,
            "phi_y_0p5_pdf_pt": PHI_Y_HALF,
            "gap_y_zero_pdf_pt": GAP_Y_ZERO,
            "gap_y_0p5_pdf_pt": GAP_Y_HALF,
            "shared_zero_tick": True,
            "estimated_marker_value_uncertainty_ry": 0.001,
        },
        "supersedes": {
            "schema": "xue2018-fig2-image-marker-digitization-v2",
            "reason": "The raster gap calibration used y_zero=900 px instead of the shared zero-axis tick near 908 px; vector axes and markers remove that offset and line/marker ambiguity.",
        },
        "series": series,
        "validation": {
            "blue_p24_p26_ry": [series["blue"][index - 1]["value"] for index in (24, 25, 26)],
            "warning": "These values are digitized source evidence, not author raw arrays or independent calculations.",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT), "validation": payload["validation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
