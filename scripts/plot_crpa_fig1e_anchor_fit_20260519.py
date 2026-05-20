#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = (
    Path("results")
    / "TBG_HF_cRPA"
    / "crpa_hf_logic_gate_20260519"
    / "fig1e_representative_points.csv"
)
DEFAULT_OUTPUT_DIR = Path("results") / "TBG_HF_cRPA" / "crpa_hf_logic_gate_20260519" / "anchor_fit"


def _load_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_values: list[float] = []
    computed: list[float] = []
    paper: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            q_values.append(float(row["q_nm_inv"]))
            computed.append(float(row["computed_eps_times_bn"]))
            paper.append(float(row["paper_eps_times_bn"]))
    if len(q_values) < 3:
        raise ValueError(f"Need at least three anchor points, got {len(q_values)} from {path}")
    order = np.argsort(q_values)
    return (
        np.asarray(q_values, dtype=float)[order],
        np.asarray(computed, dtype=float)[order],
        np.asarray(paper, dtype=float)[order],
    )


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic slopes for a shape-preserving fit."""

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    if y.size == 2:
        d[:] = delta[0]
        return d

    for i in range(1, y.size - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            d[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d[0]) != np.sign(delta[0]):
        d[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d[0]) > abs(3.0 * delta[0]):
        d[0] = 3.0 * delta[0]

    d[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(d[-1]) != np.sign(delta[-1]):
        d[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(d[-1]) > abs(3.0 * delta[-1]):
        d[-1] = 3.0 * delta[-1]

    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    d = _pchip_slopes(x, y)
    y_new = np.empty_like(x_new, dtype=float)
    for idx, value in enumerate(x_new):
        if value <= x[0]:
            interval = 0
        elif value >= x[-1]:
            interval = x.size - 2
        else:
            interval = int(np.searchsorted(x, value) - 1)
        h = x[interval + 1] - x[interval]
        t = (value - x[interval]) / h
        h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
        h10 = t * (1.0 - t) ** 2
        h01 = t**2 * (3.0 - 2.0 * t)
        h11 = t**2 * (t - 1.0)
        y_new[idx] = (
            h00 * y[interval]
            + h10 * h * d[interval]
            + h01 * y[interval + 1]
            + h11 * h * d[interval + 1]
        )
    return y_new


def _write_curve(path: Path, x: np.ndarray, computed: np.ndarray, paper: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_nm_inv", "computed_fit_eps_times_bn", "paper_fit_eps_times_bn", "computed_minus_paper"])
        for row in zip(x, computed, paper, computed - paper, strict=True):
            writer.writerow([f"{value:.12g}" for value in row])


def _plot(
    output_path: Path,
    q: np.ndarray,
    computed: np.ndarray,
    paper: np.ndarray,
    q_dense: np.ndarray,
    computed_fit: np.ndarray,
    paper_fit: np.ndarray,
) -> None:
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        constrained_layout=True,
    )

    ax.plot(q_dense, paper_fit, color="#111111", linewidth=2.2, label="Paper anchors fit")
    ax.scatter(q, paper, color="#111111", marker="x", s=56, linewidths=1.8, label="Paper anchors")
    ax.plot(q_dense, computed_fit, color="#1f77b4", linewidth=2.2, label="cRPA computed fit")
    ax.scatter(q, computed, color="#1f77b4", s=40, label="cRPA computed anchors")
    ax.set_ylabel(r"$\epsilon(q)\,\epsilon_{\rm BN}$")
    ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Zhang Fig. 1(e) Anchor Fit")

    residual = computed - paper
    residual_fit = computed_fit - paper_fit
    residual_ax.axhline(0.0, color="#111111", linewidth=1.0)
    residual_ax.plot(q_dense, residual_fit, color="#d62728", linewidth=2.0)
    residual_ax.scatter(q, residual, color="#d62728", s=34)
    residual_ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    residual_ax.set_ylabel("diff")
    residual_ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit smooth curves through the sparse Zhang Fig. 1(e) anchor table.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    q, computed, paper = _load_points(args.input_csv)
    q_dense = np.linspace(float(np.min(q)), float(np.max(q)), 400)
    computed_fit = _pchip_eval(q, computed, q_dense)
    paper_fit = _pchip_eval(q, paper, q_dense)

    output_dir = Path(args.output_dir)
    plot_path = output_dir / "fig1e_anchor_fit.png"
    curve_path = output_dir / "fig1e_anchor_fit_curve.csv"
    summary_path = output_dir / "fig1e_anchor_fit_summary.json"

    _plot(plot_path, q, computed, paper, q_dense, computed_fit, paper_fit)
    _write_curve(curve_path, q_dense, computed_fit, paper_fit)

    diff = computed - paper
    summary = {
        "input_csv": str(args.input_csv),
        "fit": "shape_preserving_pchip_through_sparse_anchors",
        "point_count": int(q.size),
        "q_min_nm_inv": float(np.min(q)),
        "q_max_nm_inv": float(np.max(q)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "plot_png": str(plot_path),
        "plot_pdf": str(plot_path.with_suffix(".pdf")),
        "curve_csv": str(curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[fig1e-anchor-fit] plot_png={plot_path}")
    print(f"[fig1e-anchor-fit] plot_pdf={plot_path.with_suffix('.pdf')}")
    print(f"[fig1e-anchor-fit] curve_csv={curve_path}")
    print(f"[fig1e-anchor-fit] summary_json={summary_path}")
    print(
        "[fig1e-anchor-fit] "
        f"rmse={summary['rmse']:.12g} max_abs={summary['max_abs']:.12g} mean_abs={summary['mean_abs']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
