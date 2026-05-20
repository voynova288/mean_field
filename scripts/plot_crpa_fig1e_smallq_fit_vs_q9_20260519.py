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


DEFAULT_Q9_CURVE = (
    Path("results")
    / "TBG_HF_cRPA"
    / "crpa_lk24_lg9_q9_zhang_appendix_fig4_merged"
    / "epsilon_fig1e_window_curve.csv"
)
DEFAULT_Q11_CURVE = (
    Path("results")
    / "TBG_HF_cRPA"
    / "crpa_lk24_lg9_q11_zhang_recheck_20260516_merged"
    / "epsilon_fig1e_window_curve.csv"
)
DEFAULT_OUTPUT_DIR = Path("results") / "TBG_HF_cRPA" / "crpa_hf_logic_gate_20260519" / "smallq_fit_vs_q9"


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q: list[float] = []
    eps: list[float] = []
    counts: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            q.append(float(row["q_abs_nm_inv"]))
            eps.append(float(row["eps_total_median"]))
            counts.append(float(row.get("n_points", 0.0)))
    order = np.argsort(q)
    return (
        np.asarray(q, dtype=float)[order],
        np.asarray(eps, dtype=float)[order],
        np.asarray(counts, dtype=float)[order],
    )


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        y_new[idx] = h00 * y[interval] + h10 * h * d[interval] + h01 * y[interval + 1] + h11 * h * d[interval + 1]
    return y_new


def _write_selected_points(path: Path, q: np.ndarray, eps: np.ndarray, q9: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_nm_inv", "q11_selected_eps_times_bn", "q9_reference_eps_times_bn", "q11_minus_q9"])
        for row in zip(q, eps, q9, eps - q9, strict=True):
            writer.writerow([f"{value:.12g}" for value in row])


def _write_fit_curve(path: Path, q: np.ndarray, selected_fit: np.ndarray, q9_fit: np.ndarray, q11_full: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_nm_inv",
                "selected_point_fit_eps_times_bn",
                "q9_reference_eps_times_bn",
                "q11_full_curve_eps_times_bn",
                "selected_fit_minus_q9",
                "q11_full_minus_q9",
            ]
        )
        for row in zip(q, selected_fit, q9_fit, q11_full, selected_fit - q9_fit, q11_full - q9_fit, strict=True):
            writer.writerow([f"{value:.12g}" for value in row])


def _plot(
    path: Path,
    dense_q: np.ndarray,
    q9_fit: np.ndarray,
    q11_full_fit: np.ndarray,
    selected_fit: np.ndarray,
    selected_q: np.ndarray,
    selected_eps: np.ndarray,
) -> None:
    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(7.6, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
        constrained_layout=True,
    )

    ax.plot(dense_q, q9_fit, color="#111111", linewidth=2.2, label="q9 original window curve")
    ax.plot(dense_q, q11_full_fit, color="#7f7f7f", linewidth=1.6, linestyle="--", label="q11 full window curve")
    ax.plot(dense_q, selected_fit, color="#1f77b4", linewidth=2.1, label="q11 selected-point fit")
    ax.scatter(selected_q, selected_eps, color="#1f77b4", s=36, zorder=4, label="selected q11 points")
    ax.set_xlim(0.0, 1.2)
    ax.set_ylabel(r"$\epsilon(q)\,\epsilon_{\rm BN}$")
    ax.set_title("Fig. 1(e) Window: Small-q Selected-Point Fit vs q9 Curve")
    ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.85)
    ax.legend(frameon=False, fontsize=9, loc="best")

    residual_ax.axhline(0.0, color="#111111", linewidth=1.0)
    residual_ax.plot(dense_q, selected_fit - q9_fit, color="#d62728", linewidth=2.0, label="selected fit - q9")
    residual_ax.plot(dense_q, q11_full_fit - q9_fit, color="#7f7f7f", linewidth=1.2, linestyle="--", label="q11 full - q9")
    residual_ax.scatter(selected_q, selected_eps - np.interp(selected_q, dense_q, q9_fit), color="#d62728", s=28, zorder=4)
    residual_ax.set_xlim(0.0, 1.2)
    residual_ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    residual_ax.set_ylabel("diff")
    residual_ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.85)
    residual_ax.legend(frameon=False, fontsize=8, loc="best")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a small-q selected-point q11 fit against the q9 Fig. 1(e) curve.")
    parser.add_argument("--q9-curve", type=Path, default=DEFAULT_Q9_CURVE)
    parser.add_argument("--q11-curve", type=Path, default=DEFAULT_Q11_CURVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selected-q",
        default="0,0.04,0.08,0.12,0.16,0.20,0.30,0.40,0.50,0.60,0.80,1.00,1.20",
        help="Comma-separated q values in nm^-1 for the selected-point q11 fit.",
    )
    args = parser.parse_args()

    q9_q, q9_eps, _ = _load_curve(args.q9_curve)
    q11_q, q11_eps, _ = _load_curve(args.q11_curve)
    selected_q = np.asarray([float(piece) for piece in str(args.selected_q).split(",") if piece.strip()], dtype=float)
    selected_q.sort()
    selected_eps = np.interp(selected_q, q11_q, q11_eps)
    selected_q9_eps = np.interp(selected_q, q9_q, q9_eps)

    dense_q = np.linspace(0.0, 1.2, 600)
    q9_fit = np.interp(dense_q, q9_q, q9_eps)
    q11_full_fit = np.interp(dense_q, q11_q, q11_eps)
    selected_fit = _pchip_eval(selected_q, selected_eps, dense_q)

    out = Path(args.output_dir)
    plot_path = out / "fig1e_smallq_selected_fit_vs_q9.png"
    selected_path = out / "selected_points.csv"
    curve_path = out / "fit_curve.csv"
    summary_path = out / "summary.json"

    _plot(plot_path, dense_q, q9_fit, q11_full_fit, selected_fit, selected_q, selected_eps)
    _write_selected_points(selected_path, selected_q, selected_eps, selected_q9_eps)
    _write_fit_curve(curve_path, dense_q, selected_fit, q9_fit, q11_full_fit)

    residual = selected_fit - q9_fit
    full_residual = q11_full_fit - q9_fit
    summary = {
        "q9_curve": str(args.q9_curve),
        "q11_curve": str(args.q11_curve),
        "selected_q": [float(v) for v in selected_q],
        "fit": "shape_preserving_pchip_through_selected_q11_points",
        "q_range_nm_inv": [0.0, 1.2],
        "selected_fit_vs_q9_rmse": float(np.sqrt(np.mean(residual * residual))),
        "selected_fit_vs_q9_max_abs": float(np.max(np.abs(residual))),
        "selected_fit_vs_q9_mean_abs": float(np.mean(np.abs(residual))),
        "q11_full_vs_q9_rmse": float(np.sqrt(np.mean(full_residual * full_residual))),
        "q11_full_vs_q9_max_abs": float(np.max(np.abs(full_residual))),
        "q11_full_vs_q9_mean_abs": float(np.mean(np.abs(full_residual))),
        "plot_png": str(plot_path),
        "plot_pdf": str(plot_path.with_suffix(".pdf")),
        "selected_points_csv": str(selected_path),
        "fit_curve_csv": str(curve_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[fig1e-smallq-fit] plot_png={plot_path}")
    print(f"[fig1e-smallq-fit] plot_pdf={plot_path.with_suffix('.pdf')}")
    print(f"[fig1e-smallq-fit] selected_points_csv={selected_path}")
    print(f"[fig1e-smallq-fit] fit_curve_csv={curve_path}")
    print(f"[fig1e-smallq-fit] summary_json={summary_path}")
    print(
        "[fig1e-smallq-fit] "
        f"selected_fit_vs_q9_rmse={summary['selected_fit_vs_q9_rmse']:.12g} "
        f"max_abs={summary['selected_fit_vs_q9_max_abs']:.12g} "
        f"mean_abs={summary['selected_fit_vs_q9_mean_abs']:.12g}"
    )
    print(
        "[fig1e-smallq-fit] "
        f"q11_full_vs_q9_rmse={summary['q11_full_vs_q9_rmse']:.12g} "
        f"max_abs={summary['q11_full_vs_q9_max_abs']:.12g} "
        f"mean_abs={summary['q11_full_vs_q9_mean_abs']:.12g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
