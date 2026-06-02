#!/usr/bin/env python3
"""Compare several cRPA epsilon radial curves against sparse paper anchors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Curve:
    label: str
    path: Path
    q_nm_inv: np.ndarray
    eps_times_bn: np.ndarray
    n_points: np.ndarray


def _read_curve(label: str, path: Path) -> Curve:
    q_values: list[float] = []
    eps_values: list[float] = []
    counts: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"q_abs_nm_inv", "eps_total_median", "n_points"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            q_values.append(float(row["q_abs_nm_inv"]))
            eps_values.append(float(row["eps_total_median"]))
            counts.append(int(float(row["n_points"])))
    order = np.argsort(np.asarray(q_values, dtype=float))
    return Curve(
        label=label,
        path=path,
        q_nm_inv=np.asarray(q_values, dtype=float)[order],
        eps_times_bn=np.asarray(eps_values, dtype=float)[order],
        n_points=np.asarray(counts, dtype=int)[order],
    )


def _read_paper_anchors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    q_values: list[float] = []
    eps_values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"q_nm_inv", "paper_eps_times_bn"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        for row in reader:
            q_values.append(float(row["q_nm_inv"]))
            eps_values.append(float(row["paper_eps_times_bn"]))
    order = np.argsort(np.asarray(q_values, dtype=float))
    return np.asarray(q_values, dtype=float)[order], np.asarray(eps_values, dtype=float)[order]


def _parse_curve_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("curve arguments must be LABEL=PATH")
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("curve label must be non-empty")
    return label, Path(path_text)


def _write_anchor_table(path: Path, curves: list[Curve], anchor_q: np.ndarray, paper_eps: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["curve_label", "q_nm_inv", "paper_eps_times_bn", "curve_eps_times_bn", "curve_minus_paper"])
        for curve in curves:
            if anchor_q.min() < curve.q_nm_inv.min() or anchor_q.max() > curve.q_nm_inv.max():
                raise ValueError(f"{curve.label} does not cover the paper anchor q range")
            interp = np.interp(anchor_q, curve.q_nm_inv, curve.eps_times_bn)
            for q_value, paper_value, curve_value in zip(anchor_q, paper_eps, interp, strict=True):
                writer.writerow(
                    [
                        curve.label,
                        f"{q_value:.12g}",
                        f"{paper_value:.12g}",
                        f"{curve_value:.12g}",
                        f"{curve_value - paper_value:.12g}",
                    ]
                )


def _write_metric_table(path: Path, curves: list[Curve], anchor_q: np.ndarray, paper_eps: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["curve_label", "rmse", "mean_diff", "median_diff", "max_abs_diff"])
        for curve in curves:
            interp = np.interp(anchor_q, curve.q_nm_inv, curve.eps_times_bn)
            diff = interp - paper_eps
            writer.writerow(
                [
                    curve.label,
                    f"{np.sqrt(np.mean(diff**2)):.12g}",
                    f"{np.mean(diff):.12g}",
                    f"{np.median(diff):.12g}",
                    f"{np.max(np.abs(diff)):.12g}",
                ]
            )


def _write_summary(
    path: Path,
    *,
    curves: list[Curve],
    anchors_path: Path,
    metric_csv: Path,
    anchor_csv: Path,
    plot_png: Path,
    plot_pdf: Path,
) -> None:
    lines = [
        "# cRPA Epsilon Curve Comparison",
        "",
        "This is a lightweight postprocess of existing cRPA radial-median curves.",
        "It does not rerun cRPA or HF.",
        "",
        "## Inputs",
        "",
        f"- paper anchors: `{anchors_path}`",
    ]
    for curve in curves:
        lines.append(f"- {curve.label}: `{curve.path}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{metric_csv}`",
            f"- `{anchor_csv}`",
            f"- `{plot_png}`",
            f"- `{plot_pdf}`",
            "",
            "Interpretation note: curves labelled `hf-compatible lg9/q11` use the old lg9/q11 artifact and are convention diagnostics only; that artifact is alias-invalid for final HF-compatible production.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(path_png: Path, path_pdf: Path, curves: list[Curve], anchor_q: np.ndarray, paper_eps: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for curve in curves:
        mask = curve.q_nm_inv <= 1.2
        ax.plot(curve.q_nm_inv[mask], curve.eps_times_bn[mask], linewidth=1.8, marker="o", markersize=2.5, label=curve.label)
    ax.scatter(anchor_q, paper_eps, marker="x", color="#d62728", s=65, linewidth=1.9, label="paper anchors")
    ax.set_xlabel("q (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon(q)\,\epsilon_{\mathrm{BN}}$")
    ax.set_xlim(0.0, 1.2)
    ax.set_ylim(0.0, None)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, loc="best", fontsize=9)
    ax.set_title("cRPA epsilon radial-median convention comparison")
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", required=True, type=_parse_curve_arg, help="LABEL=PATH")
    parser.add_argument("--paper-anchors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    curves = [_read_curve(label, path) for label, path in args.curve]
    anchor_q, paper_eps = _read_paper_anchors(args.paper_anchors)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_csv = args.output_dir / "epsilon_curve_anchor_metrics.csv"
    anchor_csv = args.output_dir / "epsilon_curve_anchor_values.csv"
    plot_png = args.output_dir / "epsilon_curve_convention_comparison.png"
    plot_pdf = args.output_dir / "epsilon_curve_convention_comparison.pdf"
    summary_md = args.output_dir / "summary.md"

    _write_metric_table(metric_csv, curves, anchor_q, paper_eps)
    _write_anchor_table(anchor_csv, curves, anchor_q, paper_eps)
    _plot(plot_png, plot_pdf, curves, anchor_q, paper_eps)
    _write_summary(
        summary_md,
        curves=curves,
        anchors_path=args.paper_anchors,
        metric_csv=metric_csv,
        anchor_csv=anchor_csv,
        plot_png=plot_png,
        plot_pdf=plot_pdf,
    )

    print(f"wrote {metric_csv}")
    print(f"wrote {anchor_csv}")
    print(f"wrote {plot_png}")
    print(f"wrote {plot_pdf}")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
