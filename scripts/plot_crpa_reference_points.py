#!/usr/bin/env python3
"""Overlay a cRPA epsilon radial curve with sparse paper reference points."""

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
class ComparisonRow:
    q_nm_inv: float
    paper_eps_times_bn: float
    computed_eps_times_bn: float

    @property
    def diff(self) -> float:
        return self.computed_eps_times_bn - self.paper_eps_times_bn


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_values: list[float] = []
    eps_values: list[float] = []
    counts: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"q_abs_nm_inv", "eps_total_median", "n_points"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            q_values.append(float(row["q_abs_nm_inv"]))
            eps_values.append(float(row["eps_total_median"]))
            counts.append(int(float(row["n_points"])))
    if not q_values:
        raise ValueError(f"{path} contains no curve rows")
    order = np.argsort(np.asarray(q_values, dtype=float))
    return (
        np.asarray(q_values, dtype=float)[order],
        np.asarray(eps_values, dtype=float)[order],
        np.asarray(counts, dtype=int)[order],
    )


def _read_paper_anchors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    q_values: list[float] = []
    eps_values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"q_nm_inv", "paper_eps_times_bn"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            q_values.append(float(row["q_nm_inv"]))
            eps_values.append(float(row["paper_eps_times_bn"]))
    if not q_values:
        raise ValueError(f"{path} contains no paper anchor rows")
    order = np.argsort(np.asarray(q_values, dtype=float))
    return (
        np.asarray(q_values, dtype=float)[order],
        np.asarray(eps_values, dtype=float)[order],
    )


def _nearest_counts(curve_q: np.ndarray, curve_counts: np.ndarray, anchor_q: np.ndarray) -> np.ndarray:
    nearest = np.abs(curve_q[:, None] - anchor_q[None, :]).argmin(axis=0)
    return curve_counts[nearest]


def _write_comparison(path: Path, rows: list[ComparisonRow], nearest_counts: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "q_nm_inv",
                "paper_eps_times_bn",
                "computed_eps_times_bn",
                "computed_minus_paper",
                "nearest_curve_n_points",
            ]
        )
        for row, count in zip(rows, nearest_counts, strict=True):
            writer.writerow(
                [
                    f"{row.q_nm_inv:.12g}",
                    f"{row.paper_eps_times_bn:.12g}",
                    f"{row.computed_eps_times_bn:.12g}",
                    f"{row.diff:.12g}",
                    int(count),
                ]
            )


def _write_summary(
    path: Path,
    *,
    artifact_label: str,
    curve_path: Path,
    anchors_path: Path,
    rows: list[ComparisonRow],
    rmse: float,
    max_abs: float,
    plot_png: Path,
    plot_pdf: Path,
    comparison_csv: Path,
) -> None:
    lines = [
        "# cRPA Epsilon Reference-Point Overlay",
        "",
        "This is a lightweight postprocess of an existing cRPA artifact. It does not rerun cRPA or HF.",
        "",
        "## Inputs",
        "",
        f"- artifact_label: {artifact_label}",
        f"- cRPA curve: `{curve_path}`",
        f"- paper anchors: `{anchors_path}`",
        "",
        "Only `paper_eps_times_bn` is used from the paper-anchor CSV. Any older computed column in that file is ignored.",
        "",
        "## Metrics",
        "",
        f"- anchor_count: {len(rows)}",
        f"- rmse_computed_minus_paper: {rmse:.6g}",
        f"- max_abs_computed_minus_paper: {max_abs:.6g}",
        "",
        "## Reference Points",
        "",
        "| q (nm^-1) | paper eps*eps_BN | current computed eps*eps_BN | computed - paper |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.q_nm_inv:.6g} | {row.paper_eps_times_bn:.6g} | "
            f"{row.computed_eps_times_bn:.6g} | {row.diff:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{comparison_csv}`",
            f"- `{plot_png}`",
            f"- `{plot_pdf}`",
            "",
            "Interpretation note: this curve is from the legal small no-alias artifact, not a production lk24/lg13/q11 result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(
    *,
    path_png: Path,
    path_pdf: Path,
    curve_q: np.ndarray,
    curve_eps: np.ndarray,
    anchor_q: np.ndarray,
    paper_eps: np.ndarray,
    computed_at_anchor: np.ndarray,
    artifact_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.plot(
        curve_q,
        curve_eps,
        color="#1f77b4",
        linewidth=1.8,
        marker="o",
        markersize=3.4,
        label="current radial median",
    )
    ax.scatter(
        anchor_q,
        computed_at_anchor,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=4,
        label="current at paper q",
    )
    ax.scatter(
        anchor_q,
        paper_eps,
        color="#d62728",
        marker="x",
        linewidth=1.8,
        s=62,
        zorder=5,
        label="paper anchor",
    )
    for q_value, paper_value, computed_value in zip(anchor_q, paper_eps, computed_at_anchor, strict=True):
        ax.plot(
            [q_value, q_value],
            [paper_value, computed_value],
            color="#666666",
            alpha=0.35,
            linewidth=0.8,
            zorder=2,
        )
    ax.set_xlabel("q (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon(q)\,\epsilon_{\mathrm{BN}}$")
    ax.set_title(artifact_label)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, loc="best")
    ax.set_xlim(left=0.0, right=max(1.2, float(anchor_q.max()) * 1.05))
    y_max = max(float(curve_eps.max()), float(paper_eps.max())) * 1.08
    ax.set_ylim(bottom=0.0, top=y_max)
    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-csv", required=True, type=Path)
    parser.add_argument("--paper-anchors", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact-label", default="cRPA epsilon reference points")
    args = parser.parse_args()

    curve_q, curve_eps, curve_counts = _read_curve(args.curve_csv)
    anchor_q, paper_eps = _read_paper_anchors(args.paper_anchors)
    if anchor_q.min() < curve_q.min() or anchor_q.max() > curve_q.max():
        raise ValueError(
            "paper anchor q range is outside the cRPA curve range: "
            f"anchors=[{anchor_q.min()}, {anchor_q.max()}], curve=[{curve_q.min()}, {curve_q.max()}]"
        )

    computed = np.interp(anchor_q, curve_q, curve_eps)
    rows = [
        ComparisonRow(q_nm_inv=float(q), paper_eps_times_bn=float(paper), computed_eps_times_bn=float(comp))
        for q, paper, comp in zip(anchor_q, paper_eps, computed, strict=True)
    ]
    diffs = np.asarray([row.diff for row in rows], dtype=float)
    rmse = float(np.sqrt(np.mean(diffs**2)))
    max_abs = float(np.max(np.abs(diffs)))
    nearest_counts = _nearest_counts(curve_q, curve_counts, anchor_q)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "epsilon_reference_point_comparison.csv"
    plot_png = args.output_dir / "epsilon_reference_points.png"
    plot_pdf = args.output_dir / "epsilon_reference_points.pdf"
    summary_md = args.output_dir / "summary.md"

    _write_comparison(comparison_csv, rows, nearest_counts)
    _plot(
        path_png=plot_png,
        path_pdf=plot_pdf,
        curve_q=curve_q,
        curve_eps=curve_eps,
        anchor_q=anchor_q,
        paper_eps=paper_eps,
        computed_at_anchor=computed,
        artifact_label=args.artifact_label,
    )
    _write_summary(
        summary_md,
        artifact_label=args.artifact_label,
        curve_path=args.curve_csv,
        anchors_path=args.paper_anchors,
        rows=rows,
        rmse=rmse,
        max_abs=max_abs,
        plot_png=plot_png,
        plot_pdf=plot_pdf,
        comparison_csv=comparison_csv,
    )

    print(f"wrote {comparison_csv}")
    print(f"wrote {plot_png}")
    print(f"wrote {plot_pdf}")
    print(f"wrote {summary_md}")
    print(f"rmse={rmse:.6g} max_abs={max_abs:.6g}")


if __name__ == "__main__":
    main()
