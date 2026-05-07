from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile

import numpy as np

from mean_field.systems.tbg import TBGParameters
from mean_field.systems.tbg.zero_field.model import build_b0_uniform_lattice
from mean_field.systems.tbg.zero_field.path_advisor import (
    KPathCompatibility,
    moire_bz_vertices,
    rank_kpath_candidates_for_lk,
    recommend_lk_values_for_path_family,
    sampled_cell_vertices,
)


def _parse_lk_values(text: str | None, fallback: int) -> tuple[int, ...]:
    if text is None or not text.strip():
        values = (fallback,)
    else:
        values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one lk value.")
    return values


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig_mean_field"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def _plot_candidates(
    output_dir: Path,
    compatibilities: tuple[KPathCompatibility, ...],
    *,
    stem: str,
    title: str,
    params: TBGParameters,
) -> dict[str, Path]:
    _configure_matplotlib()
    import matplotlib

    matplotlib.use(os.environ["MPLBACKEND"])
    import matplotlib.pyplot as plt

    if not compatibilities:
        raise ValueError("Expected at least one compatibility entry to plot.")

    ncols = len(compatibilities)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.4), squeeze=False)
    axes_flat = axes.ravel()

    for ax, compatibility in zip(axes_flat, compatibilities, strict=True):
        grid = build_b0_uniform_lattice(params, compatibility.lk)
        grid_kvec = np.asarray(grid.kvec, dtype=np.complex128)
        path_kvec = np.asarray(compatibility.candidate.path.kvec, dtype=np.complex128)
        node_kvec = np.asarray([complex(node.kvec) for node in compatibility.candidate.path.nodes], dtype=np.complex128)
        cell_vertices = np.asarray(sampled_cell_vertices(params), dtype=np.complex128)
        cell_loop = np.concatenate([cell_vertices, cell_vertices[:1]])
        bz_vertices = np.asarray(moire_bz_vertices(params), dtype=np.complex128)
        bz_loop = np.concatenate([bz_vertices, bz_vertices[:1]])

        ax.scatter(grid_kvec.real, grid_kvec.imag, s=11, color="#c7c7c7", alpha=0.85, linewidths=0.0)
        ax.plot(cell_loop.real, cell_loop.imag, color="#7f7f7f", lw=1.0, ls="--")
        ax.plot(bz_loop.real, bz_loop.imag, color="#9467bd", lw=1.2, alpha=0.9)
        ax.plot(path_kvec.real, path_kvec.imag, color="#1f77b4", lw=1.6)
        if compatibility.exact_grid_kvec.size > 0:
            ax.scatter(
                compatibility.exact_grid_kvec.real,
                compatibility.exact_grid_kvec.imag,
                s=22,
                color="#d62728",
                edgecolors="#ffffff",
                linewidths=0.35,
                zorder=3,
            )
        ax.scatter(node_kvec.real, node_kvec.imag, s=22, color="#111111", zorder=4)
        for node in compatibility.candidate.path.nodes:
            ax.text(node.kx, node.ky, f" {node.label}", fontsize=8, va="bottom")

        ax.set_aspect("equal")
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")
        ax.set_title(
            f"{compatibility.candidate.name}\n"
            f"exact={compatibility.exact_count}, "
            f"segments={compatibility.exact_segment_counts}\n"
            f"mean={compatibility.mean_nearest_distance:.4f}, "
            f"max={compatibility.max_nearest_distance:.4f}",
            fontsize=9,
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def _write_candidate_summary(path: Path, compatibilities: tuple[KPathCompatibility, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "lk",
                "candidate",
                "m_real",
                "m_imag",
                "exact_count",
                "exact_node_hit_count",
                "exact_segment_counts",
                "mean_nearest_distance",
                "max_nearest_distance",
                "node_min_distances",
            ]
        )
        for rank, compatibility in enumerate(compatibilities, start=1):
            writer.writerow(
                [
                    rank,
                    compatibility.lk,
                    compatibility.candidate.name,
                    f"{compatibility.candidate.m_point.real:.16f}",
                    f"{compatibility.candidate.m_point.imag:.16f}",
                    compatibility.exact_count,
                    compatibility.exact_node_hit_count,
                    ",".join(str(value) for value in compatibility.exact_segment_counts),
                    f"{compatibility.mean_nearest_distance:.16e}",
                    f"{compatibility.max_nearest_distance:.16e}",
                    ",".join(f"{value:.16e}" for value in compatibility.node_min_distances),
                ]
            )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect B0 kmesh and high-symmetry path compatibility.")
    parser.add_argument("--theta-deg", type=float, default=1.20)
    parser.add_argument("--lk", type=int, default=19)
    parser.add_argument("--lk-values", type=str, default="19,23,32")
    parser.add_argument("--points-per-segment", type=int, default=120)
    parser.add_argument("--candidate-set", choices=("adjacent", "all"), default="adjacent")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/home/ziyuzhu/Mean_Field/results/kmesh_path_advisor"),
    )
    args = parser.parse_args()

    params = TBGParameters.from_degrees(args.theta_deg)
    adjacent_only = args.candidate_set == "adjacent"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = rank_kpath_candidates_for_lk(
        params,
        lk=int(args.lk),
        points_per_segment=int(args.points_per_segment),
        adjacent_only=adjacent_only,
    )
    lk_values = _parse_lk_values(args.lk_values, int(args.lk))
    recommendations = recommend_lk_values_for_path_family(
        params,
        lk_values,
        points_per_segment=int(args.points_per_segment),
        adjacent_only=adjacent_only,
    )

    candidate_summary = _write_candidate_summary(
        output_dir / f"theta_{args.theta_deg:.2f}_lk_{args.lk}_candidate_summary.tsv",
        ranked,
    )
    recommendation_summary = _write_candidate_summary(
        output_dir / f"theta_{args.theta_deg:.2f}_lk_recommendations.tsv",
        recommendations,
    )
    best_plot = _plot_candidates(
        output_dir,
        ranked[:1],
        stem=f"theta_{args.theta_deg:.2f}_lk_{args.lk}_best_path_overlay",
        title=f"theta={args.theta_deg:.2f}°, lk={args.lk}: best path candidate",
        params=params,
    )
    candidate_plot = _plot_candidates(
        output_dir,
        ranked,
        stem=f"theta_{args.theta_deg:.2f}_lk_{args.lk}_candidate_overlays",
        title=f"theta={args.theta_deg:.2f}°, lk={args.lk}: candidate path overlays",
        params=params,
    )

    print(f"[advisor] candidate_summary={candidate_summary}")
    print(f"[advisor] recommendation_summary={recommendation_summary}")
    print(f"[advisor] best_plot_png={best_plot['png']}")
    print(f"[advisor] candidate_plot_png={candidate_plot['png']}")
    if ranked:
        best = ranked[0]
        print(
            "[advisor] best_candidate="
            f"{best.candidate.name} exact_count={best.exact_count} "
            f"exact_segment_counts={best.exact_segment_counts} "
            f"mean_nearest_distance={best.mean_nearest_distance:.6e} "
            f"max_nearest_distance={best.max_nearest_distance:.6e}"
        )


if __name__ == "__main__":
    main()
