#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_node_payload(labels: np.ndarray, node_indices: np.ndarray, frac: np.ndarray) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for label, index in zip(labels, node_indices, strict=True):
        point = np.asarray(frac[int(index) - 1], dtype=float)
        out.append(
            {
                "label": str(label),
                "node_index_1based": int(index),
                "frac_g1": float(point[0]),
                "frac_g2": float(point[1]),
            }
        )
    return out


def _draw_cell(ax, *, color: str = "0.25", linestyle: str = "--", linewidth: float = 0.8) -> None:
    corners = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    ax.plot(corners[:, 0], corners[:, 1], color=color, linestyle=linestyle, linewidth=linewidth)


def _scatter_grid(ax, mesh_size: int) -> None:
    values = np.arange(int(mesh_size), dtype=float) / float(mesh_size)
    x, y = np.meshgrid(values, values, indexing="ij")
    ax.scatter(x.ravel(), y.ravel(), s=5, color="0.84", linewidths=0, zorder=0)


def _cell_from_g(g1: complex, g2: complex) -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [float(g1.real), float(g1.imag)],
            [float((g1 + g2).real), float((g1 + g2).imag)],
            [float(g2.real), float(g2.imag)],
            [0.0, 0.0],
        ],
        dtype=float,
    )


def _moire_bz_hexagon(g1: complex, g2: complex) -> np.ndarray:
    vertices = np.asarray(
        [
            (2.0 * g1 + g2) / 3.0,
            (g1 + 2.0 * g2) / 3.0,
            (-g1 + g2) / 3.0,
            (-2.0 * g1 - g2) / 3.0,
            (-g1 - 2.0 * g2) / 3.0,
            (g1 - g2) / 3.0,
        ],
        dtype=np.complex128,
    )
    order = np.argsort(np.angle(vertices))
    ordered = vertices[order]
    closed = np.concatenate([ordered, ordered[:1]])
    return np.stack([closed.real, closed.imag], axis=-1)


def _plot_connected_with_breaks(ax, frac: np.ndarray, *, color: str, linewidth: float, marker: str, label: str) -> None:
    frac = np.asarray(frac, dtype=float)
    start = 0
    for idx in range(1, frac.shape[0]):
        if np.linalg.norm(frac[idx] - frac[idx - 1]) > 0.26:
            ax.plot(
                frac[start:idx, 0],
                frac[start:idx, 1],
                color=color,
                linewidth=linewidth,
                marker=marker,
                markersize=3.2,
                label=label if start == 0 else None,
            )
            start = idx
    ax.plot(
        frac[start:, 0],
        frac[start:, 1],
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=3.2,
        label=label if start == 0 else None,
    )


def _angle_deg(v1: complex, v2: complex) -> float:
    denom = abs(v1) * abs(v2)
    if denom <= 0.0:
        return float("nan")
    cosine = (v1.real * v2.real + v1.imag * v2.imag) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _triangle_checks(node_indices: np.ndarray, kvec_extended: np.ndarray) -> dict[str, object]:
    if len(node_indices) < 7:
        return {}
    node_xy = np.asarray(kvec_extended[np.asarray(node_indices, dtype=int) - 1], dtype=float)
    node_z = np.asarray(node_xy[:, 0] + 1j * node_xy[:, 1], dtype=np.complex128)
    gamma0, k_point, kprime_point, gamma1, mprime_point, m_point, gamma2 = node_z[:7]
    return {
        "gamma_k_kprime_gamma": {
            "side_gamma_to_k_nm_inv": float(abs(k_point - gamma0)),
            "side_k_to_kprime_nm_inv": float(abs(kprime_point - k_point)),
            "side_kprime_to_gamma_nm_inv": float(abs(gamma1 - kprime_point)),
            "angle_at_gamma_deg": _angle_deg(k_point - gamma0, kprime_point - gamma0),
        },
        "gamma_mprime_m_gamma": {
            "side_gamma_to_mprime_nm_inv": float(abs(mprime_point - gamma1)),
            "side_mprime_to_m_nm_inv": float(abs(m_point - mprime_point)),
            "side_m_to_gamma_nm_inv": float(abs(gamma2 - m_point)),
            "angle_at_gamma_deg": _angle_deg(mprime_point - gamma1, m_point - gamma1),
        },
    }


def _draw_cartesian_reference(ax, *, g1: complex, g2: complex, scatter_grid: np.ndarray | None = None) -> None:
    if scatter_grid is not None:
        ax.scatter(scatter_grid[:, 0], scatter_grid[:, 1], s=5, color="0.84", linewidths=0, zorder=0, label="kmesh")
    cell = _cell_from_g(g1, g2)
    bz = _moire_bz_hexagon(g1, g2)
    ax.plot(cell[:, 0], cell[:, 1], color="0.45", linestyle="--", linewidth=1.0, label="sampled cell")
    ax.plot(bz[:, 0], bz[:, 1], color="#8e63c7", linewidth=1.5, label="moire BZ")


def _k_midpoint_payload(node_indices: np.ndarray, frac: np.ndarray, kvec: np.ndarray) -> dict[str, object]:
    indices = np.asarray(node_indices, dtype=int) - 1
    if indices.size < 3:
        return {}
    frac_mid = 0.5 * (np.asarray(frac[indices[1]], dtype=float) + np.asarray(frac[indices[2]], dtype=float))
    k_mid = 0.5 * (np.asarray(kvec[indices[1]], dtype=float) + np.asarray(kvec[indices[2]], dtype=float))
    return {
        "frac_g1": float(frac_mid[0]),
        "frac_g2": float(frac_mid[1]),
        "kx_nm_inv": float(k_mid[0]),
        "ky_nm_inv": float(k_mid[1]),
    }


def _axis_alignment_payload(node_indices: np.ndarray, kvec: np.ndarray) -> dict[str, object]:
    indices = np.asarray(node_indices, dtype=int) - 1
    if indices.size < 6:
        return {}
    values = np.asarray(kvec[indices], dtype=float)
    gamma = values[0, 0] + 1j * values[0, 1]
    k_mid_xy = 0.5 * (values[1] + values[2])
    m_mid_xy = 0.5 * (values[4] + values[5])
    k_axis = (k_mid_xy[0] + 1j * k_mid_xy[1]) - gamma
    m_axis = (m_mid_xy[0] + 1j * m_mid_xy[1]) - gamma
    angle = _angle_deg(k_axis, m_axis)
    return {
        "k_base_midpoint_kx_nm_inv": float(k_mid_xy[0]),
        "k_base_midpoint_ky_nm_inv": float(k_mid_xy[1]),
        "m_base_midpoint_kx_nm_inv": float(m_mid_xy[0]),
        "m_base_midpoint_ky_nm_inv": float(m_mid_xy[1]),
        "axis_angle_difference_deg": float(min(angle, abs(180.0 - angle))),
    }


def _draw_shared_axis(ax, node_xy: np.ndarray) -> None:
    if node_xy.shape[0] < 6:
        return
    gamma = np.asarray(node_xy[0], dtype=float)
    k_mid = 0.5 * (np.asarray(node_xy[1], dtype=float) + np.asarray(node_xy[2], dtype=float))
    m_mid = 0.5 * (np.asarray(node_xy[4], dtype=float) + np.asarray(node_xy[5], dtype=float))
    end = m_mid
    ax.plot([float(gamma[0]), float(end[0])], [float(gamma[1]), float(end[1])], color="#ff8c00", linestyle="--", linewidth=1.0, label="shared bisector")
    ax.scatter([float(k_mid[0]), float(m_mid[0])], [float(k_mid[1]), float(m_mid[1])], s=30, marker="D", color="#ff8c00", zorder=6)


def _plot_reciprocal_basis_overlay(
    *,
    output_png: Path,
    output_pdf: Path,
    g1: complex,
    g2: complex,
    grid_xy: np.ndarray,
    kvec_extended: np.ndarray,
    kvec_lookup: np.ndarray,
    labels: np.ndarray,
    node_indices: np.ndarray,
    frac_extended: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    _draw_cartesian_reference(ax, g1=g1, g2=g2, scatter_grid=grid_xy)

    ax.plot(
        kvec_extended[:, 0],
        kvec_extended[:, 1],
        color="#1f5aa6",
        linewidth=1.3,
        marker=".",
        markersize=3.0,
        label="high-symmetry path",
        zorder=3,
    )
    ax.scatter(
        kvec_lookup[:, 0],
        kvec_lookup[:, 1],
        s=12,
        color="#d62728",
        linewidths=0,
        zorder=4,
        label="exact SCF hits",
    )

    node_xy = kvec_extended[node_indices - 1]
    node_frac = frac_extended[node_indices - 1]
    ax.scatter(node_xy[:, 0], node_xy[:, 1], s=40, color="black", zorder=6, label="path nodes")
    _draw_shared_axis(ax, node_xy)
    for label, xy, frac_xy in zip(labels, node_xy, node_frac, strict=True):
        _annotate_node(ax, label, xy, frac_xy)

    arrow_props = {"arrowstyle": "-|>", "linewidth": 1.8, "color": "#006d77", "mutation_scale": 14}
    ax.annotate("", xy=(float(g1.real), float(g1.imag)), xytext=(0.0, 0.0), arrowprops=arrow_props, zorder=5)
    ax.annotate("", xy=(float(g2.real), float(g2.imag)), xytext=(0.0, 0.0), arrowprops=arrow_props, zorder=5)
    ax.annotate(
        "$\\mathbf{g}_1$",
        (float(g1.real), float(g1.imag)),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color="#005f68",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    ax.annotate(
        "$\\mathbf{g}_2$",
        (float(g2.real), float(g2.imag)),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        color="#005f68",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    corners = _cell_from_g(g1, g2)
    basis_points = np.asarray(
        [
            [0.0, 0.0],
            [float(g1.real), float(g1.imag)],
            [float(g2.real), float(g2.imag)],
            [float((g1 + g2).real), float((g1 + g2).imag)],
        ],
        dtype=float,
    )
    all_points = np.vstack([grid_xy, kvec_extended, kvec_lookup, node_xy, corners, basis_points])
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0e-9)
    pad = 0.08 * span
    ax.set_xlim(float(x_min - pad), float(x_max + pad))
    ax.set_ylim(float(y_min - pad), float(y_max + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$k_x$ (nm$^{-1}$)")
    ax.set_ylabel("$k_y$ (nm$^{-1}$)")
    ax.set_title("SCF k mesh, high-symmetry path, and reciprocal basis", fontsize=11)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)


def _node_annotation(label: object, frac_xy: np.ndarray) -> str:
    text = str(label)
    frac = np.asarray(frac_xy, dtype=float)
    if text == "$M_M$" and np.all(frac >= -1.0e-12) and np.all(frac < 1.0 - 1.0e-12):
        return text + "\n(in cell)"
    return text


def _label_offset(label: object) -> tuple[int, int]:
    text = str(label)
    if "K'_M" in text:
        return (8, -12)
    if "K_M" in text:
        return (8, 8)
    if "M'_M" in text:
        return (8, 8)
    if "M_M" in text:
        return (8, -14)
    return (8, 4)


def _annotate_node(ax, label: object, xy: np.ndarray, frac_xy: np.ndarray) -> None:
    ax.annotate(
        _node_annotation(label, frac_xy),
        (float(xy[0]), float(xy[1])),
        xytext=_label_offset(label),
        textcoords="offset points",
        fontsize=8,
        color="black",
        zorder=7,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the k-path geometry used by an RLG/hBN checkpoint SCF-path plot.")
    parser.add_argument("--bands-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", type=str, default="current_checkpoint_paper_fig6_kpath_geometry")
    parser.add_argument("--mesh-size", type=int, default=18)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    data = np.load(args.bands_npz, allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=object)
    node_indices = np.asarray(data["node_indices"], dtype=int)
    frac_extended = np.asarray(data["frac_extended"], dtype=float)
    frac_lookup = np.asarray(data["frac_lookup"], dtype=float)
    mesh_indices = np.asarray(data["mesh_indices"], dtype=int)
    kvec_extended = np.asarray(data["kvec_extended_nm_inv"], dtype=float)
    kvec_lookup = np.asarray(data["kvec_lookup_nm_inv"], dtype=float)

    checkpoint_kvec = None
    if args.checkpoint is not None:
        checkpoint = np.load(args.checkpoint)
        pairs = np.asarray(checkpoint["kvec_nm_inv"], dtype=float)
        checkpoint_kvec = np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128).reshape(-1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.stem}.png"
    output_pdf = args.output_dir / f"{args.stem}.pdf"
    output_cartesian_png = args.output_dir / f"{args.stem}_cartesian.png"
    output_cartesian_pdf = args.output_dir / f"{args.stem}_cartesian.pdf"
    output_summary = args.output_dir / f"{args.stem}_summary.json"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)

    ax = axes[0]
    _draw_cell(ax)
    # Show the nearest translated sampled cell that contains M' for visual context.
    shifted = np.asarray([[-1, -1], [0, -1], [0, 0], [-1, 0], [-1, -1]], dtype=float)
    ax.plot(shifted[:, 0], shifted[:, 1], color="0.65", linestyle=":", linewidth=0.75)
    ax.plot(frac_extended[:, 0], frac_extended[:, 1], color="#1f5aa6", linewidth=1.2, marker=".", markersize=3.2)
    node_frac_ext = frac_extended[node_indices - 1]
    ax.scatter(node_frac_ext[:, 0], node_frac_ext[:, 1], s=34, color="#d62728", zorder=4)
    for label, xy in zip(labels, node_frac_ext, strict=True):
        _annotate_node(ax, label, xy, xy)
    ax.set_title("Paper Fig. 6 extended path", fontsize=10)
    ax.set_xlabel("$g_1$ fractional coordinate")
    ax.set_ylabel("$g_2$ fractional coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(np.min(frac_extended[:, 0])) - 0.12, float(np.max(frac_extended[:, 0])) + 0.12)
    ax.set_ylim(float(np.min(frac_extended[:, 1])) - 0.12, float(np.max(frac_extended[:, 1])) + 0.12)
    ax.grid(True, color="0.9", linewidth=0.6)

    ax = axes[1]
    _scatter_grid(ax, int(args.mesh_size))
    _draw_cell(ax, color="0.2", linestyle="-", linewidth=0.8)
    _plot_connected_with_breaks(ax, frac_lookup, color="#2ca02c", linewidth=1.1, marker=".", label="exact SCF hits")
    node_frac_lookup = frac_lookup[node_indices - 1]
    ax.scatter(node_frac_lookup[:, 0], node_frac_lookup[:, 1], s=34, color="#d62728", zorder=4)
    for label, xy in zip(labels, node_frac_lookup, strict=True):
        _annotate_node(ax, label, xy, xy)
    ax.set_title("Folded lookup in saved SCF mesh", fontsize=10)
    ax.set_xlabel("$g_1$ fractional coordinate")
    ax.set_ylabel("$g_2$ fractional coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, color="0.9", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.savefig(output_png, dpi=int(args.dpi))
    fig.savefig(output_pdf)
    plt.close(fig)

    if checkpoint_kvec is not None:
        # Reconstruct the physical sampled reciprocal cell from the saved mesh
        # points and the path archive.  The path may use extended-zone paper
        # nodes, so do not infer g1/g2 from K/K' high-symmetry identities.
        frac_ext = frac_extended
        z_ext = np.asarray(kvec_extended[:, 0] + 1j * kvec_extended[:, 1], dtype=np.complex128)
        mask = np.linalg.norm(frac_ext, axis=1) > 1.0e-14
        if np.count_nonzero(mask) < 2:
            raise ValueError("Cannot infer reciprocal cell from path archive")
        g_coeffs, *_ = np.linalg.lstsq(frac_ext[mask], z_ext[mask], rcond=None)
        g1 = complex(g_coeffs[0])
        g2 = complex(g_coeffs[1])
        grid_xy = np.stack([checkpoint_kvec.real, checkpoint_kvec.imag], axis=-1)
        cell = _cell_from_g(g1, g2)

        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
        ax = axes[0]
        _draw_cartesian_reference(ax, g1=g1, g2=g2)
        shifted = _cell_from_g(g1, g2) - np.asarray([float((g1 + g2).real), float((g1 + g2).imag)], dtype=float)
        ax.plot(shifted[:, 0], shifted[:, 1], color="0.65", linestyle=":", linewidth=0.75, label="translated cell")
        ax.plot(
            kvec_extended[:, 0],
            kvec_extended[:, 1],
            color="#1f5aa6",
            linewidth=1.2,
            marker=".",
            markersize=3.2,
            label="paper path",
        )
        node_xy = kvec_extended[node_indices - 1]
        node_frac_ext = frac_extended[node_indices - 1]
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=34, color="#d62728", zorder=4)
        _draw_shared_axis(ax, node_xy)
        for label, xy, frac_xy in zip(labels, node_xy, node_frac_ext, strict=True):
            _annotate_node(ax, label, xy, frac_xy)
        ax.set_title("Paper Fig. 6 path in $k_x,k_y$", fontsize=10)
        ax.set_xlabel("$k_x$ (nm$^{-1}$)")
        ax.set_ylabel("$k_y$ (nm$^{-1}$)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=7, frameon=False)

        ax = axes[1]
        _draw_cartesian_reference(ax, g1=g1, g2=g2, scatter_grid=grid_xy)
        ax.scatter(kvec_lookup[:, 0], kvec_lookup[:, 1], s=12, color="#d62728", linewidths=0, zorder=3, label="exact SCF hits")
        node_lookup_xy = kvec_lookup[node_indices - 1]
        node_frac_lookup = frac_lookup[node_indices - 1]
        ax.scatter(node_lookup_xy[:, 0], node_lookup_xy[:, 1], s=36, color="black", zorder=4, label="path nodes")
        for label, xy, frac_xy in zip(labels, node_lookup_xy, node_frac_lookup, strict=True):
            _annotate_node(ax, label, xy, frac_xy)
        ax.set_title("Folded lookup in physical SCF cell", fontsize=10)
        ax.set_xlabel("$k_x$ (nm$^{-1}$)")
        ax.set_ylabel("$k_y$ (nm$^{-1}$)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="0.9", linewidth=0.6)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
        fig.savefig(output_cartesian_png, dpi=int(args.dpi))
        fig.savefig(output_cartesian_pdf)
        plt.close(fig)
        _plot_reciprocal_basis_overlay(
            output_png=output_png,
            output_pdf=output_pdf,
            g1=g1,
            g2=g2,
            grid_xy=grid_xy,
            kvec_extended=kvec_extended,
            kvec_lookup=kvec_lookup,
            labels=labels,
            node_indices=node_indices,
            frac_extended=frac_extended,
        )
        _plot_reciprocal_basis_overlay(
            output_png=output_cartesian_png,
            output_pdf=output_cartesian_pdf,
            g1=g1,
            g2=g2,
            grid_xy=grid_xy,
            kvec_extended=kvec_extended,
            kvec_lookup=kvec_lookup,
            labels=labels,
            node_indices=node_indices,
            frac_extended=frac_extended,
        )

    _write_json(
        output_summary,
        {
            "bands_npz": str(args.bands_npz),
            "mesh_size": int(args.mesh_size),
            "coordinate_note": "figures use physical kx/ky coordinates, show the SCF k mesh, high-symmetry path, sampled cell, moire BZ, and reciprocal basis arrows g1/g2",
            "output_cartesian_pdf": str(output_cartesian_pdf) if checkpoint_kvec is not None else "",
            "output_cartesian_png": str(output_cartesian_png) if checkpoint_kvec is not None else "",
            "output_pdf": str(output_pdf),
            "output_png": str(output_png),
            "path_labels": [str(value) for value in labels],
            "points": int(frac_extended.shape[0]),
            "unique_mesh_points": int(np.unique(mesh_indices).size),
            "triangle_checks": _triangle_checks(node_indices, kvec_extended),
            "k_kprime_midpoint_extended": _k_midpoint_payload(node_indices, frac_extended, kvec_extended),
            "k_kprime_midpoint_lookup": _k_midpoint_payload(node_indices, frac_lookup, kvec_lookup),
            "axis_alignment_extended": _axis_alignment_payload(node_indices, kvec_extended),
            "axis_alignment_lookup_note": "not evaluated after modulo folding; use axis_alignment_extended for geometry",
            "extended_nodes": _path_node_payload(labels, node_indices, frac_extended),
            "lookup_nodes": _path_node_payload(labels, node_indices, frac_lookup),
        },
    )
    print(f"[done] output_png={output_png}", flush=True)


if __name__ == "__main__":
    main()
