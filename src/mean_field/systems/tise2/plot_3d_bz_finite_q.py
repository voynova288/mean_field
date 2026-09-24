#!/usr/bin/env python3
"""Draw the 3D 1T-TiSe2 Brillouin zone and read off finite-Q vectors.

The construction follows the normal-state geometry used by Monney et al.
(arXiv:0809.1930): the relevant valence pocket is centered at Gamma and the
three symmetry-related conduction pockets are centered at L_i.  In top view,
L_i projects onto M_i, but the physical bulk ordering vectors are Gamma->L_i,
not Gamma->M_i.

Coordinates are normalized: |Gamma M|=1 in-plane and k_GammaA=1 vertically.
The reciprocal basis therefore has b3=(0,0,2), so the top BZ plane is l=1/2.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from monney import MonneyTiSe2Parameters, monney_bare_energies


SQRT3 = np.sqrt(3.0)

# Normalized reciprocal basis. b1 and b2 have length 2, hence |Gamma M|=1.
B1 = np.array([SQRT3, -1.0, 0.0])
B2 = np.array([0.0, 2.0, 0.0])
B3 = np.array([0.0, 0.0, 2.0])
GAMMA = np.zeros(3)
A_POINT = 0.5 * B3

# A C3-symmetric choice of the three L representatives. Other choices differ
# by reciprocal lattice vectors and describe the same torus momenta.
Q_FRACTIONAL = np.array(
    [
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
    ]
)


def cartesian_from_fractional(frac: np.ndarray) -> np.ndarray:
    frac = np.asarray(frac, dtype=float)
    basis = np.stack((B1, B2, B3), axis=0)
    return frac @ basis


def canonical_mod_one(frac: np.ndarray) -> np.ndarray:
    """Canonical reciprocal-torus label in [0,1), stable at half coordinates."""
    return np.mod(np.round(np.asarray(frac, dtype=float), decimals=12), 1.0)


def hexagonal_bz_geometry():
    radius_k = 2.0 / SQRT3
    angles = np.arange(6, dtype=float) * np.pi / 3.0
    k_bottom = np.column_stack(
        (radius_k * np.cos(angles), radius_k * np.sin(angles), np.zeros(6))
    )
    h_top = k_bottom + A_POINT
    m_bottom = 0.5 * (k_bottom + np.roll(k_bottom, -1, axis=0))
    l_top = m_bottom + A_POINT
    return k_bottom, h_top, m_bottom, l_top


def draw_closed_polygon_3d(ax, points: np.ndarray, **kwargs) -> None:
    closed = np.vstack((points, points[0]))
    ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], **kwargs)


def draw_side_view(ax, *, q_color: str, valence_color: str) -> None:
    """Draw the Gamma--M/A--L projection used by the composite figure."""
    ax.plot(
        [-0.06, 1.08, 1.08, -0.06, -0.06],
        [0, 0, 1, 1, 0],
        color="0.25",
        lw=1.3,
    )
    ax.scatter(
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        color=[valence_color, "0.2", "0.55", q_color],
        s=34,
        zorder=4,
    )
    ax.text(-0.03, -0.12, r"$\Gamma$", ha="center")
    ax.text(-0.03, 1.08, r"$A$", ha="center")
    ax.text(1.0, -0.12, r"$M_i$", ha="center")
    ax.text(1.0, 1.08, r"$L_i$", ha="center", color=q_color)
    ax.annotate(
        "",
        xy=(1.0, 1.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": q_color, "lw": 2.2},
    )
    ax.text(
        0.48,
        0.57,
        r"$\mathbf{Q}_i=\overrightarrow{\Gamma L_i}$",
        color=q_color,
        rotation=34,
        ha="center",
    )
    ax.plot([0.0, 1.0], [0.0, 0.0], color="0.65", lw=1.0, ls="--")
    ax.plot([1.0, 1.0], [0.0, 1.0], color="0.65", lw=1.0, ls="--")
    ax.text(0.52, -0.12, r"$Q_{i,\parallel}=\Gamma M_i$", ha="center", fontsize=8.8)
    ax.text(
        1.11,
        0.52,
        r"$Q_{i,z}=k_{\Gamma A}$",
        rotation=90,
        va="center",
        fontsize=8.8,
    )
    ax.set_xlim(-0.16, 1.30)
    ax.set_ylim(-0.18, 1.20)
    ax.axis("off")
    ax.set_title("Side view")


def draw_band_edge_schematic(ax, *, conduction_color: str, valence_color: str) -> None:
    """Draw actual Monney k.p energies in their declared local windows."""
    params = MonneyTiSe2Parameters(conduction_z_convention="paper_literal")
    p_valence = np.linspace(-0.18, 0.18, 361)
    p_conduction = np.linspace(-0.25, 0.25, 501)
    valence_points = np.column_stack(
        (p_valence, np.zeros_like(p_valence), np.zeros_like(p_valence))
    )
    conduction_points = np.column_stack(
        (p_conduction, np.zeros_like(p_conduction), np.zeros_like(p_conduction))
    )
    e_v = 1.0e3 * monney_bare_energies(valence_points, params=params)[0]
    e_c = 1.0e3 * monney_bare_energies(conduction_points, params=params)[1]

    # External lattice constant a=3.54 Angstrom fixes the physical in-plane
    # Gamma--M_i projection used by the finite-Q geometry.
    q_parallel_Ainv = 2.0 * np.pi / (SQRT3 * 3.54)
    k_valence = p_valence
    k_conduction = q_parallel_Ainv + p_conduction

    ax.plot(k_valence, e_v, color=valence_color, lw=2.6)
    ax.plot(k_conduction, e_c, color=conduction_color, lw=2.6)
    e_v_gamma = float(
        1.0e3 * monney_bare_energies(np.zeros((1, 3)), params=params)[0, 0]
    )
    e_c_l = float(
        1.0e3 * monney_bare_energies(np.zeros((1, 3)), params=params)[1, 0]
    )
    ax.scatter(
        [0.0, q_parallel_Ainv],
        [e_v_gamma, e_c_l],
        color=[valence_color, conduction_color],
        s=42,
        zorder=4,
    )
    arrow_y = -33.0
    ax.annotate(
        "",
        xy=(q_parallel_Ainv, arrow_y),
        xytext=(0.0, arrow_y),
        arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 1.4},
    )
    ax.text(
        0.5 * q_parallel_Ainv,
        arrow_y - 12.0,
        r"$Q_{i,\parallel}=\Gamma M_i$",
        ha="center",
    )
    ax.text(
        0.0,
        e_v_gamma + 14.0,
        rf"$E_v(\Gamma)={e_v_gamma:.0f}\,\mathrm{{meV}}$",
        color=valence_color,
        ha="center",
    )
    ax.text(
        q_parallel_Ainv,
        e_c_l + 22.0,
        rf"$E_c(L_i)={e_c_l:.0f}\,\mathrm{{meV}}$",
        color=conduction_color,
        ha="center",
    )
    ax.text(
        0.5 * q_parallel_Ainv,
        44.0,
        rf"$E_c(L_i)-E_v(\Gamma)={e_c_l - e_v_gamma:.0f}\,\mathrm{{meV}}$",
        ha="center",
        fontsize=9.0,
    )
    ax.set_xlim(-0.21, q_parallel_Ainv + 0.28)
    ax.set_ylim(-50.0, 50.0)
    ax.set_yticks([-50.0, -25.0, 0.0, 25.0, 50.0])
    ax.set_xticks([0.0, q_parallel_Ainv], [r"$\Gamma$", r"$L_i$"])
    ax.set_ylabel("Energy (meV)")
    ax.set_title(r"Noninteracting $k\!\cdot\!p$ band edges")
    ax.grid(axis="y", color="0.90", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    k_bottom, h_top, m_bottom, l_top = hexagonal_bz_geometry()
    q_cart = cartesian_from_fractional(Q_FRACTIONAL)

    # The chosen representatives land on alternating L points of the top hexagon.
    selected_l_indices = np.array([5, 1, 3])
    selected_l = l_top[selected_l_indices]
    selected_m = m_bottom[selected_l_indices]

    assert np.allclose(q_cart, selected_l, rtol=0.0, atol=1e-12)
    assert np.allclose(q_cart[:, :2], selected_m[:, :2], rtol=0.0, atol=1e-12)
    assert np.allclose(q_cart[:, 2], 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(q_cart[:, :2], axis=1), 1.0, atol=1e-12)

    # Every L_i is self-conjugate modulo a reciprocal lattice vector.
    self_conjugate = [
        np.allclose(
            canonical_mod_one(-q_frac),
            canonical_mod_one(q_frac),
            rtol=0.0,
            atol=1e-12,
        )
        for q_frac in Q_FRACTIONAL
    ]
    assert all(self_conjugate)

    # The three Q_i generate the complete 2x2x2 reciprocal quotient (8 sectors).
    closure = {
        tuple(canonical_mod_one(sum(bits[i] * Q_FRACTIONAL[i] for i in range(3))))
        for bits in product((0, 1), repeat=3)
    }
    assert len(closure) == 8

    colors = ("#005f73", "#0a9396", "#ca6702")
    valence_color = "#9b2226"

    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 300,
        }
    )

    fig = plt.figure(figsize=(13.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.13, top=0.86, wspace=0.25)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.0, 1.12))
    ax_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_top = fig.add_subplot(grid[0, 1])
    right = grid[0, 2].subgridspec(2, 1, height_ratios=(1.0, 1.05), hspace=0.34)
    ax_side = fig.add_subplot(right[0, 0])
    ax_band = fig.add_subplot(right[1, 0])

    # --- 3D hexagonal BZ -------------------------------------------------
    draw_closed_polygon_3d(ax_3d, k_bottom, color="0.25", lw=1.3)
    draw_closed_polygon_3d(ax_3d, h_top, color="0.25", lw=1.3)
    for lower, upper in zip(k_bottom, h_top):
        ax_3d.plot(
            [lower[0], upper[0]],
            [lower[1], upper[1]],
            [lower[2], upper[2]],
            color="0.55",
            lw=0.85,
        )

    ax_3d.scatter(*GAMMA, color=valence_color, s=45, depthshade=False, zorder=5)
    ax_3d.scatter(*A_POINT, color="0.18", s=28, depthshade=False, zorder=5)
    ax_3d.scatter(
        selected_l[:, 0],
        selected_l[:, 1],
        selected_l[:, 2],
        color=colors,
        s=45,
        depthshade=False,
        zorder=6,
    )
    ax_3d.scatter(
        selected_m[:, 0],
        selected_m[:, 1],
        selected_m[:, 2],
        facecolor="white",
        edgecolor=colors,
        s=32,
        depthshade=False,
        zorder=5,
    )

    for index, (q_vec, color) in enumerate(zip(q_cart, colors), start=1):
        ax_3d.quiver(
            0.0,
            0.0,
            0.0,
            q_vec[0],
            q_vec[1],
            q_vec[2],
            color=color,
            lw=2.0,
            arrow_length_ratio=0.08,
        )
        midpoint = 0.55 * q_vec
        ax_3d.text(
            midpoint[0],
            midpoint[1],
            midpoint[2] + 0.04,
            rf"$Q_{index}$",
            color=color,
            fontsize=10,
        )

    ax_3d.text(0.02, 0.02, -0.08, r"$\Gamma$ ($E_v$)", color=valence_color)
    ax_3d.text(0.03, 0.03, 1.06, r"$A$")
    for index, point in enumerate(selected_l, start=1):
        ax_3d.text(
            point[0] * 1.08,
            point[1] * 1.08,
            point[2] + 0.04,
            rf"$L_{index}$ ($E_c^{{({index})}}$)",
            color=colors[index - 1],
            fontsize=9,
        )

    # Label one representative member of each standard high-symmetry family.
    ax_3d.text(k_bottom[0, 0] * 1.08, k_bottom[0, 1], 0.0, r"$K$")
    ax_3d.text(h_top[0, 0] * 1.08, h_top[0, 1], 1.02, r"$H$")
    ax_3d.text(m_bottom[0, 0] * 1.08, m_bottom[0, 1] * 1.08, -0.03, r"$M$")
    ax_3d.text(l_top[0, 0] * 1.08, l_top[0, 1] * 1.08, 1.02, r"$L$")

    ax_3d.set_title("3D hexagonal Brillouin zone")
    ax_3d.set_xlabel(r"$k_x/|\Gamma M|$", labelpad=5)
    ax_3d.set_ylabel(r"$k_y/|\Gamma M|$", labelpad=5)
    ax_3d.set_zlabel(r"$k_z/k_{\Gamma A}$", labelpad=5)
    ax_3d.set_box_aspect((2.1, 2.1, 1.35))
    ax_3d.set_xlim(-1.35, 1.35)
    ax_3d.set_ylim(-1.35, 1.35)
    ax_3d.set_zlim(-0.08, 1.12)
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([0.0, 1.0], ["0", "1"])
    ax_3d.view_init(elev=23, azim=-54)
    for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        axis.pane.set_alpha(0.0)
    ax_3d.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=valence_color, label=r"valence pocket at $\Gamma$"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0], label=r"three conduction pockets at $L_i$"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.97),
        fontsize=8.5,
    )

    # --- Top view: why L looks like M in a 2D projection -----------------
    polygon = np.vstack((k_bottom[:, :2], k_bottom[0, :2]))
    ax_top.plot(polygon[:, 0], polygon[:, 1], color="0.25", lw=1.4)
    ax_top.scatter(0.0, 0.0, color=valence_color, s=45, zorder=5)
    ax_top.text(0.04, 0.04, r"$\Gamma$", color=valence_color, fontsize=11)
    ax_top.scatter(
        m_bottom[:, 0],
        m_bottom[:, 1],
        facecolor="white",
        edgecolor="0.65",
        s=24,
        zorder=3,
    )
    for index, (point, color) in enumerate(zip(selected_m, colors), start=1):
        ax_top.annotate(
            "",
            xy=point[:2],
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
        )
        ax_top.scatter(
            point[0], point[1], facecolor=color, edgecolor="white", s=42, zorder=5
        )
        ax_top.text(
            point[0] * 0.76,
            point[1] * 0.76,
            rf"$M_{index}\equiv \mathrm{{proj}}(L_{index})$",
            color=color,
            ha="center",
            va="center",
            fontsize=8.3,
        )
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-1.35, 1.35)
    ax_top.set_ylim(-1.35, 1.35)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_title(r"Top view: $L_i\mapsto M_i$")
    ax_top.text(
        0.5,
        -0.06,
        r"Only the in-plane projection is $\Gamma M_i$",
        transform=ax_top.transAxes,
        ha="center",
        fontsize=9,
    )

    # --- Side view: the missing out-of-plane component -------------------
    ax_side.plot([-0.06, 1.08, 1.08, -0.06, -0.06], [0, 0, 1, 1, 0], color="0.25", lw=1.3)
    ax_side.scatter([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0], color=[valence_color, "0.2", "0.55", colors[0]], s=34, zorder=4)
    ax_side.text(-0.03, -0.12, r"$\Gamma$", ha="center")
    ax_side.text(-0.03, 1.08, r"$A$", ha="center")
    ax_side.text(1.0, -0.12, r"$M_i$", ha="center")
    ax_side.text(1.0, 1.08, r"$L_i$", ha="center", color=colors[0])
    ax_side.annotate(
        "",
        xy=(1.0, 1.0),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "color": colors[0], "lw": 2.2},
    )
    ax_side.text(0.48, 0.57, r"$\mathbf{Q}_i=\overrightarrow{\Gamma L_i}$", color=colors[0], rotation=34, ha="center")
    ax_side.plot([0.0, 1.0], [0.0, 0.0], color="0.65", lw=1.0, ls="--")
    ax_side.plot([1.0, 1.0], [0.0, 1.0], color="0.65", lw=1.0, ls="--")
    ax_side.text(0.52, -0.12, r"$Q_{i,\parallel}=\Gamma M_i$", ha="center", fontsize=8.8)
    ax_side.text(1.11, 0.52, r"$Q_{i,z}=k_{\Gamma A}$", rotation=90, va="center", fontsize=8.8)
    ax_side.set_xlim(-0.16, 1.30)
    ax_side.set_ylim(-0.18, 1.20)
    ax_side.axis("off")
    ax_side.set_title("Side view")

    # --- Local band-edge schematic: no unsupported full-zone interpolation
    x_v = np.linspace(-0.28, 0.28, 200)
    x_c = np.linspace(0.72, 1.28, 200)
    e_v = 0.055 - 1.1 * x_v**2
    e_c = 0.025 + 0.95 * (x_c - 1.0) ** 2
    ax_band.plot(x_v, e_v, color=valence_color, lw=2.2)
    ax_band.plot(x_c, e_c, color=colors[0], lw=2.2)
    ax_band.scatter([0.0, 1.0], [e_v.max(), e_c.min()], color=[valence_color, colors[0]], s=30, zorder=4)
    ax_band.annotate(
        "",
        xy=(1.0, -0.012),
        xytext=(0.0, -0.012),
        arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 1.4},
    )
    ax_band.text(0.5, -0.026, r"$\mathbf{Q}_i=\Gamma L_i$", ha="center")
    ax_band.text(0.0, 0.065, r"$E_v$ at $\Gamma$", color=valence_color, ha="center")
    ax_band.text(1.0, 0.065, r"$E_c^{(i)}$ at $L_i$", color=colors[0], ha="center")
    ax_band.set_xlim(-0.34, 1.34)
    ax_band.set_ylim(-0.04, 0.09)
    ax_band.set_xticks([0.0, 1.0], [r"$\Gamma$", r"$L_i$"])
    ax_band.set_yticks([])
    ax_band.set_ylabel("Energy (schematic)")
    ax_band.set_title("Relevant normal-state band edges")
    ax_band.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        r"1T-TiSe$_2$: 3D high-symmetry points and the finite-$Q$ choice",
        fontsize=13,
    )
    fig.text(
        0.5,
        -0.015,
        r"Read-off: $\mathbf{Q}_i=\overrightarrow{\Gamma L_i}$ ($i=1,2,3$); "
        r"$M_i$ are only their basal-plane projections.  Minimal paper basis: "
        r"$\{0,Q_1,Q_2,Q_3\}$.",
        ha="center",
        fontsize=10.3,
    )

    png_path = output_dir / "tise2_3d_high_symmetry_finite_Q.png"
    svg_path = output_dir / "tise2_3d_high_symmetry_finite_Q.svg"
    csv_path = output_dir / "tise2_3d_finite_Q_vectors.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    # Publish the two right-column panels as standalone, editable figures.
    side_fig, side_only_ax = plt.subplots(figsize=(6.4, 5.2))
    draw_side_view(side_only_ax, q_color=colors[0], valence_color=valence_color)
    side_fig.tight_layout()
    side_stem = output_dir / "tise2_finite_Q_side_view"
    side_fig.savefig(side_stem.with_suffix(".png"), bbox_inches="tight")
    side_fig.savefig(side_stem.with_suffix(".svg"), bbox_inches="tight")
    side_fig.savefig(side_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(side_fig)

    band_fig, band_only_ax = plt.subplots(figsize=(6.8, 4.8))
    draw_band_edge_schematic(
        band_only_ax,
        conduction_color=colors[0],
        valence_color=valence_color,
    )
    band_fig.tight_layout()
    band_stem = output_dir / "tise2_normal_state_band_edges"
    band_fig.savefig(band_stem.with_suffix(".png"), bbox_inches="tight")
    band_fig.savefig(band_stem.with_suffix(".svg"), bbox_inches="tight")
    band_fig.savefig(band_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(band_fig)

    rows = []
    for index, (frac, cart) in enumerate(zip(Q_FRACTIONAL, q_cart), start=1):
        rows.append(
            [
                index,
                frac[0],
                frac[1],
                frac[2],
                cart[0],
                cart[1],
                cart[2],
                int(self_conjugate[index - 1]),
            ]
        )
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=float),
        delimiter=",",
        header="i,h_b1,k_b2,l_b3,Qx_over_GammaM,Qy_over_GammaM,Qz_over_GammaA,self_conjugate",
        comments="",
    )

    print("Finite-Q vectors in reciprocal fractional coordinates:")
    for index, frac in enumerate(Q_FRACTIONAL, start=1):
        print(f"Q{index} = ({frac[0]:+.1f}, {frac[1]:+.1f}, {frac[2]:+.1f})")
    print(f"All Q_i self-conjugate modulo G: {all(self_conjugate)}")
    print(f"Generated reciprocal-quotient sectors: {len(closure)}")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
