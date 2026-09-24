#!/usr/bin/env python3
"""Plot the projected K-Gamma-M-K path for the 1T-TiSe2 bare bands.

The energies use the local effective-mass model of Monney et al.
(arXiv:0809.1930), with the excitonic order parameter Delta set to zero.
The valence energy E_v is expanded around Gamma; the plotted conduction
energy E_c is c_1 expanded around L_1, whose in-plane projection is M.

Because the paper does not provide a periodic full-zone Hamiltonian, dotted
parts are only the formal parabolic extrapolation. Solid parts mark the local
momentum windows around Gamma and M where the low-energy model is intended to
be interpreted.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_noninteracting_bands import conduction_energy, valence_energy

# Basal-plane lattice constant used only to convert standard hexagonal
# high-symmetry coordinates to inverse Angstrom. It is configurable because it
# is not one of the effective-mass fit parameters listed in the paper.
A_LATTICE = 3.54  # Angstrom

# Conservative visualization windows for the local effective-mass expansions.
Q_VALID_V = 0.18  # inverse Angstrom around Gamma
Q_VALID_C = 0.25  # inverse Angstrom around the L_1 projection M


def reciprocal_geometry(a_lattice: float):
    """Return reciprocal vectors and selected 2D hexagonal high-symmetry points."""
    factor = 2.0 * np.pi / a_lattice
    b1 = factor * np.array([1.0, -1.0 / np.sqrt(3.0)])
    b2 = factor * np.array([0.0, 2.0 / np.sqrt(3.0)])
    gamma = np.zeros(2)
    k_point = (2.0 * b1 + b2) / 3.0
    m_point = (b1 + b2) / 2.0
    return b1, b2, k_point, gamma, m_point


def interpolate_path(points, points_per_segment: int = 350):
    """Interpolate a piecewise-linear path without duplicate segment endpoints."""
    k_chunks = []
    s_chunks = []
    ticks = [0.0]
    cumulative = 0.0

    for index, (start, stop) in enumerate(zip(points[:-1], points[1:])):
        endpoint = index == len(points) - 2
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=endpoint)
        segment = start[None, :] + t[:, None] * (stop - start)[None, :]
        segment_length = np.linalg.norm(stop - start)
        s_segment = cumulative + t * segment_length
        k_chunks.append(segment)
        s_chunks.append(s_segment)
        cumulative += segment_length
        ticks.append(cumulative)

    return np.vstack(k_chunks), np.concatenate(s_chunks), np.asarray(ticks)


def first_bz_vertices(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Vertices of the first 2D hexagonal Brillouin zone in cyclic order."""
    return np.asarray(
        [
            (2.0 * b1 + b2) / 3.0,
            (b1 + 2.0 * b2) / 3.0,
            (-b1 + b2) / 3.0,
            -(2.0 * b1 + b2) / 3.0,
            -(b1 + 2.0 * b2) / 3.0,
            (b1 - b2) / 3.0,
        ]
    )


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    b1, b2, k_point, gamma, m_point = reciprocal_geometry(A_LATTICE)
    path_points = (k_point, gamma, m_point, k_point)
    k_path, path_coordinate, ticks = interpolate_path(path_points)

    # E_v is local to Gamma. E_c(c_1) is local to L_1; in the projected 2D
    # path, its in-plane center is M and its long axis follows Gamma->M.
    q_v = k_path - gamma
    q_c = k_path - m_point
    valley_angle = float(np.arctan2(m_point[1], m_point[0]))

    ev = valence_energy(q_v[:, 0], q_v[:, 1], qz_reduced=0.0)
    ec1 = conduction_energy(
        q_c[:, 0], q_c[:, 1], valley_angle=valley_angle, qz_reduced=0.0
    )

    radius_v = np.linalg.norm(q_v, axis=1)
    radius_c = np.linalg.norm(q_c, axis=1)
    ev_local = np.where(radius_v <= Q_VALID_V, ev, np.nan)
    ec1_local = np.where(radius_c <= Q_VALID_C, ec1, np.nan)

    # Transcription/geometry checks.
    gamma_index = int(np.argmin(np.abs(path_coordinate - ticks[1])))
    m_index = int(np.argmin(np.abs(path_coordinate - ticks[2])))
    assert np.allclose(path_points[0], path_points[-1])
    assert np.linalg.norm(k_path[gamma_index] - gamma) < 1e-12
    assert np.linalg.norm(k_path[m_index] - m_point) < 1e-12
    assert np.isclose(ev[gamma_index], 0.030, atol=1e-12)
    assert np.isclose(ec1[m_index], 0.020, atol=1e-12)

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
        }
    )

    fig, (ax_band, ax_bz) = plt.subplots(
        1,
        2,
        figsize=(9.3, 4.1),
        gridspec_kw={"width_ratios": (2.15, 1.0)},
        constrained_layout=True,
    )

    v_color = "#9b2226"
    c_color = "#005f73"

    # Formal full-path extrapolations are deliberately faint and dotted.
    ax_band.plot(
        path_coordinate,
        ev,
        color=v_color,
        lw=1.25,
        ls=":",
        alpha=0.42,
        label=r"$E_v$ formal extrapolation",
    )
    ax_band.plot(
        path_coordinate,
        ec1,
        color=c_color,
        lw=1.25,
        ls=":",
        alpha=0.42,
        label=r"$E_c$ ($c_1$) formal extrapolation",
    )
    ax_band.plot(
        path_coordinate,
        ev_local,
        color=v_color,
        lw=2.4,
        label=rf"$E_v$ local ($|q_\Gamma|\leq {Q_VALID_V:.2f}\,\AA^{{-1}}$)",
    )
    ax_band.plot(
        path_coordinate,
        ec1_local,
        color=c_color,
        lw=2.4,
        label=rf"$E_c$: $c_1$ local ($|q_M|\leq {Q_VALID_C:.2f}\,\AA^{{-1}}$)",
    )

    for tick in ticks:
        ax_band.axvline(tick, color="0.78", lw=0.8, zorder=0)
    ax_band.axhline(0.0, color="0.82", lw=0.8, zorder=0)
    ax_band.set_xlim(ticks[0], ticks[-1])
    ax_band.set_ylim(-0.62, 0.90)
    ax_band.set_xticks(ticks, [r"$K$", r"$\Gamma$", r"$M$", r"$K$"])
    ax_band.set_ylabel("Energy (eV; paper convention)")
    ax_band.set_title(r"Projected $K$–$\Gamma$–$M$–$K$ bare bands ($\Delta=0$)")
    ax_band.grid(axis="y", color="0.91", lw=0.6)
    ax_band.legend(frameon=False, loc="upper right", fontsize=8.4)

    ax_band.scatter(
        [ticks[1], ticks[2]],
        [ev[gamma_index], ec1[m_index]],
        color=[v_color, c_color],
        edgecolor="white",
        linewidth=0.8,
        s=34,
        zorder=5,
    )
    ax_band.annotate(
        r"$E_v(\Gamma)=0.030$ eV",
        xy=(ticks[1], ev[gamma_index]),
        xytext=(ticks[1] - 0.34, 0.16),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )
    ax_band.annotate(
        r"$E_c(M)=0.020$ eV",
        xy=(ticks[2], ec1[m_index]),
        xytext=(ticks[2] - 0.10, -0.17),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )

    # Brillouin-zone path schematic.
    bz = first_bz_vertices(b1, b2)
    bz_closed = np.vstack((bz, bz[0]))
    ax_bz.plot(bz_closed[:, 0], bz_closed[:, 1], color="0.25", lw=1.4)
    projected_path = np.vstack(path_points)
    ax_bz.plot(
        projected_path[:, 0],
        projected_path[:, 1],
        color="#ca6702",
        lw=2.2,
        marker="o",
        ms=4.2,
    )
    ax_bz.annotate(
        "",
        xy=gamma,
        xytext=k_point,
        arrowprops={"arrowstyle": "->", "color": "#ca6702", "lw": 1.2},
    )
    ax_bz.annotate(
        "",
        xy=m_point,
        xytext=gamma,
        arrowprops={"arrowstyle": "->", "color": "#ca6702", "lw": 1.2},
    )
    ax_bz.annotate(
        "",
        xy=k_point,
        xytext=m_point,
        arrowprops={"arrowstyle": "->", "color": "#ca6702", "lw": 1.2},
    )
    ax_bz.text(k_point[0] + 0.06, k_point[1] - 0.06, r"$K$", fontsize=11)
    ax_bz.text(gamma[0] - 0.12, gamma[1] - 0.12, r"$\Gamma$", fontsize=11)
    ax_bz.text(m_point[0] - 0.02, m_point[1] + 0.09, r"$M$ ($L_1$ projection)", ha="center", fontsize=9)
    ax_bz.set_aspect("equal")
    ax_bz.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
    ax_bz.set_ylabel(r"$k_y$ ($\AA^{-1}$)")
    ax_bz.set_title("Basal-plane BZ path")
    ax_bz.grid(color="0.92", lw=0.6)

    fig.suptitle(
        r"Noninteracting low-energy model of 1T-TiSe$_2$ (Monney et al.)",
        fontsize=12,
    )

    png_path = output_dir / "tise2_K_Gamma_M_K_noninteracting.png"
    svg_path = output_dir / "tise2_K_Gamma_M_K_noninteracting.svg"
    csv_path = output_dir / "tise2_K_Gamma_M_K_noninteracting.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    table = np.column_stack(
        (
            path_coordinate,
            k_path[:, 0],
            k_path[:, 1],
            ev,
            ec1,
            radius_v <= Q_VALID_V,
            radius_c <= Q_VALID_C,
        )
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="path_A^-1,kx_A^-1,ky_A^-1,E_v_eV,E_c1_eV,E_v_local,E_c1_local",
        comments="",
    )

    print(f"a = {A_LATTICE:.4f} A")
    print(f"K = ({k_point[0]:.6f}, {k_point[1]:.6f}) A^-1")
    print(f"M = ({m_point[0]:.6f}, {m_point[1]:.6f}) A^-1")
    print(f"E_v(Gamma) = {ev[gamma_index]:.6f} eV")
    print(f"E_c1(M) = {ec1[m_index]:.6f} eV")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
