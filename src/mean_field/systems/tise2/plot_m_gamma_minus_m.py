#!/usr/bin/env python3
"""Compare the reciprocal-equivalent M and -M endpoints in the TiSe2 model.

For the hexagonal basal plane used here,

    M_plus  = (b1 + b2)/2,
    M_minus = -(b1 + b2)/2 = M_plus - (b1 + b2).

Thus M_plus and M_minus are the same crystal momentum modulo a reciprocal
lattice vector.  The Monney et al. effective-mass model is local rather than
periodic, so each endpoint must be evaluated in its own reciprocal-lattice
chart.  Extending the M_plus-centered parabola all the way to M_minus would be
an invalid local-model extrapolation and would falsely make the endpoints look
inequivalent.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_k_gamma_m_k import A_LATTICE, Q_VALID_C, Q_VALID_V, reciprocal_geometry
from plot_noninteracting_bands import conduction_energy, valence_energy


# Extend 30% beyond both reciprocal-equivalent M representatives.
PATH_EXTENT = 1.30


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    b1, b2, _, gamma, m_plus = reciprocal_geometry(A_LATTICE)
    reciprocal_shift = b1 + b2
    m_minus = m_plus - reciprocal_shift

    # u=-1,0,+1 correspond to M_plus, Gamma, M_minus.  The larger range
    # deliberately continues a short distance beyond both M endpoints.
    u = np.linspace(-PATH_EXTENT, PATH_EXTENT, 2081)
    k_path = -u[:, None] * m_plus[None, :]
    valley_angle = float(np.arctan2(m_plus[1], m_plus[0]))

    q_v = k_path - gamma
    ev = valence_energy(q_v[:, 0], q_v[:, 1], qz_reduced=0.0)
    ev_local = np.where(np.linalg.norm(q_v, axis=1) <= Q_VALID_V, ev, np.nan)

    left = u <= 0.0
    right = u >= 0.0
    q_c_plus = k_path[left] - m_plus
    q_c_minus = k_path[right] - m_minus
    ec_plus = conduction_energy(
        q_c_plus[:, 0], q_c_plus[:, 1], valley_angle=valley_angle, qz_reduced=0.0
    )
    ec_minus = conduction_energy(
        q_c_minus[:, 0], q_c_minus[:, 1], valley_angle=valley_angle, qz_reduced=0.0
    )
    ec_plus_local = np.where(
        np.linalg.norm(q_c_plus, axis=1) <= Q_VALID_C, ec_plus, np.nan
    )
    ec_minus_local = np.where(
        np.linalg.norm(q_c_minus, axis=1) <= Q_VALID_C, ec_minus, np.nan
    )

    # Fold both halves onto signed distance from their own M representative.
    # Positive distance points toward Gamma; negative distance points outward.
    m_radius = float(np.linalg.norm(m_plus))
    outward_distance = (PATH_EXTENT - 1.0) * m_radius
    signed_distance = np.linspace(-outward_distance, m_radius, 1001)
    e_long = m_plus / m_radius
    q_from_plus = -signed_distance[:, None] * e_long[None, :]
    q_from_minus = signed_distance[:, None] * e_long[None, :]
    ec_fold_plus = conduction_energy(
        q_from_plus[:, 0],
        q_from_plus[:, 1],
        valley_angle=valley_angle,
        qz_reduced=0.0,
    )
    ec_fold_minus = conduction_energy(
        q_from_minus[:, 0],
        q_from_minus[:, 1],
        valley_angle=valley_angle,
        qz_reduced=0.0,
    )

    endpoint_plus = float(
        conduction_energy(np.array(0.0), np.array(0.0), valley_angle=valley_angle)
    )
    endpoint_minus = endpoint_plus
    max_fold_residual = float(np.max(np.abs(ec_fold_plus - ec_fold_minus)))

    # A deliberate diagnostic of the wrong operation: using only the M_plus
    # local chart at M_minus. This is not a physical endpoint prediction.
    q_wrong = m_minus - m_plus
    wrong_single_chart_endpoint = float(
        conduction_energy(
            np.array(q_wrong[0]),
            np.array(q_wrong[1]),
            valley_angle=valley_angle,
            qz_reduced=0.0,
        )
    )

    assert np.allclose(m_minus, -m_plus, rtol=0.0, atol=1e-14)
    assert np.allclose(m_plus - m_minus, reciprocal_shift, rtol=0.0, atol=1e-14)
    assert np.isclose(endpoint_plus, 0.020, atol=1e-12)
    assert np.isclose(endpoint_minus, 0.020, atol=1e-12)
    assert max_fold_residual <= 1e-13

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

    fig, (ax_path, ax_fold) = plt.subplots(
        1,
        2,
        figsize=(9.3, 4.0),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )

    v_color = "#9b2226"
    plus_color = "#005f73"
    minus_color = "#ca6702"

    # Faint dotted curves are formal local-parabola extensions toward Gamma.
    ax_path.plot(u, ev, color=v_color, lw=1.2, ls=":", alpha=0.42)
    ax_path.plot(u[left], ec_plus, color=plus_color, lw=1.3, ls=":", alpha=0.55)
    ax_path.plot(u[right], ec_minus, color=minus_color, lw=1.3, ls=":", alpha=0.55)

    ax_path.plot(u, ev_local, color=v_color, lw=2.4, label=r"$E_v$ near $\Gamma$")
    ax_path.plot(
        u[left],
        ec_plus_local,
        color=plus_color,
        lw=2.5,
        label=r"$E_c$ near $M_+$",
    )
    ax_path.plot(
        u[right],
        ec_minus_local,
        color=minus_color,
        lw=2.5,
        ls="--",
        label=r"$E_c$ near $M_-$",
    )

    for position in (-1.0, 0.0, 1.0):
        ax_path.axvline(position, color="0.80", lw=0.8, zorder=0)
    ax_path.axhline(0.0, color="0.84", lw=0.8, zorder=0)
    ax_path.scatter(
        [-1.0, 1.0],
        [endpoint_plus, endpoint_minus],
        color=[plus_color, minus_color],
        edgecolor="white",
        linewidth=0.8,
        s=38,
        zorder=5,
    )
    ax_path.set_xlim(-PATH_EXTENT, PATH_EXTENT)
    ax_path.set_ylim(-0.62, 0.90)
    ax_path.set_xticks([-1.0, 0.0, 1.0], [r"$M_+$", r"$\Gamma$", r"$M_-$"])
    ax_path.set_ylabel("Energy (eV; paper convention)")
    ax_path.set_title(r"$M_+\rightarrow\Gamma\rightarrow M_-$ ($\Delta=0$)")
    ax_path.grid(axis="y", color="0.91", lw=0.6)
    ax_path.legend(frameon=False, loc="lower left", fontsize=9)
    ax_path.text(
        0.03,
        0.96,
        r"$M_-=M_+-(\mathbf{b}_1+\mathbf{b}_2)$" + "\n"
        + rf"$E_c(M_+)=E_c(M_-)={endpoint_plus:.3f}$ eV",
        transform=ax_path.transAxes,
        va="top",
        fontsize=9.5,
    )

    ax_fold.plot(
        signed_distance,
        ec_fold_plus,
        color=plus_color,
        lw=2.4,
        label=r"from $M_+$ toward $\Gamma$",
    )
    ax_fold.plot(
        signed_distance,
        ec_fold_minus,
        color=minus_color,
        lw=2.0,
        ls="--",
        label=r"from $M_-$ toward $\Gamma$",
    )
    ax_fold.axvspan(-Q_VALID_C, Q_VALID_C, color="0.85", alpha=0.32, lw=0.0)
    ax_fold.axvline(0.0, color="0.55", lw=0.8)
    ax_fold.set_xlim(-outward_distance, m_radius)
    ax_fold.set_ylim(0.0, 0.80)
    ax_fold.set_xlabel(r"Signed distance from $M$; $+\!\to\Gamma$ ($\AA^{-1}$)")
    ax_fold.set_ylabel(r"$E_c$ (eV)")
    ax_fold.set_title("Folded endpoint comparison")
    ax_fold.grid(color="0.91", lw=0.6)
    ax_fold.legend(frameon=False, loc="upper left", fontsize=8.8)
    ax_fold.text(
        0.05,
        0.58,
        rf"max $|E_c^+-E_c^-|={max_fold_residual:.1e}$ eV" + "\n"
        + rf"one-chart value at $M_-$: {wrong_single_chart_endpoint:.2f} eV" + "\n"
        + "(invalid extrapolation)",
        transform=ax_fold.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.8"},
    )

    fig.suptitle(
        r"Reciprocal equivalence of $M_+$ and $M_-$ in the 1T-TiSe$_2$ local model",
        fontsize=12,
    )

    png_path = output_dir / "tise2_M_Gamma_minusM_equivalence.png"
    svg_path = output_dir / "tise2_M_Gamma_minusM_equivalence.svg"
    csv_path = output_dir / "tise2_M_Gamma_minusM_equivalence.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    # Store the path with the stitched periodic chart used for c_1.
    ec_stitched = np.empty_like(u)
    ec_stitched[left] = ec_plus
    ec_stitched[right] = ec_minus
    table = np.column_stack((u, k_path[:, 0], k_path[:, 1], ev, ec_stitched))
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="u,kx_A^-1,ky_A^-1,E_v_eV,E_c1_periodic_chart_eV",
        comments="",
    )

    print(f"M_plus  = ({m_plus[0]:.9f}, {m_plus[1]:.9f}) A^-1")
    print(f"M_minus = ({m_minus[0]:.9f}, {m_minus[1]:.9f}) A^-1")
    print(f"M_plus-M_minus = b1+b2: {np.allclose(m_plus-m_minus, reciprocal_shift)}")
    print(f"E_c(M_plus)  = {endpoint_plus:.12f} eV")
    print(f"E_c(M_minus) = {endpoint_minus:.12f} eV")
    print(f"max folded residual = {max_fold_residual:.3e} eV")
    print(f"invalid one-chart E_c(M_minus) = {wrong_single_chart_endpoint:.6f} eV")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
