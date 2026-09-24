#!/usr/bin/env python3
"""Plot the noninteracting low-energy bands of 1T-TiSe2.

Model and fit parameters are taken from C. Monney et al.,
"Spontaneous exciton condensation in 1T-TiSe2: a BCS-like approach",
arXiv:0809.1930, Sec. II A and Ref. 16.

The plot uses the folded low-energy representation of the paper: q is measured
from Gamma for the valence band and from each symmetry-related L valley for the
three conduction bands.  Setting Delta=0 leaves the four bare dispersions
uncoupled.  This local effective-mass model should not be extrapolated into a
full-Brillouin-zone band structure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# hbar^2/(2 m_e), in eV Angstrom^2 (CODATA value rounded for plotting).
HBAR2_OVER_2ME = 3.80998212

# Paper parameters (energies in eV; masses in units of the free-electron mass).
EPS_V0 = -0.030
M_V = -0.23
T_V = 0.060

EPS_C0 = -0.010
M_LONG = 5.5
M_SHORT = 2.2
T_C = 0.030

# In-plane long-axis directions of the three symmetry-related L pockets.
VALLEY_ANGLES = (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)


def valence_energy(qx: np.ndarray, qy: np.ndarray, qz_reduced: float = 0.0) -> np.ndarray:
    """Bare Se-4p valence dispersion epsilon_v in eV.

    qx and qy are in inverse Angstrom. qz_reduced is qz/k_{Gamma A}; thus the
    cosine in the paper is cos(pi*qz_reduced).
    """
    in_plane = HBAR2_OVER_2ME * (qx**2 + qy**2) / M_V
    return in_plane + T_V * np.cos(np.pi * qz_reduced) + EPS_V0


def conduction_energy(
    qx: np.ndarray,
    qy: np.ndarray,
    valley_angle: float,
    qz_reduced: float = 0.0,
) -> np.ndarray:
    """Bare Ti-3d conduction dispersion for one L valley in eV.

    q is measured relative to that valley center. valley_angle specifies the
    in-plane long axis of its elliptical pocket.
    """
    c, s = np.cos(valley_angle), np.sin(valley_angle)
    q_long = c * qx + s * qy
    q_short = -s * qx + c * qy
    in_plane = HBAR2_OVER_2ME * (q_long**2 / M_LONG + q_short**2 / M_SHORT)
    return in_plane + T_C * np.cos(np.pi * qz_reduced) + EPS_C0


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    q = np.linspace(-0.18, 0.18, 1201)
    zeros = np.zeros_like(q)

    ev = valence_energy(q, zeros)
    ec = [conduction_energy(q, zeros, angle) for angle in VALLEY_ANGLES]

    # Internal checks for the symmetry and paper-parameter transcription.
    ev_gamma = float(valence_energy(np.array(0.0), np.array(0.0)))
    ec_l = float(conduction_energy(np.array(0.0), np.array(0.0), 0.0))
    assert np.allclose(ec[1], ec[2], rtol=0.0, atol=1e-13)
    assert np.isclose(ev_gamma, EPS_V0 + T_V)
    assert np.isclose(ec_l, EPS_C0 + T_C)

    # Analytic crossing of v and c1 along this cut (positive-q branch).
    denom = HBAR2_OVER_2ME * (1.0 / M_LONG - 1.0 / M_V)
    q_cross = np.sqrt((ev_gamma - ec_l) / denom)
    e_cross = float(valence_energy(np.array(q_cross), np.array(0.0)))

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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.0, 3.8),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
        constrained_layout=True,
    )

    colors = {"v": "#9b2226", "c1": "#005f73", "c23": "#0a9396"}

    for ax in axes:
        ax.plot(q, ev, color=colors["v"], lw=2.2, label=r"$v$ at $\Gamma$")
        ax.plot(q, ec[0], color=colors["c1"], lw=2.0, label=r"$c_1$ at $L_1$")
        ax.plot(
            q,
            ec[1],
            color=colors["c23"],
            lw=1.8,
            ls="--",
            label=r"$c_2=c_3$ at $L_{2,3}$",
        )
        ax.axvline(0.0, color="0.78", lw=0.8, zorder=0)
        ax.set_xlabel(r"Reduced in-plane momentum $q_{\parallel\Gamma M}$ ($\AA^{-1}$)")
        ax.grid(which="major", color="0.90", lw=0.6)

    axes[0].set_ylabel("Energy (eV; paper convention)")
    axes[0].set_xlim(-0.18, 0.18)
    axes[0].set_ylim(-0.52, 0.11)
    axes[0].set_title(r"Bare bands ($\Delta=0$)")
    axes[0].legend(frameon=False, loc="lower center", fontsize=9)

    axes[1].set_xlim(-0.065, 0.065)
    axes[1].set_ylim(-0.045, 0.075)
    axes[1].set_title("Near the band extrema")
    axes[1].scatter(
        [-q_cross, q_cross],
        [e_cross, e_cross],
        s=24,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    axes[1].annotate(
        rf"$q_\times={q_cross:.3f}\,\AA^{{-1}}$",
        xy=(q_cross, e_cross),
        xytext=(0.004, -0.024),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=9,
    )
    axes[1].text(
        0.04,
        0.92,
        rf"$E_v(0)={ev_gamma:.3f}$ eV" + "\n" + rf"$E_c(0)={ec_l:.3f}$ eV",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        r"Noninteracting low-energy bands of 1T-TiSe$_2$ (Monney et al.)",
        fontsize=12,
    )

    png_path = output_dir / "tise2_noninteracting_bands.png"
    svg_path = output_dir / "tise2_noninteracting_bands.svg"
    csv_path = output_dir / "tise2_noninteracting_bands.csv"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    table = np.column_stack((q, ev, ec[0], ec[1], ec[2]))
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="q_A^-1,E_v_eV,E_c1_eV,E_c2_eV,E_c3_eV",
        comments="",
    )

    print(f"E_v(q=0) = {ev_gamma:.6f} eV")
    print(f"E_c_i(q=0) = {ec_l:.6f} eV")
    print(f"Bare indirect overlap E_c-E_v = {(ec_l - ev_gamma) * 1e3:.3f} meV")
    print(f"v-c1 crossing: |q| = {q_cross:.6f} A^-1, E = {e_cross:.6f} eV")
    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
