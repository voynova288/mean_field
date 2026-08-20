from __future__ import annotations

"""Regenerate the Du2017 report SVGs from internal, hash-pinned NPZ artifacts.

The calculation artifacts are intentionally not distributed in this public branch.
Set DU2017_SOURCE_ROOT to the internal calculation archive root when regenerating.
"""

import json
from pathlib import Path
import hashlib
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(os.environ.get("DU2017_SOURCE_ROOT", "/data/home/ziyuzhu/Mean_Field"))
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

KP = ROOT / "results/inas_gasb_matrix_ei_experiment/runs/phase0_du2017_N95_plane_wave_canonical_scf_k024_20260730/canonical_N95_kane_poisson.npz"
NOTE2 = ROOT / "results/du2017_inas_gasb_excitonic_insulator/runs/du2017_literal_note2_fixed_density_bcs_20260816/nr320_k024.npz"
MATRIX = ROOT / "results/inas_gasb_matrix_ei_experiment/runs/phase2_du2017_N95_refined_raw_radial_nr640_forecast_20260818/fixed_mu_matrix_bcs_normal_tr/result.npz"
BUNDLE = ROOT / "results/inas_gasb_matrix_ei_experiment/runs/phase2_du2017_N95_refined_raw_radial_nr640_forecast_20260818/refined_raw_radial_kane4_nr640_forecast.npz"
OPTICAL = ROOT / "results/inas_gasb_matrix_ei_experiment/runs/phase2_du2017_N95_refined_raw_radial_nr640_forecast_20260818/bare_current_response_nr640/bare_current_response_nr640.npz"
CONTRACT_CORRECTION = KP.parent / "CONTRACT_CORRECTION.json"
NOTE2_REPORT = NOTE2.parent / "report.json"
SCALAR_CONTROL_REPORT = ROOT / "results/du2017_inas_gasb_excitonic_insulator/runs/du2017_kane8_band_views_20260811/kane_poisson_fixed_canonical_mu_literal_note2_bcs/report.json"
MATRIX_META = MATRIX.parent / "metadata.json"
BARE_REPORT = OPTICAL.parent / "report.json"
POSTFLIGHT = OPTICAL.parent.parent / "bare_current_response_nr320_nr640_postflight/report.json"
POSTFLIGHT_REJECTED = POSTFLIGHT.parent / "POSTFLIGHT_REJECTED.json"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "svg.hashsalt": "mean-field-du2017-report-v1",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
BLUE = "#2457a6"
RED = "#c53f3f"
GOLD = "#d49a1f"
TEAL = "#19857b"
DARK = "#24292f"
GRAY = "#6b7280"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_label(ax, text):
    ax.text(-0.13, 1.03, text, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")

# 1. Canonical N95 Kane-Poisson checkpoint and exact saved-grid quartet.
with np.load(KP, allow_pickle=False) as d:
    z = d["z_nm"]
    wz = d["z_weights_nm"]
    potential = d["potential_mev"]
    rhoe = d["electron_density_nm3"]
    rhoh = d["hole_density_nm3"]
    k = d["k_cart_nm_inv"][:, 0]
    energies = d["energies_mev"]
    micro = d["micro_wavefunctions"]
    mu = float(d["history__mu_mev"][-1])
    residual = float(d["history__fixed_point_residual_mev"][-1])
    ne = float(np.dot(wz, rhoe))
    nh = float(np.dot(wz, rhoh))
    gamma6 = np.einsum("kzmb,z->kb", np.abs(micro[:, :, :2, :])**2, wz, optimize=True)

fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.02, 1.18]})
ax = axes[0]
ax.plot(z, potential - np.average(potential, weights=wz), color=DARK, lw=1.8, label=r"$U(z)-\langle U\rangle$")
ax.set_xlabel("z (nm)")
ax.set_ylabel("electrostatic potential (meV)")
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(z, rhoe, color=BLUE, lw=1.3, label=r"$n_e(z)$")
ax2.plot(z, rhoh, color=RED, lw=1.3, label=r"$n_h(z)$")
ax2.set_ylabel(r"carrier density (nm$^{-3}$)")
lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right", fontsize=8)
ax.text(0.02, 0.075, rf"$n_e={ne:.7f}$ nm$^{{-2}}$", transform=ax.transAxes, fontsize=8, va="bottom")
ax.text(0.02, 0.025, rf"$n_h={nh:.7f}$ nm$^{{-2}}$", transform=ax.transAxes, fontsize=8, va="bottom")
panel_label(ax, "a")

ax = axes[1]
for band in range(4):
    y = energies[band] - mu
    ax.plot(k, y, color="#b9bec5", lw=0.8, zorder=1)
    points = np.column_stack([k, y])
    sc = ax.scatter(points[:, 0], points[:, 1], c=gamma6[:, band], cmap="coolwarm", vmin=0, vmax=1, s=12, edgecolors="none", zorder=2)
ax.axhline(0, color=DARK, lw=0.9, ls="--")
ax.set_xlabel(r"radial $k$ (nm$^{-1}$)")
ax.set_ylabel(r"$E-\mu_{\rm KP}$ (meV)")
ax.set_xlim(k.min(), k.max())
ax.text(0.02, 0.075, rf"$\mu_{{\rm KP}}={mu:.9f}$ meV", transform=ax.transAxes, fontsize=8, va="bottom")
ax.text(0.02, 0.025, rf"final $\|U_{{out}}-U_{{in}}\|_\infty={residual:.2e}$ meV", transform=ax.transAxes, fontsize=8, va="bottom")
cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
cbar.set_label(r"projected $\Gamma_6$ weight")
panel_label(ax, "b")
fig.suptitle("Canonical split=0 N=95 Kane–Poisson fixed point — fixed regulator, radial ray", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "canonical_n95_kane_poisson.svg", bbox_inches="tight", metadata={"Title":"Canonical N95 Kane-Poisson fixed-regulator checkpoint","Creator":"Mean_Field Du2017 report generator","Date":"2026-08-20","Description":"Independent saved-grid replot; radial ray only; no off-grid reconstruction."})
plt.close(fig)

# 2. Original schematic of paper Fig. 2 semantics (not digitization).
k_s = np.linspace(0, 0.035, 500)
# Smooth illustrative curves matching only the topology and quoted scales.
E_s = 1.5 + 5.5 * (1 - np.exp(-((k_s - 0.024)/0.010)**2)) + 4.0 * np.clip((k_s - 0.029)/0.006, 0, None)**2
Delta_s = 0.45 + 1.05 * np.exp(-((k_s - 0.024)/0.012)**2)
omega_s = np.linspace(0, 9.5, 1000)
jdos_s = 0.18*np.exp(-0.5*((omega_s-1.55)/0.18)**2) + 0.72*np.exp(-0.5*((omega_s-7.0)/0.28)**2)
jdos_s /= jdos_s.max()
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.25), sharey=True, gridspec_kw={"width_ratios": [1.15, 0.85]})
ax = axes[0]
ax.plot(k_s, E_s, color=BLUE, lw=2, ls="--", label=r"$E(k)$")
ax.plot(k_s, Delta_s, color=RED, lw=2, ls="--", label=r"$\Delta(k)$")
ax.scatter([0.024, 0], [1.5, 7.0], color=[RED, BLUE], s=25, zorder=4)
ax.annotate("ring minimum\n" + r"$E_{min}\approx1.5$ meV", (0.024, 1.5), (0.012, 2.8), arrowprops={"arrowstyle":"->","lw":0.8}, fontsize=8)
ax.annotate(r"$k\approx0$ feature" + "\n" + r"$E\approx7$ meV", (0.0005, 7.0), (0.009, 7.8), arrowprops={"arrowstyle":"->","lw":0.8}, fontsize=8)
ax.set(xlabel=r"$k$ (nm$^{-1}$)", ylabel="energy (meV)", xlim=(0,0.035), ylim=(0,9.5))
ax.legend(frameon=False, loc="upper center")
panel_label(ax, "a")
ax = axes[1]
ax.plot(jdos_s, omega_s, color=BLUE, lw=2)
ax.set(xlabel="JDOS (schematic, arb. units)", xlim=(0,1.05))
ax.set_xticks([])
ax.text(0.03, 0.075, "shared vertical energy axis", transform=ax.transAxes, fontsize=8, color=GRAY)
ax.text(0.03, 0.025, "illustrative widths/amplitudes; high doublet not resolved", transform=ax.transAxes, fontsize=7, color=GRAY)
panel_label(ax, "b")
fig.suptitle("What Du et al. Fig. 2(a,b) asserts — conceptual redraw, not digitized data", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "paper_fig2_semantics_schematic.svg", bbox_inches="tight", metadata={"Title":"Du2017 Fig. 2 semantics schematic","Creator":"Mean_Field Du2017 report generator","Date":"2026-08-20","Description":"Original conceptual redraw; not digitized, not to scale, and not numerical evidence."})
plt.close(fig)

# 3. Independently calculated literal Supplementary-Note-2 model.
with np.load(NOTE2, allow_pickle=False) as d:
    k2 = d["k_nm_inv"]
    pair = d["pair_energy_mev"]
    delta = d["delta_mev"]
    omega = d["omega_mev"]
    jdos = d["normalized_physical_jdos"]
    lambda_max = float(d["linearized_lambda_max"])
    e0 = float(d["exact_k0_pair_energy_mev"])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.25), sharey=True, gridspec_kw={"width_ratios": [1.15, 0.85]})
ax = axes[0]
ax.plot(k2, pair, color=BLUE, lw=2, label=r"$E_{pair}=\sqrt{\xi^2+\Delta^2}$")
ax.plot(k2, delta, color=RED, lw=2, label=r"$\Delta(k)$")
imin = int(np.argmin(pair)); imax = int(np.argmax(delta))
ax.scatter([k2[imin], k2[imax]], [pair[imin], delta[imax]], color=[BLUE, RED], s=24, zorder=3)
ax.set(xlabel=r"$k$ (nm$^{-1}$)", ylabel="energy (meV)", xlim=(0,0.12), ylim=(0,10.2))
ax.legend(frameon=False, fontsize=8)
ax.text(0.98, 0.12, rf"$\lambda_{{max}}={lambda_max:.4f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
ax.text(0.98, 0.07, rf"$E_{{min}}={pair[imin]:.4f}$ meV", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
ax.text(0.98, 0.02, rf"$E(0)={e0:.4f}$ meV", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
ax.text(0.02, 0.96, r"display: $k\leq0.12$; solver: $k_{max}=0.24$ nm$^{-1}$", transform=ax.transAxes, va="top", fontsize=7, color=GRAY)
panel_label(ax, "a")
ax = axes[1]
ax.plot(jdos, omega, color=BLUE, lw=2)
ax.set(xlabel=r"physical $k\,dk/(2\pi)$ JDOS", xlim=(0,1.05))
ax.set_xticks([])
ax.axhline(omega[np.argmax(jdos)], color=GRAY, lw=0.8, ls=":")
panel_label(ax, "b")
fig.suptitle("Literal printed Note-2 equations; physical radial JDOS with gamma=0.1 meV", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "literal_note2_bcs_result.svg", bbox_inches="tight", metadata={"Title":"Literal Du2017 Supplementary Note 2 BCS result","Creator":"Mean_Field Du2017 report generator","Date":"2026-08-20","Description":"Independent scalar-model calculation; physical radial JDOS with gamma=0.1 meV."})
plt.close(fig)

# 4. Same-parent N95 matrix HF/BCS state and bare-current diagnostic.
with np.load(MATRIX, allow_pickle=False) as d:
    sigma = d["sigma_fock_mev"]
    emat = d["energies_mev"]
with np.load(BUNDLE, allow_pickle=False) as d:
    km = d["k_cart_nm_inv"][:,0]
with np.load(OPTICAL, allow_pickle=False) as d:
    om = d["omega_mev"]
    optical_shape = d["bare_current_shape_saved"]
sv = np.empty((2, km.size))
for i in range(km.size):
    sv[:,i] = np.linalg.svd(sigma[:2,2:,i], compute_uv=False)
middle = emat[2]-emat[1]

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.15), gridspec_kw={"width_ratios":[1,1,0.8]})
ax=axes[0]
ax.plot(km, sv[0], color=RED, lw=1.8, label=r"$s_1[\Delta_{EH}]$")
ax.plot(km, sv[1], color=GOLD, lw=1.8, label=r"$s_2[\Delta_{EH}]$")
ax.set(xlabel=r"$k$ (nm$^{-1}$)", ylabel="matrix order singular value (meV)", xlim=(0,km.max()))
ax.legend(frameon=False, fontsize=8)
ax.text(0.03,0.96,rf"max $s_1={sv[0].max():.4f}$ meV",transform=ax.transAxes,va="top",fontsize=8)
ax.text(0.03,0.90,rf"max $s_2={sv[1].max():.4f}$ meV",transform=ax.transAxes,va="top",fontsize=8)
panel_label(ax,"a")
ax=axes[1]
for b,color in zip(range(4),("#85a7db",BLUE,RED,"#df9393")):
    ax.plot(km, emat[b]-113.8688022886634, color=color, lw=1.3)
ax.axhline(0,color=DARK,lw=0.8,ls="--")
ax.set(xlabel=r"$k$ (nm$^{-1}$)", ylabel=r"$E-\mu_{KP}$ (meV)", xlim=(0,km.max()))
ax.text(0.03,0.09,"min direct middle gap",transform=ax.transAxes,fontsize=8)
ax.text(0.03,0.04,rf"$={middle.min():.4f}$ meV",transform=ax.transAxes,fontsize=8)
panel_label(ax,"b")
ax=axes[2]
ax.plot(optical_shape, om, color=TEAL, lw=1.8)
ax.set(xlabel=r"peak-normalized bare current ($\gamma=0.2$ meV)", ylabel="transition energy (meV)", ylim=(0,25), xlim=(0,1.04))
ax.set_xticks([])
for val,label in ((8.2465,"low group"),(18.1315,"high group")):
    ax.axhline(val,color=GRAY,lw=0.8,ls=":")
    ax.text(0.05,val+0.35,f"{label}: {val:.4f} meV",fontsize=8)
ax.text(0.04,0.97,"postflight rejected: shape L1=0.0312 > 0.02",transform=ax.transAxes,va="top",fontsize=7,color=RED)
panel_label(ax,"c")
fig.suptitle(r"Same-parent fixed-$\mu_{KP}$ N=95 matrix state; bare-current peak gates pass, lineshape gate fails",fontsize=11.5,y=1.02)
fig.tight_layout()
fig.savefig(OUT / "same_parent_nr640_matrix_bcs_and_bare_current.svg", bbox_inches="tight", metadata={"Title":"Same-parent N95 matrix state and bare-current diagnostic","Creator":"Mean_Field Du2017 report generator","Date":"2026-08-20","Description":"Independent radial exchange-only matrix state; peak gates pass but overall bare-current lineshape postflight rejected."})
plt.close(fig)

# Matplotlib emits path-data lines with trailing spaces. Normalize generated SVG
# text so repository whitespace checks are deterministic without changing geometry.
figure_names = (
    "canonical_n95_kane_poisson.svg",
    "paper_fig2_semantics_schematic.svg",
    "literal_note2_bcs_result.svg",
    "same_parent_nr640_matrix_bcs_and_bare_current.svg",
)
for figure_name in figure_names:
    figure_path = OUT / figure_name
    normalized = "\n".join(line.rstrip() for line in figure_path.read_text().splitlines()) + "\n"
    figure_path.write_text(normalized)

# Provenance record. Internal source paths identify a hash-pinned archive but the
# NPZ/report bytes themselves are intentionally not distributed in this branch.
generator_path = Path(__file__).resolve()
provenance = {
    "schema": "du2017-report-figure-provenance-v2",
    "source_root": str(ROOT),
    "source_distribution": "internal calculation archive; hashes and transformations public, source bytes not distributed",
    "reuse_rights": "no separate license is granted for report figures; repository policy applies",
    "regeneration_command": "DU2017_SOURCE_ROOT=/path/to/Mean_Field python docs/figures/du2017/generate_figures.py",
    "figures": {
        "canonical_n95_kane_poisson.svg": {
            "classification": "independently calculated fixed-regulator saved-grid checkpoint replot; radial ray only",
            "sources": {
                str(KP.relative_to(ROOT)): sha256(KP),
                str(CONTRACT_CORRECTION.relative_to(ROOT)): sha256(CONTRACT_CORRECTION),
            },
            "transformations": [
                "potential shifted by its z-weighted mean for display",
                "active quartet energies shifted by saved mu_KP",
                "marker color is z-integrated Gamma6 weight",
                "no interpolation or off-grid reconstruction",
            ],
            "metrics": {
                "mu_mev": mu,
                "electron_density_nm2": ne,
                "hole_density_nm2": nh,
                "potential_peak_to_peak_mev": float(np.ptp(potential)),
                "final_fixed_point_residual_mev": residual,
                "minimum_middle_rank_gap_mev": float(np.min(energies[2]-energies[1])),
                "minimum_middle_rank_gap_k_nm_inv": float(k[np.argmin(energies[2]-energies[1])]),
                "gamma_active_quartet_separation_mev": float(energies[2,0]-energies[1,0]),
            },
        },
        "paper_fig2_semantics_schematic.svg": {
            "classification": "original conceptual redraw; not digitization, not to scale, and not numerical evidence",
            "source": "Du et al., Nature Communications 8, 1971 (2017), Fig. 2 caption and main-text pair-breaking discussion, DOI 10.1038/s41467-017-01988-1",
            "manually_read_approximate_values": {
                "ring_minimum_energy_mev": 1.5,
                "ring_minimum_k_nm_inv": 0.024,
                "k_zero_feature_energy_mev": 7.0,
            },
            "transformations": [
                "curve shapes, widths, and relative JDOS amplitudes are illustrative",
                "reported high-energy fine doublet is not resolved in the schematic",
            ],
        },
        "literal_note2_bcs_result.svg": {
            "classification": "independently calculated literal printed Note-2 scalar-model output",
            "sources": {
                str(NOTE2.relative_to(ROOT)): sha256(NOTE2),
                str(NOTE2_REPORT.relative_to(ROOT)): sha256(NOTE2_REPORT),
            },
            "transformations": [
                "panel a display cropped to k <= 0.12 nm^-1 from a kmax=0.24 nm^-1 solver",
                "panel b uses saved physical k dk/(2pi) JDOS with Gaussian gamma=0.1 meV",
                "JDOS normalized by its saved maximum",
            ],
        },
        "same_parent_nr640_matrix_bcs_and_bare_current.svg": {
            "classification": "independently calculated fixed-mu radial exchange-only matrix state plus postflight-rejected bare-current lineshape",
            "sources": {
                **{str(p.relative_to(ROOT)): sha256(p) for p in (MATRIX, MATRIX_META, BUNDLE, OPTICAL, BARE_REPORT, POSTFLIGHT, POSTFLIGHT_REJECTED)},
            },
            "transformations": [
                "matrix-order curves are singular values of saved Sigma_F[:2,2:]",
                "band energies shifted by fixed mu_KP",
                "bare-current curve is the saved peak-normalized gamma=0.2 meV shape",
                "peak-position and gauge gates passed; overall postflight rejected because shape L1=0.03120597 > 0.02",
            ],
        },
    },
    "additional_report_sources": {
        str(SCALAR_CONTROL_REPORT.relative_to(ROOT)): sha256(SCALAR_CONTROL_REPORT),
    },
    "generator": {
        "path": "docs/figures/du2017/generate_figures.py",
        "sha256": sha256(generator_path),
    },
}
for name in provenance["figures"]:
    provenance["figures"][name]["figure_sha256"] = sha256(OUT/name)
(OUT/"provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False)+"\n")
print(json.dumps(provenance, indent=2, sort_keys=True))
