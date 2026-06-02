from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .run_chaudhary2021_noninteracting import _safe_key


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "results" / "shift_current_tbg" / "chaudhary2021_paperstyle_comparison"
PAPER_RENDER = REPO_ROOT / "tmp" / "pdfs" / "chaudhary2021" / "render"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style comparison using Chaudhary filling grid and paper-normalized axes.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--nonint-dir", type=Path, default=REPO_ROOT / "results/shift_current_tbg/chaudhary2021_b0_nonint_paperfill_lg7_m16_c3")
    p.add_argument("--hartree-dir", type=Path, default=REPO_ROOT / "results/shift_current_tbg/chaudhary2021_hartree_response_paperfill_lg7_m10_c3_eps15_T15K")
    p.add_argument("--paper-render-dir", type=Path, default=PAPER_RENDER)
    p.add_argument("--our-sign", type=float, default=-1.0, help="global diagnostic sign for the plotted current convention")
    return p.parse_args()


def _crop(page: Path, box: tuple[int, int, int, int]) -> Image.Image:
    return Image.open(page).convert("RGB").crop(box)


def _show(ax, img: Image.Image, title: str) -> None:
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")


def _fillings_from_summary(summary_path: Path) -> tuple[float, ...]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    run = s.get("run", {})
    values = run.get("fillings_nu_or_mu_labels", run.get("fillings_nu", run.get("filling", None)))
    if values is None:
        values = [-3.95, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 3.95]
    return tuple(float(x) for x in values)


def _spectrum(data: np.lib.npyio.NpzFile, *, prefix: str, filling: float, group: str, component: str = "y;xx") -> np.ndarray:
    key = f"nu={float(filling):g}|{group}|{component}"
    return np.asarray(data[f"{prefix}_{_safe_key(key)}"], dtype=float)


def _colors(fillings: tuple[float, ...]) -> dict[float, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("jet")
    lo = min(fillings)
    hi = max(fillings)
    return {nu: cmap((nu - lo) / (hi - lo)) for nu in fillings}


def _plot_ff_normalized(ax, data, summary: dict, fillings: tuple[float, ...], *, our_sign: float) -> float:
    moments = summary.get("transition_energy_moments", {})
    eps_ff_ev = float(moments.get("nu=0|FF", {}).get("mean_ev") or 0.0)
    if eps_ff_ev <= 0.0:
        eps_ff_ev = 0.0166
    x = np.asarray(data["photon_energies_ev"], dtype=float) / eps_ff_ev
    colors = _colors(fillings)
    for nu in fillings:
        y = _spectrum(data, prefix="sigma", filling=nu, group="FF") * float(our_sign)
        ax.plot(x, y, color=colors[nu], lw=1.25)
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set_xlim(0.2, 2.5)
    ax.set_ylim(-1300, 1300)
    ax.set_xlabel(r"$\omega/\langle\epsilon_{ff}\rangle$")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    ax.set_title(rf"Ours Fig. 2(b) style: $\langle\epsilon_{{ff}}\rangle={eps_ff_ev*1e3:.1f}$ meV, spectra$\times${our_sign:g}", loc="left", fontsize=10.5, fontweight="bold")
    return eps_ff_ev


def _plot_ff_hartree(ax, nonint, hartree, fillings: tuple[float, ...], *, our_sign: float) -> None:
    colors = _colors(fillings)
    x_non = np.asarray(nonint["photon_energies_ev"], dtype=float) * 1e3
    x_h = np.asarray(hartree["photon_energies_ev"], dtype=float) * 1e3
    for nu in fillings:
        ax.plot(x_non, _spectrum(nonint, prefix="sigma", filling=nu, group="FF") * our_sign, color=colors[nu], lw=0.8, ls="--", alpha=0.55)
        ax.plot(x_h, _spectrum(hartree, prefix="spectrum", filling=nu, group="FF") * our_sign, color=colors[nu], lw=1.2)
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set_xlim(3, 40)
    ax.set_ylim(-11000, 9500)
    ax.set_xlabel(r"$\omega$ (meV)")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    ax.set_title("Ours Fig. 3(b) style: dashed nonint, solid Hartree T=15 K", loc="left", fontsize=10.5, fontweight="bold")


def _plot_fd_hartree(ax, nonint, hartree, fillings: tuple[float, ...], *, our_sign: float) -> None:
    colors = _colors(fillings)
    x_non = np.asarray(nonint["photon_energies_ev"], dtype=float) * 1e3
    x_h = np.asarray(hartree["photon_energies_ev"], dtype=float) * 1e3
    for nu in fillings:
        ax.plot(x_non, _spectrum(nonint, prefix="sigma", filling=nu, group="FD") * our_sign, color=colors[nu], lw=0.85, ls="--", alpha=0.6)
        ax.plot(x_h, _spectrum(hartree, prefix="spectrum", filling=nu, group="FD") * our_sign, color=colors[nu], lw=1.15)
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set_xlim(0, 105)
    ax.set_ylim(-28000, 28000)
    ax.set_xlabel(r"$\omega$ (meV)")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    ax.set_title("Ours Fig. 4(b,c) style: dashed nonint, solid Hartree T=15 K", loc="left", fontsize=10.5, fontweight="bold")


def _add_filling_colorbar(fig, ax, fillings: tuple[float, ...]) -> None:
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(min(fillings), max(fillings))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label(r"filling $\nu$")
    cbar.set_ticks([-3.95, -2, 0, 2, 3.95])


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nonint_dir = Path(args.nonint_dir)
    hartree_dir = Path(args.hartree_dir)
    nonint_data = np.load(nonint_dir / "chaudhary2021_b0_noninteracting.npz")
    hartree_data = np.load(hartree_dir / "spectra_histograms.npz")
    nonint_summary = json.loads((nonint_dir / "summary.json").read_text(encoding="utf-8"))
    hartree_summary = json.loads((hartree_dir / "summary.json").read_text(encoding="utf-8"))
    fillings = _fillings_from_summary(nonint_dir / "summary.json")

    paper_dir = Path(args.paper_render_dir)
    crops = {
        "fig2b": _crop(paper_dir / "page-07.png", (605, 125, 1000, 355)),
        "fig3b": _crop(paper_dir / "page-10.png", (125, 715, 650, 1090)),
        "fig4bc": _crop(paper_dir / "page-11.png", (515, 205, 1390, 445)),
    }
    for key, img in crops.items():
        img.save(out / f"paper_crop_{key}.png")

    fig = plt.figure(figsize=(13.8, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.35), height_ratios=(0.82, 1.18, 0.95))
    _show(fig.add_subplot(gs[0, 0]), crops["fig2b"], "Paper Fig. 2(b): normalized noninteracting FF")
    ax2 = fig.add_subplot(gs[0, 1])
    eps_ff_ev = _plot_ff_normalized(ax2, nonint_data, nonint_summary, fillings, our_sign=float(args.our_sign))
    _add_filling_colorbar(fig, ax2, fillings)

    _show(fig.add_subplot(gs[1, 0]), crops["fig3b"], "Paper Fig. 3(b): FF nonint vs Hartree")
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_ff_hartree(ax3, nonint_data, hartree_data, fillings, our_sign=float(args.our_sign))

    _show(fig.add_subplot(gs[2, 0]), crops["fig4bc"], "Paper Fig. 4(b,c): FD nonint vs Hartree")
    ax4 = fig.add_subplot(gs[2, 1])
    _plot_fd_hartree(ax4, nonint_data, hartree_data, fillings, our_sign=float(args.our_sign))

    fig.suptitle("Chaudhary 2021 comparison after paper filling-grid and paper-style axes", fontsize=15, fontweight="bold")
    png = out / "chaudhary2021_paperstyle_vs_ours.png"
    pdf = out / "chaudhary2021_paperstyle_vs_ours.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    def peak(group: str, prefix: str, data, nu: float) -> dict[str, float]:
        x = np.asarray(data["photon_energies_ev"], dtype=float)
        y = _spectrum(data, prefix=prefix, filling=nu, group=group) * float(args.our_sign)
        i = int(np.argmax(np.abs(y)))
        return {"nu": float(nu), "energy_mev": float(x[i] * 1e3), "value": float(y[i]), "abs": float(abs(y[i]))}

    hartree_run = hartree_summary.get("run", {})
    eps_label = hartree_run.get("epsilon_r", "unknown")
    temp_label = hartree_run.get("occupation_temperature_k", "unknown")
    diagnostic = {
        "status": "paper-style side-by-side comparison, paper curves are raster crops not digitized data",
        "output_png": str(png),
        "output_pdf": str(pdf),
        "inputs": {
            "nonint_dir": str(nonint_dir),
            "hartree_dir": str(hartree_dir),
            "our_sign": float(args.our_sign),
            "fillings": [float(x) for x in fillings],
            "epsilon_ff_mean_mev": float(eps_ff_ev * 1e3),
            "hartree_run": hartree_run,
        },
        "peaks_selected": {
            "FF_nonint_nu_m2": peak("FF", "sigma", nonint_data, -2.0),
            "FF_nonint_nu_p2": peak("FF", "sigma", nonint_data, 2.0),
            "FF_hartree_nu_m2": peak("FF", "spectrum", hartree_data, -2.0),
            "FF_hartree_nu_p2": peak("FF", "spectrum", hartree_data, 2.0),
            "FD_nonint_nu_m3p95": peak("FD", "sigma", nonint_data, -3.95),
            "FD_nonint_nu_p3p95": peak("FD", "sigma", nonint_data, 3.95),
            "FD_hartree_nu_m3p95": peak("FD", "spectrum", hartree_data, -3.95),
            "FD_hartree_nu_p3p95": peak("FD", "spectrum", hartree_data, 3.95),
        },
        "known_gaps": [
            "Fig. 2(b) now uses the paper's filling grid and normalized x-axis; remaining differences are mostly small peak-position/weight shifts.",
            "FD panels remain convention-sensitive: exact filling-derived chemical potentials suppress the noninteracting edge peak, while gap chemical potentials at the edge labels recover a giant Fig. 4(b)-like peak.",
            f"Hartree panels use epsilon_r={eps_label} and response occupation T={temp_label} K; paper states T=0 Hartree and an effective dielectric fitted as in Ref. 65, so exact screening/temperature convention is still unresolved.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(diagnostic, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
