from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from analysis.shift_current_htg.constants import eta_mev_to_ev
from .run_chaudhary2021_noninteracting import _safe_key, _spectrum_from_histogram


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER_RENDER = REPO_ROOT / "tmp" / "pdfs" / "chaudhary2021" / "render"
DEFAULT_OUT = REPO_ROOT / "results" / "shift_current_tbg" / "chaudhary2021_paper_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make side-by-side comparison panels: Chaudhary paper raster crops vs our calculated spectra.")
    parser.add_argument("--paper-render-dir", type=Path, default=DEFAULT_PAPER_RENDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nonint-dir", type=Path, default=REPO_ROOT / "results" / "shift_current_tbg" / "chaudhary2021_b0_nonint_fig2_fd_same_lg7_m16_c3")
    parser.add_argument("--mu-dir", type=Path, default=REPO_ROOT / "results" / "shift_current_tbg" / "chaudhary2021_b0_nonint_fig2_mu_m30_0_p30_fd1_lg7_m16_c3")
    parser.add_argument("--hartree-dir", type=Path, default=REPO_ROOT / "results" / "shift_current_tbg" / "chaudhary2021_hartree_response_lg7_m12_c3_eps15_T15K")
    parser.add_argument("--response-degeneracy-scale", type=float, default=0.25, help="scale old noninteracting deg=4 spectra to per-flavor deg=1")
    parser.add_argument("--our-sign", type=float, default=1.0, help="diagnostic global sign applied to all calculated spectra; use -1 to align paper current/sign convention visually")
    parser.add_argument("--tag", default="raw", help="filename tag for outputs")
    return parser.parse_args()


def _crop(path: Path, box: tuple[int, int, int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").crop(box)


def _show_image(ax, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")


def _load_npz(directory: Path, filename: str = "chaudhary2021_b0_noninteracting.npz"):
    return np.load(Path(directory) / filename)


def _spectrum(data, *, prefix: str, filling: float, group: str, component: str = "y;xx") -> np.ndarray:
    return np.asarray(data[f"{prefix}_{_safe_key(f'nu={filling:g}_{group}_{component}')}"], dtype=float)


def _hist_spectrum(data, *, filling: float, group: str, component: str = "y;xx", eta_mev: float) -> np.ndarray:
    hist = np.asarray(data[f"hist_{_safe_key(f'nu={filling:g}_{group}_{component}')}"], dtype=np.complex128)
    return _spectrum_from_histogram(
        np.asarray(data["photon_energies_ev"], dtype=float),
        np.asarray(data["energy_edges_ev"], dtype=float),
        hist,
        eta_ev=eta_mev_to_ev(float(eta_mev)),
    )


def _nu_colors(fillings: tuple[float, ...]):
    cmap = plt.get_cmap("coolwarm")
    if len(fillings) == 1:
        return {fillings[0]: cmap(0.5)}
    return {nu: cmap(i / (len(fillings) - 1)) for i, nu in enumerate(fillings)}


def _plot_nonint_ff(ax, data, scale: float, *, our_sign: float = 1.0) -> None:
    fillings = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    colors = _nu_colors(fillings)
    e = np.asarray(data["photon_energies_ev"], dtype=float) * 1.0e3
    for nu in fillings:
        ax.plot(e, _spectrum(data, prefix="sigma", filling=nu, group="FF") * scale * float(our_sign), color=colors[nu], lw=1.8, label=rf"$\nu={nu:g}$")
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlim(0, 45)
    ax.set_xlabel(r"$\omega$ (meV)")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    suffix = "" if float(our_sign) == 1.0 else rf" (global sign x {our_sign:g})"
    ax.set_title("Our noninteracting FF, per flavor" + suffix, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=True)


def _plot_mu_fd(ax, data, scale: float, *, our_sign: float = 1.0) -> None:
    colors = {-30.0: "#4b0082", 0.0: "#00cc55", 30.0: "#aa6666"}
    e = np.asarray(data["photon_energies_ev"], dtype=float) * 1.0e3
    for mu in (-30.0, 0.0, 30.0):
        sigma = _hist_spectrum(data, filling=mu, group="FD", eta_mev=10.0) * scale * float(our_sign)
        ax.plot(e, sigma, color=colors[mu], lw=1.8, label=rf"$\mu={mu:g}$ meV")
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlim(0, 55)
    ax.set_xlabel(r"$\omega$ (meV)")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    suffix = "" if float(our_sign) == 1.0 else rf" (global sign x {our_sign:g})"
    ax.set_title("Our Fig. 2(e)-style FD, eta=10 meV, per flavor" + suffix, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=7, frameon=True)


def _plot_hartree_compare(ax, nonint, hartree, group: str, scale: float, *, our_sign: float = 1.0) -> None:
    fillings = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    colors = _nu_colors(fillings)
    e_non = np.asarray(nonint["photon_energies_ev"], dtype=float) * 1.0e3
    e_h = np.asarray(hartree["photon_energies_ev"], dtype=float) * 1.0e3
    for nu in fillings:
        ax.plot(e_non, _spectrum(nonint, prefix="sigma", filling=nu, group=group) * scale * float(our_sign), color=colors[nu], lw=1.0, ls="--", alpha=0.72)
        ax.plot(e_h, _spectrum(hartree, prefix="spectrum", filling=nu, group=group) * float(our_sign), color=colors[nu], lw=1.55, label=rf"$\nu={nu:g}$")
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.set_xlim(0, 110 if group == "FD" else 45)
    ax.set_xlabel(r"$\omega$ (meV)")
    ax.set_ylabel(r"$\sigma^{y;xx}$ ($\mu$A nm V$^{-2}$)")
    suffix = "" if float(our_sign) == 1.0 else rf" (global sign x {our_sign:g})"
    ax.set_title(f"Our {group}: dashed nonint, solid Hartree T=15 K" + suffix, fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=7, ncol=2, frameon=True)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paper = Path(args.paper_render_dir)
    crops = {
        "fig2b": _crop(paper / "page-07.png", (540, 115, 930, 345)),
        "fig2e": _crop(paper / "page-07.png", (720, 385, 1310, 510)),
        "fig3b": _crop(paper / "page-10.png", (130, 720, 635, 1080)),
        "fig4bc": _crop(paper / "page-11.png", (520, 200, 1385, 440)),
    }
    for name, img in crops.items():
        img.save(out / f"paper_crop_{name}.png")

    nonint = _load_npz(Path(args.nonint_dir))
    mu_data = _load_npz(Path(args.mu_dir))
    hartree = np.load(Path(args.hartree_dir) / "spectra_histograms.npz")
    scale = float(args.response_degeneracy_scale)
    our_sign = float(args.our_sign)
    tag = str(args.tag).strip() or "raw"

    fig = plt.figure(figsize=(13.6, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=(1.0, 1.2), height_ratios=(1.0, 0.62, 1.05, 0.82))

    _show_image(fig.add_subplot(gs[0, 0]), crops["fig2b"], "Paper Fig. 2(b): noninteracting FF")
    _plot_nonint_ff(fig.add_subplot(gs[0, 1]), nonint, scale, our_sign=our_sign)

    _show_image(fig.add_subplot(gs[1, 0]), crops["fig2e"], "Paper Fig. 2(e), top: FD at fixed mu")
    _plot_mu_fd(fig.add_subplot(gs[1, 1]), mu_data, scale, our_sign=our_sign)

    _show_image(fig.add_subplot(gs[2, 0]), crops["fig3b"], "Paper Fig. 3(b): FF, nonint vs Hartree")
    _plot_hartree_compare(fig.add_subplot(gs[2, 1]), nonint, hartree, "FF", scale, our_sign=our_sign)

    _show_image(fig.add_subplot(gs[3, 0]), crops["fig4bc"], "Paper Fig. 4(b,c): FD, nonint vs Hartree")
    _plot_hartree_compare(fig.add_subplot(gs[3, 1]), nonint, hartree, "FD", scale, our_sign=our_sign)

    fig.suptitle(
        "Chaudhary 2021 paper panels vs current reproduction (side-by-side, not digitized overlay)" + ("" if our_sign == 1.0 else f"; calculated spectra x {our_sign:g}"),
        fontsize=15,
        fontweight="bold",
    )
    output_png = out / f"chaudhary2021_paper_vs_ours_{tag}.png"
    output_pdf = out / f"chaudhary2021_paper_vs_ours_{tag}.pdf"
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)

    summary = {
        "status": "Side-by-side visual comparison using rendered paper crops; paper curves are not digitized, so this is not a numerical overlay.",
        "output_png": str(output_png),
        "output_pdf": str(output_pdf),
        "paper_crops": {key: str(out / f"paper_crop_{key}.png") for key in crops},
        "our_inputs": {
            "noninteracting_dir": str(args.nonint_dir),
            "mu_dir": str(args.mu_dir),
            "hartree_dir": str(args.hartree_dir),
            "response_degeneracy_scale": scale,
            "our_sign": our_sign,
            "hartree_convention": "T=15 K, epsilon_r=15, lg7/m12 response, fd_bands=10 for FD panel",
        },
        "comparison_notes": [
            "Fig. 2(b): noninteracting FF peak scale and filling sign pattern agree qualitatively; x-axis differs because paper normalizes frequency by the average flat-flat gap.",
            "Fig. 2(e): explicit mu cuts reproduce opposite signs and zero/near-zero neutrality; plotted with eta=10 meV per-flavor scale.",
            "Fig. 3(b): Hartree FF enhancement and narrowed low-energy resonance are reproduced qualitatively.",
            "Fig. 4(b,c): Hartree shifts FD weight from the noninteracting high-energy scale to strong low-energy peaks; exact amplitude/epsilon convention remains under audit.",
        ],
    }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# Chaudhary 2021 paper-vs-ours comparison\n\n"
        "Generated side-by-side comparison.  The left column uses raster crops from the rendered paper; "
        "the right column uses the current calculated spectra.  This is a visual comparison, not a digitized numerical overlay.\n\n"
        f"- Main PNG: `{output_png}`\n"
        f"- Main PDF: `{output_pdf}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
