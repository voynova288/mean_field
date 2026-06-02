from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _safe_key(text: str) -> str:
    return (
        str(text)
        .replace(";", "_")
        .replace(",", "_")
        .replace(":", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace("|", "_")
        .replace(".", "p")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make Chaudhary 2021 Fig. 2-style plots from a noninteracting run.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--component", default="y;xx")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fillings", default=None, help="comma-separated fillings; default uses summary run list")
    parser.add_argument("--fd-reference-filling", type=float, default=0.0)
    parser.add_argument("--ff-reference-filling", type=float, default=0.0)
    parser.add_argument("--fd-reference", choices=("peak", "mean"), default="peak")
    parser.add_argument("--ff-reference", choices=("peak", "mean"), default="mean")
    return parser.parse_args()


def _load_energy_reference(summary: dict, *, group: str, filling: float, component: str, source: str) -> float:
    if source == "mean":
        key = f"nu={filling:g}|{group}"
        moment = summary.get("transition_energy_moments", {}).get(key)
        if moment is None or moment.get("mean_ev") is None:
            raise KeyError(f"No transition-energy moment {key!r} in summary")
        return float(moment["mean_ev"])
    key = f"nu={filling:g}|{group}|{component}"
    peak = summary.get("peaks", {}).get(key)
    if peak is None:
        raise KeyError(f"No peak summary {key!r} in summary")
    return float(peak["energy_at_max_abs_ev"])


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    component = str(args.component)
    fillings = (
        tuple(float(part.strip()) for part in str(args.fillings).split(",") if part.strip())
        if args.fillings is not None
        else tuple(float(x) for x in summary["run"].get("fillings_nu", summary["run"].get("fillings_nu_or_mu_labels", ())))
    )
    explicit_mu_mev = summary.get("run", {}).get("explicit_mu_mev")
    label_is_mu = explicit_mu_mev is not None
    photon_ev = np.asarray(data["photon_energies_ev"], dtype=float)
    photon_mev = 1.0e3 * photon_ev
    path_kdist = np.asarray(data["path_kdist"], dtype=float)
    bands = np.asarray(data["path_bands_ev"], dtype=float)

    eps_ff = _load_energy_reference(
        summary,
        group="FF",
        filling=float(args.ff_reference_filling),
        component=component,
        source=str(args.ff_reference),
    )
    eps_fd = _load_energy_reference(
        summary,
        group="FD",
        filling=float(args.fd_reference_filling),
        component=component,
        source=str(args.fd_reference),
    )
    lm_nm = float(summary["config"]["moire_length_nm"])
    fd_scale = eps_fd * eps_fd / lm_nm

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    ax_band, ax_ff, ax_fd_raw, ax_fd_scaled = axes.ravel()

    for ib in range(bands.shape[1]):
        ax_band.plot(path_kdist, 1.0e3 * bands[:, ib], color="black", lw=0.8)
    ax_band.axhline(0.0, color="0.6", lw=0.8)
    ax_band.set_title("TBG bands near charge neutrality")
    ax_band.set_ylabel("E [meV]")
    ax_band.set_xticks([])

    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(fillings)))
    for filling, color in zip(fillings, colors, strict=True):
        curve_label = (rf"$\mu$={filling:g} meV" if label_is_mu else f"nu={filling:g}")
        for group, ax in (("FF", ax_ff), ("FD", ax_fd_raw)):
            key = f"nu={filling:g}|{group}|{component}"
            arr_key = f"sigma_{_safe_key(key)}"
            if arr_key not in data:
                continue
            sigma = np.asarray(data[arr_key], dtype=float)
            if group == "FF":
                ax.plot(photon_ev / eps_ff, sigma, color=color, lw=1.4, label=curve_label)
            else:
                ax.plot(photon_mev, sigma, color=color, lw=1.1, alpha=0.55, label=curve_label)
                ax_fd_scaled.plot(photon_ev / eps_fd, sigma * fd_scale, color=color, lw=1.4, label=curve_label)

    ax_ff.axhline(0.0, color="0.6", lw=0.8)
    ax_ff.set_title(rf"FF: $\omega/\epsilon_{{FF}}$, $\epsilon_{{FF}}={1e3*eps_ff:.1f}$ meV")
    ax_ff.set_xlabel(r"$\omega/\epsilon_{FF}$")
    ax_ff.set_ylabel(rf"$\sigma^{{{component}}}$ [$\mu$A nm V$^{{-2}}$]")
    ax_ff.legend(fontsize=8, ncols=2)

    ax_fd_raw.axhline(0.0, color="0.6", lw=0.8)
    ax_fd_raw.set_title("FD raw spectrum")
    ax_fd_raw.set_xlabel("photon energy [meV]")
    ax_fd_raw.set_ylabel(rf"$\sigma^{{{component}}}$ [$\mu$A nm V$^{{-2}}$]")

    ax_fd_scaled.axhline(0.0, color="0.6", lw=0.8)
    ax_fd_scaled.set_title(rf"FD: $\omega/\epsilon_{{FD}}$, $\epsilon_{{FD}}={1e3*eps_fd:.1f}$ meV; scaled by $\epsilon_{{FD}}^2/L_M$")
    ax_fd_scaled.set_xlabel(r"$\omega/\epsilon_{FD}$")
    ax_fd_scaled.set_ylabel(r"scaled conductivity [arb.]")
    ax_fd_scaled.legend(fontsize=8, ncols=2)

    fig.suptitle(
        f"Chaudhary 2021 noninteracting TBG, theta={summary['config']['theta_deg']}°, "
        f"Delta=({summary['config']['delta1_ev']*1e3:.1f},{summary['config']['delta2_ev']*1e3:.1f}) meV, "
        f"mesh={summary['run']['mesh_size']}, shells={summary['config']['n_shells']}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
