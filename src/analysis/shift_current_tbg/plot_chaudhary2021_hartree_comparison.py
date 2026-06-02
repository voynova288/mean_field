from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .run_chaudhary2021_noninteracting import _parse_float_csv, _safe_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare noninteracting and Hartree-corrected Chaudhary spectra.")
    parser.add_argument("--nonint-dir", type=Path, required=True)
    parser.add_argument("--hartree-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fillings", type=_parse_float_csv, default=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0))
    parser.add_argument("--component", default="y;xx")
    parser.add_argument("--nonint-degeneracy-scale", type=float, default=None, help="manual scale applied to noninteracting spectra; default rescales to per-flavor response degeneracy 1 from summary")
    parser.add_argument("--xmax-mev", type=float, default=120.0)
    return parser.parse_args()


def _load_summary(directory: Path) -> dict:
    path = Path(directory) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _nonint_scale(summary: dict, manual: float | None) -> float:
    if manual is not None:
        return float(manual)
    run = summary.get("run", {})
    deg = run.get("response_degeneracy_multiplier", run.get("degeneracy_multiplier", 1.0))
    return 1.0 / float(deg)


def _peak(energies_ev: np.ndarray, sigma: np.ndarray) -> dict[str, float]:
    if sigma.size == 0:
        return {"energy_mev": float("nan"), "signed_uA_nm_per_V2": float("nan"), "max_abs_uA_nm_per_V2": float("nan")}
    idx = int(np.argmax(np.abs(sigma)))
    return {
        "energy_mev": float(energies_ev[idx] * 1.0e3),
        "signed_uA_nm_per_V2": float(sigma[idx]),
        "max_abs_uA_nm_per_V2": float(np.max(np.abs(sigma))),
    }


def _get_spectrum(data: np.lib.npyio.NpzFile, *, prefix: str, filling: float, group: str, component: str) -> np.ndarray:
    key = f"{prefix}_{_safe_key(f'nu={filling:g}_{group}_{component}')}"
    return np.asarray(data[key], dtype=float)


def main() -> None:
    args = parse_args()
    nonint_summary = _load_summary(args.nonint_dir)
    hartree_summary = _load_summary(args.hartree_dir)
    nonint_npz = np.load(Path(args.nonint_dir) / "chaudhary2021_b0_noninteracting.npz")
    hartree_npz = np.load(Path(args.hartree_dir) / "spectra_histograms.npz")
    e_non = np.asarray(nonint_npz["photon_energies_ev"], dtype=float)
    e_h = np.asarray(hartree_npz["photon_energies_ev"], dtype=float)
    scale_non = _nonint_scale(nonint_summary, args.nonint_degeneracy_scale)

    fillings = tuple(float(x) for x in args.fillings)
    groups = ("FF", "FD")
    rows: list[dict[str, object]] = []
    for filling in fillings:
        for group in groups:
            non = _get_spectrum(nonint_npz, prefix="sigma", filling=filling, group=group, component=args.component) * scale_non
            har = _get_spectrum(hartree_npz, prefix="spectrum", filling=filling, group=group, component=args.component)
            p_non = _peak(e_non, non)
            p_h = _peak(e_h, har)
            rows.append(
                {
                    "filling": float(filling),
                    "group": group,
                    "nonint": p_non,
                    "hartree": p_h,
                    "max_abs_ratio_hartree_over_nonint": float(
                        p_h["max_abs_uA_nm_per_V2"] / p_non["max_abs_uA_nm_per_V2"]
                    )
                    if p_non["max_abs_uA_nm_per_V2"] not in (0.0, float("nan"))
                    else None,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "Diagnostic comparison; noninteracting spectra are rescaled to response degeneracy 1.",
        "nonint_dir": str(args.nonint_dir),
        "hartree_dir": str(args.hartree_dir),
        "component": str(args.component),
        "nonint_scale_applied": float(scale_non),
        "hartree_run": hartree_summary.get("run", {}),
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Chaudhary Hartree vs noninteracting comparison",
        "",
        "Noninteracting spectra are rescaled to response degeneracy `1`.",
        "",
        "| nu | group | nonint peak | Hartree peak | ratio |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        p_non = row["nonint"]
        p_h = row["hartree"]
        ratio = row["max_abs_ratio_hartree_over_nonint"]
        lines.append(
            f"| {row['filling']:g} | {row['group']} | "
            f"{p_non['signed_uA_nm_per_V2']:.3g} @ {p_non['energy_mev']:.2f} meV | "
            f"{p_h['signed_uA_nm_per_V2']:.3g} @ {p_h['energy_mev']:.2f} meV | "
            f"{ratio:.3g} |"
            if ratio is not None
            else f"| {row['filling']:g} | {row['group']} | {p_non['signed_uA_nm_per_V2']:.3g} | {p_h['signed_uA_nm_per_V2']:.3g} | n/a |"
        )
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3), constrained_layout=True)
    for ax, group in zip(axes, groups, strict=True):
        for idx, filling in enumerate(fillings):
            color = colors(idx % 10)
            non = _get_spectrum(nonint_npz, prefix="sigma", filling=filling, group=group, component=args.component) * scale_non
            har = _get_spectrum(hartree_npz, prefix="spectrum", filling=filling, group=group, component=args.component)
            ax.plot(e_non * 1.0e3, non, ls="--", lw=1.0, color=color, alpha=0.75)
            ax.plot(e_h * 1.0e3, har, ls="-", lw=1.4, color=color, label=rf"$\nu={filling:g}$")
        ax.axhline(0.0, color="0.65", lw=0.7)
        ax.set_xlim(0.0, float(args.xmax_mev))
        ax.set_xlabel("photon energy (meV)")
        ax.set_ylabel(r"$\sigma$ ($\mu$A nm V$^{-2}$)")
        ax.set_title(f"{group} {args.component}: dashed nonint, solid Hartree")
        ax.legend(fontsize=8, ncol=2)
    fig.savefig(args.output_dir / "hartree_vs_nonint.png", dpi=220)
    fig.savefig(args.output_dir / "hartree_vs_nonint.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
