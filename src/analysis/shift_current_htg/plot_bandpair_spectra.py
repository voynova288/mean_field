from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def _eta_tag(eta_mev: float) -> str:
    return f"eta_{float(eta_mev):g}meV".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot spectra produced by run_htg_bandpair_spectra.py")
    parser.add_argument("--input", type=Path, required=True, help="htg_bandpair_spectra.npz")
    parser.add_argument("--summary", type=Path, default=None, help="summary.json; defaults to sibling summary.json")
    parser.add_argument("--group", default="central_flat")
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--components", nargs="+", default=["x;yy", "y;xx"])
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument(
        "--reflect-y",
        action="store_true",
        help="Apply y -> -y as part of a full tensor-basis transformation.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=0.0,
        help="Rotate the tensor basis by this angle in degrees after optional y reflection.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _component_tuple(component: str) -> tuple[int, int, int]:
    if ";" not in component:
        raise ValueError(f"Component must look like x;yy, got {component!r}")
    left, right = component.split(";", 1)
    labels = {"x": 0, "y": 1}
    return labels[left], labels[right[0]], labels[right[1]]


def _component_key(group: str, component: str, eta: str) -> str:
    return f"sigma_{_safe_key(f'{group}|{component}|{eta}') }"


def _basis_transform_matrix(rotation_deg: float, reflect_y: bool) -> np.ndarray:
    angle = np.deg2rad(float(rotation_deg))
    rot = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    reflection = np.diag([1.0, -1.0]) if reflect_y else np.eye(2)
    return rot @ reflection


def _load_transformed_component(
    data: np.lib.npyio.NpzFile,
    *,
    group: str,
    eta: str,
    component: str,
    transform: np.ndarray,
) -> np.ndarray:
    old_labels = ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")
    sample_key = _component_key(group, old_labels[0], eta)
    if sample_key not in data:
        direct_key = _component_key(group, component, eta)
        if direct_key not in data:
            available = "\n".join(sorted(k for k in data.files if k.startswith("sigma_")))
            raise KeyError(f"Missing all-component key {sample_key!r} and direct key {direct_key!r}. Available:\n{available}")
        direct = np.asarray(data[direct_key], dtype=float)
        if np.allclose(np.abs(transform), np.eye(2)):
            sign = np.prod([transform[index, index] for index in _component_tuple(component)])
            return float(sign) * direct
        raise KeyError("Full tensor rotation requires all eight tensor components in the input npz.")

    spectra = np.empty((2, 2, 2, np.asarray(data[sample_key]).size), dtype=float)
    for label in old_labels:
        key = _component_key(group, label, eta)
        if key not in data:
            raise KeyError(f"Missing component {label!r} required for tensor transformation: {key}")
        a, b, c = _component_tuple(label)
        spectra[a, b, c] = np.asarray(data[key], dtype=float)
    a_new, b_new, c_new = _component_tuple(component)
    return np.einsum("i,j,k,ijkw->w", transform[a_new], transform[b_new], transform[c_new], spectra)


def main() -> None:
    args = parse_args()
    summary_path = args.summary if args.summary is not None else args.input.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    data = np.load(args.input)
    photon = np.asarray(data["photon_energies_ev"], dtype=float)
    eta = _eta_tag(float(args.eta_mev))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    colors = {"x;yy": "#0047ff", "y;xx": "#e41a1c"}
    transform = _basis_transform_matrix(float(args.rotation_deg), bool(args.reflect_y))
    has_transform = bool(args.reflect_y) or abs(float(args.rotation_deg)) > 1.0e-12
    for component in args.components:
        values = _load_transformed_component(
            data,
            group=str(args.group),
            eta=eta,
            component=str(component),
            transform=transform,
        )
        label_suffix = ""
        if has_transform:
            label_suffix = f" (rot={args.rotation_deg:g}°, reflect_y={bool(args.reflect_y)})"
        ax.plot(
            photon,
            values,
            lw=2.0,
            color=colors.get(component),
            label=rf"$\sigma^{{{component}}}$" + label_suffix,
        )

    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel(r"photon energy $E_\gamma$ [eV]")
    ax.set_ylabel(r"$\sigma$ [$\mu$A nm V$^{-2}$]")
    convention_parts = []
    if args.reflect_y:
        convention_parts.append("y reflected")
    if abs(float(args.rotation_deg)) > 1.0e-12:
        convention_parts.append(f"rot={args.rotation_deg:g}°")
    convention = ", " + ", ".join(convention_parts) if convention_parts else ""
    ax.set_title(f"hTTG {args.group}, eta={args.eta_mev:g} meV{convention}")
    if args.xmax is not None:
        ax.set_xlim(float(np.min(photon)), float(args.xmax))
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.22, lw=0.6)

    text_lines = []
    config = summary.get("config", {})
    if config:
        text_lines.append(
            f"theta={config.get('theta_deg', '?')} deg, r={config.get('corrugation_r', '?')}, "
            f"Nk={config.get('mesh_size', '?')}, shells={config.get('n_shells', '?')}"
        )
    pair_groups = summary.get("pair_groups", {})
    if args.group in pair_groups:
        text_lines.append(f"pairs={pair_groups[args.group].get('pair_count', '?')}")
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85})

    output = args.output
    if output is None:
        output = args.input.with_name(f"{_safe_key(args.group)}_{eta}_spectra.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"))
    print(output)


if __name__ == "__main__":
    main()
