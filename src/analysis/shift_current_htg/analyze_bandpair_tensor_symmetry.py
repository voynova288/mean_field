from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")


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


def _component_tuple(component: str) -> tuple[int, int, int]:
    left, right = component.split(";", 1)
    labels = {"x": 0, "y": 1}
    return labels[left], labels[right[0]], labels[right[1]]


def _component_key(group: str, component: str, eta: str) -> str:
    return f"sigma_{_safe_key(f'{group}|{component}|{eta}') }"


def _load_tensor(data: np.lib.npyio.NpzFile, group: str, eta: str) -> np.ndarray:
    sample = _component_key(group, COMPONENTS[0], eta)
    if sample not in data:
        available = "\n".join(sorted(k for k in data.files if k.startswith("sigma_")))
        raise KeyError(f"Need all eight tensor components; missing {sample!r}. Available:\n{available}")
    n_energy = np.asarray(data[sample]).size
    tensor = np.empty((2, 2, 2, n_energy), dtype=float)
    for component in COMPONENTS:
        key = _component_key(group, component, eta)
        if key not in data:
            raise KeyError(f"Missing component {component!r}: {key}")
        a, b, c = _component_tuple(component)
        tensor[a, b, c] = np.asarray(data[key], dtype=float)
    return tensor


def _basis_transform_matrix(rotation_deg: float, reflect_y: bool = False) -> np.ndarray:
    angle = np.deg2rad(float(rotation_deg))
    rot = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=float)
    reflection = np.diag([1.0, -1.0]) if reflect_y else np.eye(2)
    return rot @ reflection


def _transform_tensor(tensor: np.ndarray, rotation_deg: float, reflect_y: bool = False) -> np.ndarray:
    transform = _basis_transform_matrix(rotation_deg, reflect_y=reflect_y)
    return np.einsum("ai,bj,ck,ijkw->abcw", transform, transform, transform, tensor, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose C3/C3v tensor structure of hTTG band-pair spectra without visual fitting."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--group", default="central_flat")
    parser.add_argument("--eta-mev", type=float, default=2.0)
    parser.add_argument("--reflect-y", action="store_true")
    parser.add_argument("--rotation-grid-deg", default="-30,-20,-15,-10,-5,0,5,10,15,20,30")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    photon = np.asarray(data["photon_energies_ev"], dtype=float)
    eta = _eta_tag(float(args.eta_mev))
    tensor = _load_tensor(data, str(args.group), eta)

    # C3 irreducible coefficients in the paper's Eq. (15) notation.
    group1_stack = np.stack([tensor[0, 1, 1], -tensor[0, 0, 0], tensor[1, 1, 0], tensor[1, 0, 1]], axis=0)
    group2_stack = np.stack([tensor[1, 0, 0], -tensor[1, 1, 1], tensor[0, 0, 1], tensor[0, 1, 0]], axis=0)
    c3_group1 = np.mean(group1_stack, axis=0)
    c3_group2 = np.mean(group2_stack, axis=0)
    group1_error = float(np.max(np.abs(group1_stack - c3_group1[None, :])))
    group2_error = float(np.max(np.abs(group2_stack - c3_group2[None, :])))
    max_group1 = float(np.max(np.abs(c3_group1)))
    max_group2 = float(np.max(np.abs(c3_group2)))
    mirror_like_ratio = max_group1 / max_group2 if max_group2 > 0 else None

    rotations = [float(x.strip()) for x in str(args.rotation_grid_deg).split(",") if x.strip()]
    rotation_summary: dict[str, dict[str, float]] = {}
    for angle in rotations:
        transformed = _transform_tensor(tensor, angle, reflect_y=bool(args.reflect_y))
        xyy = transformed[0, 1, 1]
        yxx = transformed[1, 0, 0]
        rotation_summary[f"{angle:g}"] = {
            "max_abs_x_yy_uA_nm_per_V2": float(np.max(np.abs(xyy))),
            "energy_x_yy_peak_ev": float(photon[int(np.argmax(np.abs(xyy)))]),
            "signed_x_yy_at_own_peak_uA_nm_per_V2": float(xyy[int(np.argmax(np.abs(xyy)))]),
            "max_abs_y_xx_uA_nm_per_V2": float(np.max(np.abs(yxx))),
            "energy_y_xx_peak_ev": float(photon[int(np.argmax(np.abs(yxx)))]),
            "signed_y_xx_at_own_peak_uA_nm_per_V2": float(yxx[int(np.argmax(np.abs(yxx)))]),
        }

    out_dir = args.output_dir if args.output_dir is not None else args.input.with_name("tensor_symmetry_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(args.input),
        "group": str(args.group),
        "eta_mev": float(args.eta_mev),
        "reflect_y_for_rotation_scan": bool(args.reflect_y),
        "c3_errors_uA_nm_per_V2": {
            "group1_max_abs_deviation": group1_error,
            "group2_max_abs_deviation": group2_error,
        },
        "c3_irrep_peak_uA_nm_per_V2": {
            "group1_A_like_max_abs": max_group1,
            "group2_B_like_max_abs": max_group2,
            "group1_over_group2_peak_ratio": mirror_like_ratio,
        },
        "rotation_scan": rotation_summary,
    }
    (out_dir / "tensor_symmetry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    axes[0].plot(photon, c3_group1, label="C3 group1 A", color="#0047ff")
    axes[0].plot(photon, c3_group2, label="C3 group2 B", color="#e41a1c")
    axes[0].axhline(0.0, color="0.4", lw=0.7)
    axes[0].set_xlabel(r"photon energy $E_\gamma$ [eV]")
    axes[0].set_ylabel(r"C3 coefficient [$\mu$A nm V$^{-2}$]")
    axes[0].legend(frameon=False)
    axes[0].set_title("C3 tensor coefficients")

    angles = np.asarray(rotations, dtype=float)
    xyy_peaks = np.asarray([rotation_summary[f"{angle:g}"]["max_abs_x_yy_uA_nm_per_V2"] for angle in angles])
    yxx_peaks = np.asarray([rotation_summary[f"{angle:g}"]["max_abs_y_xx_uA_nm_per_V2"] for angle in angles])
    axes[1].plot(angles, xyy_peaks, "o-", label=r"$|\sigma^{x;yy}|$ peak")
    axes[1].plot(angles, yxx_peaks, "s--", label=r"$|\sigma^{y;xx}|$ peak")
    axes[1].set_xlabel("basis rotation [deg]")
    axes[1].set_ylabel(r"peak $|\sigma|$ [$\mu$A nm V$^{-2}$]")
    axes[1].legend(frameon=False)
    axes[1].set_title("axis-dependence diagnostic")
    fig.savefig(out_dir / "tensor_symmetry_audit.png", dpi=180)
    fig.savefig(out_dir / "tensor_symmetry_audit.pdf")
    plt.close(fig)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(out_dir / "tensor_symmetry_audit.png")


if __name__ == "__main__":
    main()
