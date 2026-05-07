#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYTWIST_ROOT = Path("/data/home/ziyuzhu/pytwist")
DEFAULT_BM_REFERENCE_DIR = REPO_ROOT / "results" / "BM" / "theta_105_vf2416_w0_88_w1_110_mu0_strain0_20260422"
DEFAULT_TTG_CHIRAL_DIR = DEFAULT_PYTWIST_ROOT / "custom_outputs" / "ttg_theta_1p53_phi_0p0_epsilon_0p0_D_0p0_cut_5p0_u_0p0_up_0p0975"
DEFAULT_TTG_REALISTIC_DIR = DEFAULT_PYTWIST_ROOT / "custom_outputs" / "ttg_theta_1p53_phi_0p0_epsilon_0p0_D_0p0_cut_5p0_u_0p034125_up_0p0975"

from mean_field.runtime import collect_runtime_environment, current_timestamp
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import build_b0_benchmark_kpath, solve_bm_model


@dataclass(frozen=True)
class AlignmentSummary:
    name: str
    max_abs_diff: float
    rms_diff: float
    unit: str
    artifact_pngs: dict[str, str]


@dataclass(frozen=True)
class MappedPathBands:
    kpath: np.ndarray
    combined_energies: np.ndarray
    subspace_energies: tuple[np.ndarray, ...]
    labels: tuple[str, ...]
    singular_values: np.ndarray


@dataclass(frozen=True)
class FamilyPlotCase:
    n_layers: int
    theta_deg: float
    bands: MappedPathBands
    energy_scale_vfkdtheta: float
    alpha_dimless: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the n=2/n=3 ATMG reference family and generate n=4/5/6 band plots."
    )
    parser.add_argument("--bm-reference-dir", type=Path, default=DEFAULT_BM_REFERENCE_DIR)
    parser.add_argument("--ttg-chiral-dir", type=Path, default=DEFAULT_TTG_CHIRAL_DIR)
    parser.add_argument("--ttg-realistic-dir", type=Path, default=DEFAULT_TTG_REALISTIC_DIR)
    parser.add_argument("--pytwist-root", type=Path, default=DEFAULT_PYTWIST_ROOT)
    parser.add_argument("--res", type=int, default=64, help="TTG-style path resolution.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--energy-unit",
        choices=("native", "vfkdtheta", "both"),
        default="native",
        help="Plot energies in native units, E/(v_f*k_d*theta), or both.",
    )
    parser.add_argument(
        "--native-window",
        type=float,
        default=0.10,
        help="Half-window for native TTG/ATMG plots in eV.",
    )
    parser.add_argument(
        "--scaled-window",
        type=float,
        default=1.0,
        help="Half-window for scaled plots in E/(v_f*k_d*theta).",
    )
    return parser.parse_args()


def _default_output_dir() -> Path:
    return REPO_ROOT / "results" / "ATMG" / f"reference_alignment_{current_timestamp().replace(':', '').replace('-', '')}"


def _load_pytwist_module(pytwist_root: Path | str):
    pytwist_root = Path(pytwist_root)
    spec = importlib.util.spec_from_file_location("pytwist_local", pytwist_root / "pytwist.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load pytwist.py from {pytwist_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bm_reference(path_tsv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path_tsv}")

    kdist = np.asarray([float(row["k_dist"]) for row in rows], dtype=float)
    kvec = np.asarray([complex(float(row["kx"]), float(row["ky"])) for row in rows], dtype=np.complex128)
    band_columns = [name for name in rows[0] if name.startswith("band_") and name.endswith("_meV")]
    band_columns.sort()
    energies = np.asarray(
        [[float(row[column]) for column in band_columns] for row in rows],
        dtype=float,
    )
    return kdist, kvec, energies


def _load_bm_metadata(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_bm_nodes(nodes_tsv: Path) -> tuple[list[float], list[str]]:
    with nodes_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    return [float(row["k_dist"]) for row in rows], [row["label"] for row in rows]


def _build_bm_params(metadata: dict[str, object]) -> tuple[TBGParameters, int, int]:
    params = metadata["parameters"]
    assert isinstance(params, dict)
    return (
        TBGParameters(
            dtheta_rad=float(params["theta_deg"]) * math.pi / 180.0,
            convention="b0",
            vf=float(params["vf"]),
            chemical_potential=float(params["chemical_potential"]),
            w0=float(params["w0"]),
            w1=float(params["w1"]),
            delta=float(params["delta"]),
            strain=float(params["strain"]),
            strain_angle_rad=float(params["strain_angle_rad"]),
            poisson=float(params["poisson"]),
            beta_g=float(params["beta_g"]),
            alpha=float(params["alpha"]),
            deformation_potential=float(params["deformation_potential"]),
        ),
        int(params["points_per_segment"]),
        int(params["lg"]),
    )


def _rerun_bm_reference(reference_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata = _load_bm_metadata(reference_dir / "run_metadata.json")
    params, points_per_segment, lg = _build_bm_params(metadata)
    path = build_b0_benchmark_kpath(params, points_per_segment)
    solution = solve_bm_model(params, path.kvec, lg=lg, sigma_rotation=True)
    energies = np.asarray(solution.flattened_energies().T, dtype=float)
    return np.asarray(path.kdist, dtype=float), energies


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ttg_ticks(res: int) -> tuple[list[int], list[str]]:
    l1 = int(res)
    l2 = int(res)
    l3 = int(np.sqrt(3) * res / 2)
    l4 = int(res / 2)
    positions = [0, l1 - 1, l1 + l2 - 2, l1 + l2 + l3 - 3, l1 + l2 + l3 + l4 - 4]
    labels = ["K'", "K", r"$\Gamma$", "M", "K'"]
    return positions, labels


def _path_distances(kpath: np.ndarray) -> np.ndarray:
    if len(kpath) == 0:
        return np.asarray([], dtype=float)
    xcoords = np.zeros(len(kpath), dtype=float)
    if len(kpath) > 1:
        xcoords[1:] = np.cumsum(np.linalg.norm(np.diff(kpath, axis=0), axis=1))
    return xcoords


def _requested_energy_units(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return ("native", "vfkdtheta")
    return (selection,)


def _pytwist_energy_scale_vfkdtheta(theta_deg: float, vf: float, a: float) -> float:
    theta_rad = float(theta_deg) * math.pi / 180.0
    dirac_velocity = float(vf) * 2.1354 * float(a)
    k_d = 4.0 * math.pi / (3.0 * float(a))
    return dirac_velocity * k_d * theta_rad


def _bm_energy_scale_vfkdtheta(vf: float, theta_deg: float) -> float:
    theta_rad = float(theta_deg) * math.pi / 180.0
    k_d = 4.0 * math.pi / 3.0
    return float(vf) * k_d * theta_rad


@dataclass(frozen=True)
class ExactCommonBasisFamily:
    module: object
    theta_deg: float
    phi_deg: float
    epsilon: float
    vf: float
    u: float
    up: float
    cut: float
    a: float = 0.246
    beta: float = 3.14
    delta: float = 0.16

    def __post_init__(self) -> None:
        theta = float(self.theta_deg) * np.pi / 180.0
        phi = float(self.phi_deg) * np.pi / 180.0
        template = self.module.TTGModel(
            float(self.theta_deg),
            float(self.phi_deg),
            float(self.epsilon),
            0.0,
            a=float(self.a),
            beta=float(self.beta),
            delta=float(self.delta),
            vf=float(self.vf),
            u=float(self.u),
            up=float(self.up),
            cut=float(self.cut),
        )
        q = np.asarray(template.q, dtype=float)
        k_theta = float(template.k_theta)
        even = np.asarray([row[:2] for row in template.Q if int(row[2]) == 0], dtype=float)
        odd = np.asarray([row[:2] for row in template.Q if int(row[2]) == 1], dtype=float)
        even_map = {self._rounded_key(value): idx for idx, value in enumerate(even)}
        a_gauge = math.sqrt(3.0) * float(self.beta) / (2.0 * float(self.a))
        velocity = float(self.vf) * 2.1354 * float(self.a)

        object.__setattr__(self, "theta_rad", theta)
        object.__setattr__(self, "phi_rad", phi)
        object.__setattr__(self, "q_vectors", q)
        object.__setattr__(self, "k_theta", k_theta)
        object.__setattr__(self, "odd_q", odd)
        object.__setattr__(self, "even_q", even)
        object.__setattr__(self, "even_q_map", even_map)
        object.__setattr__(self, "gauge_prefactor", a_gauge)
        object.__setattr__(self, "dirac_velocity", velocity)

    @staticmethod
    def _rounded_key(value: np.ndarray, digits: int = 3) -> tuple[float, float]:
        return (round(float(value[0]), digits), round(float(value[1]), digits))

    def ttg_style_kpath(self, res: int) -> np.ndarray:
        l1 = int(res)
        l2 = int(res)
        l3 = int(np.sqrt(3.0) * res / 2.0)
        l4 = int(res / 2.0)

        kprime = np.asarray([0.0, 0.0], dtype=float)
        kpoint = np.asarray(self.q_vectors[0], dtype=float)
        gamma = np.asarray(self.q_vectors[0] + self.q_vectors[1], dtype=float)
        mpoint = np.asarray(self.q_vectors[0] / 2.0, dtype=float)

        points: list[np.ndarray] = []
        for t in np.linspace(0.0, 1.0, l1):
            points.append(kprime + t * (kpoint - kprime))
        for t in np.linspace(0.0, 1.0, l2):
            points.append(kpoint + t * (gamma - kpoint))
        for t in np.linspace(0.0, 1.0, l3):
            points.append(gamma + t * (mpoint - gamma))
        for t in np.linspace(0.0, 1.0, l4):
            points.append(mpoint + t * (kprime - mpoint))
        return np.asarray(points, dtype=float)

    def energy_scale_vfkdtheta(self) -> float:
        return _pytwist_energy_scale_vfkdtheta(self.theta_deg, self.vf, self.a)

    def _diagonal_block(self, q_points: np.ndarray, sign: int, kvec: np.ndarray, valley: int) -> np.ndarray:
        m_matrix = self.module.Et(
            sign * valley * self.theta_rad / 2.0,
            self.phi_rad,
            sign * valley * float(self.epsilon) / 2.0,
            float(self.delta),
        )
        strain = (m_matrix + m_matrix.T) / 2.0
        exx = float(strain[0, 0])
        eyy = float(strain[1, 1])
        exy = float(strain[0, 1])
        gauge_shift = valley * self.gauge_prefactor * np.asarray([exx - eyy, -2.0 * exy], dtype=float)

        hamiltonian = np.zeros((2 * len(q_points), 2 * len(q_points)), dtype=np.complex128)
        for index, qvec in enumerate(q_points):
            kj = (self.module.I + m_matrix) @ (np.asarray(kvec, dtype=float) + qvec + gauge_shift)
            km = valley * kj[0] - 1j * kj[1]
            hamiltonian[2 * index, 2 * index + 1] = -self.dirac_velocity * km
        return hamiltonian + hamiltonian.conjugate().T

    def _coupling_block(self, valley: int) -> np.ndarray:
        omega = np.exp(1j * 2.0 * math.pi / 3.0)
        couplings = (
            np.asarray([[self.u, self.up], [self.up, self.u]], dtype=np.complex128),
            np.asarray([[self.u, self.up * omega ** (-valley)], [self.up * omega**valley, self.u]], dtype=np.complex128),
            np.asarray([[self.u, self.up * omega**valley], [self.up * omega ** (-valley), self.u]], dtype=np.complex128),
        )

        block = np.zeros((2 * len(self.odd_q), 2 * len(self.even_q)), dtype=np.complex128)
        for odd_index, odd_qvec in enumerate(self.odd_q):
            for channel, matrix in enumerate(couplings):
                even_index = self.even_q_map.get(self._rounded_key(odd_qvec + self.q_vectors[channel]))
                if even_index is None:
                    continue
                block[2 * odd_index : 2 * odd_index + 2, 2 * even_index : 2 * even_index + 2] = matrix
        return block

    def mapped_path_bands(self, n_layers: int, kpath: np.ndarray, *, valley: int = 1) -> MappedPathBands:
        n_layers = int(n_layers)
        n_even = n_layers // 2
        singular_values = np.asarray(
            [2.0 * math.cos(math.pi * k / (n_layers + 1)) for k in range(1, n_even + 1)],
            dtype=float,
        )
        labels = [f"$\\lambda_{index}$" for index in range(1, singular_values.size + 1)]
        if n_layers % 2 == 1:
            labels.append("Dirac")

        odd_dim = 2 * len(self.odd_q)
        even_dim = 2 * len(self.even_q)
        subspace_sizes = [odd_dim + even_dim] * singular_values.size
        if n_layers % 2 == 1:
            subspace_sizes.append(odd_dim)

        subspace_energies = [np.zeros((len(kpath), size), dtype=float) for size in subspace_sizes]
        combined = np.zeros((len(kpath), sum(subspace_sizes)), dtype=float)
        coupling = self._coupling_block(valley)

        for ik, kvec in enumerate(kpath):
            h_odd = self._diagonal_block(self.odd_q, +1, kvec, valley)
            h_even = self._diagonal_block(self.even_q, -1, kvec, valley)
            pieces: list[np.ndarray] = []
            for block_index, singular in enumerate(singular_values):
                block = np.block(
                    [
                        [h_odd, singular * coupling],
                        [singular * coupling.conjugate().T, h_even],
                    ]
                )
                evals = np.linalg.eigvalsh(block)
                subspace_energies[block_index][ik, :] = evals
                pieces.append(evals)
            if n_layers % 2 == 1:
                mono = np.linalg.eigvalsh(h_odd)
                subspace_energies[-1][ik, :] = mono
                pieces.append(mono)
            combined[ik, :] = np.sort(np.concatenate(pieces))

        return MappedPathBands(
            kpath=np.asarray(kpath, dtype=float),
            combined_energies=combined,
            subspace_energies=tuple(subspace_energies),
            labels=tuple(labels),
            singular_values=singular_values,
        )


def _alignment_metrics(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))


def _plot_bm_overlay(
    output_path: Path,
    kdist: np.ndarray,
    reference_energies: np.ndarray,
    rerun_energies: np.ndarray,
    node_positions: list[float],
    node_labels: list[str],
    summary: AlignmentSummary,
    *,
    energy_scale: float,
    ylabel: str,
    diff_unit: str,
    y_window: tuple[float, float] | None,
    title_suffix: str,
) -> None:
    ref_plot = np.asarray(reference_energies, dtype=float) / float(energy_scale)
    rerun_plot = np.asarray(rerun_energies, dtype=float) / float(energy_scale)
    max_abs_diff = float(summary.max_abs_diff) / float(energy_scale)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for xpos in node_positions[1:-1]:
        ax.axvline(xpos, color="0.85", linewidth=0.8, zorder=0)

    for band in range(ref_plot.shape[1]):
        ax.plot(kdist, ref_plot[:, band], color="black", linewidth=0.8, alpha=0.9)
        ax.plot(kdist, rerun_plot[:, band], color="tab:red", linestyle="--", linewidth=0.8, alpha=0.85)

    ax.set_title(f"n=2 BM Reference vs Rerun{title_suffix}\nmax |ΔE| = {max_abs_diff:.3e} {diff_unit}")
    ax.set_ylabel(ylabel)
    ax.set_xticks(node_positions, node_labels)
    ax.set_xlim(float(kdist[0]), float(kdist[-1]))
    if y_window is None:
        ax.set_ylim(float(np.min(ref_plot)) - 0.5, float(np.max(ref_plot)) + 0.5)
    else:
        ax.set_ylim(*y_window)
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.2, label="reference"),
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.2, label="rerun"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_ttg_overlay(
    output_path: Path,
    reference: np.ndarray,
    mapped: np.ndarray,
    kpath: np.ndarray,
    res: int,
    title: str,
    max_abs_diff: float,
    *,
    energy_scale: float,
    ylabel: str,
    diff_unit: str,
    y_window: tuple[float, float],
    title_suffix: str,
) -> None:
    tick_indices, tick_labels = _ttg_ticks(res)
    xcoords = _path_distances(kpath)
    reference_plot = np.asarray(reference, dtype=float) / float(energy_scale)
    mapped_plot = np.asarray(mapped, dtype=float) / float(energy_scale)
    band_indices = _select_window_bands(
        reference_plot,
        emin=float(y_window[0]),
        emax=float(y_window[1]),
        fallback_each_side=10,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for index in tick_indices[1:-1]:
        ax.axvline(xcoords[index], color="0.85", linewidth=0.8, zorder=0)

    for band in band_indices:
        ax.plot(xcoords, reference_plot[:, band], color="black", linewidth=0.7, alpha=0.9)
        ax.plot(xcoords, mapped_plot[:, band], color="tab:blue", linestyle="--", linewidth=0.8, alpha=0.85)

    ax.set_title(f"{title}{title_suffix}\nmax |ΔE| = {max_abs_diff / float(energy_scale):.3e} {diff_unit}")
    ax.set_ylabel(ylabel)
    ax.set_xticks([xcoords[index] for index in tick_indices], tick_labels)
    ax.set_xlim(float(xcoords[0]), float(xcoords[-1]))
    ax.set_ylim(*y_window)
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=1.2, label="reference"),
            Line2D([0], [0], color="tab:blue", linestyle="--", linewidth=1.2, label="exact common-basis map"),
        ],
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _select_window_bands(energies: np.ndarray, *, emin: float, emax: float, fallback_each_side: int) -> np.ndarray:
    band_min = np.min(energies, axis=0)
    band_max = np.max(energies, axis=0)
    indices = np.nonzero((band_max >= emin) & (band_min <= emax))[0]
    if indices.size > 0:
        return indices
    center = energies.shape[1] // 2
    lower = max(0, center - fallback_each_side)
    upper = min(energies.shape[1], center + fallback_each_side)
    return np.arange(lower, upper, dtype=int)


def _plot_family_grid(
    output_path: Path,
    cases: list[FamilyPlotCase],
    *,
    res: int,
    title_prefix: str,
    energy_unit: str,
    y_window: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(len(cases), 1, figsize=(8.2, 11.0), sharex=False)
    axes_array = np.atleast_1d(axes)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    ylabel = "Energy (eV)" if energy_unit == "native" else r"$E / (v_f k_d \theta)$"
    title_suffix = "" if energy_unit == "native" else r", unit $E/(v_f k_d \theta)$"

    for axis, case in zip(axes_array, cases, strict=True):
        n_layers = case.n_layers
        alpha_dimless = case.alpha_dimless
        bands = case.bands
        xcoords = _path_distances(bands.kpath)
        tick_indices, tick_labels = _ttg_ticks(res)
        energy_scale = 1.0 if energy_unit == "native" else float(case.energy_scale_vfkdtheta)
        for index in tick_indices[1:-1]:
            axis.axvline(xcoords[index], color="0.87", linewidth=0.8, zorder=0)

        for block_index, (label, energies) in enumerate(zip(bands.labels, bands.subspace_energies, strict=True)):
            color = colors[min(block_index, len(colors) - 1)]
            energies_plot = np.asarray(energies, dtype=float) / energy_scale
            band_indices = _select_window_bands(
                energies_plot,
                emin=float(y_window[0]),
                emax=float(y_window[1]),
                fallback_each_side=8,
            )
            for band in band_indices:
                axis.plot(xcoords, energies_plot[:, band], color=color, linewidth=0.8, alpha=0.85)

        axis.set_title(
            f"{title_prefix}: n={n_layers}, alpha={alpha_dimless:.6f}, "
            f"$\\lambda_{{max}}={bands.singular_values[0]:.6f}${title_suffix}"
        )
        axis.set_ylabel(ylabel)
        axis.set_xticks([xcoords[index] for index in tick_indices], tick_labels)
        axis.set_xlim(float(xcoords[0]), float(xcoords[-1]))
        axis.set_ylim(*y_window)
        axis.grid(True, alpha=0.25)
        axis.legend(
            handles=[
                Line2D([0], [0], color=colors[min(index, len(colors) - 1)], linewidth=1.2, label=label)
                for index, label in enumerate(bands.labels)
            ],
            frameon=False,
            loc="upper right",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _theta_from_reference(theta_ref_deg: float, n_layers: int, reference_n: int = 3) -> float:
    s_ref = 2.0 * math.cos(math.pi / (reference_n + 1))
    s_n = 2.0 * math.cos(math.pi / (n_layers + 1))
    return float(theta_ref_deg * s_n / s_ref)


def _save_npz(path: Path, payload: dict[str, np.ndarray | list[str] | list[float] | float | int]) -> None:
    np.savez(path, **payload)


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pytwist_module = _load_pytwist_module(args.pytwist_root.resolve())

    bm_kdist_ref, _, bm_energies_ref = _load_bm_reference(args.bm_reference_dir / "computed_bm_path.tsv")
    bm_kdist_rerun, bm_energies_rerun = _rerun_bm_reference(args.bm_reference_dir)
    bm_max, bm_rms = _alignment_metrics(bm_energies_ref, bm_energies_rerun)
    bm_metadata = _load_bm_metadata(args.bm_reference_dir / "run_metadata.json")
    bm_scale_vfkdtheta = _bm_energy_scale_vfkdtheta(
        float(bm_metadata["parameters"]["vf"]),
        float(bm_metadata["parameters"]["theta_deg"]),
    )
    bm_summary = AlignmentSummary(
        name="n2_bm_reference_vs_rerun",
        max_abs_diff=bm_max,
        rms_diff=bm_rms,
        unit="meV",
        artifact_pngs={},
    )
    node_positions, node_labels = _load_bm_nodes(args.bm_reference_dir / "computed_nodes.tsv")
    for energy_unit in _requested_energy_units(args.energy_unit):
        if energy_unit == "native":
            plot_path = output_dir / "n2_bm_reference_vs_rerun.png"
            _plot_bm_overlay(
                plot_path,
                bm_kdist_ref,
                bm_energies_ref,
                bm_energies_rerun,
                node_positions,
                node_labels,
                bm_summary,
                energy_scale=1.0,
                ylabel="Energy (meV)",
                diff_unit="meV",
                y_window=None,
                title_suffix="",
            )
        else:
            plot_path = output_dir / "n2_bm_reference_vs_rerun_vfkdtheta.png"
            _plot_bm_overlay(
                plot_path,
                bm_kdist_ref,
                bm_energies_ref,
                bm_energies_rerun,
                node_positions,
                node_labels,
                bm_summary,
                energy_scale=bm_scale_vfkdtheta,
                ylabel=r"$E / (v_f k_d \theta)$",
                diff_unit=r"$v_f k_d \theta$",
                y_window=(-float(args.scaled_window), float(args.scaled_window)),
                title_suffix=r", unit $E/(v_f k_d \theta)$",
            )
        bm_summary.artifact_pngs[energy_unit] = str(plot_path)

    ttg_cases = [
        ("n3_ttg_chiral", args.ttg_chiral_dir.resolve()),
        ("n3_ttg_realistic", args.ttg_realistic_dir.resolve()),
    ]
    ttg_summaries: list[AlignmentSummary] = []
    ttg_reference_payloads: dict[str, dict[str, object]] = {}

    for name, case_dir in ttg_cases:
        summary = _read_json(case_dir / "summary.json")
        band_data = np.load(case_dir / "band_data.npz")
        theta = float(summary["parameters"]["theta"])
        phi = float(summary["parameters"]["phi"])
        epsilon = float(summary["parameters"]["epsilon"])
        cut = float(summary["parameters"]["cut"])
        u = float(summary["parameters"]["u"])
        up = float(summary["parameters"]["up"])
        res = int(summary["resolution"])
        kpath = np.asarray(band_data["kpath"], dtype=float)
        reference_p = np.asarray(band_data["evals_p"], dtype=float)
        reference_m = np.asarray(band_data["evals_m"], dtype=float)

        family = ExactCommonBasisFamily(
            module=pytwist_module,
            theta_deg=theta,
            phi_deg=phi,
            epsilon=epsilon,
            vf=1.0,
            u=u,
            up=up,
            cut=cut,
        )
        mapped_p = family.mapped_path_bands(3, kpath, valley=1)
        mapped_m = family.mapped_path_bands(3, kpath, valley=-1)
        max_p, rms_p = _alignment_metrics(reference_p, mapped_p.combined_energies)
        max_m, rms_m = _alignment_metrics(reference_m, mapped_m.combined_energies)
        max_abs = max(max_p, max_m)
        rms_abs = max(rms_p, rms_m)
        energy_scale_vfkdtheta = family.energy_scale_vfkdtheta()
        artifact_pngs: dict[str, str] = {}
        for energy_unit in _requested_energy_units(args.energy_unit):
            if energy_unit == "native":
                plot_path = output_dir / f"{name}_reference_vs_exact_map.png"
                _plot_ttg_overlay(
                    plot_path,
                    reference_p,
                    mapped_p.combined_energies,
                    kpath,
                    res,
                    f"{name.replace('_', ' ').upper()} reference vs exact common-basis map",
                    max_abs,
                    energy_scale=1.0,
                    ylabel="Energy (eV)",
                    diff_unit="eV",
                    y_window=(-float(args.native_window), float(args.native_window)),
                    title_suffix="",
                )
            else:
                plot_path = output_dir / f"{name}_reference_vs_exact_map_vfkdtheta.png"
                _plot_ttg_overlay(
                    plot_path,
                    reference_p,
                    mapped_p.combined_energies,
                    kpath,
                    res,
                    f"{name.replace('_', ' ').upper()} reference vs exact common-basis map",
                    max_abs,
                    energy_scale=energy_scale_vfkdtheta,
                    ylabel=r"$E / (v_f k_d \theta)$",
                    diff_unit=r"$v_f k_d \theta$",
                    y_window=(-float(args.scaled_window), float(args.scaled_window)),
                    title_suffix=r", unit $E/(v_f k_d \theta)$",
                )
            artifact_pngs[energy_unit] = str(plot_path)
        ttg_summaries.append(
            AlignmentSummary(
                name=name,
                max_abs_diff=max_abs,
                rms_diff=rms_abs,
                unit="eV",
                artifact_pngs=artifact_pngs,
            )
        )
        ttg_reference_payloads[name] = {
            "theta_deg": theta,
            "phi_deg": phi,
            "epsilon": epsilon,
            "cut": cut,
            "u": u,
            "up": up,
            "res": res,
            "vf": 1.0,
            "energy_scale_vfkdtheta": energy_scale_vfkdtheta,
        }

    family_plots: dict[str, dict[str, str]] = {}
    family_theta_summary: dict[str, dict[str, float]] = {}
    family_scale_summary: dict[str, dict[str, float]] = {}
    family_alpha_summary: dict[str, dict[str, float]] = {}
    for family_name, case_dir in (("chiral", args.ttg_chiral_dir.resolve()), ("realistic", args.ttg_realistic_dir.resolve())):
        summary = _read_json(case_dir / "summary.json")
        params = summary["parameters"]
        theta_ref = float(params["theta"])
        phi = float(params["phi"])
        epsilon = float(params["epsilon"])
        cut = float(params["cut"])
        u = float(params["u"])
        up = float(params["up"])
        res = int(summary["resolution"])

        cases: list[FamilyPlotCase] = []
        theta_map: dict[str, float] = {}
        scale_map: dict[str, float] = {}
        alpha_map: dict[str, float] = {}
        for n_layers in (4, 5, 6):
            theta_n = _theta_from_reference(theta_ref, n_layers, reference_n=3)
            family = ExactCommonBasisFamily(
                module=pytwist_module,
                theta_deg=theta_n,
                phi_deg=phi,
                epsilon=epsilon,
                vf=1.0,
                u=u,
                up=up,
                cut=cut,
            )
            kpath = family.ttg_style_kpath(res)
            bands = family.mapped_path_bands(n_layers, kpath, valley=1)
            alpha_n = float(up) / float(family.energy_scale_vfkdtheta())
            cases.append(
                FamilyPlotCase(
                    n_layers=n_layers,
                    theta_deg=theta_n,
                    bands=bands,
                    energy_scale_vfkdtheta=family.energy_scale_vfkdtheta(),
                    alpha_dimless=alpha_n,
                )
            )
            theta_map[f"n{n_layers}"] = theta_n
            scale_map[f"n{n_layers}"] = family.energy_scale_vfkdtheta()
            alpha_map[f"n{n_layers}"] = alpha_n
            _save_npz(
                output_dir / f"atmg_{family_name}_n{n_layers}_bands.npz",
                {
                    "kpath": kpath,
                    "combined_energies": bands.combined_energies,
                    "singular_values": bands.singular_values,
                    "theta_deg": np.asarray(theta_n, dtype=float),
                    "alpha_dimless": np.asarray(alpha_n, dtype=float),
                    "u": np.asarray(u, dtype=float),
                    "up": np.asarray(up, dtype=float),
                    "energy_scale_vfkdtheta": np.asarray(family.energy_scale_vfkdtheta(), dtype=float),
                    "labels": np.asarray(bands.labels, dtype=object),
                    **{f"subspace_{index + 1}": energy for index, energy in enumerate(bands.subspace_energies)},
                },
            )
        plot_variants: dict[str, str] = {}
        for energy_unit in _requested_energy_units(args.energy_unit):
            if energy_unit == "native":
                plot_path = output_dir / f"atmg_{family_name}_n4_n5_n6_bands.png"
                _plot_family_grid(
                    plot_path,
                    cases,
                    res=res,
                    title_prefix=f"ATMG {family_name}",
                    energy_unit="native",
                    y_window=(-float(args.native_window), float(args.native_window)),
                )
            else:
                plot_path = output_dir / f"atmg_{family_name}_n4_n5_n6_bands_vfkdtheta.png"
                _plot_family_grid(
                    plot_path,
                    cases,
                    res=res,
                    title_prefix=f"ATMG {family_name}",
                    energy_unit="vfkdtheta",
                    y_window=(-float(args.scaled_window), float(args.scaled_window)),
                )
            plot_variants[energy_unit] = str(plot_path)
        family_plots[family_name] = plot_variants
        family_theta_summary[family_name] = theta_map
        family_scale_summary[family_name] = scale_map
        family_alpha_summary[family_name] = alpha_map

    summary_payload = {
        "generated_at": current_timestamp(),
        "output_dir": str(output_dir),
        "runtime_environment": asdict(collect_runtime_environment()),
        "bm_reference_dir": str(args.bm_reference_dir.resolve()),
        "ttg_chiral_dir": str(args.ttg_chiral_dir.resolve()),
        "ttg_realistic_dir": str(args.ttg_realistic_dir.resolve()),
        "alignment": {
            bm_summary.name: {
                "max_abs_diff": bm_summary.max_abs_diff,
                "rms_diff": bm_summary.rms_diff,
                "unit": bm_summary.unit,
                "artifact_pngs": bm_summary.artifact_pngs,
                "energy_scale_vfkdtheta": bm_scale_vfkdtheta,
            },
            **{
                item.name: {
                    "max_abs_diff": item.max_abs_diff,
                    "rms_diff": item.rms_diff,
                    "unit": item.unit,
                    "artifact_pngs": item.artifact_pngs,
                }
                for item in ttg_summaries
            },
        },
        "reference_cases": ttg_reference_payloads,
        "family_plots": family_plots,
        "family_theta_deg": family_theta_summary,
        "family_energy_scale_vfkdtheta": family_scale_summary,
        "family_alpha_dimless": family_alpha_summary,
        "plot_settings": {
            "energy_unit": args.energy_unit,
            "native_window": args.native_window,
            "scaled_window": args.scaled_window,
        },
    }
    _write_summary(output_dir / "alignment_summary.json", summary_payload)

    print(f"output_dir={output_dir}")
    for energy_unit, plot_path in bm_summary.artifact_pngs.items():
        print(f"n2_bm_plot_{energy_unit}={plot_path}")
    for item in ttg_summaries:
        for energy_unit, plot_path in item.artifact_pngs.items():
            print(f"{item.name}_plot_{energy_unit}={plot_path}")
    for family_name, plot_variants in family_plots.items():
        for energy_unit, plot_path in plot_variants.items():
            print(f"{family_name}_n456_plot_{energy_unit}={plot_path}")
    print(f"summary_json={output_dir / 'alignment_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
