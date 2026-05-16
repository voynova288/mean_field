from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from mean_field.core.lattice import KPath
from mean_field.devtools._runtime import ensure_not_running_compute_on_login_node, write_json
from mean_field.runtime import collect_runtime_environment
from mean_field.systems.tdbg import (
    PathBandsResult,
    TDBGModel,
    TDBGParameters,
    compute_topology_from_grid_result,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "TDBG"
VALID_STACKINGS = ("AB-AB", "AB-BA")
DEFAULT_DELTAS_EV = (0.0, 0.005, 0.020)


@dataclass(frozen=True)
class Fig3PanelSpec:
    stacking: str
    delta_ev: float

    @property
    def key(self) -> str:
        return f"{self.stacking.lower().replace('-', '')}_delta_{int(round(self.delta_ev * 1000.0)):03d}mev"

    @property
    def title(self) -> str:
        return f"{self.stacking}, Delta = {self.delta_ev * 1000.0:.0f} meV"


@dataclass(frozen=True)
class PanelBandData:
    spec: Fig3PanelSpec
    full_plus: PathBandsResult
    full_minus: PathBandsResult
    minimal_plus: PathBandsResult
    minimal_minus: PathBandsResult
    matrix_dim: int
    valence_index: int
    conduction_index: int


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    return values


def _parse_csv_stackings(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated stacking name.")
    invalid = [item for item in values if item not in VALID_STACKINGS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported stacking(s): {invalid}; expected values in {VALID_STACKINGS}.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce Koshino 2019 Fig. 3 TDBG full-parameter band panels with "
            "K-valley Chern and integrated Chern annotations."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--theta-deg", type=float, default=1.33)
    parser.add_argument("--phi-deg", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--cut", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--topology-mesh", type=int, default=13)
    parser.add_argument(
        "--topology-boundary",
        choices=("open", "periodic"),
        default="open",
        help=(
            "Use open plaquette integration for the finite-cutoff Q lattice, or the periodic wrapped FHS grid. "
            "The open mode avoids artificial boundary sewing from the truncated basis."
        ),
    )
    parser.add_argument("--deltas-ev", type=_parse_csv_floats, default=DEFAULT_DELTAS_EV)
    parser.add_argument("--stackings", type=_parse_csv_stackings, default=VALID_STACKINGS)
    parser.add_argument("--window-ev", type=float, default=0.050)
    parser.add_argument(
        "--central-touching-threshold-ev",
        type=float,
        default=1.0e-4,
        help="Central direct gap below which individual central-band Chern numbers are treated as unresolved.",
    )
    parser.add_argument(
        "--display-chern-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Multiply raw Chern numbers by this sign for plot labels when matching a paper convention.",
    )
    parser.add_argument(
        "--skip-topology",
        action="store_true",
        help="Only generate band panels and path data; skip Fukui-Hatsugai Chern calculations.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse bands_path.npz and chern_numbers.json files under --output-dir and only redraw reports/figures.",
    )
    return parser.parse_args()


def _default_output_dir() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        stem = f"tdbg_fig3_chern_{job_id}"
    else:
        stem = f"tdbg_fig3_chern_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return DEFAULT_OUTPUT_ROOT / stem


def _format_delta_dir(delta_ev: float) -> str:
    return f"delta_{int(round(delta_ev * 1000.0)):+04d}mev"


def _fig3_segment_point_counts(resolution: int) -> tuple[int, int, int]:
    return (int(resolution), int(np.sqrt(3.0) * int(resolution) / 2.0), int(int(resolution) / 2.0))


def _build_fig3_valley_path(model: TDBGModel, *, valley: int, resolution: int) -> KPath:
    """Build the valley-specific path used for the paper-style Fig. 3 overlay.

    In the pytwist coordinate convention the K-valley path is K-Gamma-M-Kprime.
    For the opposite valley, the same symbolic labels correspond to the opposite
    valley corners, so the physical endpoints are Kprime-Gamma-M-K while the
    plot keeps the same path-distance axis.
    """

    if int(valley) == 1:
        return model.standard_kpath(resolution=resolution)
    if int(valley) != -1:
        raise ValueError(f"Expected valley +/-1, got {valley}")
    lattice = model.lattice
    return model.build_kpath(
        (lattice.kprime_m, lattice.gamma_m, lattice.m_m, lattice.k_m),
        ("K", "Gamma", "M", "Kprime"),
        segment_point_counts=_fig3_segment_point_counts(resolution),
        duplicate_nodes=True,
    )


def _display_node_label(label: str) -> str:
    return {"Gamma": r"$\Gamma$", "Kprime": "K'", "KPrime": "K'"}.get(label, label)


def _window_band_indices(*energy_sets: np.ndarray, emin: float, emax: float, fallback_each_side: int = 8) -> np.ndarray:
    if not energy_sets:
        raise ValueError("Expected at least one energy set.")
    n_bands = min(np.asarray(energies).shape[1] for energies in energy_sets)
    keep: set[int] = set()
    for energies in energy_sets:
        values = np.asarray(energies, dtype=float)[:, :n_bands]
        band_min = np.min(values, axis=0)
        band_max = np.max(values, axis=0)
        keep.update(int(index) for index in np.nonzero((band_max >= emin) & (band_min <= emax))[0])
    if keep:
        return np.asarray(sorted(keep), dtype=int)
    center = n_bands // 2
    lower = max(0, center - fallback_each_side)
    upper = min(n_bands, center + fallback_each_side)
    return np.arange(lower, upper, dtype=int)


def _chern_label(value: object, *, sign: int) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(round(sign * float(value))))
    except (TypeError, ValueError):
        return "n/a"


def _rounded_chern(result: object | None) -> int | None:
    if result is None:
        return None
    if not isinstance(result, dict):
        return None
    rounded = result.get("rounded_chern_number")
    if rounded is None:
        return None
    return int(rounded)


def _topology_payload(result) -> dict[str, object]:
    return {
        "band_indices": list(int(index) for index in result.band_indices),
        "valley": int(result.valley),
        "chern_number": float(result.chern_number),
        "rounded_chern_number": int(result.rounded_chern_number),
        "integer_residual": float(result.integer_residual),
        "is_nearly_integer": bool(result.is_nearly_integer),
    }


def _normalize_band_indices(band_indices: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(band_indices, (int, np.integer)):
        normalized = (int(band_indices),)
    else:
        normalized = tuple(int(index) for index in band_indices)
    if not normalized:
        raise ValueError("Expected at least one target band index.")
    if min(normalized) < 0:
        raise ValueError(f"Band indices must be non-negative, got {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Band indices must be unique, got {normalized}")
    return normalized


def _unit_link(overlap: complex, *, atol: float = 1.0e-14) -> complex:
    magnitude = abs(overlap)
    if magnitude <= atol:
        raise ValueError(
            "Encountered a near-zero overlap link while building a plaquette field. "
            "The selected band or subspace is likely not isolated on this grid."
        )
    return overlap / magnitude


def _subspace_link(
    left_vectors: np.ndarray,
    right_vectors: np.ndarray,
    *,
    atol: float = 1.0e-14,
    regularization: float = 1.0e-12,
) -> complex:
    overlap_matrix = left_vectors.conjugate().T @ right_vectors
    if overlap_matrix.shape == (1, 1):
        return _unit_link(complex(overlap_matrix[0, 0]), atol=atol)

    singular_values = np.linalg.svd(overlap_matrix, compute_uv=False)
    if np.min(singular_values) <= atol:
        overlap_matrix = overlap_matrix + regularization * np.eye(overlap_matrix.shape[0], dtype=np.complex128)
    u_mat, _, vh_mat = np.linalg.svd(overlap_matrix, full_matrices=False)
    phase_matrix = u_mat @ vh_mat
    return _unit_link(complex(np.linalg.det(phase_matrix)), atol=atol)


def _compute_open_topology_payload(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int,
) -> dict[str, object]:
    if grid_result.eigenvectors is None:
        raise ValueError("Grid eigenvectors are required for topology.")
    eigenvectors = np.asarray(grid_result.eigenvectors, dtype=np.complex128)
    normalized = _normalize_band_indices(band_indices)
    if max(normalized) >= eigenvectors.shape[-1]:
        raise ValueError(f"Band index {max(normalized)} exceeds available eigenvector count {eigenvectors.shape[-1]}")

    mesh_x, mesh_y = eigenvectors.shape[:2]
    if mesh_x < 2 or mesh_y < 2:
        raise ValueError(f"Open plaquette topology needs at least a 2x2 grid, got {mesh_x}x{mesh_y}.")

    selected = np.take(eigenvectors, normalized, axis=-1)
    berry_curvature = np.zeros((mesh_x - 1, mesh_y - 1), dtype=float)
    for ix in range(mesh_x - 1):
        ix_next = ix + 1
        for iy in range(mesh_y - 1):
            iy_next = iy + 1
            ux = _subspace_link(selected[ix, iy], selected[ix_next, iy])
            uy_next_x = _subspace_link(selected[ix_next, iy], selected[ix_next, iy_next])
            ux_next_y = _subspace_link(selected[ix, iy_next], selected[ix_next, iy_next])
            uy = _subspace_link(selected[ix, iy], selected[ix, iy_next])
            plaquette = ux * uy_next_x / (ux_next_y * uy)
            berry_curvature[ix, iy] = float(np.angle(plaquette))

    chern_number = float(np.sum(berry_curvature) / (2.0 * np.pi))
    rounded_chern_number = int(np.rint(chern_number))
    integer_residual = float(abs(chern_number - rounded_chern_number))
    return {
        "band_indices": list(int(index) for index in normalized),
        "valley": int(valley),
        "chern_number": chern_number,
        "rounded_chern_number": rounded_chern_number,
        "integer_residual": integer_residual,
        "is_nearly_integer": bool(integer_residual < 5.0e-2),
        "plaquette_shape": [int(mesh_x - 1), int(mesh_y - 1)],
    }


def _try_topology(
    grid_result,
    band_indices: int | Iterable[int],
    *,
    valley: int,
    topology_boundary: str,
) -> dict[str, object]:
    try:
        if topology_boundary == "periodic":
            result = _topology_payload(compute_topology_from_grid_result(grid_result, band_indices, valley=valley))
        else:
            result = _compute_open_topology_payload(grid_result, band_indices, valley=valley)
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}
    return {"status": "ok", "topology_boundary": topology_boundary, **result}


def _derived_topology_difference(
    upper_payload: object,
    lower_payload: object,
    *,
    band_indices: tuple[int, ...],
    valley: int,
    note: str,
) -> dict[str, object]:
    if not isinstance(upper_payload, dict) or not isinstance(lower_payload, dict):
        return {"status": "failed", "error": "Cannot derive Chern difference from non-dict payloads."}
    if upper_payload.get("status") != "ok" or lower_payload.get("status") != "ok":
        return {
            "status": "failed",
            "error": "Cannot derive Chern difference because at least one cumulative topology failed.",
            "upper_status": upper_payload.get("status"),
            "lower_status": lower_payload.get("status"),
        }

    chern_number = float(upper_payload["chern_number"]) - float(lower_payload["chern_number"])
    rounded_chern_number = int(upper_payload["rounded_chern_number"]) - int(lower_payload["rounded_chern_number"])
    integer_residual = float(abs(chern_number - rounded_chern_number))
    return {
        "status": "derived",
        "band_indices": list(int(index) for index in band_indices),
        "valley": int(valley),
        "chern_number": chern_number,
        "rounded_chern_number": rounded_chern_number,
        "integer_residual": integer_residual,
        "is_nearly_integer": bool(integer_residual < 1.0e-6),
        "note": note,
    }


def _derived_topology_sum(
    payloads: tuple[object, ...],
    *,
    band_indices: tuple[int, ...],
    valley: int,
    note: str,
) -> dict[str, object]:
    normalized_payloads = []
    for payload in payloads:
        if not isinstance(payload, dict):
            return {"status": "failed", "error": "Cannot derive Chern sum from a non-dict payload."}
        if payload.get("status") not in {"ok", "derived"}:
            return {
                "status": "failed",
                "error": "Cannot derive Chern sum because at least one source topology failed.",
                "source_statuses": [item.get("status") for item in payloads if isinstance(item, dict)],
            }
        normalized_payloads.append(payload)

    chern_number = float(sum(float(payload["chern_number"]) for payload in normalized_payloads))
    rounded_chern_number = int(sum(int(payload["rounded_chern_number"]) for payload in normalized_payloads))
    integer_residual = float(abs(chern_number - rounded_chern_number))
    return {
        "status": "derived",
        "band_indices": list(int(index) for index in band_indices),
        "valley": int(valley),
        "chern_number": chern_number,
        "rounded_chern_number": rounded_chern_number,
        "integer_residual": integer_residual,
        "is_nearly_integer": bool(integer_residual < 5.0e-2),
        "note": note,
    }


def _build_panel_data(
    spec: Fig3PanelSpec,
    *,
    theta_deg: float,
    phi_deg: float,
    epsilon: float,
    cut: float,
    resolution: int,
) -> PanelBandData:
    full_params = TDBGParameters.full(
        stacking=spec.stacking,
        Delta=spec.delta_ev,
        phi_deg=phi_deg,
        epsilon=epsilon,
    )
    minimal_params = TDBGParameters.minimal(
        stacking=spec.stacking,
        Delta=spec.delta_ev,
        phi_deg=phi_deg,
        epsilon=epsilon,
    )
    full_model = TDBGModel.from_config(theta_deg, cut=cut, params=full_params)
    minimal_model = TDBGModel.from_config(theta_deg, cut=cut, params=minimal_params)
    path_plus = _build_fig3_valley_path(full_model, valley=1, resolution=resolution)
    path_minus = _build_fig3_valley_path(full_model, valley=-1, resolution=resolution)

    full_plus = full_model.bands_along_path(path_plus, valley=1, n_bands=full_model.matrix_dim)
    full_minus = full_model.bands_along_path(path_minus, valley=-1, n_bands=full_model.matrix_dim)
    minimal_plus = minimal_model.bands_along_path(path_plus, valley=1, n_bands=minimal_model.matrix_dim)
    minimal_minus = minimal_model.bands_along_path(path_minus, valley=-1, n_bands=minimal_model.matrix_dim)

    valence_index = full_model.matrix_dim // 2 - 1
    conduction_index = full_model.matrix_dim // 2
    return PanelBandData(
        spec=spec,
        full_plus=full_plus,
        full_minus=full_minus,
        minimal_plus=minimal_plus,
        minimal_minus=minimal_minus,
        matrix_dim=full_model.matrix_dim,
        valence_index=valence_index,
        conduction_index=conduction_index,
    )


def _compute_chern_data(
    spec: Fig3PanelSpec,
    *,
    theta_deg: float,
    phi_deg: float,
    epsilon: float,
    cut: float,
    topology_mesh: int,
    topology_boundary: str,
    central_touching_threshold_ev: float,
) -> dict[str, object]:
    params = TDBGParameters.full(
        stacking=spec.stacking,
        Delta=spec.delta_ev,
        phi_deg=phi_deg,
        epsilon=epsilon,
    )
    model = TDBGModel.from_config(theta_deg, cut=cut, params=params)
    valence = model.matrix_dim // 2 - 1
    conduction = model.matrix_dim // 2
    first_upper = conduction + 1
    n_bands = first_upper + 1

    grid_mesh_size = int(topology_mesh) + 1 if topology_boundary == "open" else int(topology_mesh)
    print(
        f"[topology] {spec.stacking} Delta={spec.delta_ev * 1000.0:.0f} meV: "
        f"mesh={topology_mesh}, boundary={topology_boundary}, grid_points={grid_mesh_size}, "
        f"n_bands={n_bands}, matrix_dim={model.matrix_dim}",
        flush=True,
    )
    grid_result = model.bands_on_grid(
        grid_mesh_size,
        valley=1,
        n_bands=n_bands,
        return_eigenvectors=True,
        endpoint=(topology_boundary == "open"),
        frac_shift=(0.0, 0.0),
    )
    energies = np.asarray(grid_result.energies, dtype=float)
    gap_metrics = {
        "lower_direct_gap_ev": float(np.min(energies[:, :, valence] - energies[:, :, valence - 1])),
        "central_direct_gap_ev": float(np.min(energies[:, :, conduction] - energies[:, :, valence])),
        "upper_direct_gap_ev": float(np.min(energies[:, :, first_upper] - energies[:, :, conduction])),
    }

    symmetry_protected_touching = spec.stacking == "AB-AB" and abs(spec.delta_ev) < central_touching_threshold_ev
    central_touches = bool(
        symmetry_protected_touching
        or abs(gap_metrics["central_direct_gap_ev"]) < central_touching_threshold_ev
    )
    topology: dict[str, object] = {
        "lower_gap_occupied": _try_topology(
            grid_result,
            range(0, valence),
            valley=1,
            topology_boundary=topology_boundary,
        )
    }

    if central_touches:
        topology["upper_gap_occupied"] = _try_topology(
            grid_result,
            range(0, conduction + 1),
            valley=1,
            topology_boundary=topology_boundary,
        )
        topology["central_pair"] = _derived_topology_difference(
            topology["upper_gap_occupied"],
            topology["lower_gap_occupied"],
            band_indices=(valence, conduction),
            valley=1,
            note=(
                "Central bands are not individually isolated; total Chern is derived from "
                "the difference between upper- and lower-gap occupied-subspace Chern numbers."
            ),
        )
        topology["central_gap_occupied"] = {
            "status": "skipped",
            "reason": "central_gap_not_isolated",
        }
        topology["valence_band"] = {
            "status": "skipped",
            "reason": "central_gap_not_isolated",
        }
        topology["conduction_band"] = {
            "status": "skipped",
            "reason": "central_gap_not_isolated",
        }
    else:
        topology["valence_band"] = _try_topology(
            grid_result,
            valence,
            valley=1,
            topology_boundary=topology_boundary,
        )
        topology["conduction_band"] = _try_topology(
            grid_result,
            conduction,
            valley=1,
            topology_boundary=topology_boundary,
        )
        topology["central_gap_occupied"] = _derived_topology_sum(
            (topology["lower_gap_occupied"], topology["valence_band"]),
            band_indices=tuple(range(0, valence + 1)),
            valley=1,
            note="Derived from lower-gap occupied-subspace Chern plus the isolated valence-band Chern.",
        )
        topology["upper_gap_occupied"] = _derived_topology_sum(
            (topology["central_gap_occupied"], topology["conduction_band"]),
            band_indices=tuple(range(0, conduction + 1)),
            valley=1,
            note="Derived from central-gap occupied-subspace Chern plus the isolated conduction-band Chern.",
        )
        topology["central_pair"] = _derived_topology_sum(
            (topology["valence_band"], topology["conduction_band"]),
            band_indices=(valence, conduction),
            valley=1,
            note="Derived from isolated central valence and conduction Chern numbers.",
        )

    return {
        "stacking": spec.stacking,
        "delta_ev": float(spec.delta_ev),
        "topology_mesh": int(topology_mesh),
        "topology_boundary": topology_boundary,
        "valley": 1,
        "matrix_dim": int(model.matrix_dim),
        "band_indices": {
            "valence": int(valence),
            "conduction": int(conduction),
            "first_upper": int(first_upper),
        },
        "gap_metrics": gap_metrics,
        "central_touches": central_touches,
        "topology": topology,
    }


def _save_panel_npz(output_path: Path, panel: PanelBandData) -> None:
    path = panel.full_plus.path
    np.savez_compressed(
        output_path,
        kdist=np.asarray(path.kdist, dtype=float),
        kpath=np.stack([path.kvec.real, path.kvec.imag], axis=-1),
        full_valley_minus_kpath=np.stack(
            [panel.full_minus.path.kvec.real, panel.full_minus.path.kvec.imag],
            axis=-1,
        ),
        minimal_valley_minus_kpath=np.stack(
            [panel.minimal_minus.path.kvec.real, panel.minimal_minus.path.kvec.imag],
            axis=-1,
        ),
        node_indices=np.asarray(path.node_indices, dtype=int),
        labels=np.asarray(path.labels, dtype=object),
        full_valley_plus_ev=np.asarray(panel.full_plus.energies, dtype=float),
        full_valley_minus_ev=np.asarray(panel.full_minus.energies, dtype=float),
        minimal_valley_plus_ev=np.asarray(panel.minimal_plus.energies, dtype=float),
        minimal_valley_minus_ev=np.asarray(panel.minimal_minus.energies, dtype=float),
        stacking=np.asarray(panel.spec.stacking, dtype=object),
        delta_ev=np.asarray(panel.spec.delta_ev, dtype=float),
        matrix_dim=np.asarray(panel.matrix_dim, dtype=int),
        valence_index=np.asarray(panel.valence_index, dtype=int),
        conduction_index=np.asarray(panel.conduction_index, dtype=int),
    )


def _load_panel_npz(input_path: Path, spec: Fig3PanelSpec) -> PanelBandData:
    payload = np.load(input_path, allow_pickle=True)
    kxy = np.asarray(payload["kpath"], dtype=float)
    kvec = np.asarray(kxy[:, 0] + 1j * kxy[:, 1], dtype=np.complex128)
    labels = tuple(str(item) for item in np.asarray(payload["labels"], dtype=object).tolist())
    node_indices = tuple(int(item) for item in np.asarray(payload["node_indices"], dtype=int).tolist())
    path = KPath(
        kvec=kvec,
        kdist=np.asarray(payload["kdist"], dtype=float),
        labels=labels,
        node_indices=node_indices,
    )
    full_minus_path = path
    if "full_valley_minus_kpath" in payload:
        full_minus_kxy = np.asarray(payload["full_valley_minus_kpath"], dtype=float)
        full_minus_path = KPath(
            kvec=np.asarray(full_minus_kxy[:, 0] + 1j * full_minus_kxy[:, 1], dtype=np.complex128),
            kdist=np.asarray(payload["kdist"], dtype=float),
            labels=labels,
            node_indices=node_indices,
        )
    minimal_minus_path = full_minus_path
    if "minimal_valley_minus_kpath" in payload:
        minimal_minus_kxy = np.asarray(payload["minimal_valley_minus_kpath"], dtype=float)
        minimal_minus_path = KPath(
            kvec=np.asarray(minimal_minus_kxy[:, 0] + 1j * minimal_minus_kxy[:, 1], dtype=np.complex128),
            kdist=np.asarray(payload["kdist"], dtype=float),
            labels=labels,
            node_indices=node_indices,
        )
    return PanelBandData(
        spec=spec,
        full_plus=PathBandsResult(path=path, energies=np.asarray(payload["full_valley_plus_ev"], dtype=float)),
        full_minus=PathBandsResult(path=full_minus_path, energies=np.asarray(payload["full_valley_minus_ev"], dtype=float)),
        minimal_plus=PathBandsResult(path=path, energies=np.asarray(payload["minimal_valley_plus_ev"], dtype=float)),
        minimal_minus=PathBandsResult(
            path=minimal_minus_path,
            energies=np.asarray(payload["minimal_valley_minus_ev"], dtype=float),
        ),
        matrix_dim=int(np.asarray(payload["matrix_dim"]).item()),
        valence_index=int(np.asarray(payload["valence_index"]).item()),
        conduction_index=int(np.asarray(payload["conduction_index"]).item()),
    )


def _plot_fig3_grid(
    output_dir: Path,
    panels: list[PanelBandData],
    chern_by_key: dict[str, dict[str, object]],
    *,
    theta_deg: float,
    window_ev: float,
    display_chern_sign: int,
) -> dict[str, str]:
    if len(panels) == 0:
        raise ValueError("Expected at least one panel to plot.")
    stackings = list(dict.fromkeys(panel.spec.stacking for panel in panels))
    deltas = list(dict.fromkeys(float(panel.spec.delta_ev) for panel in panels))
    panel_lookup = {(panel.spec.stacking, float(panel.spec.delta_ev)): panel for panel in panels}

    fig, axes = plt.subplots(
        len(stackings),
        len(deltas),
        figsize=(3.35 * len(deltas) + 1.0, 3.2 * len(stackings) + 0.6),
        sharex=False,
        sharey=True,
    )
    axes_array = np.atleast_2d(axes)
    ymin = -abs(float(window_ev)) * 1000.0
    ymax = abs(float(window_ev)) * 1000.0

    for row, stacking in enumerate(stackings):
        for col, delta_ev in enumerate(deltas):
            panel = panel_lookup[(stacking, delta_ev)]
            axis = axes_array[row, col]
            path = panel.full_plus.path
            node_x = [float(node.k_dist) for node in path.nodes]
            node_labels = [_display_node_label(node.label) for node in path.nodes]
            for xpos in node_x:
                axis.axvline(xpos, color="0.55", linewidth=0.55, zorder=0)

            band_indices = _window_band_indices(
                panel.full_plus.energies,
                panel.full_minus.energies,
                panel.minimal_plus.energies,
                panel.minimal_minus.energies,
                emin=-abs(float(window_ev)),
                emax=abs(float(window_ev)),
            )
            for band_index in band_indices:
                axis.plot(
                    path.kdist,
                    panel.minimal_plus.energies[:, band_index] * 1000.0,
                    color="#64d66c",
                    linewidth=0.45,
                    alpha=0.70,
                    zorder=1,
                )
                axis.plot(
                    path.kdist,
                    panel.minimal_minus.energies[:, band_index] * 1000.0,
                    color="#64d66c",
                    linewidth=0.45,
                    linestyle="--",
                    alpha=0.65,
                    zorder=1,
                )
                axis.plot(
                    path.kdist,
                    panel.full_plus.energies[:, band_index] * 1000.0,
                    color="black",
                    linewidth=0.78,
                    alpha=0.95,
                    zorder=2,
                )
                axis.plot(
                    path.kdist,
                    panel.full_minus.energies[:, band_index] * 1000.0,
                    color="red",
                    linestyle=(0, (3, 2)),
                    linewidth=0.72,
                    alpha=0.88,
                    zorder=2,
                )

            chern = chern_by_key.get(panel.spec.key, {})
            topology = chern.get("topology", {}) if isinstance(chern, dict) else {}
            central_touches = bool(chern.get("central_touches", False)) if isinstance(chern, dict) else False
            if topology:
                lower_gap = _rounded_chern(topology.get("lower_gap_occupied"))
                central_gap = _rounded_chern(topology.get("central_gap_occupied"))
                upper_gap = _rounded_chern(topology.get("upper_gap_occupied"))
                conduction = _rounded_chern(topology.get("conduction_band"))
                valence = _rounded_chern(topology.get("valence_band"))
                pair = _rounded_chern(topology.get("central_pair"))

                x0 = float(path.kdist[0])
                x1 = float(path.kdist[-1])
                xr = x1 - x0
                if central_touches:
                    axis.text(
                        x0 + 0.10 * xr,
                        23.0,
                        f"total: {_chern_label(pair, sign=display_chern_sign)}",
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )
                else:
                    axis.text(
                        x0 + 0.18 * xr,
                        23.0,
                        _chern_label(conduction, sign=display_chern_sign),
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )
                    axis.text(
                        x0 + 0.18 * xr,
                        -3.0,
                        _chern_label(valence, sign=display_chern_sign),
                        color="black",
                        fontsize=9,
                        fontweight="bold",
                    )

                if upper_gap is not None:
                    axis.text(
                        x0 + 0.63 * xr,
                        27.0,
                        _chern_label(upper_gap, sign=display_chern_sign),
                        color="#00a7ff",
                        fontsize=9,
                        fontweight="bold",
                    )
                if central_gap is not None:
                    axis.text(
                        x0 + 0.61 * xr,
                        4.0,
                        _chern_label(central_gap, sign=display_chern_sign),
                        color="#00a7ff",
                        fontsize=9,
                        fontweight="bold",
                    )
                if lower_gap is not None:
                    axis.text(
                        x0 + 0.63 * xr,
                        -12.0,
                        _chern_label(lower_gap, sign=display_chern_sign),
                        color="#00a7ff",
                        fontsize=9,
                        fontweight="bold",
                    )

            axis.set_title(rf"$\Delta={delta_ev * 1000.0:.0f}$ meV", fontsize=10)
            axis.set_xticks(node_x, node_labels)
            axis.set_xlim(float(node_x[0]), float(node_x[-1]))
            axis.set_ylim(ymin, ymax)
            axis.grid(False)
            if col == 0:
                axis.set_ylabel(f"{stacking}\nEnergy (meV)")
            else:
                axis.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.0, label=r"$\xi=+$ full"),
        Line2D([0], [0], color="red", linestyle=(0, (3, 2)), linewidth=1.0, label=r"$\xi=-$ full"),
        Line2D([0], [0], color="#64d66c", linewidth=0.8, label="minimal model"),
        Line2D([0], [0], color="#00a7ff", linewidth=0, marker="", label="blue: integrated Chern"),
    ]
    fig.suptitle(
        rf"TDBG Fig. 3 reproduction, $\theta={theta_deg:.2f}^\circ$ [full parameter model]",
        fontsize=12,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.055, 1.0, 0.94))

    png_path = output_dir / "tdbg_fig3_chern_bands.png"
    pdf_path = output_dir / "tdbg_fig3_chern_bands.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"fig3_chern_bands_png": str(png_path), "fig3_chern_bands_pdf": str(pdf_path)}


def _write_report(
    path: Path,
    *,
    summary: dict[str, object],
    chern_by_key: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# TDBG Fig. 3 Chern Band Reproduction",
        "",
        "## Parameters",
        "",
    ]
    parameters = summary["parameters"]
    assert isinstance(parameters, dict)
    for key in (
        "theta_deg",
        "phi_deg",
        "epsilon",
        "cut",
        "resolution",
        "valley_minus_path_convention",
        "topology_mesh",
        "topology_boundary",
        "display_chern_sign",
    ):
        lines.append(f"- `{key} = {parameters[key]}`")
    lines.extend(["", "## Artifacts", ""])
    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    for key, value in artifacts.items():
        lines.append(f"- `{key} = {value}`")

    lines.extend(["", "## Chern Summary", ""])
    for key, payload in sorted(chern_by_key.items()):
        topology = payload.get("topology", {})
        assert isinstance(topology, dict)
        gap_metrics = payload.get("gap_metrics", {})
        assert isinstance(gap_metrics, dict)
        valence = _rounded_chern(topology.get("valence_band"))
        conduction = _rounded_chern(topology.get("conduction_band"))
        pair = _rounded_chern(topology.get("central_pair"))
        lower_gap = _rounded_chern(topology.get("lower_gap_occupied"))
        central_gap = _rounded_chern(topology.get("central_gap_occupied"))
        upper_gap = _rounded_chern(topology.get("upper_gap_occupied"))
        lines.extend(
            [
                f"### {key}",
                "",
                f"- `central_direct_gap_ev = {gap_metrics.get('central_direct_gap_ev')}`",
                f"- `valence_chern = {valence}`",
                f"- `conduction_chern = {conduction}`",
                f"- `central_pair_chern = {pair}`",
                f"- `integrated_lower_gap = {lower_gap}`",
                f"- `integrated_central_gap = {central_gap}`",
                f"- `integrated_upper_gap = {upper_gap}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    total_start = perf_counter()
    args = _parse_args()
    if not args.reuse_existing:
        ensure_not_running_compute_on_login_node("TDBG Fig. 3 Chern band reproduction")

    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    deltas = tuple(float(value) for value in args.deltas_ev)
    stackings = tuple(str(value) for value in args.stackings)
    specs = [Fig3PanelSpec(stacking=stacking, delta_ev=delta) for stacking in stackings for delta in deltas]

    panels: list[PanelBandData] = []
    chern_by_key: dict[str, dict[str, object]] = {}
    panel_npz_paths: dict[str, str] = {}

    for spec in specs:
        panel_dir = output_dir / spec.stacking.replace("-", "").lower() / _format_delta_dir(spec.delta_ev)
        panel_npz_path = panel_dir / "bands_path.npz"
        panel_npz_paths[spec.key] = str(panel_npz_path)

        if args.reuse_existing:
            if not panel_npz_path.exists():
                raise FileNotFoundError(f"Cannot reuse missing panel data: {panel_npz_path}")
            print(f"[reuse] {panel_npz_path}", flush=True)
            panel = _load_panel_npz(panel_npz_path, spec)
            panels.append(panel)
            panel_chern_path = panel_dir / "chern_numbers.json"
            if panel_chern_path.exists():
                chern_by_key[spec.key] = json.loads(panel_chern_path.read_text(encoding="utf-8"))
            continue

        print(f"[bands] {spec.stacking} Delta={spec.delta_ev * 1000.0:.0f} meV", flush=True)
        panel = _build_panel_data(
            spec,
            theta_deg=float(args.theta_deg),
            phi_deg=float(args.phi_deg),
            epsilon=float(args.epsilon),
            cut=float(args.cut),
            resolution=int(args.resolution),
        )
        panels.append(panel)
        panel_dir.mkdir(parents=True, exist_ok=True)
        _save_panel_npz(panel_npz_path, panel)

        if not args.skip_topology:
            chern_by_key[spec.key] = _compute_chern_data(
                spec,
                theta_deg=float(args.theta_deg),
                phi_deg=float(args.phi_deg),
                epsilon=float(args.epsilon),
                cut=float(args.cut),
                topology_mesh=int(args.topology_mesh),
                topology_boundary=str(args.topology_boundary),
                central_touching_threshold_ev=float(args.central_touching_threshold_ev),
            )
            write_json(panel_dir / "chern_numbers.json", chern_by_key[spec.key])

    figure_paths = _plot_fig3_grid(
        output_dir,
        panels,
        chern_by_key,
        theta_deg=float(args.theta_deg),
        window_ev=float(args.window_ev),
        display_chern_sign=int(args.display_chern_sign),
    )

    total_elapsed = perf_counter() - total_start
    artifacts = {
        **figure_paths,
        "chern_numbers_json": str(output_dir / "chern_numbers.json"),
        "summary_json": str(output_dir / "summary.json"),
        "report_md": str(output_dir / "fig3_chern_report.md"),
    }
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "parameters": {
            "theta_deg": float(args.theta_deg),
            "phi_deg": float(args.phi_deg),
            "epsilon": float(args.epsilon),
            "cut": float(args.cut),
            "resolution": int(args.resolution),
            "valley_minus_path_convention": "Kprime-Gamma-M-K physical path drawn on K-Gamma-M-Kprime axis",
            "topology_mesh": int(args.topology_mesh),
            "topology_boundary": str(args.topology_boundary),
            "deltas_ev": list(deltas),
            "stackings": list(stackings),
            "window_ev": float(args.window_ev),
            "central_touching_threshold_ev": float(args.central_touching_threshold_ev),
            "display_chern_sign": int(args.display_chern_sign),
            "skip_topology": bool(args.skip_topology),
            "reuse_existing": bool(args.reuse_existing),
        },
        "runtime": {
            "total_elapsed_sec": float(total_elapsed),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        },
        "runtime_environment": asdict(collect_runtime_environment()),
        "artifacts": artifacts,
        "panel_npz_paths": panel_npz_paths,
        "chern_by_panel": chern_by_key,
    }
    write_json(output_dir / "chern_numbers.json", chern_by_key)
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "fig3_chern_report.md", summary=summary, chern_by_key=chern_by_key)

    print(f"[done] output_dir={output_dir}")
    print(f"fig3_chern_bands_png={figure_paths['fig3_chern_bands_png']}")
    print(f"chern_numbers_json={output_dir / 'chern_numbers.json'}")
    print(f"summary_json={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
