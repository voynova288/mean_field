from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mean_field.core.hf import HartreeFockRun, ProjectedWavefunctionBasis, empty_overlap_block_set

from ._hf_types import HTGHartreeFockRun, HTGHartreeFockState, HTGProjectedBasisData
from .model import HTGModel
from .params import HTGParams, InteractionParams


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    arr = np.asarray(value)
    if arr.shape == ():
        raw = arr.item()
    else:
        raw = arr.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"Expected JSON string metadata, got {type(raw).__name__}")


def _model_from_metadata(metadata: Mapping[str, Any]) -> HTGModel:
    params_payload = dict(metadata.get("params", {}))
    params_payload.pop("vf_ev_nm", None)
    params = HTGParams(**params_payload) if params_payload else HTGParams.default()
    return HTGModel.from_config(
        float(metadata["theta_deg"]),
        n_shells=int(metadata.get("n_shells", 5)),
        params=params,
    )


def _interaction_from_metadata(metadata: Mapping[str, Any]) -> InteractionParams:
    payload = dict(metadata)
    return InteractionParams(**payload)


def _scalar(data: Mapping[str, np.ndarray], key: str, default: object | None = None) -> object:
    if key not in data:
        if default is not None:
            return default
        raise KeyError(key)
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.reshape(-1)[0]


def load_htg_hf_run_from_archive(root: str | Path) -> HTGHartreeFockRun:
    """Load a complete primitive HTG HF archive without recomputing physics.

    The loader refuses summary-only artifacts: ``hf_ground_state.npz`` and
    ``hf_projected_basis.npz`` must contain the raw arrays and explicit
    model/interaction metadata needed by the canonical adapter.
    """

    base = Path(root)
    if base.is_file():
        state_path = base
        base = state_path.parent
    else:
        state_path = base / "hf_ground_state.npz"
    basis_path = base / "hf_projected_basis.npz"
    params_path = base / "hf_params.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not basis_path.is_file():
        raise FileNotFoundError(basis_path)
    with np.load(state_path, allow_pickle=False) as state_npz, np.load(basis_path, allow_pickle=False) as basis_npz:
        state_data = {key: state_npz[key] for key in state_npz.files}
        basis_data = {key: basis_npz[key] for key in basis_npz.files}
    metadata = _json_file(params_path) if params_path.is_file() else {}
    model_metadata = _json_value(basis_data["model_params"]) if "model_params" in basis_data else dict(metadata.get("model", {}))
    interaction_metadata = (
        _json_value(basis_data["interaction_params"])
        if "interaction_params" in basis_data
        else dict(metadata.get("interaction", {}))
    )
    model = _model_from_metadata(model_metadata)
    interaction = _interaction_from_metadata(interaction_metadata)
    wavefunctions = np.asarray(basis_data["wavefunctions"], dtype=np.complex128)
    h0 = np.asarray(state_data["h0"], dtype=np.complex128)
    n_spin = int(h0.shape[0] // (wavefunctions.shape[1] * wavefunctions.shape[2]))
    if n_spin <= 0 or n_spin * wavefunctions.shape[1] * wavefunctions.shape[2] != h0.shape[0]:
        raise ValueError("cannot infer HTG spin/flavor/band dimensions from archive")
    basis = ProjectedWavefunctionBasis(
        wavefunctions,
        grid_shape=tuple(int(v) for v in np.asarray(basis_data["reciprocal_grid_shape"]).reshape(-1)),
        n_spin=n_spin,
        boundary_mode="zero_fill",
    )
    projected_basis = HTGProjectedBasisData(
        model=model,
        interaction=interaction,
        mesh_size=int(round(np.sqrt(np.asarray(state_data["kvec_nm_inv"]).size))),
        kvec=np.asarray(state_data["kvec_nm_inv"], dtype=np.complex128),
        k_grid_frac=np.asarray(state_data["k_grid_frac"], dtype=float),
        basis=basis,
        h0=h0,
        sigma_z=np.asarray(state_data["sigma_z"], dtype=np.complex128),
        band_sigma_z=np.asarray(basis_data["band_sigma_z"], dtype=float),
        central_band_indices=tuple(int(v) for v in np.asarray(basis_data["central_band_indices"]).reshape(-1)),
        projected_band_indices=tuple(int(v) for v in np.asarray(basis_data["projected_band_indices"]).reshape(-1)),
        reciprocal_grid_shape=tuple(int(v) for v in np.asarray(basis_data["reciprocal_grid_shape"]).reshape(-1)),
        reciprocal_grid_origin=tuple(int(v) for v in np.asarray(basis_data["reciprocal_grid_origin"]).reshape(-1)),
        moire_cell_area_nm2=float(_scalar(basis_data, "moire_cell_area_nm2")),
    )
    state = HTGHartreeFockState(
        h0=h0,
        density=np.asarray(state_data["density"], dtype=np.complex128),
        hamiltonian=np.asarray(state_data["hamiltonian"], dtype=np.complex128),
        energies=np.asarray(state_data["energies_ev"], dtype=float),
        sigma_z=np.asarray(state_data["sigma_z"], dtype=np.complex128),
        nu=float(_scalar(state_data, "nu")),
        v0=1.0 / float(projected_basis.moire_cell_area_nm2),
        mu=float(_scalar(state_data, "mu")),
        precision=float(_scalar(state_data, "precision")),
        n_spin=n_spin,
        n_eta=int(wavefunctions.shape[2]),
        n_band=int(wavefunctions.shape[1]),
        occupation_counts=None,
        diagnostics={"hf_energy": float(np.asarray(state_data.get("iter_energy_ev", [np.nan])).reshape(-1)[-1])},
    )
    run = HartreeFockRun(
        state=state,
        iter_energy=np.asarray(state_data["iter_energy_ev"], dtype=float),
        iter_err=np.asarray(state_data["iter_err"], dtype=float),
        iter_oda=np.asarray(state_data["iter_oda"], dtype=float),
        init_mode=str(_scalar(state_data, "init_mode", "archive")),
        seed=int(_scalar(state_data, "seed", 0)),
        converged=bool(_scalar(state_data, "converged")),
        exit_reason=str(_scalar(state_data, "exit_reason")),
    )
    return HTGHartreeFockRun(
        state=state,
        iter_energy=run.iter_energy,
        iter_err=run.iter_err,
        iter_oda=run.iter_oda,
        init_mode=run.init_mode,
        seed=run.seed,
        converged=run.converged,
        exit_reason=run.exit_reason,
        overlap_blocks=empty_overlap_block_set(),
        basis_data=projected_basis,
    )


__all__ = ["load_htg_hf_run_from_archive"]
