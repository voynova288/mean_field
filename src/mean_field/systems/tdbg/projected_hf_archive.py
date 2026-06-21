from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mean_field.core.hf import HartreeFockRun

from .model import TDBGModel
from .params import TDBGParameters
from .projected_hf_config import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedWindow
from .projected_hf_state import TDBGProjectedHFData, TDBGProjectedHFResult, TDBGProjectedHFState, TDBGStateLabel


def _json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_value(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    arr = np.asarray(value)
    raw = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"Expected JSON string metadata, got {type(raw).__name__}")


def _scalar(data: Mapping[str, np.ndarray], key: str, default: object | None = None) -> object:
    if key not in data:
        if default is not None:
            return default
        raise KeyError(key)
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value.reshape(-1)[0]


def _config_from_metadata(metadata: Mapping[str, Any]) -> TDBGProjectedHFConfig:
    window_payload = dict(metadata.get("window", {}))
    interaction_payload = dict(metadata.get("interaction", {}))
    payload = dict(metadata)
    payload["window"] = TDBGProjectedWindow(**window_payload) if window_payload else TDBGProjectedWindow()
    payload["interaction"] = TDBGInteractionSettings(**interaction_payload) if interaction_payload else TDBGInteractionSettings()
    return TDBGProjectedHFConfig(**payload)


def _model_from_metadata(metadata: Mapping[str, Any], config: TDBGProjectedHFConfig) -> TDBGModel:
    params_payload = dict(metadata.get("params", {}))
    for derived in ("vf", "v3", "v4", "phi_rad"):
        params_payload.pop(derived, None)
    params = TDBGParameters(**params_payload) if params_payload else TDBGParameters.full(stacking=config.stacking)
    return TDBGModel.from_config(float(metadata.get("theta_deg", config.theta_deg)), cut=float(metadata.get("cut", config.cut)), params=params)


def _labels_from_json(path: Path) -> tuple[TDBGStateLabel, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected state_labels.json to contain a list, got {type(payload).__name__}")
    labels: list[TDBGStateLabel] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("state_labels.json entries must be objects")
        labels.append(
            TDBGStateLabel(
                index=int(item["index"]),
                spin=str(item["spin"]),
                valley=int(item["valley"]),
                band_position=int(item["band_position"]),
                band_index=int(item["band_index"]),
            )
        )
    return tuple(labels)


def _valley_params_from_basis(data: Mapping[str, np.ndarray]) -> Mapping[int, TDBGParameters] | None:
    if "valley_params" not in data:
        return None
    payload = _json_value(data["valley_params"])
    out: dict[int, TDBGParameters] = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            raise ValueError("valley_params entries must be objects")
        item = dict(value)
        for derived in ("vf", "v3", "v4", "phi_rad"):
            item.pop(derived, None)
        out[int(key)] = TDBGParameters(**item)
    return out


def load_tdbg_projected_hf_result_from_archive(root: str | Path) -> TDBGProjectedHFResult:
    """Load a complete TDBG projected-HF archive without recomputing physics."""

    base = Path(root)
    if base.is_file():
        state_path = base
        base = state_path.parent
    else:
        state_path = base / "hf_state.npz"
    basis_path = base / "projected_basis.npz"
    labels_path = base / "state_labels.json"
    summary_path = base / "projected_hf_summary.json"
    config_path = base / "config.json"
    model_path = base / "model.json"
    for path in (state_path, basis_path, labels_path, summary_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(state_path, allow_pickle=False) as state_npz, np.load(basis_path, allow_pickle=False) as basis_npz:
        state_data = {key: state_npz[key] for key in state_npz.files}
        basis_data = {key: basis_npz[key] for key in basis_npz.files}
    config = _config_from_metadata(_json_file(config_path))
    model_metadata = _json_file(model_path) if model_path.is_file() else {"theta_deg": config.theta_deg, "cut": config.cut}
    model = _model_from_metadata(model_metadata, config)
    summary = _json_file(summary_path)
    state = TDBGProjectedHFState(
        h0=np.asarray(state_data["h0"], dtype=np.complex128),
        density=np.asarray(state_data["density"], dtype=np.complex128),
        hamiltonian=np.asarray(state_data["hamiltonian"], dtype=np.complex128),
        energies=np.asarray(state_data["energies"], dtype=float),
        mu=float(_scalar(state_data, "mu")),
        precision=float(config.precision),
        diagnostics={"hf_energy": float(np.asarray(state_data.get("iter_energy", [np.nan])).reshape(-1)[-1])},
    )
    run = HartreeFockRun(
        state=state,
        iter_energy=np.asarray(state_data["iter_energy"], dtype=float),
        iter_err=np.asarray(state_data["iter_err"], dtype=float),
        iter_oda=np.asarray(state_data["iter_oda"], dtype=float),
        init_mode=str(summary.get("init_mode", "archive")),
        seed=int(summary.get("seed", 0)),
        converged=bool(summary.get("converged", False)),
        exit_reason=str(summary.get("exit_reason", "archive_loaded")),
    )
    data = TDBGProjectedHFData(
        model=model,
        config=config,
        k_grid_frac=np.asarray(state_data["k_grid_frac"], dtype=float),
        kvec=np.asarray(state_data["kvec_nm_inv"], dtype=np.complex128),
        band_indices=tuple(int(v) for v in np.asarray(state_data["band_indices"]).reshape(-1)),
        labels=_labels_from_json(labels_path),
        h0=np.asarray(state_data["h0"], dtype=np.complex128),
        wavefunctions=np.asarray(basis_data["wavefunctions"], dtype=np.complex128),
        reference_density=np.asarray(state_data["reference_density"], dtype=np.complex128),
        n_occupied_per_k=int(_scalar(state_data, "n_occupied_per_k")),
        lower_band_count=int(_scalar(state_data, "lower_band_count")),
        moire_area_nm2=float(_scalar(basis_data, "moire_area_nm2")),
        shifts=tuple(tuple(int(x) for x in row) for row in np.asarray(basis_data["shifts"], dtype=int)),
        shift_gvecs=np.asarray(basis_data["shift_gvecs"], dtype=np.complex128),
        shift_srcmaps=tuple(np.asarray(item, dtype=int) for item in np.asarray(basis_data["shift_srcmaps"])),
        valley_params=_valley_params_from_basis(basis_data),
    )
    return TDBGProjectedHFResult(
        run=run,
        data=data,
        init_mode=str(summary.get("init_mode", run.init_mode)),
        seed=int(summary.get("seed", run.seed)),
        order_parameters=dict(summary.get("order_parameters", {})),
        energy_components=dict(summary.get("energy_components_ev", {})),
        hamiltonian_components=None,
    )


__all__ = ["load_tdbg_projected_hf_result_from_archive"]
