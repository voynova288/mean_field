from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...api.artifacts import ModelRecord, write_contract_artifacts
from ...core.io import write_json_artifact
from .projected_hf import TDBGProjectedHFResult, liu2022_projected_hf_metadata, tdbg_hf_grid_band_summary


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ensure_fresh_tdbg_artifact_root(root: Path, *, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()) and not bool(overwrite):
        raise FileExistsError(
            f"Refusing to write TDBG projected-HF artifacts into non-empty root {root}. "
            "Use a fresh output directory or pass overwrite=True explicitly."
        )
    root.mkdir(parents=True, exist_ok=True)


def _label_payload(labels: object) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for label in labels:
        to_dict = getattr(label, "to_dict", None)
        payload.append(dict(to_dict()) if callable(to_dict) else {"repr": repr(label)})
    return payload


def write_tdbg_projected_hf_artifacts(
    output_dir: str | Path,
    result: TDBGProjectedHFResult,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a computed TDBG projected-HF result root plus public sidecars.

    This writer serializes an already computed ``TDBGProjectedHFResult``.  It
    does not run self-consistency, diagonalization, target-path reconstruction,
    or any other numerical workflow.
    """

    root = Path(output_dir)
    _ensure_fresh_tdbg_artifact_root(root, overwrite=overwrite)

    state = result.run.state
    data = result.data
    state_path = root / "hf_state.npz"
    summary_path = root / "projected_hf_summary.json"
    labels_path = root / "state_labels.json"

    np.savez_compressed(
        state_path,
        density=np.asarray(state.density, dtype=np.complex128),
        hamiltonian=np.asarray(state.hamiltonian, dtype=np.complex128),
        h0=np.asarray(state.h0, dtype=np.complex128),
        energies=np.asarray(state.energies, dtype=float),
        k_grid_frac=np.asarray(data.k_grid_frac, dtype=float),
        kvec_nm_inv=np.stack(
            [np.asarray(data.kvec.real, dtype=float), np.asarray(data.kvec.imag, dtype=float)],
            axis=-1,
        ),
        band_indices=np.asarray(data.band_indices, dtype=int),
        reference_density=np.asarray(data.reference_density, dtype=np.complex128),
        density_convention=np.asarray("projector"),
        density_axis_order=np.asarray("abk"),
        hamiltonian_axis_order=np.asarray("abk"),
    )

    config_payload = liu2022_projected_hf_metadata(data.config)
    config_payload.update({"init_mode": result.init_mode, "seed": int(result.seed)})
    summary_payload = {
        **result.to_summary_dict(),
        "grid_band_summary": tdbg_hf_grid_band_summary(result),
        "state_archive": "hf_state.npz",
    }
    write_json_artifact(summary_payload, summary_path, default=_json_default)
    write_json_artifact(_label_payload(data.labels), labels_path, default=_json_default)

    validation_payload = {
        "status": "pass" if bool(result.run.converged) else "not_converged",
        "converged": bool(result.run.converged),
        "exit_reason": str(result.run.exit_reason),
        "iterations": int(result.run.iterations),
    }
    sidecars = write_contract_artifacts(
        root,
        workflow="tdbg.projected_hf",
        system_name="tdbg",
        model=ModelRecord(
            system_name="tdbg",
            params=dict(config_payload),
            lattice={"theta_deg": float(data.config.theta_deg), "cut": float(data.config.cut)},
        ),
        config=config_payload,
        conventions={
            "energy_unit": "eV",
            "momentum_unit": "nm^-1",
            "density_convention": "projector",
            "density_axis_order": "abk",
            "hamiltonian_axis_order": "abk",
            "gauge": "tdbg_projected_hf_system_defined",
        },
        environment={},
        validation=validation_payload,
        observables=summary_payload,
        files={
            "hf_state": "hf_state.npz",
            "projected_hf_summary": "projected_hf_summary.json",
            "state_labels": "state_labels.json",
        },
        metadata={
            "runner_kind": "tdbg_projected_hf",
            "init_mode": result.init_mode,
            "seed": int(result.seed),
        },
        array_files=(state_path,),
    )
    return {
        "hf_state_npz": state_path,
        "projected_hf_summary.json": summary_path,
        "state_labels.json": labels_path,
        **sidecars,
    }


__all__ = ["write_tdbg_projected_hf_artifacts"]
