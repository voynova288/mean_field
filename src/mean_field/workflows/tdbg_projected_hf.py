from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from mean_field.api.hf import HFConfig
from mean_field.runtime import ensure_not_running_compute_on_login_node


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _mapping_payload(payload: dict[str, Any], key: str, *, required: bool = True) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required config section {key!r}")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {key!r} must be an object")
    return dict(value)


def _reject_unknown_keys(section: str, payload: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys in {section}: {unknown}")


def tdbg_projected_hf_config_from_payload(payload: dict[str, Any]):
    from mean_field.systems.tdbg import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedWindow

    projected = _mapping_payload(payload, "tdbg_projected_hf")
    _reject_unknown_keys(
        "tdbg_projected_hf",
        projected,
        {
            "theta_deg",
            "cut",
            "mesh_size",
            "paper_ud_ev",
            "paper_ud_convention",
            "stacking",
            "window",
            "filling",
            "interaction",
            "precision",
            "max_iter",
            "mix_fallback",
            "frac_shift",
            "orbital_zeeman_b_t",
            "orbital_zeeman_delta_k_nm_inv",
        },
    )
    window_payload = projected.get("window", {})
    if isinstance(window_payload, str):
        window = TDBGProjectedWindow(name=window_payload)
    elif isinstance(window_payload, dict):
        _reject_unknown_keys("tdbg_projected_hf.window", dict(window_payload), {"name", "band_indices"})
        band_indices = window_payload.get("band_indices")
        window = TDBGProjectedWindow(
            name=str(window_payload.get("name", "two_flat")),
            band_indices=None if band_indices is None else tuple(int(value) for value in band_indices),
        )
    else:
        raise ValueError("tdbg_projected_hf.window must be a string or object")

    interaction_payload = dict(projected.get("interaction") or {})
    _reject_unknown_keys(
        "tdbg_projected_hf.interaction",
        interaction_payload,
        {
            "include_intersite",
            "include_onsite",
            "hubbard_u_ev",
            "epsilon_r",
            "kappa_nm_inv",
            "g_shells",
            "hartree_reference",
            "fock_density",
            "onsite_valley_policy",
            "drop_g0_hartree",
        },
    )
    interaction = TDBGInteractionSettings(**interaction_payload)
    return TDBGProjectedHFConfig(
        theta_deg=float(projected.get("theta_deg", 1.38)),
        cut=float(projected.get("cut", 5.0)),
        mesh_size=int(projected.get("mesh_size", 9)),
        paper_ud_ev=float(projected.get("paper_ud_ev", 0.09)),
        paper_ud_convention=projected.get("paper_ud_convention", "same_delta_minus_ud_over3"),
        stacking=str(projected.get("stacking", "AB-BA")),
        window=window,
        filling=int(projected.get("filling", 2)),
        interaction=interaction,
        precision=float(projected.get("precision", 1.0e-7)),
        max_iter=int(projected.get("max_iter", 300)),
        mix_fallback=None if projected.get("mix_fallback") is None else float(projected["mix_fallback"]),
        frac_shift=None if projected.get("frac_shift") is None else tuple(float(value) for value in projected["frac_shift"]),
        orbital_zeeman_b_t=float(projected.get("orbital_zeeman_b_t", 0.0)),
        orbital_zeeman_delta_k_nm_inv=float(projected.get("orbital_zeeman_delta_k_nm_inv", 1.0e-5)),
    )


def tdbg_hf_config_from_payload(payload: dict[str, Any], tdbg_config: Any) -> HFConfig:
    hf_payload = _mapping_payload(payload, "hf", required=False)
    _reject_unknown_keys(
        "hf",
        hf_payload,
        {
            "filling",
            "mesh",
            "density_convention",
            "max_iter",
            "precision",
            "interaction_scheme",
            "epsilon_r",
            "dsc_nm",
            "coulomb_kernel",
            "seeds",
            "metadata",
        },
    )
    mesh_raw = hf_payload.get("mesh", [int(tdbg_config.mesh_size), int(tdbg_config.mesh_size)])
    if len(mesh_raw) != 2:
        raise ValueError(f"hf.mesh must have two entries, got {mesh_raw!r}")
    return HFConfig(
        filling=float(hf_payload.get("filling", tdbg_config.filling)),
        mesh=(int(mesh_raw[0]), int(mesh_raw[1])),
        density_convention=hf_payload.get("density_convention", "projector"),
        max_iter=int(hf_payload.get("max_iter", tdbg_config.max_iter)),
        precision=float(hf_payload.get("precision", tdbg_config.precision)),
        interaction_scheme=hf_payload.get("interaction_scheme", "average"),
        epsilon_r=float(hf_payload.get("epsilon_r", tdbg_config.interaction.epsilon_r)),
        dsc_nm=float(hf_payload.get("dsc_nm", 10.0)),
        coulomb_kernel=hf_payload.get("coulomb_kernel", "2d_gate"),
        seeds=tuple(str(value) for value in hf_payload.get("seeds", ("random",))),
        metadata=dict(hf_payload.get("metadata", {})),
    )


def tdbg_run_config_from_payload(payload: dict[str, Any]) -> tuple[str, int]:
    run_payload = _mapping_payload(payload, "run")
    _reject_unknown_keys("run", run_payload, {"init_mode", "seed"})
    if "init_mode" not in run_payload:
        raise ValueError("run.init_mode is required for TDBG projected HF")
    return str(run_payload["init_mode"]), int(run_payload.get("seed", 1))


def tdbg_output_dir_from_payload(payload: dict[str, Any], output_dir: Path | None) -> Path | None:
    if output_dir is not None:
        return Path(output_dir)
    result_payload = _mapping_payload(payload, "result", required=False)
    _reject_unknown_keys("result", result_payload, {"output_dir"})
    value = result_payload.get("output_dir")
    return None if value is None else Path(str(value))


def validate_tdbg_cli_hf_config(hf_config: HFConfig, tdbg_config: Any) -> None:
    if int(hf_config.mesh[0]) != int(hf_config.mesh[1]) or int(hf_config.mesh[0]) != int(tdbg_config.mesh_size):
        raise ValueError(
            "TDBG projected-HF CLI requires hf.mesh=(mesh_size, mesh_size) matching "
            f"tdbg_projected_hf.mesh_size={tdbg_config.mesh_size}, got {hf_config.mesh}"
        )
    if float(hf_config.filling) != float(int(tdbg_config.filling)):
        raise ValueError(f"TDBG projected-HF CLI requires hf.filling={tdbg_config.filling}, got {hf_config.filling}")
    if int(hf_config.max_iter) != int(tdbg_config.max_iter):
        raise ValueError(
            f"TDBG projected-HF CLI requires hf.max_iter={tdbg_config.max_iter}, got {hf_config.max_iter}"
        )
    if not np.isclose(float(hf_config.precision), float(tdbg_config.precision)):
        raise ValueError(
            f"TDBG projected-HF CLI requires hf.precision={tdbg_config.precision}, got {hf_config.precision}"
        )
    if hf_config.density_convention != "projector":
        raise ValueError("TDBG projected-HF CLI requires hf.density_convention='projector'")
    if hf_config.active_window is not None or hf_config.active_band_indices is not None:
        raise NotImplementedError(
            "TDBG projected-HF CLI takes the projected window from tdbg_projected_hf.window; "
            "leave hf.active_window/active_band_indices unset"
        )


def validate_tdbg_output_root_is_fresh(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to run TDBG projected HF into non-empty output directory {output_dir}. "
            "Use a fresh directory."
        )


def save_tdbg_projected_hf_result(result: Any, output_dir: Path) -> Path:
    from mean_field.systems.tdbg.artifacts import write_tdbg_projected_hf_artifacts
    from mean_field.systems.tdbg.projected_hf import TDBGProjectedHFResult

    raw_state = getattr(result, "state", None)
    if not isinstance(raw_state, TDBGProjectedHFResult):
        raise TypeError(
            "TDBG projected-HF CLI expected mean_field.api.run_hf to return an HFResult wrapping "
            f"TDBGProjectedHFResult, got {type(raw_state).__name__}"
        )
    paths = write_tdbg_projected_hf_artifacts(output_dir, raw_state)
    return paths["manifest.json"]


def validate_tdbg_workflow_payload(payload: dict[str, Any]) -> None:
    _reject_unknown_keys(
        str(payload.get("workflow", "workflow")),
        payload,
        {"schema_version", "workflow", "system", "tdbg_projected_hf", "hf", "run", "result"},
    )
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError(f"Unsupported TDBG workflow schema_version={payload.get('schema_version')!r}")
    if payload.get("workflow") != "tdbg.projected_hf.explicit_config":
        raise ValueError("TDBG projected-HF CLI requires workflow='tdbg.projected_hf.explicit_config'")
    if str(payload.get("system", "tdbg")).lower().replace("-", "_") != "tdbg":
        raise ValueError("TDBG projected-HF CLI only supports system='tdbg'")


def run_tdbg_projected_hf_workflow(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    dry_run: bool = False,
    make_model_fn: Callable[..., object] | None = None,
    run_hf_fn: Callable[..., object] | None = None,
    compute_guard: Callable[[str], None] = ensure_not_running_compute_on_login_node,
) -> int:
    from mean_field.api.hf import run_hf as default_run_hf
    from mean_field.api.models import make_model as default_make_model
    from mean_field.systems.tdbg.projected_hf_config import (
        tdbg_parameters_from_paper_ud_for_valley,
        validate_tdbg_projected_hf_config,
    )

    make_model_call = default_make_model if make_model_fn is None else make_model_fn
    run_hf_call = default_run_hf if run_hf_fn is None else run_hf_fn

    payload = _read_json_object(config_path)
    validate_tdbg_workflow_payload(payload)
    tdbg_config = tdbg_projected_hf_config_from_payload(payload)
    validate_tdbg_projected_hf_config(tdbg_config)
    hf_config = tdbg_hf_config_from_payload(payload, tdbg_config)
    validate_tdbg_cli_hf_config(hf_config, tdbg_config)
    init_mode, seed = tdbg_run_config_from_payload(payload)
    resolved_output_dir = tdbg_output_dir_from_payload(payload, output_dir)

    if dry_run:
        print(
            "workflow=tdbg.projected_hf.explicit_config\t"
            f"theta_deg={tdbg_config.theta_deg:.12g}\t"
            f"cut={tdbg_config.cut:.12g}\t"
            f"mesh_size={tdbg_config.mesh_size}\t"
            f"filling={tdbg_config.filling}\t"
            f"init_mode={init_mode}\t"
            f"seed={seed}\t"
            f"output_dir={'' if resolved_output_dir is None else resolved_output_dir}"
        )
        return 0

    if resolved_output_dir is None:
        raise ValueError("TDBG projected-HF run requires --output-dir or result.output_dir in the config")
    validate_tdbg_output_root_is_fresh(resolved_output_dir)
    compute_guard("TDBG projected HF")
    params = tdbg_parameters_from_paper_ud_for_valley(
        tdbg_config.paper_ud_ev,
        stacking=tdbg_config.stacking,
        valley=1,
        convention=tdbg_config.paper_ud_convention,
    )
    model = make_model_call("tdbg", theta_deg=tdbg_config.theta_deg, cut=tdbg_config.cut, params=params)
    result = run_hf_call(model, hf_config, tdbg_config=tdbg_config, init_mode=init_mode, seed=seed)
    manifest_path = save_tdbg_projected_hf_result(result, resolved_output_dir)
    print(f"manifest={manifest_path}\toutput_dir={resolved_output_dir}")
    return 0


__all__ = [
    "run_tdbg_projected_hf_workflow",
    "save_tdbg_projected_hf_result",
    "tdbg_hf_config_from_payload",
    "tdbg_output_dir_from_payload",
    "tdbg_projected_hf_config_from_payload",
    "tdbg_run_config_from_payload",
    "validate_tdbg_cli_hf_config",
    "validate_tdbg_output_root_is_fresh",
    "validate_tdbg_workflow_payload",
]
