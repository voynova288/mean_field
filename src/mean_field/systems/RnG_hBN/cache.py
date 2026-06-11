from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from ...core.hf import ProjectedWavefunctionBasis
from ...core.io import write_json_artifact
from .hf import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    RLGhBNLayerOverlapBlockSet,
    RLGhBNProjectedBasisData,
    active_band_indices_for_interaction,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    rlg_hbn_layer_component_groups,
)
from .interaction import RLGhBNInteractionParams
from .model import RLGhBNModel
from .screening import (
    ScreenedInterlayerPotentialResult,
    screening_result_from_dict,
    screening_result_to_dict,
    solve_screened_interlayer_potential,
    solve_screened_interlayer_potential_grid,
)


CACHE_POLICY_CHOICES = ("reuse", "refresh", "off")


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_cache_array(path: Path, *, dtype: np.dtype | type, mmap: bool = False) -> np.ndarray:
    array = np.load(path, mmap_mode="r" if mmap else None)
    expected_dtype = np.dtype(dtype)
    if array.dtype == expected_dtype:
        return array
    return np.asarray(array, dtype=expected_dtype)


@dataclass(frozen=True)
class RLGhBNCacheResult:
    value: object
    key: str
    path: Path | None
    hit: bool
    manifest: dict[str, object]


class RLGhBNCacheMiss(RuntimeError):
    pass


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)


def _write_json(path: Path, payload: object) -> None:
    write_json_artifact(payload, path, default=_json_default)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _complex_pairs(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    return np.stack([array.real, array.imag], axis=-1)


def _complex_from_pairs(values: np.ndarray) -> np.ndarray:
    pairs = np.asarray(values, dtype=float)
    if pairs.shape[-1] != 2:
        raise ValueError(f"Expected complex-pair final axis length 2, got {pairs.shape}")
    return np.asarray(pairs[..., 0] + 1j * pairs[..., 1], dtype=np.complex128)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _git_commit_sha() -> str | None:
    root = _repo_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def _code_version_payload() -> dict[str, object]:
    sha = _git_commit_sha()
    return {
        "package": "mean-field",
        "git_commit_sha": sha or "unavailable",
    }


def _model_cache_payload(model: RLGhBNModel) -> dict[str, object]:
    summary = model.lattice_summary()
    return {
        "layer_count": int(summary["layer_count"]),
        "xi": int(summary["xi"]),
        "theta_deg": float(summary["theta_deg"]),
        "shell_count": int(summary["shell_count"]),
        "displacement_field_mev": float(summary["displacement_field_mev"]),
        "params": model.params.to_summary_dict(),
        "lattice": model.lattice.to_summary_dict(),
    }


def _model_from_payload(payload: dict[str, object]) -> RLGhBNModel:
    return RLGhBNModel.from_config(
        layer_count=int(payload["layer_count"]),
        xi=int(payload["xi"]),
        theta_deg=float(payload["theta_deg"]),
        displacement_field_mev=float(payload["displacement_field_mev"]),
        shell_count=int(payload["shell_count"]),
    )


def _interaction_from_payload(payload: dict[str, object]) -> RLGhBNInteractionParams:
    return RLGhBNInteractionParams(
        epsilon_r=float(payload["epsilon_r"]),
        gate_distance_nm=float(payload["gate_distance_nm"]),
        scheme=str(payload["scheme"]),
        interaction_dimension=str(payload.get("interaction_dimension", "3d_layer_dependent")),
        active_valence_bands=int(payload["active_valence_bands"]),
        active_conduction_bands=int(payload["active_conduction_bands"]),
        k_mesh_size=int(payload["k_mesh_size"]),
        hilbert_cutoff_q1=float(payload.get("hilbert_cutoff_q1", 4.0)),
        interaction_cutoff_q1=float(payload["interaction_cutoff_q1"]),
        use_screened_basis=bool(payload["use_screened_basis"]),
    )


def _manifest_for(
    kind: str,
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "kind": str(kind),
        "code_version": _code_version_payload(),
        "model": _model_cache_payload(model),
        "interaction": interaction.to_summary_dict(),
        "extra": {} if extra is None else dict(extra),
    }


def rlg_hbn_cache_key(
    kind: str,
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    extra: dict[str, object] | None = None,
) -> str:
    manifest = _manifest_for(kind, model, interaction, extra=extra)
    digest = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()[:24]
    return f"{str(kind)}_{digest}"


def _cache_path(cache_dir: Path, kind: str, key: str) -> Path:
    return Path(cache_dir).resolve() / kind / key


def _check_policy(cache_policy: str) -> str:
    policy = str(cache_policy)
    if policy not in CACHE_POLICY_CHOICES:
        raise ValueError(f"cache_policy must be one of {CACHE_POLICY_CHOICES}, got {cache_policy!r}")
    return policy


def _manifest_matches(path: Path, expected: dict[str, object]) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        actual = _read_json(manifest_path)
    except Exception:
        return False
    return _canonical_json(actual) == _canonical_json(expected)


def _prepare_write_dir(path: Path) -> Path:
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _finish_write_dir(tmp_path: Path, final_path: Path) -> None:
    if final_path.exists():
        shutil.rmtree(final_path)
    tmp_path.replace(final_path)


def load_or_solve_screening(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    cache_dir: Path,
    cache_policy: str = "reuse",
    solver: str = "grid",
    mesh_size: int | None = None,
    u_min_mev: float = -100.0,
    u_max_mev: float = 200.0,
    n_grid: int = 121,
    root_tolerance_mev: float = 1.0e-5,
    fixed_point_max_iter: int = 50,
    fixed_point_tolerance_mev: float = 1.0e-6,
    fixed_point_mixing: float = 0.5,
) -> RLGhBNCacheResult:
    policy = _check_policy(cache_policy)
    if solver not in {"grid", "fixed_point"}:
        raise ValueError(f"screening solver must be 'grid' or 'fixed_point', got {solver!r}")
    resolved_mesh = interaction.k_mesh_size if mesh_size is None else int(mesh_size)
    extra = {
        "screening_solver": str(solver),
        "screening_mesh_size": int(resolved_mesh),
        "screening_u_min_mev": float(u_min_mev),
        "screening_u_max_mev": float(u_max_mev),
        "screening_u_grid_points": int(n_grid),
        "screening_root_tolerance_mev": float(root_tolerance_mev),
        "screening_hartree_projection": "appendix_b5_centered_layer_scalar_2_over_L_v1",
    }
    key = rlg_hbn_cache_key("screening", model, interaction, extra)
    manifest = _manifest_for("screening", model, interaction, extra=extra)
    path = _cache_path(cache_dir, "screening", key)
    result_path = path / "screening_result.json"
    if policy == "reuse" and _manifest_matches(path, manifest) and result_path.exists():
        result = screening_result_from_dict(_read_json(result_path))
        return RLGhBNCacheResult(value=result, key=key, path=path, hit=True, manifest=manifest)

    if solver == "grid":
        result = solve_screened_interlayer_potential_grid(
            model,
            interaction,
            mesh_size=resolved_mesh,
            u_min_mev=float(u_min_mev),
            u_max_mev=float(u_max_mev),
            n_grid=int(n_grid),
            root_tolerance_mev=float(root_tolerance_mev),
        )
    else:
        result = solve_screened_interlayer_potential(
            model,
            interaction,
            mesh_size=resolved_mesh,
            max_iter=int(fixed_point_max_iter),
            tolerance_mev=float(fixed_point_tolerance_mev),
            mixing=float(fixed_point_mixing),
        )

    if policy != "off":
        tmp_path = _prepare_write_dir(path)
        _write_json(tmp_path / "screening_result.json", screening_result_to_dict(result))
        _write_json(tmp_path / "manifest.json", manifest)
        _finish_write_dir(tmp_path, path)
    return RLGhBNCacheResult(value=result, key=key, path=None if policy == "off" else path, hit=False, manifest=manifest)


def save_projected_basis_cache(
    cache_dir: Path,
    key: str,
    basis_data: RLGhBNProjectedBasisData,
    manifest: dict[str, object],
) -> Path:
    path = _cache_path(cache_dir, "basis", key)
    tmp_path = _prepare_write_dir(path)
    np.save(tmp_path / "wavefunctions.npy", np.asarray(basis_data.basis.wavefunctions, dtype=np.complex128))
    np.save(tmp_path / "h0.npy", np.asarray(basis_data.h0, dtype=np.complex128))
    np.save(tmp_path / "physical_h0.npy", np.asarray(basis_data.physical_h0, dtype=np.complex128))
    np.save(
        tmp_path / "fixed_remote_hamiltonian.npy",
        np.asarray(basis_data.fixed_remote_hamiltonian, dtype=np.complex128),
    )
    np.save(tmp_path / "band_energies.npy", np.asarray(basis_data.band_energies, dtype=float))
    np.save(tmp_path / "kvec_complex_pairs.npy", _complex_pairs(basis_data.kvec))
    np.save(tmp_path / "k_grid_frac.npy", np.asarray(basis_data.k_grid_frac, dtype=float))
    np.save(tmp_path / "active_band_indices.npy", np.asarray(basis_data.active_band_indices, dtype=int))
    np.save(tmp_path / "flat_band_indices.npy", np.asarray(basis_data.flat_band_indices, dtype=int))
    np.save(tmp_path / "valleys.npy", np.asarray(basis_data.valleys, dtype=int))
    np.save(tmp_path / "reciprocal_grid_shape.npy", np.asarray(basis_data.reciprocal_grid_shape, dtype=int))
    np.save(tmp_path / "reciprocal_grid_origin.npy", np.asarray(basis_data.reciprocal_grid_origin, dtype=int))
    np.save(tmp_path / "moire_cell_area_nm2.npy", np.asarray([basis_data.moire_cell_area_nm2], dtype=float))
    if basis_data.screening is not None:
        _write_json(tmp_path / "screening_result.json", screening_result_to_dict(basis_data.screening))
    _write_json(tmp_path / "manifest.json", manifest)
    _finish_write_dir(tmp_path, path)
    return path


def load_projected_basis_cache(cache_dir: Path, key: str) -> RLGhBNProjectedBasisData:
    path = _cache_path(cache_dir, "basis", key)
    required = (
        "manifest.json",
        "wavefunctions.npy",
        "h0.npy",
        "physical_h0.npy",
        "fixed_remote_hamiltonian.npy",
        "band_energies.npy",
        "kvec_complex_pairs.npy",
        "k_grid_frac.npy",
        "active_band_indices.npy",
        "flat_band_indices.npy",
        "valleys.npy",
        "reciprocal_grid_shape.npy",
        "reciprocal_grid_origin.npy",
        "moire_cell_area_nm2.npy",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RLGhBNCacheMiss(f"Basis cache {path} is missing {missing}")
    manifest = _read_json(path / "manifest.json")
    model_payload = manifest["model"]
    extra = manifest.get("extra", {})
    if not isinstance(model_payload, dict) or not isinstance(extra, dict):
        raise RLGhBNCacheMiss(f"Basis cache {path} has an invalid manifest")
    basis_model_payload = extra.get("basis_model")
    if not isinstance(basis_model_payload, dict):
        raise RLGhBNCacheMiss(f"Basis cache {path} lacks basis_model metadata")
    if extra.get("basis_periodic_gauge") != RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION:
        raise RLGhBNCacheMiss(
            f"Basis cache {path} uses stale periodic-gauge metadata "
            f"{extra.get('basis_periodic_gauge')!r}; expected {RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION!r}"
        )
    if extra.get("form_factor_convention") != RLG_HBN_FORM_FACTOR_CONVENTION_VERSION:
        raise RLGhBNCacheMiss(
            f"Basis cache {path} uses stale form-factor convention "
            f"{extra.get('form_factor_convention')!r}; expected {RLG_HBN_FORM_FACTOR_CONVENTION_VERSION!r}"
        )
    model = _model_from_payload(model_payload)
    basis_model = _model_from_payload(basis_model_payload)
    interaction_payload = manifest["interaction"]
    if not isinstance(interaction_payload, dict):
        raise RLGhBNCacheMiss(f"Basis cache {path} lacks interaction metadata")
    interaction = _interaction_from_payload(interaction_payload)
    screening = None
    if (path / "screening_result.json").exists():
        screening = screening_result_from_dict(_read_json(path / "screening_result.json"))
    wavefunctions = np.load(path / "wavefunctions.npy")
    reciprocal_grid_shape = tuple(int(value) for value in np.load(path / "reciprocal_grid_shape.npy").reshape(-1))
    basis = ProjectedWavefunctionBasis(
        wavefunctions=np.asarray(wavefunctions, dtype=np.complex128),
        grid_shape=reciprocal_grid_shape,  # type: ignore[arg-type]
        n_spin=2,
        local_basis_size=2 * int(basis_model.params.layer_count),
        name=str(extra.get("basis_name", "rlg_hbn_screened_active")),
        component_groups=rlg_hbn_layer_component_groups(basis_model.params.layer_count),
    )
    return RLGhBNProjectedBasisData(
        model=model,
        basis_model=basis_model,
        interaction=interaction,
        screening=screening,
        mesh_size=int(extra.get("mesh_size", interaction.k_mesh_size)),
        kvec=_complex_from_pairs(np.load(path / "kvec_complex_pairs.npy")),
        k_grid_frac=np.asarray(np.load(path / "k_grid_frac.npy"), dtype=float),
        basis=basis,
        h0=np.asarray(np.load(path / "h0.npy"), dtype=np.complex128),
        band_energies=np.asarray(np.load(path / "band_energies.npy"), dtype=float),
        active_band_indices=tuple(int(value) for value in np.load(path / "active_band_indices.npy").reshape(-1)),
        flat_band_indices=tuple(int(value) for value in np.load(path / "flat_band_indices.npy").reshape(-1)),
        valleys=tuple(int(value) for value in np.load(path / "valleys.npy").reshape(-1)),
        reciprocal_grid_shape=reciprocal_grid_shape,  # type: ignore[arg-type]
        reciprocal_grid_origin=tuple(int(value) for value in np.load(path / "reciprocal_grid_origin.npy").reshape(-1)),  # type: ignore[arg-type]
        moire_cell_area_nm2=float(np.load(path / "moire_cell_area_nm2.npy").reshape(-1)[0]),
        physical_h0=np.asarray(np.load(path / "physical_h0.npy"), dtype=np.complex128),
        fixed_remote_hamiltonian=np.asarray(np.load(path / "fixed_remote_hamiltonian.npy"), dtype=np.complex128),
    )


def load_or_build_projected_basis(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    cache_dir: Path,
    cache_policy: str = "reuse",
    mesh_size: int | None = None,
    frac_shift: tuple[float, float] = (0.0, 0.0),
    valleys: tuple[int, ...] = (1, -1),
    screening: ScreenedInterlayerPotentialResult | None = None,
    screening_solver: str = "grid",
    screening_mesh_size: int | None = None,
    screening_u_min_mev: float = -100.0,
    screening_u_max_mev: float = 200.0,
    screening_u_grid_points: int = 121,
    screening_root_tolerance_mev: float = 1.0e-5,
) -> RLGhBNCacheResult:
    policy = _check_policy(cache_policy)
    resolved_mesh = interaction.k_mesh_size if mesh_size is None else int(mesh_size)
    resolved_screening_mesh = resolved_mesh if screening_mesh_size is None else int(screening_mesh_size)
    if interaction.use_screened_basis and screening is None:
        screening_cache = load_or_solve_screening(
            model,
            interaction,
            cache_dir=cache_dir,
            cache_policy=cache_policy,
            solver=screening_solver,
            mesh_size=resolved_screening_mesh,
            u_min_mev=screening_u_min_mev,
            u_max_mev=screening_u_max_mev,
            n_grid=screening_u_grid_points,
            root_tolerance_mev=screening_root_tolerance_mev,
        )
        screening = screening_cache.value  # type: ignore[assignment]
    if screening is not None and interaction.use_screened_basis:
        screened_params = replace(model.params, displacement_field_mev=float(screening.screened_u_mev))
        basis_model = RLGhBNModel.from_config(
            layer_count=model.params.layer_count,
            xi=model.params.xi,
            theta_deg=model.lattice.theta_deg,
            displacement_field_mev=float(screening.screened_u_mev),
            shell_count=model.lattice.shell_count,
            params=screened_params,
        )
    else:
        basis_model = model
    active_indices = active_band_indices_for_interaction(basis_model, interaction)
    extra = {
        "mesh_size": int(resolved_mesh),
        "frac_shift": [float(frac_shift[0]), float(frac_shift[1])],
        "valleys": [int(value) for value in valleys],
        "screening_mesh_size": int(resolved_screening_mesh),
        "screening_solver": str(screening_solver),
        "screened_u_mev": None if screening is None else float(screening.screened_u_mev),
        "physical_v_mev": float(model.params.displacement_field_mev),
        "basis_model": _model_cache_payload(basis_model),
        "basis_name": "rlg_hbn_screened_active",
        "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
        "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
        "active_band_indices": [int(value) for value in active_indices],
        "flat_band_indices": [int(value) for value in basis_model.flat_band_indices],
        "screening_u_min_mev": float(screening_u_min_mev),
        "screening_u_max_mev": float(screening_u_max_mev),
        "screening_u_grid_points": int(screening_u_grid_points),
        "screening_root_tolerance_mev": float(screening_root_tolerance_mev),
    }
    key = rlg_hbn_cache_key("basis", model, interaction, extra)
    manifest = _manifest_for("basis", model, interaction, extra=extra)
    path = _cache_path(cache_dir, "basis", key)
    if policy == "reuse" and _manifest_matches(path, manifest):
        try:
            value = load_projected_basis_cache(cache_dir, key)
        except Exception:
            pass
        else:
            return RLGhBNCacheResult(value=value, key=key, path=path, hit=True, manifest=manifest)

    value = build_rlg_hbn_projected_basis(
        model,
        interaction,
        mesh_size=resolved_mesh,
        frac_shift=frac_shift,
        valleys=valleys,
        screening_mesh_size=resolved_screening_mesh,
        screening_result=screening,
        screening_solver=screening_solver,
        screening_u_min_mev=screening_u_min_mev,
        screening_u_max_mev=screening_u_max_mev,
        screening_u_grid_points=screening_u_grid_points,
        screening_root_tolerance_mev=screening_root_tolerance_mev,
    )
    if policy != "off":
        path = save_projected_basis_cache(cache_dir, key, value, manifest)
    return RLGhBNCacheResult(value=value, key=key, path=None if policy == "off" else path, hit=False, manifest=manifest)


def save_layer_overlap_blocks_cache(
    cache_dir: Path,
    key: str,
    blocks: RLGhBNLayerOverlapBlockSet,
    manifest: dict[str, object],
) -> Path:
    path = _cache_path(cache_dir, "overlap", key)
    tmp_path = _prepare_write_dir(path)
    np.save(tmp_path / "shifts.npy", np.asarray(blocks.shifts, dtype=int))
    np.save(tmp_path / "gvecs_complex_pairs.npy", _complex_pairs(blocks.gvecs))
    for shift in blocks.shifts:
        m, n = int(shift[0]), int(shift[1])
        shift_dir = tmp_path / f"shift_{m}_{n}"
        shift_dir.mkdir(parents=True, exist_ok=True)
        np.save(shift_dir / "layer_overlap.npy", np.asarray(blocks.layer_overlaps[shift], dtype=np.complex128))
        np.save(
            shift_dir / "layer_diagonal_overlap.npy",
            np.asarray(blocks.layer_diagonal_overlaps[shift], dtype=np.complex128),
        )
        np.save(
            shift_dir / "hartree_layer_coulomb.npy",
            np.asarray(blocks.hartree_layer_coulomb[shift], dtype=float),
        )
        np.save(
            shift_dir / "fock_layer_coulomb.npy",
            np.asarray(blocks.fock_layer_coulomb[shift], dtype=float),
        )
    _write_json(tmp_path / "manifest.json", manifest)
    _finish_write_dir(tmp_path, path)
    return path


def load_layer_overlap_blocks_cache(cache_dir: Path, key: str) -> RLGhBNLayerOverlapBlockSet:
    path = _cache_path(cache_dir, "overlap", key)
    if not (path / "manifest.json").exists() or not (path / "shifts.npy").exists():
        raise RLGhBNCacheMiss(f"Overlap cache {path} is missing manifest or shifts")
    manifest = _read_json(path / "manifest.json")
    extra = manifest.get("extra", {})
    if not isinstance(extra, dict):
        raise RLGhBNCacheMiss(f"Overlap cache {path} has an invalid manifest")
    if extra.get("basis_periodic_gauge") != RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION:
        raise RLGhBNCacheMiss(
            f"Overlap cache {path} uses stale periodic-gauge metadata "
            f"{extra.get('basis_periodic_gauge')!r}; expected {RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION!r}"
        )
    if extra.get("form_factor_convention") != RLG_HBN_FORM_FACTOR_CONVENTION_VERSION:
        raise RLGhBNCacheMiss(
            f"Overlap cache {path} uses stale form-factor convention "
            f"{extra.get('form_factor_convention')!r}; expected {RLG_HBN_FORM_FACTOR_CONVENTION_VERSION!r}"
        )
    mmap = _env_flag_enabled("MEAN_FIELD_RLG_HBN_MMAP_OVERLAP_CACHE", default=False)
    shifts = tuple((int(row[0]), int(row[1])) for row in np.asarray(np.load(path / "shifts.npy"), dtype=int))
    gvecs_path = path / "gvecs_complex_pairs.npy"
    if not gvecs_path.exists():
        raise RLGhBNCacheMiss(f"Overlap cache {path} is missing gvecs_complex_pairs.npy")
    gvecs = _complex_from_pairs(np.load(gvecs_path))
    layer_overlaps: dict[tuple[int, int], np.ndarray] = {}
    layer_diagonal_overlaps: dict[tuple[int, int], np.ndarray] = {}
    hartree_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    fock_layer_coulomb: dict[tuple[int, int], np.ndarray] = {}
    for shift in shifts:
        shift_dir = path / f"shift_{shift[0]}_{shift[1]}"
        files = {
            "layer_overlap": shift_dir / "layer_overlap.npy",
            "layer_diagonal_overlap": shift_dir / "layer_diagonal_overlap.npy",
            "hartree_layer_coulomb": shift_dir / "hartree_layer_coulomb.npy",
            "fock_layer_coulomb": shift_dir / "fock_layer_coulomb.npy",
        }
        missing = [name for name, file_path in files.items() if not file_path.exists()]
        if missing:
            raise RLGhBNCacheMiss(f"Overlap cache {path} is corrupt; shift {shift} is missing {missing}")
        layer_overlaps[shift] = _load_cache_array(files["layer_overlap"], dtype=np.complex128, mmap=mmap)
        layer_diagonal_overlaps[shift] = _load_cache_array(
            files["layer_diagonal_overlap"], dtype=np.complex128, mmap=mmap
        )
        hartree_layer_coulomb[shift] = _load_cache_array(files["hartree_layer_coulomb"], dtype=float, mmap=mmap)
        fock_layer_coulomb[shift] = _load_cache_array(files["fock_layer_coulomb"], dtype=float, mmap=mmap)
    return RLGhBNLayerOverlapBlockSet(
        shifts=shifts,
        gvecs=np.asarray(gvecs, dtype=np.complex128),
        layer_overlaps=layer_overlaps,
        layer_diagonal_overlaps=layer_diagonal_overlaps,
        hartree_layer_coulomb=hartree_layer_coulomb,
        fock_layer_coulomb=fock_layer_coulomb,
    )


def load_or_build_layer_overlap_blocks(
    basis_data: RLGhBNProjectedBasisData,
    *,
    cache_dir: Path,
    cache_policy: str = "reuse",
    basis_cache_key: str | None = None,
    shifts: tuple[tuple[int, int], ...] | None = None,
) -> RLGhBNCacheResult:
    policy = _check_policy(cache_policy)
    resolved_shifts = shifts
    extra = {
        "basis_cache_key": basis_cache_key or "",
        "mesh_size": int(basis_data.mesh_size),
        "screened_u_mev": float(basis_data.basis_model.params.displacement_field_mev),
        "physical_v_mev": float(basis_data.model.params.displacement_field_mev),
        "active_band_indices": [int(value) for value in basis_data.active_band_indices],
        "valleys": [int(value) for value in basis_data.valleys],
        "frac_shift": np.asarray(basis_data.k_grid_frac[0], dtype=float).tolist()
        if np.asarray(basis_data.k_grid_frac).size
        else [0.0, 0.0],
        "shifts": None if resolved_shifts is None else [[int(m), int(n)] for m, n in resolved_shifts],
        "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
        "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    }
    key = rlg_hbn_cache_key("overlap", basis_data.model, basis_data.interaction, extra)
    manifest = _manifest_for("overlap", basis_data.model, basis_data.interaction, extra=extra)
    path = _cache_path(cache_dir, "overlap", key)
    if policy == "reuse" and _manifest_matches(path, manifest):
        try:
            blocks = load_layer_overlap_blocks_cache(cache_dir, key)
        except Exception:
            pass
        else:
            return RLGhBNCacheResult(value=blocks, key=key, path=path, hit=True, manifest=manifest)
    blocks = build_rlg_hbn_layer_overlap_blocks(basis_data, shifts=resolved_shifts)
    if policy != "off":
        path = save_layer_overlap_blocks_cache(cache_dir, key, blocks, manifest)
    return RLGhBNCacheResult(value=blocks, key=key, path=None if policy == "off" else path, hit=False, manifest=manifest)


def hf_ground_state_archive_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_cache_key(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
    *,
    source_archive: Path,
    path_payload: dict[str, object],
    chunk_size: int,
    beta: float,
    spin_index: int,
    panel: str,
) -> tuple[str, dict[str, object]]:
    extra = {
        "source_hf_ground_state_archive": str(Path(source_archive).resolve()),
        "source_hf_ground_state_archive_sha256": hf_ground_state_archive_hash(source_archive),
        "path": path_payload,
        "chunk_size": int(chunk_size),
        "beta": float(beta),
        "spin_index": int(spin_index),
        "source_panel": str(panel),
        "basis_periodic_gauge": RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        "basis_periodic_gauge_padding": int(RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING),
        "form_factor_convention": RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    }
    key = rlg_hbn_cache_key("path_bands", model, interaction, extra)
    return key, _manifest_for("path_bands", model, interaction, extra=extra)


def load_path_band_cache(cache_dir: Path, key: str) -> dict[str, object]:
    path = _cache_path(cache_dir, "path_bands", key)
    required = (
        "manifest.json",
        "path_hamiltonian.npy",
        "path_energies.npy",
        "path_kvec_complex_pairs.npy",
        "kdist.npy",
        "labels.json",
        "hf_bands_path.npz",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RLGhBNCacheMiss(f"Path-band cache {path} is missing {missing}")
    labels_payload = _read_json(path / "labels.json")
    archive = np.load(path / "hf_bands_path.npz")
    return {
        "path": path,
        "manifest": _read_json(path / "manifest.json"),
        "hamiltonian": np.asarray(np.load(path / "path_hamiltonian.npy"), dtype=np.complex128),
        "energies": np.asarray(np.load(path / "path_energies.npy"), dtype=float),
        "kvec": _complex_from_pairs(np.load(path / "path_kvec_complex_pairs.npy")),
        "kdist": np.asarray(np.load(path / "kdist.npy"), dtype=float),
        "labels": labels_payload,
        "hf_bands_path": {name: np.asarray(archive[name]) for name in archive.files},
    }


def save_path_band_cache(
    cache_dir: Path,
    key: str,
    manifest: dict[str, object],
    *,
    path_hamiltonian: np.ndarray,
    path_energies: np.ndarray,
    path_kvec: np.ndarray,
    kdist: np.ndarray,
    labels_payload: dict[str, object],
    hf_bands_payload: dict[str, object],
) -> Path:
    path = _cache_path(cache_dir, "path_bands", key)
    tmp_path = _prepare_write_dir(path)
    np.save(tmp_path / "path_hamiltonian.npy", np.asarray(path_hamiltonian, dtype=np.complex128))
    np.save(tmp_path / "path_energies.npy", np.asarray(path_energies, dtype=float))
    np.save(tmp_path / "path_kvec_complex_pairs.npy", _complex_pairs(path_kvec))
    np.save(tmp_path / "kdist.npy", np.asarray(kdist, dtype=float))
    _write_json(tmp_path / "labels.json", labels_payload)
    np.savez(tmp_path / "hf_bands_path.npz", **hf_bands_payload)
    _write_json(tmp_path / "manifest.json", manifest)
    _finish_write_dir(tmp_path, path)
    return path


def update_cache_manifest_file(
    output_manifest_path: Path,
    *,
    cache_dir: Path,
    kind: str,
    key: str,
    hit: bool,
    path: Path | None,
    panel: str | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    if output_manifest_path.exists():
        payload = _read_json(output_manifest_path)
    else:
        payload = {"cache_dir": str(Path(cache_dir).resolve()), "entries": [], "summary": {}}
    entries = payload.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
        payload["entries"] = entries
    entry = {
        "kind": str(kind),
        "key": str(key),
        "hit": bool(hit),
        "path": "" if path is None else str(path),
        "panel": "" if panel is None else str(panel),
    }
    if extra:
        entry.update(extra)
    entries.append(entry)
    summary = payload.setdefault("summary", {})
    if not isinstance(summary, dict):
        summary = {}
        payload["summary"] = summary
    kind_summary = summary.setdefault(str(kind), {"hit": 0, "miss": 0})
    if not isinstance(kind_summary, dict):
        kind_summary = {"hit": 0, "miss": 0}
        summary[str(kind)] = kind_summary
    key_name = "hit" if hit else "miss"
    kind_summary[key_name] = int(kind_summary.get(key_name, 0)) + 1
    _write_json(output_manifest_path, payload)


__all__ = [
    "CACHE_POLICY_CHOICES",
    "RLGhBNCacheMiss",
    "RLGhBNCacheResult",
    "hf_ground_state_archive_hash",
    "load_layer_overlap_blocks_cache",
    "load_or_build_layer_overlap_blocks",
    "load_or_build_projected_basis",
    "load_or_solve_screening",
    "load_path_band_cache",
    "load_projected_basis_cache",
    "path_cache_key",
    "rlg_hbn_cache_key",
    "save_layer_overlap_blocks_cache",
    "save_path_band_cache",
    "save_projected_basis_cache",
    "update_cache_manifest_file",
]
