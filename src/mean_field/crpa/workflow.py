from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from time import strftime

import numpy as np

from ..systems.tbg.params import TBGParameters
from .band_classifier import BandClassification, classify_flat_bands
from .bm import AllBandBMSolution, solve_all_band_bm_model
from .coulomb import CRPACoulombParams, coulomb_potential_table_mev
from .dielectric import compute_dielectric
from .form_factor import (
    LEGACY_ZERO_FILL_TEST_MODE,
    PRODUCTION_FORM_FACTOR_MODE,
    normalize_form_factor_mode,
)
from .grid import CRPAKGrid, build_q_shift_table, build_uniform_crpa_grid, q_shift_vectors
from .susceptibility import compute_constrained_chi0


@dataclass(frozen=True)
class CRPAResult:
    theta_deg: float
    lk: int
    lg: int
    q_lg: int
    bands_per_valley: int | None
    eta_mev: float
    coulomb_params: CRPACoulombParams
    q_indices: np.ndarray
    q_tilde: np.ndarray
    q_shifts: np.ndarray
    q_vectors: np.ndarray
    physical_q_vectors: np.ndarray
    chi0: np.ndarray
    dielectric_matrix: np.ndarray
    epsilon_inv: np.ndarray
    screened_v: np.ndarray
    effective_epsilon: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)


def _normalize_q_indices(q_indices: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] | None, grid: CRPAKGrid) -> np.ndarray:
    if q_indices is None:
        coords = np.asarray([grid.unravel_index(iq) for iq in range(grid.nk)], dtype=int)
        return coords
    coords: list[tuple[int, int]] = []
    for item in q_indices:
        if isinstance(item, tuple):
            coords.append((int(item[0]) % grid.lk, int(item[1]) % grid.lk))
        else:
            coords.append(grid.unravel_index(int(item)))
    return np.asarray(coords, dtype=int)


def resolve_crpa_runtime_convention(
    *,
    periodic_g_grid: bool,
    form_factor_mode: str,
    allow_legacy_zero_fill_test: bool = False,
) -> str:
    """Return the normalized form-factor mode after enforcing production gates."""

    resolved_mode = normalize_form_factor_mode(form_factor_mode)
    if bool(periodic_g_grid) and resolved_mode == PRODUCTION_FORM_FACTOR_MODE:
        return resolved_mode
    if (
        bool(allow_legacy_zero_fill_test)
        and not bool(periodic_g_grid)
        and resolved_mode == LEGACY_ZERO_FILL_TEST_MODE
    ):
        return resolved_mode
    raise ValueError(
        "Production cRPA requires periodic_g_grid=True and form_factor_mode='hf_periodic'. "
        "The zhang_zero_fill/periodic_g_grid=False convention is retained only for explicit "
        "legacy regression tests via allow_legacy_zero_fill_test=True."
    )


def compute_crpa(
    params: TBGParameters,
    *,
    theta_deg: float,
    lk: int = 6,
    lg: int = 3,
    q_lg: int = 3,
    bands_per_valley: int | None = None,
    q_indices: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] | None = None,
    coulomb_params: CRPACoulombParams | None = None,
    eta_mev: float = 1.0,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = True,
    form_factor_mode: str = PRODUCTION_FORM_FACTOR_MODE,
    allow_legacy_zero_fill_test: bool = False,
    occupation_mode: str = "cnp_index",
    flat_method: str = "center",
) -> CRPAResult:
    resolved_form_factor_mode = resolve_crpa_runtime_convention(
        periodic_g_grid=bool(periodic_g_grid),
        form_factor_mode=form_factor_mode,
        allow_legacy_zero_fill_test=bool(allow_legacy_zero_fill_test),
    )
    coulomb = CRPACoulombParams() if coulomb_params is None else coulomb_params
    grid = build_uniform_crpa_grid(params, lk)
    solution = solve_all_band_bm_model(
        params,
        grid.kvec,
        lg=lg,
        bands_per_valley=bands_per_valley,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
    )
    classification = classify_flat_bands(solution.spectrum, method=flat_method)
    return compute_crpa_from_solution(
        solution,
        classification,
        grid,
        theta_deg=theta_deg,
        q_lg=q_lg,
        bands_per_valley=bands_per_valley,
        q_indices=q_indices,
        coulomb_params=coulomb,
        eta_mev=eta_mev,
        form_factor_mode=resolved_form_factor_mode,
        allow_legacy_zero_fill_test=bool(allow_legacy_zero_fill_test),
        occupation_mode=occupation_mode,
        flat_method=flat_method,
        metadata={
            "vf": float(params.vf),
            "w0": float(params.w0),
            "w1": float(params.w1),
            "sigma_rotation": bool(sigma_rotation),
            "periodic_g_grid": bool(periodic_g_grid),
            "form_factor_mode": resolved_form_factor_mode,
            "legacy_zero_fill_test": bool(
                allow_legacy_zero_fill_test and resolved_form_factor_mode == LEGACY_ZERO_FILL_TEST_MODE
            ),
            "occupation_mode": str(occupation_mode),
            "flat_band_classifier": str(flat_method),
            "k_grid_kind": "uniform_crpa",
        },
    )


def compute_crpa_from_solution(
    solution: AllBandBMSolution,
    classification: BandClassification,
    grid: CRPAKGrid,
    *,
    theta_deg: float,
    q_lg: int,
    bands_per_valley: int | None,
    q_indices: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] | None,
    coulomb_params: CRPACoulombParams,
    eta_mev: float,
    form_factor_mode: str = PRODUCTION_FORM_FACTOR_MODE,
    allow_legacy_zero_fill_test: bool = False,
    occupation_mode: str = "cnp_index",
    flat_method: str = "center",
    metadata: dict[str, object] | None = None,
) -> CRPAResult:
    resolved_form_factor_mode = resolve_crpa_runtime_convention(
        periodic_g_grid=bool(solution.periodic_g_grid),
        form_factor_mode=form_factor_mode,
        allow_legacy_zero_fill_test=bool(allow_legacy_zero_fill_test),
    )
    q_shift_labels, q_shift_coords = build_q_shift_table(q_lg)
    q_vecs = q_shift_vectors(solution.params, q_shift_labels)
    q_coords = _normalize_q_indices(q_indices, grid)

    chi0_list: list[np.ndarray] = []
    eps_list: list[np.ndarray] = []
    eps_inv_list: list[np.ndarray] = []
    screened_list: list[np.ndarray] = []
    eff_list: list[np.ndarray] = []
    q_tilde_list: list[complex] = []
    physical_q_list: list[np.ndarray] = []

    for qi, qj in q_coords:
        q_tilde = grid.centered_q_vector((int(qi), int(qj)))
        chi0 = compute_constrained_chi0(
            solution,
            classification,
            grid,
            (int(qi), int(qj)),
            q_shift_labels,
            eta_mev=eta_mev,
            form_factor_mode=resolved_form_factor_mode,
            occupation_mode=occupation_mode,
        )
        v_q = coulomb_potential_table_mev(q_tilde, q_vecs, solution.params, coulomb_params)
        dielectric = compute_dielectric(chi0, v_q)
        chi0_list.append(chi0)
        eps_list.append(dielectric.epsilon)
        eps_inv_list.append(dielectric.epsilon_inv)
        screened_list.append(dielectric.screened_v)
        eff_list.append(dielectric.effective_epsilon)
        q_tilde_list.append(q_tilde)
        physical_q_list.append(q_tilde + q_vecs)

    resolved_metadata: dict[str, object] = {
        "vf": float(solution.params.vf),
        "w0": float(solution.params.w0),
        "w1": float(solution.params.w1),
        "sigma_rotation": bool(solution.sigma_rotation),
        "periodic_g_grid": bool(solution.periodic_g_grid),
        "form_factor_mode": resolved_form_factor_mode,
        "legacy_zero_fill_test": bool(
            allow_legacy_zero_fill_test and resolved_form_factor_mode == LEGACY_ZERO_FILL_TEST_MODE
        ),
        "occupation_mode": str(occupation_mode),
        "flat_band_classifier": str(flat_method),
        "k_grid_kind": str(solution.k_grid_kind),
        "n_valleys_explicit": int(solution.n_eta),
        "spin_degeneracy": 2.0,
        "spin_degeneracy_handling": "implicit factor 2 in chi0 prefactor",
        "valley_degeneracy_handling": "explicit valley sum over BM solution valleys",
        "temperature_mev": 0.0,
        "fermi_level_mev": 0.0,
        "flat_band_count_per_valley": 2,
        "band_start": int(solution.band_start),
        "band_stop": int(solution.band_stop),
        "basis_dimension_per_valley": int(solution.basis_dimension),
    }
    if metadata:
        resolved_metadata.update(metadata)
    resolved_metadata.update(
        {
            "periodic_g_grid": bool(solution.periodic_g_grid),
            "form_factor_mode": resolved_form_factor_mode,
            "legacy_zero_fill_test": bool(
                allow_legacy_zero_fill_test and resolved_form_factor_mode == LEGACY_ZERO_FILL_TEST_MODE
            ),
        }
    )

    return CRPAResult(
        theta_deg=float(theta_deg),
        lk=int(grid.lk),
        lg=int(solution.lg),
        q_lg=int(q_lg),
        bands_per_valley=None if bands_per_valley is None else int(bands_per_valley),
        eta_mev=float(eta_mev),
        coulomb_params=coulomb_params,
        q_indices=q_coords,
        q_tilde=np.asarray(q_tilde_list, dtype=np.complex128),
        q_shifts=q_shift_coords,
        q_vectors=q_vecs,
        physical_q_vectors=np.asarray(physical_q_list, dtype=np.complex128),
        chi0=np.asarray(chi0_list, dtype=np.complex128),
        dielectric_matrix=np.asarray(eps_list, dtype=np.complex128),
        epsilon_inv=np.asarray(eps_inv_list, dtype=np.complex128),
        screened_v=np.asarray(screened_list, dtype=np.complex128),
        effective_epsilon=np.asarray(eff_list, dtype=float),
        metadata=resolved_metadata,
    )


def default_crpa_output_dir(root: Path | str = "outputs") -> Path:
    return Path(root) / f"crpa_{strftime('%Y%m%d_%H%M%S')}"


def _complex_as_columns(values: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(values)
    return {"real": arr.real, "imag": arr.imag}


def write_crpa_outputs(result: CRPAResult, output_dir: Path | str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    params_json = {
        "theta_deg": result.theta_deg,
        "lk": result.lk,
        "lg": result.lg,
        "q_lg": result.q_lg,
        "bands_per_valley": result.bands_per_valley,
        "eta_mev": result.eta_mev,
        "coulomb_params": asdict(result.coulomb_params),
        "q_point_count": int(result.q_indices.shape[0]),
        "q_shift_count": int(result.q_shifts.shape[0]),
        "metadata": result.metadata,
    }
    (out / "crpa_params.json").write_text(json.dumps(params_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "chi0_q.npz",
        chi0=result.chi0,
        q_indices=result.q_indices,
        q_tilde_real=result.q_tilde.real,
        q_tilde_imag=result.q_tilde.imag,
        q_shifts=result.q_shifts,
    )
    np.savez_compressed(
        out / "dielectric_matrix.npz",
        epsilon=result.dielectric_matrix,
        epsilon_inv=result.epsilon_inv,
        q_indices=result.q_indices,
        q_shifts=result.q_shifts,
    )
    np.savez_compressed(
        out / "effective_epsilon.npz",
        effective_epsilon=result.effective_epsilon,
        epsilon_times_bn=result.effective_epsilon * float(result.coulomb_params.epsilon_bn),
        q_abs=np.abs(result.physical_q_vectors),
        q_abs_nm_inv=np.abs(result.physical_q_vectors) / (float(result.coulomb_params.graphene_lattice_angstrom) / 10.0),
        q_real=result.physical_q_vectors.real,
        q_imag=result.physical_q_vectors.imag,
        q_indices=result.q_indices,
        q_shifts=result.q_shifts,
    )
    np.savez_compressed(
        out / "screened_coulomb.npz",
        screened_v=result.screened_v,
        effective_epsilon=result.effective_epsilon,
        q_indices=result.q_indices,
        q_shifts=result.q_shifts,
        q_abs_nm_inv=np.abs(result.physical_q_vectors) / (float(result.coulomb_params.graphene_lattice_angstrom) / 10.0),
        q_vectors_real=result.physical_q_vectors.real,
        q_vectors_imag=result.physical_q_vectors.imag,
    )
    return out


def load_crpa_result(output_dir: Path | str) -> CRPAResult:
    """Load a cRPA artifact directory written by ``write_crpa_outputs``."""

    path = Path(output_dir)
    params_json = json.loads((path / "crpa_params.json").read_text(encoding="utf-8"))
    coulomb = CRPACoulombParams(**params_json["coulomb_params"])

    with np.load(path / "chi0_q.npz") as chi0_npz:
        chi0 = np.asarray(chi0_npz["chi0"], dtype=np.complex128)
        q_indices = np.asarray(chi0_npz["q_indices"], dtype=int)
        q_tilde = np.asarray(chi0_npz["q_tilde_real"], dtype=float) + 1j * np.asarray(
            chi0_npz["q_tilde_imag"], dtype=float
        )
        q_shifts = np.asarray(chi0_npz["q_shifts"], dtype=int)

    with np.load(path / "dielectric_matrix.npz") as dielectric_npz:
        dielectric_matrix = np.asarray(dielectric_npz["epsilon"], dtype=np.complex128)
        epsilon_inv = np.asarray(dielectric_npz["epsilon_inv"], dtype=np.complex128)

    with np.load(path / "screened_coulomb.npz") as screened_npz:
        screened_v = np.asarray(screened_npz["screened_v"], dtype=np.complex128)
        effective_epsilon = np.asarray(screened_npz["effective_epsilon"], dtype=float)
        physical_q_vectors = np.asarray(screened_npz["q_vectors_real"], dtype=float) + 1j * np.asarray(
            screened_npz["q_vectors_imag"], dtype=float
        )

    if physical_q_vectors.ndim != 2:
        raise ValueError(f"Expected physical_q_vectors to have rank 2, got {physical_q_vectors.shape}")
    if q_tilde.size == 0:
        q_vectors = np.zeros((q_shifts.shape[0],), dtype=np.complex128)
    else:
        q_vectors = np.asarray(physical_q_vectors[0] - q_tilde[0], dtype=np.complex128)

    return CRPAResult(
        theta_deg=float(params_json["theta_deg"]),
        lk=int(params_json["lk"]),
        lg=int(params_json["lg"]),
        q_lg=int(params_json["q_lg"]),
        bands_per_valley=None
        if params_json.get("bands_per_valley") is None
        else int(params_json["bands_per_valley"]),
        eta_mev=float(params_json["eta_mev"]),
        coulomb_params=coulomb,
        q_indices=q_indices,
        q_tilde=q_tilde,
        q_shifts=q_shifts,
        q_vectors=q_vectors,
        physical_q_vectors=physical_q_vectors,
        chi0=chi0,
        dielectric_matrix=dielectric_matrix,
        epsilon_inv=epsilon_inv,
        screened_v=screened_v,
        effective_epsilon=effective_epsilon,
        metadata=dict(params_json.get("metadata", {})),
    )
