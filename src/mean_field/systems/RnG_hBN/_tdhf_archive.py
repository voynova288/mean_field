from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_types import *  # noqa: F401,F403

def load_rlg_hbn_tdhf_run_from_archive(
    archive_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    precision: float = 1.0e-6,
) -> RLGhBNHartreeFockRun:
    """Load a saved RLG/hBN HF archive as a TDHF-ready run object.

    Archives written by ``run_rlg_hbn_paper_hf`` store final HF matrices plus
    cache keys for the projected basis and layer-overlap blocks.  This loader
    restores those cached objects and attaches the saved HF state without
    rerunning SCF.  It is intended for TDHF postprocessing jobs.
    """

    path = Path(archive_path).expanduser().resolve()
    with np.load(path) as data:
        archive = {key: data[key] for key in data.files}
    if _archive_bool(archive, "zero_literal_q0_fock", default=False):
        raise ValueError(
            "HF archive was generated with MEAN_FIELD_RLG_HBN_ZERO_LITERAL_Q0_FOCK=1; "
            "TDHF postprocessing requires the physical q=0 Fock convention."
        )
    resolved_summary_path = Path(summary_path) if summary_path is not None else path.with_name("hf_run_summary.json")
    summary: dict[str, object] = {}
    if resolved_summary_path.exists():
        summary = json.loads(resolved_summary_path.read_text(encoding="utf-8"))

    resolved_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    if resolved_cache_dir is None:
        cache_dir_text = _archive_string(archive, "cache_dir") or str(summary.get("cache_dir", ""))
        if not cache_dir_text:
            raise ValueError("HF archive does not record cache_dir; pass cache_dir explicitly")
        resolved_cache_dir = Path(cache_dir_text).expanduser().resolve()

    basis_key = _archive_string(archive, "cache_key_basis") or str(summary.get("cache_key_basis", ""))
    overlap_key = _archive_string(archive, "cache_key_overlap") or str(summary.get("cache_key_overlap", ""))
    if not basis_key or not overlap_key:
        raise ValueError("HF archive must contain cache_key_basis and cache_key_overlap for TDHF postprocessing")

    basis_data = load_projected_basis_cache(resolved_cache_dir, basis_key)
    overlap_blocks = load_layer_overlap_blocks_cache(resolved_cache_dir, overlap_key)

    nu = _archive_scalar_float(archive, "nu", default=float(summary.get("filling", 1.0)))
    occupation_counts_array = np.asarray(archive.get("occupation_counts", np.asarray([], dtype=int)), dtype=int).reshape(-1)
    occupation_counts = None if occupation_counts_array.size == 0 else tuple(int(v) for v in occupation_counts_array)
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=nu,
        precision=float(precision),
        occupation_counts=occupation_counts,
    )
    _assign_archive_array(state, "density", archive, "density")
    _assign_archive_array(state, "hamiltonian", archive, "hamiltonian")
    _assign_archive_array(state, "h0", archive, "h0")
    _assign_archive_array(state, "energies", archive, "energies_mev")
    if "reference_density" in archive:
        _assign_archive_array(state, "reference_density", archive, "reference_density")
    state.mu = _archive_scalar_float(archive, "mu_mev", default=float("nan"))
    state.diagnostics.update(
        {
            "hf_energy": float(summary.get("final_energy_mev", np.nan)),
            "hf_gap": float(summary.get("hf_gap_mev", np.nan)),
            "filling": float(summary.get("filling", nu)),
            "projector_idempotency_residual": float(summary.get("projector_idempotency_residual", np.nan)),
            "density_hermitian_residual": float(summary.get("density_hermitian_residual", np.nan)),
            "hamiltonian_hermitian_residual": float(summary.get("hamiltonian_hermitian_residual", np.nan)),
        }
    )

    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=np.asarray(archive.get("iter_energy_mev", np.asarray([], dtype=float)), dtype=float),
        iter_err=np.asarray(archive.get("iter_err", np.asarray([], dtype=float)), dtype=float),
        iter_oda=np.asarray(archive.get("iter_oda", np.asarray([], dtype=float)), dtype=float),
        init_mode=str(summary.get("init_mode", "archive")),
        seed=int(summary.get("seed", 0)),
        converged=bool(summary.get("converged", False)),
        exit_reason=str(summary.get("exit_reason", "loaded_archive")),
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


def _archive_string(archive: dict[str, np.ndarray], key: str) -> str:
    if key not in archive:
        return ""
    value = np.asarray(archive[key])
    if value.size == 0:
        return ""
    return str(value.reshape(-1)[0])


def _archive_scalar_float(archive: dict[str, np.ndarray], key: str, *, default: float) -> float:
    if key not in archive:
        return float(default)
    value = np.asarray(archive[key], dtype=float).reshape(-1)
    if value.size == 0:
        return float(default)
    return float(value[0])


def _archive_bool(archive: dict[str, np.ndarray], key: str, *, default: bool) -> bool:
    if key not in archive:
        return bool(default)
    value = np.asarray(archive[key]).reshape(-1)
    if value.size == 0:
        return bool(default)
    item = value[0]
    if isinstance(item, np.bool_ | bool):
        return bool(item)
    if isinstance(item, np.integer | int):
        return bool(int(item))
    return str(item).strip().lower() not in {"", "0", "false", "no", "off"}


def _assign_archive_array(
    state: RLGhBNHartreeFockState,
    attribute: str,
    archive: dict[str, np.ndarray],
    key: str,
) -> None:
    if key not in archive:
        raise ValueError(f"HF archive is missing required array {key!r}")
    value = np.asarray(archive[key])
    expected = np.asarray(getattr(state, attribute)).shape
    if value.shape != expected:
        raise ValueError(f"Archive array {key!r} has shape {value.shape}, expected {expected}")
    setattr(state, attribute, value.astype(np.asarray(getattr(state, attribute)).dtype, copy=True))

__all__ = [name for name in globals() if not name.startswith('__')]
