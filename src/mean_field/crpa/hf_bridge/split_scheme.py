from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from .density import physical_projector_from_delta


_CRPA_SPLIT_MODE_ALIASES = {
    "remote_projector": "remote_projector",
    "remote_bare_projector": "remote_projector",
    "remote+p": "remote_projector",
    "legacy_remote_projector": "remote_projector",
    "production": "active_cnp_fock_reference_projector",
    "remote_delta": "remote_delta",
    "remote_bare_delta": "remote_delta",
    "remote+d": "remote_delta",
    "remote_hartree_delta_fock_projector": "remote_hartree_delta_fock_projector",
    "remote_hartree_d_fock_p": "remote_hartree_delta_fock_projector",
    "remote_hd_fp": "remote_hartree_delta_fock_projector",
    "remote_fock_projector": "remote_fock_projector",
    "remote_fock_p": "remote_fock_projector",
    "remote_fock_only_projector": "remote_fock_projector",
    "remote_fp": "remote_fock_projector",
    "remote_fock_active_cnp_fock_reference_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fock_active_cnp_fock_ref_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fock_cnp_fock_reference_projector": "remote_fock_active_cnp_fock_reference_projector",
    "remote_fp_cnp_fp": "remote_fock_active_cnp_fock_reference_projector",
    "active_cnp_reference_projector": "active_cnp_reference_projector",
    "active_cnp_ref_projector": "active_cnp_reference_projector",
    "cnp_reference_projector": "active_cnp_reference_projector",
    "active_cnp_fock_reference_projector": "active_cnp_fock_reference_projector",
    "active_cnp_fock_ref_projector": "active_cnp_fock_reference_projector",
    "cnp_fock_reference_projector": "active_cnp_fock_reference_projector",
    "minus_active_cnp_fock_projector": "active_cnp_fock_reference_projector",
    "active_cnp_fock_reference_hartree_delta_projector": "active_cnp_fock_reference_hartree_delta_projector",
    "active_cnp_fock_ref_hartree_delta": "active_cnp_fock_reference_hartree_delta_projector",
    "active_cnp_fock_ref_hd_fp": "active_cnp_fock_reference_hartree_delta_projector",
    "cnp_fock_ref_hd_fp": "active_cnp_fock_reference_hartree_delta_projector",
    "no_remote_projector": "no_remote_projector",
    "projector_only": "no_remote_projector",
    "p_only": "no_remote_projector",
    "no_remote_delta": "no_remote_delta",
    "delta_only": "no_remote_delta",
    "d_only": "no_remote_delta",
}


def crpa_split_mode() -> str:
    """Return the diagnostic cRPA split mode.

    The production convention is the flat-subspace cRPA HF split validated
    against Zhang's projected Eq. (17)-(20): build the active cRPA self-energy
    from the physical projector P = D + 0.5 I and subtract the CNP lower-flat
    Fock reference.  The old bare remote-projector split remains available as
    an explicit diagnostic/legacy mode.
    """

    raw = os.environ.get("MEAN_FIELD_CRPA_SPLIT_MODE", "active_cnp_fock_reference_projector")
    normalized = raw.strip().lower().replace("-", "_")
    try:
        return _CRPA_SPLIT_MODE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_CRPA_SPLIT_MODE_ALIASES.values())))
        raise ValueError(
            f"Unsupported MEAN_FIELD_CRPA_SPLIT_MODE={raw!r}; allowed canonical modes: {allowed}"
        ) from exc


def crpa_split_uses_remote_bare(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_projector",
        "remote_delta",
        "remote_hartree_delta_fock_projector",
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_projector(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_projector",
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
        "active_cnp_reference_projector",
        "active_cnp_fock_reference_projector",
        "no_remote_projector",
    }


def crpa_split_uses_hartree_delta_fock_projector(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_hartree_delta_fock_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
    }


def crpa_split_uses_remote_fock_only(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "remote_fock_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_active_cnp_reference(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "active_cnp_reference_projector",
        "active_cnp_fock_reference_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_split_uses_active_cnp_fock_only(mode: str | None = None) -> bool:
    resolved = crpa_split_mode() if mode is None else str(mode)
    return resolved in {
        "active_cnp_fock_reference_projector",
        "active_cnp_fock_reference_hartree_delta_projector",
        "remote_fock_active_cnp_fock_reference_projector",
    }


def crpa_remote_bare_scale() -> float:
    """Diagnostic scale factor for the bare remote/reference cRPA split term."""

    raw = os.environ.get("MEAN_FIELD_CRPA_REMOTE_BARE_SCALE", "1.0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Unsupported MEAN_FIELD_CRPA_REMOTE_BARE_SCALE={raw!r}; expected a float") from exc
    if not np.isfinite(value):
        raise ValueError(f"Unsupported MEAN_FIELD_CRPA_REMOTE_BARE_SCALE={raw!r}; expected a finite float")
    return value


def crpa_active_density_from_delta(density_delta: np.ndarray, mode: str | None = None) -> np.ndarray:
    resolved = crpa_split_mode() if mode is None else str(mode)
    if crpa_split_uses_projector(resolved):
        return physical_projector_from_delta(density_delta)
    return np.asarray(density_delta, dtype=np.complex128)


def select_remote_reference_components(
    hartree: np.ndarray,
    fock: np.ndarray,
    mode: str | None = None,
) -> np.ndarray:
    """Select the fixed remote/reference one-body term for a diagnostic split."""

    resolved = crpa_split_mode() if mode is None else str(mode)
    if not crpa_split_uses_remote_bare(resolved):
        return np.zeros_like(np.asarray(hartree, dtype=np.complex128))
    hartree_arr = np.asarray(hartree, dtype=np.complex128)
    fock_arr = np.asarray(fock, dtype=np.complex128)
    if hartree_arr.shape != fock_arr.shape:
        raise ValueError(f"Expected matching Hartree/Fock shapes, got {hartree_arr.shape} and {fock_arr.shape}")
    if crpa_split_uses_remote_fock_only(resolved):
        return fock_arr.copy()
    return hartree_arr + fock_arr


def select_active_cnp_reference_components(
    hartree: np.ndarray,
    fock: np.ndarray,
    mode: str | None = None,
) -> np.ndarray:
    """Return the fixed CNP active-reference subtraction for a split mode."""

    resolved = crpa_split_mode() if mode is None else str(mode)
    if not crpa_split_uses_active_cnp_reference(resolved):
        return np.zeros_like(np.asarray(hartree, dtype=np.complex128))
    hartree_arr = np.asarray(hartree, dtype=np.complex128)
    fock_arr = np.asarray(fock, dtype=np.complex128)
    if hartree_arr.shape != fock_arr.shape:
        raise ValueError(f"Expected matching Hartree/Fock shapes, got {hartree_arr.shape} and {fock_arr.shape}")
    if crpa_split_uses_active_cnp_fock_only(resolved):
        return -fock_arr.copy()
    return -(hartree_arr + fock_arr)


__all__ = [name for name, value in globals().items() if callable(value) and getattr(value, '__module__', None) == __name__ and not name.startswith('_')]
