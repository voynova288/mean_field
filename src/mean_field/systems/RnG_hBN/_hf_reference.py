from __future__ import annotations

from ._hf_shared import *  # noqa: F401,F403

def active_band_indices_for_interaction(
    model: RLGhBNModel,
    interaction: RLGhBNInteractionParams,
) -> tuple[int, ...]:
    n_valence = int(interaction.active_valence_bands)
    n_conduction = int(interaction.active_conduction_bands)
    center = valence_band_count(model.lattice, model.params)
    start = center - n_valence
    stop = center + n_conduction
    if start < 0 or stop > model.matrix_dim:
        raise ValueError(
            "Active RLG/hBN band window is outside the single-particle spectrum: "
            f"requested {n_valence} valence and {n_conduction} conduction bands around "
            f"center={center}, but matrix_dim={model.matrix_dim}."
        )
    return tuple(range(start, stop))


def rlg_hbn_average_reference_density(nt: int, nk: int) -> np.ndarray:
    return core_average_reference_density(int(nt), int(nk), value=0.5)


def _infer_rlg_hbn_band_count(nt: int, *, n_spin: int = 2, n_eta: int = 2) -> int:
    n_flavor = int(n_spin) * int(n_eta)
    if int(nt) % n_flavor != 0:
        raise ValueError(f"Projected dimension nt={nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}")
    n_band = int(nt) // n_flavor
    if n_band <= 0:
        raise ValueError(f"Projected band count must be positive, got {n_band}")
    return n_band


def _rlg_hbn_reference_density_diagonal(
    nt: int,
    nk: int,
    *,
    scheme: str,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    if scheme not in VALID_INTERACTION_SCHEMES:
        raise ValueError(f"scheme must be one of {VALID_INTERACTION_SCHEMES}, got {scheme!r}")
    n_band = _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    n_valence = int(active_valence_bands)
    if n_valence < 0 or n_valence > n_band:
        raise ValueError(f"active_valence_bands must lie in [0, {n_band}], got {active_valence_bands}")

    diagonal = np.zeros((int(nt), int(nk)), dtype=float)
    idx = np.arange(int(nt), dtype=int).reshape((int(n_spin), int(n_eta), n_band), order="F")
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            for iband in range(n_band):
                if scheme == "average":
                    value = 0.5
                else:
                    value = 1.0 if iband < n_valence else 0.0
                diagonal[int(idx[ispin, ieta, iband]), :] = value
    return diagonal


def rlg_hbn_reference_density(
    nt: int,
    nk: int,
    *,
    scheme: str = "average",
    active_valence_bands: int = 0,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    if scheme == "average":
        return rlg_hbn_average_reference_density(nt, nk)
    diagonal = _rlg_hbn_reference_density_diagonal(
        nt,
        nk,
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )
    reference = np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128)
    rows = np.arange(int(nt), dtype=int)
    for ik in range(int(nk)):
        reference[rows, rows, ik] = diagonal[:, ik]
    return reference


def average_scheme_density_delta(occupation_density: np.ndarray) -> np.ndarray:
    density = np.asarray(occupation_density, dtype=np.complex128)
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected occupation_density shape (nt, nt, nk), got {density.shape}")
    return density - rlg_hbn_average_reference_density(density.shape[0], density.shape[2])


def rlg_hbn_density_delta(
    occupation_density: np.ndarray,
    *,
    scheme: str,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> np.ndarray:
    density = np.asarray(occupation_density, dtype=np.complex128)
    if density.ndim != 3 or density.shape[0] != density.shape[1]:
        raise ValueError(f"Expected occupation_density shape (nt, nt, nk), got {density.shape}")
    return density - rlg_hbn_reference_density(
        density.shape[0],
        density.shape[2],
        scheme=scheme,
        active_valence_bands=active_valence_bands,
        n_spin=n_spin,
        n_eta=n_eta,
    )


def rlg_hbn_projector_from_density(density_delta: np.ndarray, reference_density: np.ndarray) -> np.ndarray:
    density_delta = np.asarray(density_delta, dtype=np.complex128)
    reference_density = np.asarray(reference_density, dtype=np.complex128)
    if density_delta.shape != reference_density.shape:
        raise ValueError(f"Expected reference density shape {density_delta.shape}, got {reference_density.shape}")
    return density_delta + reference_density


def rlg_hbn_occupied_state_count(
    nu: float,
    nt: int,
    nk: int,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    raw = (float(int(n_spin) * int(n_eta) * int(active_valence_bands)) + float(nu)) * int(nk)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer occupied-state count {raw}")
    if rounded < 0 or rounded > int(nt) * int(nk):
        raise ValueError(f"Filling nu={nu} gives occupied-state count {rounded} outside [0, {int(nt) * int(nk)}]")
    return rounded


def rlg_hbn_occupied_bands_per_k(
    nu: float,
    nt: int,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> int:
    _infer_rlg_hbn_band_count(nt, n_spin=n_spin, n_eta=n_eta)
    raw = float(int(n_spin) * int(n_eta) * int(active_valence_bands)) + float(nu)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"Filling nu={nu} gives non-integer per-k occupation {raw}")
    if rounded < 0 or rounded > int(nt):
        raise ValueError(f"Filling nu={nu} gives per-k occupation {rounded} outside [0, {int(nt)}]")
    return rounded


def rlg_hbn_filling_from_density(
    density_delta: np.ndarray,
    reference_density: np.ndarray,
    *,
    active_valence_bands: int,
    n_spin: int = 2,
    n_eta: int = 2,
) -> float:
    projector = rlg_hbn_projector_from_density(density_delta, reference_density)
    total_particles = float(np.trace(projector, axis1=0, axis2=1).real.sum())
    particles_per_k = total_particles / float(projector.shape[2])
    return float(particles_per_k - float(int(n_spin) * int(n_eta) * int(active_valence_bands)))

__all__ = [name for name in globals() if not name.startswith('__')]
