from __future__ import annotations

from ._hf_types import *  # noqa: F401,F403
from ._hf_reference import *  # noqa: F401,F403
from ._hf_initialization import *  # noqa: F401,F403
from ._hf_basis import *  # noqa: F401,F403

def build_htg_interaction_components(
    density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet,
    *,
    v0: float,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGInteractionComponents:
    hartree_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
    )
    fock_blocks = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        fock_screening=overlap_blocks.fock_screening,
    )
    hartree = build_projected_interaction_hamiltonian(
        density,
        hartree_blocks,
        v0=v0,
        beta=beta,
        use_numba=use_numba,
    )
    fock = build_projected_interaction_hamiltonian(
        density,
        fock_blocks,
        v0=v0,
        beta=beta,
        use_numba=use_numba,
    )
    total = hartree + fock
    hartree_eigs = np.zeros((hartree.shape[0], hartree.shape[2]), dtype=float)
    fock_eigs = np.zeros_like(hartree_eigs)
    for ik in range(hartree.shape[2]):
        hartree_eigs[:, ik] = np.linalg.eigvalsh(hartree[:, :, ik])
        fock_eigs[:, ik] = np.linalg.eigvalsh(fock[:, :, ik])
    return HTGInteractionComponents(
        hartree=hartree,
        fock=fock,
        total=total,
        hartree_eigenvalues=hartree_eigs,
        fock_eigenvalues=fock_eigs,
    )


def _hartree_only_blocks(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
    )


def _fock_only_blocks(overlap_blocks: HFOverlapBlockSet) -> HFOverlapBlockSet:
    return HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        fock_screening=overlap_blocks.fock_screening,
    )


def _flavor_band_diagonal(matrix: np.ndarray, *, n_spin: int, n_eta: int, n_band: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    nt, nt_rhs, nk = matrix.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square matrix blocks, got {matrix.shape}")
    if nt != n_spin * n_eta * n_band:
        raise ValueError(f"Matrix dimension {nt} is incompatible with n_spin={n_spin}, n_eta={n_eta}, n_band={n_band}")
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    diagonal = np.zeros((n_spin, n_eta, n_band, nk), dtype=float)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            for iband in range(n_band):
                row = int(idx[ispin, ieta, iband])
                diagonal[ispin, ieta, iband, :] = np.real(matrix[row, row, :])
    return diagonal


def evaluate_htg_interaction_path(
    hf_run: HTGHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGInteractionPathResult:
    source_basis_data = hf_run.basis_data
    resolved_g_shells = _infer_g_shells_from_overlap_blocks(hf_run.overlap_blocks) if g_shells is None else int(g_shells)
    resolved_path = (
        source_basis_data.model.paper_hf_kpath(points_per_segment=points_per_segment)
        if path is None
        else path
    )
    path_basis_data = build_htg_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        projected_band_count=hf_run.state.n_band,
    )
    source_overlap_blocks = hf_run.overlap_blocks
    target_overlap_blocks = build_htg_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    zero_base = np.zeros_like(path_basis_data.h0)
    hartree = build_projected_target_hamiltonian(
        zero_base,
        hf_run.state.density,
        source_overlap_blocks=_hartree_only_blocks(source_overlap_blocks),
        target_overlap_blocks=_hartree_only_blocks(target_overlap_blocks),
        target_source_overlap_blocks=HFOverlapBlockSet(
            shifts=target_source_overlap_blocks.shifts,
            gvecs=target_source_overlap_blocks.gvecs,
            overlaps=target_source_overlap_blocks.overlaps,
        ),
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    fock = build_projected_target_hamiltonian(
        zero_base,
        hf_run.state.density,
        source_overlap_blocks=HFOverlapBlockSet(
            shifts=source_overlap_blocks.shifts,
            gvecs=source_overlap_blocks.gvecs,
            overlaps=source_overlap_blocks.overlaps,
        ),
        target_overlap_blocks=HFOverlapBlockSet(
            shifts=target_overlap_blocks.shifts,
            gvecs=target_overlap_blocks.gvecs,
            overlaps=target_overlap_blocks.overlaps,
        ),
        target_source_overlap_blocks=_fock_only_blocks(target_source_overlap_blocks),
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    total = hartree + fock
    return HTGInteractionPathResult(
        path=resolved_path,
        hartree=hartree,
        fock=fock,
        total=total,
        hartree_diagonal_ev=_flavor_band_diagonal(
            hartree,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        fock_diagonal_ev=_flavor_band_diagonal(
            fock,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        total_diagonal_ev=_flavor_band_diagonal(
            total,
            n_spin=hf_run.state.n_spin,
            n_eta=hf_run.state.n_eta,
            n_band=hf_run.state.n_band,
        ),
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )


def _diagonalize_path_hamiltonian(
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nt, _, nk = hamiltonian.shape
    energies = np.zeros((nk, nt), dtype=float)
    sigma_z_expectation = np.zeros((nk, nt), dtype=float)
    for ik in range(nk):
        eigvals, eigvecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[ik, :] = eigvals
        sigma_z_expectation[ik, :] = np.real(np.diag(eigvecs.conjugate().T @ sigma_z[:, :, ik] @ eigvecs))
    return energies, sigma_z_expectation


def evaluate_htg_hf_path(
    hf_run: HTGHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGHFPathResult:
    source_basis_data = hf_run.basis_data
    resolved_g_shells = _infer_g_shells_from_overlap_blocks(hf_run.overlap_blocks) if g_shells is None else int(g_shells)
    resolved_path = (
        source_basis_data.model.paper_hf_kpath(points_per_segment=points_per_segment)
        if path is None
        else path
    )
    path_basis_data = build_htg_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        projected_band_count=hf_run.state.n_band,
    )
    source_overlap_blocks = hf_run.overlap_blocks
    target_overlap_blocks = build_htg_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    h_path = build_projected_target_hamiltonian(
        path_basis_data.h0,
        hf_run.state.density,
        source_overlap_blocks=source_overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    energies, sigma_z_expectation = _diagonalize_path_hamiltonian(h_path, path_basis_data.sigma_z)
    band_data = build_flavor_band_data(
        h_path,
        n_spin=hf_run.state.n_spin,
        n_eta=hf_run.state.n_eta,
        n_band=hf_run.state.n_band,
    )
    return HTGHFPathResult(
        path=resolved_path,
        hamiltonian=h_path,
        energies=energies,
        sigma_z_expectation=sigma_z_expectation,
        sigma_z_operator=path_basis_data.sigma_z,
        band_data=band_data,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )

__all__ = [name for name in globals() if not name.startswith('__')]
