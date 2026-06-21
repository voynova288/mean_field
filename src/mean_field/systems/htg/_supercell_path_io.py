from __future__ import annotations

from ._supercell_shared import *  # noqa: F401,F403
from ._supercell_types import *  # noqa: F401,F403
from ._supercell_basis import *  # noqa: F401,F403
from ._supercell_runner import *  # noqa: F401,F403

def build_htg_supercell_gamma_path(
    basis_data: HTGSupercellProjectedBasisData,
    points_per_segment: int = 80,
) -> KPath:
    """Continuous HTG paper-style supercell path.

    This is an off-grid reconstruction path.  Mean-field band figures should use
    exact SCF-grid samples by default; call this only for explicitly diagnostic
    off-grid reconstruction.
    """

    gamma = 0.0 + 0.0j
    kappa = (basis_data.super_g1 + basis_data.super_g2) / 3.0
    kappa_prime_edge = -(basis_data.super_g1 + basis_data.super_g2) / 3.0 + basis_data.super_g1
    m_point = basis_data.super_g1 / 2.0
    gamma_across_m = gamma + basis_data.super_g1
    return build_kpath_from_nodes(
        (gamma, kappa, kappa_prime_edge, gamma, m_point, gamma_across_m),
        ("Gamma_s", "kappa_s", "kappa_prime_s", "Gamma_s", "M_s", "Gamma_s+G1"),
        int(points_per_segment),
    )


def evaluate_htg_supercell_hf_path(
    hf_run: HTGSupercellHartreeFockRun,
    *,
    path: KPath | None = None,
    points_per_segment: int = 80,
    g_shells: int | None = None,
    beta: float = 1.0,
    use_numba: bool | None = None,
) -> HTGSupercellPathResult:
    source_basis_data = hf_run.basis_data
    resolved_path = build_htg_supercell_gamma_path(source_basis_data, points_per_segment=points_per_segment) if path is None else path
    resolved_g_shells = source_basis_data.interaction.g_shells if g_shells is None else int(g_shells)
    path_basis_data = build_htg_supercell_projected_basis_for_kvec(
        source_basis_data.model,
        source_basis_data.interaction,
        resolved_path.kvec,
        supercell=source_basis_data.supercell,
        projected_band_count=source_basis_data.primitive_band_count,
    )
    target_overlap_blocks = build_htg_supercell_overlap_blocks(path_basis_data, g_shells=resolved_g_shells)
    target_source_overlap_blocks = build_htg_supercell_overlap_blocks_between(
        path_basis_data,
        source_basis_data,
        g_shells=resolved_g_shells,
        include_hartree=False,
    )
    h_path = build_projected_target_hamiltonian(
        path_basis_data.h0,
        hf_run.state.density,
        source_overlap_blocks=hf_run.overlap_blocks,
        target_overlap_blocks=target_overlap_blocks,
        target_source_overlap_blocks=target_source_overlap_blocks,
        v0=hf_run.state.v0,
        beta=beta,
        use_numba=use_numba,
    )
    energies = np.zeros((resolved_path.kvec.size, h_path.shape[0]), dtype=float)
    for ik in range(resolved_path.kvec.size):
        energies[ik, :] = np.linalg.eigvalsh(h_path[:, :, ik])
    return HTGSupercellPathResult(
        path=resolved_path,
        hamiltonian=h_path,
        energies=energies,
        mu=hf_run.state.mu,
        nu=hf_run.state.nu,
        init_mode=hf_run.init_mode,
        seed=hf_run.seed,
        exit_reason=hf_run.exit_reason,
        points_per_segment=int(points_per_segment),
    )


def save_htg_supercell_run_npz(path: str, run: HTGSupercellHartreeFockRun) -> None:
    np.savez_compressed(
        path,
        density=run.state.density,
        hamiltonian=run.state.hamiltonian,
        h0=run.state.h0,
        energies=run.state.energies,
        kvec=run.basis_data.kvec,
        k_grid_frac=np.asarray([]) if run.basis_data.k_grid_frac is None else run.basis_data.k_grid_frac,
        iter_energy=run.iter_energy,
        iter_err=run.iter_err,
        iter_oda=run.iter_oda,
        reference_diagonal=run.state.reference_diagonal,
        fold_representatives=np.asarray(run.basis_data.fold_representatives, dtype=int),
        supercell_matrix=np.asarray(
            [
                [run.basis_data.supercell.n11, run.basis_data.supercell.n12],
                [run.basis_data.supercell.n21, run.basis_data.supercell.n22],
            ],
            dtype=int,
        ),
        primitive_nu=float(run.state.nu),
        init_mode=str(run.init_mode),
        seed=int(run.seed),
        converged=bool(run.converged),
        exit_reason=str(run.exit_reason),
        diagnostics=np.asarray([run.state.diagnostics], dtype=object),
    )


def save_htg_supercell_path_npz(path: str, result: HTGSupercellPathResult) -> None:
    np.savez_compressed(
        path,
        kvec=result.path.kvec,
        kdist=result.path.kdist,
        labels=np.asarray(result.path.labels, dtype=object),
        node_indices=np.asarray(result.path.node_indices, dtype=int),
        hamiltonian=result.hamiltonian,
        energies=result.energies,
        mu=float(result.mu),
        primitive_nu=float(result.nu),
        init_mode=str(result.init_mode),
        seed=int(result.seed),
        exit_reason=str(result.exit_reason),
    )

__all__ = [name for name in globals() if not name.startswith('__')]
