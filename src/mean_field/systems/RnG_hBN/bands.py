from __future__ import annotations


from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh

from ...api.artifacts import update_artifact_manifest
from ...core.bands import GridBandsResult, PathBandsResult, compute_grid_bands, compute_path_bands
from ...core.lattice import KPath
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian, flat_band_indices, hamiltonian_dimension
from .lattice import RLGhBNLattice, build_kpath_from_nodes, build_moire_k_grid
from .params import RLGhBNParams


def build_fig6_paper_hf_path(model: object, points_per_segment: int) -> KPath:
    """Return the paper-style RnG/hBN HF path used by the Fig. 6 workflow."""

    lattice = model.lattice
    g2 = lattice.g_m2
    # Keep the Fig. 6 M' representative explicit.  The projected-basis builder
    # folds each target k for diagonalization and then relabels plane-wave
    # coefficients back into this raw path convention before form factors are
    # evaluated.
    mprime_fig6 = g2 / 2.0
    m_fig6 = lattice.m_m
    nodes = (
        lattice.gamma_m,
        lattice.k_m,
        lattice.kprime_m,
        lattice.gamma_m,
        mprime_fig6,
        m_fig6,
        lattice.gamma_m,
    )
    labels = ("$\\Gamma_M$", "$K_M$", "$K'_M$", "$\\Gamma_M$", "$M'_M$", "$M_M$", "$\\Gamma_M$")
    return build_kpath_from_nodes(
        nodes,
        labels,
        tuple(int(points_per_segment) for _ in range(len(nodes) - 1)),
    )


def update_paper_hf_band_plot_manifest(
    source_dir: Path | str,
    *,
    paper_target: str,
    panel_names: list[str],
    status: str,
) -> Path:
    """Record RnG/hBN paper-HF band-plot files without overwriting core sidecars."""

    source_dir = Path(source_dir)
    files: dict[str, object] = {
        "hf_band_plot_config": "hf_band_plot_config.json",
        "hf_band_plot_panels": panel_names,
    }
    if status != "dry_run":
        files.update(
            {
                "hf_band_plot_summary": "hf_band_plot_summary.json",
                "hf_band_plot_combined_png": f"paper_{paper_target}_hf_bands.png",
                "hf_band_plot_combined_pdf": f"paper_{paper_target}_hf_bands.pdf",
            }
        )
    return update_artifact_manifest(
        source_dir,
        files=files,
        metadata={
            "band_plot": {
                "workflow": "rlg_hbn.paper_hf_bands",
                "status": str(status),
                "paper_target": str(paper_target),
                "panel_count": int(len(panel_names)),
            }
        },
    )


def neutrality_energy_mev(path_result: PathBandsResult, lattice: RLGhBNLattice, params: RLGhBNParams) -> float:
    """Energy zero used for paper-style RLG/hBN band plots.

    The continuum Hamiltonian includes an arbitrary fitted onsite offset from
    the RLG remote parameters.  Fig. 2 of Kwan et al. plots the single-particle
    spectrum relative to charge neutrality, so the useful reference is the
    midpoint between the path maximum of the central valence band and the path
    minimum of the central conduction band.
    """

    flat_valence, flat_conduction = flat_band_indices(lattice, params)
    energies = np.asarray(path_result.energies, dtype=float)
    if energies.ndim != 2:
        raise ValueError(f"Expected path energies with shape (n_k, n_bands), got {energies.shape}")
    if flat_conduction >= energies.shape[1]:
        raise ValueError(
            f"Path result only contains {energies.shape[1]} bands, but neutrality reference "
            f"requires central conduction index {flat_conduction}."
        )
    valence_top = float(np.max(energies[:, flat_valence]))
    conduction_bottom = float(np.min(energies[:, flat_conduction]))
    return 0.5 * (valence_top + conduction_bottom)


def _make_diagonalizer(lattice: RLGhBNLattice, params: RLGhBNParams, *, valley: int, basis_dim: int):
    def _diagonalize(kval: complex, resolved_n_bands: int, want_eigenvectors: bool):
        if want_eigenvectors:
            return diagonalize_hamiltonian(kval, lattice, params, valley=valley, n_bands=resolved_n_bands)
        hamiltonian = build_hamiltonian(kval, lattice, params, valley=valley)
        if resolved_n_bands >= basis_dim:
            return np.asarray(eigvalsh(hamiltonian), dtype=float), None
        return np.asarray(eigvalsh(hamiltonian, subset_by_index=[0, resolved_n_bands - 1]), dtype=float), None

    return _diagonalize


def compute_bands_along_path(
    path: KPath,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
) -> PathBandsResult:
    basis_dim = hamiltonian_dimension(lattice, params)

    return compute_path_bands(
        path,
        matrix_dim=basis_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_make_diagonalizer(lattice, params, valley=valley, basis_dim=basis_dim),
    )


def compute_bands_on_grid(
    mesh_size: int,
    lattice: RLGhBNLattice,
    params: RLGhBNParams,
    *,
    valley: int = 1,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
    endpoint: bool = False,
    frac_shift: tuple[float, float] = (0.0, 0.0),
) -> GridBandsResult:
    basis_dim = hamiltonian_dimension(lattice, params)
    k_grid_frac, kvec = build_moire_k_grid(lattice, mesh_size, endpoint=endpoint, frac_shift=frac_shift)

    return compute_grid_bands(
        k_grid_frac=k_grid_frac,
        kvec=kvec,
        matrix_dim=basis_dim,
        n_bands=n_bands,
        return_eigenvectors=return_eigenvectors,
        diagonalize=_make_diagonalizer(lattice, params, valley=valley, basis_dim=basis_dim),
    )


__all__ = [
    "GridBandsResult",
    "PathBandsResult",
    "build_fig6_paper_hf_path",
    "compute_bands_along_path",
    "compute_bands_on_grid",
    "neutrality_energy_mev",
    "update_paper_hf_band_plot_manifest",
]
