from __future__ import annotations

import numpy as np

from mean_field.systems.htg.bands import (
    compute_bands_along_path,
    compute_bands_on_grid,
    estimate_central_band_metrics,
)
from mean_field.systems.htg.hamiltonian import build_coupling_table, centered_band_indices, diagonalize_hamiltonian
from mean_field.systems.htg.lattice import build_htg_lattice, build_kpath_from_nodes
from mean_field.systems.htg.params import HTGParams
from mean_field.systems.htg.plot import HTGPathPlotTrace, write_htg_path_band_plot


def test_htg_path_band_wrapper_matches_direct_diagonalization() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    valley = -1
    path = build_kpath_from_nodes(
        (lattice.gamma_m, lattice.kappa_m),
        ("Gamma", "kappa"),
        2,
    )
    band_indices = centered_band_indices(lattice.matrix_dim, 6)
    d_top = -0.25 * lattice.delta
    d_bot = 0.50 * lattice.delta

    result = compute_bands_along_path(
        path,
        lattice,
        params,
        valley=valley,
        d_top=d_top,
        d_bot=d_bot,
        band_indices=band_indices,
        return_eigenvectors=True,
    )

    assert result.energies.shape == (path.kvec.size, len(band_indices))
    assert result.band_indices == band_indices
    assert result.eigenvectors is not None
    assert result.eigenvectors.shape == (path.kvec.size, lattice.matrix_dim, len(band_indices))
    assert result.metadata["system"] == "HTG"

    top_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)
    for ik, kval in enumerate(path.kvec):
        evals, evecs = diagonalize_hamiltonian(
            kval,
            lattice,
            params,
            valley=valley,
            d_top=d_top,
            d_bot=d_bot,
            top_coupling_table=top_table,
            bottom_coupling_table=bottom_table,
            band_indices=band_indices,
            return_eigenvectors=True,
        )
        np.testing.assert_allclose(result.energies[ik], evals, atol=0.0, rtol=0.0)
        assert evecs is not None
        np.testing.assert_allclose(np.abs(result.eigenvectors[ik]), np.abs(evecs), atol=0.0, rtol=0.0)

    metrics = estimate_central_band_metrics(result, lattice.matrix_dim)
    assert metrics["central_bandwidth_ev"] is not None


def test_htg_grid_band_wrapper_resolves_central_band_count() -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.chiral()
    valley = 1

    result = compute_bands_on_grid(
        2,
        lattice,
        params,
        valley=valley,
        central_band_count=4,
        return_eigenvectors=False,
        frac_shift=(0.25, 0.5),
    )

    expected_indices = centered_band_indices(lattice.matrix_dim, 4)
    assert result.k_grid_frac.shape == (2, 2, 2)
    assert result.kvec.shape == (2, 2)
    assert result.energies.shape == (2, 2, 4)
    assert result.band_indices == expected_indices
    assert result.eigenvectors is None
    assert result.metadata["system"] == "HTG"
    assert result.metadata["valley"] == valley

    top_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=1)
    bottom_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley, shift_sign=-1)
    evals, evecs = diagonalize_hamiltonian(
        result.kvec[0, 0],
        lattice,
        params,
        valley=valley,
        top_coupling_table=top_table,
        bottom_coupling_table=bottom_table,
        band_indices=expected_indices,
        return_eigenvectors=False,
    )
    assert evecs is None
    np.testing.assert_allclose(result.energies[0, 0], evals, atol=0.0, rtol=0.0)


def test_htg_path_band_plot_uses_common_plotting_helpers(tmp_path) -> None:
    lattice = build_htg_lattice(1.5, n_shells=1)
    params = HTGParams.default()
    path = build_kpath_from_nodes(
        (lattice.gamma_m, lattice.kappa_m),
        ("Gamma", "kappa"),
        2,
    )
    result = compute_bands_along_path(
        path,
        lattice,
        params,
        valley=-1,
        band_indices=centered_band_indices(lattice.matrix_dim, 4),
        return_eigenvectors=False,
    )

    paths = write_htg_path_band_plot(
        tmp_path,
        (HTGPathPlotTrace(label="smoke", path_result=result),),
        stem="htg_path_smoke",
        annotate="smoke",
    )

    assert set(paths) == {"band_plot_png", "band_plot_pdf"}
    assert paths["band_plot_png"].is_file()
    assert paths["band_plot_pdf"].is_file()
