from __future__ import annotations

import numpy as np
import pytest

from mean_field.systems.tmbg.full_flavor_ivc import (
    FullFlavorLayout,
    flatten_full_density,
    full_density_to_wang_stored,
    ivc_order_metrics,
    odd_integer_filling_summary,
    sector_block_representability,
    uniform_ivc_projector,
    unflatten_full_density,
    validate_projector,
    validate_sector_block_representable,
    wang_stored_to_full_density,
)


def test_full_flavor_flatten_round_trip_preserves_valley_offdiagonal_blocks() -> None:
    layout = FullFlavorLayout(n_band=2)
    blocks = np.zeros(layout.full_density_shape(nk=2), dtype=np.complex128)
    kkprime = np.asarray([0.25 + 0.50j, -0.125 + 0.75j], dtype=np.complex128)
    blocks[0, 0, 0, 0, 1, 1, :] = kkprime
    blocks[0, 1, 1, 0, 0, 0, :] = np.conj(kkprime)

    flat = flatten_full_density(blocks, layout=layout)
    i_k0 = layout.flat_index(0, 0, 0)
    i_kp1 = layout.flat_index(0, 1, 1)

    assert flat.shape == (layout.nt, layout.nt, 2)
    assert np.allclose(flat[i_k0, i_kp1, :], kkprime)
    assert np.allclose(flat[i_kp1, i_k0, :], np.conj(kkprime))
    assert np.allclose(unflatten_full_density(flat, layout), blocks)

    stored = full_density_to_wang_stored(blocks, layout=layout)
    assert np.allclose(stored[i_k0, i_kp1, :], np.conj(kkprime))
    assert np.allclose(wang_stored_to_full_density(stored, layout), blocks)


def test_odd_integer_filling_summary_uses_total_rank_not_sector_counts() -> None:
    summary = odd_integer_filling_summary((25, 26, 27, 28), 27, filling_nu=1, area_ratio=2)

    assert summary.target_fold_indices == (4, 5)
    assert summary.reference_diagonal.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert summary.reference_total_per_k == 16
    assert summary.target_occupied_total_per_k == 2
    assert summary.n_occupied_total_per_k == 18
    assert np.isclose(summary.primitive_nu_from_total, 1.0)
    assert summary.matches_odd_integer_contract
    assert summary.to_dict()["sector_counts_valid_for_ivc"] is False

    target_only_nu3 = odd_integer_filling_summary((27,), 27, filling_nu=3, area_ratio=2)
    assert target_only_nu3.reference_total_per_k == 0
    assert target_only_nu3.target_occupied_total_per_k == 6
    assert target_only_nu3.n_occupied_total_per_k == 6

    with pytest.raises(ValueError, match="odd"):
        odd_integer_filling_summary((27,), 27, filling_nu=2, area_ratio=2)


def test_uniform_ivc_projector_metrics_are_array_only_contract_checks() -> None:
    layout = FullFlavorLayout(n_band=1)
    projector = uniform_ivc_projector(layout, spin_index=0, band_indices=(0,), phase=0.0, nk=3)

    validation = validate_projector(projector, expected_trace=1.0)
    assert validation.is_valid
    assert np.allclose(validation.trace_per_k, np.ones(3))

    metrics = ivc_order_metrics(projector, layout=layout, target_band_indices=(0,))
    assert np.isclose(metrics.ivc_abs_mean, 0.5, atol=1.0e-12)
    assert np.isclose(metrics.ivc_abs_max, 0.5, atol=1.0e-12)
    assert np.allclose(metrics.per_k_frobenius, 0.5)
    assert np.allclose(metrics.per_spin_frobenius_mean, [0.5, 0.0])
    assert np.allclose(metrics.raw_phase_field, 0.0)
    assert "placeholder" in metrics.phase_field_status
    assert np.isnan(metrics.trs_residual)
    assert "sewing" in metrics.trs_residual_status


def test_sector_block_validation_rejects_rho_kkprime_ivc_density() -> None:
    layout = FullFlavorLayout(n_band=1)
    ivc_projector = uniform_ivc_projector(layout, spin_index=0, band_indices=(0,), phase=0.0, nk=1)

    report = sector_block_representability(ivc_projector, layout=layout)
    assert not report.is_representable
    assert report.valley_offdiag_norm > 0.0
    with pytest.raises(ValueError, match="rho_KKprime"):
        validate_sector_block_representable(ivc_projector, layout=layout)

    sector_diagonal = np.zeros_like(ivc_projector)
    sector_diagonal[layout.flat_index(0, 0, 0), layout.flat_index(0, 0, 0), 0] = 1.0
    accepted = validate_sector_block_representable(sector_diagonal, layout=layout)
    assert accepted.is_representable
    assert np.isclose(accepted.off_sector_norm, 0.0)
