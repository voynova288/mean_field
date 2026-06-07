from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf import HartreeFockProblem, diagonal_overlap_blocks
from mean_field.systems.tdbg.hf_plotting import tdbg_hf_band_data
from mean_field.systems.tdbg.projected_hf import (
    SPIN_LABELS,
    VALLEY_SEQUENCE,
    TDBGInteractionSettings,
    TDBGProjectedHFConfig,
    TDBGProjectedHFData,
    TDBGProjectedWindow,
    TDBGStateLabel,
    build_tdbg_projected_hf_data,
    build_tdbg_projected_hf_problem,
    build_tdbg_total_overlap_blocks,
    _tdbg_total_overlap_between,
    _total_diagonal_overlap_from_wavefunctions,
    initialize_tdbg_density,
    initialize_tdbg_nu2_density,
    tdbg_density_from_hamiltonian,
    tdbg_energy_components,
    tdbg_delta_from_paper_ud_for_valley,
    tdbg_order_parameters,
    validate_tdbg_interaction_settings,
)
from mean_field.systems.tdbg.topology import translation_srcmap


def _toy_data(*, filling: int = 2) -> TDBGProjectedHFData:
    labels = []
    band_indices = (10, 11)
    n_band = 2
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band in enumerate(band_indices):
                idx = iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)
                labels.append(TDBGStateLabel(idx, spin, valley, iband, band))
    nt = len(labels)
    nk = 1
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    reference = np.zeros_like(h0)
    for label in labels:
        if label.band_position == 0:
            reference[label.index, label.index, :] = 1.0
    lattice = SimpleNamespace(n_q=1)
    model = SimpleNamespace(lattice=lattice)
    return TDBGProjectedHFData(
        model=model,  # type: ignore[arg-type]
        config=TDBGProjectedHFConfig(
            window=TDBGProjectedWindow("two_flat"),
            filling=int(filling),
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        ),
        k_grid_frac=np.zeros((1, 1, 2)),
        kvec=np.zeros(1, dtype=np.complex128),
        band_indices=band_indices,
        labels=tuple(labels),
        h0=h0,
        wavefunctions=np.zeros((nt, nk, 1, 4), dtype=np.complex128),
        reference_density=reference,
        n_occupied_per_k=4 + int(filling),
        lower_band_count=1,
        moire_area_nm2=1.0,
        shifts=(),
        shift_gvecs=np.zeros(0, dtype=np.complex128),
        shift_srcmaps=(),
    )


def _toy_overlap_data() -> TDBGProjectedHFData:
    labels = []
    band_indices = (10, 11)
    n_band = 2
    for ispin, spin in enumerate(SPIN_LABELS):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for iband, band in enumerate(band_indices):
                idx = iband + n_band * (ivalley + len(VALLEY_SEQUENCE) * ispin)
                labels.append(TDBGStateLabel(idx, spin, valley, iband, band))
    nt = len(labels)
    nk = 2
    g1 = 1.0 + 0.0j
    g2 = 0.0 + 1.0j
    q0 = 0.2 + 0.1j
    q_sites = []
    for axis0 in range(2):
        for axis1 in range(2):
            for sector in (0, 1):
                coord = axis0 * g1 + axis1 * g2 - sector * q0
                q_sites.append([coord.real, coord.imag, float(sector)])
    q_sites_array = np.asarray(q_sites, dtype=float)
    lattice = SimpleNamespace(
        n_q=q_sites_array.shape[0],
        q_sites=q_sites_array,
        q_complex=np.asarray([q0], dtype=np.complex128),
        g_m1=g1,
        g_m2=g2,
        cut=1.0,
    )
    model = SimpleNamespace(lattice=lattice)
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    reference = np.zeros_like(h0)
    for label in labels:
        if label.band_position == 0:
            reference[label.index, label.index, :] = 1.0
    rng = np.random.default_rng(123)
    wavefunctions = np.zeros((nt, nk, lattice.n_q, 4), dtype=np.complex128)
    for valley in VALLEY_SEQUENCE:
        for iband in range(n_band):
            values = rng.standard_normal((nk, lattice.n_q, 4)) + 1j * rng.standard_normal((nk, lattice.n_q, 4))
            for label in labels:
                if int(label.valley) == int(valley) and label.band_position == iband:
                    wavefunctions[label.index] = values
    shift = (1, 0)
    return TDBGProjectedHFData(
        model=model,  # type: ignore[arg-type]
        config=TDBGProjectedHFConfig(window=TDBGProjectedWindow("two_flat"), interaction=TDBGInteractionSettings(include_intersite=True, include_onsite=False)),
        k_grid_frac=np.zeros((1, nk, 2)),
        kvec=np.asarray([0.0 + 0.0j, 0.1 + 0.2j], dtype=np.complex128),
        band_indices=band_indices,
        labels=tuple(labels),
        h0=h0,
        wavefunctions=wavefunctions,
        reference_density=reference,
        n_occupied_per_k=6,
        lower_band_count=1,
        moire_area_nm2=1.0,
        shifts=(shift,),
        shift_gvecs=np.asarray([g1], dtype=np.complex128),
        shift_srcmaps=(translation_srcmap(lattice, g1),),
    )

def test_tdbg_nu2_initializers_classify_sp_and_vp() -> None:
    data = _toy_data()
    sp = tdbg_order_parameters(data, initialize_tdbg_nu2_density(data, init_mode="sp", seed=1))
    vp = tdbg_order_parameters(data, initialize_tdbg_nu2_density(data, init_mode="vp_k", seed=1))
    assert sp["classification"] == "SP_up"
    assert vp["classification"] == "VP_K"
    assert sp["cb_spin_polarization"] == 2.0
    assert vp["cb_valley_polarization"] == 2.0


def test_tdbg_target_diagonal_overlap_matches_full_overlap_blocks() -> None:
    data = build_tdbg_projected_hf_data(
        TDBGProjectedHFConfig(
            theta_deg=1.38,
            cut=1.0,
            mesh_size=1,
            paper_ud_ev=0.09,
            paper_ud_convention="minus_xi_ud_over3",
            window=TDBGProjectedWindow("two_flat"),
            filling=2,
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        )
    )

    for ishift, shift in enumerate(data.shifts[: min(3, len(data.shifts))]):
        full = _tdbg_total_overlap_between(data, data.wavefunctions, data.wavefunctions, shift)
        expected = diagonal_overlap_blocks(full, nt=data.nt, nk=data.nk)
        actual = _total_diagonal_overlap_from_wavefunctions(data, data.wavefunctions, ishift)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_tdbg_paper_ud_convention_maps_valley_resolved_delta() -> None:
    assert tdbg_delta_from_paper_ud_for_valley(0.09, 1, convention="same_delta_minus_ud_over3") == pytest.approx(-0.03)
    assert tdbg_delta_from_paper_ud_for_valley(0.09, -1, convention="same_delta_minus_ud_over3") == pytest.approx(-0.03)
    assert tdbg_delta_from_paper_ud_for_valley(0.09, 1, convention="minus_xi_ud_over3") == pytest.approx(-0.03)
    assert tdbg_delta_from_paper_ud_for_valley(0.09, -1, convention="minus_xi_ud_over3") == pytest.approx(0.03)


def test_tdbg_projected_hf_data_uses_minus_xi_ud_over3_valley_params() -> None:
    data = build_tdbg_projected_hf_data(
        TDBGProjectedHFConfig(
            theta_deg=1.38,
            cut=1.0,
            mesh_size=1,
            paper_ud_ev=0.09,
            paper_ud_convention="minus_xi_ud_over3",
            window=TDBGProjectedWindow("two_flat"),
            filling=2,
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        )
    )

    assert data.valley_params is not None
    assert data.valley_params[1].Delta == pytest.approx(-0.03)
    assert data.valley_params[-1].Delta == pytest.approx(0.03)


def test_tdbg_negative_filling_initializer_removes_active_valence_projectors() -> None:
    data = _toy_data(filling=-2)
    density = initialize_tdbg_density(data, init_mode="vp_k", seed=1)
    order = tdbg_order_parameters(data, density)

    occupations = {item["index"]: item["occupation"] for item in order["occupations"]}
    # The charge-neutral reference fills all four valence flavors; nu=-2 with
    # vp_k removes the two K-valley valence projectors.
    assert occupations[0] == pytest.approx(0.0)
    assert occupations[4] == pytest.approx(0.0)
    assert occupations[2] == pytest.approx(1.0)
    assert occupations[6] == pytest.approx(1.0)
    assert order["active_valley_polarization"] == pytest.approx(-2.0)
    assert order["classification"] == "VP_Kprime"

def test_tdbg_negative_filling_biased_ivc_initializer_sets_valley_and_ivc_seed() -> None:
    data = _toy_data(filling=-2)
    density = initialize_tdbg_density(data, init_mode="ivc_k85", seed=1)
    order = tdbg_order_parameters(data, density)

    assert order["classification"] == "IVC_or_valley_coherent"
    assert order["active_valley_polarization"] == pytest.approx(-1.4)
    assert order["ivc_amplitude"] == pytest.approx(2.0 * np.sqrt(0.85 * 0.15))


def test_tdbg_total_overlap_blocks_match_finite_q_site_manual_contraction() -> None:
    data = _toy_overlap_data()
    blocks = build_tdbg_total_overlap_blocks(data)
    block = blocks.overlaps[(1, 0)]

    expected = np.zeros_like(block)
    src = data.shift_srcmaps[0]
    valid = src >= 0
    for a, la in enumerate(data.labels):
        wa = np.conj(data.wavefunctions[a][:, valid, :])
        for b, lb in enumerate(data.labels):
            if la.spin != lb.spin or int(la.valley) != int(lb.valley):
                continue
            wb = data.wavefunctions[b][:, src[valid], :]
            expected[a, :, b, :] = np.einsum("tqa,sqa->ts", wa, wb, optimize=True)

    assert np.allclose(block, expected)
    assert np.allclose(blocks.diagonal_overlaps[(1, 0)], np.diagonal(expected, axis1=1, axis2=3))

def test_tdbg_density_builder_uses_core_stored_projector_convention() -> None:
    # A diagonal Hamiltonian with two occupied states should produce a stored
    # projector whose diagonal occupations are the two lowest eigenstates.
    h = np.zeros((4, 4, 1), dtype=np.complex128)
    h[:, :, 0] = np.diag([0.0, 2.0, -1.0, 3.0])
    density, energies, mu, mask = tdbg_density_from_hamiltonian(h, 2)
    assert np.allclose(energies[:, 0], [-1.0, 0.0, 2.0, 3.0])
    assert mask[:2, 0].all()
    assert not mask[2:, 0].any()
    # In the original basis, states 2 and 0 are occupied.
    assert np.allclose(np.diag(density[:, :, 0]).real, [1.0, 0.0, 1.0, 0.0])
    assert mu == 1.0


def test_tdbg_density_builder_stores_offdiagonal_projector_transpose() -> None:
    h = np.zeros((2, 2, 1), dtype=np.complex128)
    h[:, :, 0] = np.array([[0.25, 0.2 - 0.3j], [0.2 + 0.3j, -0.1]], dtype=np.complex128)
    density, energies, _, mask = tdbg_density_from_hamiltonian(h, 1)

    vals, vecs = np.linalg.eigh(h[:, :, 0])
    conventional = vecs[:, :1] @ vecs[:, :1].conjugate().T
    assert np.allclose(energies[:, 0], vals)
    assert mask[0, 0]
    assert not mask[1, 0]
    assert np.allclose(density[:, :, 0], conventional.T)
    assert np.isclose(
        np.einsum("ab,ab->", h[:, :, 0], density[:, :, 0]).real,
        np.trace(h[:, :, 0] @ conventional).real,
    )


def test_tdbg_reference_subtracted_hartree_energy_contracts_policy_density() -> None:
    data = _toy_data()
    density = initialize_tdbg_nu2_density(data, init_mode="sp", seed=1)
    hartree_h = np.zeros_like(density)
    for idx in range(data.nt):
        hartree_h[idx, idx, :] = 1.0

    components = tdbg_energy_components(data, density, interaction_components={"hartree": hartree_h})

    # The toy two-flat reference fills four lower-band flavors; nu=2 adds only
    # two conduction flavors.  Charge-neutral Hartree energy must contract with
    # P-P_ref, giving 1/2 * 2, not 1/2 * 6 from the absolute projector.
    assert components["hartree_ev"] == pytest.approx(1.0)
    assert components["interaction_ev"] == pytest.approx(1.0)
    assert components["total_ev"] == pytest.approx(1.0)


def test_tdbg_projected_hf_problem_uses_core_problem_surface() -> None:
    data = _toy_data()
    problem = build_tdbg_projected_hf_problem(data)
    assert isinstance(problem, HartreeFockProblem)


def test_tdbg_interaction_policy_validation_rejects_unknown_strings() -> None:
    with pytest.raises(ValueError, match="Hartree reference"):
        validate_tdbg_interaction_settings(TDBGInteractionSettings(hartree_reference="background"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Fock density"):
        validate_tdbg_interaction_settings(TDBGInteractionSettings(fock_density="delta"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="onsite valley"):
        validate_tdbg_interaction_settings(TDBGInteractionSettings(onsite_valley_policy="intervalley"))  # type: ignore[arg-type]


def test_tdbg_hf_band_data_labels_by_tdbg_flavor_sectors() -> None:
    data = _toy_data()
    h = np.zeros_like(data.h0)
    for label in data.labels:
        h[label.index, label.index, 0] = float(label.index)

    band_data = tdbg_hf_band_data(data, h)

    assert band_data.energies_ev.shape == (data.nt, data.nk)
    assert band_data.band_labels[0].startswith("K_up_")
    assert band_data.band_labels[2].startswith("Kprime_up_")
    assert band_data.band_labels[4].startswith("K_down_")
    assert band_data.band_labels[6].startswith("Kprime_down_")
