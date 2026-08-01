from __future__ import annotations

from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.core.hf import build_projected_interaction_hamiltonian
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field import (
    BMSolution,
    HFOverlapBlockSet,
    RestrictedHartreeFockRun,
    RestrictedHartreeFockState,
    TBGZeroFieldHFSourceReceipt,
    TBGZeroFieldTDHFContext,
    TBGZeroFieldTDHFOccupationResiduals,
    TBGZeroFieldTDHFOrbitals,
    TBGZeroFieldTDHFProvenance,
    build_tbg_zero_field_hf_source_receipt,
    build_tbg_zero_field_tdhf_orbitals,
    build_tbg_zero_field_tdhf_q0_matrices,
    build_tbg_zero_field_tdhf_q0_pairs,
    conventional_projector_to_stored_density,
    conventional_tangent_to_stored_tangent,
    stored_density_to_conventional_projector,
    stored_tangent_to_conventional_tangent,
    tbg_zero_field_lattice_kvec_sha256,
    tbg_zero_field_overlap_kernel_inventory_fingerprint,
    validate_tbg_zero_field_tdhf_tangent_columns,
)


def _provenance(
    label: str,
    receipt: TBGZeroFieldHFSourceReceipt,
) -> TBGZeroFieldTDHFProvenance:
    return TBGZeroFieldTDHFProvenance(
        hf_run_source=f"synthetic:{label}:hf",
        overlap_blocks_source=f"synthetic:{label}:overlaps",
        interaction_parameters_source=f"synthetic:{label}:interaction",
        reference_density_source=f"synthetic:{label}:centered-half-identity",
        expected_hf_source_receipt_sha256=receipt.fingerprint,
        hf_mode="full",
    )


def _bm_solution(
    *,
    n_spin: int,
    n_eta: int,
    n_band: int,
    nk: int,
    h0: np.ndarray | None = None,
) -> BMSolution:
    params = TBGParameters.from_degrees(1.05)
    nt = n_spin * n_eta * n_band
    plane_wave_dimension = 4
    lattice_kvec = np.asarray([0.0 + 0.0j, 0.17 + 0.09j], dtype=np.complex128)[:nk]
    spectrum = np.zeros((n_band, n_eta, nk), dtype=float)
    if h0 is not None:
        resolved_h0 = np.asarray(h0, dtype=np.complex128)
        assert resolved_h0.shape == (nt, nt, nk)
        row = 0
        for iband in range(n_band):
            for ieta in range(n_eta):
                spin_values = np.asarray(
                    [resolved_h0[row + ispin, row + ispin, :] for ispin in range(n_spin)]
                )
                np.testing.assert_allclose(spin_values, spin_values[[0], :], rtol=0.0, atol=1.0e-14)
                spectrum[iband, ieta, :] = spin_values[0].real
                row += n_spin
    return BMSolution(
        params=params,
        lattice_kvec=lattice_kvec,
        lg=1,
        nlocal=4,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=n_band,
        hamiltonian=np.zeros(
            (plane_wave_dimension, plane_wave_dimension, n_eta, nk),
            dtype=np.complex128,
        ),
        sigma_z=np.zeros((nt, nt, nk), dtype=np.complex128),
        uk=np.zeros((plane_wave_dimension, n_band, n_eta, nk), dtype=np.complex128),
        spectrum=spectrum,
        gvec=np.asarray([0.0 + 0.0j], dtype=np.complex128),
    )


def _two_k_source() -> tuple[BMSolution, RestrictedHartreeFockRun]:
    nt = 2
    nk = 2
    beta = 0.6
    v0 = 0.8
    hf_energies = np.asarray([[-1.0, -0.8], [0.7, 1.1]], dtype=float)
    hamiltonian = np.zeros((nt, nt, nk), dtype=np.complex128)
    projector = np.zeros_like(hamiltonian)
    for ik in range(nk):
        hamiltonian[:, :, ik] = np.diag(hf_energies[:, ik])
        projector[:, :, ik] = np.diag([1.0, 0.0])

    overlap = np.zeros((nt, nk, nt, nk), dtype=np.complex128)
    overlap[:, 0, :, 0] = np.diag([1.0, 0.8])
    overlap[:, 1, :, 1] = np.diag([0.9, 1.1])
    overlap[:, 0, :, 1] = np.diag([0.7 + 0.1j, 0.6 - 0.05j])
    overlap[:, 1, :, 0] = overlap[:, 0, :, 1].conjugate().T
    diagonal = np.diagonal(overlap, axis1=1, axis2=3)
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): diagonal},
        hartree_screening={(0, 0): 0.3},
        fock_screening={(0, 0): np.asarray([[0.9, 0.4], [0.4, 1.1]], dtype=float)},
    )
    density = conventional_projector_to_stored_density(projector)
    interaction = build_projected_interaction_hamiltonian(
        density,
        blocks,
        v0=v0,
        beta=beta,
    )
    h0 = hamiltonian - interaction
    solution = _bm_solution(n_spin=1, n_eta=1, n_band=2, nk=nk, h0=h0)
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros_like(hamiltonian),
        density=density,
        hamiltonian=hamiltonian,
        energies=hf_energies,
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=v0,
        n_spin=1,
        n_eta=1,
        n_band=2,
        diagnostics={"beta": beta},
    )
    state.hf_source_receipt = build_tbg_zero_field_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-1.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="synthetic-full",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    return solution, run


def _source_with_overlap_blocks(
    blocks: HFOverlapBlockSet,
) -> tuple[BMSolution, RestrictedHartreeFockRun]:
    solution, run = _two_k_source()
    state = run.state
    beta = float(state.diagnostics["beta"])
    interaction = build_projected_interaction_hamiltonian(
        state.density,
        blocks,
        v0=state.v0,
        beta=beta,
    )
    state.h0 = state.hamiltonian - interaction
    solution = _bm_solution(
        n_spin=state.n_spin,
        n_eta=state.n_eta,
        n_band=state.n_band,
        nk=state.nk,
        h0=state.h0,
    )
    state.hf_source_receipt = build_tbg_zero_field_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    return solution, replace(run, overlap_blocks=blocks)


def _context_from_source(
    solution: BMSolution,
    run: RestrictedHartreeFockRun,
    *,
    label: str = "two-k",
) -> TBGZeroFieldTDHFContext:
    return TBGZeroFieldTDHFContext(
        grid_solution=solution,
        run=run,
        beta=float(run.state.diagnostics["beta"]),
        provenance=_provenance(label, run.state.hf_source_receipt),
        closure_tolerance=1.0e-12,
    )


def _two_k_context() -> TBGZeroFieldTDHFContext:
    return _context_from_source(*_two_k_source())


def test_source_receipt_fingerprint_binds_v0() -> None:
    _solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    changed_v0_receipt = replace(receipt, v0=receipt.v0 + 1.0)

    assert receipt.schema_version == 2
    assert changed_v0_receipt.fingerprint != receipt.fingerprint


def test_source_receipt_json_metadata_round_trip_reconstructs_and_verifies() -> None:
    _solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    loaded_metadata = json.loads(
        json.dumps(receipt.to_metadata(), sort_keys=True, allow_nan=False)
    )

    reconstructed = TBGZeroFieldHFSourceReceipt.from_metadata(loaded_metadata)
    assert reconstructed == receipt
    assert reconstructed.fingerprint == loaded_metadata["fingerprint"]

    loaded_metadata["v0"] = receipt.v0 + 1.0
    with pytest.raises(ValueError, match="fingerprint does not match"):
        TBGZeroFieldHFSourceReceipt.from_metadata(loaded_metadata)


@pytest.mark.parametrize(
    ("module_name", "runner_name", "expected_mode", "init_mode"),
    [
        (
            "mean_field.systems.tbg.zero_field._hf_full",
            "run_full_hartree_fock",
            "full",
            "flavor",
        ),
        (
            "mean_field.systems.tbg.zero_field._hf_restricted",
            "run_restricted_hartree_fock",
            "restricted",
            "educated",
        ),
    ],
)
def test_hf_runners_persist_the_screened_blocks_used_by_the_kernel(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    runner_name: str,
    expected_mode: str,
    init_mode: str,
) -> None:
    solution, source_run = _two_k_source()
    unscreened = replace(
        source_run.overlap_blocks,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening={},
    )
    captured: dict[str, object] = {}
    module = importlib.import_module(module_name)

    def fake_build_projected_hf_kernel(state, overlap_blocks, **kwargs):
        captured["overlap_blocks"] = overlap_blocks
        captured["oda_parameterizer"] = kwargs["oda_parameterizer"]
        return SimpleNamespace()

    def fake_run_hartree_fock_problem(state, problem, **kwargs):
        return SimpleNamespace(
            iter_energy=np.asarray([], dtype=float),
            iter_err=np.asarray([], dtype=float),
            iter_oda=np.asarray([], dtype=float),
            init_mode=kwargs["init_mode"],
            seed=kwargs["seed"],
            converged=True,
            exit_reason="synthetic-converged",
        )

    monkeypatch.setattr(module, "build_projected_hf_kernel", fake_build_projected_hf_kernel)
    monkeypatch.setattr(module, "run_hartree_fock_problem", fake_run_hartree_fock_problem)
    result = getattr(module, runner_name)(
        source_run.state,
        unscreened,
        solution.lattice_kvec,
        solution.params,
        init_mode=init_mode,
        max_iter=0,
    )

    assert result.overlap_blocks is captured["overlap_blocks"]
    oda_parameterizer = captured["oda_parameterizer"]
    assert callable(oda_parameterizer)
    assert any(
        cell.cell_contents is result.overlap_blocks
        for cell in oda_parameterizer.__closure__ or ()
    )
    assert result.overlap_blocks.diagonal_overlaps
    assert result.overlap_blocks.hartree_screening
    assert result.overlap_blocks.fock_screening
    assert all(
        isinstance(value, (int, float, np.integer, np.floating))
        for value in result.state.diagnostics.values()
    )
    receipt = result.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    assert receipt.hf_mode == expected_mode
    assert receipt.schema_version == 2
    assert receipt.v0 == result.state.v0
    assert receipt.lattice_kvec_sha256 == tbg_zero_field_lattice_kvec_sha256(
        solution.lattice_kvec
    )
    assert receipt.overlap_kernel_inventory_sha256 == (
        tbg_zero_field_overlap_kernel_inventory_fingerprint(result.overlap_blocks)
    )
    assert TBGZeroFieldHFSourceReceipt.from_metadata(receipt.to_metadata()) == receipt


def test_stored_and_conventional_density_conversions_are_explicit() -> None:
    vector = np.asarray([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)
    projector = (vector[:, None] @ vector[None, :].conjugate())[:, :, None]
    stored = conventional_projector_to_stored_density(projector)
    np.testing.assert_allclose(stored_density_to_conventional_projector(stored), projector, atol=0.0)

    tangent = np.asarray([[0.0, 0.3 + 0.2j], [-0.1j, 0.0]], dtype=np.complex128)[:, :, None]
    np.testing.assert_allclose(
        stored_tangent_to_conventional_tangent(conventional_tangent_to_stored_tangent(tangent)),
        tangent,
        atol=0.0,
    )


def test_q0_pair_order_is_k_then_hole_then_particle() -> None:
    energies = np.arange(8, dtype=float).reshape((4, 2), order="F")
    occupied = np.asarray(
        [[True, False], [False, True], [True, False], [False, True]],
        dtype=bool,
    )
    projector = np.zeros((4, 4, 2), dtype=np.complex128)
    for ik in range(2):
        projector[:, :, ik] = np.diag(occupied[:, ik].astype(float))
    orbitals = TBGZeroFieldTDHFOrbitals(
        energies=energies,
        eigenvectors=np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], 2, axis=2),
        occupied_mask=occupied,
        conventional_projector=projector,
        mu=0.0,
        occupation_residuals=TBGZeroFieldTDHFOccupationResiduals(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0e-12,
            1.0e-12,
        ),
        source_hamiltonian_sha256="0" * 64,
        source_density_sha256="1" * 64,
    )

    pairs = build_tbg_zero_field_tdhf_q0_pairs(orbitals)
    decoded = [
        (*orbitals.decode_global_index(pair.particle), *orbitals.decode_global_index(pair.hole))
        for pair in pairs
    ]
    assert decoded == [
        (1, 0, 0, 0),
        (3, 0, 0, 0),
        (1, 0, 2, 0),
        (3, 0, 2, 0),
        (0, 1, 1, 1),
        (2, 1, 1, 1),
        (0, 1, 3, 1),
        (2, 1, 3, 1),
    ]


def test_orbitals_use_stored_projector_and_reject_noninteger_occupations() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    np.testing.assert_array_equal(orbitals.occupied_mask, [[True, True], [False, False]])
    np.testing.assert_allclose(
        orbitals.conventional_projector,
        stored_density_to_conventional_projector(context.run.state.density),
    )

    context.run.state.density[0, 0, 0] -= 0.2
    with pytest.raises(ValueError, match="0/1 projector"):
        build_tbg_zero_field_tdhf_orbitals(context.run)


def test_vectorized_q0_a_b_structure_and_tangent_column_parity() -> None:
    context = _two_k_context()
    orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, orbitals)

    assert [(pair.particle_momentum, pair.hole_momentum) for pair in matrices.pairs] == [(0, 0), (1, 1)]
    np.testing.assert_allclose(matrices.A, matrices.A.conjugate().T, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(matrices.B, matrices.B.T, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        matrices.L,
        np.block([[matrices.A, matrices.B], [-matrices.B.conjugate(), -matrices.A.conjugate()]]),
        rtol=0.0,
        atol=0.0,
    )
    assert matrices.structure.ok

    parity = validate_tbg_zero_field_tdhf_tangent_columns(
        context,
        orbitals,
        matrices,
        columns=(0, 1),
        tolerance=1.0e-12,
    )
    assert parity.ok
    assert parity.max_a_residual <= 1.0e-12
    assert parity.max_b_residual <= 1.0e-12


def test_dense_complex_fock_only_nontrivial_unitary_tangent_parity() -> None:
    nt = 4
    nk = 1
    beta = 0.7
    v0 = 0.9
    projector = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.complex128)[:, :, None]
    density = conventional_projector_to_stored_density(projector)
    identity2 = np.eye(2, dtype=np.complex128)

    def unitary2(angle: float, phase: float) -> np.ndarray:
        return np.asarray(
            [
                [np.cos(angle), np.sin(angle) * np.exp(1.0j * phase)],
                [-np.sin(angle) * np.exp(-1.0j * phase), np.cos(angle)],
            ],
            dtype=np.complex128,
        )

    cross_block = 0.73 * unitary2(0.41, 0.37)
    dense_hermitian_overlap = np.block(
        [
            [0.61 * identity2, cross_block],
            [cross_block.conjugate().T, -0.27 * identity2],
        ]
    )
    centered_sign = np.diag([1.0, 1.0, -1.0, -1.0])
    overlap_matrices = (
        dense_hermitian_overlap,
        centered_sign @ dense_hermitian_overlap @ centered_sign,
    )
    assert np.max(np.abs(dense_hermitian_overlap.imag)) > 0.1
    shifts = ((0, 0), (1, 0))
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    for shift, matrix in zip(shifts, overlap_matrices, strict=True):
        overlap = np.empty((nt, nk, nt, nk), dtype=np.complex128)
        overlap[:, 0, :, 0] = matrix
        overlaps[shift] = overlap
    blocks = HFOverlapBlockSet(
        shifts=shifts,
        gvecs=np.asarray([0.0 + 0.0j, 0.2 - 0.1j], dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening={
            shift: np.asarray([[weight]], dtype=float)
            for shift, weight in zip(shifts, (0.8, 0.8), strict=True)
        },
    )
    interaction = build_projected_interaction_hamiltonian(
        density,
        blocks,
        v0=v0,
        beta=beta,
    )
    hamiltonian = np.diag([-0.7, -0.7, 0.9, 0.9]).astype(np.complex128)[:, :, None]
    h0 = hamiltonian - interaction
    np.testing.assert_allclose(
        h0,
        np.einsum("iik,ij->ijk", h0, np.eye(nt)),
        rtol=0.0,
        atol=1.0e-13,
    )
    solution = _bm_solution(n_spin=1, n_eta=1, n_band=nt, nk=nk, h0=h0)
    state = RestrictedHartreeFockState(
        h0=h0,
        sigma_z=np.zeros_like(hamiltonian),
        density=density,
        hamiltonian=hamiltonian,
        energies=np.asarray([[-0.7], [-0.7], [0.9], [0.9]], dtype=float),
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=v0,
        n_spin=1,
        n_eta=1,
        n_band=nt,
        diagnostics={"beta": beta},
    )
    state.hf_source_receipt = build_tbg_zero_field_hf_source_receipt(
        hf_mode="full",
        beta=beta,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-1.0], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="dense-complex-fock-only",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    context = _context_from_source(solution, run, label="dense-complex-fock-only")
    base_orbitals = build_tbg_zero_field_tdhf_orbitals(run)
    nontrivial_unitary = np.block(
        [
            [unitary2(0.33, 0.61), np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), unitary2(0.27, -0.47)],
        ]
    )[:, :, None]
    assert np.max(np.abs(nontrivial_unitary.imag)) > 0.1
    orbitals = replace(base_orbitals, eigenvectors=nontrivial_unitary)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, orbitals)
    parity = validate_tbg_zero_field_tdhf_tangent_columns(
        context,
        orbitals,
        matrices,
        columns=tuple(range(len(matrices.pairs))),
        tolerance=2.0e-12,
    )
    assert parity.ok


def test_flavor_symmetric_toy_has_exact_su2_spin_rotation_goldstone() -> None:
    nt = 2
    nk = 1
    interaction_strength = 1.0
    projector = np.diag([1.0, 0.0]).astype(np.complex128)[:, :, None]
    hamiltonian = np.diag([-0.5, 0.5]).astype(np.complex128)[:, :, None]
    overlap = np.eye(nt, dtype=np.complex128)[:, None, :, None]
    blocks = HFOverlapBlockSet(
        shifts=((0, 0),),
        gvecs=np.asarray([0.0 + 0.0j], dtype=np.complex128),
        overlaps={(0, 0): overlap},
        diagonal_overlaps={(0, 0): np.eye(nt, dtype=np.complex128)[:, :, None]},
        hartree_screening={(0, 0): 0.7},
        fock_screening={(0, 0): np.ones((1, 1), dtype=float)},
    )
    solution = _bm_solution(n_spin=2, n_eta=1, n_band=1, nk=1)
    state = RestrictedHartreeFockState(
        h0=np.zeros_like(hamiltonian),
        sigma_z=np.zeros_like(hamiltonian),
        density=conventional_projector_to_stored_density(projector),
        hamiltonian=hamiltonian,
        energies=np.asarray([[-0.5], [0.5]], dtype=float),
        sigma_ztauz=np.zeros((nt, nk), dtype=float),
        nu=0.0,
        v0=interaction_strength,
        n_spin=2,
        n_eta=1,
        n_band=1,
        diagnostics={"beta": 1.0},
    )
    state.hf_source_receipt = build_tbg_zero_field_hf_source_receipt(
        hf_mode="full",
        beta=1.0,
        v0=state.v0,
        lattice_kvec=solution.lattice_kvec,
        overlap_blocks=blocks,
    )
    run = RestrictedHartreeFockRun(
        state=state,
        overlap_blocks=blocks,
        iter_energy=np.asarray([-0.25], dtype=float),
        iter_err=np.asarray([0.0], dtype=float),
        iter_oda=np.asarray([1.0], dtype=float),
        init_mode="spin-polarized-toy",
        seed=0,
        converged=True,
        exit_reason="synthetic-converged",
    )
    context = _context_from_source(solution, run, label="su2-goldstone")

    matrices = build_tbg_zero_field_tdhf_q0_matrices(context)
    assert len(matrices.pairs) == 1
    np.testing.assert_allclose(matrices.A, np.zeros((1, 1)), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(matrices.B, np.zeros((1, 1)), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.eigvals(matrices.L), np.zeros(2), rtol=0.0, atol=1.0e-14)


def test_context_rejects_unscreened_kernel_inventory() -> None:
    _solution, source_run = _two_k_source()
    blocks = replace(
        source_run.overlap_blocks,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening={},
    )
    solution, run = _source_with_overlap_blocks(blocks)
    with pytest.raises(ValueError, match="unscreened"):
        _context_from_source(solution, run)


@pytest.mark.parametrize("active_kind", ["hartree", "fock"])
def test_context_accepts_independently_active_exact_functionals(active_kind: str) -> None:
    _solution, source_run = _two_k_source()
    if active_kind == "hartree":
        blocks = replace(source_run.overlap_blocks, fock_screening={})
    else:
        blocks = replace(
            source_run.overlap_blocks,
            diagonal_overlaps={},
            hartree_screening={},
        )
    solution, run = _source_with_overlap_blocks(blocks)
    context = _context_from_source(solution, run, label=f"{active_kind}-only")
    assert set(context.overlap_blocks.hartree_screening) == (
        {(0, 0)} if active_kind == "hartree" else set()
    )
    assert set(context.overlap_blocks.fock_screening) == (
        {(0, 0)} if active_kind == "fock" else set()
    )


def test_context_rejects_bm_h0_mismatch() -> None:
    solution, run = _two_k_source()
    run.state.h0[0, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="BM source h0 does not match"):
        _context_from_source(solution, run)


def test_context_requires_typed_receipt_and_matching_provenance_fingerprint() -> None:
    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    provenance = _provenance("missing-typed-receipt", receipt)
    run.state.hf_source_receipt = None
    with pytest.raises(ValueError, match="typed TBGZeroFieldHFSourceReceipt"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=provenance,
            closure_tolerance=1.0e-12,
        )

    solution, run = _two_k_source()
    receipt = run.state.hf_source_receipt
    assert isinstance(receipt, TBGZeroFieldHFSourceReceipt)
    provenance = replace(
        _provenance("wrong-expected-fingerprint", receipt),
        expected_hf_source_receipt_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="provenance expected fingerprint"):
        TBGZeroFieldTDHFContext(
            grid_solution=solution,
            run=run,
            beta=receipt.beta,
            provenance=provenance,
            closure_tolerance=1.0e-12,
        )


def test_context_rejects_non_full_hf_mode() -> None:
    solution, run = _two_k_source()
    run.state.hf_source_receipt = replace(
        run.state.hf_source_receipt,
        hf_mode="restricted",
    )
    with pytest.raises(ValueError, match="hf_mode.*full"):
        _context_from_source(solution, run)


def test_context_rejects_wrong_lattice_kvec_source_receipt() -> None:
    solution, run = _two_k_source()
    run.state.hf_source_receipt = replace(
        run.state.hf_source_receipt,
        lattice_kvec_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="lattice_kvec"):
        _context_from_source(solution, run)


def test_context_rejects_source_hamiltonian_closure_failure() -> None:
    solution, run = _two_k_source()
    run.state.hamiltonian[0, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="not closed by h0 plus the saved interaction"):
        _context_from_source(solution, run)


def test_context_rejects_pre_context_v0_substitution_even_if_hamiltonian_is_reclosed() -> None:
    solution, run = _two_k_source()
    state = run.state
    state.v0 += 0.125
    state.hamiltonian[:, :, :] = state.h0 + build_projected_interaction_hamiltonian(
        state.density,
        run.overlap_blocks,
        v0=state.v0,
        beta=float(state.diagnostics["beta"]),
    )

    with pytest.raises(ValueError, match=r"receipt v0 does not match run\.state\.v0 exactly"):
        _context_from_source(solution, run, label="pre-context-v0-substitution")


@pytest.mark.parametrize(
    "mutated_source",
    ["hamiltonian", "density", "fock_kernel", "v0"],
)
def test_matrix_and_parity_calls_reject_post_context_live_source_mutation_with_rebuilt_orbitals(
    mutated_source: str,
) -> None:
    context = _two_k_context()
    original_orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)
    matrices = build_tbg_zero_field_tdhf_q0_matrices(context, original_orbitals)
    state = context.run.state
    if mutated_source == "hamiltonian":
        state.hamiltonian[0, 0, 0] += 1.0e-6
    elif mutated_source == "density":
        state.density[0, 0, 0] = -0.5
        state.density[1, 1, 0] = 0.5
    elif mutated_source == "fock_kernel":
        context.overlap_blocks.fock_screening[(0, 0)][0, 0] += 1.0e-6
    else:
        state.v0 += 1.0e-6
    rebuilt_orbitals = build_tbg_zero_field_tdhf_orbitals(context.run)

    with pytest.raises(ValueError, match="live HF source changed"):
        build_tbg_zero_field_tdhf_q0_matrices(context, rebuilt_orbitals)
    with pytest.raises(ValueError, match="live HF source changed"):
        validate_tbg_zero_field_tdhf_tangent_columns(
            context,
            rebuilt_orbitals,
            matrices,
            columns=(0,),
        )
