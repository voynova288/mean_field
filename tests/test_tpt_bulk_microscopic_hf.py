from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

import mean_field.systems.tpt_bulk as tpt_bulk_api
import mean_field.systems.tpt_bulk.microscopic_hf as microscopic_hf_module
import mean_field.systems.tpt_bulk.source as tpt_bulk_source_module
from mean_field.core.hf.density_vertex import literal_wick_four_index_action
from mean_field.systems.tpt_bulk import (
    TPT_BULK_HR_SHA256,
    TPT_BULK_SOURCE_MESH,
    TPTBulkSource,
    build_tpt_bulk_hf_problem,
    evaluate_tpt_bulk_microscopic_hf,
    load_tpt_bulk_microscopic_hf_inputs,
)
from mean_field.systems.tpt_bulk.microscopic_hf import (
    TPTBulkPhysicalContract,
    _build_tpt_bulk_interaction_model,
    _evaluate_tpt_bulk_interaction,
    _tpt_bulk_interaction_action,
)
from mean_field.systems.tpt_bulk.source import _uniform_k_fractional


def _contract(
    *,
    construction_method: str = "all_electron_real_space_bloch_wavefunction_integral",
) -> TPTBulkPhysicalContract:
    return TPTBulkPhysicalContract(
        construction_method=construction_method,
        density_vertex_definition=(
            "rho_mn(k,q+G)=<psi_m,k+q|exp(i(q+G).r)|psi_n,k>"
        ),
        wavefunction_source_kind="spinor_all_electron_real_space_bloch_wavefunctions",
        wavefunction_source_hashes=("1" * 64,),
        vertex_generator_sha256="2" * 64,
        wavefunction_provenance="manufactured same-parent psi gauge",
        paw_augmentation_provenance=(
            "manufactured all-electron oracle; PAW reconstruction not applicable"
        ),
        local_field_provenance="manufactured complete symmetric G inventory",
        hartree_screening="manufactured W(Q) direct channel",
        fock_screening="manufactured W(Q) exchange channel",
        periodic_coulomb_convention=(
            "3D periodic background removes only total-charge Q=0 Hartree mode"
        ),
        fock_self_cell_convention="explicit manufactured finite Q=0 cell weight",
        normalization_provenance="weights include 1/Nk quadrature and cell volume",
        filling_provenance="two occupied states over the two-point mesh",
        reference_definition="accepted_hr_neutral_fixed_rank_projector",
        double_counting_convention="subtract the same explicit neutral reference",
        ensemble="canonical_zero_temperature_fixed_rank",
        occupation_boundary_status="insulating",
        total_occupied_states=2,
    )


def _manufactured_arrays() -> dict[str, np.ndarray]:
    mesh = (2, 1, 1)
    labels = np.asarray([0, 1, -1, 2, -2], dtype=np.int64)
    q = np.asarray(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]],
        dtype=np.int64,
    )
    g = np.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )
    rho = np.zeros((5, 2, 2, 2), dtype=np.complex128)
    rho[0] = np.eye(2, dtype=np.complex128)[None, :, :]

    a0 = np.asarray([[1.0, 0.2j], [-0.2j, -0.7]], dtype=np.complex128)
    a1 = np.asarray([[0.8, 0.1 + 0.05j], [0.1 - 0.05j, -0.5]])
    rho[1, 0] = a0
    rho[1, 1] = a1
    rho[2, 0] = a0.conj().T
    rho[2, 1] = a1.conj().T

    b0 = np.asarray([[0.7, 0.2 + 0.1j], [-0.3j, 0.4]])
    b1 = np.asarray([[0.6j, -0.15], [0.25 + 0.05j, 0.5]])
    rho[3, 0] = b0
    rho[3, 1] = b1
    # +q maps source 0->1 and 1->0.  The reverse data are indexed by
    # their own source, so rho_-q[1]=rho_+q[0]^dagger and vice versa.
    rho[4, 0] = b1.conj().T
    rho[4, 1] = b0.conj().T

    reference = np.broadcast_to(
        np.diag([1.0, 0.0]).astype(np.complex128), (2, 2, 2)
    ).copy()
    return {
        "mesh": np.asarray(mesh, dtype=np.int64),
        "rho": rho,
        "labels": labels,
        "q": q,
        "g": g,
        "hartree_weight": np.asarray([0.0, 0.6, 0.6, 0.4, 0.4]),
        "fock_weight": np.asarray([0.2, 0.6, 0.6, 0.4, 0.4]),
        "reference": reference,
    }


def _model():
    values = _manufactured_arrays()
    return _build_tpt_bulk_interaction_model(
        mesh=values["mesh"],
        rho=values["rho"],
        channel_labels=values["labels"],
        q_mesh_shifts=values["q"],
        g_integer=values["g"],
        hartree_weight_ev=values["hartree_weight"],
        fock_weight_ev=values["fock_weight"],
        reference_projector_ket=values["reference"],
        contract=_contract(),
        closure_tolerance=1.0e-13,
        provenance="manufactured direct psi/rho fixture",
        require_complete_q_mesh=True,
    )


def _projector() -> np.ndarray:
    reference = _manufactured_arrays()["reference"]
    delta = np.asarray(
        (
            [[0.08, 0.04 + 0.02j], [0.04 - 0.02j, -0.03]],
            [[-0.02, -0.03j], [0.03j, 0.07]],
        ),
        dtype=np.complex128,
    )
    return reference + delta


def test_tpt_bulk_source_mesh_is_accepted_wannier_mesh() -> None:
    assert TPT_BULK_SOURCE_MESH == (12, 4, 4)


def test_tpt_bulk_k_order_matches_wannier90_x_fastest_signed_list() -> None:
    points = _uniform_k_fractional((3, 2, 2))
    np.testing.assert_allclose(
        points[:4],
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0 / 3.0, 0.0, 0.0],
                [-1.0 / 3.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
            ]
        ),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(points[6], [0.0, 0.0, 0.5])


def test_tpt_bulk_public_api_has_no_unsourced_interaction_constructor() -> None:
    assert not hasattr(tpt_bulk_api, "build_tpt_bulk_interaction_model")
    assert not hasattr(tpt_bulk_api, "evaluate_tpt_bulk_interaction")
    assert not hasattr(tpt_bulk_api, "tpt_bulk_interaction_action")


def test_tpt_bulk_public_evaluator_rejects_forged_inputs() -> None:
    forged = SimpleNamespace(
        _authority_token=object(),
        interaction=SimpleNamespace(_authority_token=object()),
        h0_ket_ev=np.zeros((2, 2, 2), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="loader authority"):
        evaluate_tpt_bulk_microscopic_hf(_projector(), forged)
    with pytest.raises(ValueError, match="loader-issued"):
        build_tpt_bulk_hf_problem(
            forged,
            mixing=0.5,
            fermi_degeneracy_tolerance=1.0e-10,
        )


def test_tpt_bulk_contract_rejects_null_provenance() -> None:
    values = {**_contract().__dict__, "hartree_screening": None}
    with pytest.raises(TypeError, match="hartree_screening must be an explicit string"):
        TPTBulkPhysicalContract(**values)


def test_tpt_bulk_rejects_nonmicroscopic_form_factor_constructions() -> None:
    with pytest.raises(ValueError, match="point-centre phases"):
        _contract(construction_method="wannier90_mmn_neighbor_overlap")
    with pytest.raises(ValueError, match="point-centre phases"):
        _contract(construction_method="direct_plane_wave_coefficient_contraction")


def test_tpt_bulk_rho_reverse_closure_is_fail_closed() -> None:
    values = _manufactured_arrays()
    values["rho"][4, 1, 0, 0] += 1.0e-4
    with pytest.raises(ValueError, match="rho_-Q"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"],
            fock_weight_ev=values["fock_weight"],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="broken reverse-closure fixture",
            require_complete_q_mesh=True,
        )


def test_tpt_bulk_q_inventory_must_cover_source_mesh() -> None:
    values = _manufactured_arrays()
    keep = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match="every source-mesh q transfer"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"][keep],
            channel_labels=values["labels"][keep],
            q_mesh_shifts=values["q"][keep],
            g_integer=values["g"][keep],
            hartree_weight_ev=values["hartree_weight"][keep],
            fock_weight_ev=values["fock_weight"][keep],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="incomplete q fixture",
            require_complete_q_mesh=True,
        )


def test_tpt_bulk_rejects_caller_weakened_closure_tolerance() -> None:
    values = _manufactured_arrays()
    with pytest.raises(ValueError, match="code-owned fail-closed maximum"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"],
            fock_weight_ev=values["fock_weight"],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0,
            provenance="weakened tolerance rejection fixture",
            require_complete_q_mesh=True,
        )


def test_tpt_bulk_rejects_complex_coulomb_weights_before_conversion() -> None:
    values = _manufactured_arrays()
    with pytest.raises(TypeError, match="Hartree weights must be a real"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"].astype(np.complex128),
            fock_weight_ev=values["fock_weight"],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="complex weight rejection fixture",
            require_complete_q_mesh=True,
        )

    with pytest.raises(TypeError, match="Fock weights must be a real"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"],
            fock_weight_ev=values["fock_weight"].astype(np.complex128),
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="complex weight rejection fixture",
            require_complete_q_mesh=True,
        )


def test_tpt_bulk_rejects_zero_nonuniform_hartree_or_fock_cell_weight() -> None:
    values = _manufactured_arrays()
    values["hartree_weight"][[1, 2]] = 0.0
    with pytest.raises(ValueError, match="only the uniform Hartree"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"],
            fock_weight_ev=values["fock_weight"],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="zero nonuniform Hartree fixture",
            require_complete_q_mesh=True,
        )

    values = _manufactured_arrays()
    values["fock_weight"][0] = 0.0
    with pytest.raises(ValueError, match="every Fock channel"):
        _build_tpt_bulk_interaction_model(
            mesh=values["mesh"],
            rho=values["rho"],
            channel_labels=values["labels"],
            q_mesh_shifts=values["q"],
            g_integer=values["g"],
            hartree_weight_ev=values["hartree_weight"],
            fock_weight_ev=values["fock_weight"],
            reference_projector_ket=values["reference"],
            contract=_contract(),
            closure_tolerance=1.0e-13,
            provenance="missing Fock self-cell fixture",
            require_complete_q_mesh=True,
        )


def test_tpt_bulk_hartree_removes_only_uniform_charge_mode() -> None:
    model = _model()
    by_h_label = {vertex.label: vertex for vertex in model.hartree_vertices}
    by_f_label = {vertex.label: vertex for vertex in model.fock_vertices}
    assert by_h_label[(0, 0)].weight == 0.0
    assert by_f_label[(0, 0)].weight == pytest.approx(0.2)

    action = _tpt_bulk_interaction_action(_projector(), model)
    assert np.linalg.norm(action.hartree) > 1.0e-8
    np.testing.assert_allclose(action.total, action.hartree + action.fock)

    reference_action = _tpt_bulk_interaction_action(
        model.reference_projector_ket, model
    )
    np.testing.assert_allclose(reference_action.hartree, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(reference_action.fock, 0.0, atol=1.0e-14)


def test_tpt_bulk_sparse_hartree_fock_matches_literal_wick_oracle() -> None:
    model = _model()
    projector = _projector()
    action = _tpt_bulk_interaction_action(projector, model)

    literal_h = literal_wick_four_index_action(
        projector,
        model.hartree_vertices,
        model.hartree_spec,
        model.references,
        max_orbitals=4,
    )
    literal_f = literal_wick_four_index_action(
        projector,
        model.fock_vertices,
        model.fock_spec,
        model.references,
        max_orbitals=4,
    )
    np.testing.assert_allclose(action.hartree, literal_h.hartree, atol=1.0e-14)
    np.testing.assert_allclose(action.fock, literal_f.fock, atol=1.0e-14)


def test_tpt_bulk_three_point_mapping_matches_independent_dense_gamma_oracle() -> None:
    mesh = (3, 1, 1)
    labels = np.asarray([0, 1, -1, 2, -2], dtype=np.int64)
    q = np.asarray(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
        dtype=np.int64,
    )
    g = np.asarray(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]],
        dtype=np.int64,
    )
    rho = np.zeros((5, 3, 2, 2), dtype=np.complex128)
    rho[0] = np.eye(2, dtype=np.complex128)[None, :, :]
    plus = np.asarray(
        [
            [[0.8, 0.1j], [0.2, 0.5]],
            [[0.4, -0.3j], [0.1 + 0.2j, 0.7]],
            [[0.6j, 0.2], [-0.15j, 0.3]],
        ],
        dtype=np.complex128,
    )
    rho[1] = plus
    for source in range(3):
        rho[2, (source + 1) % 3] = plus[source].conj().T
    local_field = np.asarray([[1.0, 0.2j], [-0.2j, -0.4]])
    rho[3] = local_field[None, :, :]
    rho[4] = local_field.conj().T[None, :, :]
    reference = np.broadcast_to(
        np.diag([1.0, 0.0]).astype(np.complex128), (3, 2, 2)
    ).copy()
    model = _build_tpt_bulk_interaction_model(
        mesh=mesh,
        rho=rho,
        channel_labels=labels,
        q_mesh_shifts=q,
        g_integer=g,
        hartree_weight_ev=np.asarray([0.0, 0.4, 0.4, 0.7, 0.7]),
        fock_weight_ev=np.asarray([0.3, 0.4, 0.4, 0.7, 0.7]),
        reference_projector_ket=reference,
        contract=TPTBulkPhysicalContract(
            **{
                **_contract().__dict__,
                "total_occupied_states": 3,
                "filling_provenance": "three occupied states over three K blocks",
            }
        ),
        closure_tolerance=1.0e-13,
        provenance="three-point non-Nyquist direct psi/rho fixture",
        require_complete_q_mesh=True,
    )
    projector = reference + np.asarray(
        [
            [[0.03, 0.02j], [-0.02j, -0.01]],
            [[-0.02, 0.04], [0.04, 0.05]],
            [[0.01, -0.03j], [0.03j, -0.02]],
        ],
        dtype=np.complex128,
    )
    action = _tpt_bulk_interaction_action(projector, model)

    n_global = 6
    delta_global = np.zeros((n_global, n_global), dtype=np.complex128)
    for k_index in range(3):
        block = slice(2 * k_index, 2 * k_index + 2)
        delta_global[block, block] = projector[k_index] - reference[k_index]
    gamma: dict[int, np.ndarray] = {}
    for channel, label in enumerate(labels):
        matrix = np.zeros((n_global, n_global), dtype=np.complex128)
        for source in range(3):
            target = (source + int(q[channel, 0])) % 3
            target_block = slice(2 * target, 2 * target + 2)
            source_block = slice(2 * source, 2 * source + 2)
            matrix[target_block, source_block] = rho[channel, source]
        gamma[int(label)] = matrix

    hartree_global = np.zeros_like(delta_global)
    fock_global = np.zeros_like(delta_global)
    h_weights = dict(zip(labels.tolist(), [0.0, 0.4, 0.4, 0.7, 0.7], strict=True))
    f_weights = dict(zip(labels.tolist(), [0.3, 0.4, 0.4, 0.7, 0.7], strict=True))
    for label in labels:
        integer_label = int(label)
        charge = np.trace(gamma[-integer_label] @ delta_global)
        hartree_global += h_weights[integer_label] * charge * gamma[integer_label]
        fock_global -= (
            f_weights[integer_label]
            * gamma[integer_label]
            @ delta_global
            @ gamma[integer_label].conj().T
        )
    hartree_blocks = np.asarray(
        [hartree_global[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] for k in range(3)]
    )
    fock_blocks = np.asarray(
        [fock_global[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] for k in range(3)]
    )
    np.testing.assert_allclose(action.hartree, hartree_blocks, atol=1.0e-14)
    np.testing.assert_allclose(action.fock, fock_blocks, atol=1.0e-14)


def test_tpt_bulk_energy_derivative_matches_h0_plus_hartree_plus_fock() -> None:
    model = _model()
    projector = _projector()
    h0 = np.asarray(
        (
            [[0.2, 0.03j], [-0.03j, -0.1]],
            [[-0.07, 0.04 + 0.02j], [0.04 - 0.02j, 0.16]],
        ),
        dtype=np.complex128,
    )
    direction = np.asarray(
        (
            [[0.17, -0.04 + 0.09j], [-0.04 - 0.09j, -0.08]],
            [[-0.05, 0.07 - 0.03j], [0.07 + 0.03j, 0.11]],
        ),
        dtype=np.complex128,
    )
    evaluation = _evaluate_tpt_bulk_interaction(projector, h0, model)
    analytic = (
        np.einsum(
            "kab,kba->",
            h0 + evaluation.action.total,
            direction,
            optimize=True,
        ).real
        / model.nk
    )
    step = 2.0e-6
    upper = _evaluate_tpt_bulk_interaction(
        projector + step * direction, h0, model
    ).energy_ev_per_cell
    lower = _evaluate_tpt_bulk_interaction(
        projector - step * direction, h0, model
    ).energy_ev_per_cell
    finite_difference = (upper - lower) / (2.0 * step)
    assert finite_difference == pytest.approx(analytic, abs=2.0e-10)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return _file_sha256(path)


def _write_minimal_loader_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_reference: bool = False,
    wrong_receipt_parent: bool = False,
    occupation_gap_ev: float = 1.9,
) -> tuple[Path, TPTBulkSource]:
    bundle = tmp_path / "bundle"
    authority = bundle / "authority"
    authority.mkdir(parents=True)
    source_root = tmp_path / "source"
    source_root.mkdir()
    hr_path = source_root / "wannier90_hr.dat"
    hr_path.write_bytes(b"synthetic accepted HR bytes\n")
    hr_hash = _file_sha256(hr_path)

    monkeypatch.setattr(microscopic_hf_module, "TPT_BULK_HR_SHA256", hr_hash)
    monkeypatch.setattr(microscopic_hf_module, "TPT_BULK_NUM_WANN", 2)
    monkeypatch.setattr(microscopic_hf_module, "TPT_BULK_SOURCE_MESH", (2, 1, 1))
    monkeypatch.setattr(
        microscopic_hf_module, "TPT_BULK_ACTIVE_RANKS_1BASED", (1, 2)
    )
    monkeypatch.setattr(microscopic_hf_module, "TPT_BULK_ACTIVE_DIMENSION", 2)

    source = TPTBulkSource(
        root=source_root,
        lattice_angstrom=np.eye(3),
        species=("Ta", "Te", "Pd"),
        counts=(8, 20, 12),
        positions_fractional=np.zeros((40, 3)),
        hr_sha256=hr_hash,
    )
    k_fractional = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    active_vectors = np.repeat(np.eye(2, dtype=np.complex128)[:, :, None], 2, axis=2)
    conduction_min = -0.9 + float(occupation_gap_ev)
    eigensystem = tpt_bulk_source_module.TPTBulkActiveEigensystem(
        source=source,
        mesh=(2, 1, 1),
        active_ranks_1based=(1, 2),
        k_fractional=k_fractional,
        active_energies_ev=np.asarray(
            [[-1.0, -0.9], [conduction_min, conduction_min + 0.1]]
        ),
        active_eigenvectors=active_vectors,
        max_antihermitian_before_symmetrization_ev=0.0,
        max_eigenpair_residual_ev=0.0,
    )
    monkeypatch.setattr(
        microscopic_hf_module,
        "build_tpt_bulk_active_eigensystem",
        lambda *_args, **_kwargs: eigensystem,
    )

    psi = np.repeat(np.eye(2, dtype=np.complex128)[None, :, :], 2, axis=0)
    rho = np.repeat(
        np.eye(2, dtype=np.complex128)[None, None, :, :], 3, axis=0
    )
    rho = np.repeat(rho, 2, axis=1)
    reference_one = np.diag([0.0, 1.0] if wrong_reference else [1.0, 0.0])
    reference = np.repeat(reference_one[None, :, :], 2, axis=0).astype(np.complex128)
    data_path = bundle / "microscopic_hf_data.npz"
    np.savez(
        data_path,
        k_fractional=k_fractional,
        psi_wannier=psi,
        rho=rho,
        channel_labels=np.asarray([0, 1, -1], dtype=np.int64),
        q_mesh_shifts=np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64),
        g_integer=np.zeros((3, 3), dtype=np.int64),
        hartree_weight_ev=np.asarray([0.0, 0.4, 0.4]),
        fock_weight_ev=np.asarray([0.2, 0.4, 0.4]),
        reference_projector_ket=reference,
    )
    data_hash = _file_sha256(data_path)

    wavefunction_path = authority / "spinor_psi.bin"
    wavefunction_path.write_bytes(b"synthetic spinor wavefunctions\n")
    wavefunction_hash = _file_sha256(wavefunction_path)
    generator_path = authority / "generate_density_vertices.py"
    generator_path.write_text("# synthetic direct contraction generator\n")
    generator_hash = _file_sha256(generator_path)

    active_path = authority / "ACTIVE_SPACE_AUTHORITY.json"
    active_hash = _write_json(
        active_path,
        {
            "schema": "tpt-bulk-active-space-authority-v1",
            "status": "validated_for_microscopic_hf",
            "parent_hr_sha256": hr_hash,
            "active_ranks_1based": [1, 2],
            "gauge": "psi_wannier_columns_in_accepted_parent_active_basis",
        },
    )
    filling_provenance = "synthetic neutral one-band-per-k authority"
    filling_path = authority / "NEUTRAL_FILLING_AUTHORITY.json"
    filling_hash = _write_json(
        filling_path,
        {
            "schema": "tpt-bulk-neutral-filling-authority-v1",
            "status": "neutral_insulating_fixed_rank",
            "parent_hr_sha256": hr_hash,
            "source_mesh": [2, 1, 1],
            "ensemble": "canonical_zero_temperature_fixed_rank",
            "reference_definition": "accepted_hr_neutral_fixed_rank_projector",
            "total_occupied_states": 2,
            "provenance": filling_provenance,
        },
    )
    wavefunction_hashes = {"authority/spinor_psi.bin": wavefunction_hash}
    receipt_path = authority / "DENSITY_VERTEX_RECEIPT.json"
    receipt_hash = _write_json(
        receipt_path,
        {
            "schema": "tpt-bulk-density-vertex-receipt-v2",
            "status": "complete",
            "parent_hr_sha256": (
                "0" * 64 if wrong_receipt_parent else hr_hash
            ),
            "payload_sha256": data_hash,
            "generator_sha256": generator_hash,
            "wavefunction_source_sha256": wavefunction_hashes,
            "density_vertex_definition": (
                "rho_mn(k,q+G)=<psi_m,k+q|exp(i(q+G).r)|psi_n,k>"
            ),
            "construction_method": (
                "all_electron_real_space_bloch_wavefunction_integral"
            ),
            "paw_augmentation_provenance": (
                "synthetic all-electron oracle; PAW reconstruction not applicable"
            ),
            "source_mesh": [2, 1, 1],
            "active_ranks_1based": [1, 2],
            "k_ordering": "wannier90_x_fastest_signed_fractional",
        },
    )
    manifest = {
        "format_version": "tpt-bulk-microscopic-hf-v2",
        "parent": {
            "hr_sha256": hr_hash,
            "num_wann": 2,
            "source_mesh": [2, 1, 1],
            "active_ranks_1based": [1, 2],
            "gauge": "psi_wannier_columns_in_accepted_parent_active_basis",
            "k_ordering": "wannier90_x_fastest_signed_fractional",
            "active_space_authority": {
                "path": "authority/ACTIVE_SPACE_AUTHORITY.json",
                "sha256": active_hash,
            },
        },
        "files": {"microscopic_hf_data.npz": {"sha256": data_hash}},
        "physics": {
            "construction_method": (
                "all_electron_real_space_bloch_wavefunction_integral"
            ),
            "density_vertex_definition": (
                "rho_mn(k,q+G)=<psi_m,k+q|exp(i(q+G).r)|psi_n,k>"
            ),
            "wavefunction_sources": {
                "source_kind": (
                    "spinor_all_electron_real_space_bloch_wavefunctions"
                ),
                "sha256": wavefunction_hashes,
            },
            "vertex_generator": {
                "path": "authority/generate_density_vertices.py",
                "sha256": generator_hash,
            },
            "vertex_receipt": {
                "path": "authority/DENSITY_VERTEX_RECEIPT.json",
                "sha256": receipt_hash,
            },
            "filling_authority": {
                "path": "authority/NEUTRAL_FILLING_AUTHORITY.json",
                "sha256": filling_hash,
            },
            "wavefunction_provenance": "synthetic same-parent spinor psi",
            "paw_augmentation_provenance": (
                "synthetic all-electron oracle; PAW reconstruction not applicable"
            ),
            "local_field_provenance": "synthetic complete G inventory",
            "screening": {
                "hartree_screening": "synthetic direct weights",
                "fock_screening": "synthetic exchange weights",
            },
            "periodic_coulomb_convention": "synthetic 3D neutral background",
            "fock_self_cell_convention": "synthetic positive Q=0 self cell",
            "normalization_provenance": "synthetic eV per cell weights",
            "filling_provenance": filling_provenance,
            "reference_definition": "accepted_hr_neutral_fixed_rank_projector",
            "double_counting_convention": "subtract the common neutral projector",
            "ensemble": "canonical_zero_temperature_fixed_rank",
            "occupation_boundary_status": "insulating",
            "total_occupied_states": 2,
            "hartree_reference_policy": "subtract_explicit",
            "fock_reference_policy": "subtract_explicit",
            "zero_mode_policy": "background_removed_hartree_only",
            "local_field_inventory_complete": True,
            "q_mesh_inventory_complete": True,
            "closure_tolerance": 1.0e-12,
        },
    }
    manifest_hash = _write_json(bundle / "MICROSCOPIC_HF_MANIFEST.json", manifest)
    monkeypatch.setattr(
        microscopic_hf_module,
        "TPT_BULK_APPROVED_MICROSCOPIC_MANIFEST_SHA256",
        frozenset({manifest_hash}),
    )
    return bundle, source


def _rewrite_bundle_as_vasp_paw(
    bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_payloads = {
        "authority/WAVECAR": b"synthetic ncl wavecar\n",
        "authority/POTCAR": b"synthetic paw potentials\n",
        "authority/POSCAR": b"synthetic bulk structure\n",
        "authority/wannier90.chk": b"synthetic checkpoint\n",
        "authority/wannier90.eig": b"synthetic eig table\n",
        "authority/wannier90.mmn": b"synthetic paw overlap oracle\n",
    }
    source_hashes: dict[str, str] = {}
    for relative, payload in source_payloads.items():
        path = bundle / relative
        path.write_bytes(payload)
        source_hashes[relative] = _file_sha256(path)
    for attribute, relative in (
        ("TPT_BULK_WAVECAR_SHA256", "authority/WAVECAR"),
        ("TPT_BULK_POTCAR_SHA256", "authority/POTCAR"),
        ("TPT_BULK_POSCAR_SHA256", "authority/POSCAR"),
        ("TPT_BULK_WANNIER90_CHK_SHA256", "authority/wannier90.chk"),
        ("TPT_BULK_WANNIER90_EIG_SHA256", "authority/wannier90.eig"),
        ("TPT_BULK_WANNIER90_MMN_SHA256", "authority/wannier90.mmn"),
    ):
        monkeypatch.setattr(
            microscopic_hf_module,
            attribute,
            source_hashes[relative],
        )

    receipt_path = bundle / "authority/DENSITY_VERTEX_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["construction_method"] = "vasp_paw_augmented_plane_wave_contraction"
    receipt["wavefunction_source_sha256"] = source_hashes
    receipt["paw_augmentation_provenance"] = (
        "synthetic VASP-compatible one-centre augmentation oracle"
    )
    receipt_hash = _write_json(receipt_path, receipt)

    manifest_path = bundle / "MICROSCOPIC_HF_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    physics = manifest["physics"]
    physics["construction_method"] = "vasp_paw_augmented_plane_wave_contraction"
    physics["wavefunction_sources"] = {
        "source_kind": "vasp_ncl_wavecar_potcar_paw",
        "sha256": source_hashes,
    }
    physics["paw_augmentation_provenance"] = (
        "synthetic VASP-compatible one-centre augmentation oracle"
    )
    physics["vertex_receipt"]["sha256"] = receipt_hash
    manifest_hash = _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        microscopic_hf_module,
        "TPT_BULK_APPROVED_MICROSCOPIC_MANIFEST_SHA256",
        frozenset({manifest_hash}),
    )


def test_tpt_bulk_loader_closes_hash_bound_authority_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(tmp_path, monkeypatch)
    real_np_load = np.load
    load_inputs: list[object] = []

    def recording_np_load(file: object, *args: object, **kwargs: object):
        load_inputs.append(file)
        return real_np_load(file, *args, **kwargs)

    monkeypatch.setattr(microscopic_hf_module.np, "load", recording_np_load)
    inputs = load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)
    assert inputs.occupation_boundary_gap_ev == pytest.approx(1.9)
    assert load_inputs and hasattr(load_inputs[0], "read")
    evaluated = evaluate_tpt_bulk_microscopic_hf(
        inputs.interaction.reference_projector_ket,
        inputs,
    )
    assert evaluated.energy_ev_per_cell == pytest.approx(-0.95)


def test_tpt_bulk_loader_accepts_exact_vasp_paw_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(tmp_path, monkeypatch)
    _rewrite_bundle_as_vasp_paw(bundle, monkeypatch)
    inputs = load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)
    assert (
        inputs.interaction.contract.construction_method
        == "vasp_paw_augmented_plane_wave_contraction"
    )


def test_tpt_bulk_loader_rejects_missing_vasp_paw_authority_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(tmp_path, monkeypatch)
    _rewrite_bundle_as_vasp_paw(bundle, monkeypatch)
    (bundle / "authority/POTCAR").unlink()
    with pytest.raises(FileNotFoundError, match="missing source-bound authority"):
        load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)


@pytest.mark.parametrize("gap_ev", [1.0e-6, 0.5e-6])
def test_tpt_bulk_loader_rejects_gap_at_or_below_code_minimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gap_ev: float,
) -> None:
    bundle, source = _write_minimal_loader_bundle(
        tmp_path,
        monkeypatch,
        occupation_gap_ev=gap_ev,
    )
    with pytest.raises(ValueError, match="minimum insulating gap"):
        load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)


def test_tpt_bulk_loader_accepts_gap_above_code_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(
        tmp_path,
        monkeypatch,
        occupation_gap_ev=1.1e-6,
    )
    inputs = load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)
    assert inputs.occupation_boundary_gap_ev == pytest.approx(1.1e-6)


def test_tpt_bulk_loader_requires_nonbundle_manifest_trust_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        microscopic_hf_module,
        "TPT_BULK_APPROVED_MICROSCOPIC_MANIFEST_SHA256",
        frozenset(),
    )
    with pytest.raises(ValueError, match="external trust root"):
        load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)


def test_tpt_bulk_loader_parses_and_rejects_wrong_vertex_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(
        tmp_path, monkeypatch, wrong_receipt_parent=True
    )
    with pytest.raises(ValueError, match="receipt is bound to a different parent"):
        load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)


def test_tpt_bulk_loader_rejects_reference_not_equal_to_hr_occupied_projector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, source = _write_minimal_loader_bundle(
        tmp_path, monkeypatch, wrong_reference=True
    )
    with pytest.raises(ValueError, match="not the accepted-HR neutral occupied projector"):
        load_tpt_bulk_microscopic_hf_inputs(bundle, source=source)


def test_tpt_bulk_hr_parser_hashes_before_parsing_same_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tpt_bulk_source_module, "TPT_BULK_NUM_WANN", 1)
    monkeypatch.setattr(tpt_bulk_source_module, "TPT_BULK_NUM_RPTS", 1)
    path = tmp_path / "tiny_hr.dat"
    path.write_text("synthetic\n1\n1\n1\n0 0 0 1 1 2.0 0.0\n")
    digest = _file_sha256(path)
    grid, num_wann, nrpts = tpt_bulk_source_module._parse_hr_mod_grid(
        path, (1, 1, 1), expected_sha256=digest
    )
    assert (num_wann, nrpts) == (1, 1)
    assert grid[0, 0, 0, 0, 0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="changed before eigensystem"):
        tpt_bulk_source_module._parse_hr_mod_grid(
            path, (1, 1, 1), expected_sha256="0" * 64
        )


def test_tpt_bulk_loader_stops_before_hr_only_fallback(tmp_path: Path) -> None:
    dummy_source = TPTBulkSource(
        root=tmp_path,
        lattice_angstrom=np.eye(3),
        species=("Ta", "Te", "Pd"),
        counts=(8, 20, 12),
        positions_fractional=np.zeros((40, 3)),
        hr_sha256=TPT_BULK_HR_SHA256,
    )
    with pytest.raises(FileNotFoundError, match="microscopic manifest"):
        load_tpt_bulk_microscopic_hf_inputs(tmp_path, source=dummy_source)
