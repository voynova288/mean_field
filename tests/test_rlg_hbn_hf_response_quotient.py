from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.systems.RnG_hBN import (
    RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
    RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
    RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    RLGhBNHFInteractionProvenance,
    RLGhBNHartreeFockRun,
    RLGhBNHartreeFockState,
    RLGhBNInteractionParams,
    RLGhBNModel,
    build_rlg_hbn_hf_c3_quotient_interaction_components,
    build_rlg_hbn_hf_c3_quotient_interaction_context,
    build_rlg_hbn_hf_problem,
    build_rlg_hbn_layer_overlap_blocks,
    build_rlg_hbn_projected_basis,
    build_rlg_hbn_tdhf_c3_quotient_cycle,
    build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
 build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs,
 build_rlg_hbn_tdhf_finite_q_quotient_context,
 build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs,
 build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs,
 build_rlg_hbn_tdhf_orbitals,
    build_rlg_hbn_tdhf_q_pairs,
    rlg_hbn_flavor_occupation_counts_for_init_mode,
)
from mean_field.systems.RnG_hBN._hf_c3_quotient import (
    RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
    _contract_hartree_between,
    _contract_ws_fock,
    _expanded_fixed_density,
)
from mean_field.systems.RnG_hBN._hf_response_finite_q import (
    RLGhBNFiniteQDensityTangent,
    apply_rlg_hbn_hf_quotient_response,
)
from mean_field.systems.RnG_hBN._tdhf_archive import (
    _archive_interaction_provenance,
)
from mean_field.systems.RnG_hBN import _tdhf_fixed_quotient as fixed_quotient_module
from mean_field.systems.RnG_hBN._tdhf_fixed_quotient import (
 _expanded_node_transfer_vector,
 fixed_role_masks,
 sparse_quotient_leg_vector,
)


MESH = 3
PHYSICAL_SHIFTS = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, -1),
    (1, 1),
    (-1, 0),
    (0, -1),
)
TOLERANCE_MEV = 1.0e-10


def _build_reduced_run() -> RLGhBNHartreeFockRun:
    model = RLGhBNModel.from_config(
        layer_count=3,
        xi=1,
        theta_deg=0.77,
        displacement_field_mev=24.0,
        shell_count=1,
    )
    interaction = RLGhBNInteractionParams(
        active_valence_bands=0,
        active_conduction_bands=2,
        k_mesh_size=MESH,
        interaction_cutoff_q1=1.0,
        use_screened_basis=False,
    )
    basis_data = build_rlg_hbn_projected_basis(model, interaction, mesh_size=MESH)
    overlap_blocks = build_rlg_hbn_layer_overlap_blocks(
        basis_data,
        shifts=PHYSICAL_SHIFTS,
    )
    counts = rlg_hbn_flavor_occupation_counts_for_init_mode(
        "flavor",
        nu=1.0,
        active_valence_bands=basis_data.interaction.active_valence_bands,
        n_spin=basis_data.basis.n_spin,
        n_eta=basis_data.basis.n_flavor,
        n_band=basis_data.basis.n_band,
    )
    state = RLGhBNHartreeFockState.from_projected_basis(
        basis_data,
        nu=1.0,
        occupation_counts=counts,
    )
    problem = build_rlg_hbn_hf_problem(state, overlap_blocks)
    problem.initializer(state, init_mode="flavor", seed=1)
    update = problem.kernel.density_builder(state.h0)
    state.density[:, :, :] = update.density
    state.hamiltonian[:, :, :] = state.h0
    state.energies[:, :] = update.energies
    state.mu = update.mu
    return RLGhBNHartreeFockRun(
        state=state,
        iter_energy=(),
        iter_err=(),
        iter_oda=(),
        init_mode="flavor",
        seed=1,
        converged=False,
        exit_reason="reduced-q0-hessian-diagnostic",
        overlap_blocks=overlap_blocks,
        basis_data=basis_data,
    )


def _intraflavor_pairs(run, orbitals, q_shift=(0, 0)):
    result = []
    for pair in build_rlg_hbn_tdhf_q_pairs(orbitals, run.basis_data, q_shift):
        if pair.particle_flavor is None or pair.hole_flavor is None:
            raise AssertionError("finite-q pair lacks flavor metadata")
        if (
            pair.particle_flavor.spin == pair.hole_flavor.spin
            and pair.particle_flavor.valley == pair.hole_flavor.valley
        ):
            result.append(pair)
    return tuple(result)


def _q0_tangent(run, orbitals, pair, role: str) -> RLGhBNFiniteQDensityTangent:
    p_local, p_k = orbitals.decode_global_index(pair.particle)
    h_local, h_k = orbitals.decode_global_index(pair.hole)
    if int(p_k) != int(h_k):
        raise AssertionError(f"q0 pair connects different k: {p_k} != {h_k}")
    u_p = np.asarray(orbitals.eigenvectors[:, p_local, p_k], dtype=np.complex128)
    u_h = np.asarray(orbitals.eigenvectors[:, h_local, h_k], dtype=np.complex128)
    blocks = np.zeros(
        (run.basis_data.nt, run.basis_data.nt, run.basis_data.nk),
        dtype=np.complex128,
    )
    if role == "ph":
        blocks[:, :, h_k] = np.outer(np.conj(u_h), u_p)
    elif role == "hp":
        blocks[:, :, h_k] = np.outer(np.conj(u_p), u_h)
    else:
        raise ValueError(role)
    k_indices = np.arange(run.basis_data.nk, dtype=int)
    return RLGhBNFiniteQDensityTangent(
        q_shift=(0, 0),
        target_k=k_indices,
        source_k=k_indices,
        blocks=blocks,
        role=role,  # type: ignore[arg-type]
    )


def _project_response(orbitals, row_pair, response: np.ndarray) -> complex:
    p_local, p_k = orbitals.decode_global_index(row_pair.particle)
    h_local, h_k = orbitals.decode_global_index(row_pair.hole)
    if int(p_k) != int(h_k):
        raise AssertionError("q0 response row is not momentum diagonal")
    u_p = np.asarray(orbitals.eigenvectors[:, p_local, p_k], dtype=np.complex128)
    u_h = np.asarray(orbitals.eigenvectors[:, h_local, h_k], dtype=np.complex128)
    return complex(np.vdot(u_p, response[:, :, p_k] @ u_h))


def _max_abs(values: np.ndarray) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _explicit_copy_density(
    density: np.ndarray,
    transforms: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [transform.conj() @ density @ transform.T / 3.0 for transform in transforms],
        axis=2,
    )


def _explicit_copy_descent(
    hamiltonian: np.ndarray,
    transforms: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    return sum(
        (
            transform.conj().T
            @ np.asarray(hamiltonian)[:, :, copy]
            @ transform
            / 3.0
        )
        for copy, transform in enumerate(transforms)
    )


def _explicit_variational_copy_oracle(run, context, density: np.ndarray):
    """Assemble L^sharp V L without calling the production quotient builder."""

    values = np.asarray(density, dtype=np.complex128)
    scale = float(run.state.v0) / float(run.basis_data.nk)
    ordinary_density = values.copy()
    ordinary_density[:, :, np.asarray(context.fixed_indices, dtype=int)] = 0.0
    expanded = tuple(
        _explicit_copy_density(
            values[:, :, source.source_index],
            source.copy_transforms,
        )
        for source in context.fixed_sources
    )
    target_shape = values.shape
    base_hartree = _contract_hartree_between(
        ordinary_density,
        context.base_blocks,
        context.base_blocks,
        context.physical_shifts,
        scale=scale,
        target_shape=target_shape,
    )
    base_fock = _contract_ws_fock(
        ordinary_density,
        context.ordinary_fock_overlaps,
        context.ordinary_fock_kernels,
        context.ordinary_fock_weights,
        scale=scale,
        target_shape=target_shape,
    )
    for source, source_density in zip(context.fixed_sources, expanded, strict=True):
        sample_overlap = next(iter(source.fock_overlaps.values()))
        if sample_overlap.shape[2] != run.basis_data.nk:
            raise AssertionError("ordinary-target/fixed-source overlap target axis is wrong")
        if sample_overlap.shape[4] != source.source_basis.nk:
            raise AssertionError("ordinary-target/fixed-source overlap source axis is wrong")
        base_hartree += _contract_hartree_between(
            source_density,
            context.base_blocks,
            source.hartree_blocks,
            context.physical_shifts,
            scale=scale,
            target_shape=target_shape,
        )
        base_fock += _contract_ws_fock(
            source_density,
            source.fock_overlaps,
            source.fock_kernels,
            source.fock_weights,
            scale=scale,
            target_shape=target_shape,
        )

    descended_hartree = np.array(base_hartree, copy=True)
    descended_fock = np.array(base_fock, copy=True)
    physical_targets = []
    for target in context.fixed_targets:
        copy_shape = (target.target_basis.nt, target.target_basis.nt, 3)
        ordinary_overlap = next(iter(target.ordinary_fock.overlaps.values()))
        if ordinary_overlap.shape[2] != target.target_basis.nk:
            raise AssertionError("fixed-target/ordinary-source overlap target axis is wrong")
        if ordinary_overlap.shape[4] != run.basis_data.nk:
            raise AssertionError("fixed-target/ordinary-source overlap source axis is wrong")
        physical_hartree = _contract_hartree_between(
            ordinary_density,
            target.hartree_blocks,
            context.base_blocks,
            context.physical_shifts,
            scale=scale,
            target_shape=copy_shape,
        )
        physical_fock = _contract_ws_fock(
            ordinary_density,
            target.ordinary_fock.overlaps,
            target.ordinary_fock.kernels,
            target.ordinary_fock.weights,
            scale=scale,
            target_shape=copy_shape,
        )
        for source, source_density, fock_context in zip(
            context.fixed_sources,
            expanded,
            target.fixed_fock,
            strict=True,
        ):
            physical_hartree += _contract_hartree_between(
                source_density,
                target.hartree_blocks,
                source.hartree_blocks,
                context.physical_shifts,
                scale=scale,
                target_shape=copy_shape,
            )
            physical_fock += _contract_ws_fock(
                source_density,
                fock_context.overlaps,
                fock_context.kernels,
                fock_context.weights,
                scale=scale,
                target_shape=copy_shape,
            )
        descended_hartree[:, :, target.target_index] = _explicit_copy_descent(
            physical_hartree,
            target.copy_transforms,
        )
        descended_fock[:, :, target.target_index] = _explicit_copy_descent(
            physical_fock,
            target.copy_transforms,
        )
        physical_targets.append((physical_hartree, physical_fock))
    return {
        "hartree": descended_hartree,
        "fock": descended_fock,
        "total": descended_hartree + descended_fock,
        "base_hartree": base_hartree,
        "base_fock": base_fock,
        "physical_targets": tuple(physical_targets),
        "expanded": expanded,
    }


def test_rlg_hbn_fixed_copy_lift_preserves_stored_density_convention() -> None:
    """A ket sewing S acts on stored ΔD_ab=<c_a†c_b>-R_ab as S* ΔD S^T."""

    vector = np.asarray([1.0 + 2.0j, -0.3 + 0.7j], dtype=np.complex128)
    vector /= np.linalg.norm(vector)
    sewing = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]],
        dtype=np.complex128,
    ) / np.sqrt(2.0)
    np.testing.assert_allclose(
        sewing.conj().T @ sewing,
        np.eye(2),
        rtol=0.0,
        atol=1.0e-15,
    )
    stored_density = np.outer(np.conj(vector), vector)
    transformed_vector = sewing @ vector
    expected_copy = np.outer(np.conj(transformed_vector), transformed_vector) / 3.0
    context = SimpleNamespace(
        source_basis=SimpleNamespace(nt=2),
        copy_transforms=(np.eye(2, dtype=np.complex128), sewing, sewing @ sewing),
    )

    expanded = _expanded_fixed_density(stored_density, context)

    np.testing.assert_allclose(expanded[:, :, 1], expected_copy, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.trace(expanded, axis1=0, axis2=1).sum(),
        np.trace(stored_density),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_rlg_hbn_archive_interaction_provenance_is_typed_and_fail_closed() -> None:
    archive = {
        "hf_interaction_convention": np.asarray(
            RLG_HBN_HF_INTERACTION_CONVENTION_VERSION
        ),
        "hf_quotient_enabled": np.asarray([True], dtype=bool),
        "hf_beta": np.asarray([1.0]),
        "hf_physical_shifts": np.asarray(PHYSICAL_SHIFTS, dtype=int),
        "zero_literal_q0_fock": np.asarray([False], dtype=bool),
        "hf_basis_periodic_gauge": np.asarray(
            RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION
        ),
        "hf_basis_periodic_gauge_padding": np.asarray(
            [RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING],
            dtype=int,
        ),
        "hf_form_factor_convention": np.asarray(
            RLG_HBN_FORM_FACTOR_CONVENTION_VERSION
        ),
    }
    provenance = _archive_interaction_provenance(
        archive,
        basis_cache_key="basis-test",
        overlap_cache_key="overlap-test",
    )
    assert provenance is not None
    assert provenance.convention == RLG_HBN_HF_INTERACTION_CONVENTION_VERSION
    assert provenance.quotient_enabled
    assert provenance.physical_shifts == PHYSICAL_SHIFTS
    assert provenance.basis_cache_key == "basis-test"
    assert provenance.overlap_cache_key == "overlap-test"
    assert _archive_interaction_provenance(
        {},
        basis_cache_key="basis-test",
        overlap_cache_key="overlap-test",
    ) is None
    incomplete = dict(archive)
    incomplete.pop("hf_beta")
    with pytest.raises(ValueError, match="partial interaction provenance"):
        _archive_interaction_provenance(
            incomplete,
            basis_cache_key="basis-test",
            overlap_cache_key="overlap-test",
        )


def test_expanded_node_transfer_canonicalizes_analytic_zero() -> None:
 lattice = SimpleNamespace(g_m1=0.7 + 0.2j, g_m2=-0.3 + 0.8j)
 fractional = np.asarray(
  [
   [2.0 / 3.0, 1.0],
   [2.0 / 3.0, 0.0],
   [2.0 / 3.0, -1.0],
  ],
  dtype=float,
 )
 forward = _expanded_node_transfer_vector(
  lattice,
  fractional,
  0,
  1,
  (0, -1),
  tolerance=1.0e-12,
 )
 reverse = _expanded_node_transfer_vector(
  lattice,
  fractional,
  2,
  1,
  (0, 1),
  tolerance=1.0e-12,
 )
 assert forward == 0.0j
 assert reverse == 0.0j


def test_sparse_quotient_leg_uses_analytic_periodic_lift_for_ordinary_wrap(
 monkeypatch,
) -> None:
 sentinel = np.asarray([1.0 + 2.0j, 3.0 - 4.0j])
 calls = []

 def fake_periodic_lift(
  basis_data,
  orbitals,
  *,
  local_index: int,
  k_index: int,
  wrap: tuple[int, int],
 ) -> np.ndarray:
  calls.append((basis_data, orbitals, local_index, k_index, wrap))
  return sentinel

 monkeypatch.setattr(
  fixed_quotient_module,
  "_hf_full_vector_in_periodic_gauge",
  fake_periodic_lift,
 )
 component_shape = (7, 1, 1, 1)
 periodic_basis = SimpleNamespace(
  basis=SimpleNamespace(wavefunctions=np.zeros(component_shape)),
 )
 context = SimpleNamespace(
  basis_data=SimpleNamespace(
   basis=SimpleNamespace(wavefunctions=np.zeros(component_shape)),
  ),
  builder=SimpleNamespace(
   node_keys=(
    SimpleNamespace(
     stored_k=5,
     wrap=(1, -1),
     reciprocal_shift=(1, -1),
    ),
   ),
  ),
 )
 orbitals = object()
 result = sparse_quotient_leg_vector(
  context,
  periodic_basis,
  orbitals,
  local_index=3,
  node=0,
  fixed_transform_maps={},
 )
 assert result is sentinel
 assert calls == [(periodic_basis, orbitals, 3, 5, (1, -1))]


@pytest.mark.parametrize(
 "builder",
 (
  build_rlg_hbn_tdhf_finite_q_exchange_matrices_from_pairs,
  build_rlg_hbn_tdhf_finite_q_intraflavor_matrices_from_pairs,
 ),
)
def test_typed_variational_hf_source_rejects_legacy_pair_assembly(builder) -> None:
 run = _build_reduced_run()
 context = build_rlg_hbn_hf_c3_quotient_interaction_context(
  run.basis_data,
  run.overlap_blocks,
 )
 run = replace(
  run,
  interaction_provenance=RLGhBNHFInteractionProvenance(
   convention=RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
   quotient_enabled=True,
   beta=1.0,
   physical_shifts=tuple(context.physical_shifts),
   zero_literal_q0_fock=False,
   basis_periodic_gauge=RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
   basis_periodic_gauge_padding=RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
   form_factor_convention=RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
  ),
 )
 orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
 with pytest.raises(NotImplementedError, match="finite_q_quotient_matrices"):
  builder(run, orbitals, (), (1, 0))


def test_rlg_hbn_variational_copy_builder_matches_expanded_space_oracle() -> None:
    run = _build_reduced_run()
    context = build_rlg_hbn_hf_c3_quotient_interaction_context(
        run.basis_data,
        run.overlap_blocks,
    )
    rng = np.random.default_rng(20260716)
    shape = run.state.density.shape
    fixed_indices = tuple(int(index) for index in context.fixed_indices)
    ordinary_indices = tuple(
        index for index in range(run.basis_data.nk) if index not in fixed_indices
    )
    if not fixed_indices or not ordinary_indices:
        raise AssertionError("expanded-space oracle requires ordinary and fixed nodes")

    cases: dict[str, np.ndarray] = {}
    supports = {
        "ordinary_ordinary": (ordinary_indices[0],),
        "fixed0_all_targets": (fixed_indices[0],),
        "fixed1_all_targets": (fixed_indices[1],),
        "mixed": (ordinary_indices[1], *fixed_indices),
    }
    for name, support in supports.items():
        density = np.zeros(shape, dtype=np.complex128)
        for index in support:
            density[:, :, index] = 0.1 * (
                rng.normal(size=shape[:2]) + 1.0j * rng.normal(size=shape[:2])
            )
        cases[name] = density

    residuals: dict[str, float] = {}
    energy_residuals: dict[str, float] = {}
    for case_name, density in cases.items():
        production = build_rlg_hbn_hf_c3_quotient_interaction_components(
            density,
            context,
            v0=run.state.v0,
        )
        oracle = _explicit_variational_copy_oracle(run, context, density)
        for component_name in ("hartree", "fock", "total"):
            production_component = np.asarray(getattr(production, component_name))
            oracle_component = np.asarray(oracle[component_name])
            for target_name, target_indices in (
                ("ordinary", ordinary_indices),
                ("fixed", fixed_indices),
            ):
                residuals[
                    f"{case_name}.{component_name}.{target_name}"
                ] = _max_abs(
                    production_component[:, :, target_indices]
                    - oracle_component[:, :, target_indices]
                )

            active_energy = 0.5 * np.einsum(
                "abk,abk->",
                production_component,
                density,
                optimize=True,
            )
            base_component = np.asarray(
                oracle["base_hartree" if component_name == "hartree" else "base_fock"]
            )
            if component_name == "total":
                base_component = np.asarray(oracle["base_hartree"]) + np.asarray(
                    oracle["base_fock"]
                )
            physical_energy = 0.5 * np.einsum(
                "abk,abk->",
                base_component[:, :, ordinary_indices],
                density[:, :, ordinary_indices],
                optimize=True,
            )
            for target_position, target in enumerate(context.fixed_targets):
                source = context.fixed_sources[target_position]
                if target.pair != source.pair:
                    raise AssertionError("fixed target/source context order changed")
                physical_pair = oracle["physical_targets"][target_position]
                physical_component = np.asarray(
                    physical_pair[0] if component_name == "hartree" else physical_pair[1]
                )
                if component_name == "total":
                    physical_component = np.asarray(physical_pair[0]) + np.asarray(
                        physical_pair[1]
                    )
                physical_energy += 0.5 * np.einsum(
                    "abr,abr->",
                    physical_component,
                    oracle["expanded"][target_position],
                    optimize=True,
                )
            energy_residuals[f"{case_name}.{component_name}"] = float(
                abs(active_energy - physical_energy)
            )

    d1 = cases["mixed"]
    d2 = cases["ordinary_ordinary"] + cases["fixed1_all_targets"]
    k1 = build_rlg_hbn_hf_c3_quotient_interaction_components(
        d1,
        context,
        v0=run.state.v0,
    )
    k2 = build_rlg_hbn_hf_c3_quotient_interaction_components(
        d2,
        context,
        v0=run.state.v0,
    )
    bilinear_residuals = {
        name: float(
            abs(
                np.einsum("abk,abk->", getattr(k1, name), d2, optimize=True)
                - np.einsum("abk,abk->", getattr(k2, name), d1, optimize=True)
            )
        )
        for name in ("hartree", "fock", "total")
    }
    summary = {
        "hf_interaction_convention": RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
        "production_vs_expanded_oracle_mev": residuals,
        "active_vs_expanded_energy_mev": energy_residuals,
        "bilinear_self_adjoint_residual_mev": bilinear_residuals,
    }
    output = os.environ.get("MEAN_FIELD_RLG_HBN_Q0_EXPANDED_ORACLE_JSON")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    assert max(residuals.values(), default=0.0) <= TOLERANCE_MEV
    assert max(energy_residuals.values(), default=0.0) <= TOLERANCE_MEV
    assert max(bilinear_residuals.values(), default=0.0) <= TOLERANCE_MEV


def test_rlg_hbn_q0_hf_quotient_response_is_the_corrected_hessian() -> None:
    run = _build_reduced_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pairs = _intraflavor_pairs(run, orbitals)
    context = build_rlg_hbn_hf_c3_quotient_interaction_context(
        run.basis_data,
        run.overlap_blocks,
    )
    pair_count = len(pairs)
    response_terms = {
        name: np.zeros((pair_count, pair_count), dtype=np.complex128)
        for name in ("A_direct", "A_exchange", "B_direct", "B_exchange")
    }
    ph_responses = []
    hp_responses = []
    for column, pair in enumerate(pairs):
        ph = apply_rlg_hbn_hf_quotient_response(
            run,
            _q0_tangent(run, orbitals, pair, "ph"),
            context=context,
            require_converged=False,
            require_provenance=False,
        )
        hp = apply_rlg_hbn_hf_quotient_response(
            run,
            _q0_tangent(run, orbitals, pair, "hp"),
            context=context,
            require_converged=False,
            require_provenance=False,
        )
        ph_responses.append(ph)
        hp_responses.append(hp)
        for row, row_pair in enumerate(pairs):
            response_terms["A_direct"][row, column] = _project_response(
                orbitals,
                row_pair,
                ph.hartree,
            )
            response_terms["A_exchange"][row, column] = _project_response(
                orbitals,
                row_pair,
                ph.fock,
            )
            response_terms["B_direct"][row, column] = _project_response(
                orbitals,
                row_pair,
                hp.hartree,
            )
            response_terms["B_exchange"][row, column] = _project_response(
                orbitals,
                row_pair,
                hp.fock,
            )

    cycle = build_rlg_hbn_tdhf_c3_quotient_cycle(
        run,
        orbitals,
        (0, 0),
        physical_shifts=PHYSICAL_SHIFTS,
        structure_tolerance=1.0e-10,
        closure_tolerance=1.0e-9,
        require_closure=False,
    )
    direct_terms = cycle.steps[0].terms["q"]
    fixed_indices = {
        int(pair[0]) * MESH + int(pair[1])
        for pair in run.basis_data.c3_fixed_representative_pairs
    }
    x_fixed, y_fixed = fixed_role_masks(
        orbitals,
        pairs,
        (0, 0),
        fixed_indices,
        MESH,
    )
    np.testing.assert_array_equal(x_fixed, y_fixed)
    ordinary = ~x_fixed
    ordinary_entries = ordinary[:, None] & ordinary[None, :]
    ordinary_residuals = {
        name: _max_abs((response_terms[name] - direct_terms[name])[ordinary_entries])
        for name in response_terms
    }

    a0 = np.asarray(direct_terms["A0"], dtype=np.complex128)
    a_response = a0 + response_terms["A_direct"] + response_terms["A_exchange"]
    b_response = response_terms["B_direct"] + response_terms["B_exchange"]
    structure = {
        "A_hermitian": _max_abs(a_response - a_response.conj().T),
        "B_symmetric": _max_abs(b_response - b_response.T),
    }
    prepared = build_rlg_hbn_tdhf_finite_q_quotient_context(
        run,
        periodic_gauge_padding=2,
        require_provenance=False,
    )
    production_q0 = build_rlg_hbn_tdhf_finite_q_quotient_matrices_from_pairs(
        run,
        orbitals,
        pairs,
        (0, 0),
        prepared_context=prepared,
        structure_tolerance=1.0e-10,
        require_provenance=False,
    )
    np.testing.assert_allclose(production_q0.A, a_response, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(production_q0.B, b_response, rtol=0.0, atol=1.0e-10)

    fixed_columns = np.flatnonzero(x_fixed)
    if fixed_columns.size == 0:
        raise AssertionError("q0 Hessian gate did not include a fixed-touched tangent")
    base = build_rlg_hbn_hf_c3_quotient_interaction_components(
        run.state.density,
        context,
        v0=run.state.v0,
    )
    finite_difference_residuals = {}
    epsilon = 0.25
    for role, responses in (("ph", ph_responses), ("hp", hp_responses)):
        column = int(fixed_columns[0])
        tangent = _q0_tangent(run, orbitals, pairs[column], role)
        perturbed = build_rlg_hbn_hf_c3_quotient_interaction_components(
            run.state.density + epsilon * tangent.blocks,
            context,
            v0=run.state.v0,
        )
        analytic = responses[column]
        for name in ("hartree", "fock", "total"):
            finite_difference = (
                np.asarray(getattr(perturbed, name)) - np.asarray(getattr(base, name))
            ) / epsilon
            finite_difference_residuals[f"{role}_{name}"] = _max_abs(
                finite_difference - np.asarray(getattr(analytic, name))
            )

    beta_zero = apply_rlg_hbn_hf_quotient_response(
        run,
        _q0_tangent(run, orbitals, pairs[0], "ph"),
        context=context,
        beta=0.0,
        require_converged=False,
        require_provenance=False,
    )
    beta_zero_residual = max(
        _max_abs(beta_zero.hartree),
        _max_abs(beta_zero.fock),
        _max_abs(beta_zero.total),
    )

    summary = {
        "hf_interaction_convention": RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
        "configuration": {
            "mesh": [MESH, MESH],
            "active_conduction_bands": 2,
            "physical_shifts": [list(shift) for shift in PHYSICAL_SHIFTS],
            "pair_count": pair_count,
            "fixed_pair_count": int(np.count_nonzero(x_fixed)),
        },
        "ordinary_D18_residuals_mev": ordinary_residuals,
        "finite_difference_residuals_mev": finite_difference_residuals,
        "structure_residuals_mev": structure,
        "beta_zero_residual_mev": beta_zero_residual,
        "fixed_D18_comparison_is_oracle": False,
    }
    output = os.environ.get("MEAN_FIELD_RLG_HBN_Q0_HESSIAN_GATE_JSON")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    assert max(ordinary_residuals.values(), default=0.0) <= TOLERANCE_MEV
    assert max(finite_difference_residuals.values(), default=0.0) <= TOLERANCE_MEV
    assert max(structure.values(), default=0.0) <= TOLERANCE_MEV
    assert beta_zero_residual == 0.0


def test_finite_q_quotient_public_api_has_raw_structures_and_a_c3() -> None:
    run = _build_reduced_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    prepared = build_rlg_hbn_tdhf_finite_q_quotient_context(
        run,
        periodic_gauge_padding=2,
        require_provenance=False,
    )
    sectors = []
    for q_shift in ((1, 0), (0, 1)):
        pairs = _intraflavor_pairs(run, orbitals, q_shift)
        signed = build_rlg_hbn_tdhf_finite_q_quotient_matrix_pair_from_pairs(
            run,
            orbitals,
            pairs,
            q_shift,
            prepared_context=prepared,
            structure_tolerance=1.0e-10,
            require_provenance=False,
        )
        matrices = signed.plus
        identity = np.eye(len(pairs), dtype=np.complex128)
        zero = np.zeros_like(identity)
        tau_x = np.block([[zero, identity], [identity, zero]])
        np.testing.assert_allclose(
            tau_x @ signed.plus.L.conj() @ tau_x,
            -signed.minus.L,
            rtol=0.0,
            atol=1.0e-12,
        )
        assert matrices.structure.a_hermitian <= 1.0e-10
        assert matrices.structure.b_symmetric <= 1.0e-10
        sectors.append(matrices)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(sectors[0].A),
        np.linalg.eigvalsh(sectors[1].A),
        rtol=0.0,
        atol=1.0e-9,
    )
    for matrices in sectors:
        assert matrices.L.shape == (2 * len(matrices.pairs), 2 * len(matrices.pairs))
        assert np.all(np.isfinite(matrices.L))


def test_rlg_hbn_q0_response_rejects_nonzero_reciprocal_alias() -> None:
    run = _build_reduced_run()
    orbitals = build_rlg_hbn_tdhf_orbitals(run.state)
    pair = _intraflavor_pairs(run, orbitals)[0]
    tangent = _q0_tangent(run, orbitals, pair, "ph")
    aliased = RLGhBNFiniteQDensityTangent(
        q_shift=(MESH, 0),
        target_k=tangent.target_k,
        source_k=tangent.source_k,
        blocks=tangent.blocks,
        role=tangent.role,
    )
    with pytest.raises(NotImplementedError, match="Only the q=0"):
        apply_rlg_hbn_hf_quotient_response(
            run,
            aliased,
            require_converged=False,
            require_provenance=False,
        )

    fractional = replace(tangent, q_shift=(0.5, 0))
    with pytest.raises(ValueError, match="exact integers"):
        apply_rlg_hbn_hf_quotient_response(
            run,
            fractional,
            require_converged=False,
            require_provenance=False,
        )

    context = build_rlg_hbn_hf_c3_quotient_interaction_context(
        run.basis_data,
        run.overlap_blocks,
    )
    with pytest.raises(ValueError, match="no typed interaction provenance"):
        apply_rlg_hbn_hf_quotient_response(
            run,
            tangent,
            context=context,
            require_converged=False,
        )

    provenance = RLGhBNHFInteractionProvenance(
        convention=RLG_HBN_HF_INTERACTION_CONVENTION_VERSION,
        quotient_enabled=True,
        beta=1.0,
        physical_shifts=tuple(context.physical_shifts),
        zero_literal_q0_fock=False,
        basis_periodic_gauge=RLG_HBN_BASIS_PERIODIC_GAUGE_VERSION,
        basis_periodic_gauge_padding=RLG_HBN_BASIS_PERIODIC_GAUGE_PADDING,
        form_factor_convention=RLG_HBN_FORM_FACTOR_CONVENTION_VERSION,
    )
    validated = apply_rlg_hbn_hf_quotient_response(
        replace(run, interaction_provenance=provenance),
        tangent,
        context=context,
        require_converged=False,
    )
    assert validated.provenance["source_provenance_validated"] is True

    with pytest.raises(ValueError, match="does not match"):
        apply_rlg_hbn_hf_quotient_response(
            replace(
                run,
                interaction_provenance=replace(
                    provenance,
                    convention="actual_node_ws_fixed_source_copy_v1",
                ),
            ),
            tangent,
            context=context,
            require_converged=False,
        )
