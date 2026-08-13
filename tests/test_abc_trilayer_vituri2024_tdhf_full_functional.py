"""Core-oracle tests for the factorized Vituri full projected functional."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from itertools import product
from pathlib import Path
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from mean_field.systems.abc_trilayer.vituri2024 import (
    SM_TEX_SHA256,
    third_lowest_active_band,
)
from mean_field.systems.abc_trilayer.vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from mean_field.systems.abc_trilayer.vituri2024_interaction import (
    Vituri2024InteractionBinding,
    Vituri2024InteractionChoiceReceipt,
)
import mean_field.systems.abc_trilayer as abc
import mean_field.systems.abc_trilayer.vituri2024_tdhf_full_functional as full_functional
import mean_field.systems.abc_trilayer.vituri2024_hf as translational_hf
from mean_field.systems.abc_trilayer.vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    make_vituri2024_finite_domain_mesh_receipt,
    make_vituri2024_translational_q0_reproduction_choice,
    vituri2024_conventional_k_diagonal_to_native_density,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_full_scalar import (
    vituri2024_full_operator_to_payload_k_diagonal,
    vituri2024_payload_density_to_full_projector,
    vituri2024_payload_operator_to_full_dense,
)
from mean_field.systems.abc_trilayer.vituri2024_tdhf_full_functional import (
    VITURI2024_FULL_FUNCTIONAL_AUTHORITY,
    VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY,
    Vituri2024FullProjectedFunctionalKernel,
    validate_vituri2024_full_projected_supplied_arrays,
)
from mean_field.systems.abc_trilayer.vituri2024_vertex import (
    Vituri2024Flavor,
    Vituri2024FourPointKinematicsReceipt,
    Vituri2024Orbital,
    vituri2024_antisymmetrized_projected_vertex,
)


def _digest(label: str) -> str:
    return sha256(label.encode()).hexdigest()


def _interaction(*, q0: str = "analytic_kernel_limit_only"):
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=250.0,
        coulomb_e2_ev_angstrom=14.3996,
        q0_evaluation=q0,
        provider_sha256=_digest("full-functional-interaction-choice"),
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text="Explicit test choice for actual Vituri projected-vertex algebra.",
    )


def _states(mesh: np.ndarray, delta1: float) -> np.ndarray:
    result = np.empty((2, 6, mesh.shape[0]), dtype=np.complex128)
    for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
        for k_index, momentum in enumerate(mesh):
            result[valley_index, :, k_index] = third_lowest_active_band(
                momentum, valley, delta1
            ).eigenvector
    return result


def _hermitian(rng: np.random.Generator, dimension: int) -> np.ndarray:
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    return np.asarray(0.5 * (raw + raw.conj().T), dtype=np.complex128)


def _kernel(*, q0: str = "analytic_kernel_limit_only"):
    mesh = np.asarray([[0.0, 0.0], [0.013, 0.0]], dtype=np.float64)
    delta1 = 0.028
    states = _states(mesh, delta1)
    dimension = len(INTERNAL_FLAVOR_ORDER) * len(mesh)
    rng = np.random.default_rng(1701)
    h0 = 0.01 * _hermitian(rng, dimension)
    reference_columns = rng.normal(size=(dimension, dimension // 2)) + 1j * rng.normal(
        size=(dimension, dimension // 2)
    )
    reference_orbitals, _ = np.linalg.qr(reference_columns)
    reference = np.asarray(
        reference_orbitals @ reference_orbitals.conj().T,
        dtype=np.complex128,
    )
    kernel = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=mesh,
        active_band_states=states,
        h0_full=h0,
        normal_order_reference=reference,
        area_angstrom_squared=7300.0,
        interaction=_interaction(q0=q0),
        normal_order_reference_fingerprint=_digest(
            "explicit-complex-projector-reference"
        ),
        q0_policy_fingerprint=_digest("analytic-q0-value-no-background-authority"),
        source_artifact_sha256=_digest("small-actual-vertex-source-fixture"),
        provenance=(
            "Small actual-code-path projected-Hamiltonian oracle; no immutable "
            "source, q0-background, production, or paper authority."
        ),
    )
    return kernel, delta1


def _orbitals(kernel: Vituri2024FullProjectedFunctionalKernel):
    return tuple(
        Vituri2024Orbital(
            flavor=Vituri2024Flavor(valley=valley, spin=spin),
            momentum_inverse_angstrom=tuple(kernel.ordered_mesh[k_index]),
        )
        for valley, spin in INTERNAL_FLAVOR_ORDER
        for k_index in range(kernel.nk)
    )


def _explicit_wbar(
    kernel: Vituri2024FullProjectedFunctionalKernel,
    delta1: float,
) -> np.ndarray:
    orbitals = _orbitals(kernel)
    number = len(orbitals)
    result = np.zeros((number, number, number, number), dtype=np.complex128)
    for alpha, beta, gamma, delta in product(range(number), repeat=4):
        quartet = tuple(orbitals[index] for index in (alpha, beta, gamma, delta))
        momenta = tuple(item.momentum_inverse_angstrom for item in quartet)
        residual = (
            momenta[0][0] + momenta[1][0] - momenta[2][0] - momenta[3][0],
            momenta[0][1] + momenta[1][1] - momenta[2][1] - momenta[3][1],
        )
        if residual != (0.0, 0.0):
            continue
        kinematics = Vituri2024FourPointKinematicsReceipt(
            alpha=quartet[0],
            beta=quartet[1],
            gamma=quartet[2],
            delta=quartet[3],
            momentum_tolerance_inverse_angstrom=0.0,
            provider_sha256=_digest("literal-quartet-provider"),
            derivation_source_sm_sha256=SM_TEX_SHA256,
            source_text="Independent literal N^4 actual-vertex oracle.",
        )
        result[alpha, beta, gamma, delta] = (
            vituri2024_antisymmetrized_projected_vertex(
                kinematics, delta1, kernel.interaction_receipt
            ).value
            / kernel.area_angstrom_squared
        )
    return result


def test_public_package_exports_full_functional_kernel() -> None:
    assert abc.Vituri2024FullProjectedFunctionalKernel is (
        Vituri2024FullProjectedFunctionalKernel
    )
    assert abc.validate_vituri2024_full_projected_supplied_arrays is (
        validate_vituri2024_full_projected_supplied_arrays
    )


def test_factorized_action_and_scalar_match_actual_vertex_tensor() -> None:
    kernel, delta1 = _kernel()
    assert kernel.authority == VITURI2024_FULL_FUNCTIONAL_AUTHORITY
    assert kernel.source_closure_established is False
    assert kernel.production_ready is False
    wbar = _explicit_wbar(kernel, delta1)
    assert np.max(np.abs(wbar + wbar.swapaxes(0, 1))) < 2.0e-13
    assert np.max(np.abs(wbar + wbar.swapaxes(2, 3))) < 2.0e-13
    assert np.max(np.abs(wbar - wbar.transpose(3, 2, 1, 0).conj())) < 2.0e-13

    rng = np.random.default_rng(811)
    density = _hermitian(rng, kernel.dimension)
    actual = kernel.interaction_action(density)
    expected = np.einsum("ibgj,gb->ij", wbar, density, optimize=False)
    assert np.max(np.abs(actual - expected)) < 2.0e-12

    projector = _hermitian(rng, kernel.dimension)
    q = projector - kernel.normal_order_reference
    explicit_energy = np.einsum(
        "ij,ji->", kernel.h0_full, projector, optimize=False
    ) + 0.5 * np.einsum("abgd,da,gb->", wbar, q, q, optimize=False)
    explicit_fock = kernel.h0_full + np.einsum(
        "ibgj,gb->ij", wbar, q, optimize=False
    )
    assert kernel.energy(projector) == pytest.approx(explicit_energy.real, abs=2.0e-12)
    assert abs(explicit_energy.imag) < 2.0e-12
    assert np.max(np.abs(kernel.fock(projector) - explicit_fock)) < 2.0e-12


def test_normal_reference_rejects_nonrepresentable_eigenvalues() -> None:
    kernel, _ = _kernel()
    negative = kernel.normal_order_reference.copy()
    negative -= 2.0 * np.eye(kernel.dimension, dtype=np.complex128)
    with pytest.raises(ValueError, match="0<=R<=I"):
        replace(
            kernel,
            normal_order_reference=negative,
            normal_order_reference_fingerprint=_digest("negative-reference"),
        )

    above_one = kernel.normal_order_reference.copy()
    above_one += 2.0 * np.eye(kernel.dimension, dtype=np.complex128)
    with pytest.raises(ValueError, match="0<=R<=I"):
        replace(
            kernel,
            normal_order_reference=above_one,
            normal_order_reference_fingerprint=_digest("above-one-reference"),
        )


def test_energy_fock_df_and_self_adjoint_pairing() -> None:
    kernel, _ = _kernel()
    rng = np.random.default_rng(908)
    p = 0.1 * _hermitian(rng, kernel.dimension)
    d = _hermitian(rng, kernel.dimension)
    d /= np.linalg.norm(d)
    e = _hermitian(rng, kernel.dimension)
    e /= np.linalg.norm(e)
    step = 2.0e-5

    reference_energy = np.einsum(
        "ij,ji->",
        kernel.h0_full,
        kernel.normal_order_reference,
        optimize=False,
    )
    assert kernel.energy(kernel.normal_order_reference) == pytest.approx(
        reference_energy.real, abs=2.0e-12
    )
    assert np.max(
        np.abs(kernel.fock(kernel.normal_order_reference) - kernel.h0_full)
    ) < 2.0e-12

    energy_derivative = (kernel.energy(p + step * d) - kernel.energy(p - step * d)) / (
        2.0 * step
    )
    fock_pairing = np.einsum("ij,ji->", kernel.fock(p), d, optimize=False)
    assert energy_derivative == pytest.approx(fock_pairing.real, abs=2.0e-9)
    assert abs(fock_pairing.imag) < 2.0e-12

    finite_df = (kernel.fock(p + step * d) - kernel.fock(p - step * d)) / (
        2.0 * step
    )
    exact_df = kernel.fock_derivative(p, d)
    assert np.max(np.abs(finite_df - exact_df)) < 2.0e-10
    assert np.max(np.abs(kernel.fock_derivative(p + e, d) - exact_df)) == 0.0

    left = np.einsum(
        "ij,ji->", d, kernel.fock_derivative(p, e), optimize=False
    )
    right = np.einsum(
        "ij,ji->", e, kernel.fock_derivative(p, d), optimize=False
    )
    assert abs(left.imag) < 2.0e-12
    assert abs(right.imag) < 2.0e-12
    assert left.real == pytest.approx(right.real, abs=2.0e-12)


def test_source_gauge_covariance_uses_explicit_active_states() -> None:
    kernel, _ = _kernel()
    rng = np.random.default_rng(2718)
    p = _hermitian(rng, kernel.dimension)
    phases_by_valley = rng.uniform(-np.pi, np.pi, size=(2, kernel.nk))
    gauged_states = kernel.active_band_states * np.exp(
        1j * phases_by_valley[:, None, :]
    )
    valley_index = {
        valley: index
        for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
    }
    orbital_phases = np.asarray(
        [
            phases_by_valley[valley_index[valley], k_index]
            for valley, _spin in INTERNAL_FLAVOR_ORDER
            for k_index in range(kernel.nk)
        ]
    )
    gauge = np.diag(np.exp(1j * orbital_phases))

    def transform(value: np.ndarray) -> np.ndarray:
        return np.asarray(gauge.conj().T @ value @ gauge, dtype=np.complex128)

    gauged = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=kernel.ordered_mesh.copy(),
        active_band_states=np.asarray(gauged_states, dtype=np.complex128),
        h0_full=transform(kernel.h0_full),
        normal_order_reference=transform(kernel.normal_order_reference),
        area_angstrom_squared=kernel.area_angstrom_squared,
        interaction=kernel.interaction_receipt,
        normal_order_reference_fingerprint=_digest("gauge-transformed-reference"),
        q0_policy_fingerprint=kernel.q0_policy_fingerprint,
        source_artifact_sha256=_digest("gauge-transformed-source"),
        provenance="Gauge-covariance oracle in the same explicit source-state basis.",
    )
    transformed_p = transform(p)
    assert gauged.energy(transformed_p) == pytest.approx(kernel.energy(p), abs=2.0e-11)
    assert np.max(np.abs(gauged.fock(transformed_p) - transform(kernel.fock(p)))) < 2.0e-11


def test_supplied_array_consistency_is_fail_closed_and_authority_limited() -> None:
    kernel, _ = _kernel()
    rng = np.random.default_rng(1307)
    orbitals = rng.normal(size=(kernel.dimension, kernel.dimension // 2)) + 1j * rng.normal(
        size=(kernel.dimension, kernel.dimension // 2)
    )
    occupied, _ = np.linalg.qr(orbitals)
    p0 = np.asarray(occupied @ occupied.conj().T, dtype=np.complex128)
    interaction_h = kernel.interaction_action(p0 - kernel.normal_order_reference)
    fock = kernel.fock(p0)
    receipt = validate_vituri2024_full_projected_supplied_arrays(
        kernel=kernel,
        source_projector=p0,
        supplied_interaction_h=interaction_h,
        supplied_fock=fock,
    )
    assert receipt.passed is True
    assert receipt.authority == VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_AUTHORITY
    assert receipt.production_ready is False

    wrong = fock.copy()
    wrong[0, 0] += 2.0e-6
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        validate_vituri2024_full_projected_supplied_arrays(
            kernel=kernel,
            source_projector=p0,
            supplied_interaction_h=interaction_h,
            supplied_fock=wrong,
        )

    shared_shift = np.zeros_like(fock)
    shared_shift[0, 0] = 2.0e-6
    with pytest.raises(ValueError, match="supplied-array consistency failed"):
        validate_vituri2024_full_projected_supplied_arrays(
            kernel=kernel,
            source_projector=p0,
            supplied_interaction_h=interaction_h + shared_shift,
            supplied_fock=fock + shared_shift,
        )


def test_exact_local_mask_rejects_roundoff_asymmetry_and_torus_carry() -> None:
    q = 0.013
    shifted = np.nextafter(q, np.inf)
    pathological = np.asarray(
        [[0.0, 0.0], [q, 0.0], [shifted, 0.0]], dtype=np.float64
    )
    with pytest.raises(ValueError, match="pair-Hermitian closed"):
        full_functional._exact_local_mask(pathological)

    kernel, _ = _kernel()
    assert kernel.exact_local_mask[1, 0, 0, 0] == np.bool_(False)
    assert kernel.exact_local_mask[1, 0, 1, 0] == np.bool_(True)
    # If one artificially declared a reciprocal period G=2q, this quartet
    # would conserve only modulo G.  The local continuum kernel must reject it.
    reciprocal_period = 2.0 * q
    wrap_only_residual = 2.0 * q
    assert wrap_only_residual / reciprocal_period == 1.0
    assert kernel.exact_local_mask[1, 1, 0, 0] == np.bool_(False)


def test_live_state_is_bytes_backed_and_implementation_bound(monkeypatch) -> None:
    kernel, _ = _kernel()
    for value in (
        kernel.ordered_mesh,
        kernel.active_band_states,
        kernel.h0_full,
        kernel.normal_order_reference,
        kernel.form_factors_by_flavor,
        kernel.kernel_by_mesh_pair,
        kernel.exact_local_mask,
    ):
        assert value.flags.writeable is False
        assert value.flags.owndata is False
        with pytest.raises(ValueError):
            value.setflags(write=True)
    kernel.validate_live_state()
    object.__setattr__(
        kernel, "area_angstrom_squared", 2.0 * kernel.area_angstrom_squared
    )
    with pytest.raises(ValueError, match="kernel fingerprint drifted"):
        kernel.validate_live_state()

    kernel, _ = _kernel()
    object.__setattr__(kernel, "production_ready", True)
    with pytest.raises(ValueError, match="authority fields drifted"):
        kernel.validate_live_state()

    kernel, _ = _kernel()

    def altered_energy(self, projector):
        return 0.0

    with monkeypatch.context() as patch:
        patch.setattr(
            Vituri2024FullProjectedFunctionalKernel, "energy", altered_energy
        )
        with pytest.raises(ValueError, match="implementation code drifted"):
            kernel.validate_live_state()

    kernel, _ = _kernel()

    def altered_max_abs(value):
        return 0.0

    with monkeypatch.context() as patch:
        patch.setattr(full_functional, "_max_abs", altered_max_abs)
        with pytest.raises(ValueError, match="implementation code drifted"):
            kernel.validate_live_state()

    kernel, _ = _kernel()
    with monkeypatch.context() as patch:
        patch.setattr(
            full_functional,
            "VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_TOLERANCE",
            1.0,
        )
        with pytest.raises(RuntimeError, match="tolerance drifted from locked v1"):
            kernel.validate_live_state()

    with monkeypatch.context() as patch:
        patch.setattr(
            full_functional,
            "VITURI2024_FULL_FUNCTIONAL_SUPPLIED_ARRAY_TOLERANCE",
            1.0,
        )
        with pytest.raises(RuntimeError, match="tolerance drifted from locked v1"):
            _kernel()


def test_implementation_fingerprint_is_reproducible_across_fresh_processes() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import mean_field.systems.abc_trilayer.vituri2024_tdhf_full_functional as m;"
        "print(m._kernel_implementation_fingerprint("
        "m.Vituri2024FullProjectedFunctionalKernel))"
    )
    command = [sys.executable, "-I", "-B", "-c", script]
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    first = subprocess.check_output(command, env=environment, text=True).strip()
    second = subprocess.check_output(command, env=environment, text=True).strip()
    assert len(first) == 64
    assert first == second


def test_translational_implementation_fingerprint_is_reproducible_across_fresh_processes() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "import mean_field.systems.abc_trilayer.vituri2024_hf as m;"
        "print(m._implementation_fingerprint())"
    )
    command = [sys.executable, "-I", "-B", "-c", script]
    environment = {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    first = subprocess.check_output(command, env=environment, text=True).strip()
    second = subprocess.check_output(command, env=environment, text=True).strip()
    assert len(first) == 64
    assert first == second


def test_dimension80_factorized_path_avoids_dense_rank4_orbital_tensor() -> None:
    nk = 20
    mesh = np.asarray(
        [(first / 1024.0, second / 1024.0) for first in range(4) for second in range(5)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(80)
    states = rng.normal(size=(2, 6, nk)) + 1j * rng.normal(size=(2, 6, nk))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    dimension = 4 * nk
    zeros = np.zeros((dimension, dimension), dtype=np.complex128)
    tracemalloc.start()
    tracemalloc.clear_traces()
    kernel = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=mesh,
        active_band_states=np.asarray(states, dtype=np.complex128),
        h0_full=zeros,
        normal_order_reference=zeros,
        area_angstrom_squared=7300.0,
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("n80-zero-reference"),
        q0_policy_fingerprint=_digest("n80-analytic-q0"),
        source_artifact_sha256=_digest("n80-runtime-fixture"),
        provenance="Dimension-80 scaling fixture only; no scientific authority.",
    )
    assert kernel.dimension == 80
    assert kernel.exact_local_mask.shape == (20, 20, 20, 20)
    assert kernel.exact_local_mask.nbytes == 20**4
    raw = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    density = np.asarray(0.5 * (raw + raw.conj().T), dtype=np.complex128)
    action = kernel.interaction_action(density)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 64 * 1024**2
    assert action.shape == (80, 80)
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action - action.conj().T)) < 1.0e-11


def test_translational_specialization_matches_full_oracle_for_k_diagonal_E_F_dF() -> None:
    mesh = np.asarray([[-0.013, 0.0], [0.0, 0.0], [0.013, 0.0]], dtype=np.float64)
    states = _states(mesh, 0.028)
    nk = mesh.shape[0]
    rng = np.random.default_rng(731)
    h0_raw = rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))
    h0_native = np.asarray(
        0.01 * (h0_raw + h0_raw.swapaxes(0, 1).conj()) / 2.0,
        dtype=np.complex128,
    )
    reference_native = np.zeros_like(h0_native)
    reference_native[0, 0, :] = 0.20
    reference_native[1, 1, :] = 0.35
    reference_native[0, 1, :] = 0.04 + 0.03j
    reference_native[1, 0, :] = 0.04 - 0.03j
    full_h0 = vituri2024_payload_operator_to_full_dense(h0_native)
    full_reference = vituri2024_payload_density_to_full_projector(reference_native)
    full = Vituri2024FullProjectedFunctionalKernel(
        ordered_mesh=mesh,
        active_band_states=states,
        h0_full=full_h0,
        normal_order_reference=full_reference,
        area_angstrom_squared=7300.0,
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("translation-R0"),
        q0_policy_fingerprint=_digest("translation-q0"),
        source_artifact_sha256=_digest("translation-source"),
        provenance="Reduced translational/full oracle comparison only.",
    )
    translational = Vituri2024TranslationalHFFunctional(
        ordered_mesh=mesh,
        active_band_states=states,
        h0_native=h0_native,
        normal_order_reference_native=reference_native,
        mesh_receipt=make_vituri2024_finite_domain_mesh_receipt(
            ordered_mesh=mesh,
            area_angstrom_squared=7300.0,
            provenance="Reduced explicit finite-domain mesh choice.",
        ),
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("translation-R0"),
        q0_choice=make_vituri2024_translational_q0_reproduction_choice(
            evidence=(
                "Retain finite dual-gate q0 direct/exchange at fixed rank; "
                "independent reproduction choice, not paper authority."
            )
        ),
        provenance="Reduced translational/full oracle comparison only.",
    )
    raw = rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))
    conventional = np.asarray(0.5 * (raw + raw.swapaxes(0, 1).conj()), dtype=np.complex128)
    native = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    raw_d = rng.normal(size=(4, 4, nk)) + 1j * rng.normal(size=(4, 4, nk))
    conventional_d = np.asarray(
        0.5 * (raw_d + raw_d.swapaxes(0, 1).conj()), dtype=np.complex128
    )
    native_d = vituri2024_conventional_k_diagonal_to_native_density(conventional_d)
    full_p = vituri2024_payload_density_to_full_projector(native)
    full_d = vituri2024_payload_density_to_full_projector(native_d)
    full_action = vituri2024_full_operator_to_payload_k_diagonal(
        full.interaction_action(full_p)
    )
    translated_action = translational.interaction_action_conventional(conventional)
    assert np.max(np.abs(full_action - translated_action)) < 2.0e-12
    assert translational.energy(native) == pytest.approx(full.energy(full_p), abs=2.0e-12)
    native_pairing = np.einsum("abk,abk->", h0_native, native, optimize=False)
    full_pairing = np.einsum("ij,ji->", full_h0, full_p, optimize=False)
    assert native_pairing == pytest.approx(full_pairing, abs=2.0e-12)
    assert native[0, 1, 0] == conventional[1, 0, 0]
    assert full_p[0, nk,] == native[1, 0, 0]
    translated_fock = translational.fock(native)
    full_fock = vituri2024_full_operator_to_payload_k_diagonal(full.fock(full_p))
    assert np.max(np.abs(translated_fock - full_fock)) < 2.0e-12
    translated_df = translational.fock_derivative(native, native_d)
    full_df = vituri2024_full_operator_to_payload_k_diagonal(
        full.fock_derivative(full_p, full_d)
    )
    assert np.max(np.abs(translated_df - full_df)) < 2.0e-12
    step = 2.0e-5
    finite_energy = (
        translational.energy(native + step * native_d)
        - translational.energy(native - step * native_d)
    ) / (2.0 * step)
    fock_pairing = np.einsum(
        "abk,abk->", translated_fock, native_d, optimize=False
    )
    assert finite_energy == pytest.approx(float(np.real(fock_pairing)), abs=2.0e-8)
    finite_df = (
        translational.fock(native + step * native_d)
        - translational.fock(native - step * native_d)
    ) / (2.0 * step)
    assert np.max(np.abs(finite_df - translated_df)) < 2.0e-9
    anchor_df = translational.fock_derivative(native + 0.2 * native_d, native_d)
    assert np.array_equal(anchor_df, translated_df)
    reverse_pairing = np.einsum(
        "abk,abk->",
        translational.fock_derivative(native, native),
        native_d,
        optimize=False,
    )
    forward_pairing = np.einsum(
        "abk,abk->", translated_df, native, optimize=False
    )
    assert reverse_pairing == pytest.approx(forward_pairing, abs=2.0e-12)
    assert translational.source_stationarity_established is False
    assert translational.q0_background_authority_established is False
    assert translational.production_ready is False
    wrong_nk = np.zeros((4, 4, 1), dtype=np.complex128)
    for method in (
        translational.interaction_action_conventional,
        translational.energy,
        translational.fock,
    ):
        with pytest.raises(ValueError, match="Nk mismatch"):
            method(wrong_nk)
    with pytest.raises(ValueError, match="Nk mismatch"):
        translational.fock_derivative(native, wrong_nk)
    object.__setattr__(
        translational.mesh_receipt,
        "uniform_weight_inverse_angstrom_squared",
        2.0 / translational.mesh_receipt.area_angstrom_squared,
    )
    with pytest.raises(ValueError, match="mesh receipt live state drifted"):
        translational.validate_live_state()
    object.__setattr__(
        translational.mesh_receipt,
        "uniform_weight_inverse_angstrom_squared",
        1.0 / translational.mesh_receipt.area_angstrom_squared,
    )
    object.__setattr__(
        translational.q0_choice,
        "establishes_paper_or_source_q0_background",
        True,
    )
    with pytest.raises(ValueError, match="q0 authority was inflated"):
        translational.validate_live_state()
    object.__setattr__(
        translational.q0_choice,
        "establishes_paper_or_source_q0_background",
        False,
    )
    inflated_binding = Vituri2024InteractionBinding(receipt=_interaction())
    object.__setattr__(inflated_binding, "establishes_hf_q0_background", True)
    object.__setattr__(translational, "interaction", inflated_binding)
    with pytest.raises(ValueError, match="stale or inflated"):
        translational.validate_live_state()
    original_interaction = _interaction()
    object.__setattr__(translational, "interaction", original_interaction)
    original_vtf = translational_hf.vituri2024_vtf
    try:
        translational_hf.vituri2024_vtf = lambda q, interaction: original_vtf(
            q, interaction
        )
        with pytest.raises(ValueError, match="implementation drifted"):
            translational.validate_live_state()
    finally:
        translational_hf.vituri2024_vtf = original_vtf
    object.__setattr__(translational, "interaction", _interaction(q0="reject"))
    with pytest.raises(ValueError, match="finite analytic q=0 kernel"):
        translational.validate_live_state()


def test_translational_two_dimensional_mesh_permutation_covariance() -> None:
    mesh = np.asarray(
        [[-0.015625, -0.03125], [0.0, 0.0], [0.03125, -0.015625], [0.015625, 0.03125]],
        dtype=np.float64,
    )
    states = _states(mesh, 0.028)
    rng = np.random.default_rng(904)
    raw = rng.normal(size=(4, 4, 4)) + 1j * rng.normal(size=(4, 4, 4))
    conventional = np.asarray(
        0.5 * (raw + raw.swapaxes(0, 1).conj()), dtype=np.complex128
    )
    native = vituri2024_conventional_k_diagonal_to_native_density(conventional)
    zeros = np.zeros((4, 4, 4), dtype=np.complex128)
    choice = make_vituri2024_translational_q0_reproduction_choice(
        evidence="2D permutation covariance q0 reproduction choice."
    )
    original = Vituri2024TranslationalHFFunctional(
        ordered_mesh=mesh,
        active_band_states=states,
        h0_native=zeros,
        normal_order_reference_native=zeros,
        mesh_receipt=make_vituri2024_finite_domain_mesh_receipt(
            ordered_mesh=mesh, area_angstrom_squared=7300.0,
            provenance="2D irregular finite-domain algebra mesh.",
        ),
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("2d-R0"),
        q0_choice=choice,
        provenance="2D translational permutation oracle.",
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted_mesh = mesh[permutation]
    permuted = Vituri2024TranslationalHFFunctional(
        ordered_mesh=permuted_mesh,
        active_band_states=states[:, :, permutation],
        h0_native=zeros[:, :, permutation],
        normal_order_reference_native=zeros[:, :, permutation],
        mesh_receipt=make_vituri2024_finite_domain_mesh_receipt(
            ordered_mesh=permuted_mesh, area_angstrom_squared=7300.0,
            provenance="Permuted 2D irregular finite-domain algebra mesh.",
        ),
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("2d-R0"),
        q0_choice=choice,
        provenance="2D translational permutation oracle.",
    )
    action = original.interaction_action(native)
    permuted_action = permuted.interaction_action(native[:, :, permutation])
    inverse = np.argsort(permutation)
    assert np.max(np.abs(action - permuted_action[:, :, inverse])) < 2.0e-12


def test_translational_constructor_rejects_duplicate_but_accepts_nondyadic_uniform_mesh() -> None:
    duplicate = np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    states = _states(duplicate, 0.028)
    zeros = np.zeros((4, 4, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="duplicate exact coordinates"):
        Vituri2024TranslationalHFFunctional(
            ordered_mesh=duplicate,
            active_band_states=states,
            h0_native=zeros,
            normal_order_reference_native=zeros,
            mesh_receipt=make_vituri2024_finite_domain_mesh_receipt(
                ordered_mesh=duplicate,
                area_angstrom_squared=7300.0,
                provenance="Duplicate-mesh rejection choice.",
            ),
            interaction=_interaction(),
            normal_order_reference_fingerprint=_digest("duplicate-R0"),
            q0_choice=make_vituri2024_translational_q0_reproduction_choice(
                evidence="Duplicate-mesh q0 choice."
            ),
            provenance="Duplicate-mesh rejection canary.",
        )
    spacing = np.float64(2.0 * np.pi / np.sqrt(100000.0))
    nondyadic = np.asarray(
        [[-spacing, 0.0], [0.0, 0.0], [spacing, 0.0]], dtype=np.float64
    )
    states = _states(nondyadic, 0.028)
    three_zeros = np.zeros((4, 4, 3), dtype=np.complex128)
    candidate = Vituri2024TranslationalHFFunctional(
        ordered_mesh=nondyadic,
        active_band_states=states,
        h0_native=three_zeros,
        normal_order_reference_native=three_zeros,
        mesh_receipt=make_vituri2024_finite_domain_mesh_receipt(
            ordered_mesh=nondyadic,
            area_angstrom_squared=100000.0,
            provenance="Physical non-dyadic finite-volume mesh canary.",
        ),
        interaction=_interaction(),
        normal_order_reference_fingerprint=_digest("nondyadic-R0"),
        q0_choice=make_vituri2024_translational_q0_reproduction_choice(
            evidence="Nondyadic-grid q0 choice."
        ),
        provenance="Index-conserving translational nondyadic-grid canary.",
    )
    assert candidate.nk == 3


def test_full_functional_rejects_missing_q0_kernel_policy() -> None:
    kernel, _ = _kernel()
    assert kernel.kernel_by_mesh_pair[0, 0] > 0.0
    assert kernel.q0_background_authority_established is False
    assert kernel.production_ready is False
    with pytest.raises(ValueError, match="requires an explicit finite q=0"):
        _kernel(q0="reject")
