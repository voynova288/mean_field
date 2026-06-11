from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import (
    apply_band_gauge_to_axis_matrix,
    apply_band_gauge_to_matrix,
    berry_connection_generalized_derivative,
    berry_connection_generalized_derivative_pair,
    covariant_derivative_matrix,
    degenerate_band_groups,
    hamiltonian_gauge_data,
    link_shift_vector,
    random_block_unitary,
    shift_integrand_from_generalized_derivative,
    shift_integrand_from_pair_generalized_derivative,
    shift_vector_from_generalized_derivative,
    shift_vector_from_pair_generalized_derivative,
    trace_subspace,
    wannierberri_matrix_gen_derivative_ln,
    wannierberri_matrix_gen_derivative_nn,
    wannierberri_shift_current_group_trace,
    wannierberri_shift_current_internal_imn,
)
from mean_field.systems.htg.shift_current import generalized_derivative_from_D, second_derivative_matrices, velocity_matrices
from analysis.shift_current.toy_models.slg_toy import GappedSLGParams, d2hdk, dhdk, diagonalize


def _load_wannierberri_formula_module():
    """Load the actual upstream formula.py with minimal dependency stubs.

    This avoids importing the full WannierBerri package, whose optional runtime
    dependencies are not installed in this environment, while still executing
    the reference ``Matrix_GenDer_ln`` source file.
    """

    root = Path(__file__).resolve().parents[1]
    formula_path = root / "reference/upstream/wannier-berri/wannierberri/formula/formula.py"
    package = "wannierberri_ref_test"
    for name in [
        package,
        f"{package}.formula",
        f"{package}.utility",
        f"{package}.symmetry",
        f"{package}.symmetry.point_symmetry",
    ]:
        sys.modules.pop(name, None)

    pkg = types.ModuleType(package)
    pkg.__path__ = []
    sys.modules[package] = pkg
    formula_pkg = types.ModuleType(f"{package}.formula")
    formula_pkg.__path__ = []
    sys.modules[f"{package}.formula"] = formula_pkg

    utility = types.ModuleType(f"{package}.utility")
    utility.cached_einsum = lambda subscripts, *operands: np.einsum(subscripts, *operands, optimize=True)
    sys.modules[f"{package}.utility"] = utility

    symmetry_pkg = types.ModuleType(f"{package}.symmetry")
    symmetry_pkg.__path__ = []
    sys.modules[f"{package}.symmetry"] = symmetry_pkg
    point_symmetry = types.ModuleType(f"{package}.symmetry.point_symmetry")

    class TransformProduct:  # pragma: no cover - only needed to satisfy import-time reference
        def __init__(self, transforms):
            self.transforms = tuple(transforms)

    point_symmetry.TransformProduct = TransformProduct
    sys.modules[f"{package}.symmetry.point_symmetry"] = point_symmetry

    spec = importlib.util.spec_from_file_location(f"{package}.formula.formula", formula_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_wannierberri_dynamic_module():
    """Load upstream dynamic.py with minimal stubs for ShiftCurrentFormula."""

    root = Path(__file__).resolve().parents[1]
    dynamic_path = root / "reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py"
    package = "wannierberri_dynamic_ref_test"
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            sys.modules.pop(name, None)

    pkg = types.ModuleType(package)
    pkg.__path__ = []
    sys.modules[package] = pkg
    calculators_pkg = types.ModuleType(f"{package}.calculators")
    calculators_pkg.__path__ = []
    sys.modules[f"{package}.calculators"] = calculators_pkg

    utility = types.ModuleType(f"{package}.utility")
    utility.cached_einsum = lambda subscripts, *operands: np.einsum(subscripts, *operands, optimize=True)
    utility.Gaussian = lambda *args, **kwargs: None
    utility.Lorentzian = lambda *args, **kwargs: None
    utility.FermiDirac = lambda *args, **kwargs: None
    sys.modules[f"{package}.utility"] = utility

    result = types.ModuleType(f"{package}.result")
    result.EnergyResult = object
    sys.modules[f"{package}.result"] = result

    calculator_module = types.ModuleType(f"{package}.calculators.calculator")

    class Calculator:  # pragma: no cover - only needed for import-time base class
        def __init__(self, **kwargs):
            self.degen_thresh = kwargs.get("degen_thresh", 1.0e-4)
            self.degen_Kramers = kwargs.get("degen_Kramers", False)
            self.save_mode = kwargs.get("save_mode", "bin")

    calculator_module.Calculator = Calculator
    sys.modules[f"{package}.calculators.calculator"] = calculator_module

    formula_pkg = types.ModuleType(f"{package}.formula")
    formula_pkg.__path__ = []

    class Formula:  # pragma: no cover - only needed as upstream base class
        def __init__(self, data_K=None, **parameters):
            self.data_K = data_K
            self.external_terms = parameters.get("external_terms", False)

    formula_pkg.Formula = Formula
    sys.modules[f"{package}.formula"] = formula_pkg

    covariant = types.ModuleType(f"{package}.formula.covariant")

    class SpinVelocity:  # pragma: no cover - unused by ShiftCurrentFormula test
        def __init__(self, *args, **kwargs):
            self.matrix = None

    covariant.SpinVelocity = SpinVelocity
    sys.modules[f"{package}.formula.covariant"] = covariant
    formula_pkg.covariant = covariant

    symmetry_pkg = types.ModuleType(f"{package}.symmetry")
    symmetry_pkg.__path__ = []
    sys.modules[f"{package}.symmetry"] = symmetry_pkg
    point_symmetry = types.ModuleType(f"{package}.symmetry.point_symmetry")
    for transform_name in ["transform_ident", "transform_trans", "transform_odd", "transform_odd_trans_021"]:
        setattr(point_symmetry, transform_name, object())
    sys.modules[f"{package}.symmetry.point_symmetry"] = point_symmetry

    factors = types.ModuleType(f"{package}.factors")
    factors.factor_opt = 1.0
    factors.factor_shc = 1.0
    factors.factor_shift_current = 1.0
    sys.modules[f"{package}.factors"] = factors

    spec = importlib.util.spec_from_file_location(f"{package}.calculators.dynamic", dynamic_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeWannierBerriDataK:
    """One-k-point data_K subset consumed by upstream ShiftCurrentFormula."""

    def __init__(self, energies: np.ndarray, velocity_h: np.ndarray, second_velocity_h: np.ndarray | None = None):
        e = np.asarray(energies, dtype=float)
        v_axis = np.asarray(velocity_h, dtype=np.complex128)
        if v_axis.ndim != 3 or v_axis.shape[1:] != (e.size, e.size):
            raise ValueError(f"velocity_h has shape {v_axis.shape}, incompatible with {e.shape}")
        ndim = v_axis.shape[0]
        self.E_K = e[None, :]
        self.nk = 1
        self.cell_volume = 1.0
        self._velocity = np.moveaxis(v_axis, 0, -1)[None, :, :, :]
        self._second_velocity = np.zeros((1, e.size, e.size, ndim, ndim), dtype=np.complex128)
        if second_velocity_h is not None:
            w_axis = np.asarray(second_velocity_h, dtype=np.complex128)
            if w_axis.shape != (ndim, ndim, e.size, e.size):
                raise ValueError(f"second_velocity_h has shape {w_axis.shape}")
            self._second_velocity[0] = np.transpose(w_axis, (2, 3, 0, 1))
        d_eig = self.E_K[:, :, None] - self.E_K[:, None, :]
        self.dEig_inv = np.zeros_like(d_eig, dtype=float)
        mask = np.abs(d_eig) > 1.0e-10
        self.dEig_inv[mask] = 1.0 / d_eig[mask]
        self.D_H = -self._velocity * self.dEig_inv[:, :, :, None]

    def Xbar(self, name: str, der: int = 0):
        if name == "Ham" and der == 1:
            return self._velocity
        if name == "Ham" and der == 2:
            return self._second_velocity
        if name == "AA" and der == 0:
            return np.zeros_like(self._velocity)
        if name == "AA" and der == 1:
            ndim = self._velocity.shape[-1]
            nb = self._velocity.shape[1]
            return np.zeros((1, nb, nb, ndim, ndim), dtype=np.complex128)
        raise KeyError((name, der))

    def get_A_H(self, external_terms: bool = True):
        return 1.0j * self.D_H


def test_wannierberri_matrix_gen_derivative_block_matches_upstream_code():
    wb_formula = _load_wannierberri_formula_module()
    rng = np.random.default_rng(1234)
    nk = 1
    nb = 5
    ndim = 2
    extra = 3
    matrix = rng.normal(size=(nk, nb, nb, extra)) + 1j * rng.normal(size=(nk, nb, nb, extra))
    comma = rng.normal(size=(nk, nb, nb, extra, ndim)) + 1j * rng.normal(size=(nk, nb, nb, extra, ndim))
    dcov_last = rng.normal(size=(nk, nb, nb, ndim)) + 1j * rng.normal(size=(nk, nb, nb, ndim))
    inn = np.asarray([1, 3], dtype=int)
    out = np.asarray([0, 2, 4], dtype=int)

    ref = wb_formula.Matrix_GenDer_ln(
        wb_formula.Matrix_ln(matrix),
        wb_formula.Matrix_ln(comma),
        wb_formula.Matrix_ln(dcov_last),
    )
    ours_dcov = np.moveaxis(dcov_last[0], -1, 0)

    np.testing.assert_allclose(
        wannierberri_matrix_gen_derivative_ln(matrix[0], comma[0], ours_dcov, inn, out),
        ref.ln(0, inn, out),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        wannierberri_matrix_gen_derivative_nn(matrix[0], comma[0], ours_dcov, inn, out),
        ref.nn(0, inn, out),
        rtol=0.0,
        atol=1.0e-13,
    )


def _toy_hamiltonian(kx: float, ky: float):
    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    mass = 0.7
    beta = 0.18
    velocity = 1.1
    lam = 0.23
    s = float(kx + ky)
    h = (velocity * kx + lam * s * s) * sx + velocity * ky * sy + (mass + beta * kx) * sz
    dhdk = np.stack(
        [
            (velocity + 2.0 * lam * s) * sx + beta * sz,
            (2.0 * lam * s) * sx + velocity * sy,
        ],
        axis=0,
    )
    d2 = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            d2[a, b] = 2.0 * lam * sx
    return h, dhdk, d2


def _toy_gauge_data(kx: float, ky: float):
    h, dhdk, d2 = _toy_hamiltonian(kx, ky)
    evals, evecs = eigh(h)
    return hamiltonian_gauge_data(evals, evecs, dhdk, d2hdk=d2, denominator_cutoff=1.0e-12)


def test_berry_connection_generalized_derivative_matches_existing_sum_rule():
    data = _toy_gauge_data(0.31, -0.17)
    ours = berry_connection_generalized_derivative(
        data.velocity_h,
        data.energies,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    existing = generalized_derivative_from_D(
        data.velocity_h,
        data.energies,
        W=data.second_velocity_h,
        denominator_cutoff_ev=1.0e-12,
    )
    np.testing.assert_allclose(ours.values, existing.values, rtol=0.0, atol=1.0e-14)


def test_selected_pair_generalized_derivative_matches_full_tensor():
    data = _toy_gauge_data(0.22, 0.19)
    full = berry_connection_generalized_derivative(
        data.velocity_h,
        data.energies,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    pair = berry_connection_generalized_derivative_pair(
        data.velocity_h,
        data.energies,
        0,
        1,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    np.testing.assert_allclose(pair.values, full.values[:, :, 0, 1], rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        shift_integrand_from_pair_generalized_derivative(
            data.berry_connection,
            pair.values,
            initial_band=0,
            final_band=1,
            deriv_axis=1,
            optical_axis=0,
        ),
        shift_integrand_from_generalized_derivative(
            data.berry_connection,
            full.values,
            initial_band=0,
            final_band=1,
            deriv_axis=1,
            optical_axis=0,
        ),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        shift_vector_from_pair_generalized_derivative(
            data.berry_connection,
            pair.values,
            initial_band=0,
            final_band=1,
            deriv_axis=1,
            optical_axis=0,
        ),
        shift_vector_from_generalized_derivative(
            data.berry_connection,
            full.values,
            initial_band=0,
            final_band=1,
            deriv_axis=1,
            optical_axis=0,
        ),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert pair.skipped_small_denominators == 0


def test_wannierberri_shift_current_internal_imn_matches_upstream_dynamic_module():
    params = GappedSLGParams(mass_ev=1.5, hopping_ev=2.73)
    k_xy = np.asarray([0.11, -0.07], dtype=float)
    evals, evecs = diagonalize(k_xy, params)
    D = velocity_matrices(evecs, dhdk(k_xy, params))
    W = second_derivative_matrices(evecs, d2hdk(k_xy, params))
    ours = wannierberri_shift_current_internal_imn(D, evals, second_velocity_h=W, sc_eta=0.04, denominator_cutoff=1.0e-10)

    dynamic = _load_wannierberri_dynamic_module()
    upstream = dynamic.ShiftCurrentFormula(
        _FakeWannierBerriDataK(evals, D, W),
        sc_eta=0.04,
        external_terms=False,
    ).Imn[0]
    np.testing.assert_allclose(ours, upstream, rtol=0.0, atol=1.0e-13)

    np.testing.assert_allclose(
        wannierberri_shift_current_group_trace(ours, [0], [1]),
        ours[0, 1],
        rtol=0.0,
        atol=1.0e-15,
    )


def test_principal_value_regularized_pair_matches_full_tensor():
    data = _toy_gauge_data(0.22, 0.19)
    full = berry_connection_generalized_derivative(
        data.velocity_h,
        data.energies,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
        principal_value_eta=0.03,
    )
    pair = berry_connection_generalized_derivative_pair(
        data.velocity_h,
        data.energies,
        0,
        1,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
        principal_value_eta=0.03,
    )
    np.testing.assert_allclose(pair.values, full.values[:, :, 0, 1], rtol=0.0, atol=1.0e-14)


def test_shift_integrand_is_invariant_under_random_u1_gauge():
    data = _toy_gauge_data(0.31, -0.17)
    gd = berry_connection_generalized_derivative(
        data.velocity_h,
        data.energies,
        second_velocity_h=data.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    base = shift_integrand_from_generalized_derivative(
        data.berry_connection,
        gd.values,
        initial_band=0,
        final_band=1,
        deriv_axis=1,
        optical_axis=0,
    )

    phases = np.exp(1j * np.asarray([0.73, -1.21]))
    evecs_g = data.eigenvectors * phases[None, :]
    _h, dhdk, d2 = _toy_hamiltonian(0.31, -0.17)
    gauged = hamiltonian_gauge_data(data.energies, evecs_g, dhdk, d2hdk=d2, denominator_cutoff=1.0e-12)
    gd_g = berry_connection_generalized_derivative(
        gauged.velocity_h,
        gauged.energies,
        second_velocity_h=gauged.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    shifted = shift_integrand_from_generalized_derivative(
        gauged.berry_connection,
        gd_g.values,
        initial_band=0,
        final_band=1,
        deriv_axis=1,
        optical_axis=0,
    )
    assert abs(base - shifted) < 1.0e-13


def test_covariant_derivative_trace_is_invariant_under_degenerate_unitary_gauge():
    rng = np.random.default_rng(5678)
    energies = np.asarray([-1.0, -0.2, -0.2 + 2.0e-6, 0.7])
    groups = degenerate_band_groups(energies, threshold=1.0e-4)
    assert groups == [(0, 1), (1, 3), (3, 4)]
    nb = energies.size
    ndim = 2
    A = rng.normal(size=(nb, nb, 3)) + 1j * rng.normal(size=(nb, nb, 3))
    dA = rng.normal(size=(nb, nb, 3, ndim)) + 1j * rng.normal(size=(nb, nb, 3, ndim))
    D = rng.normal(size=(ndim, nb, nb)) + 1j * rng.normal(size=(ndim, nb, nb))
    gen = covariant_derivative_matrix(A, dA, D)

    gauge = random_block_unitary(groups, nb, rng)
    A_g = apply_band_gauge_to_matrix(A, gauge)
    dA_g = apply_band_gauge_to_matrix(dA, gauge)
    D_g = apply_band_gauge_to_axis_matrix(D, gauge)
    gen_g = covariant_derivative_matrix(A_g, dA_g, D_g)
    gen_expected = apply_band_gauge_to_matrix(gen, gauge)
    np.testing.assert_allclose(gen_g, gen_expected, rtol=0.0, atol=2.0e-13)
    for group in groups:
        np.testing.assert_allclose(trace_subspace(gen_g[..., 0], group), trace_subspace(gen[..., 0], group), atol=2.0e-13)


def test_wilson_link_phase_derivative_is_invariant_under_independent_u1_gauges():
    kx = 0.31
    ky = -0.17
    step = 1.0e-6
    data0 = _toy_gauge_data(kx, ky)
    data1 = _toy_gauge_data(kx, ky + step)
    base = link_shift_vector(
        data0.eigenvectors,
        data1.eigenvectors,
        data0.berry_connection,
        data1.berry_connection,
        initial_band=0,
        final_band=1,
        optical_axis=0,
        step=step,
    )
    phase0 = np.exp(1j * np.asarray([0.37, -1.13]))
    phase1 = np.exp(1j * np.asarray([-0.81, 2.40]))
    g0 = np.diag(phase0)
    g1 = np.diag(phase1)
    shifted = link_shift_vector(
        data0.eigenvectors @ g0,
        data1.eigenvectors @ g1,
        apply_band_gauge_to_axis_matrix(data0.berry_connection, g0),
        apply_band_gauge_to_axis_matrix(data1.berry_connection, g1),
        initial_band=0,
        final_band=1,
        optical_axis=0,
        step=step,
    )
    assert abs(base - shifted) < 1.0e-10


def test_wilson_link_phase_derivative_matches_covariant_shift_vector():
    kx = 0.31
    ky = -0.17
    step = 1.0e-6
    data0 = _toy_gauge_data(kx, ky)
    data1 = _toy_gauge_data(kx, ky + step)
    gd0 = berry_connection_generalized_derivative(
        data0.velocity_h,
        data0.energies,
        second_velocity_h=data0.second_velocity_h,
        denominator_cutoff=1.0e-12,
    )
    covariant_shift = shift_vector_from_generalized_derivative(
        data0.berry_connection,
        gd0.values,
        initial_band=0,
        final_band=1,
        deriv_axis=1,
        optical_axis=0,
    )
    link_shift = link_shift_vector(
        data0.eigenvectors,
        data1.eigenvectors,
        data0.berry_connection,
        data1.berry_connection,
        initial_band=0,
        final_band=1,
        optical_axis=0,
        step=step,
    )
    assert np.isfinite(covariant_shift)
    assert abs(covariant_shift - link_shift) < 2.0e-5
