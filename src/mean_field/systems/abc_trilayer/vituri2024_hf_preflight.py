"""Receipt-only preflight schema for a Vituri-2024 half-metal HF source.

This module records paper facts and validates immutable authority receipts.  It
is not an HF solver: generic SCF/ODA remains in :mod:`mean_field.core.hf`, and
none of the provider methods described below is executed by this preflight.
A complete receipt set establishes only ``receipt_set_complete``.  Scientific
execution, array recomputation, provider-method execution, and paper
reproduction remain explicitly false and execution replay remains unresolved.

Primary paper authority is arXiv:2408.10309v1.  In particular,
``Delta_1=28 meV`` is direct for Fig. 2 and supplementary charge figures, but
the Fig. 3 caption does not state ``Delta_1``.  Its use here is therefore a
reproduction choice, not a Fig. 3 paper-direct fact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .vituri2024 import ARXIV_IDENTIFIER, ARXIV_SOURCE_SHA256, SM_TEX_SHA256

VITURI2024_MAIN_TEX_SHA256 = (
    "eb0a142bda1f594686fab818820a2b0ee700dfeef0fa54055718bfd1fe56ee56"
)
VITURI2024_HALF_METAL_HF_SCOPE = (
    "vituri2024_spin_polarized_half_metal_hf_receipt_preflight_v2"
)

ReceiptAuthority = Literal["reproduction_choice", "independent_provider_explicit"]
FunctionalRole = Literal[
    "scalar_energy",
    "fock_derivative",
    "finite_q_hessian",
    "interaction_form_factor",
]
FiniteDifferenceKind = Literal["fock_first_derivative", "finite_q_hessian"]
FiniteDifferenceComparisonIdentity = Literal[
    "scalar_energy_vs_fock_derivative",
    "fock_derivative_vs_finite_q_hessian",
]
SCFExitReason = Literal["converged", "oda_stall", "max_iter"]
SCFCallbackRole = Literal[
    "initializer",
    "interaction_builder",
    "density_builder",
    "energy_functional",
    "oda_parameterizer",
    "oda_delta_interaction_builder",
    "hamiltonian_postprocessor",
    "density_postprocessor",
    "step_callback",
    "final_state_callback",
]

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntegerArray = NDArray[np.int64]

_GEOMETRY_POLICY = (
    "finite_domain_no_wrap",
    "not_a_reciprocal_torus",
    "no_reciprocal_carry",
)
_SCF_CALLBACK_ROLES: tuple[SCFCallbackRole, ...] = (
    "initializer",
    "interaction_builder",
    "density_builder",
    "energy_functional",
    "oda_parameterizer",
    "oda_delta_interaction_builder",
    "hamiltonian_postprocessor",
    "density_postprocessor",
    "step_callback",
    "final_state_callback",
)
_REQUIRED_CALLABLE_CALLBACKS = frozenset(
    {
        "initializer",
        "interaction_builder",
        "density_builder",
        "energy_functional",
        "oda_delta_interaction_builder",
    }
)


def _strict_int(value: object, label: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{label} must be a strict integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> float:
    result = _finite(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase 40- or 64-character commit")
    return value


def _authority(value: object, label: str) -> ReceiptAuthority:
    if value not in ("reproduction_choice", "independent_provider_explicit"):
        raise ValueError(f"{label} must be provider/reproduction authority")
    return value  # type: ignore[return-value]


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _require_residual(
    residual: object, tolerance: object, label: str
) -> tuple[float, float]:
    clean_residual = _nonnegative(residual, f"{label} residual")
    clean_tolerance = _positive(tolerance, f"{label} tolerance")
    if clean_residual > clean_tolerance:
        raise ValueError(f"{label} residual exceeds its declared tolerance")
    return clean_residual, clean_tolerance


def _require_abs_residual(
    residual: float, left: float, right: float, label: str
) -> None:
    expected = abs(left - right)
    scale = max(abs(left), abs(right), expected, 1.0)
    if not math.isclose(
        residual,
        expected,
        rel_tol=1.0e-12,
        abs_tol=64.0 * math.ulp(scale),
    ):
        raise ValueError(f"{label} residual does not match the bound values")


def _require_context_fingerprint(
    actual: object, payload: object, label: str
) -> str:
    digest = _sha256(actual, label)
    if digest != _fingerprint(payload):
        raise ValueError(f"{label} does not match its bound context")
    return digest


def _validate_core_scf_exit(
    *,
    reason: SCFExitReason,
    iterations: int,
    terminal_norm_raw: float,
    terminal_norm_mixed: float,
    terminal_norm_selected: float,
    terminal_oda_lambda: float,
    convergence_rule: Literal["raw", "mixed"],
    precision: float,
    max_iter: int,
    oda_stall_threshold: float,
    max_oda_lambda: float,
    label: str,
) -> None:
    if reason not in ("converged", "oda_stall", "max_iter"):
        raise ValueError(f"{label} has an unsupported SCF exit reason")
    if iterations < 1 or iterations > max_iter:
        raise ValueError(f"{label} iteration count is outside [1,max_iter]")
    expected_selected = (
        terminal_norm_raw if convergence_rule == "raw" else terminal_norm_mixed
    )
    if terminal_norm_selected != expected_selected:
        raise ValueError(f"{label} terminal selected norm contradicts convergence_rule")
    if terminal_oda_lambda > max_oda_lambda:
        raise ValueError(f"{label} terminal ODA lambda exceeds max_oda_lambda")
    if reason == "converged":
        if terminal_norm_selected > precision:
            raise ValueError(f"{label} converged exit exceeds SCF precision")
        return
    if terminal_norm_selected <= precision:
        raise ValueError(f"{label} non-converged exit contradicts SCF precision")
    if reason == "oda_stall":
        if terminal_oda_lambda >= oda_stall_threshold:
            raise ValueError(f"{label} ODA-stall exit contradicts stall threshold")
        return
    if iterations != max_iter or terminal_oda_lambda < oda_stall_threshold:
        raise ValueError(f"{label} max-iteration exit contradicts core exit semantics")


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalPaperTarget:
    """Exact paper statements with their necessary scope qualifiers."""

    arxiv: str = ARXIV_IDENTIFIER
    source_archive_sha256: str = ARXIV_SOURCE_SHA256
    main_tex_path: str = "main.tex"
    main_tex_sha256: str = VITURI2024_MAIN_TEX_SHA256
    sm_tex_path: str = "SM.tex"
    sm_tex_sha256: str = SM_TEX_SHA256
    half_metal_spin_polarized: bool = True
    one_hole_pocket_per_valley: bool = True
    intervalley_hund_omitted: bool = True
    independent_valley_spin_rotations: bool = True
    fig3a_density_cm2: float = -1.0e12
    fig3bd_density_range_cm2: tuple[float, float] = (-1.1e12, -1.0e12)
    unrestricted_hf: bool = True
    optimal_damping: bool = True
    many_broken_symmetry_initial_conditions: bool = True
    direct_branch_energy_comparison: bool = True
    transfer_learning_near_transitions: bool = True
    exact_numerical_scf_policy_reported: bool = False
    fig2_delta1_mev: float = 28.0
    fig3_delta1_mev: float | None = None
    hole_pocket_qualifier: str = (
        "HF half-metal close to the IVC transition; main.tex line 215 also says "
        "the predicted non-annular FS is at odds with experiment"
    )
    hund_qualifier: str = (
        "model simplification; main.tex lines 101 and 183, not a claim that the "
        "real material has zero intervalley Hund coupling"
    )
    scf_qualifier: str = (
        "SM.tex lines 122-132 state ODA, many broken-symmetry starts, direct "
        "energy comparison, and transfer learning, but no exact tolerances, "
        "iteration limit, seed inventory, or restart policy"
    )
    fig3_delta1_qualifier: str = (
        "unresolved: 28 meV is explicit for Fig. 2 (main.tex lines 124/152) and "
        "supplementary charge figures (SM.tex lines 283-295), not the Fig. 3 caption"
    )

    def __post_init__(self) -> None:
        expected = (
            ARXIV_IDENTIFIER,
            ARXIV_SOURCE_SHA256,
            "main.tex",
            VITURI2024_MAIN_TEX_SHA256,
            "SM.tex",
            SM_TEX_SHA256,
            True,
            True,
            True,
            True,
            -1.0e12,
            (-1.1e12, -1.0e12),
            True,
            True,
            True,
            True,
            True,
            False,
            28.0,
            None,
        )
        actual = (
            self.arxiv,
            self.source_archive_sha256,
            self.main_tex_path,
            self.main_tex_sha256,
            self.sm_tex_path,
            self.sm_tex_sha256,
            self.half_metal_spin_polarized,
            self.one_hole_pocket_per_valley,
            self.intervalley_hund_omitted,
            self.independent_valley_spin_rotations,
            float(self.fig3a_density_cm2),
            tuple(float(value) for value in self.fig3bd_density_range_cm2),
            self.unrestricted_hf,
            self.optimal_damping,
            self.many_broken_symmetry_initial_conditions,
            self.direct_branch_energy_comparison,
            self.transfer_learning_near_transitions,
            self.exact_numerical_scf_policy_reported,
            float(self.fig2_delta1_mev),
            self.fig3_delta1_mev,
        )
        if actual != expected:
            raise ValueError("Vituri half-metal paper-direct facts were changed")
        for value, label in (
            (self.source_archive_sha256, "source archive"),
            (self.main_tex_sha256, "main.tex"),
            (self.sm_tex_sha256, "SM.tex"),
        ):
            _sha256(value, label)
        for value, label in (
            (self.hole_pocket_qualifier, "hole-pocket qualifier"),
            (self.hund_qualifier, "Hund qualifier"),
            (self.scf_qualifier, "SCF qualifier"),
            (self.fig3_delta1_qualifier, "Fig. 3 Delta1 qualifier"),
        ):
            _text(value, label)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024HFGeometryReceipt:
    """Finite-domain geometry and the unit state-sum used by core ODA."""

    area_angstrom_squared: float
    finite_area_receipt_fingerprint: str
    mesh_shape: tuple[int, int]
    mesh_point_count: int
    core_state_nk: int
    per_valley_k_count: int
    valley_representation: Literal["internal_flavor_axis"]
    spin_count: int
    total_active_state_count: int
    selected_spin_state_count: int
    array_layout: Literal["core_state_k_then_internal_valley_then_spin"]
    ordered_momentum_mesh_sha256: str
    mesh_order: Literal["row_major_cartesian_k"]
    momentum_units: Literal["inverse_angstrom"]
    quadrature_rule: Literal["uniform_finite_volume_state_sum"]
    state_sum_weight: float
    state_sum_weight_units: Literal["dimensionless"]
    state_sum_weight_sum: float
    state_sum_weight_sum_residual: float
    state_sum_weight_sum_tolerance: float
    state_sum_weight_sum_evidence_sha256: str
    reciprocal_basis_sha256: str
    reciprocal_basis_convention: Literal["columns_b1_b2_cartesian"]
    axis_origin_convention: Literal["kx_crystal_axis_origin_at_valley_center"]
    boundary_policy: Literal["finite_domain_no_wrap"]
    torus_policy: Literal["not_a_reciprocal_torus"]
    reciprocal_carry_policy: Literal["no_reciprocal_carry"]
    uv_cutoff_inverse_angstrom: float
    delta1_mev: float
    active_band_index: int
    valleys: tuple[int, int]
    domain_minimum_third_band_direct_gap_ev: float
    domain_gap_tolerance_ev: float
    domain_gap_point_count: int
    domain_gap_evidence_sha256: str
    domain_gap_context_fingerprint: str
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    authority_kind: ReceiptAuthority
    provenance: str
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        area = _positive(self.area_angstrom_squared, "finite area")
        object.__setattr__(self, "area_angstrom_squared", area)
        shape = tuple(_strict_int(value, "mesh dimension") for value in self.mesh_shape)
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("mesh_shape must contain two positive dimensions")
        count = _strict_int(self.mesh_point_count, "mesh_point_count")
        if count != shape[0] * shape[1]:
            raise ValueError("mesh_point_count does not match mesh_shape")
        object.__setattr__(self, "mesh_shape", shape)
        object.__setattr__(self, "mesh_point_count", count)
        state_counts = (
            _strict_int(self.core_state_nk, "core_state_nk"),
            _strict_int(self.per_valley_k_count, "per-valley k count"),
            _strict_int(self.spin_count, "spin count"),
            _strict_int(self.total_active_state_count, "total active state count"),
            _strict_int(self.selected_spin_state_count, "selected-spin state count"),
        )
        if state_counts != (count, count, 2, 4 * count, 2 * count):
            raise ValueError(
                "geometry state counts must close as Nk, Nk, 2, 4Nk, and 2Nk"
            )
        for name, value in zip(
            (
                "core_state_nk",
                "per_valley_k_count",
                "spin_count",
                "total_active_state_count",
                "selected_spin_state_count",
            ),
            state_counts,
        ):
            object.__setattr__(self, name, value)

        if (
            self.mesh_order != "row_major_cartesian_k"
            or self.valley_representation != "internal_flavor_axis"
            or self.array_layout != "core_state_k_then_internal_valley_then_spin"
            or self.momentum_units != "inverse_angstrom"
            or self.quadrature_rule != "uniform_finite_volume_state_sum"
            or self.state_sum_weight_units != "dimensionless"
        ):
            raise ValueError(
                "geometry must use the locked k-only mesh and internal-flavor array layout"
            )
        weight = _finite(self.state_sum_weight, "dimensionless state-sum weight")
        if weight != 1.0:
            raise ValueError("current core ODA requires dimensionless state-sum weight exactly 1")
        weight_sum = _finite(self.state_sum_weight_sum, "state-sum weight sum")
        residual, tolerance = _require_residual(
            self.state_sum_weight_sum_residual,
            self.state_sum_weight_sum_tolerance,
            "state-sum weight closure",
        )
        _require_abs_residual(residual, weight_sum, float(count), "state-sum weight sum")
        object.__setattr__(self, "state_sum_weight", weight)
        object.__setattr__(self, "state_sum_weight_sum", weight_sum)
        object.__setattr__(self, "state_sum_weight_sum_residual", residual)
        object.__setattr__(self, "state_sum_weight_sum_tolerance", tolerance)

        if (
            self.boundary_policy,
            self.torus_policy,
            self.reciprocal_carry_policy,
        ) != _GEOMETRY_POLICY:
            raise ValueError(
                "boundary/torus/carry must use the finite-domain no-wrap/no-carry policy"
            )
        if (
            self.reciprocal_basis_convention != "columns_b1_b2_cartesian"
            or self.axis_origin_convention
            != "kx_crystal_axis_origin_at_valley_center"
        ):
            raise ValueError("geometry basis or axis/origin convention was changed")

        object.__setattr__(
            self,
            "uv_cutoff_inverse_angstrom",
            _positive(self.uv_cutoff_inverse_angstrom, "UV cutoff"),
        )
        delta1 = _finite(self.delta1_mev, "Delta1")
        active_band = _strict_int(self.active_band_index, "active_band_index")
        valleys = tuple(_strict_int(value, "valley label") for value in self.valleys)
        if active_band != 2 or valleys != (-1, 1):
            raise ValueError("domain gap must cover active band index 2 and valleys (-1,+1)")
        object.__setattr__(self, "delta1_mev", delta1)
        object.__setattr__(self, "active_band_index", active_band)
        object.__setattr__(self, "valleys", valleys)

        gap_count = _strict_int(self.domain_gap_point_count, "domain_gap_point_count")
        if gap_count != len(valleys) * count:
            raise ValueError(
                "domain-wide gap evidence must cover every valley-k sample (2*Nk)"
            )
        gap = _positive(self.domain_minimum_third_band_direct_gap_ev, "domain gap")
        gap_tolerance = _positive(self.domain_gap_tolerance_ev, "domain gap tolerance")
        if gap <= gap_tolerance:
            raise ValueError("domain-wide third-band gap does not clear its tolerance")
        object.__setattr__(self, "domain_gap_point_count", gap_count)
        object.__setattr__(self, "domain_minimum_third_band_direct_gap_ev", gap)
        object.__setattr__(self, "domain_gap_tolerance_ev", gap_tolerance)

        for value, label in (
            (self.finite_area_receipt_fingerprint, "finite-area receipt"),
            (self.ordered_momentum_mesh_sha256, "ordered momentum mesh"),
            (self.state_sum_weight_sum_evidence_sha256, "state-sum evidence"),
            (self.reciprocal_basis_sha256, "reciprocal basis"),
            (self.domain_gap_evidence_sha256, "domain-gap evidence"),
            (self.provider_fingerprint, "geometry provider"),
            (self.source_artifact_sha256, "geometry source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "geometry source commit")
        _authority(self.authority_kind, "geometry authority")
        _text(self.provenance, "geometry provenance")
        _require_context_fingerprint(
            self.domain_gap_context_fingerprint,
            {
                "ordered_momentum_mesh_sha256": self.ordered_momentum_mesh_sha256,
                "mesh_point_count": count,
                "core_state_nk": self.core_state_nk,
                "per_valley_k_count": self.per_valley_k_count,
                "valley_representation": self.valley_representation,
                "array_layout": self.array_layout,
                "delta1_mev": delta1,
                "active_band_index": active_band,
                "valleys": valleys,
                "domain_gap_point_count": gap_count,
                "domain_minimum_third_band_direct_gap_ev": gap,
                "domain_gap_tolerance_ev": gap_tolerance,
                "domain_gap_evidence_sha256": self.domain_gap_evidence_sha256,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "domain-gap context fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024HFEnsembleReceipt:
    """Locked fixed-density canonical ensemble and reference/q=0 policies."""

    ensemble: Literal["fixed_density"]
    target_density_cm2: float
    density_tolerance_cm2: float
    delta1_mev: float
    electron_hole_counting: Literal["holes_negative_relative_to_neutral_reference"]
    normal_order_reference_kind: Literal["provider_neutral_active_band_reference"]
    normal_order_reference_evidence_sha256: str
    normal_order_reference_fingerprint: str
    q0_neutralizing_background_kind: Literal[
        "remove_uniform_hartree_charge_against_normal_order_reference"
    ]
    q0_background_evidence_sha256: str
    q0_policy_fingerprint: str
    interaction_analytic_kernel_q0_policy: Literal[
        "reject", "analytic_kernel_limit_only"
    ]
    interaction_receipt_fingerprint: str
    chemical_potential_policy: Literal["solve_global_mu_for_exact_fixed_state_count"]
    occupation_policy: Literal["zero_temperature_stable_aufbau_exact_state_count"]
    branch_thermodynamic_functional: Literal["fixed_density_canonical_energy_ev"]
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    authority_kind: ReceiptAuthority
    provenance: str
    delta1_authority: Literal["reproduction_choice"] = field(
        default="reproduction_choice", init=False
    )
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.ensemble != "fixed_density":
            raise ValueError("Vituri Fig. 3 target is fixed-density canonical only")
        density = _finite(self.target_density_cm2, "target density")
        if density >= 0.0:
            raise ValueError("Vituri half-metal target must use a hole density")
        object.__setattr__(self, "target_density_cm2", density)
        object.__setattr__(
            self,
            "density_tolerance_cm2",
            _positive(self.density_tolerance_cm2, "density tolerance"),
        )
        delta1 = _finite(self.delta1_mev, "Delta1")
        object.__setattr__(self, "delta1_mev", delta1)
        locked = (
            self.electron_hole_counting
            == "holes_negative_relative_to_neutral_reference",
            self.normal_order_reference_kind
            == "provider_neutral_active_band_reference",
            self.q0_neutralizing_background_kind
            == "remove_uniform_hartree_charge_against_normal_order_reference",
            self.chemical_potential_policy
            == "solve_global_mu_for_exact_fixed_state_count",
            self.occupation_policy
            == "zero_temperature_stable_aufbau_exact_state_count",
            self.branch_thermodynamic_functional
            == "fixed_density_canonical_energy_ev",
            self.delta1_authority == "reproduction_choice",
        )
        if not all(locked):
            raise ValueError("Vituri Fig. 3 ensemble labels must use the locked typed policies")
        if self.interaction_analytic_kernel_q0_policy not in (
            "reject",
            "analytic_kernel_limit_only",
        ):
            raise ValueError("invalid interaction analytic q=0 policy")
        if (
            self.q0_neutralizing_background_kind
            == self.interaction_analytic_kernel_q0_policy
        ):
            raise ValueError("HF q=0 background must remain distinct from analytic kernel policy")

        for value, label in (
            (self.normal_order_reference_evidence_sha256, "normal-order evidence"),
            (self.q0_background_evidence_sha256, "q=0 background evidence"),
            (self.interaction_receipt_fingerprint, "interaction receipt"),
            (self.provider_fingerprint, "ensemble provider"),
            (self.source_artifact_sha256, "ensemble source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "ensemble source commit")
        _require_context_fingerprint(
            self.normal_order_reference_fingerprint,
            {
                "normal_order_reference_kind": self.normal_order_reference_kind,
                "normal_order_reference_evidence_sha256": (
                    self.normal_order_reference_evidence_sha256
                ),
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "normal-order reference fingerprint",
        )
        _require_context_fingerprint(
            self.q0_policy_fingerprint,
            {
                "q0_neutralizing_background_kind": (
                    self.q0_neutralizing_background_kind
                ),
                "q0_background_evidence_sha256": self.q0_background_evidence_sha256,
                "interaction_analytic_kernel_q0_policy": (
                    self.interaction_analytic_kernel_q0_policy
                ),
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "q0-policy fingerprint",
        )
        _authority(self.authority_kind, "ensemble authority")
        _text(self.provenance, "ensemble provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024SCFSeedReceipt:
    """One exact integer RNG seed paired to its label and core init mode."""

    seed_label: str
    seed_value: int
    init_mode: str

    def __post_init__(self) -> None:
        _text(self.seed_label, "seed label")
        object.__setattr__(self, "seed_value", _strict_int(self.seed_value, "RNG seed"))
        _text(self.init_mode, "seed init mode")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024SCFCallbackReceipt:
    """Exact callable/null implementation identity for one generic-core hook."""

    role: SCFCallbackRole
    implementation_kind: Literal["callable", "none"]
    implementation_fingerprint: str

    def __post_init__(self) -> None:
        if self.role not in _SCF_CALLBACK_ROLES:
            raise ValueError("invalid generic-core callback role")
        if self.implementation_kind not in ("callable", "none"):
            raise ValueError("callback implementation kind must be callable or none")
        _sha256(self.implementation_fingerprint, f"{self.role} implementation")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024HFSCFPolicyReceipt:
    """Exact generic-core SCF/ODA arguments, metrics, callbacks, and seeds."""

    convergence_rule: Literal["raw", "mixed"]
    convergence_metric_identity: Literal[
        "mean_field.core.hf.occupations.calculate_norm_convergence"
    ]
    convergence_metric_normalization: Literal[
        "frobenius_updated_minus_previous_over_frobenius_updated"
    ]
    precision: float
    branch_energy_tolerance_ev: float
    stationarity_tolerance_ev: float
    max_iter: int
    oda_stall_threshold: float
    max_oda_lambda: float
    seed_records: tuple[Vituri2024SCFSeedReceipt, ...]
    callback_receipts: tuple[Vituri2024SCFCallbackReceipt, ...]
    branch_energy_comparison_policy: Literal[
        "compare_canonical_energy_of_all_attested_converged_branches"
    ]
    transfer_learning_policy: Literal[
        "include_hash_bound_neighbor_sources_from_both_density_sides"
    ]
    restart_checkpoint_policy: Literal[
        "hash_bound_atomic_checkpoint_with_exact_policy_restart"
    ]
    checkpoint_interval: int
    uniform_weight_representation: Literal[
        "implicit_dimensionless_unit_weight_per_finite_mesh_state"
    ]
    final_exit_semantics: Literal[
        "core_exit_reason_plus_recomputed_final_raw_metric"
    ]
    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    authority_kind: ReceiptAuthority
    provenance: str
    core_module: Literal["mean_field.core.hf"] = "mean_field.core.hf"
    core_entrypoint: Literal["run_hartree_fock_problem"] = "run_hartree_fock_problem"
    oda_policy: Literal["generic_core_oda_delta_interaction_builder"] = (
        "generic_core_oda_delta_interaction_builder"
    )
    exact_numerical_policy_paper_reported: bool = field(default=False, init=False)
    forks_solver: bool = field(default=False, init=False)
    weighted_quadrature_claimed: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.convergence_rule not in ("raw", "mixed"):
            raise ValueError("unsupported generic-core convergence rule")
        if (
            self.convergence_metric_identity
            != "mean_field.core.hf.occupations.calculate_norm_convergence"
            or self.convergence_metric_normalization
            != "frobenius_updated_minus_previous_over_frobenius_updated"
        ):
            raise ValueError("SCF convergence metric identity/normalization was changed")
        precision = _positive(self.precision, "SCF precision")
        branch_tolerance = _positive(
            self.branch_energy_tolerance_ev, "branch-energy tolerance"
        )
        stationarity = _positive(
            self.stationarity_tolerance_ev, "stationarity tolerance"
        )
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "branch_energy_tolerance_ev", branch_tolerance)
        object.__setattr__(self, "stationarity_tolerance_ev", stationarity)
        max_iter = _strict_int(self.max_iter, "max_iter")
        checkpoint_interval = _strict_int(self.checkpoint_interval, "checkpoint_interval")
        if max_iter < 1 or checkpoint_interval < 1 or checkpoint_interval > max_iter:
            raise ValueError("invalid SCF iteration/checkpoint policy")
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "checkpoint_interval", checkpoint_interval)
        object.__setattr__(
            self,
            "oda_stall_threshold",
            _positive(self.oda_stall_threshold, "ODA stall threshold"),
        )
        max_lambda = _positive(self.max_oda_lambda, "maximum ODA lambda")
        if max_lambda > 1.0:
            raise ValueError("max_oda_lambda must lie in (0,1]")
        object.__setattr__(self, "max_oda_lambda", max_lambda)

        seeds = tuple(self.seed_records)
        if len(seeds) < 2 or any(type(seed) is not Vituri2024SCFSeedReceipt for seed in seeds):
            raise TypeError("SCF policy requires multiple typed seed records")
        labels = tuple(seed.seed_label for seed in seeds)
        seed_pairs = tuple((seed.seed_label, seed.seed_value, seed.init_mode) for seed in seeds)
        if len(set(labels)) != len(labels) or len(set(seed_pairs)) != len(seed_pairs):
            raise ValueError("SCF seed labels and label/value/init-mode tuples must be unique")
        object.__setattr__(self, "seed_records", seeds)

        callbacks = tuple(self.callback_receipts)
        if any(type(item) is not Vituri2024SCFCallbackReceipt for item in callbacks):
            raise TypeError("SCF callbacks require typed callback receipts")
        if tuple(item.role for item in callbacks) != _SCF_CALLBACK_ROLES:
            raise ValueError("SCF callback inventory must match the generic-core hook order")
        for callback in callbacks:
            if (
                callback.role in _REQUIRED_CALLABLE_CALLBACKS
                and callback.implementation_kind != "callable"
            ):
                raise ValueError(f"SCF callback {callback.role} must be callable")
        oda_parameterizer = callbacks[_SCF_CALLBACK_ROLES.index("oda_parameterizer")]
        if oda_parameterizer.implementation_kind != "none":
            raise ValueError("delta-interaction ODA must not also install oda_parameterizer")
        object.__setattr__(self, "callback_receipts", callbacks)

        locked = (
            self.branch_energy_comparison_policy
            == "compare_canonical_energy_of_all_attested_converged_branches",
            self.transfer_learning_policy
            == "include_hash_bound_neighbor_sources_from_both_density_sides",
            self.restart_checkpoint_policy
            == "hash_bound_atomic_checkpoint_with_exact_policy_restart",
            self.uniform_weight_representation
            == "implicit_dimensionless_unit_weight_per_finite_mesh_state",
            self.final_exit_semantics
            == "core_exit_reason_plus_recomputed_final_raw_metric",
            self.core_module == "mean_field.core.hf",
            self.core_entrypoint == "run_hartree_fock_problem",
            self.oda_policy == "generic_core_oda_delta_interaction_builder",
            self.forks_solver is False,
            self.weighted_quadrature_claimed is False,
        )
        if not all(locked):
            raise ValueError("Vituri SCF policy must use the exact generic core SCF/ODA contract")
        for value, label in (
            (self.provider_fingerprint, "SCF provider"),
            (self.source_artifact_sha256, "SCF source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "SCF source commit")
        _authority(self.authority_kind, "SCF authority")
        _text(self.provenance, "SCF provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024FunctionalComponentReceipt:
    """Source-bound identity of one shared-functional implementation role."""

    role: FunctionalRole
    symbol: str
    implementation_fingerprint: str
    source_commit: str
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if self.role not in (
            "scalar_energy",
            "fock_derivative",
            "finite_q_hessian",
            "interaction_form_factor",
        ):
            raise ValueError("invalid shared-functional component role")
        _text(self.symbol, f"{self.role} symbol")
        _sha256(self.implementation_fingerprint, f"{self.role} implementation")
        _commit(self.source_commit, f"{self.role} source commit")
        _sha256(self.source_artifact_sha256, f"{self.role} source artifact")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024FiniteDifferenceEvidenceReceipt:
    """Context-complete external finite-difference evidence; never replayed here."""

    validation_kind: FiniteDifferenceKind
    residual: float
    tolerance: float
    source_state_sha256: str
    geometry_receipt_fingerprint: str
    ensemble_receipt_fingerprint: str
    perturbation_inventory_sha256: str
    perturbation_normalization: Literal[
        "unit_frobenius_norm_hermitian_projector_tangent"
    ]
    matrix_norm: Literal["frobenius"]
    q_probe_inventory_sha256: str | None
    finite_difference_step_ladder: tuple[float, ...]
    comparison_identity: FiniteDifferenceComparisonIdentity
    left_implementation_fingerprint: str
    right_implementation_fingerprint: str
    evidence_context_fingerprint: str
    evidence_artifact_sha256: str
    source_commit: str
    source_artifact_sha256: str
    provenance: str
    recomputed_by_preflight: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.validation_kind not in ("fock_first_derivative", "finite_q_hessian"):
            raise ValueError("invalid finite-difference validation kind")
        residual, tolerance = _require_residual(
            self.residual, self.tolerance, self.validation_kind
        )
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "tolerance", tolerance)
        if (
            self.perturbation_normalization
            != "unit_frobenius_norm_hermitian_projector_tangent"
            or self.matrix_norm != "frobenius"
        ):
            raise ValueError("finite-difference perturbation normalization/norm was changed")
        steps = tuple(
            _positive(value, "finite-difference step")
            for value in self.finite_difference_step_ladder
        )
        if len(steps) < 3 or any(left <= right for left, right in zip(steps, steps[1:])):
            raise ValueError("finite-difference step ladder must have >=3 decreasing steps")
        object.__setattr__(self, "finite_difference_step_ladder", steps)
        expected_comparison: FiniteDifferenceComparisonIdentity
        if self.validation_kind == "finite_q_hessian":
            _sha256(self.q_probe_inventory_sha256, "finite-q Hessian q/probe inventory")
            expected_comparison = "fock_derivative_vs_finite_q_hessian"
        else:
            if self.q_probe_inventory_sha256 is not None:
                raise ValueError(
                    "Fock first-derivative evidence must not claim a q/probe inventory"
                )
            expected_comparison = "scalar_energy_vs_fock_derivative"
        if self.comparison_identity != expected_comparison:
            raise ValueError(
                "finite-difference comparison identity contradicts validation kind"
            )
        for value, label in (
            (self.source_state_sha256, "finite-difference source state"),
            (self.geometry_receipt_fingerprint, "finite-difference geometry"),
            (self.ensemble_receipt_fingerprint, "finite-difference ensemble"),
            (self.perturbation_inventory_sha256, "perturbation inventory"),
            (
                self.left_implementation_fingerprint,
                "left compared implementation",
            ),
            (
                self.right_implementation_fingerprint,
                "right compared implementation",
            ),
            (self.evidence_artifact_sha256, "finite-difference artifact"),
            (self.source_artifact_sha256, "finite-difference source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "finite-difference source commit")
        _text(self.provenance, "finite-difference provenance")
        _require_context_fingerprint(
            self.evidence_context_fingerprint,
            {
                "validation_kind": self.validation_kind,
                "residual": residual,
                "tolerance": tolerance,
                "source_state_sha256": self.source_state_sha256,
                "geometry_receipt_fingerprint": self.geometry_receipt_fingerprint,
                "ensemble_receipt_fingerprint": self.ensemble_receipt_fingerprint,
                "perturbation_inventory_sha256": self.perturbation_inventory_sha256,
                "perturbation_normalization": self.perturbation_normalization,
                "matrix_norm": self.matrix_norm,
                "q_probe_inventory_sha256": self.q_probe_inventory_sha256,
                "finite_difference_step_ladder": steps,
                "comparison_identity": self.comparison_identity,
                "left_implementation_fingerprint": (
                    self.left_implementation_fingerprint
                ),
                "right_implementation_fingerprint": (
                    self.right_implementation_fingerprint
                ),
                "evidence_artifact_sha256": self.evidence_artifact_sha256,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "finite-difference evidence context fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024SharedFunctionalReceipt:
    """One source for E, dE/dP, d2E, interaction, reference, and q=0 policy."""

    source_commit: str
    source_artifact_sha256: str
    provider_fingerprint: str
    geometry_receipt_fingerprint: str
    ensemble_receipt_fingerprint: str
    normal_order_reference_fingerprint: str
    q0_policy_fingerprint: str
    scalar_energy: Vituri2024FunctionalComponentReceipt
    fock_derivative: Vituri2024FunctionalComponentReceipt
    finite_q_hessian: Vituri2024FunctionalComponentReceipt
    interaction_form_factor: Vituri2024FunctionalComponentReceipt
    interaction_receipt_fingerprint: str
    fock_finite_difference: Vituri2024FiniteDifferenceEvidenceReceipt
    hessian_finite_difference: Vituri2024FiniteDifferenceEvidenceReceipt
    authority_kind: ReceiptAuthority
    provenance: str
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _commit(self.source_commit, "shared-functional source commit")
        for value, label in (
            (self.source_artifact_sha256, "shared-functional source artifact"),
            (self.provider_fingerprint, "shared-functional provider"),
            (self.geometry_receipt_fingerprint, "shared-functional geometry"),
            (self.ensemble_receipt_fingerprint, "shared-functional ensemble"),
            (self.normal_order_reference_fingerprint, "shared-functional reference"),
            (self.q0_policy_fingerprint, "shared-functional q0 policy"),
            (self.interaction_receipt_fingerprint, "interaction receipt"),
        ):
            _sha256(value, label)
        expected = (
            (self.scalar_energy, "scalar_energy"),
            (self.fock_derivative, "fock_derivative"),
            (self.finite_q_hessian, "finite_q_hessian"),
            (self.interaction_form_factor, "interaction_form_factor"),
        )
        for component, role in expected:
            if type(component) is not Vituri2024FunctionalComponentReceipt:
                raise TypeError(f"{role} requires a typed component receipt")
            if component.role != role:
                raise ValueError(f"shared-functional {role} receipt has the wrong role")
            if (
                component.source_commit != self.source_commit
                or component.source_artifact_sha256 != self.source_artifact_sha256
            ):
                raise ValueError("all functional components must share one source artifact+commit")
        evidence_expected = (
            (
                self.fock_finite_difference,
                "fock_first_derivative",
                "scalar_energy_vs_fock_derivative",
                self.scalar_energy.implementation_fingerprint,
                self.fock_derivative.implementation_fingerprint,
            ),
            (
                self.hessian_finite_difference,
                "finite_q_hessian",
                "fock_derivative_vs_finite_q_hessian",
                self.fock_derivative.implementation_fingerprint,
                self.finite_q_hessian.implementation_fingerprint,
            ),
        )
        source_states: set[str] = set()
        for (
            evidence,
            kind,
            comparison,
            left_implementation,
            right_implementation,
        ) in evidence_expected:
            if type(evidence) is not Vituri2024FiniteDifferenceEvidenceReceipt:
                raise TypeError(f"{kind} requires typed finite-difference evidence")
            if evidence.validation_kind != kind:
                raise ValueError(f"finite-difference evidence kind mismatch for {kind}")
            if (
                evidence.source_commit != self.source_commit
                or evidence.source_artifact_sha256 != self.source_artifact_sha256
            ):
                raise ValueError(
                    "finite-difference evidence source does not match functional source"
                )
            if (
                evidence.geometry_receipt_fingerprint
                != self.geometry_receipt_fingerprint
                or evidence.ensemble_receipt_fingerprint
                != self.ensemble_receipt_fingerprint
            ):
                raise ValueError("finite-difference geometry/ensemble context mismatch")
            if evidence.comparison_identity != comparison:
                raise ValueError("finite-difference comparison identity mismatch")
            if (
                evidence.left_implementation_fingerprint != left_implementation
                or evidence.right_implementation_fingerprint != right_implementation
            ):
                raise ValueError(
                    "finite-difference compared implementation pair mismatch"
                )
            source_states.add(evidence.source_state_sha256)
        if len(source_states) != 1:
            raise ValueError("finite-difference checks must use one source-state hash")
        _authority(self.authority_kind, "shared-functional authority")
        _text(self.provenance, "shared-functional provenance")

    @property
    def source_state_sha256(self) -> str:
        return self.fock_finite_difference.source_state_sha256

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024MetallicityEvidenceReceipt:
    """Typed finite-volume evidence that the selected-spin band straddles mu."""

    source_state_sha256: str
    geometry_receipt_fingerprint: str
    ordered_momentum_mesh_sha256: str
    ordered_energies_sha256: str
    ordered_occupations_sha256: str
    selected_spin: Literal[-1, 1]
    chemical_potential_ev: float
    selected_spin_band_min_ev: float
    selected_spin_band_max_ev: float
    selected_spin_occupied_state_count: int
    selected_spin_unoccupied_state_count: int
    metallicity_tolerance_ev: float
    evidence_sha256: str
    context_fingerprint: str
    source_commit: str
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_state_sha256, "metallicity source state"),
            (self.geometry_receipt_fingerprint, "metallicity geometry"),
            (self.ordered_momentum_mesh_sha256, "metallicity ordered mesh"),
            (self.ordered_energies_sha256, "metallicity ordered energies"),
            (self.ordered_occupations_sha256, "metallicity ordered occupations"),
            (self.evidence_sha256, "metallicity evidence"),
            (self.source_artifact_sha256, "metallicity source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "metallicity source commit")
        if self.selected_spin not in (-1, 1):
            raise ValueError("metallicity selected_spin must be exactly -1 or +1")
        mu = _finite(self.chemical_potential_ev, "chemical potential")
        minimum = _finite(self.selected_spin_band_min_ev, "selected-spin band minimum")
        maximum = _finite(self.selected_spin_band_max_ev, "selected-spin band maximum")
        tolerance = _positive(self.metallicity_tolerance_ev, "metallicity tolerance")
        occupied = _strict_int(
            self.selected_spin_occupied_state_count, "selected-spin occupied count"
        )
        unoccupied = _strict_int(
            self.selected_spin_unoccupied_state_count, "selected-spin unoccupied count"
        )
        if occupied < 1 or unoccupied < 1:
            raise ValueError("metallicity evidence requires occupied and unoccupied states")
        if not (minimum < mu - tolerance and maximum > mu + tolerance):
            raise ValueError("selected-spin band does not straddle chemical potential")
        for name, value in (
            ("chemical_potential_ev", mu),
            ("selected_spin_band_min_ev", minimum),
            ("selected_spin_band_max_ev", maximum),
            ("selected_spin_occupied_state_count", occupied),
            ("selected_spin_unoccupied_state_count", unoccupied),
            ("metallicity_tolerance_ev", tolerance),
        ):
            object.__setattr__(self, name, value)
        _require_context_fingerprint(
            self.context_fingerprint,
            {
                "source_state_sha256": self.source_state_sha256,
                "geometry_receipt_fingerprint": self.geometry_receipt_fingerprint,
                "ordered_momentum_mesh_sha256": self.ordered_momentum_mesh_sha256,
                "ordered_energies_sha256": self.ordered_energies_sha256,
                "ordered_occupations_sha256": self.ordered_occupations_sha256,
                "selected_spin": self.selected_spin,
                "chemical_potential_ev": mu,
                "selected_spin_band_min_ev": minimum,
                "selected_spin_band_max_ev": maximum,
                "selected_spin_occupied_state_count": occupied,
                "selected_spin_unoccupied_state_count": unoccupied,
                "metallicity_tolerance_ev": tolerance,
                "evidence_sha256": self.evidence_sha256,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "metallicity context fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024ValleyPocketEvidenceReceipt:
    """One no-wrap connected-hole-component receipt for one valley."""

    valley: Literal[-1, 1]
    source_state_sha256: str
    geometry_receipt_fingerprint: str
    ordered_momentum_mesh_sha256: str
    ordered_occupations_sha256: str
    selected_spin: Literal[-1, 1]
    hole_component_count: int
    hole_state_count: int
    adjacency_convention: Literal["four_neighbor_finite_domain_no_wrap"]
    base_mesh_point_count: int
    component_evidence_sha256: str
    refinement_mesh_sha256: str
    refinement_point_count: int
    refinement_evidence_sha256: str
    lifshitz_margin_ev: float
    lifshitz_tolerance_ev: float
    context_fingerprint: str
    source_commit: str
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if self.valley not in (-1, 1) or self.selected_spin not in (-1, 1):
            raise ValueError("pocket valley and selected spin must be exactly -1 or +1")
        components = _strict_int(self.hole_component_count, "hole component count")
        holes = _strict_int(self.hole_state_count, "pocket hole-state count")
        if components != 1:
            raise ValueError("each valley must have exactly one connected hole component")
        if holes < 1:
            raise ValueError("each valley pocket must contain at least one hole state")
        if self.adjacency_convention != "four_neighbor_finite_domain_no_wrap":
            raise ValueError("pocket adjacency must use the finite-domain no-wrap convention")
        base_count = _strict_int(self.base_mesh_point_count, "pocket base point count")
        refinement_count = _strict_int(
            self.refinement_point_count, "pocket refinement point count"
        )
        if base_count < 1 or refinement_count <= base_count:
            raise ValueError("pocket refinement must contain more points than the base mesh")
        margin = _positive(self.lifshitz_margin_ev, "Lifshitz margin")
        tolerance = _positive(self.lifshitz_tolerance_ev, "Lifshitz tolerance")
        if margin <= tolerance:
            raise ValueError("positive Lifshitz margin must exceed its tolerance")
        for name, value in (
            ("hole_component_count", components),
            ("hole_state_count", holes),
            ("base_mesh_point_count", base_count),
            ("refinement_point_count", refinement_count),
            ("lifshitz_margin_ev", margin),
            ("lifshitz_tolerance_ev", tolerance),
        ):
            object.__setattr__(self, name, value)
        for value, label in (
            (self.source_state_sha256, "pocket source state"),
            (self.geometry_receipt_fingerprint, "pocket geometry"),
            (self.ordered_momentum_mesh_sha256, "pocket ordered mesh"),
            (self.ordered_occupations_sha256, "pocket ordered occupations"),
            (self.component_evidence_sha256, "pocket component evidence"),
            (self.refinement_mesh_sha256, "pocket refinement mesh"),
            (self.refinement_evidence_sha256, "pocket refinement evidence"),
            (self.source_artifact_sha256, "pocket source artifact"),
        ):
            _sha256(value, label)
        _commit(self.source_commit, "pocket source commit")
        _require_context_fingerprint(
            self.context_fingerprint,
            {
                "valley": self.valley,
                "source_state_sha256": self.source_state_sha256,
                "geometry_receipt_fingerprint": self.geometry_receipt_fingerprint,
                "ordered_momentum_mesh_sha256": self.ordered_momentum_mesh_sha256,
                "ordered_occupations_sha256": self.ordered_occupations_sha256,
                "selected_spin": self.selected_spin,
                "hole_component_count": components,
                "hole_state_count": holes,
                "adjacency_convention": self.adjacency_convention,
                "base_mesh_point_count": base_count,
                "component_evidence_sha256": self.component_evidence_sha256,
                "refinement_mesh_sha256": self.refinement_mesh_sha256,
                "refinement_point_count": refinement_count,
                "refinement_evidence_sha256": self.refinement_evidence_sha256,
                "lifshitz_margin_ev": margin,
                "lifshitz_tolerance_ev": tolerance,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "pocket context fingerprint",
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024BranchEnergyReceipt:
    """One seed-bound canonical branch-energy and exact core-exit table row."""

    seed: Vituri2024SCFSeedReceipt
    attested_exit_reason: SCFExitReason
    iterations: int
    terminal_norm_raw: float
    terminal_norm_mixed: float
    terminal_norm_selected: float
    terminal_oda_lambda: float
    final_replay_raw_metric: float
    canonical_energy_ev: float

    def __post_init__(self) -> None:
        if type(self.seed) is not Vituri2024SCFSeedReceipt:
            raise TypeError("branch row requires a typed seed receipt")
        if self.attested_exit_reason not in ("converged", "oda_stall", "max_iter"):
            raise ValueError("branch row has an unsupported exit reason")
        iterations = _strict_int(self.iterations, "branch iterations")
        if iterations < 1:
            raise ValueError("branch iterations must be positive")
        object.__setattr__(self, "iterations", iterations)
        for name, value, label in (
            ("terminal_norm_raw", self.terminal_norm_raw, "terminal raw norm"),
            (
                "terminal_norm_mixed",
                self.terminal_norm_mixed,
                "terminal mixed norm",
            ),
            (
                "terminal_norm_selected",
                self.terminal_norm_selected,
                "terminal selected norm",
            ),
            (
                "terminal_oda_lambda",
                self.terminal_oda_lambda,
                "terminal ODA lambda",
            ),
            (
                "final_replay_raw_metric",
                self.final_replay_raw_metric,
                "final replay raw metric",
            ),
        ):
            object.__setattr__(self, name, _nonnegative(value, f"branch {label}"))
        object.__setattr__(
            self,
            "canonical_energy_ev",
            _finite(self.canonical_energy_ev, "branch canonical energy"),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024AttestedHalfMetalSourceReceipt:
    """Hash-bound receipt for a provider-attested source; arrays are not loaded."""

    source_commit: str
    source_artifact_sha256: str
    provider_fingerprint: str
    source_state_sha256: str
    ordered_orbitals_sha256: str
    ordered_energies_sha256: str
    ordered_occupations_sha256: str
    ordered_projector_sha256: str
    ordered_fock_sha256: str
    geometry_receipt_fingerprint: str
    ensemble_receipt_fingerprint: str
    scf_policy_receipt_fingerprint: str
    shared_functional_receipt_fingerprint: str
    area_angstrom_squared: float
    finite_area_receipt_fingerprint: str
    ordered_momentum_mesh_sha256: str
    target_density_cm2: float
    measured_density_cm2: float
    density_residual_cm2: float
    density_tolerance_cm2: float
    chemical_potential_ev: float
    attested_exit_reason: SCFExitReason
    final_replay_raw_metric: float
    final_replay_raw_precision: float
    fock_projector_commutator_residual_ev: float
    stationarity_tolerance_ev: float
    projector_idempotency_residual: float
    projector_idempotency_tolerance: float
    projector_hermiticity_residual: float
    projector_hermiticity_tolerance: float
    fock_hermiticity_residual_ev: float
    fock_hermiticity_tolerance_ev: float
    aufbau_min_unoccupied_minus_max_occupied_ev: float
    aufbau_occupation_violation_ev: float
    aufbau_tolerance_ev: float
    selected_spin: Literal[-1, 1]
    valley_plus_hole_count: int
    valley_minus_hole_count: int
    selected_spin_hole_count: int
    opposite_spin_hole_count: int
    metallicity_evidence: Vituri2024MetallicityEvidenceReceipt
    pocket_evidence: tuple[
        Vituri2024ValleyPocketEvidenceReceipt,
        Vituri2024ValleyPocketEvidenceReceipt,
    ]
    branch_comparison_evidence_sha256: str
    branch_energy_table_sha256: str
    branch_records: tuple[Vituri2024BranchEnergyReceipt, ...]
    branch_energy_functional_fingerprint: str
    branch_table_context_fingerprint: str
    selected_branch_label: str
    selected_branch_energy_ev: float
    minimum_compared_branch_energy_ev: float
    branch_energy_residual_ev: float
    branch_energy_tolerance_ev: float
    provenance: str
    arrays_recomputed_by_preflight: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)
    paper_direct_claim_allowed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _commit(self.source_commit, "attested-source commit")
        for value, label in (
            (self.source_artifact_sha256, "attested-source artifact"),
            (self.provider_fingerprint, "attested-source provider"),
            (self.source_state_sha256, "source-state hash"),
            (self.ordered_orbitals_sha256, "ordered orbitals"),
            (self.ordered_energies_sha256, "ordered energies"),
            (self.ordered_occupations_sha256, "ordered occupations"),
            (self.ordered_projector_sha256, "ordered projector"),
            (self.ordered_fock_sha256, "ordered Fock"),
            (self.geometry_receipt_fingerprint, "geometry receipt"),
            (self.ensemble_receipt_fingerprint, "ensemble receipt"),
            (self.scf_policy_receipt_fingerprint, "SCF-policy receipt"),
            (self.shared_functional_receipt_fingerprint, "shared-functional receipt"),
            (self.finite_area_receipt_fingerprint, "finite-area receipt"),
            (self.ordered_momentum_mesh_sha256, "source ordered momentum mesh"),
            (self.branch_comparison_evidence_sha256, "branch-comparison evidence"),
            (self.branch_energy_table_sha256, "branch energy table"),
            (self.branch_energy_functional_fingerprint, "branch energy functional"),
        ):
            _sha256(value, label)
        area = _positive(self.area_angstrom_squared, "source finite area")
        object.__setattr__(self, "area_angstrom_squared", area)
        _require_context_fingerprint(
            self.source_state_sha256,
            {
                "ordered_orbitals_sha256": self.ordered_orbitals_sha256,
                "ordered_energies_sha256": self.ordered_energies_sha256,
                "ordered_occupations_sha256": self.ordered_occupations_sha256,
                "ordered_projector_sha256": self.ordered_projector_sha256,
                "ordered_fock_sha256": self.ordered_fock_sha256,
                "geometry_receipt_fingerprint": self.geometry_receipt_fingerprint,
                "ensemble_receipt_fingerprint": self.ensemble_receipt_fingerprint,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "source-state hash",
        )

        target_density = _finite(self.target_density_cm2, "target density")
        measured_density = _finite(self.measured_density_cm2, "measured density")
        density_residual, density_tolerance = _require_residual(
            self.density_residual_cm2, self.density_tolerance_cm2, "density closure"
        )
        _require_abs_residual(
            density_residual, target_density, measured_density, "density"
        )
        for name, value in (
            ("target_density_cm2", target_density),
            ("measured_density_cm2", measured_density),
            ("density_residual_cm2", density_residual),
            ("density_tolerance_cm2", density_tolerance),
            (
                "chemical_potential_ev",
                _finite(self.chemical_potential_ev, "chemical potential"),
            ),
        ):
            object.__setattr__(self, name, value)

        if self.attested_exit_reason != "converged":
            raise ValueError("attested selected source exit must be converged")
        final_replay = _nonnegative(
            self.final_replay_raw_metric, "final replay raw SCF metric"
        )
        final_precision = _positive(
            self.final_replay_raw_precision, "final replay raw SCF precision"
        )
        if final_replay > final_precision:
            raise ValueError("attested source final replay exceeds SCF precision")
        object.__setattr__(self, "final_replay_raw_metric", final_replay)
        object.__setattr__(self, "final_replay_raw_precision", final_precision)
        residual_pairs = (
            (
                "fock_projector_commutator_residual_ev",
                "stationarity_tolerance_ev",
                "[F,P] stationarity",
            ),
            (
                "projector_idempotency_residual",
                "projector_idempotency_tolerance",
                "projector idempotency",
            ),
            (
                "projector_hermiticity_residual",
                "projector_hermiticity_tolerance",
                "projector Hermiticity",
            ),
            (
                "fock_hermiticity_residual_ev",
                "fock_hermiticity_tolerance_ev",
                "Fock Hermiticity",
            ),
            (
                "aufbau_occupation_violation_ev",
                "aufbau_tolerance_ev",
                "Aufbau occupation",
            ),
            (
                "branch_energy_residual_ev",
                "branch_energy_tolerance_ev",
                "branch energy selection",
            ),
        )
        for residual_name, tolerance_name, label in residual_pairs:
            residual, tolerance = _require_residual(
                getattr(self, residual_name), getattr(self, tolerance_name), label
            )
            object.__setattr__(self, residual_name, residual)
            object.__setattr__(self, tolerance_name, tolerance)
        gap = _finite(
            self.aufbau_min_unoccupied_minus_max_occupied_ev,
            "Aufbau finite-volume gap",
        )
        if gap < -self.aufbau_tolerance_ev:
            raise ValueError("finite-volume Aufbau gap violates occupation tolerance")
        object.__setattr__(self, "aufbau_min_unoccupied_minus_max_occupied_ev", gap)

        if self.selected_spin not in (-1, 1):
            raise ValueError("selected_spin must be exactly -1 or +1")
        plus = _strict_int(self.valley_plus_hole_count, "K-valley hole count")
        minus = _strict_int(self.valley_minus_hole_count, "K'-valley hole count")
        selected = _strict_int(
            self.selected_spin_hole_count, "selected-spin total hole count"
        )
        opposite = _strict_int(self.opposite_spin_hole_count, "opposite-spin hole count")
        if (
            plus < 1
            or plus != minus
            or selected != plus + minus
            or opposite != 0
        ):
            raise ValueError(
                "source must have equal nonzero two-valley selected-spin hole counts"
            )
        for name, value in (
            ("valley_plus_hole_count", plus),
            ("valley_minus_hole_count", minus),
            ("selected_spin_hole_count", selected),
            ("opposite_spin_hole_count", opposite),
        ):
            object.__setattr__(self, name, value)

        if type(self.metallicity_evidence) is not Vituri2024MetallicityEvidenceReceipt:
            raise TypeError("source requires typed selected-spin metallicity evidence")
        pockets = tuple(self.pocket_evidence)
        if len(pockets) != 2 or any(
            type(item) is not Vituri2024ValleyPocketEvidenceReceipt for item in pockets
        ):
            raise TypeError("source requires two typed valley-pocket evidence receipts")
        if tuple(item.valley for item in pockets) != (-1, 1):
            raise ValueError("pocket evidence must be ordered as valleys (-1,+1)")
        if tuple(item.hole_state_count for item in pockets) != (minus, plus):
            raise ValueError(
                "pocket hole-state count does not match its corresponding valley count"
            )
        if (
            self.metallicity_evidence.selected_spin_unoccupied_state_count
            != selected
        ):
            raise ValueError(
                "metallicity unoccupied count must equal selected-spin hole count"
            )
        object.__setattr__(self, "pocket_evidence", pockets)

        records = tuple(self.branch_records)
        if len(records) < 2 or any(
            type(item) is not Vituri2024BranchEnergyReceipt for item in records
        ):
            raise TypeError("branch table requires multiple typed rows")
        labels = tuple(item.seed.seed_label for item in records)
        if len(set(labels)) != len(labels):
            raise ValueError("branch table seed labels must be unique")
        object.__setattr__(self, "branch_records", records)
        _require_context_fingerprint(
            self.branch_table_context_fingerprint,
            {
                "branch_energy_table_sha256": self.branch_energy_table_sha256,
                "branch_comparison_evidence_sha256": (
                    self.branch_comparison_evidence_sha256
                ),
                "branch_records": [asdict(item) for item in records],
                "branch_energy_functional_fingerprint": (
                    self.branch_energy_functional_fingerprint
                ),
                "source_state_sha256": self.source_state_sha256,
                "source_commit": self.source_commit,
                "source_artifact_sha256": self.source_artifact_sha256,
            },
            "branch-table context fingerprint",
        )
        selected_label = _text(self.selected_branch_label, "selected branch label")
        selected_energy = _finite(self.selected_branch_energy_ev, "selected branch energy")
        minimum_energy = _finite(
            self.minimum_compared_branch_energy_ev, "minimum compared branch energy"
        )
        selected_rows = [item for item in records if item.seed.seed_label == selected_label]
        if len(selected_rows) != 1:
            raise ValueError("selected branch label does not identify exactly one table row")
        selected_row = selected_rows[0]
        if (
            selected_row.attested_exit_reason != "converged"
            or selected_row.attested_exit_reason != self.attested_exit_reason
            or selected_row.final_replay_raw_metric != final_replay
            or selected_row.canonical_energy_ev != selected_energy
        ):
            raise ValueError(
                "selected branch row/source exit, final replay, or energy mismatch"
            )
        _require_abs_residual(
            self.branch_energy_residual_ev,
            selected_energy,
            minimum_energy,
            "branch energy",
        )
        object.__setattr__(self, "selected_branch_energy_ev", selected_energy)
        object.__setattr__(self, "minimum_compared_branch_energy_ev", minimum_energy)
        _text(self.provenance, "attested-source provenance")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Vituri2024HFReceiptSetStatus:
    """Receipt completeness status with all scientific execution claims false."""

    receipt_set_complete: bool
    missing_receipts: tuple[str, ...]
    scientific_execution_verified: bool = field(default=False, init=False)
    arrays_recomputed: bool = field(default=False, init=False)
    provider_methods_executed: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        missing = tuple(_text(item, "missing receipt label") for item in self.missing_receipts)
        if self.receipt_set_complete != (not missing):
            raise ValueError("receipt-set status contradicts its missing-receipt inventory")
        object.__setattr__(self, "missing_receipts", missing)
        if any(
            (
                self.scientific_execution_verified,
                self.arrays_recomputed,
                self.provider_methods_executed,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("receipt status cannot claim scientific execution")


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFSpec:
    """Paper target plus optional authority receipts; execution stays unresolved."""

    paper_target: Vituri2024HalfMetalPaperTarget = field(
        default_factory=Vituri2024HalfMetalPaperTarget
    )
    geometry: Vituri2024HFGeometryReceipt | None = None
    ensemble: Vituri2024HFEnsembleReceipt | None = None
    scf_policy: Vituri2024HFSCFPolicyReceipt | None = None
    shared_functional: Vituri2024SharedFunctionalReceipt | None = None
    attested_source: Vituri2024AttestedHalfMetalSourceReceipt | None = None
    scope: str = VITURI2024_HALF_METAL_HF_SCOPE

    def __post_init__(self) -> None:
        if type(self.paper_target) is not Vituri2024HalfMetalPaperTarget:
            raise TypeError("paper_target must be a typed Vituri paper target")
        if self.scope != VITURI2024_HALF_METAL_HF_SCOPE:
            raise ValueError("Vituri half-metal HF receipt-preflight scope was changed")
        expected_types = (
            (self.geometry, Vituri2024HFGeometryReceipt, "geometry"),
            (self.ensemble, Vituri2024HFEnsembleReceipt, "ensemble"),
            (self.scf_policy, Vituri2024HFSCFPolicyReceipt, "SCF policy"),
            (self.shared_functional, Vituri2024SharedFunctionalReceipt, "shared functional"),
            (
                self.attested_source,
                Vituri2024AttestedHalfMetalSourceReceipt,
                "attested source",
            ),
        )
        for receipt, receipt_type, label in expected_types:
            if receipt is not None and type(receipt) is not receipt_type:
                raise TypeError(f"{label} must use its typed receipt")

        authority_receipts = (
            self.geometry,
            self.ensemble,
            self.scf_policy,
            self.shared_functional,
            self.attested_source,
        )
        source_identities = {
            (receipt.source_commit, receipt.source_artifact_sha256)
            for receipt in authority_receipts
            if receipt is not None
        }
        if len(source_identities) > 1:
            raise ValueError("aggregate receipt source artifact+commit identities do not close")
        provider_fingerprints = {
            receipt.provider_fingerprint
            for receipt in authority_receipts
            if receipt is not None
        }
        if len(provider_fingerprints) > 1:
            raise ValueError("aggregate provider fingerprints do not close")

        if self.geometry is not None and self.ensemble is not None:
            if self.geometry.delta1_mev != self.ensemble.delta1_mev:
                raise ValueError("geometry/ensemble Delta1 reproduction choice mismatch")
        if self.ensemble is not None:
            if self.ensemble.target_density_cm2 != self.paper_target.fig3a_density_cm2:
                raise ValueError("ensemble density does not match the Fig. 3a paper target")
        if self.ensemble is not None and self.shared_functional is not None:
            expected_shared = (
                (
                    self.shared_functional.ensemble_receipt_fingerprint,
                    self.ensemble.fingerprint,
                    "ensemble",
                ),
                (
                    self.shared_functional.normal_order_reference_fingerprint,
                    self.ensemble.normal_order_reference_fingerprint,
                    "normal-order reference",
                ),
                (
                    self.shared_functional.q0_policy_fingerprint,
                    self.ensemble.q0_policy_fingerprint,
                    "q0 policy",
                ),
                (
                    self.shared_functional.interaction_receipt_fingerprint,
                    self.ensemble.interaction_receipt_fingerprint,
                    "interaction receipt",
                ),
            )
            for actual, required, label in expected_shared:
                if actual != required:
                    raise ValueError(f"ensemble/shared-functional {label} mismatch")
        if self.geometry is not None and self.shared_functional is not None:
            if (
                self.shared_functional.geometry_receipt_fingerprint
                != self.geometry.fingerprint
            ):
                raise ValueError("geometry/shared-functional fingerprint mismatch")

        if self.attested_source is not None:
            if None in (
                self.geometry,
                self.ensemble,
                self.scf_policy,
                self.shared_functional,
            ):
                raise ValueError("attested source cannot precede prerequisite receipts")
            assert self.geometry is not None
            assert self.ensemble is not None
            assert self.scf_policy is not None
            assert self.shared_functional is not None
            source = self.attested_source
            expected = (
                (source.geometry_receipt_fingerprint, self.geometry.fingerprint, "geometry"),
                (source.ensemble_receipt_fingerprint, self.ensemble.fingerprint, "ensemble"),
                (
                    source.scf_policy_receipt_fingerprint,
                    self.scf_policy.fingerprint,
                    "SCF policy",
                ),
                (
                    source.shared_functional_receipt_fingerprint,
                    self.shared_functional.fingerprint,
                    "shared functional",
                ),
                (
                    source.finite_area_receipt_fingerprint,
                    self.geometry.finite_area_receipt_fingerprint,
                    "finite area",
                ),
                (
                    source.ordered_momentum_mesh_sha256,
                    self.geometry.ordered_momentum_mesh_sha256,
                    "ordered momentum mesh",
                ),
                (
                    source.source_state_sha256,
                    self.shared_functional.source_state_sha256,
                    "finite-difference source state",
                ),
                (
                    source.branch_energy_functional_fingerprint,
                    self.shared_functional.scalar_energy.fingerprint,
                    "branch energy functional",
                ),
            )
            for actual, required, label in expected:
                if actual != required:
                    raise ValueError(f"attested-source/{label} fingerprint mismatch")
            if source.area_angstrom_squared != self.geometry.area_angstrom_squared:
                raise ValueError("attested-source/geometry finite-area value mismatch")
            if source.target_density_cm2 != self.ensemble.target_density_cm2:
                raise ValueError("attested-source target density mismatch")
            if source.density_tolerance_cm2 != self.ensemble.density_tolerance_cm2:
                raise ValueError("attested-source density tolerance mismatch")
            expected_density = (
                -float(source.selected_spin_hole_count)
                / self.geometry.area_angstrom_squared
                * 1.0e16
            )
            if source.target_density_cm2 != expected_density:
                raise ValueError(
                    "finite-volume density identity n=-N_h/A*1e16 cm^-2 does not close"
                )
            if (
                source.final_replay_raw_precision != self.scf_policy.precision
                or source.stationarity_tolerance_ev
                != self.scf_policy.stationarity_tolerance_ev
                or source.branch_energy_tolerance_ev
                != self.scf_policy.branch_energy_tolerance_ev
            ):
                raise ValueError("attested-source SCF tolerances mismatch")
            if tuple(item.seed for item in source.branch_records) != self.scf_policy.seed_records:
                raise ValueError("branch table does not bind the exact SCF seed inventory")
            converged_records = tuple(
                item
                for item in source.branch_records
                if item.attested_exit_reason == "converged"
            )
            if len(converged_records) < 2:
                raise ValueError("branch comparison needs at least two attested converged rows")
            for record in source.branch_records:
                _validate_core_scf_exit(
                    reason=record.attested_exit_reason,
                    iterations=record.iterations,
                    terminal_norm_raw=record.terminal_norm_raw,
                    terminal_norm_mixed=record.terminal_norm_mixed,
                    terminal_norm_selected=record.terminal_norm_selected,
                    terminal_oda_lambda=record.terminal_oda_lambda,
                    convergence_rule=self.scf_policy.convergence_rule,
                    precision=self.scf_policy.precision,
                    max_iter=self.scf_policy.max_iter,
                    oda_stall_threshold=self.scf_policy.oda_stall_threshold,
                    max_oda_lambda=self.scf_policy.max_oda_lambda,
                    label=f"branch {record.seed.seed_label}",
                )
            minimum = min(item.canonical_energy_ev for item in converged_records)
            if source.minimum_compared_branch_energy_ev != minimum:
                raise ValueError("branch-table minimum canonical energy mismatch")

            metallicity = source.metallicity_evidence
            common_metal = (
                metallicity.source_state_sha256 == source.source_state_sha256,
                metallicity.geometry_receipt_fingerprint == self.geometry.fingerprint,
                metallicity.ordered_momentum_mesh_sha256
                == self.geometry.ordered_momentum_mesh_sha256,
                metallicity.ordered_energies_sha256 == source.ordered_energies_sha256,
                metallicity.ordered_occupations_sha256
                == source.ordered_occupations_sha256,
                metallicity.selected_spin == source.selected_spin,
                metallicity.chemical_potential_ev == source.chemical_potential_ev,
                metallicity.source_commit == source.source_commit,
                metallicity.source_artifact_sha256 == source.source_artifact_sha256,
            )
            if not all(common_metal):
                raise ValueError("metallicity evidence/source/geometry relation mismatch")
            if (
                metallicity.selected_spin_unoccupied_state_count
                != source.selected_spin_hole_count
                or metallicity.selected_spin_occupied_state_count
                + metallicity.selected_spin_unoccupied_state_count
                != self.geometry.selected_spin_state_count
            ):
                raise ValueError(
                    "metallicity occupied/unoccupied state-count closure mismatch"
                )
            for pocket in source.pocket_evidence:
                common_pocket = (
                    pocket.source_state_sha256 == source.source_state_sha256,
                    pocket.geometry_receipt_fingerprint == self.geometry.fingerprint,
                    pocket.ordered_momentum_mesh_sha256
                    == self.geometry.ordered_momentum_mesh_sha256,
                    pocket.ordered_occupations_sha256
                    == source.ordered_occupations_sha256,
                    pocket.selected_spin == source.selected_spin,
                    pocket.base_mesh_point_count == self.geometry.mesh_point_count,
                    pocket.source_commit == source.source_commit,
                    pocket.source_artifact_sha256 == source.source_artifact_sha256,
                )
                if not all(common_pocket):
                    raise ValueError("pocket evidence/source/geometry relation mismatch")

    @classmethod
    def paper_default(cls) -> "Vituri2024HalfMetalHFSpec":
        return cls()

    @property
    def missing_receipts(self) -> tuple[str, ...]:
        return tuple(
            label
            for label in (
                "geometry",
                "ensemble",
                "scf_policy",
                "shared_functional",
                "attested_source",
            )
            if getattr(self, label) is None
        )

    @property
    def receipt_set_complete(self) -> bool:
        return not self.missing_receipts

    @property
    def status(self) -> Vituri2024HFReceiptSetStatus:
        return Vituri2024HFReceiptSetStatus(
            receipt_set_complete=self.receipt_set_complete,
            missing_receipts=self.missing_receipts,
        )

    @property
    def unresolved_authorities(self) -> tuple[str, ...]:
        return self.missing_receipts + ("execution_replay",)

    @property
    def paper_direct_claim_allowed(self) -> bool:
        return False

    def require_receipt_set_complete(self) -> None:
        if self.missing_receipts:
            raise RuntimeError(
                "Vituri 2024 half-metal HF receipt set is incomplete: "
                + ", ".join(self.missing_receipts)
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


class Vituri2024AttestedHalfMetalSourceArrays(TypedDict):
    """Typed provider return annotation; the binding never requests these arrays."""

    orbitals: ComplexArray
    energies: FloatArray
    occupations: IntegerArray
    projector: ComplexArray
    fock: ComplexArray


@runtime_checkable
class Vituri2024HalfMetalHFProviderProtocol(Protocol):
    """Typed provider surface checked structurally but never executed."""

    provider_fingerprint: str
    source_commit: str
    source_artifact_sha256: str
    spec_fingerprint: str
    geometry_receipt_fingerprint: str
    ensemble_receipt_fingerprint: str
    scf_policy_receipt_fingerprint: str
    shared_functional_receipt_fingerprint: str
    attested_source_receipt_fingerprint: str
    finite_area_receipt_fingerprint: str
    interaction_receipt_fingerprint: str
    normal_order_reference_fingerprint: str
    q0_policy_fingerprint: str
    source_state_sha256: str
    scalar_energy_implementation_fingerprint: str
    fock_derivative_implementation_fingerprint: str
    finite_q_hessian_implementation_fingerprint: str
    interaction_form_factor_implementation_fingerprint: str

    def evaluate_scalar_energy(
        self,
        interaction_h: ComplexArray,
        h0: ComplexArray,
        density: ComplexArray,
    ) -> float: ...

    def evaluate_fock_derivative(self, density: ComplexArray) -> ComplexArray: ...

    def evaluate_finite_q_hessian(
        self, perturbation: ComplexArray, *, q_probe_index: int
    ) -> ComplexArray: ...

    def load_attested_source_arrays(
        self, source_artifact_sha256: str
    ) -> Vituri2024AttestedHalfMetalSourceArrays: ...


@dataclass(frozen=True, slots=True)
class Vituri2024ProviderMetadataAttestedStatus:
    """Provider-metadata attestation with execution statuses locked false."""

    metadata_status: Literal["provider_metadata_attested"] = (
        "provider_metadata_attested"
    )
    binding_execution_verified: bool = field(default=False, init=False)
    scientific_execution_verified: bool = field(default=False, init=False)
    arrays_recomputed: bool = field(default=False, init=False)
    provider_methods_executed: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.metadata_status != "provider_metadata_attested":
            raise ValueError("provider metadata status was changed")
        if any(
            (
                self.binding_execution_verified,
                self.scientific_execution_verified,
                self.arrays_recomputed,
                self.provider_methods_executed,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("provider metadata attestation cannot claim execution")


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalHFProviderBinding:
    """Provider-metadata attestation only; no provider method is called."""

    spec: Vituri2024HalfMetalHFSpec
    provider: Vituri2024HalfMetalHFProviderProtocol
    status: Vituri2024ProviderMetadataAttestedStatus = field(
        default_factory=Vituri2024ProviderMetadataAttestedStatus, init=False
    )

    def __post_init__(self) -> None:
        if type(self.spec) is not Vituri2024HalfMetalHFSpec:
            raise TypeError("provider binding requires a typed Vituri HF spec")
        self.spec.require_receipt_set_complete()
        if not isinstance(self.provider, Vituri2024HalfMetalHFProviderProtocol):
            raise TypeError("provider is missing required metadata or typed methods")
        for method_name in (
            "evaluate_scalar_energy",
            "evaluate_fock_derivative",
            "evaluate_finite_q_hessian",
            "load_attested_source_arrays",
        ):
            if not callable(getattr(self.provider, method_name, None)):
                raise TypeError(f"provider {method_name} must be callable")
        _sha256(self.provider.provider_fingerprint, "provider fingerprint")
        _commit(self.provider.source_commit, "provider source commit")
        _sha256(self.provider.source_artifact_sha256, "provider source artifact")
        assert self.spec.geometry is not None
        assert self.spec.ensemble is not None
        assert self.spec.scf_policy is not None
        assert self.spec.shared_functional is not None
        assert self.spec.attested_source is not None
        expected = (
            (
                self.provider.provider_fingerprint,
                self.spec.geometry.provider_fingerprint,
                "provider identity",
            ),
            (self.provider.spec_fingerprint, self.spec.fingerprint, "spec"),
            (
                self.provider.geometry_receipt_fingerprint,
                self.spec.geometry.fingerprint,
                "geometry receipt",
            ),
            (
                self.provider.ensemble_receipt_fingerprint,
                self.spec.ensemble.fingerprint,
                "ensemble receipt",
            ),
            (
                self.provider.scf_policy_receipt_fingerprint,
                self.spec.scf_policy.fingerprint,
                "SCF-policy receipt",
            ),
            (
                self.provider.shared_functional_receipt_fingerprint,
                self.spec.shared_functional.fingerprint,
                "shared-functional receipt",
            ),
            (
                self.provider.attested_source_receipt_fingerprint,
                self.spec.attested_source.fingerprint,
                "attested-source receipt",
            ),
            (
                self.provider.finite_area_receipt_fingerprint,
                self.spec.geometry.finite_area_receipt_fingerprint,
                "finite-area receipt",
            ),
            (
                self.provider.interaction_receipt_fingerprint,
                self.spec.shared_functional.interaction_receipt_fingerprint,
                "interaction receipt",
            ),
            (
                self.provider.normal_order_reference_fingerprint,
                self.spec.ensemble.normal_order_reference_fingerprint,
                "normal-order reference",
            ),
            (
                self.provider.q0_policy_fingerprint,
                self.spec.ensemble.q0_policy_fingerprint,
                "q0 policy",
            ),
            (
                self.provider.source_state_sha256,
                self.spec.attested_source.source_state_sha256,
                "source state",
            ),
            (
                self.provider.scalar_energy_implementation_fingerprint,
                self.spec.shared_functional.scalar_energy.implementation_fingerprint,
                "scalar-energy implementation",
            ),
            (
                self.provider.fock_derivative_implementation_fingerprint,
                self.spec.shared_functional.fock_derivative.implementation_fingerprint,
                "Fock-derivative implementation",
            ),
            (
                self.provider.finite_q_hessian_implementation_fingerprint,
                self.spec.shared_functional.finite_q_hessian.implementation_fingerprint,
                "finite-q-Hessian implementation",
            ),
            (
                self.provider.interaction_form_factor_implementation_fingerprint,
                self.spec.shared_functional.interaction_form_factor.implementation_fingerprint,
                "interaction/form-factor implementation",
            ),
        )
        for actual, required, label in expected:
            _sha256(actual, f"provider {label}")
            if actual != required:
                raise ValueError(f"provider/{label} fingerprint mismatch")
        if (
            self.provider.source_commit != self.spec.shared_functional.source_commit
            or self.provider.source_artifact_sha256
            != self.spec.shared_functional.source_artifact_sha256
        ):
            raise ValueError("provider source artifact+commit mismatch")


__all__ = [
    "VITURI2024_HALF_METAL_HF_SCOPE",
    "VITURI2024_MAIN_TEX_SHA256",
    "Vituri2024AttestedHalfMetalSourceArrays",
    "Vituri2024AttestedHalfMetalSourceReceipt",
    "Vituri2024BranchEnergyReceipt",
    "Vituri2024FiniteDifferenceEvidenceReceipt",
    "Vituri2024FunctionalComponentReceipt",
    "Vituri2024HFEnsembleReceipt",
    "Vituri2024HFGeometryReceipt",
    "Vituri2024HFReceiptSetStatus",
    "Vituri2024HFSCFPolicyReceipt",
    "Vituri2024HalfMetalHFProviderBinding",
    "Vituri2024HalfMetalHFProviderProtocol",
    "Vituri2024HalfMetalHFSpec",
    "Vituri2024HalfMetalPaperTarget",
    "Vituri2024MetallicityEvidenceReceipt",
    "Vituri2024ProviderMetadataAttestedStatus",
    "Vituri2024SCFCallbackReceipt",
    "Vituri2024SCFSeedReceipt",
    "Vituri2024SharedFunctionalReceipt",
    "Vituri2024ValleyPocketEvidenceReceipt",
]
