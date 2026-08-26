"""Independent homogeneous half-metal HF source for Vituri-2024 ABC trilayer.

This module turns the validated translational ``E/F/dF`` algebra into a
fixed-density SCF problem through the reusable :mod:`mean_field.core.hf`
engine.  It deliberately implements only a translation-preserving one-active-
band ansatz.  In particular, it cannot find the incommensurate IVC crystal.

Independent reproduction choices
--------------------------------
For a total hole density ``n_h`` and an integer number ``H_v`` of holes in each
valley, the finite area and reciprocal spacing are

``A = 2 H_v / n_h`` and ``Delta k = 2 pi / sqrt(A)``.

The odd Cartesian mesh is a finite subset of this reciprocal lattice with one
state per area ``A`` and equal weight ``1/A``.  There is no momentum wrap,
reciprocal carry, or off-grid interpolation.  The default gate distance and
UV-domain size are explicit reproduction choices and require sensitivity and
mesh convergence before a production claim.

The stored density is ``rho_ab(k)=<c_a^dagger c_b>``.  The Hamiltonian remains
an operator and is never transposed.  At zero temperature, all ``4*Nk``
one-particle levels are diagonalized and exactly ``4*Nk-2*H_v`` are occupied
by one deterministic global Aufbau operation.  The same functional supplies
SCF energy, Fock action, and ODA derivative action.

This establishes neither author-exact source data nor paper/production TDHF.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from itertools import combinations
import json
import math
from numbers import Real
from typing import Final, Literal

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockStepResult,
    HartreeFockRun,
    StateBoundPreviousDensityBuilder,
 run_hartree_fock_problem,
 select_maximum_overlap_rank_projector,
)

from .vituri2024 import (
    SM_TEX_SHA256,
    VITURI2024_PARAMETERS,
    third_lowest_active_band,
)
from .vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    make_vituri2024_finite_domain_mesh_receipt,
    make_vituri2024_translational_q0_reproduction_choice,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from .vituri2024_hf_fft import Vituri2024TranslationalHFFFTFunctional
from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from .vituri2024_interaction import Vituri2024InteractionChoiceReceipt

Array = np.ndarray
SeedMode = Literal[
    "half_metal_sz_plus",
    "half_metal_sz_minus",
    "half_metal_sx",
    "half_metal_sy",
    "valley_minus",
    "valley_plus",
    "ivc_x",
    "ivc_y",
    "random_projector",
]

VITURI2024_HF_SCF_API_VERSION: Final[str] = "vituri2024_homogeneous_hf_scf.v4"
VITURI2024_MAXIMUM_OVERLAP_AUFBAU_API_VERSION: Final[str] = (
 "vituri2024_maximum_overlap_aufbau.v2"
)
VITURI2024_HF_SCF_AUTHORITY: Final[str] = (
    "independent_homogeneous_reproduction_choice_not_author_source_"
    "incommensurate_phase_production_tdhf_or_paper_reproduction"
)
VITURI2024_TOTAL_HOLE_DENSITY_CM2: Final[float] = 1.2e12
VITURI2024_DELTA1_EV: Final[float] = 0.028
VITURI2024_GATE_DISTANCE_ANGSTROM: Final[float] = 250.0
VITURI2024_COULOMB_E2_EV_ANGSTROM: Final[float] = 14.3996454784255
VITURI2024_CM2_TO_ANGSTROM2: Final[float] = 1.0e-16
VITURI2024_DEFAULT_PRECISION: Final[float] = 1.0e-9
VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV: Final[float] = 1.0e-12
VITURI2024_FIXED_DENSITY_SCF_POLICY: Final[str] = (
    "R0_fixed_integer_rank_retain_finite_dual_gate_q0_direct_and_exchange_"
    "uniform_direct_is_constant_energy_plus_identity_fock_no_term_dropped_"
    "absolute_paper_energy_not_authorized"
)


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _finite_real(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _array_sha256(value: Array) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _readonly(value: Array, dtype: np.dtype | None = None) -> Array:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


@dataclass(frozen=True, slots=True)
class Vituri2024CartesianHFSpec:
    """Typed independent finite-volume/quadrature and SCF choices."""

    mesh_size: int
    holes_per_valley: int
    total_hole_density_cm2: float = VITURI2024_TOTAL_HOLE_DENSITY_CM2
    delta1_ev: float = VITURI2024_DELTA1_EV
    gate_distance_angstrom: float = VITURI2024_GATE_DISTANCE_ANGSTROM
    coulomb_e2_ev_angstrom: float = VITURI2024_COULOMB_E2_EV_ANGSTROM
    precision: float = VITURI2024_DEFAULT_PRECISION
    construction_mode: Literal["density_derived", "explicit_spacing"] = (
        "density_derived"
    )
    requested_delta_k_a0: float | None = None
    area_angstrom_squared: float = field(init=False)
    side_length_angstrom: float = field(init=False)
    delta_k_inverse_angstrom: float = field(init=False)
    k_cutoff_a0: float = field(init=False)
    total_holes: int = field(init=False)
    total_electrons: int = field(init=False)
    axial_k_cutoff_a0: float = field(init=False)
    corner_k_cutoff_a0: float = field(init=False)
    fingerprint: str = field(init=False)
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        size = _strict_int(self.mesh_size, "mesh_size")
        holes_per_valley = _strict_int(self.holes_per_valley, "holes_per_valley")
        if size < 3 or size % 2 != 1:
            raise ValueError("mesh_size must be an odd integer >=3")
        nk = size * size
        if holes_per_valley < 1 or holes_per_valley > nk:
            raise ValueError("holes_per_valley must satisfy 1<=H_v<=Nk")
        density_cm2 = _finite_real(
            self.total_hole_density_cm2, "total_hole_density_cm2", positive=True
        )
        delta1 = _finite_real(self.delta1_ev, "delta1_ev")
        gate = _finite_real(
            self.gate_distance_angstrom, "gate_distance_angstrom", positive=True
        )
        e2 = _finite_real(
            self.coulomb_e2_ev_angstrom,
            "coulomb_e2_ev_angstrom",
            positive=True,
        )
        precision = _finite_real(self.precision, "precision", positive=True)
        if self.construction_mode not in ("density_derived", "explicit_spacing"):
            raise ValueError("invalid Vituri Cartesian HF construction_mode")
        construction_mode = self.construction_mode
        if construction_mode == "density_derived":
            if self.requested_delta_k_a0 is not None:
                raise ValueError(
                    "density-derived construction requires requested_delta_k_a0=None"
                )
            requested_delta_k_a0 = None
        else:
            if self.requested_delta_k_a0 is None:
                raise ValueError(
                    "explicit-spacing construction requires requested_delta_k_a0"
                )
            requested_delta_k_a0 = _finite_real(
                self.requested_delta_k_a0,
                "requested_delta_k_a0",
                positive=True,
            )
        density_a2 = density_cm2 * VITURI2024_CM2_TO_ANGSTROM2
        total_holes = 2 * holes_per_valley
        area = total_holes / density_a2
        side = math.sqrt(area)
        delta_k = 2.0 * math.pi / side
        realized_delta_k_a0 = delta_k * VITURI2024_PARAMETERS.a0
        if requested_delta_k_a0 is not None and not math.isclose(
            realized_delta_k_a0,
            requested_delta_k_a0,
            rel_tol=8.0 * float(np.finfo(np.float64).eps),
            abs_tol=0.0,
        ):
            raise ValueError(
                "explicit-spacing construction is inconsistent with total density"
            )
        axial_k_cutoff_a0 = (size // 2) * realized_delta_k_a0
        corner_k_cutoff_a0 = math.sqrt(2.0) * axial_k_cutoff_a0
        total_electrons = 4 * nk - total_holes
        object.__setattr__(self, "mesh_size", size)
        object.__setattr__(self, "holes_per_valley", holes_per_valley)
        object.__setattr__(self, "total_hole_density_cm2", density_cm2)
        object.__setattr__(self, "delta1_ev", delta1)
        object.__setattr__(self, "gate_distance_angstrom", gate)
        object.__setattr__(self, "coulomb_e2_ev_angstrom", e2)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "construction_mode", construction_mode)
        object.__setattr__(
            self, "requested_delta_k_a0", requested_delta_k_a0
        )
        object.__setattr__(self, "area_angstrom_squared", area)
        object.__setattr__(self, "side_length_angstrom", side)
        object.__setattr__(self, "delta_k_inverse_angstrom", delta_k)
        object.__setattr__(self, "k_cutoff_a0", axial_k_cutoff_a0)
        object.__setattr__(self, "axial_k_cutoff_a0", axial_k_cutoff_a0)
        object.__setattr__(self, "corner_k_cutoff_a0", corner_k_cutoff_a0)
        object.__setattr__(self, "total_holes", total_holes)
        object.__setattr__(self, "total_electrons", total_electrons)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        payload = {
            "api_version": VITURI2024_HF_SCF_API_VERSION,
            "mesh_size": self.mesh_size,
            "holes_per_valley": self.holes_per_valley,
            "total_hole_density_cm2": self.total_hole_density_cm2,
            "delta1_ev": self.delta1_ev,
            "gate_distance_angstrom": self.gate_distance_angstrom,
            "coulomb_e2_ev_angstrom": self.coulomb_e2_ev_angstrom,
            "precision": self.precision,
            "construction_mode": self.construction_mode,
            "requested_delta_k_a0": self.requested_delta_k_a0,
            "area_angstrom_squared": self.area_angstrom_squared,
            "side_length_angstrom": self.side_length_angstrom,
            "delta_k_inverse_angstrom": self.delta_k_inverse_angstrom,
            "axial_k_cutoff_a0": self.axial_k_cutoff_a0,
            "corner_k_cutoff_a0": self.corner_k_cutoff_a0,
            "total_holes": self.total_holes,
            "total_electrons": self.total_electrons,
            "authority": self.authority,
            "production_ready": self.production_ready,
            "paper_reproduction_verified": self.paper_reproduction_verified,
        }
        return sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def validate_live_state(self) -> None:
        density_a2 = self.total_hole_density_cm2 * VITURI2024_CM2_TO_ANGSTROM2
        expected_area = self.total_holes / density_a2
        expected_side = math.sqrt(expected_area)
        expected_delta_k = 2.0 * math.pi / expected_side
        expected_delta_k_a0 = expected_delta_k * VITURI2024_PARAMETERS.a0
        expected_axial = (self.mesh_size // 2) * expected_delta_k_a0
        mode_consistent = (
            self.construction_mode == "density_derived"
            and self.requested_delta_k_a0 is None
        ) or (
            self.construction_mode == "explicit_spacing"
            and type(self.requested_delta_k_a0) is float
            and math.isfinite(self.requested_delta_k_a0)
            and self.requested_delta_k_a0 > 0.0
            and math.isclose(
                expected_delta_k_a0,
                self.requested_delta_k_a0,
                rel_tol=8.0 * float(np.finfo(np.float64).eps),
                abs_tol=0.0,
            )
        )
        locked = (
            type(self.mesh_size) is int,
            self.mesh_size >= 3,
            self.mesh_size % 2 == 1,
            type(self.holes_per_valley) is int,
            self.holes_per_valley >= 1,
            self.holes_per_valley <= self.mesh_size * self.mesh_size,
            self.total_holes == 2 * self.holes_per_valley,
            self.total_electrons
            == 4 * self.mesh_size * self.mesh_size - self.total_holes,
            self.area_angstrom_squared == expected_area,
            self.side_length_angstrom == expected_side,
            self.delta_k_inverse_angstrom == expected_delta_k,
            self.axial_k_cutoff_a0 == expected_axial,
            self.k_cutoff_a0 == expected_axial,
            self.corner_k_cutoff_a0 == math.sqrt(2.0) * expected_axial,
            mode_consistent,
            self.authority == VITURI2024_HF_SCF_AUTHORITY,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked) or self._current_fingerprint() != self.fingerprint:
            raise ValueError("Vituri Cartesian HF spec live state drifted")

    @property
    def nk(self) -> int:
        return self.mesh_size * self.mesh_size

    @property
    def actual_total_hole_density_cm2(self) -> float:
        return self.total_holes / self.area_angstrom_squared / VITURI2024_CM2_TO_ANGSTROM2


def make_vituri2024_cartesian_hf_spec_from_spacing(
    mesh_size: int,
    holes_per_valley: int,
    delta_k_a0: float,
    *,
    delta1_ev: float = VITURI2024_DELTA1_EV,
    gate_distance_angstrom: float = VITURI2024_GATE_DISTANCE_ANGSTROM,
    coulomb_e2_ev_angstrom: float = VITURI2024_COULOMB_E2_EV_ANGSTROM,
    precision: float = VITURI2024_DEFAULT_PRECISION,
) -> Vituri2024CartesianHFSpec:
    """Derive the realized fixed density from an explicit Cartesian spacing.

    ``delta_k_a0`` is the dimensionless nearest-neighbor grid spacing
    ``Delta k * a0``.  The returned finite-volume spec immutably records the
    explicit-spacing construction mode and exact requested value.  It grants
    no UV-convergence or paper authority.
    """

    size = _strict_int(mesh_size, "mesh_size")
    holes = _strict_int(holes_per_valley, "holes_per_valley")
    spacing_a0 = _finite_real(delta_k_a0, "delta_k_a0", positive=True)
    delta_k = spacing_a0 / VITURI2024_PARAMETERS.a0
    area = (2.0 * math.pi / delta_k) ** 2
    density_cm2 = 2 * holes / area / VITURI2024_CM2_TO_ANGSTROM2
    spec = Vituri2024CartesianHFSpec(
        mesh_size=size,
        holes_per_valley=holes,
        total_hole_density_cm2=density_cm2,
        delta1_ev=delta1_ev,
        gate_distance_angstrom=gate_distance_angstrom,
        coulomb_e2_ev_angstrom=coulomb_e2_ev_angstrom,
        precision=precision,
        construction_mode="explicit_spacing",
        requested_delta_k_a0=spacing_a0,
    )
    realized_spacing_a0 = (
        spec.delta_k_inverse_angstrom * VITURI2024_PARAMETERS.a0
    )
    if not math.isclose(
        realized_spacing_a0,
        spacing_a0,
        rel_tol=8.0 * float(np.finfo(np.float64).eps),
        abs_tol=0.0,
    ):
        raise RuntimeError("explicit-spacing Vituri spec failed to realize delta_k_a0")
    return spec


Vituri2024HomogeneousHFFunctional = (
    Vituri2024TranslationalHFFunctional | Vituri2024TranslationalHFFFTFunctional
)


@dataclass(frozen=True, slots=True)
class Vituri2024PreparedHomogeneousHF:
    spec: Vituri2024CartesianHFSpec
    ordered_mesh: Array
    integer_mesh_labels: Array
    active_band_states: Array
    active_band_energies_by_valley: Array
    h0_native: Array
    functional: Vituri2024HomogeneousHFFunctional
    fixed_density_scf_choice: "Vituri2024FixedDensitySCFChoice"
    minimum_lower_gap_ev: float
    minimum_upper_gap_ev: float
    fingerprint: str
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    source_closure_established: bool = field(default=False, init=False)
    source_stationarity_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.spec.validate_live_state()
        self.fixed_density_scf_choice.validate_live_state()
        nk = self.spec.nk
        arrays = (
            (self.ordered_mesh, np.dtype(np.float64), (nk, 2), "ordered_mesh"),
            (self.integer_mesh_labels, np.dtype(np.int64), (nk, 2), "integer_mesh_labels"),
            (
                self.active_band_states,
                np.dtype(np.complex128),
                (2, 6, nk),
                "active_band_states",
            ),
            (
                self.active_band_energies_by_valley,
                np.dtype(np.float64),
                (2, nk),
                "active_band_energies_by_valley",
            ),
            (self.h0_native, np.dtype(np.complex128), (4, 4, nk), "h0_native"),
        )
        for value, dtype, shape, label in arrays:
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"prepared {label} drifted")
        if type(self.functional) not in (
            Vituri2024TranslationalHFFunctional,
            Vituri2024TranslationalHFFFTFunctional,
        ):
            raise TypeError("prepared HF requires an approved dense or FFT functional")
        if (
            self.functional.nk != nk
            or not np.array_equal(self.functional.ordered_mesh, self.ordered_mesh)
            or not np.array_equal(
                self.functional.active_band_states, self.active_band_states
            )
            or not np.array_equal(self.functional.h0_native, self.h0_native)
        ):
            raise ValueError("prepared functional bundle binding mismatch")
        if type(self.fixed_density_scf_choice) is not Vituri2024FixedDensitySCFChoice:
            raise TypeError("prepared HF requires an exact fixed-density SCF choice")
        if not math.isfinite(self.minimum_lower_gap_ev) or self.minimum_lower_gap_ev <= 0.0:
            raise ValueError("prepared minimum lower active-band gap must be positive")
        if not math.isfinite(self.minimum_upper_gap_ev) or self.minimum_upper_gap_ev <= 0.0:
            raise ValueError("prepared minimum upper active-band gap must be positive")
        expected_mesh, expected_labels = build_vituri2024_cartesian_mesh(self.spec)
        regenerated_states = np.empty_like(self.active_band_states)
        regenerated_energies = np.empty_like(self.active_band_energies_by_valley)
        regenerated_lower_gaps = np.empty((2, nk), dtype=np.float64)
        regenerated_upper_gaps = np.empty((2, nk), dtype=np.float64)
        for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
            for momentum_index, momentum in enumerate(self.ordered_mesh):
                solution = third_lowest_active_band(
                    momentum, valley, self.spec.delta1_ev
                )
                regenerated_states[valley_index, :, momentum_index] = (
                    _largest_component_positive_gauge(solution.eigenvector)
                )
                regenerated_energies[valley_index, momentum_index] = solution.energy
                regenerated_lower_gaps[valley_index, momentum_index] = solution.lower_gap
                regenerated_upper_gaps[valley_index, momentum_index] = solution.upper_gap
        expected_h0 = np.zeros_like(self.h0_native)
        regenerated_valley_index = {
            valley: index
            for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
            expected_h0[flavor, flavor, :] = regenerated_energies[
                regenerated_valley_index[valley], :
            ]
        if (
            not np.array_equal(self.ordered_mesh, expected_mesh)
            or not np.array_equal(self.integer_mesh_labels, expected_labels)
            or not np.array_equal(
                self.active_band_energies_by_valley, regenerated_energies
            )
            or not np.array_equal(self.active_band_states, regenerated_states)
            or not np.array_equal(self.h0_native, expected_h0)
            or self.minimum_lower_gap_ev != float(np.min(regenerated_lower_gaps))
            or self.minimum_upper_gap_ev != float(np.min(regenerated_upper_gaps))
            or self.functional.mesh_receipt.area_angstrom_squared
            != self.spec.area_angstrom_squared
            or self.functional.interaction_receipt.gate_distance_angstrom
            != self.spec.gate_distance_angstrom
            or self.functional.interaction_receipt.coulomb_e2_ev_angstrom
            != self.spec.coulomb_e2_ev_angstrom
        ):
            raise ValueError("prepared homogeneous HF semantic binding mismatch")
        expected_fingerprint = _prepared_fingerprint(
            spec=self.spec,
            mesh=self.ordered_mesh,
            labels=self.integer_mesh_labels,
            states=self.active_band_states,
            energies=self.active_band_energies_by_valley,
            h0=self.h0_native,
            functional=self.functional,
            fixed_density_choice=self.fixed_density_scf_choice,
            minimum_lower_gap_ev=self.minimum_lower_gap_ev,
            minimum_upper_gap_ev=self.minimum_upper_gap_ev,
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("prepared homogeneous HF fingerprint mismatch")
        if self.authority != VITURI2024_HF_SCF_AUTHORITY or any(
            (
                self.source_closure_established,
                self.source_stationarity_established,
                self.production_ready,
                self.paper_reproduction_verified,
            )
        ):
            raise ValueError("prepared homogeneous HF authority was inflated")

    def validate_live_state(self) -> None:
        """Re-run complete source, bundle, fingerprint, and authority gates."""

        self.__post_init__()


@dataclass(slots=True)
class Vituri2024HFState:
    h0: Array
    density: Array
    hamiltonian: Array
    energies: Array
    mu: float
    precision: float
    diagnostics: dict[str, float]

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True, slots=True)
class Vituri2024FixedDensitySCFChoice:
    policy: str = VITURI2024_FIXED_DENSITY_SCF_POLICY
    normal_order_reference: str = "R=0_empty_active_electron_vacuum_representative"
    rank_is_exactly_fixed_each_aufbau_step: bool = True
    q0_direct_and_exchange_retained: bool = True
    q0_direct_dropped_or_fitted: bool = False
    absolute_paper_energy_authorized: bool = False
    paper_background_authority_established: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        self._validate_fields()
        object.__setattr__(self, "fingerprint", self._current_fingerprint())

    def _validate_fields(self) -> None:
        locked = (
            self.policy == VITURI2024_FIXED_DENSITY_SCF_POLICY,
            self.normal_order_reference
            == "R=0_empty_active_electron_vacuum_representative",
            self.rank_is_exactly_fixed_each_aufbau_step is True,
            self.q0_direct_and_exchange_retained is True,
            self.q0_direct_dropped_or_fitted is False,
            self.absolute_paper_energy_authorized is False,
            self.paper_background_authority_established is False,
        )
        if not all(locked):
            raise ValueError("fixed-density SCF choice authority was inflated")

    def _current_fingerprint(self) -> str:
        return sha256(
            json.dumps(
                {
                    "policy": self.policy,
                    "normal_order_reference": self.normal_order_reference,
                    "rank_is_exactly_fixed_each_aufbau_step": (
                        self.rank_is_exactly_fixed_each_aufbau_step
                    ),
                    "q0_direct_and_exchange_retained": (
                        self.q0_direct_and_exchange_retained
                    ),
                    "q0_direct_dropped_or_fitted": self.q0_direct_dropped_or_fitted,
                    "absolute_paper_energy_authorized": (
                        self.absolute_paper_energy_authorized
                    ),
                    "paper_background_authority_established": (
                        self.paper_background_authority_established
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def validate_live_state(self) -> None:
        self._validate_fields()
        if self._current_fingerprint() != self.fingerprint:
            raise ValueError("fixed-density SCF choice fingerprint drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024ExplicitShellBranchChoice:
 """One coordinate branch in an exhaustively declared exact shell fanout."""

 trigger_fock_sha256: str
 trigger_previous_density_sha256: str
 expected_shell_flat_indices: tuple[int, ...]
 selected_shell_flat_indices: tuple[int, ...]
 branch_index: int
 branch_count: int
 coordinate_branch_only: bool = True
 author_exact_numerical_policy: bool = False
 branch_set_fingerprint: str = field(init=False)
 fingerprint: str = field(init=False)

 def __post_init__(self) -> None:
  for value, label in (
   (self.trigger_fock_sha256, "trigger_fock_sha256"),
   (self.trigger_previous_density_sha256, "trigger_previous_density_sha256"),
  ):
   if not isinstance(value, str) or len(value) != 64 or any(
    character not in "0123456789abcdef" for character in value
   ):
    raise ValueError(f"{label} must be a lowercase SHA256 digest")
  if type(self.expected_shell_flat_indices) is not tuple or type(
   self.selected_shell_flat_indices
  ) is not tuple:
   raise TypeError("shell branch indices must be exact tuples")
  shell = tuple(
   _strict_int(value, "expected shell flat index")
   for value in self.expected_shell_flat_indices
  )
  selected = tuple(
   _strict_int(value, "selected shell flat index")
   for value in self.selected_shell_flat_indices
  )
  if shell != tuple(sorted(set(shell))) or selected != tuple(
   sorted(set(selected))
  ):
   raise ValueError("shell branch indices must be sorted and unique")
  if not shell or not selected or len(selected) >= len(shell):
   raise ValueError("explicit shell branch must be a proper nonempty subset")
  if not set(selected).issubset(shell):
   raise ValueError("selected shell branch is outside the expected shell")
  branch_index = _strict_int(self.branch_index, "branch_index")
  branch_count = _strict_int(self.branch_count, "branch_count")
  if branch_count < 1 or branch_count > 64 or not 0 <= branch_index < branch_count:
   raise ValueError("invalid branch index/count")
  expected_branch_count = math.comb(len(shell), len(selected))
  if branch_count != expected_branch_count:
   raise ValueError("branch count does not exhaust the coordinate shell fanout")
  canonical_branches = tuple(combinations(shell, len(selected)))
  if selected != tuple(canonical_branches[branch_index]):
   raise ValueError("branch index does not match the canonical shell combination")
  if self.coordinate_branch_only is not True:
   raise ValueError("only coordinate shell branches are implemented")
  if self.author_exact_numerical_policy is not False:
   raise ValueError("author-exact shell branch policy is not established")
  branch_set_payload = {
   "trigger_fock_sha256": self.trigger_fock_sha256,
   "trigger_previous_density_sha256": (
    self.trigger_previous_density_sha256
   ),
   "expected_shell_flat_indices": shell,
   "selected_rank": len(selected),
   "branch_count": branch_count,
   "coordinate_branch_only": self.coordinate_branch_only,
  }
  branch_set_fingerprint = sha256(
   json.dumps(
    branch_set_payload,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
   ).encode()
  ).hexdigest()
  payload = {
   "branch_set_fingerprint": branch_set_fingerprint,
   "trigger_fock_sha256": self.trigger_fock_sha256,
   "trigger_previous_density_sha256": (
    self.trigger_previous_density_sha256
   ),
   "expected_shell_flat_indices": shell,
   "selected_shell_flat_indices": selected,
   "branch_index": branch_index,
   "branch_count": branch_count,
   "coordinate_branch_only": self.coordinate_branch_only,
   "author_exact_numerical_policy": self.author_exact_numerical_policy,
  }
  object.__setattr__(self, "expected_shell_flat_indices", shell)
  object.__setattr__(self, "selected_shell_flat_indices", selected)
  object.__setattr__(self, "branch_set_fingerprint", branch_set_fingerprint)
  object.__setattr__(
   self,
   "fingerprint",
   sha256(
    json.dumps(
     payload,
     sort_keys=True,
     separators=(",", ":"),
     allow_nan=False,
    ).encode()
   ).hexdigest(),
  )

 def validate_live_state(self) -> None:
  expected = Vituri2024ExplicitShellBranchChoice(
   trigger_fock_sha256=self.trigger_fock_sha256,
   trigger_previous_density_sha256=self.trigger_previous_density_sha256,
   expected_shell_flat_indices=self.expected_shell_flat_indices,
   selected_shell_flat_indices=self.selected_shell_flat_indices,
   branch_index=self.branch_index,
   branch_count=self.branch_count,
   coordinate_branch_only=self.coordinate_branch_only,
   author_exact_numerical_policy=self.author_exact_numerical_policy,
  )
  if expected.fingerprint != self.fingerprint:
   raise ValueError("explicit shell branch fingerprint drifted")


def make_vituri2024_explicit_shell_branch_choices(
 *,
 trigger_fock_sha256: str,
 trigger_previous_density_sha256: str,
 shell_flat_indices: tuple[int, ...],
 selected_rank: int,
 maximum_branch_count: int = 64,
) -> tuple[Vituri2024ExplicitShellBranchChoice, ...]:
 """Enumerate all coordinate pure branches for one declared shell."""

 shell = tuple(
  sorted(
   _strict_int(value, "shell flat index") for value in shell_flat_indices
  )
 )
 if len(set(shell)) != len(shell) or not shell:
  raise ValueError("shell_flat_indices must be nonempty and unique")
 rank = _strict_int(selected_rank, "selected_rank")
 if not 0 < rank < len(shell):
  raise ValueError("selected_rank must be inside the shell")
 limit = _strict_int(maximum_branch_count, "maximum_branch_count")
 if limit < 1 or limit > 64:
  raise ValueError("maximum_branch_count must be in [1, 64]")
 count = math.comb(len(shell), rank)
 if count > limit:
  raise ValueError("coordinate shell branch count exceeds the declared limit")
 candidates = tuple(combinations(shell, rank))
 return tuple(
  Vituri2024ExplicitShellBranchChoice(
   trigger_fock_sha256=trigger_fock_sha256,
   trigger_previous_density_sha256=trigger_previous_density_sha256,
   expected_shell_flat_indices=shell,
   selected_shell_flat_indices=tuple(selected),
   branch_index=index,
   branch_count=count,
  )
  for index, selected in enumerate(candidates)
 )


@dataclass(frozen=True, slots=True)
class Vituri2024ExplicitShellBranchPath:
 """One ordered exact-trigger leaf path through unresolved Fock shells.

 This object does not certify sibling completeness. Scientific orchestration
 must enumerate every coordinate choice at every reached shell without
 postselection.
 """

 branches: tuple[Vituri2024ExplicitShellBranchChoice, ...]
 author_exact_numerical_policy: bool = False
 fingerprint: str = field(init=False)

 def __post_init__(self) -> None:
  if type(self.branches) is not tuple:
   raise TypeError("explicit shell branch path must be an exact tuple")
  if not 1 <= len(self.branches) <= 64:
   raise ValueError("explicit shell branch path length must be in [1, 64]")
  triggers: list[tuple[str, str]] = []
  branch_fingerprints: list[str] = []
  for branch in self.branches:
   if type(branch) is not Vituri2024ExplicitShellBranchChoice:
    raise TypeError("branch path entries must be typed shell choices")
   branch.validate_live_state()
   triggers.append(
    (
     branch.trigger_fock_sha256,
     branch.trigger_previous_density_sha256,
    )
   )
   branch_fingerprints.append(branch.fingerprint)
  if len(set(triggers)) != len(triggers):
   raise ValueError("explicit shell branch path repeats an exact trigger")
  if self.author_exact_numerical_policy is not False:
   raise ValueError("author-exact shell branch path is not established")
  payload = {
   "api_version": VITURI2024_MAXIMUM_OVERLAP_AUFBAU_API_VERSION,
   "ordered_branch_fingerprints": branch_fingerprints,
   "branch_path_length": len(branch_fingerprints),
   "author_exact_numerical_policy": self.author_exact_numerical_policy,
  }
  object.__setattr__(
   self,
   "fingerprint",
   sha256(
    json.dumps(
     payload,
     sort_keys=True,
     separators=(",", ":"),
     allow_nan=False,
    ).encode()
   ).hexdigest(),
  )

 def validate_live_state(self) -> None:
  expected = Vituri2024ExplicitShellBranchPath(
   branches=self.branches,
   author_exact_numerical_policy=self.author_exact_numerical_policy,
  )
  if expected.fingerprint != self.fingerprint:
   raise ValueError("explicit shell branch path fingerprint drifted")

@dataclass(frozen=True, slots=True)
class Vituri2024MaximumOverlapAufbauChoice:
 """Independent ODA-compatible policy for a set-valued Fock ground state.

 Only an exactly equal-energy shell may use maximum-overlap continuation.
 A tied overlap cutoff remains unresolved unless one exact-triggered member of
 an exhaustively declared coordinate branch fanout is supplied. Stable orbital
 ordering is never used as a physical branch selector.
 """

 energy_shell_tolerance_ev: float = VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV
 overlap_gap_tolerance: float = 1.0e-12
 previous_density_role: str = "current_oda_mixed_density"
 unresolved_overlap_policy: str = "reject_no_arbitrary_branch"
 explicit_branch: Vituri2024ExplicitShellBranchChoice | None = None
 exact_energy_tie_required: bool = True
 author_exact_numerical_policy: bool = False
 explicit_branch_path: Vituri2024ExplicitShellBranchPath | None = None
 fingerprint: str = field(init=False)

 def __post_init__(self) -> None:
  energy_tolerance = _finite_real(
   self.energy_shell_tolerance_ev,
   "energy_shell_tolerance_ev",
   positive=True,
  )
  overlap_tolerance = _finite_real(
   self.overlap_gap_tolerance,
   "overlap_gap_tolerance",
   positive=True,
  )
  if energy_tolerance != VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV:
   raise ValueError("energy shell tolerance is locked to the baseline Aufbau gate")
  if self.previous_density_role != "current_oda_mixed_density":
   raise ValueError("unsupported previous-density role")
  if self.explicit_branch is not None and self.explicit_branch_path is not None:
   raise ValueError("explicit_branch and explicit_branch_path are mutually exclusive")
  branch_path_fingerprint: str | None = None
  if self.explicit_branch is None and self.explicit_branch_path is None:
   if self.unresolved_overlap_policy != "reject_no_arbitrary_branch":
    raise ValueError("unsupported unresolved-overlap policy")
   branch_fingerprint = None
  elif self.explicit_branch is not None:
   if type(self.explicit_branch) is not Vituri2024ExplicitShellBranchChoice:
    raise TypeError("explicit_branch must be a typed shell branch choice")
   self.explicit_branch.validate_live_state()
   if self.unresolved_overlap_policy != "exact_triggered_coordinate_branch":
    raise ValueError("explicit branch requires exact-triggered branch policy")
   branch_fingerprint = self.explicit_branch.fingerprint
  else:
   if type(self.explicit_branch_path) is not Vituri2024ExplicitShellBranchPath:
    raise TypeError("explicit_branch_path must be a typed branch path")
   self.explicit_branch_path.validate_live_state()
   if self.unresolved_overlap_policy != "exact_triggered_coordinate_branch_path":
    raise ValueError("explicit branch path requires path policy")
   branch_fingerprint = None
   branch_path_fingerprint = self.explicit_branch_path.fingerprint
  if self.exact_energy_tie_required is not True:
   raise ValueError("only exact energy ties are authorized")
  if self.author_exact_numerical_policy is not False:
   raise ValueError("author-exact tie policy is not established")
  payload = {
   "api_version": VITURI2024_MAXIMUM_OVERLAP_AUFBAU_API_VERSION,
   "energy_shell_tolerance_ev": energy_tolerance,
   "overlap_gap_tolerance": overlap_tolerance,
   "previous_density_role": self.previous_density_role,
   "unresolved_overlap_policy": self.unresolved_overlap_policy,
   "explicit_branch_fingerprint": branch_fingerprint,
   "explicit_branch_path_fingerprint": branch_path_fingerprint,
   "exact_energy_tie_required": self.exact_energy_tie_required,
   "author_exact_numerical_policy": self.author_exact_numerical_policy,
  }
  object.__setattr__(
   self,
   "fingerprint",
   sha256(
    json.dumps(
     payload,
     sort_keys=True,
     separators=(",", ":"),
     allow_nan=False,
    ).encode()
   ).hexdigest(),
  )

 def validate_live_state(self) -> None:
  expected = Vituri2024MaximumOverlapAufbauChoice(
   energy_shell_tolerance_ev=self.energy_shell_tolerance_ev,
   overlap_gap_tolerance=self.overlap_gap_tolerance,
   previous_density_role=self.previous_density_role,
   unresolved_overlap_policy=self.unresolved_overlap_policy,
   explicit_branch=self.explicit_branch,
   explicit_branch_path=self.explicit_branch_path,
   exact_energy_tie_required=self.exact_energy_tie_required,
   author_exact_numerical_policy=self.author_exact_numerical_policy,
  )
  if expected.fingerprint != self.fingerprint:
   raise ValueError("maximum-overlap Aufbau choice fingerprint drifted")


def _explicit_shell_branch_sequence(
 choice: Vituri2024MaximumOverlapAufbauChoice,
) -> tuple[Vituri2024ExplicitShellBranchChoice, ...]:
 if choice.explicit_branch is not None:
  return (choice.explicit_branch,)
 if choice.explicit_branch_path is not None:
  return choice.explicit_branch_path.branches
 return ()


@dataclass(frozen=True, slots=True)
class _Vituri2024PendingExplicitBranchUse:
 generation: int
 branch_fingerprint: str
 trigger_fock_sha256: str
 trigger_previous_density_sha256: str
 update_object_id: int
 update_density_sha256: str


@dataclass(slots=True)
class _Vituri2024ExplicitBranchAuditState:
 next_generation: int = 0
 last_branch_iteration: int = 0
 pending: _Vituri2024PendingExplicitBranchUse | None = None
 consumed: list[tuple[int, int, int, str]] = field(default_factory=list)
 terminal_rejection: bool = False


@dataclass(frozen=True, slots=True)
class Vituri2024HalfMetalDiagnostics:
    electron_count: float
    total_holes: float
    holes_by_flavor: tuple[float, float, float, float]
    holes_by_valley: tuple[float, float]
    spin_hole_eigenvalues: tuple[float, float]
    spin_polarization_norm: float
    per_valley_spin_purity_residuals: tuple[float, float]
    common_spin_axis_residual: float
    intervalley_coherence_frobenius: float
    intervalley_coherence_max_per_k: float
    valley_balance_residual: float
    projector_idempotency_residual: float
    commutator_residual_ev: float
    final_raw_norm: float
    finite_size_fermi_gap_ev: float
    fully_spin_polarized: bool
    common_axis_spin_polarized: bool
    intervalley_incoherent: bool
    valley_balanced: bool
    stationary: bool
    valid_homogeneous_half_metal_candidate: bool
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)


@dataclass(frozen=True, slots=True)
class Vituri2024HFSeedRun:
    seed_mode: SeedMode
    seed: int
    run: HartreeFockRun
    diagnostics: Vituri2024HalfMetalDiagnostics
    final_independent_model_energy_ev: float
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    @property
    def final_energy_ev(self) -> float:
        """Compatibility alias; this is not an absolute paper energy."""

        return self.final_independent_model_energy_ev


@dataclass(frozen=True, slots=True)
class Vituri2024InitialFockBoundaryScanChoice:
    """Branch-conditioned finite-volume Aufbau-boundary scan choices.

    This is deliberately restricted to the deterministic ``sz+`` branch used
    by the independent homogeneous candidate. A closed boundary here is not a
    geometric shell and has no model-independent or paper-source authority.
    """

    mesh_size: int
    target_holes_per_valley: int
    scan_min_holes_per_valley: int
    scan_max_holes_per_valley: int
    total_hole_density_cm2: float = VITURI2024_TOTAL_HOLE_DENSITY_CM2
    delta1_ev: float = VITURI2024_DELTA1_EV
    gate_distance_angstrom: float = VITURI2024_GATE_DISTANCE_ANGSTROM
    coulomb_e2_ev_angstrom: float = VITURI2024_COULOMB_E2_EV_ANGSTROM
    precision: float = VITURI2024_DEFAULT_PRECISION
    degeneracy_tolerance_ev: float = VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV
    minimum_gap_to_eigensolver_residual_ratio: float = 1.0e6
    allow_fallback: bool = False
    target_holes_policy: Literal[
        "fixed_regulator", "nearest_physical_cutoff"
    ] = "fixed_regulator"
    target_axial_cutoff_a0: float | None = None
    initial_branch_mode: Literal["half_metal_sz_plus"] = field(
        default="half_metal_sz_plus", init=False
    )
    initial_branch_rng_used: bool = field(default=False, init=False)
    fingerprint: str = field(init=False)
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    finite_domain_cutoff_converged: bool = field(default=False, init=False)
    global_ground_state_proved: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        mesh_size = _strict_int(self.mesh_size, "mesh_size")
        target = _strict_int(
            self.target_holes_per_valley, "target_holes_per_valley"
        )
        minimum = _strict_int(
            self.scan_min_holes_per_valley, "scan_min_holes_per_valley"
        )
        maximum = _strict_int(
            self.scan_max_holes_per_valley, "scan_max_holes_per_valley"
        )
        if mesh_size < 3 or mesh_size % 2 != 1:
            raise ValueError("mesh_size must be an odd integer >=3")
        if minimum < 1 or minimum > target or target > maximum:
            raise ValueError("boundary scan must satisfy 1<=min<=target<=max")
        if maximum > mesh_size * mesh_size:
            raise ValueError("boundary scan exceeds the declared branch capacity")
        if type(self.allow_fallback) is not bool:
            raise TypeError("allow_fallback must be native bool")
        if self.target_holes_policy not in (
            "fixed_regulator",
            "nearest_physical_cutoff",
        ):
            raise ValueError("unsupported target_holes_policy")
        object.__setattr__(self, "mesh_size", mesh_size)
        object.__setattr__(self, "target_holes_per_valley", target)
        object.__setattr__(self, "scan_min_holes_per_valley", minimum)
        object.__setattr__(self, "scan_max_holes_per_valley", maximum)
        for name in (
            "total_hole_density_cm2",
            "gate_distance_angstrom",
            "coulomb_e2_ev_angstrom",
            "precision",
            "degeneracy_tolerance_ev",
            "minimum_gap_to_eigensolver_residual_ratio",
        ):
            object.__setattr__(
                self, name, _finite_real(getattr(self, name), name, positive=True)
            )
        object.__setattr__(
            self, "delta1_ev", _finite_real(self.delta1_ev, "delta1_ev")
        )
        if self.minimum_gap_to_eigensolver_residual_ratio < 1.0:
            raise ValueError(
                "minimum_gap_to_eigensolver_residual_ratio must be >=1"
            )
        target_spec = Vituri2024CartesianHFSpec(
            mesh_size=mesh_size,
            holes_per_valley=target,
            total_hole_density_cm2=self.total_hole_density_cm2,
            delta1_ev=self.delta1_ev,
            gate_distance_angstrom=self.gate_distance_angstrom,
            coulomb_e2_ev_angstrom=self.coulomb_e2_ev_angstrom,
            precision=self.precision,
        )
        if self.target_holes_policy == "fixed_regulator":
            if self.target_axial_cutoff_a0 is not None:
                raise ValueError(
                    "fixed_regulator derives target cutoff from the fixed target Hv"
                )
            target_cutoff = target_spec.axial_k_cutoff_a0
        else:
            target_cutoff = _finite_real(
                self.target_axial_cutoff_a0,
                "target_axial_cutoff_a0",
                positive=True,
            )
            density_a2 = (
                self.total_hole_density_cm2 * VITURI2024_CM2_TO_ANGSTROM2
            )
            half = mesh_size // 2
            continuous_target_holes = 0.5 * density_a2 * (
                2.0
                * math.pi
                * VITURI2024_PARAMETERS.a0
                * half
                / target_cutoff
            ) ** 2
            nearest_target_holes = int(math.floor(continuous_target_holes + 0.5))
            if target != nearest_target_holes:
                raise ValueError(
                    "target_holes_per_valley is not the nearest integer realization "
                    "of target_axial_cutoff_a0"
                )
        object.__setattr__(self, "target_axial_cutoff_a0", target_cutoff)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())

    def _current_fingerprint(self) -> str:
        payload = {
            "api_version": VITURI2024_HF_SCF_API_VERSION,
            "mesh_size": self.mesh_size,
            "target_holes_per_valley": self.target_holes_per_valley,
            "scan_min_holes_per_valley": self.scan_min_holes_per_valley,
            "scan_max_holes_per_valley": self.scan_max_holes_per_valley,
            "target_axial_cutoff_a0": self.target_axial_cutoff_a0,
            "total_hole_density_cm2": self.total_hole_density_cm2,
            "delta1_ev": self.delta1_ev,
            "gate_distance_angstrom": self.gate_distance_angstrom,
            "coulomb_e2_ev_angstrom": self.coulomb_e2_ev_angstrom,
            "precision": self.precision,
            "degeneracy_tolerance_ev": self.degeneracy_tolerance_ev,
            "minimum_gap_to_eigensolver_residual_ratio": (
                self.minimum_gap_to_eigensolver_residual_ratio
            ),
            "allow_fallback": self.allow_fallback,
            "target_holes_policy": self.target_holes_policy,
            "initial_branch_mode": self.initial_branch_mode,
            "initial_branch_rng_used": self.initial_branch_rng_used,
            "authority": self.authority,
        }
        return sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def validate_live_state(self) -> None:
        if (
            self.authority != VITURI2024_HF_SCF_AUTHORITY
            or self.initial_branch_mode != "half_metal_sz_plus"
            or self.initial_branch_rng_used is not False
            or self._current_fingerprint() != self.fingerprint
            or any(
                (
                    self.finite_domain_cutoff_converged,
                    self.global_ground_state_proved,
                    self.production_ready,
                    self.paper_reproduction_verified,
                )
            )
        ):
            raise ValueError("initial-Fock boundary scan choice drifted")

@dataclass(frozen=True, slots=True)
class Vituri2024GlobalAufbauBoundaryAnalysis:
    fock_shape: tuple[int, int, int]
    total_occupied: int
    boundary_gap_ev: float
    fock_energy_scale_ev: float
    fock_hermiticity_residual_ev: float
    maximum_eigensolver_residual_ev: float
    effective_eigensolver_residual_floor_ev: float
    gap_to_effective_eigensolver_residual_ratio: float
    shell_multiplicity: int
    occupied_in_boundary_shell: int
    degeneracy_tolerance_ev: float
    minimum_gap_to_eigensolver_residual_ratio: float
    closed_global_aufbau_boundary: bool

    def __post_init__(self) -> None:
        shape = tuple(_strict_int(value, "fock_shape") for value in self.fock_shape)
        occupied = _strict_int(self.total_occupied, "total_occupied")
        multiplicity = _strict_int(self.shell_multiplicity, "shell_multiplicity")
        occupied_in_shell = _strict_int(
            self.occupied_in_boundary_shell, "occupied_in_boundary_shell"
        )
        if (
            len(shape) != 3
            or shape[0] != shape[1]
            or shape[0] < 1
            or shape[2] < 1
            or not 1 <= occupied < shape[0] * shape[2]
        ):
            raise ValueError("invalid global Aufbau analysis shape/rank")
        if multiplicity < 1 or not 1 <= occupied_in_shell <= multiplicity:
            raise ValueError("invalid global Aufbau boundary-shell counts")
        if type(self.closed_global_aufbau_boundary) is not bool:
            raise TypeError("closed global Aufbau boundary flag must be native bool")
        object.__setattr__(self, "fock_shape", shape)
        object.__setattr__(self, "total_occupied", occupied)
        object.__setattr__(self, "shell_multiplicity", multiplicity)
        object.__setattr__(self, "occupied_in_boundary_shell", occupied_in_shell)
        nonnegative = (
            "boundary_gap_ev",
            "fock_hermiticity_residual_ev",
            "maximum_eigensolver_residual_ev",
            "gap_to_effective_eigensolver_residual_ratio",
        )
        for name in nonnegative:
            value = _finite_real(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        for name in (
            "fock_energy_scale_ev",
            "effective_eigensolver_residual_floor_ev",
            "degeneracy_tolerance_ev",
            "minimum_gap_to_eigensolver_residual_ratio",
        ):
            object.__setattr__(
                self, name, _finite_real(getattr(self, name), name, positive=True)
            )
        if self.minimum_gap_to_eigensolver_residual_ratio < 1.0:
            raise ValueError(
                "minimum_gap_to_eigensolver_residual_ratio must be >=1"
            )
        expected_floor = max(
            self.maximum_eigensolver_residual_ev,
            float(np.finfo(np.float64).eps) * self.fock_energy_scale_ev,
        )
        expected_ratio = self.boundary_gap_ev / expected_floor
        expected_closed = bool(
            self.boundary_gap_ev > self.degeneracy_tolerance_ev
            and self.boundary_gap_ev
            > self.minimum_gap_to_eigensolver_residual_ratio * expected_floor
            and self.occupied_in_boundary_shell == self.shell_multiplicity
        )
        if (
            self.fock_energy_scale_ev < 1.0
            or self.effective_eigensolver_residual_floor_ev != expected_floor
            or self.gap_to_effective_eigensolver_residual_ratio != expected_ratio
            or self.closed_global_aufbau_boundary is not expected_closed
        ):
            raise ValueError("global Aufbau boundary analysis drifted")


def analyze_vituri2024_global_aufbau_boundary(
    fock: Array,
    *,
    total_occupied: int,
    degeneracy_tolerance_ev: float = VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV,
    minimum_gap_to_eigensolver_residual_ratio: float = 1.0e6,
) -> Vituri2024GlobalAufbauBoundaryAnalysis:
    """Analyze one global occupied/unoccupied boundary from block Fock matrices."""

    matrix = np.asarray(fock, dtype=np.complex128)
    if matrix.ndim != 3 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Fock input must have shape (n,n,Nk)")
    occupied = _strict_int(total_occupied, "total_occupied")
    if not 1 <= occupied < matrix.shape[0] * matrix.shape[2]:
        raise ValueError("total_occupied must leave nonempty occupied/unoccupied sets")
    tolerance = _finite_real(
        degeneracy_tolerance_ev, "degeneracy_tolerance_ev", positive=True
    )
    minimum_ratio = _finite_real(
        minimum_gap_to_eigensolver_residual_ratio,
        "minimum_gap_to_eigensolver_residual_ratio",
        positive=True,
    )
    if minimum_ratio < 1.0:
        raise ValueError("minimum_gap_to_eigensolver_residual_ratio must be >=1")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Fock input must be finite")
    fock_scale = max(1.0, _max_abs(matrix))
    hermiticity = _max_abs(matrix - matrix.swapaxes(0, 1).conj())
    if hermiticity > 1.0e-12 * fock_scale:
        raise ValueError("Fock input is materially non-Hermitian")
    eigenvalues = np.empty((matrix.shape[0], matrix.shape[2]), dtype=np.float64)
    maximum_residual = 0.0
    for momentum in range(matrix.shape[2]):
        values, vectors = np.linalg.eigh(matrix[:, :, momentum])
        eigenvalues[:, momentum] = values
        residual_vectors = matrix[:, :, momentum] @ vectors - vectors * values
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.linalg.norm(residual_vectors, axis=0))),
        )
    sorted_energies = np.sort(eigenvalues.reshape(-1, order="C"))
    lower = float(sorted_energies[occupied - 1])
    upper = float(sorted_energies[occupied])
    gap = upper - lower
    shell_multiplicity = int(
        np.count_nonzero(np.abs(sorted_energies - lower) <= tolerance)
    )
    strictly_below = int(np.count_nonzero(sorted_energies < lower - tolerance))
    occupied_in_shell = occupied - strictly_below
    effective_floor = max(
        maximum_residual,
        float(np.finfo(np.float64).eps) * fock_scale,
    )
    ratio = gap / effective_floor
    closed = bool(
        gap > tolerance
        and gap > minimum_ratio * effective_floor
        and occupied_in_shell == shell_multiplicity
    )
    return Vituri2024GlobalAufbauBoundaryAnalysis(
        fock_shape=matrix.shape,
        total_occupied=occupied,
        boundary_gap_ev=gap,
        fock_energy_scale_ev=fock_scale,
        fock_hermiticity_residual_ev=hermiticity,
        maximum_eigensolver_residual_ev=maximum_residual,
        effective_eigensolver_residual_floor_ev=effective_floor,
        gap_to_effective_eigensolver_residual_ratio=ratio,
        shell_multiplicity=shell_multiplicity,
        occupied_in_boundary_shell=occupied_in_shell,
        degeneracy_tolerance_ev=tolerance,
        minimum_gap_to_eigensolver_residual_ratio=minimum_ratio,
        closed_global_aufbau_boundary=closed,
    )

@dataclass(frozen=True, slots=True)
class Vituri2024InitialFockBoundaryRecord:
    holes_per_valley: int
    analysis: Vituri2024GlobalAufbauBoundaryAnalysis
    area_angstrom_squared: float
    delta_k_inverse_angstrom: float
    axial_cutoff_a0: float
    spec_fingerprint: str
    initial_density_sha256: str
    initial_fock_sha256: str
    initial_branch_mode: Literal["half_metal_sz_plus"] = field(
        default="half_metal_sz_plus", init=False
    )
    initial_branch_rng_used: bool = field(default=False, init=False)
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    scf_stationarity_established: bool = field(default=False, init=False)
    finite_domain_cutoff_converged: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        holes = _strict_int(self.holes_per_valley, "holes_per_valley")
        if holes < 1 or type(self.analysis) is not Vituri2024GlobalAufbauBoundaryAnalysis:
            raise ValueError("invalid initial-Fock boundary record")
        object.__setattr__(self, "holes_per_valley", holes)
        for name in (
            "area_angstrom_squared",
            "delta_k_inverse_angstrom",
            "axial_cutoff_a0",
        ):
            object.__setattr__(
                self, name, _finite_real(getattr(self, name), name, positive=True)
            )
        for name in (
            "spec_fingerprint",
            "initial_density_sha256",
            "initial_fock_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            self.initial_branch_mode != "half_metal_sz_plus"
            or self.initial_branch_rng_used is not False
            or self.authority != VITURI2024_HF_SCF_AUTHORITY
            or any(
                (
                    self.scf_stationarity_established,
                    self.finite_domain_cutoff_converged,
                    self.production_ready,
                    self.paper_reproduction_verified,
                )
            )
        ):
            raise ValueError("initial-Fock boundary record authority was inflated")

@dataclass(frozen=True, slots=True)
class Vituri2024InitialFockBoundarySelection:
    choice: Vituri2024InitialFockBoundaryScanChoice
    records: tuple[Vituri2024InitialFockBoundaryRecord, ...]
    selected: Vituri2024InitialFockBoundaryRecord
    target_record: Vituri2024InitialFockBoundaryRecord
    target_admitted: bool
    fallback_used: bool
    selection_key: tuple[float, float, float, int]
    binding_fingerprint: str
    authority: str = field(default=VITURI2024_HF_SCF_AUTHORITY, init=False)
    branch_conditioned_regulator_admission_only: bool = field(
        default=True, init=False
    )
    scf_stationarity_established: bool = field(default=False, init=False)
    finite_domain_cutoff_converged: bool = field(default=False, init=False)
    global_ground_state_proved: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.choice) is not Vituri2024InitialFockBoundaryScanChoice:
            raise TypeError("selection choice has the wrong exact type")
        if type(self.records) is not tuple or any(
            type(record) is not Vituri2024InitialFockBoundaryRecord
            for record in self.records
        ):
            raise TypeError("selection records must be an exact tuple of records")
        if (
            type(self.selected) is not Vituri2024InitialFockBoundaryRecord
            or type(self.target_record) is not Vituri2024InitialFockBoundaryRecord
        ):
            raise TypeError("selection selected/target record has the wrong exact type")
        if type(self.target_admitted) is not bool or type(self.fallback_used) is not bool:
            raise TypeError("selection target/fallback flags must be native bool")
        self.choice.validate_live_state()
        expected_holes = tuple(
            range(
                self.choice.scan_min_holes_per_valley,
                self.choice.scan_max_holes_per_valley + 1,
            )
        )
        if tuple(record.holes_per_valley for record in self.records) != expected_holes:
            raise ValueError("initial-Fock boundary record inventory drifted")
        for record in self.records:
            spec = Vituri2024CartesianHFSpec(
                mesh_size=self.choice.mesh_size,
                holes_per_valley=record.holes_per_valley,
                total_hole_density_cm2=self.choice.total_hole_density_cm2,
                delta1_ev=self.choice.delta1_ev,
                gate_distance_angstrom=self.choice.gate_distance_angstrom,
                coulomb_e2_ev_angstrom=self.choice.coulomb_e2_ev_angstrom,
                precision=self.choice.precision,
            )
            if (
                record.initial_branch_mode != self.choice.initial_branch_mode
                or record.initial_branch_rng_used is not False
                or record.spec_fingerprint != spec.fingerprint
                or record.area_angstrom_squared != spec.area_angstrom_squared
                or record.delta_k_inverse_angstrom
                != spec.delta_k_inverse_angstrom
                or record.axial_cutoff_a0 != spec.axial_k_cutoff_a0
                or record.analysis.fock_shape != (4, 4, spec.nk)
                or record.analysis.total_occupied != spec.total_electrons
                or record.analysis.degeneracy_tolerance_ev
                != self.choice.degeneracy_tolerance_ev
                or record.analysis.minimum_gap_to_eigensolver_residual_ratio
                != self.choice.minimum_gap_to_eigensolver_residual_ratio
            ):
                raise ValueError("initial-Fock boundary record semantic drifted")
        target_record = next(
            record
            for record in self.records
            if record.holes_per_valley == self.choice.target_holes_per_valley
        )
        target_admitted = target_record.analysis.closed_global_aufbau_boundary
        if self.target_record != target_record or self.target_admitted is not target_admitted:
            raise ValueError("initial-Fock target admission receipt drifted")
        candidates = tuple(
            record
            for record in self.records
            if record.analysis.closed_global_aufbau_boundary
        )
        if not candidates:
            raise ValueError("initial-Fock boundary scan has no admitted candidate")
        fallback_used = self.selected.holes_per_valley != self.choice.target_holes_per_valley
        if self.fallback_used is not fallback_used:
            raise ValueError("initial-Fock fallback receipt drifted")
        if fallback_used and not self.choice.allow_fallback:
            raise ValueError("target boundary is not admitted and fallback is disabled")
        expected = min(
            candidates,
            key=lambda record: _initial_fock_selection_key(self.choice, record),
        )
        if self.selected != expected:
            raise ValueError("initial-Fock boundary selection drifted")
        if self.selection_key != _initial_fock_selection_key(self.choice, expected):
            raise ValueError("initial-Fock boundary selection key drifted")
        if self.binding_fingerprint != _initial_fock_binding_fingerprint(
            self.choice, self.records
        ):
            raise ValueError("initial-Fock boundary binding fingerprint drifted")
        if (
            self.authority != VITURI2024_HF_SCF_AUTHORITY
            or self.branch_conditioned_regulator_admission_only is not True
            or any(
                (
                    self.scf_stationarity_established,
                    self.finite_domain_cutoff_converged,
                    self.global_ground_state_proved,
                    self.production_ready,
                    self.paper_reproduction_verified,
                )
            )
        ):
            raise ValueError("initial-Fock boundary selection authority was inflated")

    def validate_live_state(self, *, recompute_branch_bindings: bool = False) -> None:
        """Validate receipts, optionally rebuilding every declared branch Fock."""

        if type(recompute_branch_bindings) is not bool:
            raise TypeError("recompute_branch_bindings must be native bool")
        self.__post_init__()
        if not recompute_branch_bindings:
            return
        for record in self.records:
            spec = Vituri2024CartesianHFSpec(
                mesh_size=self.choice.mesh_size,
                holes_per_valley=record.holes_per_valley,
                total_hole_density_cm2=self.choice.total_hole_density_cm2,
                delta1_ev=self.choice.delta1_ev,
                gate_distance_angstrom=self.choice.gate_distance_angstrom,
                coulomb_e2_ev_angstrom=self.choice.coulomb_e2_ev_angstrom,
                precision=self.choice.precision,
            )
            prepared = prepare_vituri2024_homogeneous_hf(spec)
            problem = make_vituri2024_hf_problem(prepared)
            state = make_vituri2024_hf_state(prepared)
            problem.initializer(state, init_mode=self.choice.initial_branch_mode, seed=0)
            fock = state.h0 + problem.kernel.interaction_builder(state.density)
            analysis = analyze_vituri2024_global_aufbau_boundary(
                fock,
                total_occupied=spec.total_electrons,
                degeneracy_tolerance_ev=self.choice.degeneracy_tolerance_ev,
                minimum_gap_to_eigensolver_residual_ratio=(
                    self.choice.minimum_gap_to_eigensolver_residual_ratio
                ),
            )
            if (
                record.initial_density_sha256 != _array_sha256(state.density)
                or record.initial_fock_sha256 != _array_sha256(fock)
                or record.analysis != analysis
            ):
                raise ValueError("initial-Fock branch/Fock live binding drifted")


def _initial_fock_binding_fingerprint(
    choice: Vituri2024InitialFockBoundaryScanChoice,
    records: tuple[Vituri2024InitialFockBoundaryRecord, ...],
) -> str:
    payload = {
        "api_version": VITURI2024_HF_SCF_API_VERSION,
        "choice": choice.fingerprint,
        "records": [asdict(record) for record in records],
    }
    return sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _initial_fock_selection_key(
    choice: Vituri2024InitialFockBoundaryScanChoice,
    record: Vituri2024InitialFockBoundaryRecord,
) -> tuple[float, float, float, int]:
    return (
        float(abs(record.holes_per_valley - choice.target_holes_per_valley)),
        abs(record.axial_cutoff_a0 - choice.target_axial_cutoff_a0),
        -record.analysis.boundary_gap_ev,
        record.holes_per_valley,
    )

def build_vituri2024_cartesian_mesh(spec: Vituri2024CartesianHFSpec) -> tuple[Array, Array]:
    if type(spec) is not Vituri2024CartesianHFSpec:
        raise TypeError("spec must be Vituri2024CartesianHFSpec")
    half = spec.mesh_size // 2
    labels = np.asarray(
        [(ix, iy) for iy in range(-half, half + 1) for ix in range(-half, half + 1)],
        dtype=np.int64,
    )
    mesh = np.asarray(labels, dtype=np.float64) * spec.delta_k_inverse_angstrom
    return _readonly(mesh, np.dtype(np.float64)), _readonly(labels, np.dtype(np.int64))


def _largest_component_positive_gauge(vector: Array) -> Array:
    state = np.asarray(vector, dtype=np.complex128).copy()
    index = int(np.argmax(np.abs(state)))
    amplitude = state[index]
    if amplitude == 0.0:
        raise ValueError("active-band state has no nonzero component")
    state *= np.exp(-1j * np.angle(amplitude))
    if state[index].real < 0.0:
        state *= -1.0
    state[index] = complex(abs(state[index]), 0.0)
    return state


def _interaction_choice(spec: Vituri2024CartesianHFSpec) -> Vituri2024InteractionChoiceReceipt:
    provider_payload = {
        "api_version": VITURI2024_HF_SCF_API_VERSION,
        "gate_distance_angstrom": spec.gate_distance_angstrom,
        "coulomb_e2_ev_angstrom": spec.coulomb_e2_ev_angstrom,
        "q0_evaluation": "analytic_kernel_limit_only",
        "scope": "independent_homogeneous_HF_reproduction_choice",
    }
    provider_sha = sha256(
        json.dumps(
            provider_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return Vituri2024InteractionChoiceReceipt(
        gate_distance_angstrom=spec.gate_distance_angstrom,
        coulomb_e2_ev_angstrom=spec.coulomb_e2_ev_angstrom,
        q0_evaluation="analytic_kernel_limit_only",
        provider_sha256=provider_sha,
        source_sha256=SM_TEX_SHA256,
        authority_kind="reproduction_choice",
        source_text=(
            "Paper-direct epsilon=8 and qTF=0.04/a0; gate distance, analytic q0 "
            "kernel evaluation, and finite-domain background handling are explicit "
            "independent reproduction choices."
        ),
    )


def _prepared_fingerprint(
    *,
    spec: Vituri2024CartesianHFSpec,
    mesh: Array,
    labels: Array,
    states: Array,
    energies: Array,
    h0: Array,
    functional: Vituri2024HomogeneousHFFunctional,
    fixed_density_choice: Vituri2024FixedDensitySCFChoice,
    minimum_lower_gap_ev: float,
    minimum_upper_gap_ev: float,
) -> str:
    return sha256(
        json.dumps(
            {
                "spec": spec.fingerprint,
                "mesh": _array_sha256(mesh),
                "labels": _array_sha256(labels),
                "states": _array_sha256(states),
                "energies": _array_sha256(energies),
                "h0": _array_sha256(h0),
                "functional": functional.fingerprint,
                "fixed_density_scf_choice": fixed_density_choice.fingerprint,
                "minimum_lower_gap_ev": minimum_lower_gap_ev,
                "minimum_upper_gap_ev": minimum_upper_gap_ev,
                "authority": VITURI2024_HF_SCF_AUTHORITY,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _prepare_vituri2024_homogeneous_hf(
    spec: Vituri2024CartesianHFSpec,
    *,
    fft_workers: int | None,
) -> Vituri2024PreparedHomogeneousHF:
    """Build common states and select only the interaction backend."""

    if type(spec) is not Vituri2024CartesianHFSpec:
        raise TypeError("spec must be Vituri2024CartesianHFSpec")
    mesh, labels = build_vituri2024_cartesian_mesh(spec)
    states = np.empty((2, 6, spec.nk), dtype=np.complex128)
    energies = np.empty((2, spec.nk), dtype=np.float64)
    lower_gaps = np.empty((2, spec.nk), dtype=np.float64)
    upper_gaps = np.empty((2, spec.nk), dtype=np.float64)
    for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
        for momentum_index, momentum in enumerate(mesh):
            solution = third_lowest_active_band(momentum, valley, spec.delta1_ev)
            states[valley_index, :, momentum_index] = _largest_component_positive_gauge(
                solution.eigenvector
            )
            energies[valley_index, momentum_index] = solution.energy
            lower_gaps[valley_index, momentum_index] = solution.lower_gap
            upper_gaps[valley_index, momentum_index] = solution.upper_gap
    h0 = np.zeros((4, 4, spec.nk), dtype=np.complex128)
    valley_index = {
        valley: index
        for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
    }
    for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
        h0[flavor, flavor, :] = energies[valley_index[valley], :]
    reference = np.zeros_like(h0)
    mesh_receipt = make_vituri2024_finite_domain_mesh_receipt(
        ordered_mesh=mesh,
        area_angstrom_squared=spec.area_angstrom_squared,
        provenance=(
            "Square finite-volume reciprocal spacing Delta k=2pi/sqrt(A), odd "
            "Cartesian UV subset, one equal-weight momentum state per A; "
            "independent reproduction choice requiring cutoff convergence."
        ),
    )
    q0_choice = make_vituri2024_translational_q0_reproduction_choice(
        evidence=(
            "Retain the finite dual-gate analytic q=0 direct and exchange terms "
            "explicitly; no identity quotient and no paper/source background claim."
        )
    )
    interaction = _interaction_choice(spec)
    functional_provenance = (
        "Independent homogeneous Vituri HF functional at Delta1=28 meV; "
        "not author source, incommensurate ansatz, or production TDHF."
    )
    common_functional_arguments = {
        "ordered_mesh": mesh,
        "active_band_states": np.asarray(states, dtype=np.complex128),
        "h0_native": h0,
        "normal_order_reference_native": reference,
        "mesh_receipt": mesh_receipt,
        "interaction": interaction,
        "normal_order_reference_fingerprint": _array_sha256(reference),
        "q0_choice": q0_choice,
        "provenance": functional_provenance,
    }
    if fft_workers is None:
        functional: Vituri2024HomogeneousHFFunctional = (
            Vituri2024TranslationalHFFunctional(**common_functional_arguments)
        )
    else:
        functional = Vituri2024TranslationalHFFFTFunctional(
            integer_mesh_labels=labels,
            delta_k_inverse_angstrom=spec.delta_k_inverse_angstrom,
            fft_workers=fft_workers,
            **common_functional_arguments,
        )
    fixed_density_choice = Vituri2024FixedDensitySCFChoice()
    readonly_states = _readonly(states, np.dtype(np.complex128))
    readonly_energies = _readonly(energies, np.dtype(np.float64))
    readonly_h0 = _readonly(h0, np.dtype(np.complex128))
    fingerprint = _prepared_fingerprint(
        spec=spec,
        mesh=mesh,
        labels=labels,
        states=readonly_states,
        energies=readonly_energies,
        h0=readonly_h0,
        functional=functional,
        fixed_density_choice=fixed_density_choice,
        minimum_lower_gap_ev=float(np.min(lower_gaps)),
        minimum_upper_gap_ev=float(np.min(upper_gaps)),
    )
    return Vituri2024PreparedHomogeneousHF(
        spec=spec,
        ordered_mesh=mesh,
        integer_mesh_labels=labels,
        active_band_states=readonly_states,
        active_band_energies_by_valley=readonly_energies,
        h0_native=readonly_h0,
        functional=functional,
        fixed_density_scf_choice=fixed_density_choice,
        minimum_lower_gap_ev=float(np.min(lower_gaps)),
        minimum_upper_gap_ev=float(np.min(upper_gaps)),
        fingerprint=fingerprint,
    )


def prepare_vituri2024_homogeneous_hf(
    spec: Vituri2024CartesianHFSpec,
) -> Vituri2024PreparedHomogeneousHF:
    """Build the unchanged dense-oracle homogeneous Vituri HF preparation."""

    return _prepare_vituri2024_homogeneous_hf(spec, fft_workers=None)


def prepare_vituri2024_homogeneous_hf_fft(
    spec: Vituri2024CartesianHFSpec,
    fft_workers: int = 1,
) -> Vituri2024PreparedHomogeneousHF:
    """Build an algebraically equivalent candidate exact no-wrap FFT backend.

    This preparation is not evidence of UV convergence, production readiness,
    a selected SCF branch, or paper reproduction.
    """

    return _prepare_vituri2024_homogeneous_hf(
        spec, fft_workers=fft_workers
    )


def _hole_momenta_by_valley(
    prepared: Vituri2024PreparedHomogeneousHF,
) -> dict[int, Array]:
    result: dict[int, Array] = {}
    for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
        order = np.argsort(
            -prepared.active_band_energies_by_valley[valley_index], kind="stable"
        )
        result[valley] = np.asarray(
            order[: prepared.spec.holes_per_valley], dtype=np.int64
        )
    return result


def _native_density_from_seed(
    prepared: Vituri2024PreparedHomogeneousHF,
    seed_mode: SeedMode,
    seed: int,
) -> Array:
    spec = prepared.spec
    conventional = np.repeat(np.eye(4, dtype=np.complex128)[:, :, None], spec.nk, axis=2)
    holes = _hole_momenta_by_valley(prepared)
    flavor_index = {
        flavor: index for index, flavor in enumerate(INTERNAL_FLAVOR_ORDER)
    }
    spinors = {
        "half_metal_sz_plus": np.asarray([0.0, 1.0], dtype=np.complex128),
        "half_metal_sz_minus": np.asarray([1.0, 0.0], dtype=np.complex128),
        "half_metal_sx": np.asarray([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),
        "half_metal_sy": np.asarray([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0),
    }
    if seed_mode in spinors:
        spinor = spinors[seed_mode]
        hole_projector = np.outer(spinor, spinor.conj())
        for valley in ACTIVE_BAND_STATES_VALLEY_ORDER:
            indices = np.asarray(
                [flavor_index[(valley, -1)], flavor_index[(valley, 1)]],
                dtype=np.int64,
            )
            for momentum in holes[valley]:
                conventional[np.ix_(indices, indices, [int(momentum)])] -= (
                    hole_projector[:, :, None]
                )
    elif seed_mode in ("valley_minus", "valley_plus"):
        valley = -1 if seed_mode == "valley_minus" else 1
        indices = [flavor_index[(valley, -1)], flavor_index[(valley, 1)]]
        selected = holes[valley]
        if selected.size != spec.holes_per_valley:
            raise ValueError("valley seed cannot realize the requested hole rank")
        # Remove two spin states at the highest H_v/2 momenta only when the
        # requested total hole rank is even.  This seed is otherwise undefined.
        if spec.total_holes % 2 != 0:
            raise ValueError("valley-polarized seed requires an even total hole rank")
        number_of_momenta = spec.total_holes // 2
        valley_order = np.argsort(
            -prepared.active_band_energies_by_valley[
                ACTIVE_BAND_STATES_VALLEY_ORDER.index(valley)
            ],
            kind="stable",
        )[:number_of_momenta]
        for momentum in valley_order:
            for flavor in indices:
                conventional[flavor, flavor, int(momentum)] = 0.0
    elif seed_mode in ("ivc_x", "ivc_y"):
        phase = 1.0 if seed_mode == "ivc_x" else 1.0j
        valley_spinor = np.asarray([1.0, phase], dtype=np.complex128) / math.sqrt(2.0)
        combined_energy = 0.5 * np.sum(
            prepared.active_band_energies_by_valley, axis=0
        )
        momenta = np.argsort(-combined_energy, kind="stable")[: spec.total_holes]
        if momenta.size != spec.total_holes:
            raise ValueError("IVC seed cannot realize the requested hole rank")
        indices = np.asarray(
            [flavor_index[(-1, 1)], flavor_index[(1, 1)]], dtype=np.int64
        )
        hole_projector = np.outer(valley_spinor, valley_spinor.conj())
        for momentum in momenta:
            conventional[np.ix_(indices, indices, [int(momentum)])] -= (
                hole_projector[:, :, None]
            )
    elif seed_mode == "random_projector":
        base = _native_density_from_seed(prepared, "half_metal_sz_plus", seed)
        conventional = vituri2024_native_density_to_conventional_k_diagonal(base).copy()
        rng = np.random.default_rng(_strict_int(seed, "seed"))
        for momentum in range(spec.nk):
            raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            hermitian = raw + raw.conj().T
            _values, unitary = np.linalg.eigh(hermitian)
            conventional[:, :, momentum] = (
                unitary @ conventional[:, :, momentum] @ unitary.conj().T
            )
    else:
        raise ValueError(f"unsupported Vituri seed_mode={seed_mode!r}")
    realized_electrons = float(
        sum(np.trace(conventional[:, :, momentum]).real for momentum in range(spec.nk))
    )
    if abs(realized_electrons - spec.total_electrons) > 1.0e-9:
        raise ValueError(
            "seed projector does not realize the exact fixed particle rank: "
            f"{realized_electrons} != {spec.total_electrons}"
        )
    native = conventional.swapaxes(0, 1)
    return np.asarray(native, dtype=np.complex128)


def _density_update_builder(
    prepared: Vituri2024PreparedHomogeneousHF,
):
    total_occupied = prepared.spec.total_electrons

    def build(hamiltonian: Array) -> DensityUpdateResult:
        matrix = np.asarray(hamiltonian, dtype=np.complex128)
        if matrix.shape != (4, 4, prepared.spec.nk):
            raise ValueError("Vituri Aufbau Hamiltonian shape mismatch")
        energies = np.empty((4, prepared.spec.nk), dtype=np.float64)
        eigenvectors = np.empty((4, 4, prepared.spec.nk), dtype=np.complex128)
        for momentum in range(prepared.spec.nk):
            values, vectors = np.linalg.eigh(matrix[:, :, momentum])
            energies[:, momentum] = values
            eigenvectors[:, :, momentum] = vectors
        # Explicit flavor-major-then-k flattening: flat=flavor*Nk+k.
        flat_energies = energies.reshape(-1, order="C")
        order = np.argsort(flat_energies, kind="stable")
        occupied_flat = np.zeros(flat_energies.size, dtype=np.bool_)
        occupied_flat[order[:total_occupied]] = True
        occupied = occupied_flat.reshape(energies.shape, order="C")
        conventional = np.zeros_like(matrix)
        for momentum in range(prepared.spec.nk):
            weights = occupied[:, momentum].astype(np.float64)
            vectors = eigenvectors[:, :, momentum]
            conventional[:, :, momentum] = (vectors * weights) @ vectors.conj().T
        native = np.asarray(conventional.swapaxes(0, 1), dtype=np.complex128)
        sorted_energies = flat_energies[order]
        if total_occupied == 0:
            mu = float(sorted_energies[0] - 1.0)
            gap = float("nan")
        elif total_occupied == sorted_energies.size:
            mu = float(sorted_energies[-1] + 1.0)
            gap = float("nan")
        else:
            lower = float(sorted_energies[total_occupied - 1])
            upper = float(sorted_energies[total_occupied])
            mu = 0.5 * (lower + upper)
            gap = upper - lower
        if math.isfinite(gap) and gap <= VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV:
            raise ValueError(
                "Vituri global Aufbau boundary is degenerate within the locked "
                "finite-volume tolerance; no partial-subspace occupation policy exists"
            )
        return DensityUpdateResult(
            density=native,
            energies=energies,
            mu=mu,
            observables={
                "finite_size_fermi_gap_ev": gap,
                "occupied_count": float(np.count_nonzero(occupied)),
            },
        )

    return build


def _maximum_overlap_density_update_builder(
 prepared: Vituri2024PreparedHomogeneousHF,
 state: Vituri2024HFState,
 choice: Vituri2024MaximumOverlapAufbauChoice,
 audit_state: _Vituri2024ExplicitBranchAuditState,
) -> StateBoundPreviousDensityBuilder:
 """Build a history-aware fixed-rank ground-state map for exact Fock ties."""

 choice.validate_live_state()
 if type(audit_state) is not _Vituri2024ExplicitBranchAuditState:
  raise TypeError("audit_state must be a private explicit-branch audit state")
 total_occupied = prepared.spec.total_electrons
 ordinary_builder = _density_update_builder(prepared)

 def build(
  hamiltonian: Array,
  previous_native_density: Array,
 ) -> DensityUpdateResult:
  choice.validate_live_state()
  if audit_state.terminal_rejection:
   raise RuntimeError("explicit shell branch audit state is terminally rejected")
  if audit_state.pending is not None:
   raise RuntimeError("explicit shell branch result was not audited")
  matrix = np.asarray(hamiltonian, dtype=np.complex128)
  previous_native = np.asarray(previous_native_density, dtype=np.complex128)
  expected_shape = (4, 4, prepared.spec.nk)
  if matrix.shape != expected_shape or previous_native.shape != expected_shape:
   raise ValueError("Vituri maximum-overlap Aufbau shape mismatch")
  if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(previous_native)):
   raise ValueError("Vituri maximum-overlap inputs must be finite")
  previous = vituri2024_native_density_to_conventional_k_diagonal(
   previous_native
  )
  hermiticity = float(
   np.max(np.abs(previous - previous.swapaxes(0, 1).conj()))
  )
  particle_number = float(
   sum(
    np.trace(previous[:, :, momentum]).real
    for momentum in range(prepared.spec.nk)
   )
  )
  if hermiticity > 1.0e-10:
   raise ValueError("previous ODA density is materially non-Hermitian")
  if abs(particle_number - total_occupied) > 1.0e-7:
   raise ValueError("previous ODA density changed the exact particle number")
  minimum_density_eigenvalue = float("inf")
  maximum_density_eigenvalue = float("-inf")
  for momentum in range(prepared.spec.nk):
   density_eigenvalues = np.linalg.eigvalsh(previous[:, :, momentum])
   minimum_density_eigenvalue = min(
    minimum_density_eigenvalue,
    float(density_eigenvalues[0]),
   )
   maximum_density_eigenvalue = max(
    maximum_density_eigenvalue,
    float(density_eigenvalues[-1]),
   )
  if minimum_density_eigenvalue < -1.0e-10 or maximum_density_eigenvalue > 1.0 + 1.0e-10:
   raise ValueError("previous ODA density left the fermionic interval [0, 1]")

  energies = np.empty((4, prepared.spec.nk), dtype=np.float64)
  eigenvectors = np.empty(
   (4, 4, prepared.spec.nk), dtype=np.complex128
  )
  for momentum in range(prepared.spec.nk):
   values, vectors = np.linalg.eigh(matrix[:, :, momentum])
   energies[:, momentum] = values
   eigenvectors[:, :, momentum] = vectors
  flat_energies = energies.reshape(-1, order="C")
  order = np.argsort(flat_energies, kind="stable")
  sorted_energies = flat_energies[order]
  lower = float(sorted_energies[total_occupied - 1])
  upper = float(sorted_energies[total_occupied])
  gap = upper - lower
  if gap > choice.energy_shell_tolerance_ev:
   return ordinary_builder(matrix)
  if choice.exact_energy_tie_required and gap != 0.0:
   raise ValueError(
    "Vituri boundary splitting is unresolved but not an exact energy tie"
   )

  shell_flat = np.flatnonzero(flat_energies == lower)
  below_flat = np.flatnonzero(flat_energies < lower)
  shell_rank = total_occupied - int(below_flat.size)
  shell_dimension = int(shell_flat.size)
  if not 0 < shell_rank < shell_dimension:
   raise RuntimeError("exact boundary shell inventory is inconsistent")

  overlap = np.zeros(
   (shell_dimension, shell_dimension), dtype=np.complex128
  )
  shell_band = shell_flat // prepared.spec.nk
  shell_momentum = shell_flat % prepared.spec.nk
  for left in range(shell_dimension):
   momentum = int(shell_momentum[left])
   vector_left = eigenvectors[:, int(shell_band[left]), momentum]
   for right in range(shell_dimension):
    if int(shell_momentum[right]) != momentum:
     continue
    vector_right = eigenvectors[:, int(shell_band[right]), momentum]
    overlap[left, right] = np.vdot(
     vector_left,
     previous[:, :, momentum] @ vector_right,
    )
  selection = select_maximum_overlap_rank_projector(
   overlap,
   shell_rank,
   overlap_gap_tolerance=choice.overlap_gap_tolerance,
  )
  explicit_branch_used = False
  explicit_branch_index = -1
  explicit_branch_generation = -1
  explicit_branch_sequence = _explicit_shell_branch_sequence(choice)
  explicit_branch_path_length = len(explicit_branch_sequence)
  if selection.unique and selection.coefficient_projector is not None:
   coefficient_projector = selection.coefficient_projector
  else:
   if not explicit_branch_sequence:
    raise ValueError(
     "Vituri maximum-overlap boundary remains nonunique; "
     "explicit broken-symmetry branch fanout is required"
    )
   explicit_branch_generation = audit_state.next_generation
   if explicit_branch_generation >= explicit_branch_path_length:
    audit_state.terminal_rejection = True
    raise ValueError(
     "Vituri maximum-overlap branch path is exhausted; "
     "a new explicit broken-symmetry branch fanout is required"
    )
   branch = explicit_branch_sequence[explicit_branch_generation]
   branch.validate_live_state()
   trigger_fock_sha256 = _array_sha256(matrix)
   trigger_previous_density_sha256 = _array_sha256(previous_native)
   if trigger_fock_sha256 != branch.trigger_fock_sha256:
    raise ValueError("explicit shell branch Fock trigger mismatch")
   if trigger_previous_density_sha256 != branch.trigger_previous_density_sha256:
    raise ValueError("explicit shell branch previous-density trigger mismatch")
   actual_shell = tuple(int(value) for value in shell_flat.tolist())
   if actual_shell != branch.expected_shell_flat_indices:
    raise ValueError("explicit shell branch inventory mismatch")
   if len(branch.selected_shell_flat_indices) != shell_rank:
    raise ValueError("explicit shell branch rank mismatch")
   if len(set(int(value) for value in shell_momentum.tolist())) != shell_dimension:
    raise ValueError(
     "coordinate shell branch requires one shell state per momentum block"
    )
   positions_by_flat = {
    int(flat_index): position
    for position, flat_index in enumerate(shell_flat.tolist())
   }
   selected_positions = tuple(
    positions_by_flat[int(flat_index)]
    for flat_index in branch.selected_shell_flat_indices
   )
   coefficient_projector = np.zeros(
    (shell_dimension, shell_dimension), dtype=np.complex128
   )
   coefficient_projector[selected_positions, selected_positions] = 1.0
   branch_overlap = float(
    np.trace(overlap @ coefficient_projector).real
   )
   if abs(branch_overlap - selection.maximum_overlap_value) > max(
    choice.overlap_gap_tolerance,
    64.0 * np.finfo(np.float64).eps * max(1.0, abs(branch_overlap)),
   ):
    raise ValueError("explicit shell branch is not a maximum-overlap minimizer")
   explicit_branch_used = True
   explicit_branch_index = branch.branch_index
  same_momentum = shell_momentum[:, None] == shell_momentum[None, :]
  cross_momentum_entries = coefficient_projector[~same_momentum]
  cross_momentum_residual = (
   0.0
   if cross_momentum_entries.size == 0
   else float(np.max(np.abs(cross_momentum_entries)))
  )
  if cross_momentum_residual > 1.0e-10:
   raise ValueError(
    "maximum-overlap selection left the translation-preserving ansatz"
   )

  conventional = np.zeros_like(matrix)
  below_mask = np.zeros(flat_energies.size, dtype=np.bool_)
  below_mask[below_flat] = True
  below = below_mask.reshape(energies.shape, order="C")
  for momentum in range(prepared.spec.nk):
   weights = below[:, momentum].astype(np.float64)
   vectors = eigenvectors[:, :, momentum]
   conventional[:, :, momentum] = (vectors * weights) @ vectors.conj().T
   positions = np.flatnonzero(shell_momentum == momentum)
   if positions.size:
    shell_vectors = eigenvectors[
     :,
     shell_band[positions],
     momentum,
    ]
    local_projector = coefficient_projector[np.ix_(positions, positions)]
    conventional[:, :, momentum] += (
     shell_vectors @ local_projector @ shell_vectors.conj().T
    )

  projector_residual = 0.0
  selected_particles = 0.0
  for momentum in range(prepared.spec.nk):
   block = conventional[:, :, momentum]
   selected_particles += float(np.trace(block).real)
   projector_residual = max(
    projector_residual,
    float(np.max(np.abs(block - block.conj().T))),
    float(np.max(np.abs(block @ block - block))),
   )
  if abs(selected_particles - total_occupied) > 1.0e-7:
   raise RuntimeError("maximum-overlap Aufbau rank drifted")
  if projector_residual > 1.0e-10:
   raise RuntimeError("maximum-overlap Aufbau projector invariants failed")
  native = np.asarray(
   conventional.swapaxes(0, 1), dtype=np.complex128
  )
  mu = 0.5 * (lower + upper)
  update = DensityUpdateResult(
   density=native,
   energies=energies,
   mu=mu,
   observables={
    "finite_size_fermi_gap_ev": gap,
    "occupied_count": selected_particles,
    "degenerate_shell_multiplicity": float(shell_dimension),
    "degenerate_shell_selected_rank": float(shell_rank),
    "maximum_overlap_cutoff_gap": float(
     selection.overlap_cutoff_gap
    ),
    "maximum_overlap_value": selection.maximum_overlap_value,
    "maximum_overlap_cross_momentum_residual": cross_momentum_residual,
    "explicit_coordinate_branch_used": float(explicit_branch_used),
    "explicit_coordinate_branch_index": float(explicit_branch_index),
    "explicit_coordinate_branch_generation": float(
     explicit_branch_generation
    ),
    "explicit_coordinate_branch_path_length": float(
     explicit_branch_path_length
    ),
   },
  )
  if explicit_branch_used:
   if audit_state.pending is not None:
    raise RuntimeError("explicit shell branch pending receipt was overwritten")
   audit_state.pending = _Vituri2024PendingExplicitBranchUse(
    generation=explicit_branch_generation,
    branch_fingerprint=branch.fingerprint,
    trigger_fock_sha256=trigger_fock_sha256,
    trigger_previous_density_sha256=trigger_previous_density_sha256,
    update_object_id=id(update),
    update_density_sha256=_array_sha256(update.density),
   )
  return update

 return StateBoundPreviousDensityBuilder(
  state=state,
  callback=build,
  policy_fingerprint=choice.fingerprint,
 )


def make_vituri2024_hf_maximum_overlap_problem(
 prepared: Vituri2024PreparedHomogeneousHF,
 state: Vituri2024HFState,
 choice: Vituri2024MaximumOverlapAufbauChoice | None = None,
) -> HartreeFockProblem:
 """Return an ODA-compatible problem with fail-closed exact-tie continuation.

 This independent numerical realization can follow an exhaustively declared,
 exact-triggered coordinate branch path. It does not infer the paper's
 unpublished many-start branch-selection policy.
 """

 if type(prepared) is not Vituri2024PreparedHomogeneousHF:
  raise TypeError("prepared must be Vituri2024PreparedHomogeneousHF")
 if type(state) is not Vituri2024HFState:
  raise TypeError("state must be Vituri2024HFState")
 prepared.validate_live_state()
 if state.h0.shape != prepared.h0_native.shape or not np.array_equal(
  state.h0, prepared.h0_native
 ):
  raise ValueError("state is not bound to the prepared Vituri problem")
 selected_choice = (
  Vituri2024MaximumOverlapAufbauChoice() if choice is None else choice
 )
 if type(selected_choice) is not Vituri2024MaximumOverlapAufbauChoice:
  raise TypeError("choice must be Vituri2024MaximumOverlapAufbauChoice")
 selected_choice.validate_live_state()
 explicit_branch_sequence = _explicit_shell_branch_sequence(selected_choice)
 audit_state = _Vituri2024ExplicitBranchAuditState()
 baseline = make_vituri2024_hf_problem(prepared)
 if baseline.kernel.step_callback is not None:
  raise RuntimeError("baseline Vituri problem unexpectedly owns a step callback")
 if baseline.kernel.final_state_callback is not None:
  raise RuntimeError("baseline Vituri problem unexpectedly owns a final callback")

 def bound_initializer(
  target_state: Vituri2024HFState,
  *,
  init_mode: str,
  seed: int,
 ) -> None:
  if target_state is not state:
   raise ValueError("maximum-overlap problem was run with a different HF state")
  if (
   audit_state.next_generation != 0
   or audit_state.last_branch_iteration != 0
   or audit_state.pending is not None
   or audit_state.consumed
   or audit_state.terminal_rejection
  ):
   raise RuntimeError("maximum-overlap problem cannot be reinitialized after use")
  baseline.initializer(target_state, init_mode=init_mode, seed=seed)

 def validate_consumed_branch_diagnostics(
  target_state: Vituri2024HFState,
 ) -> None:
  diagnostics = target_state.diagnostics
  actual_keys = {
   key for key in diagnostics if key.startswith("explicit_coordinate_branch_")
  }
  if not audit_state.consumed:
   if actual_keys:
    raise RuntimeError("explicit branch diagnostics are stale or preseeded")
   return
  expected_keys = {
   "explicit_coordinate_branch_used",
   "explicit_coordinate_branch_use_count",
   "explicit_coordinate_branch_path_length",
   "explicit_coordinate_branch_index",
   "explicit_coordinate_branch_count",
   "explicit_coordinate_branch_iteration",
   "explicit_coordinate_branch_last_generation",
   "explicit_coordinate_branch_last_index",
   "explicit_coordinate_branch_last_count",
   "explicit_coordinate_branch_last_iteration",
  }
  expected_generation_values: dict[str, float] = {}
  for generation, (index, count, iteration, _fingerprint) in enumerate(
   audit_state.consumed
  ):
   expected_generation_values.update(
    {
     f"explicit_coordinate_branch_generation_{generation}_index": float(index),
     f"explicit_coordinate_branch_generation_{generation}_count": float(count),
     f"explicit_coordinate_branch_generation_{generation}_iteration": float(
      iteration
     ),
    }
   )
  expected_keys.update(expected_generation_values)
  if actual_keys != expected_keys:
   raise RuntimeError("explicit branch diagnostic namespace drifted")
  if (
   float(diagnostics["explicit_coordinate_branch_used"]) != 1.0
   or float(diagnostics["explicit_coordinate_branch_use_count"])
   != float(len(audit_state.consumed))
   or float(diagnostics["explicit_coordinate_branch_path_length"])
   != float(len(explicit_branch_sequence))
  ):
   raise RuntimeError("explicit branch diagnostic aggregate drifted")
  if any(
   float(diagnostics[key]) != value
   for key, value in expected_generation_values.items()
  ):
   raise RuntimeError("explicit branch generation diagnostics drifted")
  first_index, first_count, first_iteration, _ = audit_state.consumed[0]
  last_index, last_count, last_iteration, _ = audit_state.consumed[-1]
  if (
   float(diagnostics["explicit_coordinate_branch_index"]) != float(first_index)
   or float(diagnostics["explicit_coordinate_branch_count"]) != float(first_count)
   or float(diagnostics["explicit_coordinate_branch_iteration"])
   != float(first_iteration)
   or float(diagnostics["explicit_coordinate_branch_last_generation"])
   != float(len(audit_state.consumed) - 1)
   or float(diagnostics["explicit_coordinate_branch_last_index"])
   != float(last_index)
   or float(diagnostics["explicit_coordinate_branch_last_count"])
   != float(last_count)
   or float(diagnostics["explicit_coordinate_branch_last_iteration"])
   != float(last_iteration)
  ):
   raise RuntimeError("explicit branch first/last diagnostics drifted")

 def branch_audit_callback(
  target_state: Vituri2024HFState,
  step: HartreeFockStepResult,
 ) -> None:
  if audit_state.terminal_rejection:
   raise RuntimeError("explicit branch audit state is terminally rejected")
  selected_choice.validate_live_state()
  if target_state is not state:
   raise RuntimeError("explicit branch callback received a different HF state")
  if type(step) is not HartreeFockStepResult:
   raise TypeError("explicit branch callback requires HartreeFockStepResult")
  density_update = step.density_update
  observables = density_update.observables
  if float(observables.get("explicit_coordinate_branch_used", 0.0)) != 1.0:
   if audit_state.pending is not None:
    raise RuntimeError("non-branch step left a pending branch receipt")
   validate_consumed_branch_diagnostics(target_state)
   return
  raw_generation = float(observables["explicit_coordinate_branch_generation"])
  raw_path_length = float(observables["explicit_coordinate_branch_path_length"])
  if any(
   not math.isfinite(value) or value < 0.0 or value != math.floor(value)
   for value in (raw_generation, raw_path_length)
  ):
   raise RuntimeError("explicit coordinate branch audit counters drifted")
  generation = int(raw_generation)
  path_length = int(raw_path_length)
  use_count = audit_state.next_generation
  pending = audit_state.pending
  if pending is None:
   raise RuntimeError("branch-use observable lacks a pending builder receipt")
  if (
   pending.update_object_id != id(density_update)
   or pending.update_density_sha256 != _array_sha256(density_update.density)
   or pending.trigger_fock_sha256 != _array_sha256(step.total_hamiltonian)
   or pending.trigger_previous_density_sha256
   != _array_sha256(step.previous_density)
   or pending.generation != generation
  ):
   raise RuntimeError("branch-use observable does not match its builder receipt")
  validate_consumed_branch_diagnostics(target_state)
  if path_length != len(explicit_branch_sequence):
   raise RuntimeError("branch-use observable path length drifted")
  if generation != use_count:
   raise RuntimeError("explicit coordinate branch trigger was reused or reordered")
  if generation >= len(explicit_branch_sequence):
   raise RuntimeError("branch-use observable exceeds the typed branch path")
  branch = explicit_branch_sequence[generation]
  branch.validate_live_state()
  if (
   pending.branch_fingerprint != branch.fingerprint
   or float(observables["explicit_coordinate_branch_index"])
   != float(branch.branch_index)
  ):
   raise RuntimeError("branch-use observable index/fingerprint drifted")
  iteration = _strict_int(step.iteration, "explicit branch iteration")
  if iteration <= audit_state.last_branch_iteration:
   raise RuntimeError("explicit branch iteration did not increase")
  oda_lambda = float(step.oda_lambda)
  if not math.isfinite(oda_lambda) or not 0.0 <= oda_lambda <= 1.0:
   raise RuntimeError("explicit branch step has an invalid ODA parameter")
  expected_mixed = (
   oda_lambda * np.asarray(density_update.density)
   + (1.0 - oda_lambda) * np.asarray(step.previous_density)
  )
  mixed = np.asarray(step.mixed_density)
  mixed_scale = max(1.0, _max_abs(expected_mixed))
  if (
   mixed.shape != expected_mixed.shape
   or not np.all(np.isfinite(mixed))
   or _max_abs(mixed - expected_mixed)
   > 32.0 * np.finfo(np.float64).eps * mixed_scale
   or not np.array_equal(np.asarray(target_state.density), mixed)
  ):
   raise RuntimeError("explicit branch receipt is not bound to the applied state")
  audit_state.next_generation = use_count + 1
  audit_state.last_branch_iteration = iteration
  audit_state.pending = None
  audit_state.consumed.append(
   (branch.branch_index, branch.branch_count, iteration, branch.fingerprint)
  )
  target_state.diagnostics["explicit_coordinate_branch_used"] = 1.0
  target_state.diagnostics["explicit_coordinate_branch_use_count"] = float(
   use_count + 1
  )
  target_state.diagnostics["explicit_coordinate_branch_path_length"] = float(
   path_length
  )
  target_state.diagnostics[
   f"explicit_coordinate_branch_generation_{generation}_index"
  ] = float(branch.branch_index)
  target_state.diagnostics[
   f"explicit_coordinate_branch_generation_{generation}_count"
  ] = float(branch.branch_count)
  target_state.diagnostics[
   f"explicit_coordinate_branch_generation_{generation}_iteration"
  ] = float(iteration)
  target_state.diagnostics["explicit_coordinate_branch_last_generation"] = float(
   generation
  )
  target_state.diagnostics["explicit_coordinate_branch_last_index"] = float(
   branch.branch_index
  )
  target_state.diagnostics["explicit_coordinate_branch_last_count"] = float(
   branch.branch_count
  )
  target_state.diagnostics["explicit_coordinate_branch_last_iteration"] = float(
   iteration
  )
  if generation == 0:
   target_state.diagnostics["explicit_coordinate_branch_index"] = float(
    branch.branch_index
   )
   target_state.diagnostics["explicit_coordinate_branch_count"] = float(
    branch.branch_count
   )
   target_state.diagnostics["explicit_coordinate_branch_iteration"] = float(
    iteration
   )

 def final_branch_audit_callback(
  target_state: Vituri2024HFState,
  density_update: DensityUpdateResult,
 ) -> None:
  if audit_state.terminal_rejection:
   raise RuntimeError("explicit branch audit state is terminally rejected")
  selected_choice.validate_live_state()
  if target_state is not state:
   raise RuntimeError("explicit final callback received a different HF state")
  if type(density_update) is not DensityUpdateResult:
   raise TypeError("explicit final callback requires DensityUpdateResult")
  observables = density_update.observables
  if float(observables.get("explicit_coordinate_branch_used", 0.0)) == 1.0:
   pending = audit_state.pending
   if (
    pending is None
    or pending.update_object_id != id(density_update)
    or pending.update_density_sha256 != _array_sha256(density_update.density)
   ):
    raise RuntimeError("final branch use lacks its builder receipt")
   audit_state.pending = None
   audit_state.terminal_rejection = True
   raise RuntimeError(
    "final recomputation encountered an unresolved explicit branch; "
    "a new branch fanout is required"
   )
  if audit_state.pending is not None:
   raise RuntimeError("final recomputation left a pending branch receipt")
  validate_consumed_branch_diagnostics(target_state)

 return replace(
  baseline,
  initializer=bound_initializer,
  kernel=replace(
   baseline.kernel,
   density_builder=_maximum_overlap_density_update_builder(
    prepared,
    state,
    selected_choice,
    audit_state,
   ),
   step_callback=branch_audit_callback,
   final_state_callback=final_branch_audit_callback,
  ),
 )


def _energy_from_engine_inputs(
    interaction_h: Array, h0: Array, native_density: Array
) -> float:
    one_body = np.einsum("abk,abk->", h0, native_density, optimize=False)
    interaction = 0.5 * np.einsum(
        "abk,abk->", interaction_h, native_density, optimize=False
    )
    total = complex(one_body + interaction)
    if abs(total.imag) > 5.0e-11 * max(1.0, abs(total), abs(one_body), abs(interaction)):
        raise ValueError("Vituri engine scalar energy is materially complex")
    return float(total.real)


def make_vituri2024_hf_problem(
    prepared: Vituri2024PreparedHomogeneousHF,
) -> HartreeFockProblem:
    if type(prepared) is not Vituri2024PreparedHomogeneousHF:
        raise TypeError("prepared must be Vituri2024PreparedHomogeneousHF")
    prepared.validate_live_state()

    def initializer(state: Vituri2024HFState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = _native_density_from_seed(
            prepared, init_mode, seed  # type: ignore[arg-type]
        )
        state.hamiltonian[:, :, :] = state.h0
        state.energies[:, :] = np.nan
        state.mu = float("nan")
        state.diagnostics.clear()

    interaction_action = prepared.functional.make_validated_interaction_action()
    return HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_action,
            density_builder=_density_update_builder(prepared),
            energy_functional=_energy_from_engine_inputs,
            oda_delta_interaction_builder=interaction_action,
            convergence_rule="raw",
        ),
    )


def make_vituri2024_hf_state(
    prepared: Vituri2024PreparedHomogeneousHF,
) -> Vituri2024HFState:
    h0 = np.asarray(prepared.h0_native, dtype=np.complex128).copy()
    return Vituri2024HFState(
        h0=h0,
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=np.full((4, prepared.spec.nk), np.nan, dtype=np.float64),
        mu=float("nan"),
        precision=prepared.spec.precision,
        diagnostics={},
    )


def scan_vituri2024_initial_fock_aufbau_boundaries(
    choice: Vituri2024InitialFockBoundaryScanChoice,
) -> Vituri2024InitialFockBoundarySelection:
    """Select a nearby closed initial-Fock boundary for one declared branch.

    Every candidate keeps the physical total hole density fixed. Changing
    ``H_v`` changes the finite area, reciprocal spacing, UV subset, and
    interaction normalization, so the result is regulator admission only and
    must never be reported as cutoff convergence or a repair of another finite
    system.
    """

    if type(choice) is not Vituri2024InitialFockBoundaryScanChoice:
        raise TypeError("choice must be Vituri2024InitialFockBoundaryScanChoice")
    choice.validate_live_state()
    records: list[Vituri2024InitialFockBoundaryRecord] = []
    for holes_per_valley in range(
        choice.scan_min_holes_per_valley,
        choice.scan_max_holes_per_valley + 1,
    ):
        spec = Vituri2024CartesianHFSpec(
            mesh_size=choice.mesh_size,
            holes_per_valley=holes_per_valley,
            total_hole_density_cm2=choice.total_hole_density_cm2,
            delta1_ev=choice.delta1_ev,
            gate_distance_angstrom=choice.gate_distance_angstrom,
            coulomb_e2_ev_angstrom=choice.coulomb_e2_ev_angstrom,
            precision=choice.precision,
        )
        prepared = prepare_vituri2024_homogeneous_hf(spec)
        problem = make_vituri2024_hf_problem(prepared)
        state = make_vituri2024_hf_state(prepared)
        problem.initializer(state, init_mode=choice.initial_branch_mode, seed=0)
        fock = state.h0 + problem.kernel.interaction_builder(state.density)
        analysis = analyze_vituri2024_global_aufbau_boundary(
            fock,
            total_occupied=spec.total_electrons,
            degeneracy_tolerance_ev=choice.degeneracy_tolerance_ev,
            minimum_gap_to_eigensolver_residual_ratio=(
                choice.minimum_gap_to_eigensolver_residual_ratio
            ),
        )
        records.append(
            Vituri2024InitialFockBoundaryRecord(
                holes_per_valley=holes_per_valley,
                analysis=analysis,
                area_angstrom_squared=spec.area_angstrom_squared,
                delta_k_inverse_angstrom=spec.delta_k_inverse_angstrom,
                axial_cutoff_a0=spec.axial_k_cutoff_a0,
                spec_fingerprint=spec.fingerprint,
                initial_density_sha256=_array_sha256(state.density),
                initial_fock_sha256=_array_sha256(fock),
            )
        )
    candidates = tuple(
        record
        for record in records
        if record.analysis.closed_global_aufbau_boundary
    )
    if not candidates:
        raise ValueError("initial-Fock boundary scan has no admitted candidate")
    selected = min(
        candidates,
        key=lambda record: _initial_fock_selection_key(choice, record),
    )
    frozen_records = tuple(records)
    target_record = next(
        record
        for record in frozen_records
        if record.holes_per_valley == choice.target_holes_per_valley
    )
    return Vituri2024InitialFockBoundarySelection(
        choice=choice,
        records=frozen_records,
        selected=selected,
        target_record=target_record,
        target_admitted=target_record.analysis.closed_global_aufbau_boundary,
        fallback_used=(selected.holes_per_valley != choice.target_holes_per_valley),
        selection_key=_initial_fock_selection_key(choice, selected),
        binding_fingerprint=_initial_fock_binding_fingerprint(
            choice, frozen_records
        ),
    )

def diagnose_vituri2024_half_metal(
    prepared: Vituri2024PreparedHomogeneousHF,
    run: HartreeFockRun,
) -> Vituri2024HalfMetalDiagnostics:
    state = run.state
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        np.asarray(state.density, dtype=np.complex128)
    )
    identity = np.eye(4, dtype=np.complex128)[:, :, None]
    hole = identity - conventional
    integrated_hole = np.sum(hole, axis=2)
    holes_by_flavor = tuple(
        float(integrated_hole[index, index].real) for index in range(4)
    )
    valley_blocks = (
        integrated_hole[0:2, 0:2],
        integrated_hole[2:4, 2:4],
    )
    holes_by_valley = tuple(float(np.trace(block).real) for block in valley_blocks)
    spin_hole = valley_blocks[0] + valley_blocks[1]
    spin_eigenvalues = np.linalg.eigvalsh(spin_hole)
    valley_spin_eigenvalues = tuple(np.linalg.eigvalsh(block) for block in valley_blocks)
    total_holes = float(np.trace(integrated_hole).real)
    pauli = (
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    )
    valley_spin_vectors = tuple(
        np.asarray([float(np.trace(block @ matrix).real) for matrix in pauli])
        for block in valley_blocks
    )
    spin_vector = valley_spin_vectors[0] + valley_spin_vectors[1]
    spin_polarization = float(np.linalg.norm(spin_vector) / max(total_holes, 1.0e-30))
    per_valley_spin_purity = tuple(
        float(max(0.0, eigenvalues[0]) / max(np.sum(eigenvalues), 1.0e-30))
        for eigenvalues in valley_spin_eigenvalues
    )
    normalized_valley_axes = tuple(
        vector / max(float(np.linalg.norm(vector)), 1.0e-30)
        for vector in valley_spin_vectors
    )
    common_axis_residual = float(
        np.linalg.norm(normalized_valley_axes[0] - normalized_valley_axes[1])
    )
    intervalley_norms = np.asarray(
        [
            np.linalg.norm(hole[0:2, 2:4, momentum])
            for momentum in range(prepared.spec.nk)
        ],
        dtype=np.float64,
    )
    intervalley_coherence = float(np.linalg.norm(intervalley_norms))
    intervalley_coherence_max = float(np.max(intervalley_norms, initial=0.0))
    valley_balance = float(abs(holes_by_valley[0] - holes_by_valley[1]))
    projector_residual = 0.0
    commutator_residual = 0.0
    for momentum in range(prepared.spec.nk):
        projector = conventional[:, :, momentum]
        fock = np.asarray(state.hamiltonian[:, :, momentum], dtype=np.complex128)
        projector_residual = max(
            projector_residual, _max_abs(projector @ projector - projector)
        )
        commutator_residual = max(
            commutator_residual, _max_abs(fock @ projector - projector @ fock)
        )
    electron_count = float(
        sum(np.trace(conventional[:, :, momentum]).real for momentum in range(prepared.spec.nk))
    )
    finite_gap = float("nan")
    if prepared.spec.total_electrons not in (0, 4 * prepared.spec.nk):
        flat = np.sort(np.asarray(state.energies, dtype=float).ravel(order="C"))
        finite_gap = float(
            flat[prepared.spec.total_electrons] - flat[prepared.spec.total_electrons - 1]
        )
    final_raw_norm = float(state.diagnostics.get("final_raw_norm", float("inf")))
    fully_spin_polarized = all(value <= 1.0e-7 for value in per_valley_spin_purity)
    common_axis_spin_polarized = (
        abs(total_holes - prepared.spec.total_holes) <= 1.0e-7
        and fully_spin_polarized
        and spin_polarization >= 1.0 - 1.0e-7
        and common_axis_residual <= 1.0e-7
        and float(spin_eigenvalues[0]) >= -1.0e-8
        and float(spin_eigenvalues[0]) <= 1.0e-7
    )
    intervalley_incoherent = intervalley_coherence_max <= 1.0e-7
    valley_balanced = valley_balance <= 1.0e-7
    stationary = (
        run.converged
        and final_raw_norm <= prepared.spec.precision
        and commutator_residual <= 1.0e-8
        and projector_residual <= 1.0e-8
    )
    valid = (
        common_axis_spin_polarized
        and intervalley_incoherent
        and valley_balanced
        and stationary
        and math.isfinite(finite_gap)
        and finite_gap > VITURI2024_DEFAULT_AUFBAU_GAP_TOLERANCE_EV
    )
    return Vituri2024HalfMetalDiagnostics(
        electron_count=electron_count,
        total_holes=total_holes,
        holes_by_flavor=holes_by_flavor,  # type: ignore[arg-type]
        holes_by_valley=holes_by_valley,  # type: ignore[arg-type]
        spin_hole_eigenvalues=(
            float(spin_eigenvalues[0]),
            float(spin_eigenvalues[1]),
        ),
        spin_polarization_norm=spin_polarization,
        per_valley_spin_purity_residuals=per_valley_spin_purity,  # type: ignore[arg-type]
        common_spin_axis_residual=common_axis_residual,
        intervalley_coherence_frobenius=intervalley_coherence,
        intervalley_coherence_max_per_k=intervalley_coherence_max,
        valley_balance_residual=valley_balance,
        projector_idempotency_residual=projector_residual,
        commutator_residual_ev=commutator_residual,
        final_raw_norm=final_raw_norm,
        finite_size_fermi_gap_ev=finite_gap,
        fully_spin_polarized=fully_spin_polarized,
        common_axis_spin_polarized=common_axis_spin_polarized,
        intervalley_incoherent=intervalley_incoherent,
        valley_balanced=valley_balanced,
        stationary=stationary,
        valid_homogeneous_half_metal_candidate=valid,
    )


def run_vituri2024_hf_seed(
    prepared: Vituri2024PreparedHomogeneousHF,
    *,
    seed_mode: SeedMode,
    seed: int,
    max_iter: int = 500,
    oda_stall_threshold: float = 1.0e-6,
    max_oda_lambda: float = 1.0,
) -> Vituri2024HFSeedRun:
    """Run one homogeneous seed through the reusable generic ODA engine."""

    state = make_vituri2024_hf_state(prepared)
    problem = make_vituri2024_hf_problem(prepared)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=seed_mode,
        seed=_strict_int(seed, "seed"),
        max_iter=_strict_int(max_iter, "max_iter"),
        oda_stall_threshold=_finite_real(
            oda_stall_threshold, "oda_stall_threshold", positive=True
        ),
        max_oda_lambda=_finite_real(
            max_oda_lambda, "max_oda_lambda", positive=True
        ),
    )
    prepared.validate_live_state()
    diagnostics = diagnose_vituri2024_half_metal(prepared, run)
    if run.converged and not diagnostics.stationary:
        run = replace(run, converged=False, exit_reason="final_map_not_converged")
        diagnostics = diagnose_vituri2024_half_metal(prepared, run)
    final_energy = prepared.functional.energy(np.asarray(run.state.density))
    engine_energy = float(run.state.diagnostics["hf_energy"])
    if abs(final_energy - engine_energy) > 1.0e-10 * max(
        1.0, abs(final_energy), abs(engine_energy)
    ):
        raise ValueError("Vituri final engine/scalar energy mismatch")
    return Vituri2024HFSeedRun(
        seed_mode=seed_mode,
        seed=int(seed),
        run=run,
        diagnostics=diagnostics,
        final_independent_model_energy_ev=final_energy,
    )


__all__ = [
    "VITURI2024_DELTA1_EV",
    "VITURI2024_GATE_DISTANCE_ANGSTROM",
    "VITURI2024_HF_SCF_API_VERSION",
 "VITURI2024_MAXIMUM_OVERLAP_AUFBAU_API_VERSION",
    "VITURI2024_HF_SCF_AUTHORITY",
    "VITURI2024_TOTAL_HOLE_DENSITY_CM2",
    "Vituri2024CartesianHFSpec",
    "Vituri2024ExplicitShellBranchChoice",
 "Vituri2024ExplicitShellBranchPath",
 "Vituri2024FixedDensitySCFChoice",
    "Vituri2024HalfMetalDiagnostics",
    "Vituri2024HFSeedRun",
    "Vituri2024GlobalAufbauBoundaryAnalysis",
    "Vituri2024InitialFockBoundaryRecord",
    "Vituri2024InitialFockBoundaryScanChoice",
    "Vituri2024InitialFockBoundarySelection",
 "Vituri2024MaximumOverlapAufbauChoice",
    "Vituri2024HFState",
    "Vituri2024HomogeneousHFFunctional",
    "Vituri2024PreparedHomogeneousHF",
    "analyze_vituri2024_global_aufbau_boundary",
    "build_vituri2024_cartesian_mesh",
    "diagnose_vituri2024_half_metal",
    "make_vituri2024_cartesian_hf_spec_from_spacing",
    "make_vituri2024_explicit_shell_branch_choices",
 "make_vituri2024_hf_maximum_overlap_problem",
 "make_vituri2024_hf_problem",
    "make_vituri2024_hf_state",
    "prepare_vituri2024_homogeneous_hf",
    "prepare_vituri2024_homogeneous_hf_fft",
    "run_vituri2024_hf_seed",
    "scan_vituri2024_initial_fock_aufbau_boundaries",
]
