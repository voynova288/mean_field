"""Candidate-only G=0 finite-q IVC-spiral Hartree--Fock adapter.

Physics provenance
------------------
The shifted basis

``p[tau,k] = k + tau*q/2``

is the ``G1=G2=0`` specialization of the Slater-determinant ansatz in
arXiv:2408.10309v1, supplemental ``SM.tex``, section "Self-consistent
Hartree-Fock calculations": ``k_i=k+G_i+tau_i*q_1/2``.  The six-band states
come from supplemental Eq. ``Ham6``.  The interaction and mean-field action
are not reimplemented here: :class:`Vituri2024TranslationalHFFunctional`
continues to implement the projected-interaction and mean-field equations in
the preceding supplemental section and the equation immediately following the
variational Hamiltonian.  The ODA update is delegated to the common HF engine,
matching supplemental Eq. ``eq:Phi``.

Only the central finite square of canonical ``k`` points is retained.  There
is no wrap, interpolation, reciprocal carry, or additional G harmonic.  Thus
this is an IVC *spiral* candidate, not the paper's multi-G IVC crystal ansatz.
The displayed-basis B3 gauge implemented here uses ``BASIS.index("B3")==1``.
The source instead calls B3 ``psi_6`` in its gauge paragraph, so this
independent gauge is explicitly not identified with the author gauge.

For one selected spin, the opposite-spin block is exactly full.  The selected
valley block has one global electron rank ``2*Nk-2*H_v``; no per-valley rank is
imposed.  Its finite-volume occupation boundary must have a strictly positive
gap above an explicit floor.  Exact and subtolerance boundaries fail closed.

The IVC observable follows the main-text definition
``phi_q=A^-1 sum_k <c^dagger_{+,k+q} c_{-,k}>``.  In the symmetric shifted
basis this is ``A^-1 sum_k rho[+, -, k]`` for the selected spin.  The reported
``|phi|/|n|`` uses ``|n|=2*H_v/A`` and is only an independent displayed-B3-
gauge candidate; it is never labeled as the author's gauge or Fig. 2
reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Final, Literal

import numpy as np

from mean_field.core.hf import (
    DensityUpdateResult,
    HartreeFockKernel,
    HartreeFockProblem,
    HartreeFockRun,
    run_hartree_fock_problem,
)

from .vituri2024 import BASIS, third_lowest_active_band
from .vituri2024_hf import (
    Vituri2024TranslationalHFFunctional,
    vituri2024_native_density_to_conventional_k_diagonal,
)
from .vituri2024_hf_preflight import (
    ACTIVE_BAND_STATES_VALLEY_ORDER,
    INTERNAL_FLAVOR_ORDER,
)
from .vituri2024_hf_scf import (
    Vituri2024HFState,
    Vituri2024PreparedHomogeneousHF,
)

Array = np.ndarray
GaugeMode = Literal["identity", "displayed_b3"]
SpiralInitializerMode = Literal["normal", "ivc_b3"]

VITURI2024_HF_SPIRAL_API_VERSION: Final[str] = "vituri2024_g0_ivc_spiral_candidate.v1"
VITURI2024_HF_SPIRAL_AUTHORITY: Final[str] = (
    "independent_displayed_b3_gauge_g0_finite_q_candidate_only_not_author_gauge_"
    "not_multig_crystal_production_or_fig2_reproduction"
)
VITURI2024_B3_COMPONENT_INDEX: Final[int] = 1
VITURI2024_DEFAULT_B3_ANCHOR_FLOOR: Final[float] = 1.0e-10
VITURI2024_DEFAULT_SPIRAL_OCCUPATION_GAP_FLOOR_EV: Final[float] = 1.0e-12
VITURI2024_DEFAULT_SPIN_BLOCK_TOLERANCE_EV: Final[float] = 1.0e-10


def _strict_real(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a strict real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _readonly(value: object, dtype: np.dtype | None = None) -> Array:
    array = np.ascontiguousarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    result.setflags(write=False)
    return result


def _array_sha256(value: object) -> str:
    array = np.ascontiguousarray(value)
    payload = (
        str(array.dtype).encode()
        + b"\0"
        + json.dumps(array.shape).encode()
        + b"\0"
        + array.view(np.uint8).tobytes()
    )
    return sha256(payload).hexdigest()


def _canonical_run_diagnostics(value: object) -> tuple[tuple[str, str, object], ...]:
    if type(value) is not dict:
        raise TypeError("run state diagnostics must be a canonical dict")
    canonical: list[tuple[str, str, object]] = []
    for key in sorted(value):
        if type(key) is not str:
            raise TypeError("run state diagnostic keys must be strings")
        scalar = value[key]
        if type(scalar) is bool:
            canonical.append((key, "bool", scalar))
        elif type(scalar) is int:
            canonical.append((key, "int", scalar))
        elif type(scalar) is float:
            if not math.isfinite(scalar):
                raise ValueError(f"run state diagnostic {key!r} must be finite")
            canonical.append((key, "float", scalar.hex()))
        elif type(scalar) is str:
            canonical.append((key, "str", scalar))
        elif scalar is None:
            canonical.append((key, "none", None))
        else:
            raise TypeError(
                f"run state diagnostic {key!r} must be a canonical scalar"
            )
    return tuple(canonical)


def _hartree_fock_run_snapshot_fingerprint(run: HartreeFockRun) -> str:
    state = run.state
    mu = _strict_real(state.mu, "run.state.mu")
    precision = _strict_real(state.precision, "run.state.precision", positive=True)
    if type(run.init_mode) is not str or type(run.seed) is not int:
        raise TypeError("run initializer metadata must be canonical scalars")
    if type(run.converged) is not bool or type(run.exit_reason) is not str:
        raise TypeError("run status metadata must be canonical scalars")
    return _fingerprint(
        {
            "state": {
                "h0": _array_sha256(state.h0),
                "density": _array_sha256(state.density),
                "hamiltonian": _array_sha256(state.hamiltonian),
                "energies": _array_sha256(state.energies),
                "mu": mu.hex(),
                "precision": precision.hex(),
                "diagnostics": _canonical_run_diagnostics(state.diagnostics),
            },
            "iter_energy": _array_sha256(run.iter_energy),
            "iter_err": _array_sha256(run.iter_err),
            "iter_oda": _array_sha256(run.iter_oda),
            "init_mode": run.init_mode,
            "seed": run.seed,
            "converged": run.converged,
            "exit_reason": run.exit_reason,
        }
    )


def _fingerprint(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _max_abs(value: object) -> float:
    array = np.asarray(value)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _flavor_indices(spin: int) -> tuple[int, int]:
    return tuple(
        index for index, (_valley, flavor_spin) in enumerate(INTERNAL_FLAVOR_ORDER)
        if flavor_spin == spin
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Vituri2024FiniteQSpiralChoice:
    """Independent finite-q, selected-spin, gauge, and numerical boundary choice."""

    q_inverse_angstrom: Array
    selected_spin: Literal[-1, 1] = 1
    gauge_mode: GaugeMode = "displayed_b3"
    b3_anchor_floor: float = VITURI2024_DEFAULT_B3_ANCHOR_FLOOR
    occupation_gap_floor_ev: float = VITURI2024_DEFAULT_SPIRAL_OCCUPATION_GAP_FLOOR_EV
    spin_block_tolerance_ev: float = VITURI2024_DEFAULT_SPIN_BLOCK_TOLERANCE_EV
    fingerprint: str = field(init=False)
    authority: str = field(default=VITURI2024_HF_SPIRAL_AUTHORITY, init=False)
    finite_square_no_wrap: bool = field(default=True, init=False)
    g0_only: bool = field(default=True, init=False)
    candidate_only: bool = field(default=True, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        q = np.asarray(self.q_inverse_angstrom)
        if q.dtype != np.dtype(np.float64) or q.shape != (2,) or not np.all(np.isfinite(q)):
            raise ValueError("q_inverse_angstrom must be finite float64 shape (2,)")
        spin = _strict_int(self.selected_spin, "selected_spin")
        if spin not in (-1, 1):
            raise ValueError("selected_spin must be -1 or +1")
        if self.gauge_mode not in ("identity", "displayed_b3"):
            raise ValueError("gauge_mode must be 'identity' or 'displayed_b3'")
        anchor_floor = _strict_real(self.b3_anchor_floor, "b3_anchor_floor", positive=True)
        gap_floor = _strict_real(
            self.occupation_gap_floor_ev, "occupation_gap_floor_ev", positive=True
        )
        block_tolerance = _strict_real(
            self.spin_block_tolerance_ev, "spin_block_tolerance_ev", positive=True
        )
        object.__setattr__(self, "q_inverse_angstrom", _readonly(q, np.dtype(np.float64)))
        object.__setattr__(self, "selected_spin", spin)
        object.__setattr__(self, "b3_anchor_floor", anchor_floor)
        object.__setattr__(self, "occupation_gap_floor_ev", gap_floor)
        object.__setattr__(self, "spin_block_tolerance_ev", block_tolerance)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_HF_SPIRAL_API_VERSION,
                "q": _array_sha256(self.q_inverse_angstrom),
                "selected_spin": self.selected_spin,
                "gauge_mode": self.gauge_mode,
                "b3_anchor_floor": self.b3_anchor_floor,
                "occupation_gap_floor_ev": self.occupation_gap_floor_ev,
                "spin_block_tolerance_ev": self.spin_block_tolerance_ev,
                "authority": self.authority,
                "flags": (
                    self.finite_square_no_wrap,
                    self.g0_only,
                    self.candidate_only,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        if (
            type(self.q_inverse_angstrom) is not np.ndarray
            or self.q_inverse_angstrom.dtype != np.dtype(np.float64)
            or self.q_inverse_angstrom.shape != (2,)
            or self.q_inverse_angstrom.flags.writeable
            or not np.all(np.isfinite(self.q_inverse_angstrom))
            or type(self.selected_spin) is not int
            or self.selected_spin not in (-1, 1)
            or self.gauge_mode not in ("identity", "displayed_b3")
            or type(self.b3_anchor_floor) is not float
            or not math.isfinite(self.b3_anchor_floor)
            or self.b3_anchor_floor <= 0.0
            or type(self.occupation_gap_floor_ev) is not float
            or not math.isfinite(self.occupation_gap_floor_ev)
            or self.occupation_gap_floor_ev <= 0.0
            or type(self.spin_block_tolerance_ev) is not float
            or not math.isfinite(self.spin_block_tolerance_ev)
            or self.spin_block_tolerance_ev <= 0.0
            or self.authority != VITURI2024_HF_SPIRAL_AUTHORITY
            or self.finite_square_no_wrap is not True
            or self.g0_only is not True
            or self.candidate_only is not True
            or self.production_ready is not False
            or self.paper_reproduction_verified is not False
            or self._current_fingerprint() != self.fingerprint
        ):
            raise ValueError("finite-q spiral choice live state drifted")


@dataclass(frozen=True, slots=True)
class Vituri2024DisplayedB3GaugeReceipt:
    """Receipt for the independent displayed-basis B3-positive gauge.

    ``gauged_states = source_to_b3_phase[:,None,:] * source_states``.  If a
    source column is rephased by ``exp(i theta)``, the recorded factor changes
    by ``exp(-i theta)`` and the B3-gauged state is unchanged.
    """

    source_to_b3_phase: Array
    anchor_magnitudes: Array
    anchor_floor: float
    fingerprint: str = field(init=False)
    basis: tuple[str, ...] = field(default=BASIS, init=False)
    b3_component_index: int = field(default=VITURI2024_B3_COMPONENT_INDEX, init=False)
    component_real_positive: bool = field(default=True, init=False)
    paper_gauge_established: bool = field(default=False, init=False)
    author_psi6_identification: bool = field(default=False, init=False)
    smooth_domain_established: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if BASIS.index("B3") != VITURI2024_B3_COMPONENT_INDEX:
            raise RuntimeError("displayed Vituri basis no longer puts B3 at index 1")
        phases = np.asarray(self.source_to_b3_phase)
        anchors = np.asarray(self.anchor_magnitudes)
        if (
            phases.dtype != np.dtype(np.complex128)
            or phases.ndim != 2
            or phases.shape[0] != 2
            or anchors.dtype != np.dtype(np.float64)
            or anchors.shape != phases.shape
            or not np.all(np.isfinite(phases))
            or not np.all(np.isfinite(anchors))
        ):
            raise ValueError("B3 gauge receipt arrays must have shapes (2,Nk)")
        floor = _strict_real(self.anchor_floor, "anchor_floor", positive=True)
        if np.any(anchors < floor):
            raise ValueError("B3 anchor is below the explicit gauge floor")
        if _max_abs(np.abs(phases) - 1.0) > 32.0 * np.finfo(np.float64).eps:
            raise ValueError("B3 source-to-gauge factors must be unit modulus")
        object.__setattr__(self, "source_to_b3_phase", _readonly(phases, np.dtype(np.complex128)))
        object.__setattr__(self, "anchor_magnitudes", _readonly(anchors, np.dtype(np.float64)))
        object.__setattr__(self, "anchor_floor", floor)
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self.validate_live_state()

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "basis": self.basis,
                "b3_component_index": self.b3_component_index,
                "phases": _array_sha256(self.source_to_b3_phase),
                "anchors": _array_sha256(self.anchor_magnitudes),
                "anchor_floor": self.anchor_floor,
                "component_real_positive": self.component_real_positive,
                "paper_gauge_established": self.paper_gauge_established,
                "author_psi6_identification": self.author_psi6_identification,
                "smooth_domain_established": self.smooth_domain_established,
            }
        )

    def validate_live_state(self) -> None:
        if (
            self.basis != BASIS
            or BASIS.index("B3") != self.b3_component_index
            or self.b3_component_index != 1
            or type(self.source_to_b3_phase) is not np.ndarray
            or self.source_to_b3_phase.dtype != np.dtype(np.complex128)
            or self.source_to_b3_phase.ndim != 2
            or self.source_to_b3_phase.shape[0] != 2
            or self.source_to_b3_phase.flags.writeable
            or not np.all(np.isfinite(self.source_to_b3_phase))
            or type(self.anchor_magnitudes) is not np.ndarray
            or self.anchor_magnitudes.dtype != np.dtype(np.float64)
            or self.anchor_magnitudes.shape != self.source_to_b3_phase.shape
            or self.anchor_magnitudes.flags.writeable
            or not np.all(np.isfinite(self.anchor_magnitudes))
            or type(self.anchor_floor) is not float
            or not math.isfinite(self.anchor_floor)
            or self.anchor_floor <= 0.0
            or np.any(self.anchor_magnitudes < self.anchor_floor)
            or _max_abs(np.abs(self.source_to_b3_phase) - 1.0)
            > 32.0 * np.finfo(np.float64).eps
            or self.component_real_positive is not True
            or self.paper_gauge_established is not False
            or self.author_psi6_identification is not False
            or self.smooth_domain_established is not False
            or self._current_fingerprint() != self.fingerprint
        ):
            raise ValueError("displayed-B3 gauge receipt live state drifted")


def apply_vituri2024_displayed_b3_gauge(
    source_states: Array,
    *,
    anchor_floor: float = VITURI2024_DEFAULT_B3_ANCHOR_FLOOR,
) -> tuple[Array, Vituri2024DisplayedB3GaugeReceipt]:
    """Gauge finite source columns by their displayed-basis B3 component."""

    states = np.asarray(source_states)
    if (
        states.dtype != np.dtype(np.complex128)
        or states.ndim != 3
        or states.shape[:2] != (2, 6)
        or states.shape[2] < 1
        or not np.all(np.isfinite(states))
    ):
        raise ValueError("source_states must be finite complex128 (2,6,Nk)")
    floor = _strict_real(anchor_floor, "anchor_floor", positive=True)
    anchors = states[:, VITURI2024_B3_COMPONENT_INDEX, :]
    magnitudes = np.asarray(np.abs(anchors), dtype=np.float64)
    if np.any(magnitudes < floor):
        minimum = float(np.min(magnitudes))
        raise ValueError(
            f"displayed B3 anchor {minimum:.17g} is below explicit floor {floor:.17g}"
        )
    phases = np.asarray(anchors.conj() / magnitudes, dtype=np.complex128)
    gauged = np.asarray(phases[:, None, :] * states, dtype=np.complex128)
    gauged[:, VITURI2024_B3_COMPONENT_INDEX, :] = magnitudes.astype(np.complex128)
    receipt = Vituri2024DisplayedB3GaugeReceipt(
        source_to_b3_phase=phases,
        anchor_magnitudes=magnitudes,
        anchor_floor=floor,
    )
    if (
        _max_abs(gauged[:, VITURI2024_B3_COMPONENT_INDEX, :].imag) != 0.0
        or np.any(gauged[:, VITURI2024_B3_COMPONENT_INDEX, :].real <= 0.0)
    ):
        raise RuntimeError("displayed-B3 positive-real gauge construction failed")
    return _readonly(gauged, np.dtype(np.complex128)), receipt


@dataclass(frozen=True, slots=True)
class Vituri2024PreparedHFSpiral:
    """Immutable shifted-state preparation bound to the unchanged dense functional."""

    base_prepared_fingerprint: str
    choice: Vituri2024FiniteQSpiralChoice
    holes_per_valley: int
    precision: float
    ordered_mesh: Array
    shifted_momenta_by_valley: Array
    active_band_states: Array
    active_band_energies_by_valley: Array
    h0_native: Array
    functional: Vituri2024TranslationalHFFunctional
    gauge_receipt: Vituri2024DisplayedB3GaugeReceipt | None
    minimum_lower_gap_ev: float
    minimum_upper_gap_ev: float
    fingerprint: str = field(init=False)
    authority: str = field(default=VITURI2024_HF_SPIRAL_AUTHORITY, init=False)
    candidate_only: bool = field(default=True, init=False)
    paper_gauge_established: bool = field(default=False, init=False)
    author_psi6_identification: bool = field(default=False, init=False)
    smooth_domain_established: bool = field(default=False, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if type(self.choice) is not Vituri2024FiniteQSpiralChoice:
            raise TypeError("spiral preparation requires an exact finite-q choice")
        self.choice.validate_live_state()
        holes = _strict_int(self.holes_per_valley, "holes_per_valley")
        precision = _strict_real(self.precision, "precision", positive=True)
        minimum_lower_gap = _strict_real(
            self.minimum_lower_gap_ev, "minimum_lower_gap_ev", positive=True
        )
        minimum_upper_gap = _strict_real(
            self.minimum_upper_gap_ev, "minimum_upper_gap_ev", positive=True
        )
        nk = int(np.asarray(self.ordered_mesh).shape[0])
        if holes < 1 or holes >= nk:
            raise ValueError("first spiral milestone requires 1<=H_v<Nk")
        object.__setattr__(self, "holes_per_valley", holes)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "minimum_lower_gap_ev", minimum_lower_gap)
        object.__setattr__(self, "minimum_upper_gap_ev", minimum_upper_gap)
        arrays = (
            (self.ordered_mesh, np.dtype(np.float64), (nk, 2), "ordered_mesh"),
            (
                self.shifted_momenta_by_valley,
                np.dtype(np.float64),
                (2, nk, 2),
                "shifted_momenta_by_valley",
            ),
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
                raise ValueError(f"prepared spiral {label} drifted")
        expected_shifted = np.stack(
            [
                self.ordered_mesh + 0.5 * valley * self.choice.q_inverse_angstrom
                for valley in ACTIVE_BAND_STATES_VALLEY_ORDER
            ],
            axis=0,
        )
        if not np.array_equal(self.shifted_momenta_by_valley, expected_shifted):
            raise ValueError("prepared spiral shifted momenta violate p_tau,k=k+tau*q/2")
        if type(self.functional) is not Vituri2024TranslationalHFFunctional:
            raise TypeError("spiral preparation requires the unchanged dense functional")
        if (
            self.functional.nk != nk
            or not np.array_equal(self.functional.ordered_mesh, self.ordered_mesh)
            or not np.array_equal(self.functional.active_band_states, self.active_band_states)
            or not np.array_equal(self.functional.h0_native, self.h0_native)
        ):
            raise ValueError("prepared spiral functional binding drifted")
        if self.choice.gauge_mode == "displayed_b3":
            if type(self.gauge_receipt) is not Vituri2024DisplayedB3GaugeReceipt:
                raise TypeError("displayed-B3 preparation requires a typed gauge receipt")
            self.gauge_receipt.validate_live_state()
            anchors = self.active_band_states[:, VITURI2024_B3_COMPONENT_INDEX, :]
            if _max_abs(anchors.imag) != 0.0 or np.any(anchors.real <= 0.0):
                raise ValueError("prepared displayed-B3 anchors are not positive real")
        elif self.gauge_receipt is not None:
            raise ValueError("identity-gauge preparation must not claim a B3 receipt")
        if (
            not math.isfinite(self.minimum_lower_gap_ev)
            or self.minimum_lower_gap_ev <= 0.0
            or not math.isfinite(self.minimum_upper_gap_ev)
            or self.minimum_upper_gap_ev <= 0.0
        ):
            raise ValueError("shifted active band must remain pointwise isolated")
        locked = (
            self.authority == VITURI2024_HF_SPIRAL_AUTHORITY,
            self.candidate_only is True,
            self.paper_gauge_established is False,
            self.author_psi6_identification is False,
            self.smooth_domain_established is False,
            self.production_ready is False,
            self.paper_reproduction_verified is False,
        )
        if not all(locked):
            raise ValueError("prepared spiral authority was inflated")
        object.__setattr__(self, "fingerprint", self._current_fingerprint())
        self.validate_live_state()

    @property
    def nk(self) -> int:
        return int(self.ordered_mesh.shape[0])

    @property
    def selected_rank(self) -> int:
        return 2 * self.nk - 2 * self.holes_per_valley

    def _current_fingerprint(self) -> str:
        return _fingerprint(
            {
                "api_version": VITURI2024_HF_SPIRAL_API_VERSION,
                "base": self.base_prepared_fingerprint,
                "choice": self.choice.fingerprint,
                "holes_per_valley": self.holes_per_valley,
                "precision": self.precision,
                "mesh": _array_sha256(self.ordered_mesh),
                "shifted": _array_sha256(self.shifted_momenta_by_valley),
                "states": _array_sha256(self.active_band_states),
                "energies": _array_sha256(self.active_band_energies_by_valley),
                "h0": _array_sha256(self.h0_native),
                "functional": self.functional.fingerprint,
                "gauge": None if self.gauge_receipt is None else self.gauge_receipt.fingerprint,
                "minimum_lower_gap_ev": self.minimum_lower_gap_ev,
                "minimum_upper_gap_ev": self.minimum_upper_gap_ev,
                "authority": self.authority,
                "flags": (
                    self.candidate_only,
                    self.paper_gauge_established,
                    self.author_psi6_identification,
                    self.smooth_domain_established,
                    self.production_ready,
                    self.paper_reproduction_verified,
                ),
            }
        )

    def validate_live_state(self) -> None:
        """Revalidate every nested binding before accepting the snapshot hash."""

        if type(self.choice) is not Vituri2024FiniteQSpiralChoice:
            raise TypeError("prepared spiral choice type drifted")
        self.choice.validate_live_state()
        if (
            type(self.base_prepared_fingerprint) is not str
            or len(self.base_prepared_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.base_prepared_fingerprint)
            or type(self.holes_per_valley) is not int
            or type(self.precision) is not float
            or not math.isfinite(self.precision)
            or self.precision <= 0.0
        ):
            raise ValueError("prepared spiral scalar metadata drifted")
        nk = self.nk
        if self.holes_per_valley < 1 or self.holes_per_valley >= nk:
            raise ValueError("prepared spiral selected rank left its finite domain")
        arrays = (
            (self.ordered_mesh, np.dtype(np.float64), (nk, 2), "ordered_mesh"),
            (
                self.shifted_momenta_by_valley,
                np.dtype(np.float64),
                (2, nk, 2),
                "shifted_momenta_by_valley",
            ),
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
                type(value) is not np.ndarray
                or value.dtype != dtype
                or value.shape != shape
                or value.flags.writeable
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"prepared spiral live {label} drifted")
        expected_shifted = np.stack(
            [
                self.ordered_mesh + 0.5 * valley * self.choice.q_inverse_angstrom
                for valley in ACTIVE_BAND_STATES_VALLEY_ORDER
            ],
            axis=0,
        )
        if not np.array_equal(self.shifted_momenta_by_valley, expected_shifted):
            raise ValueError("prepared spiral shifted-momentum identity drifted")
        valley_to_index = {
            valley: index for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        expected_h0 = np.zeros((4, 4, nk), dtype=np.complex128)
        for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
            expected_h0[flavor, flavor, :] = self.active_band_energies_by_valley[
                valley_to_index[valley]
            ]
        if not np.array_equal(self.h0_native, expected_h0):
            raise ValueError("prepared spiral energies/h0 binding drifted")
        if type(self.functional) is not Vituri2024TranslationalHFFunctional:
            raise TypeError("prepared spiral functional type drifted")
        self.functional.validate_live_state()
        if (
            self.functional.nk != nk
            or not np.array_equal(self.functional.ordered_mesh, self.ordered_mesh)
            or not np.array_equal(
                self.functional.active_band_states, self.active_band_states
            )
            or not np.array_equal(self.functional.h0_native, self.h0_native)
        ):
            raise ValueError("prepared spiral functional binding drifted")
        if self.choice.gauge_mode == "displayed_b3":
            if type(self.gauge_receipt) is not Vituri2024DisplayedB3GaugeReceipt:
                raise TypeError("prepared displayed-B3 gauge receipt type drifted")
            self.gauge_receipt.validate_live_state()
            anchors = self.active_band_states[:, VITURI2024_B3_COMPONENT_INDEX, :]
            if (
                self.gauge_receipt.anchor_floor != self.choice.b3_anchor_floor
                or not np.array_equal(
                    self.gauge_receipt.anchor_magnitudes, anchors.real
                )
                or _max_abs(anchors.imag) != 0.0
                or np.any(anchors.real <= 0.0)
            ):
                raise ValueError("prepared displayed-B3 gauge anchors drifted")
        elif self.gauge_receipt is not None:
            raise ValueError("prepared identity gauge acquired a B3 receipt")
        if (
            type(self.minimum_lower_gap_ev) is not float
            or not math.isfinite(self.minimum_lower_gap_ev)
            or self.minimum_lower_gap_ev <= 0.0
            or type(self.minimum_upper_gap_ev) is not float
            or not math.isfinite(self.minimum_upper_gap_ev)
            or self.minimum_upper_gap_ev <= 0.0
        ):
            raise ValueError("prepared spiral active-band gap drifted")
        if (
            self.authority != VITURI2024_HF_SPIRAL_AUTHORITY
            or self.candidate_only is not True
            or self.paper_gauge_established is not False
            or self.author_psi6_identification is not False
            or self.smooth_domain_established is not False
            or self.production_ready is not False
            or self.paper_reproduction_verified is not False
        ):
            raise ValueError("prepared spiral authority was inflated")
        if self._current_fingerprint() != self.fingerprint:
            raise ValueError("prepared spiral fingerprint drifted")


def prepare_vituri2024_hf_spiral(
    base: Vituri2024PreparedHomogeneousHF,
    choice: Vituri2024FiniteQSpiralChoice,
) -> Vituri2024PreparedHFSpiral:
    """Prepare the G=0 shifted basis while retaining base quadrature choices."""

    if type(base) is not Vituri2024PreparedHomogeneousHF:
        raise TypeError("base must be Vituri2024PreparedHomogeneousHF")
    if type(choice) is not Vituri2024FiniteQSpiralChoice:
        raise TypeError("choice must be Vituri2024FiniteQSpiralChoice")
    base.validate_live_state()
    choice.validate_live_state()
    if type(base.functional) is not Vituri2024TranslationalHFFunctional:
        raise TypeError("first spiral milestone requires the dense translational oracle")
    if base.spec.holes_per_valley >= base.spec.nk:
        raise ValueError("first spiral milestone requires an interior selected-spin rank")

    shifted = np.stack(
        [
            base.ordered_mesh + 0.5 * valley * choice.q_inverse_angstrom
            for valley in ACTIVE_BAND_STATES_VALLEY_ORDER
        ],
        axis=0,
    ).astype(np.float64, copy=False)
    is_exact_q0 = np.array_equal(choice.q_inverse_angstrom, np.zeros(2, dtype=np.float64))
    if is_exact_q0 and choice.gauge_mode == "identity":
        # This branch deliberately preserves the base preparation's source
        # eigenvector phase byte-for-byte.  It is the exact q=0 reduction gate.
        states = base.active_band_states
        energies = base.active_band_energies_by_valley
        h0 = base.h0_native
        functional = base.functional
        minimum_lower_gap = base.minimum_lower_gap_ev
        minimum_upper_gap = base.minimum_upper_gap_ev
        gauge_receipt = None
    else:
        source_states = np.empty((2, 6, base.spec.nk), dtype=np.complex128)
        energies_mutable = np.empty((2, base.spec.nk), dtype=np.float64)
        lower_gaps = np.empty((2, base.spec.nk), dtype=np.float64)
        upper_gaps = np.empty((2, base.spec.nk), dtype=np.float64)
        for valley_index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER):
            for momentum_index, momentum in enumerate(shifted[valley_index]):
                solution = third_lowest_active_band(momentum, valley, base.spec.delta1_ev)
                source_states[valley_index, :, momentum_index] = solution.eigenvector
                energies_mutable[valley_index, momentum_index] = solution.energy
                lower_gaps[valley_index, momentum_index] = solution.lower_gap
                upper_gaps[valley_index, momentum_index] = solution.upper_gap
        if choice.gauge_mode == "displayed_b3":
            states, gauge_receipt = apply_vituri2024_displayed_b3_gauge(
                source_states, anchor_floor=choice.b3_anchor_floor
            )
        else:
            states = _readonly(source_states, np.dtype(np.complex128))
            gauge_receipt = None
        energies = _readonly(energies_mutable, np.dtype(np.float64))
        h0_mutable = np.zeros((4, 4, base.spec.nk), dtype=np.complex128)
        valley_to_index = {
            valley: index for index, valley in enumerate(ACTIVE_BAND_STATES_VALLEY_ORDER)
        }
        for flavor, (valley, _spin) in enumerate(INTERNAL_FLAVOR_ORDER):
            h0_mutable[flavor, flavor, :] = energies[valley_to_index[valley], :]
        h0 = _readonly(h0_mutable, np.dtype(np.complex128))
        functional = Vituri2024TranslationalHFFunctional(
            ordered_mesh=base.ordered_mesh,
            active_band_states=states,
            h0_native=h0,
            normal_order_reference_native=base.functional.normal_order_reference_native,
            mesh_receipt=base.functional.mesh_receipt,
            interaction=base.functional.interaction,
            normal_order_reference_fingerprint=(
                base.functional.normal_order_reference_fingerprint
            ),
            q0_choice=base.functional.q0_choice,
            provenance=(
                "G=0 p_tau,k=k+tau*q/2 IVC-spiral candidate using the original "
                "finite square, area, kernel choice, reference, and q=0 policy."
            ),
        )
        if not np.array_equal(
            functional.kernel_by_mesh_pair,
            base.functional.kernel_by_mesh_pair,
        ):
            raise RuntimeError("shifted preparation changed the original central-mesh kernel")
        minimum_lower_gap = float(np.min(lower_gaps))
        minimum_upper_gap = float(np.min(upper_gaps))

    return Vituri2024PreparedHFSpiral(
        base_prepared_fingerprint=base.fingerprint,
        choice=choice,
        holes_per_valley=base.spec.holes_per_valley,
        precision=base.spec.precision,
        ordered_mesh=base.ordered_mesh,
        shifted_momenta_by_valley=_readonly(shifted, np.dtype(np.float64)),
        active_band_states=states,
        active_band_energies_by_valley=energies,
        h0_native=h0,
        functional=functional,
        gauge_receipt=gauge_receipt,
        minimum_lower_gap_ev=minimum_lower_gap,
        minimum_upper_gap_ev=minimum_upper_gap,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralOccupationBoundary:
    occupied_max_ev: float
    empty_min_ev: float
    gap_ev: float
    explicit_floor_ev: float
    roundoff_floor_ev: float
    effective_floor_ev: float


class Vituri2024SpiralOccupationBoundaryError(ValueError):
    """Typed fail-closed rejection of an exact or subtolerance selected boundary."""

    def __init__(self, boundary: Vituri2024SpiralOccupationBoundary) -> None:
        self.boundary = boundary
        super().__init__(
            "selected-spin global occupation boundary is exact or subtolerance: "
            f"gap={boundary.gap_ev:.17g} eV, floor={boundary.effective_floor_ev:.17g} eV"
        )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralDensityDiagnostics:
    selected_rank: int
    selected_valley_electron_populations: tuple[float, float]
    selected_valley_hole_populations: tuple[float, float]
    coherence_frobenius: float
    ivc_order_sum: complex
    boundary: Vituri2024SpiralOccupationBoundary
    spectator_full_residual: float
    projector_idempotency_residual: float
    hermiticity_residual: float
    spin_block_coupling_residual_ev: float


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralDensityMapResult:
    density_native: Array
    energies: Array
    mu: float
    diagnostics: Vituri2024SpiralDensityDiagnostics

    def as_core_update(self) -> DensityUpdateResult:
        d = self.diagnostics
        return DensityUpdateResult(
            density=self.density_native,
            energies=self.energies,
            mu=self.mu,
            observables={
                "selected_rank": float(d.selected_rank),
                "selected_valley_minus_population": d.selected_valley_electron_populations[0],
                "selected_valley_plus_population": d.selected_valley_electron_populations[1],
                "selected_coherence_frobenius": d.coherence_frobenius,
                "selected_ivc_order_sum_real": float(d.ivc_order_sum.real),
                "selected_ivc_order_sum_imag": float(d.ivc_order_sum.imag),
                "selected_boundary_lower_ev": d.boundary.occupied_max_ev,
                "selected_boundary_upper_ev": d.boundary.empty_min_ev,
                "selected_boundary_gap_ev": d.boundary.gap_ev,
            },
        )


def build_vituri2024_spiral_density_map(
    hamiltonian: Array,
    *,
    selected_spin: Literal[-1, 1],
    selected_rank: int,
    occupation_gap_floor_ev: float = VITURI2024_DEFAULT_SPIRAL_OCCUPATION_GAP_FLOOR_EV,
    spin_block_tolerance_ev: float = VITURI2024_DEFAULT_SPIN_BLOCK_TOLERANCE_EV,
) -> Vituri2024SpiralDensityMapResult:
    """Global selected-spin Aufbau map with exactly-full spectator flavors."""

    matrix = np.asarray(hamiltonian)
    if (
        matrix.dtype != np.dtype(np.complex128)
        or matrix.ndim != 3
        or matrix.shape[:2] != (4, 4)
        or matrix.shape[2] < 1
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("spiral Hamiltonian must be finite complex128 (4,4,Nk)")
    spin = _strict_int(selected_spin, "selected_spin")
    if spin not in (-1, 1):
        raise ValueError("selected_spin must be -1 or +1")
    nk = int(matrix.shape[2])
    rank = _strict_int(selected_rank, "selected_rank")
    if rank <= 0 or rank >= 2 * nk:
        raise ValueError("first spiral milestone requires 0<selected_rank<2*Nk")
    gap_floor = _strict_real(
        occupation_gap_floor_ev, "occupation_gap_floor_ev", positive=True
    )
    block_tolerance = _strict_real(
        spin_block_tolerance_ev, "spin_block_tolerance_ev", positive=True
    )
    hermiticity = _max_abs(matrix - matrix.swapaxes(0, 1).conj())
    if hermiticity > block_tolerance:
        raise ValueError("spiral Hamiltonian is materially non-Hermitian")
    selected = np.asarray(_flavor_indices(spin), dtype=np.int64)
    spectators = np.asarray(_flavor_indices(-spin), dtype=np.int64)
    cross = matrix[np.ix_(selected, spectators, np.arange(nk, dtype=np.int64))]
    cross_residual = _max_abs(cross)
    if cross_residual > block_tolerance:
        raise ValueError("forbidden selected/opposite-spin Hamiltonian coupling")

    selected_energies = np.empty((2, nk), dtype=np.float64)
    selected_vectors = np.empty((2, 2, nk), dtype=np.complex128)
    spectator_energies = np.empty((2, nk), dtype=np.float64)
    for momentum in range(nk):
        selected_block = matrix[:, :, momentum][np.ix_(selected, selected)]
        spectator_block = matrix[:, :, momentum][np.ix_(spectators, spectators)]
        values, vectors = np.linalg.eigh(selected_block)
        selected_energies[:, momentum] = values
        selected_vectors[:, :, momentum] = vectors
        spectator_energies[:, momentum] = np.linalg.eigvalsh(spectator_block)

    flat = selected_energies.reshape(-1, order="C")
    order = np.argsort(flat, kind="stable")
    occupied_max = float(flat[order[rank - 1]])
    empty_min = float(flat[order[rank]])
    gap = empty_min - occupied_max
    scale = max(1.0, float(np.max(np.abs(flat), initial=0.0)))
    roundoff_floor = 64.0 * np.finfo(np.float64).eps * scale
    effective_floor = max(gap_floor, roundoff_floor)
    boundary = Vituri2024SpiralOccupationBoundary(
        occupied_max_ev=occupied_max,
        empty_min_ev=empty_min,
        gap_ev=gap,
        explicit_floor_ev=gap_floor,
        roundoff_floor_ev=roundoff_floor,
        effective_floor_ev=effective_floor,
    )
    if gap <= effective_floor:
        raise Vituri2024SpiralOccupationBoundaryError(boundary)

    occupied_flat = np.zeros(2 * nk, dtype=np.bool_)
    occupied_flat[order[:rank]] = True
    occupied = occupied_flat.reshape((2, nk), order="C")
    conventional = np.zeros_like(matrix)
    conventional[np.ix_(spectators, spectators, np.arange(nk, dtype=np.int64))] = (
        np.eye(2, dtype=np.complex128)[:, :, None]
    )
    for momentum in range(nk):
        vectors = selected_vectors[:, :, momentum]
        weights = occupied[:, momentum].astype(np.float64)
        block = (vectors * weights) @ vectors.conj().T
        conventional[:, :, momentum][np.ix_(selected, selected)] = block

    selected_block_all = conventional[
        np.ix_(selected, selected, np.arange(nk, dtype=np.int64))
    ]
    valley_populations = tuple(
        float(np.sum(selected_block_all[index, index, :].real)) for index in range(2)
    )
    hole_populations = tuple(float(nk - value) for value in valley_populations)
    coherence = selected_block_all[0, 1, :]
    order_sum = complex(np.sum(conventional[selected[0], selected[1], :]))
    coherence_norm = float(np.linalg.norm(coherence))
    spectator_block_all = conventional[
        np.ix_(spectators, spectators, np.arange(nk, dtype=np.int64))
    ]
    spectator_residual = _max_abs(
        spectator_block_all - np.eye(2, dtype=np.complex128)[:, :, None]
    )
    projector_residual = 0.0
    for momentum in range(nk):
        p = conventional[:, :, momentum]
        projector_residual = max(projector_residual, _max_abs(p @ p - p))
    density_hermiticity = _max_abs(
        conventional - conventional.swapaxes(0, 1).conj()
    )
    trace = float(
        sum(np.trace(selected_block_all[:, :, momentum]).real for momentum in range(nk))
    )
    gate = 256.0 * np.finfo(np.float64).eps * max(1.0, float(nk))
    if (
        abs(trace - rank) > gate
        or spectator_residual != 0.0
        or projector_residual > 5.0e-12
        or density_hermiticity > 5.0e-12
    ):
        raise RuntimeError("spiral density map failed rank/projector/Hermiticity gates")

    energies = np.empty((4, nk), dtype=np.float64)
    energies[selected, :] = selected_energies
    energies[spectators, :] = spectator_energies
    native = _readonly(conventional.swapaxes(0, 1), np.dtype(np.complex128))
    diagnostics = Vituri2024SpiralDensityDiagnostics(
        selected_rank=rank,
        selected_valley_electron_populations=valley_populations,  # type: ignore[arg-type]
        selected_valley_hole_populations=hole_populations,  # type: ignore[arg-type]
        coherence_frobenius=coherence_norm,
        ivc_order_sum=order_sum,
        boundary=boundary,
        spectator_full_residual=spectator_residual,
        projector_idempotency_residual=projector_residual,
        hermiticity_residual=density_hermiticity,
        spin_block_coupling_residual_ev=cross_residual,
    )
    return Vituri2024SpiralDensityMapResult(
        density_native=native,
        energies=_readonly(energies, np.dtype(np.float64)),
        mu=0.5 * (occupied_max + empty_min),
        diagnostics=diagnostics,
    )


def _validate_spiral_density_structure(
    density_native: Array,
    *,
    selected_spin: int,
    selected_rank: int,
    tolerance: float,
    require_projector: bool,
) -> None:
    conventional = vituri2024_native_density_to_conventional_k_diagonal(
        np.asarray(density_native, dtype=np.complex128)
    )
    nk = conventional.shape[2]
    selected = np.asarray(_flavor_indices(selected_spin), dtype=np.int64)
    spectators = np.asarray(_flavor_indices(-selected_spin), dtype=np.int64)
    cross = conventional[np.ix_(selected, spectators, np.arange(nk, dtype=np.int64))]
    spectator = conventional[np.ix_(spectators, spectators, np.arange(nk, dtype=np.int64))]
    selected_block = conventional[np.ix_(selected, selected, np.arange(nk, dtype=np.int64))]
    trace = float(sum(np.trace(selected_block[:, :, k]).real for k in range(nk)))
    if (
        _max_abs(cross) > tolerance
        or _max_abs(spectator - np.eye(2, dtype=np.complex128)[:, :, None]) > tolerance
        or abs(trace - selected_rank) > tolerance * max(1.0, float(nk))
    ):
        raise ValueError("spiral density left selected-rank/full-spectator spin sector")
    if require_projector:
        residual = max(
            _max_abs(conventional[:, :, k] @ conventional[:, :, k] - conventional[:, :, k])
            for k in range(nk)
        )
        if residual > tolerance:
            raise ValueError("spiral initializer is not a momentum-local projector")


def make_vituri2024_spiral_initial_density(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    init_mode: SpiralInitializerMode,
    ivc_phase_radians: float = 0.0,
) -> Array:
    """Return an exact-rank normal or displayed-B3-gauge IVC projector."""

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be Vituri2024PreparedHFSpiral")
    prepared.validate_live_state()
    if init_mode not in ("normal", "ivc_b3"):
        raise ValueError("init_mode must be 'normal' or 'ivc_b3'")
    if init_mode == "ivc_b3" and prepared.choice.gauge_mode != "displayed_b3":
        raise ValueError("gauge-consistent IVC initializer requires displayed_b3 mode")
    phase = _strict_real(ivc_phase_radians, "ivc_phase_radians")
    nk = prepared.nk
    selected = np.asarray(_flavor_indices(prepared.choice.selected_spin), dtype=np.int64)
    spectators = np.asarray(_flavor_indices(-prepared.choice.selected_spin), dtype=np.int64)
    conventional = np.zeros((4, 4, nk), dtype=np.complex128)
    conventional[np.ix_(spectators, spectators, np.arange(nk, dtype=np.int64))] = (
        np.eye(2, dtype=np.complex128)[:, :, None]
    )
    conventional[np.ix_(selected, selected, np.arange(nk, dtype=np.int64))] = (
        np.eye(2, dtype=np.complex128)[:, :, None]
    )
    number_of_holes = 2 * prepared.holes_per_valley
    if init_mode == "normal":
        selected_h0 = np.asarray(
            [prepared.h0_native[index, index, :].real for index in selected],
            dtype=np.float64,
        )
        order = np.argsort(-selected_h0.reshape(-1, order="C"), kind="stable")
        for flat_index in order[:number_of_holes]:
            valley_index, momentum = np.unravel_index(
                int(flat_index), (2, nk), order="C"
            )
            conventional[selected[valley_index], selected[valley_index], momentum] = 0.0
    else:
        # First place one balanced coherent hole on distinct high-energy central
        # k points; only after exhausting Nk points place its orthogonal partner.
        # This is an initializer policy, not a claimed author many-start policy.
        mean_energy = np.mean(prepared.active_band_energies_by_valley, axis=0)
        momentum_order = np.argsort(-mean_energy, kind="stable")
        hole_spinors = (
            np.asarray([1.0, np.exp(1j * phase)], dtype=np.complex128) / math.sqrt(2.0),
            np.asarray([-np.exp(-1j * phase), 1.0], dtype=np.complex128) / math.sqrt(2.0),
        )
        remaining = number_of_holes
        for spinor in hole_spinors:
            for momentum in momentum_order:
                if remaining == 0:
                    break
                block = conventional[:, :, int(momentum)][np.ix_(selected, selected)]
                block -= np.outer(spinor, spinor.conj())
                conventional[:, :, int(momentum)][np.ix_(selected, selected)] = block
                remaining -= 1
            if remaining == 0:
                break
        if remaining != 0:
            raise RuntimeError("IVC initializer could not realize the selected hole rank")
    native = _readonly(conventional.swapaxes(0, 1), np.dtype(np.complex128))
    _validate_spiral_density_structure(
        native,
        selected_spin=prepared.choice.selected_spin,
        selected_rank=prepared.selected_rank,
        tolerance=5.0e-12,
        require_projector=True,
    )
    return native


def make_vituri2024_hf_spiral_state(
    prepared: Vituri2024PreparedHFSpiral,
) -> Vituri2024HFState:
    prepared.validate_live_state()
    h0 = np.asarray(prepared.h0_native, dtype=np.complex128).copy()
    return Vituri2024HFState(
        h0=h0,
        density=np.zeros_like(h0),
        hamiltonian=h0.copy(),
        energies=np.full((4, prepared.nk), np.nan, dtype=np.float64),
        mu=float("nan"),
        precision=prepared.precision,
        diagnostics={},
    )


def make_vituri2024_hf_spiral_problem(
    prepared: Vituri2024PreparedHFSpiral,
) -> HartreeFockProblem:
    """Compose the system adapter exclusively through the generic HF problem API."""

    if type(prepared) is not Vituri2024PreparedHFSpiral:
        raise TypeError("prepared must be Vituri2024PreparedHFSpiral")
    prepared.validate_live_state()
    interaction_action = prepared.functional.make_validated_interaction_action()

    def initializer(state: Vituri2024HFState, *, init_mode: str, seed: int) -> None:
        if _strict_int(seed, "seed") != 0:
            raise ValueError("deterministic first spiral milestone requires seed=0")
        if init_mode not in ("normal", "ivc_b3"):
            raise ValueError("unsupported spiral init_mode")
        state.density[:, :, :] = make_vituri2024_spiral_initial_density(
            prepared, init_mode=init_mode  # type: ignore[arg-type]
        )
        state.hamiltonian[:, :, :] = state.h0
        state.energies[:, :] = np.nan
        state.mu = float("nan")
        state.diagnostics.clear()

    def density_builder(hamiltonian: Array) -> DensityUpdateResult:
        return build_vituri2024_spiral_density_map(
            np.asarray(hamiltonian, dtype=np.complex128),
            selected_spin=prepared.choice.selected_spin,
            selected_rank=prepared.selected_rank,
            occupation_gap_floor_ev=prepared.choice.occupation_gap_floor_ev,
            spin_block_tolerance_ev=prepared.choice.spin_block_tolerance_ev,
        ).as_core_update()

    def energy_functional(_interaction_h: Array, _h0: Array, density: Array) -> float:
        return prepared.functional.energy(np.asarray(density, dtype=np.complex128))

    def hamiltonian_gate(hamiltonian: Array) -> None:
        matrix = np.asarray(hamiltonian)
        selected = np.asarray(_flavor_indices(prepared.choice.selected_spin), dtype=np.int64)
        spectators = np.asarray(_flavor_indices(-prepared.choice.selected_spin), dtype=np.int64)
        nk = prepared.nk
        cross = matrix[np.ix_(selected, spectators, np.arange(nk, dtype=np.int64))]
        if _max_abs(cross) > prepared.choice.spin_block_tolerance_ev:
            raise ValueError("forbidden spin-block coupling generated during SCF")

    def density_gate(density: Array) -> None:
        _validate_spiral_density_structure(
            np.asarray(density, dtype=np.complex128),
            selected_spin=prepared.choice.selected_spin,
            selected_rank=prepared.selected_rank,
            tolerance=5.0e-10,
            require_projector=False,
        )

    return HartreeFockProblem(
        initializer=initializer,
        kernel=HartreeFockKernel(
            interaction_builder=interaction_action,
            density_builder=density_builder,
            energy_functional=energy_functional,
            oda_delta_interaction_builder=interaction_action,
            hamiltonian_postprocessor=hamiltonian_gate,
            density_postprocessor=density_gate,
            convergence_rule="raw",
        ),
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralIVCObservable:
    """Displayed-B3 candidate normalized by the total hole density ``2 H_v/A``.

    A balanced coherent hole contributes at most one half to the off-diagonal
    valley density, so this normalization has the algebraic ceiling ``1/2``.
    Whether this is the paper's plotted normalization remains unresolved.
    """

    phi_inverse_angstrom_squared: complex
    total_hole_density_inverse_angstrom_squared: float
    absolute_phi_over_total_hole_density: float
    order_sum: complex
    gauge_receipt_fingerprint: str
    normalization: str = field(default="total_hole_density_2Hv_over_A", init=False)
    maximum_balanced_coherent_ratio: float = field(default=0.5, init=False)
    independent_displayed_b3_gauge_candidate: bool = field(default=True, init=False)
    paper_gauge_established: bool = field(default=False, init=False)
    author_psi6_identification: bool = field(default=False, init=False)
    paper_normalization_resolved: bool = field(default=False, init=False)
    paper_normalization_authority_established: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.validate_live_state()

    def validate_live_state(self) -> None:
        roundoff = 128.0 * np.finfo(np.float64).eps
        if (
            type(self.phi_inverse_angstrom_squared) is not complex
            or not math.isfinite(self.phi_inverse_angstrom_squared.real)
            or not math.isfinite(self.phi_inverse_angstrom_squared.imag)
            or type(self.total_hole_density_inverse_angstrom_squared) is not float
            or not math.isfinite(self.total_hole_density_inverse_angstrom_squared)
            or self.total_hole_density_inverse_angstrom_squared <= 0.0
            or type(self.absolute_phi_over_total_hole_density) is not float
            or not math.isfinite(self.absolute_phi_over_total_hole_density)
            or self.absolute_phi_over_total_hole_density < 0.0
            or self.absolute_phi_over_total_hole_density
            > self.maximum_balanced_coherent_ratio + roundoff
            or type(self.order_sum) is not complex
            or not math.isfinite(self.order_sum.real)
            or not math.isfinite(self.order_sum.imag)
            or type(self.gauge_receipt_fingerprint) is not str
            or len(self.gauge_receipt_fingerprint) != 64
            or self.normalization != "total_hole_density_2Hv_over_A"
            or self.maximum_balanced_coherent_ratio != 0.5
            or self.independent_displayed_b3_gauge_candidate is not True
            or self.paper_gauge_established is not False
            or self.author_psi6_identification is not False
            or self.paper_normalization_resolved is not False
            or self.paper_normalization_authority_established is not False
        ):
            raise ValueError("spiral IVC observable live state drifted")


def measure_vituri2024_spiral_ivc_order(
    prepared: Vituri2024PreparedHFSpiral,
    density_native: Array,
) -> Vituri2024SpiralIVCObservable:
    """Measure the main-text single-q IVC operator in the independent B3 gauge."""

    prepared.validate_live_state()
    if prepared.choice.gauge_mode != "displayed_b3" or prepared.gauge_receipt is None:
        raise ValueError("|phi|/|n| is reported only in the independent displayed-B3 gauge")
    density = np.asarray(density_native, dtype=np.complex128)
    conventional = vituri2024_native_density_to_conventional_k_diagonal(density)
    selected = _flavor_indices(prepared.choice.selected_spin)
    # rho[+, -] = conventional[-, +] in the native rho_ab=<c_a^dagger c_b> convention.
    order_sum = complex(np.sum(conventional[selected[0], selected[1], :]))
    area = prepared.functional.mesh_receipt.area_angstrom_squared
    carrier_density = 2.0 * prepared.holes_per_valley / area
    phi = order_sum / area
    ratio = abs(phi) / carrier_density
    roundoff = 128.0 * np.finfo(np.float64).eps
    if ratio > 0.5 + roundoff:
        raise ValueError("IVC order exceeds the balanced coherent-hole ceiling 1/2")
    return Vituri2024SpiralIVCObservable(
        phi_inverse_angstrom_squared=complex(phi),
        total_hole_density_inverse_angstrom_squared=float(carrier_density),
        absolute_phi_over_total_hole_density=float(ratio),
        order_sum=order_sum,
        gauge_receipt_fingerprint=prepared.gauge_receipt.fingerprint,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024SpiralStationarityDiagnostics:
    selected_rank: int
    selected_rank_residual: float
    spectator_full_residual: float
    spin_block_coupling_residual_ev: float
    density_hermiticity_residual: float
    density_idempotency_residual: float
    commutator_residual_ev: float
    final_raw_norm: float
    boundary: Vituri2024SpiralOccupationBoundary
    stationary: bool


def diagnose_vituri2024_hf_spiral_stationarity(
    prepared: Vituri2024PreparedHFSpiral,
    run: HartreeFockRun,
) -> tuple[Vituri2024SpiralStationarityDiagnostics, Vituri2024SpiralDensityMapResult]:
    """Recompute the final Fock and its selected-sector ground-state map afresh."""

    if not isinstance(run, HartreeFockRun):
        raise TypeError("run must be HartreeFockRun")
    density_native = np.asarray(run.state.density, dtype=np.complex128)
    fresh_fock = prepared.functional.fock(density_native)
    fresh_map = build_vituri2024_spiral_density_map(
        np.asarray(fresh_fock, dtype=np.complex128),
        selected_spin=prepared.choice.selected_spin,
        selected_rank=prepared.selected_rank,
        occupation_gap_floor_ev=prepared.choice.occupation_gap_floor_ev,
        spin_block_tolerance_ev=prepared.choice.spin_block_tolerance_ev,
    )
    conventional = vituri2024_native_density_to_conventional_k_diagonal(density_native)
    selected = np.asarray(_flavor_indices(prepared.choice.selected_spin), dtype=np.int64)
    spectators = np.asarray(_flavor_indices(-prepared.choice.selected_spin), dtype=np.int64)
    nk = prepared.nk
    selected_block = conventional[np.ix_(selected, selected, np.arange(nk, dtype=np.int64))]
    spectator_block = conventional[
        np.ix_(spectators, spectators, np.arange(nk, dtype=np.int64))
    ]
    cross = fresh_fock[np.ix_(selected, spectators, np.arange(nk, dtype=np.int64))]
    selected_trace = float(sum(np.trace(selected_block[:, :, k]).real for k in range(nk)))
    idempotency = 0.0
    commutator = 0.0
    for momentum in range(nk):
        p = conventional[:, :, momentum]
        f = fresh_fock[:, :, momentum]
        idempotency = max(idempotency, _max_abs(p @ p - p))
        commutator = max(commutator, _max_abs(f @ p - p @ f))
    rank_residual = abs(selected_trace - prepared.selected_rank)
    spectator_residual = _max_abs(
        spectator_block - np.eye(2, dtype=np.complex128)[:, :, None]
    )
    hermiticity = _max_abs(conventional - conventional.swapaxes(0, 1).conj())
    final_raw_norm = _max_abs(fresh_map.density_native - density_native)
    tolerance = max(1.0e-8, float(run.state.precision))
    stationary = (
        run.converged
        and rank_residual <= tolerance * max(1.0, float(nk))
        and spectator_residual <= tolerance
        and _max_abs(cross) <= prepared.choice.spin_block_tolerance_ev
        and hermiticity <= tolerance
        and idempotency <= tolerance
        and commutator <= tolerance
        and final_raw_norm <= tolerance
    )
    return (
        Vituri2024SpiralStationarityDiagnostics(
            selected_rank=prepared.selected_rank,
            selected_rank_residual=rank_residual,
            spectator_full_residual=spectator_residual,
            spin_block_coupling_residual_ev=_max_abs(cross),
            density_hermiticity_residual=hermiticity,
            density_idempotency_residual=idempotency,
            commutator_residual_ev=commutator,
            final_raw_norm=final_raw_norm,
            boundary=fresh_map.diagnostics.boundary,
            stationary=stationary,
        ),
        fresh_map,
    )


@dataclass(frozen=True, slots=True)
class Vituri2024HFSpiralRunResult:
    """Snapshot hashes and fresh diagnostics around a mutable core HF run."""

    prepared_fingerprint: str
    init_mode: SpiralInitializerMode
    run: HartreeFockRun
    final_fock_map_diagnostics: Vituri2024SpiralDensityDiagnostics
    stationarity: Vituri2024SpiralStationarityDiagnostics
    ivc_order: Vituri2024SpiralIVCObservable | None
    final_recomputed_energy_ev: float
    final_density_sha256: str
    fresh_map_density_sha256: str
    fresh_fock_sha256: str
    run_snapshot_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_snapshot_fingerprint",
            _hartree_fock_run_snapshot_fingerprint(self.run),
        )
    candidate_only: bool = field(default=True, init=False)
    production_ready: bool = field(default=False, init=False)
    paper_reproduction_verified: bool = field(default=False, init=False)

    def validate_live_state(self, prepared: Vituri2024PreparedHFSpiral) -> None:
        if (
            _hartree_fock_run_snapshot_fingerprint(self.run)
            != self.run_snapshot_fingerprint
        ):
            raise ValueError("spiral core HF run snapshot drifted")
        """Detect stale result claims after mutation of the nested core run."""

        if type(prepared) is not Vituri2024PreparedHFSpiral:
            raise TypeError("result validation requires exact spiral preparation")
        prepared.validate_live_state()
        hashes = (
            self.final_density_sha256,
            self.fresh_map_density_sha256,
            self.fresh_fock_sha256,
        )
        if (
            self.prepared_fingerprint != prepared.fingerprint
            or self.init_mode not in ("normal", "ivc_b3")
            or type(self.run) is not HartreeFockRun
            or self.run.init_mode != self.init_mode
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
            or self.candidate_only is not True
            or self.production_ready is not False
            or self.paper_reproduction_verified is not False
        ):
            raise ValueError("spiral run-result metadata drifted")
        density = np.asarray(self.run.state.density)
        if _array_sha256(density) != self.final_density_sha256:
            raise ValueError("spiral final-density snapshot is stale")
        fresh_fock = prepared.functional.fock(
            np.asarray(self.run.state.density, dtype=np.complex128)
        )
        if _array_sha256(fresh_fock) != self.fresh_fock_sha256:
            raise ValueError("spiral fresh-Fock snapshot is stale")
        stationarity, fresh_map = diagnose_vituri2024_hf_spiral_stationarity(
            prepared, self.run
        )
        if (
            _array_sha256(fresh_map.density_native)
            != self.fresh_map_density_sha256
            or fresh_map.diagnostics != self.final_fock_map_diagnostics
            or stationarity != self.stationarity
            or (self.run.converged and not stationarity.stationary)
        ):
            raise ValueError("spiral fresh-map/stationarity snapshot is stale")
        recomputed_energy = prepared.functional.energy(
            np.asarray(self.run.state.density, dtype=np.complex128)
        )
        recorded_energy = _strict_real(
            self.run.state.diagnostics.get("hf_energy"),
            "run.state.diagnostics['hf_energy']",
        )
        scale = max(abs(recomputed_energy), abs(recorded_energy))
        if (
            abs(recomputed_energy - recorded_energy) > 1.0e-10 * scale
            or recomputed_energy != self.final_recomputed_energy_ev
        ):
            raise ValueError("spiral recomputed/core scalar energy parity drifted")
        expected_order = (
            measure_vituri2024_spiral_ivc_order(prepared, self.run.state.density)
            if prepared.choice.gauge_mode == "displayed_b3"
            else None
        )
        if expected_order != self.ivc_order:
            raise ValueError("spiral IVC observable snapshot is stale")


def run_vituri2024_hf_spiral(
    prepared: Vituri2024PreparedHFSpiral,
    *,
    init_mode: SpiralInitializerMode,
    seed: int = 0,
    max_iter: int = 300,
    oda_stall_threshold: float = 1.0e-6,
    max_oda_lambda: float = 1.0,
) -> Vituri2024HFSpiralRunResult:
    """Run only through :func:`run_hartree_fock_problem`; no local SCF loop."""

    validated_max_iter = _strict_int(max_iter, "max_iter")
    if validated_max_iter <= 0:
        raise ValueError("max_iter must be positive")
    state = make_vituri2024_hf_spiral_state(prepared)
    problem = make_vituri2024_hf_spiral_problem(prepared)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=_strict_int(seed, "seed"),
        max_iter=validated_max_iter,
        oda_stall_threshold=_strict_real(
            oda_stall_threshold, "oda_stall_threshold", positive=True
        ),
        max_oda_lambda=_strict_real(max_oda_lambda, "max_oda_lambda", positive=True),
    )
    stationarity, fresh_map = diagnose_vituri2024_hf_spiral_stationarity(prepared, run)
    if run.converged and not stationarity.stationary:
        run = replace(run, converged=False, exit_reason="final_map_not_converged")
        stationarity, fresh_map = diagnose_vituri2024_hf_spiral_stationarity(
            prepared, run
        )
    recomputed_energy = prepared.functional.energy(run.state.density)
    recorded_energy = _strict_real(
        run.state.diagnostics.get("hf_energy"),
        "run.state.diagnostics['hf_energy']",
    )
    energy_scale = max(abs(recomputed_energy), abs(recorded_energy))
    if abs(recomputed_energy - recorded_energy) > 1.0e-10 * energy_scale:
        raise ValueError(
            "freshly recomputed spiral energy disagrees with core final energy"
        )
    order = (
        measure_vituri2024_spiral_ivc_order(prepared, run.state.density)
        if prepared.choice.gauge_mode == "displayed_b3"
        else None
    )
    fresh_fock = prepared.functional.fock(run.state.density)
    result = Vituri2024HFSpiralRunResult(
        prepared_fingerprint=prepared.fingerprint,
        init_mode=init_mode,
        run=run,
        final_fock_map_diagnostics=fresh_map.diagnostics,
        stationarity=stationarity,
        ivc_order=order,
        final_recomputed_energy_ev=recomputed_energy,
        final_density_sha256=_array_sha256(run.state.density),
        fresh_map_density_sha256=_array_sha256(fresh_map.density_native),
        fresh_fock_sha256=_array_sha256(fresh_fock),
    )
    result.validate_live_state(prepared)
    return result


def transform_vituri2024_native_density_under_band_rephasing(
    native_density: Array,
    phases_by_flavor_and_k: Array,
) -> Array:
    """Apply ``rho'_ab=e^{i theta_a} rho_ab e^{-i theta_b}``."""

    density = np.asarray(native_density)
    phases = np.asarray(phases_by_flavor_and_k)
    if (
        density.dtype != np.dtype(np.complex128)
        or density.ndim != 3
        or density.shape[:2] != (4, 4)
        or phases.dtype != np.dtype(np.complex128)
        or phases.shape != (4, density.shape[2])
        or _max_abs(np.abs(phases) - 1.0) > 5.0e-12
    ):
        raise ValueError("density/phases rephasing inputs have invalid shape or modulus")
    result = phases[:, None, :] * density * phases.conj()[None, :, :]
    return _readonly(result, np.dtype(np.complex128))


def transform_vituri2024_native_operator_under_band_rephasing(
    native_operator: Array,
    phases_by_flavor_and_k: Array,
) -> Array:
    """Apply ``h'_ab=e^{-i theta_a} h_ab e^{i theta_b}``."""

    operator = np.asarray(native_operator)
    phases = np.asarray(phases_by_flavor_and_k)
    if (
        operator.dtype != np.dtype(np.complex128)
        or operator.ndim != 3
        or operator.shape[:2] != (4, 4)
        or phases.dtype != np.dtype(np.complex128)
        or phases.shape != (4, operator.shape[2])
        or _max_abs(np.abs(phases) - 1.0) > 5.0e-12
    ):
        raise ValueError("operator/phases rephasing inputs have invalid shape or modulus")
    result = phases.conj()[:, None, :] * operator * phases[None, :, :]
    return _readonly(result, np.dtype(np.complex128))


__all__ = [
    "VITURI2024_B3_COMPONENT_INDEX",
    "VITURI2024_HF_SPIRAL_API_VERSION",
    "VITURI2024_HF_SPIRAL_AUTHORITY",
    "Vituri2024DisplayedB3GaugeReceipt",
    "Vituri2024FiniteQSpiralChoice",
    "Vituri2024HFSpiralRunResult",
    "Vituri2024PreparedHFSpiral",
    "Vituri2024SpiralDensityDiagnostics",
    "Vituri2024SpiralDensityMapResult",
    "Vituri2024SpiralIVCObservable",
    "Vituri2024SpiralOccupationBoundary",
    "Vituri2024SpiralOccupationBoundaryError",
    "Vituri2024SpiralStationarityDiagnostics",
    "apply_vituri2024_displayed_b3_gauge",
    "build_vituri2024_spiral_density_map",
    "diagnose_vituri2024_hf_spiral_stationarity",
    "make_vituri2024_hf_spiral_problem",
    "make_vituri2024_hf_spiral_state",
    "make_vituri2024_spiral_initial_density",
    "measure_vituri2024_spiral_ivc_order",
    "prepare_vituri2024_hf_spiral",
    "run_vituri2024_hf_spiral",
    "transform_vituri2024_native_density_under_band_rephasing",
    "transform_vituri2024_native_operator_under_band_rephasing",
]
