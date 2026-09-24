from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence
import math

import numpy as np

from analysis.response_derivative_gauge import (
    GeneralizedDerivativeData,
    HamiltonianGaugeData,
    PairGeneralizedDerivativeData,
    berry_connection_generalized_derivative,
    berry_connection_generalized_derivative_pair,
    energy_difference_inverse,
    hamiltonian_gauge_data,
    matrix_in_eigenbasis,
)

Axis = int
Component = tuple[Axis, Axis, Axis]
OpticalSymmetrization = Literal["none", "sum", "average"]


@dataclass(frozen=True)
class ShiftCurrentConvention:
    """Response-layer convention bundle.

    The derivative route is *not* configurable here: reusable code uses
    ``analysis.response_derivative_gauge`` and its WannierBerri Hamiltonian-gauge
    convention.  This dataclass records only response/plotting choices that
    differ between references: ordered vs symmetrized optical indices, the sign
    of the local geometric kernel, and whether the optical Lorentzian includes
    the normalizing ``1/pi``.
    """

    name: str
    optical_symmetrization: OpticalSymmetrization
    geometric_sign: float = 1.0
    normalized_lorentzian: bool = True
    description: str = ""


# Joya 2025 point audits showed that this ordered same-polarization kernel
# matches the explicit paper Eq.(7) c-sum directly, before omitted global
# conductivity prefactors, spin factor, SI conversion, and final colorbar sign.
JOYA_EQ7_GEOMETRIC_CONVENTION = ShiftCurrentConvention(
    name="joya2025_eq7_geometric",
    optical_symmetrization="none",
    geometric_sign=1.0,
    normalized_lorentzian=False,
    description="Ordered local pair kernel equals Joya 2025 Eq.(7) c-sum; optical Lorentzian has no 1/pi.",
)

# WannierBerri dynamic.py::ShiftCurrentFormula symmetrizes optical indices and
# its internal Imn has the opposite sign from the local pair product.  For b==c
# this is the audited relation Imn = -2 * ordered_pair_kernel.
WANNIERBERRI_INTERNAL_IMN_CONVENTION = ShiftCurrentConvention(
    name="wannierberri_internal_imn",
    optical_symmetrization="sum",
    geometric_sign=-1.0,
    normalized_lorentzian=True,
    description="Line-by-line convention of WannierBerri ShiftCurrentFormula internal Imn.",
)

HTG_LEGACY_CONVENTION = ShiftCurrentConvention(
    name="htg_legacy_sum",
    optical_symmetrization="sum",
    geometric_sign=1.0,
    normalized_lorentzian=True,
    description="Legacy hTG workspace convention: symmetrized optical product before the -i prefactor.",
)

E_CHARGE_C = 1.602176634e-19
HBAR_J_S = 1.054571817e-34
HBAR_EV_S = 6.582119569e-16
KB_EV_PER_K = 8.617333262145e-5

# Historical unsigned repository scale. It remains public for explicit
# Joya/legacy conversions, but it is not by itself a complete response
# convention and must not be guessed for WannierBerri positive-only sums.
SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 = math.pi * E_CHARGE_C**2 / HBAR_J_S * 1.0e6

# Fixed WannierBerri uses +pi*e^2/(4*hbar) before summing both ordered band
# pairs with f(E2)-f(E1). For each positive-energy initial->final transition,
# Imn[final,initial] = -Imn[initial,final], so those two terms reduce to
# -2*(f_initial-f_final)*Imn[initial,final]. Hence a positive-transition Imn
# integral is converted by -pi*e^2/(2*hbar), not by historical +pi*e^2/hbar.
WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2 = (
    -0.5 * SHIFT_CURRENT_PREFAC_UA_NM_PER_V2
)

_AXIS_LABELS: Mapping[Any, int] = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2, 0: 0, 1: 1, 2: 2}
_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class ShiftCurrentComponent:
    """Rank-3 shift-current component ``sigma^a_{bc}``."""

    current_axis: int
    optical_axis_1: int
    optical_axis_2: int

    @property
    def as_tuple(self) -> Component:
        return (int(self.current_axis), int(self.optical_axis_1), int(self.optical_axis_2))

    @property
    def compact_label(self) -> str:
        return "".join(_AXIS_NAMES[i] if 0 <= i < len(_AXIS_NAMES) else str(i) for i in self.as_tuple)

    @property
    def semicolon_label(self) -> str:
        a, b, c = self.as_tuple
        label = lambda i: _AXIS_NAMES[i] if 0 <= i < len(_AXIS_NAMES) else str(i)
        return f"{label(a)};{label(b)}{label(c)}"


def axis_index(axis: str | int) -> int:
    """Return an integer axis index for ``x/y/z`` or ``0/1/2`` labels."""

    key: str | int = axis.strip().lower() if isinstance(axis, str) else axis
    try:
        return int(_AXIS_LABELS[key])
    except KeyError as exc:
        raise ValueError(f"Unsupported axis {axis!r}; expected x/y/z or 0/1/2") from exc


def component_from_any(component: ShiftCurrentComponent | Sequence[int] | str) -> ShiftCurrentComponent:
    """Normalize a component label/tuple/dataclass to ``ShiftCurrentComponent``."""

    if isinstance(component, ShiftCurrentComponent):
        return component
    if isinstance(component, str):
        return parse_component(component)
    if len(component) != 3:  # type: ignore[arg-type]
        raise ValueError(f"Component must have three axes, got {component!r}")
    a, b, c = component  # type: ignore[misc]
    return ShiftCurrentComponent(axis_index(a), axis_index(b), axis_index(c))


def parse_component(text: str) -> ShiftCurrentComponent:
    """Parse labels such as ``'xxx'``, ``'x;yy'``, or ``'x_yy'``."""

    raw = str(text).strip().lower().replace("σ", "").replace("^", "").replace("_", ";").replace(" ", "")
    if ";" in raw:
        left, right = raw.split(";", 1)
        if len(left) != 1 or len(right) != 2:
            raise ValueError(f"Component must look like 'xxx' or 'x;yy', got {text!r}")
        return ShiftCurrentComponent(axis_index(left), axis_index(right[0]), axis_index(right[1]))
    if len(raw) == 3:
        return ShiftCurrentComponent(axis_index(raw[0]), axis_index(raw[1]), axis_index(raw[2]))
    raise ValueError(f"Component must look like 'xxx' or 'x;yy', got {text!r}")


def component_label(component: ShiftCurrentComponent | Sequence[int] | str, *, style: Literal["compact", "semicolon"] = "semicolon") -> str:
    comp = component_from_any(component)
    return comp.compact_label if style == "compact" else comp.semicolon_label


def _operator_array(operators: Sequence[np.ndarray] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(operators, dtype=np.complex128)
    if arr.ndim < 3 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"{name} must have shape (...,basis,basis), got {arr.shape}")
    return arr


def velocity_matrices(eigenvectors: np.ndarray, dhdk: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Return ``<u_n|partial_a H|u_m>`` with shape ``(ndim,nb,nb)``."""

    raw = _operator_array(dhdk, name="dhdk")
    if raw.ndim != 3:
        raise ValueError(f"dhdk must have shape (ndim,basis,basis), got {raw.shape}")
    return matrix_in_eigenbasis(eigenvectors, raw)


def second_derivative_matrices(eigenvectors: np.ndarray, d2hdk: Sequence[Sequence[np.ndarray]] | np.ndarray) -> np.ndarray:
    """Return ``<u_n|partial_a partial_b H|u_m>`` as ``(ndim,ndim,nb,nb)``."""

    raw = _operator_array(d2hdk, name="d2hdk")
    if raw.ndim != 4 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"d2hdk must have shape (ndim,ndim,basis,basis), got {raw.shape}")
    return matrix_in_eigenbasis(eigenvectors, raw)


def berry_connection_matrix(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
) -> np.ndarray:
    """Return interband Berry connection ``A^a_nm = -i V^a_nm/(E_n-E_m)``.

    This is exactly the ``external_terms=False`` WannierBerri Hamiltonian-gauge
    convention ``A_H = i D_H`` with ``D_H = -V/(E_n-E_m)``.
    """

    V = np.asarray(velocity_h, dtype=np.complex128)
    e = np.asarray(energies_ev, dtype=float)
    if V.ndim != 3 or V.shape[1:] != (e.size, e.size):
        raise ValueError(f"velocity_h must have shape (ndim,nb,nb), got {V.shape} for nb={e.size}")
    inv = energy_difference_inverse(e, cutoff=float(denominator_cutoff_ev))
    return 1.0j * (-V * inv[None, :, :])


def generalized_derivative_from_velocity(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
) -> GeneralizedDerivativeData:
    """Generic full-band generalized derivative of the Berry connection."""

    return berry_connection_generalized_derivative(
        velocity_h,
        energies_ev,
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=second_velocity_h,
        principal_value_eta=principal_value_eta_ev,
    )


@dataclass(frozen=True)
class ShiftCurrentTensors:
    """Reusable one-k-point ingredients for shift-current calculations."""

    energies_ev: np.ndarray
    occupations: np.ndarray
    velocity_h: np.ndarray
    berry_connection: np.ndarray
    berry_connection_gen_derivative: np.ndarray
    skipped_small_denominators: int
    second_velocity_h: np.ndarray | None = None
    gauge_data: HamiltonianGaugeData | None = None

    @property
    def D(self) -> np.ndarray:
        return self.velocity_h

    @property
    def r(self) -> np.ndarray:
        return self.berry_connection

    @property
    def r_covariant(self) -> np.ndarray:
        return self.berry_connection_gen_derivative


def fermi_occupation(energies_ev: np.ndarray, *, mu_ev: float = 0.0, temperature_k: float = 0.0) -> np.ndarray:
    """Fermi occupation for energies in eV."""

    energies = np.asarray(energies_ev, dtype=float)
    if float(temperature_k) <= 0.0:
        return (energies < float(mu_ev)).astype(float)
    beta_arg = (energies - float(mu_ev)) / (KB_EV_PER_K * float(temperature_k))
    out = np.empty_like(beta_arg, dtype=float)
    out[beta_arg > 40.0] = 0.0
    out[beta_arg < -40.0] = 1.0
    mask = (beta_arg >= -40.0) & (beta_arg <= 40.0)
    out[mask] = 1.0 / (np.exp(beta_arg[mask]) + 1.0)
    return out


def precompute_shift_current_tensors(
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    dhdk: Sequence[np.ndarray] | np.ndarray,
    *,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
    denominator_cutoff_ev: float = 1.0e-10,
    d2hdk: Sequence[Sequence[np.ndarray]] | np.ndarray | None = None,
    external_connection: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
) -> ShiftCurrentTensors:
    """Build full one-k-point shift-current tensors from a system adapter.

    A system only provides eigenpairs and Hamiltonian derivatives.  The
    Hamiltonian-gauge rotation and generalized derivative are delegated to
    ``analysis.response_derivative_gauge``.
    """

    if external_connection is not None:
        raise NotImplementedError(
            "external_connection is not supported in analysis.shift_current yet: "
            "the generalized derivative of external position/Berry terms must be implemented first. "
            "Use response_derivative_gauge directly for audited external_terms work."
        )
    raw_dh = _operator_array(dhdk, name="dhdk")
    raw_d2 = None if d2hdk is None else _operator_array(d2hdk, name="d2hdk")
    gauge = hamiltonian_gauge_data(
        energies_ev,
        eigenvectors,
        raw_dh,
        denominator_cutoff=float(denominator_cutoff_ev),
        d2hdk=raw_d2,
        external_connection=external_connection,
    )
    gd = generalized_derivative_from_velocity(
        gauge.velocity_h,
        gauge.energies,
        denominator_cutoff_ev=float(denominator_cutoff_ev),
        second_velocity_h=gauge.second_velocity_h,
        principal_value_eta_ev=principal_value_eta_ev,
    )
    return ShiftCurrentTensors(
        energies_ev=gauge.energies,
        occupations=fermi_occupation(gauge.energies, mu_ev=mu_ev, temperature_k=temperature_k),
        velocity_h=gauge.velocity_h,
        berry_connection=gauge.berry_connection,
        berry_connection_gen_derivative=gd.values,
        skipped_small_denominators=gd.skipped_small_denominators,
        second_velocity_h=gauge.second_velocity_h,
        gauge_data=gauge,
    )


def _validate_optical_symmetrization(value: OpticalSymmetrization) -> OpticalSymmetrization:
    if value not in ("none", "sum", "average"):
        raise ValueError(f"optical_symmetrization must be 'none', 'sum', or 'average', got {value!r}")
    return value


def component_amplitude_from_pair(
    berry_connection: np.ndarray,
    pair_gen_derivative: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> complex:
    """Return the complex geometric product for one direct transition.

    For ``sigma^a_{bc}``, the ordered product is
    ``A^b_mn (A^c_nm)_;a`` for transition ``n -> m``.  Optical-axis
    symmetrization and geometric sign may be supplied explicitly or through a
    named convention such as ``JOYA_EQ7_GEOMETRIC_CONVENTION``.
    """

    comp = component_from_any(component)
    if convention is not None:
        optical_symmetrization = convention.optical_symmetrization
    sym = _validate_optical_symmetrization(optical_symmetrization)
    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(pair_gen_derivative, dtype=np.complex128)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"berry_connection must have shape (ndim,nb,nb), got {A.shape}")
    if G.shape != (A.shape[0], A.shape[0]):
        raise ValueError(f"pair_gen_derivative has shape {G.shape}, expected {(A.shape[0], A.shape[0])}")
    n = int(initial_band)
    m = int(final_band)
    a, b, c = comp.as_tuple
    ordered = A[b, m, n] * G[a, c]
    if sym == "none":
        total = ordered
    else:
        swapped = A[c, m, n] * G[a, b]
        total = ordered + swapped
        if sym == "average":
            total *= 0.5
    if convention is not None:
        total *= float(convention.geometric_sign)
    return complex(total)


def component_kernel_from_pair(
    berry_connection: np.ndarray,
    pair_gen_derivative: np.ndarray,
    *,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> float:
    """Return the Abelian labeled-pair kernel ``Im[component_amplitude]``.

    This scalar is gauge invariant for a nondegenerate pair (independent U(1)
    phases). It is not invariant under arbitrary rotations of bands inside a
    degenerate manifold. Use :func:`component_group_trace_kernel` for an
    exact-degenerate transition between complete subspaces.
    """

    return float(
        np.imag(
            component_amplitude_from_pair(
                berry_connection,
                pair_gen_derivative,
                initial_band=initial_band,
                final_band=final_band,
                component=component,
                optical_symmetrization=optical_symmetrization,
                convention=convention,
            )
        )
    )


def _validated_band_group(group: Iterable[int], nb: int, *, name: str) -> np.ndarray:
    indices = np.asarray(tuple(int(index) for index in group), dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{name} must contain at least one band index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate band indices: {indices.tolist()}")
    if np.any(indices < 0) or np.any(indices >= int(nb)):
        raise ValueError(f"{name} indices {indices.tolist()} are outside [0,{int(nb)})")
    return indices


def component_group_trace_amplitude(
    berry_connection: np.ndarray,
    berry_connection_gen_derivative: np.ndarray,
    *,
    initial_group: Iterable[int],
    final_group: Iterable[int],
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> complex:
    r"""Return the proper subspace trace for one group-to-group transition.

    For initial subspace ``I``, final subspace ``F``, and component
    :math:`\sigma^a_{bc}`, the ordered amplitude is

    .. math::

       \operatorname{Tr}_{I}\!\left[(A^c_{;a})_{IF} A^b_{FI}\right].

    If ``A_FI -> U_F^dagger A_FI U_I`` and its generalized derivative
    ``G_IF -> U_I^dagger G_IF U_F``, the product transforms by similarity in
    ``I`` and this trace is invariant. This is the matrix-product meaning of
    WannierBerri ``ShiftCurrentFormula.trace_ln``; it is not a claim that its
    individual labeled-band summands are invariant.

    A single spectral transition may use this contraction only when every
    band in each group has the same energy and occupation. Arbitrary rotations
    mixing nondegenerate bands are not gauge changes and are outside this
    contract. Use :func:`exact_degenerate_group_transition_weight` for strict
    spectral validation and transition-energy bookkeeping.
    """

    comp = component_from_any(component)
    if convention is not None:
        optical_symmetrization = convention.optical_symmetrization
    sym = _validate_optical_symmetrization(optical_symmetrization)
    A = np.asarray(berry_connection, dtype=np.complex128)
    G = np.asarray(berry_connection_gen_derivative, dtype=np.complex128)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(f"berry_connection must have shape (ndim,nb,nb), got {A.shape}")
    ndim, nb, _ = A.shape
    if G.shape != (ndim, ndim, nb, nb):
        raise ValueError(f"berry_connection_gen_derivative has shape {G.shape}, expected {(ndim, ndim, nb, nb)}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(G)):
        raise ValueError("berry_connection and berry_connection_gen_derivative must be finite")
    inn = _validated_band_group(initial_group, nb, name="initial_group")
    out = _validated_band_group(final_group, nb, name="final_group")
    if np.intersect1d(inn, out).size:
        raise ValueError("initial_group and final_group must be disjoint")
    a, b, c = comp.as_tuple
    if min(a, b, c) < 0 or max(a, b, c) >= ndim:
        raise ValueError(f"component {comp.as_tuple} is incompatible with ndim={ndim}")

    def block_trace(connection_axis: int, optical_axis: int) -> complex:
        gen_if = G[a, connection_axis][np.ix_(inn, out)]
        connection_fi = A[optical_axis][np.ix_(out, inn)]
        return complex(np.trace(gen_if @ connection_fi))

    total = block_trace(c, b)
    if sym != "none":
        total += block_trace(b, c)
        if sym == "average":
            total *= 0.5
    if convention is not None:
        total *= float(convention.geometric_sign)
    return complex(total)


def component_group_trace_kernel(
    berry_connection: np.ndarray,
    berry_connection_gen_derivative: np.ndarray,
    *,
    initial_group: Iterable[int],
    final_group: Iterable[int],
    component: ShiftCurrentComponent | Sequence[int] | str,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> float:
    """Return ``Im`` of :func:`component_group_trace_amplitude`."""

    return float(
        np.imag(
            component_group_trace_amplitude(
                berry_connection,
                berry_connection_gen_derivative,
                initial_group=initial_group,
                final_group=final_group,
                component=component,
                optical_symmetrization=optical_symmetrization,
                convention=convention,
            )
        )
    )


@dataclass(frozen=True)
class ExactDegenerateGroupTransition:
    """One strict exact-degenerate group transition.

    ``geometric_amplitude`` is the unoccupied-weighted group trace, while
    ``weight`` includes the common occupation difference. ``kernel`` is
    ``Im(weight)``, the real local response accumulated into a spectrum by the
    current conventions.
    """

    initial_group: tuple[int, ...]
    final_group: tuple[int, ...]
    initial_energy_ev: float
    final_energy_ev: float
    transition_energy_ev: float
    occupation_difference: float
    geometric_amplitude: complex
    weight: complex
    kernel: float


def exact_degenerate_group_transition_weight(
    tensors: ShiftCurrentTensors,
    initial_group: Iterable[int],
    final_group: Iterable[int],
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
    intragroup_velocity_atol: float = 1.0e-10,
) -> ExactDegenerateGroupTransition:
    """Build a strict exact-degenerate group transition and trace weight.

    Each supplied group must be the complete maximal eigenspace at its energy;
    energies and occupations must be bitwise equal inside it. For the local
    bandwise generalized-derivative builder, scalar projected first-derivative
    blocks are a necessary first-order condition for a smooth persistent
    degeneracy. This condition is checked with ``intragroup_velocity_atol`` in
    the caller's derivative units; the caller must still establish persistence
    beyond one k point.

    The strict energy check is deliberate: replacing distinct pair transition
    energies by one cluster-average energy is an additional approximation and
    does not make arbitrary rotations of nondegenerate bands a gauge symmetry.
    Near-degenerate cluster spectra therefore remain outside this API.
    """

    energies = np.asarray(tensors.energies_ev, dtype=float)
    occupations = np.asarray(tensors.occupations, dtype=float)
    velocity_h = np.asarray(tensors.velocity_h, dtype=np.complex128)
    if energies.ndim != 1 or occupations.shape != energies.shape:
        raise ValueError(f"energies/occupations must have matching 1D shape, got {energies.shape} and {occupations.shape}")
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(occupations)):
        raise ValueError("energies and occupations must be finite")
    if velocity_h.ndim != 3 or velocity_h.shape[1:] != (energies.size, energies.size):
        raise ValueError(f"velocity_h must have shape (ndim,{energies.size},{energies.size}), got {velocity_h.shape}")
    if not np.all(np.isfinite(velocity_h)):
        raise ValueError("velocity_h must be finite")
    velocity_atol = float(intragroup_velocity_atol)
    if not np.isfinite(velocity_atol) or velocity_atol < 0.0:
        raise ValueError(f"intragroup_velocity_atol must be finite and non-negative, got {intragroup_velocity_atol}")

    inn = _validated_band_group(initial_group, energies.size, name="initial_group")
    out = _validated_band_group(final_group, energies.size, name="final_group")
    if np.intersect1d(inn, out).size:
        raise ValueError("initial_group and final_group must be disjoint")

    initial_energy = float(energies[inn[0]])
    final_energy = float(energies[out[0]])
    initial_occupation = float(occupations[inn[0]])
    final_occupation = float(occupations[out[0]])
    if not np.all(energies[inn] == initial_energy):
        raise ValueError(f"initial_group is not exactly degenerate: {energies[inn].tolist()}")
    if not np.all(energies[out] == final_energy):
        raise ValueError(f"final_group is not exactly degenerate: {energies[out].tolist()}")
    maximal_initial = np.flatnonzero(energies == initial_energy)
    maximal_final = np.flatnonzero(energies == final_energy)
    if not np.array_equal(np.sort(inn), maximal_initial):
        raise ValueError(f"initial_group must contain the complete exact eigenspace {maximal_initial.tolist()}, got {inn.tolist()}")
    if not np.array_equal(np.sort(out), maximal_final):
        raise ValueError(f"final_group must contain the complete exact eigenspace {maximal_final.tolist()}, got {out.tolist()}")
    if not np.all(occupations[inn] == initial_occupation):
        raise ValueError(f"initial_group occupations are not identical: {occupations[inn].tolist()}")
    if not np.all(occupations[out] == final_occupation):
        raise ValueError(f"final_group occupations are not identical: {occupations[out].tolist()}")

    for name, indices in (("initial_group", inn), ("final_group", out)):
        block = velocity_h[:, indices][:, :, indices]
        scalar = np.trace(block, axis1=1, axis2=2) / indices.size
        expected = scalar[:, None, None] * np.eye(indices.size, dtype=np.complex128)[None, :, :]
        residual = float(np.max(np.abs(block - expected)))
        if residual > velocity_atol:
            raise ValueError(
                f"{name} fails the necessary first-order exact-block condition: "
                f"max non-scalar velocity residual {residual} exceeds {velocity_atol}"
            )

    transition_energy = final_energy - initial_energy
    if not np.isfinite(transition_energy) or transition_energy <= 0.0:
        raise ValueError(f"group transition must have finite positive E_final-E_initial, got {transition_energy}")

    geometric = component_group_trace_amplitude(
        tensors.berry_connection,
        tensors.berry_connection_gen_derivative,
        initial_group=inn,
        final_group=out,
        component=component,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    occupation_difference = initial_occupation - final_occupation
    weight = complex(occupation_difference * geometric)
    return ExactDegenerateGroupTransition(
        initial_group=tuple(int(index) for index in inn),
        final_group=tuple(int(index) for index in out),
        initial_energy_ev=initial_energy,
        final_energy_ev=final_energy,
        transition_energy_ev=float(transition_energy),
        occupation_difference=float(occupation_difference),
        geometric_amplitude=geometric,
        weight=weight,
        kernel=float(np.imag(weight)),
    )


@dataclass(frozen=True)
class PairTransitionKernel:
    """Selected-pair geometric kernel using the full virtual/intermediate sum."""

    amplitude: complex
    kernel: float
    generalized_derivative: PairGeneralizedDerivativeData

    @property
    def skipped_small_denominators(self) -> int:
        return int(self.generalized_derivative.skipped_small_denominators)


@dataclass(frozen=True)
class PairTransitionWeight:
    """Occupation-weighted selected-pair result using the full virtual sum."""

    weight: complex
    kernel: float
    generalized_derivative: PairGeneralizedDerivativeData

    @property
    def skipped_small_denominators(self) -> int:
        return int(self.generalized_derivative.skipped_small_denominators)


def component_kernel_from_gauge_pair(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    berry_connection: np.ndarray,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> PairTransitionKernel:
    """Compute one raw geometric transition kernel from Hamiltonian-gauge data.

    This is the selected-pair/full-virtual-band path used by Joya-style
    heatmaps: it avoids constructing the full generalized-derivative tensor but
    still sums intermediate states over the full supplied basis.
    """

    n = int(initial_band)
    m = int(final_band)
    pair = berry_connection_generalized_derivative_pair(
        velocity_h,
        energies_ev,
        n,
        m,
        denominator_cutoff=float(denominator_cutoff_ev),
        second_velocity_h=second_velocity_h,
        principal_value_eta=principal_value_eta_ev,
    )
    amp = component_amplitude_from_pair(
        berry_connection,
        pair.values,
        initial_band=n,
        final_band=m,
        component=component,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    return PairTransitionKernel(amplitude=amp, kernel=float(np.imag(amp)), generalized_derivative=pair)


def component_transition_weight_from_gauge_pair(
    velocity_h: np.ndarray,
    energies_ev: np.ndarray,
    berry_connection: np.ndarray,
    occupations: np.ndarray,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    denominator_cutoff_ev: float = 1.0e-10,
    second_velocity_h: np.ndarray | None = None,
    principal_value_eta_ev: float | None = None,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> PairTransitionWeight:
    """Compute one occupation-weighted transition without a full GD tensor."""

    n = int(initial_band)
    m = int(final_band)
    raw = component_kernel_from_gauge_pair(
        velocity_h,
        energies_ev,
        berry_connection,
        n,
        m,
        component,
        denominator_cutoff_ev=denominator_cutoff_ev,
        second_velocity_h=second_velocity_h,
        principal_value_eta_ev=principal_value_eta_ev,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    fnm = float(np.asarray(occupations, dtype=float)[n] - np.asarray(occupations, dtype=float)[m])
    weight = complex(fnm * raw.amplitude)
    return PairTransitionWeight(weight=weight, kernel=float(np.imag(weight)), generalized_derivative=raw.generalized_derivative)


def component_transition_weight(
    tensors: ShiftCurrentTensors,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> complex:
    """Return an occupation-weighted labeled-band transition amplitude.

    This is the ordinary Abelian band-resolved formula. Inside an exact
    degenerate manifold only a complete group trace is U(N)-invariant; use
    :func:`exact_degenerate_group_transition_weight` there.
    """

    n = int(initial_band)
    m = int(final_band)
    G = np.asarray(tensors.berry_connection_gen_derivative, dtype=np.complex128)
    if G.ndim != 4 or G.shape[2:] != (tensors.energies_ev.size, tensors.energies_ev.size):
        raise ValueError(f"berry_connection_gen_derivative must have shape (ndim,ndim,nb,nb), got {G.shape}")
    amp = component_amplitude_from_pair(
        tensors.berry_connection,
        G[:, :, n, m],
        initial_band=n,
        final_band=m,
        component=component,
        optical_symmetrization=optical_symmetrization,
        convention=convention,
    )
    fnm = float(tensors.occupations[n] - tensors.occupations[m])
    return complex(fnm * amp)


def positive_transition_pairs(
    energies_ev: np.ndarray,
    occupations: np.ndarray | None = None,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
) -> list[tuple[int, int, float, float]]:
    """Return ``(n,m,E_m-E_n,f_n-f_m)`` for positive direct transitions."""

    energies = np.asarray(energies_ev, dtype=float)
    occ = None if occupations is None else np.asarray(occupations, dtype=float)
    if occ is not None and occ.shape != energies.shape:
        raise ValueError(f"occupations has shape {occ.shape}, expected {energies.shape}")
    bands = list(range(energies.size)) if selected_bands is None else [int(i) for i in selected_bands]
    out: list[tuple[int, int, float, float]] = []
    for n in bands:
        for m in bands:
            if n == m:
                continue
            transition = float(energies[m] - energies[n])
            if transition <= float(min_transition_ev):
                continue
            if max_transition_ev is not None and transition > float(max_transition_ev):
                continue
            fnm = 1.0 if occ is None else float(occ[n] - occ[m])
            if occ is not None and abs(fnm) < float(min_abs_occupation_diff):
                continue
            out.append((n, m, transition, fnm))
    return out


def positive_transition_terms(
    tensors: ShiftCurrentTensors,
    component: ShiftCurrentComponent | Sequence[int] | str,
    *,
    selected_bands: Iterable[int] | None = None,
    min_transition_ev: float = 0.0,
    max_transition_ev: float | None = None,
    min_abs_occupation_diff: float = 1.0e-14,
    optical_symmetrization: OpticalSymmetrization = "sum",
    convention: ShiftCurrentConvention | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ordinary labeled-band transition terms for one k point.

    The result is appropriate for nondegenerate bands (up to U(1) phases). It
    must not be interpreted as a U(N)-invariant decomposition inside exact
    degenerate blocks; aggregate such blocks with the strict group API.
    """

    transitions: list[float] = []
    weights: list[complex] = []
    for n, m, transition_ev, _fnm in positive_transition_pairs(
        tensors.energies_ev,
        tensors.occupations,
        selected_bands=selected_bands,
        min_transition_ev=min_transition_ev,
        max_transition_ev=max_transition_ev,
        min_abs_occupation_diff=min_abs_occupation_diff,
    ):
        weight = component_transition_weight(
            tensors,
            n,
            m,
            component,
            optical_symmetrization=optical_symmetrization,
            convention=convention,
        )
        if np.isfinite(weight.real) and np.isfinite(weight.imag):
            transitions.append(float(transition_ev))
            weights.append(complex(weight))
    return np.asarray(transitions, dtype=float), np.asarray(weights, dtype=np.complex128)


def lorentzian_delta(
    photon_energies_ev: np.ndarray,
    transition_energy_ev: float,
    eta_ev: float,
    *,
    normalized: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> np.ndarray:
    """Lorentzian delta replacement in eV units.

    ``normalized=True`` returns ``(eta/pi)/(x^2+eta^2)``.  Joya's plotting text
    uses ``eta/(x^2+eta^2)``; pass ``JOYA_EQ7_GEOMETRIC_CONVENTION`` to make
    that choice explicit.
    """

    if convention is not None:
        normalized = bool(convention.normalized_lorentzian)
    photon = np.asarray(photon_energies_ev, dtype=float)
    eta = float(eta_ev)
    if eta <= 0.0:
        raise ValueError(f"eta_ev must be positive, got {eta_ev}")
    diff = photon - float(transition_energy_ev)
    out = eta / (diff * diff + eta * eta)
    if normalized:
        out = out / math.pi
    return out


def add_transitions_to_integral(
    integral: np.ndarray,
    photon_energies_ev: np.ndarray,
    transition_energies_ev: np.ndarray,
    transition_weights: np.ndarray,
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
    normalized_lorentzian: bool = True,
    include_bz_factor: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> None:
    """Accumulate weighted transitions into a spectrum/integral in-place."""

    if np.asarray(transition_energies_ev).size == 0:
        return
    factor = float(k_weight_nm_inv_sq)
    if include_bz_factor:
        factor /= (2.0 * math.pi) ** 2
    for transition_ev, weight in zip(np.asarray(transition_energies_ev, dtype=float), np.asarray(transition_weights, dtype=np.complex128), strict=True):
        integral += factor * weight * lorentzian_delta(
            photon_energies_ev,
            float(transition_ev),
            eta_ev,
            normalized=bool(normalized_lorentzian),
            convention=convention,
        )


def wannierberri_positive_transition_conductivity_from_imn(
    imn_integral: np.ndarray | float,
) -> np.ndarray:
    r"""Convert a positive-transition WannierBerri ``Imn`` integral.

    The input must contain every positive-energy initial-to-final transition
    exactly once, weighted by ``f_initial-f_final``, with the real local tensor
    ``ShiftCurrentFormula.Imn[initial,final]``. For an ideal positive-frequency
    delta function the spectral factor is
    ``delta(E_final-E_initial-hbar*omega)``. To reproduce fixed WannierBerri's
    finite-width calculator exactly, include both its resonant and antiresonant
    broadened orientations in ``imn_integral`` before calling this helper.

    The returned conversion is ``[-pi e^2/(2 hbar)] * integral`` in the
    repository's ``microA*nm/V^2`` sheet-unit convention. The minus sign and
    factor one half are the exact reduction of WannierBerri's two ordered group
    pairs; they are not a plotting or material-specific sign.
    """

    values = np.asarray(imn_integral)
    if np.iscomplexobj(values) and np.any(np.imag(values) != 0.0):
        raise ValueError(
            "imn_integral must be the real WannierBerri internal Imn integral; "
            "take the explicitly audited imaginary part of a complex amplitude first"
        )
    return float(WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2) * np.asarray(
        np.real(values), dtype=float
    )



def conductivity_from_integral(
    integral: np.ndarray,
    *,
    prefactor: float,
    phase: complex,
) -> np.ndarray:
    """Apply an explicit caller-chosen conversion to an integrated kernel."""

    return np.real(complex(phase) * float(prefactor) * np.asarray(integral))


def spectra_from_transition_table(
    photon_energies_ev: np.ndarray,
    transition_table: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    k_weight_nm_inv_sq: float,
    eta_ev: float,
    prefactor: float,
    prefactor_phase: complex,
    normalized_lorentzian: bool = True,
    include_bz_factor: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> dict[str, np.ndarray]:
    """Build spectra using an explicit global prefactor and phase."""

    spectra: dict[str, np.ndarray] = {}
    for name, (transition_energies, weights) in transition_table.items():
        integral = np.zeros_like(photon_energies_ev, dtype=np.complex128)
        add_transitions_to_integral(
            integral,
            photon_energies_ev,
            np.asarray(transition_energies, dtype=float),
            np.asarray(weights, dtype=np.complex128),
            k_weight_nm_inv_sq=k_weight_nm_inv_sq,
            eta_ev=eta_ev,
            normalized_lorentzian=normalized_lorentzian,
            include_bz_factor=include_bz_factor,
            convention=convention,
        )
        spectra[name] = conductivity_from_integral(
            integral,
            prefactor=prefactor,
            phase=prefactor_phase,
        )
    return spectra


def fermi_window_indices(fermi_grid_ev: np.ndarray, initial_energy_ev: float, final_energy_ev: float) -> tuple[int, int]:
    """Return the Fermi-grid interval where ``initial <= E_F < final``."""

    grid = np.asarray(fermi_grid_ev, dtype=float)
    lo = min(float(initial_energy_ev), float(final_energy_ev))
    hi = max(float(initial_energy_ev), float(final_energy_ev))
    start = int(np.searchsorted(grid, lo, side="left"))
    end = int(np.searchsorted(grid, hi, side="left"))
    return max(0, min(start, grid.size)), max(0, min(end, grid.size))


def accumulate_fermi_omega_heatmap(
    heatmap: np.ndarray,
    fermi_grid_ev: np.ndarray,
    photon_energies_ev: np.ndarray,
    *,
    initial_energy_ev: float,
    final_energy_ev: float,
    transition_energy_ev: float,
    amplitude: complex | float,
    eta_ev: float,
    k_weight: float = 1.0,
    normalized_lorentzian: bool = True,
    convention: ShiftCurrentConvention | None = None,
) -> bool:
    """Accumulate one transition into an ``(E_F, omega)`` heatmap."""

    start, end = fermi_window_indices(fermi_grid_ev, initial_energy_ev, final_energy_ev)
    if start >= end:
        return False
    broaden = lorentzian_delta(
        photon_energies_ev,
        transition_energy_ev,
        eta_ev,
        normalized=normalized_lorentzian,
        convention=convention,
    )
    heatmap[start:end, :] += float(k_weight) * amplitude * broaden[None, :]
    return True


__all__ = [
    "Axis",
    "Component",
    "E_CHARGE_C",
    "ExactDegenerateGroupTransition",
    "HBAR_EV_S",
    "HBAR_J_S",
    "HTG_LEGACY_CONVENTION",
    "JOYA_EQ7_GEOMETRIC_CONVENTION",
    "KB_EV_PER_K",
    "OpticalSymmetrization",
    "PairTransitionKernel",
    "PairTransitionWeight",
    "SHIFT_CURRENT_PREFAC_UA_NM_PER_V2",
    "ShiftCurrentComponent",
    "ShiftCurrentConvention",
    "ShiftCurrentTensors",
    "WANNIERBERRI_INTERNAL_IMN_CONVENTION",
    "WANNIERBERRI_POSITIVE_TRANSITION_PREFAC_UA_NM_PER_V2",
    "accumulate_fermi_omega_heatmap",
    "add_transitions_to_integral",
    "axis_index",
    "berry_connection_matrix",
    "component_amplitude_from_pair",
    "component_from_any",
    "component_group_trace_amplitude",
    "component_group_trace_kernel",
    "component_kernel_from_gauge_pair",
    "component_kernel_from_pair",
    "component_label",
    "component_transition_weight",
    "component_transition_weight_from_gauge_pair",
    "conductivity_from_integral",
    "exact_degenerate_group_transition_weight",
    "fermi_occupation",
    "fermi_window_indices",
    "generalized_derivative_from_velocity",
    "lorentzian_delta",
    "parse_component",
    "positive_transition_pairs",
    "positive_transition_terms",
    "precompute_shift_current_tensors",
    "second_derivative_matrices",
    "spectra_from_transition_table",
    "velocity_matrices",
    "wannierberri_positive_transition_conductivity_from_imn",
]
