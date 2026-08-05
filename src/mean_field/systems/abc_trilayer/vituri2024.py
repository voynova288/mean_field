"""Isolated noninteracting ABC-trilayer adapter for Vituri et al. (2024).

Only the six-band Hamiltonian and its third-lowest-band projector are
implemented.  The interaction, Hartree--Fock, and TDHF calculations are not
implemented or claimed here because the pinned paper authority leaves the
items in :data:`VITURI2024_SPEC` unresolved.

Physics authority
-----------------
arXiv:2408.10309v1, archive-relative ``SM.tex``, especially Eq. ``Ham6``
(lines 20--35) and the model discussion in lines 9--132.  Authority is bound
by hashes below; no temporary extraction path is a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray


BASIS: Final[tuple[str, ...]] = ("A1", "B3", "B1", "A2", "B2", "A3")
ACTIVE_BAND_INDEX_ZERO_BASED: Final[int] = 2

ARXIV_IDENTIFIER: Final[str] = "2408.10309v1"
SM_TEX_AUTHORITY_PATH: Final[str] = "SM.tex"
PDF_AUTHORITY_PATH: Final[str] = "reference/2408.10309v1.pdf"
ARXIV_SOURCE_SHA256: Final[str] = (
    "c01d805c463e388989370a202f04f4f27ceb38a668294e1959e172c8fc9932f9"
)
ARXIV_PDF_SHA256: Final[str] = (
    "ec761a2b494a8e5983ff3fb6cfb842e114526cc0ba8b3e7cdc7c128f5d204bc8"
)
SM_TEX_SHA256: Final[str] = (
    "f2847fa3dc14590f4157dd82ac6983ace39328a620f55cf75d4db51f1a43be45"
)


@dataclass(frozen=True, slots=True)
class Vituri2024Parameters:
    """Pinned Eq. ``Ham6`` parameters.

    ``a0`` is in Angstrom; all hopping and onsite parameters are in eV.
    The frozen dataclass prevents mutation after construction.
    """

    a0: float = 2.46
    gamma0: float = 3.1
    gamma1: float = 0.38
    gamma2: float = -0.022
    gamma3: float = -0.29
    gamma4: float = -0.21
    delta: float = -0.0105
    Delta2: float = -0.0023

    def __post_init__(self) -> None:
        values = (
            self.a0,
            self.gamma0,
            self.gamma1,
            self.gamma2,
            self.gamma3,
            self.gamma4,
            self.delta,
            self.Delta2,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("all Vituri2024 parameters must be finite")
        if tuple(float(value) for value in values) != (
            2.46,
            3.1,
            0.38,
            -0.022,
            -0.29,
            -0.21,
            -0.0105,
            -0.0023,
        ):
            raise ValueError("Vituri2024 paper-direct parameters may not be changed")

    def velocity(self, hopping: float) -> float:
        """Return ``sqrt(3) * a0 * hopping / 2`` in eV Angstrom."""

        return float(np.sqrt(3.0) * self.a0 * hopping / 2.0)


VITURI2024_PARAMETERS: Final[Vituri2024Parameters] = Vituri2024Parameters()


@dataclass(frozen=True, slots=True)
class ApproximatePaperCheckpoint:
    """A paper-direct number that this adapter does not independently verify."""

    quantity: str
    value: float
    unit: str
    evidence: str
    qualifier: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0.0:
            raise ValueError("Vituri checkpoint value must be finite and non-negative")
        if not self.quantity.strip() or not self.unit.strip() or not self.evidence.strip():
            raise ValueError("Vituri checkpoint requires quantity/unit/evidence")
        if self.qualifier not in (
            "paper_approximate_not_acceptance_threshold",
            "source_reported_density_normalization_conflict_not_threshold",
        ):
            raise ValueError("Vituri approximate checkpoint qualifier was changed")


SPIN_STIFFNESS_CHECKPOINTS: Final[tuple[ApproximatePaperCheckpoint, ...]] = (
    ApproximatePaperCheckpoint(
        quantity="2 rho_s / |n|",
        value=780.0,
        unit="meV*a0^2",
        evidence="SM.tex line 314 (fit to the intravalley magnon dispersion)",
        qualifier="paper_approximate_not_acceptance_threshold",
    ),
    ApproximatePaperCheckpoint(
        quantity="rho_s",
        value=0.28,
        unit="meV",
        evidence=(
            "SM.tex line 314 reports 280 micro-eV; main.tex line 183 rounds "
            "this to 0.3 meV"
        ),
        qualifier="source_reported_density_normalization_conflict_not_threshold",
    ),
)


@dataclass(frozen=True, slots=True)
class UnresolvedAuthorityItem:
    """Missing or internally inconsistent authority needed beyond Eq. ``Ham6``."""

    key: str
    detail: str
    evidence: str


UNRESOLVED_AUTHORITY: Final[tuple[UnresolvedAuthorityItem, ...]] = (
    UnresolvedAuthorityItem(
        key="paper_gauge_basis_index_conflict",
        detail=(
            "The displayed basis puts B3 at one-based index 2, while SM.tex "
            "line 97 calls B3 psi_6 and U_{6,3}.  No paper gauge is imposed; "
            "only the rank-one projector is authoritative here."
        ),
        evidence="SM.tex Eq. Ham6/line 31 versus line 97",
    ),
    UnresolvedAuthorityItem(
        key="spin_stiffness_density_normalization",
        detail=(
            "The source reports 780 meV*a0^2 and n=6e11 cm^-2 per valley, "
            "but rho_s=0.28 meV follows only from the two-valley total density "
            "1.2e12 cm^-2; the normalization is internally ambiguous."
        ),
        evidence="SM.tex line 314 versus main.tex line 183",
    ),
    UnresolvedAuthorityItem(
        key="momentum_axis_and_valley_center_convention",
        detail=(
            "Ham6 defines pi=tau*kx+i*ky but does not fully pin the absolute "
            "crystalline axis/origin needed to compare finite-q directions."
        ),
        evidence="SM.tex lines 20--35",
    ),
    UnresolvedAuthorityItem(
        key="gate_distance_d",
        detail="The gate distance d needed by the screened interaction is not pinned.",
        evidence="SM.tex line 59",
    ),
    UnresolvedAuthorityItem(
        key="uv_domain_and_cutoff",
        detail=(
            "The continuum momentum domain/UV cutoff and a domain-wide minimum "
            "direct-gap certificate for the third band are not pinned."
        ),
        evidence="Not specified by the scoped model authority (SM.tex lines 9--132)",
    ),
    UnresolvedAuthorityItem(
        key="interaction_q0_policy",
        detail="The screened-interaction q=0/background prescription is not pinned.",
        evidence="SM.tex lines 58--63 specify V(q) but not its q=0 implementation",
    ),
    UnresolvedAuthorityItem(
        key="figure_q0_momentum",
        detail="The exact figure momentum called q0 is not tabulated numerically.",
        evidence="main.tex Fig. 3 and SM.tex Figs. 10--11 identify q0 graphically",
    ),
    UnresolvedAuthorityItem(
        key="mesh_and_quadrature",
        detail="The momentum mesh, weights, boundary convention, and quadrature are not pinned.",
        evidence="Not specified by the scoped model authority (SM.tex lines 9--132)",
    ),
    UnresolvedAuthorityItem(
        key="ensemble_and_source",
        detail=(
            "The thermodynamic ensemble and the externally fixed density, chemical "
            "potential, displacement-field, or electrostatic source policy are not pinned."
        ),
        evidence="Not fully specified by the scoped model authority (SM.tex lines 9--132)",
    ),
    UnresolvedAuthorityItem(
        key="cdw_harmonic_cutoff_and_q_scan",
        detail="The numerical CDW harmonic cutoff Lambda and ordering-q scan are not pinned.",
        evidence="SM.tex lines 103--110 define the ansatz but not a complete numerical scan",
    ),
    UnresolvedAuthorityItem(
        key="scf_policy",
        detail="Initialization, mixing, convergence, branch selection, and restart policy are not pinned.",
        evidence="Not fully specified by the scoped model authority (SM.tex lines 101--132)",
    ),
    UnresolvedAuthorityItem(
        key="exact_tdhf_q_tolerances_and_provider",
        detail=(
            "The exact TDHF q points, numerical tolerances, and response/eigensolver "
            "provider are not pinned."
        ),
        evidence="Outside the complete authority supplied for this isolated adapter",
    ),
)


class UnresolvedVituriAuthorityError(RuntimeError):
    """Raised when unresolved paper authority is requested as production-ready."""


@dataclass(frozen=True, slots=True)
class Vituri2024Spec:
    """Pinned scope and fail-closed authority record for this adapter."""

    basis: tuple[str, ...] = BASIS
    active_band_index_zero_based: int = ACTIVE_BAND_INDEX_ZERO_BASED
    paper_gauge_imposed: bool = False
    source_sha256: str = ARXIV_SOURCE_SHA256
    pdf_sha256: str = ARXIV_PDF_SHA256
    sm_tex_sha256: str = SM_TEX_SHA256
    unresolved_authority: tuple[UnresolvedAuthorityItem, ...] = UNRESOLVED_AUTHORITY
    spin_stiffness_checkpoints: tuple[ApproximatePaperCheckpoint, ...] = (
        SPIN_STIFFNESS_CHECKPOINTS
    )

    def __post_init__(self) -> None:
        expected_keys = tuple(item.key for item in UNRESOLVED_AUTHORITY)
        actual_keys = tuple(item.key for item in self.unresolved_authority)
        if actual_keys != expected_keys:
            raise ValueError("the pinned unresolved-authority list may not be overridden")
        if self.spin_stiffness_checkpoints != SPIN_STIFFNESS_CHECKPOINTS:
            raise ValueError("the qualified spin-stiffness checkpoints may not be changed")
        if self.paper_gauge_imposed:
            raise ValueError("the contradictory paper gauge must remain unimposed")
        pinned = (
            self.source_sha256 == ARXIV_SOURCE_SHA256,
            self.pdf_sha256 == ARXIV_PDF_SHA256,
            self.sm_tex_sha256 == SM_TEX_SHA256,
            self.basis == BASIS,
            self.active_band_index_zero_based == ACTIVE_BAND_INDEX_ZERO_BASED,
        )
        if not all(pinned):
            raise ValueError("the Vituri2024 authority pins may not be overridden")

    @property
    def unresolved_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.unresolved_authority)

    @property
    def production_ready(self) -> bool:
        return not self.unresolved_authority

    def require_resolved(self) -> None:
        """Fail closed until every listed authority item has been resolved."""

        if self.unresolved_authority:
            keys = ", ".join(self.unresolved_keys)
            raise UnresolvedVituriAuthorityError(
                f"Vituri2024 many-body specification is unresolved: {keys}"
            )


VITURI2024_SPEC: Final[Vituri2024Spec] = Vituri2024Spec()


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _strict_valley(valley: object) -> int:
    if isinstance(valley, (bool, np.bool_)):
        raise ValueError("valley must be the integer +1 or -1")
    try:
        tau = operator.index(valley)  # accepts Python and NumPy integer scalars
    except TypeError as exc:
        raise ValueError("valley must be the integer +1 or -1") from exc
    if tau not in (-1, 1):
        raise ValueError("valley must be the integer +1 or -1")
    return int(tau)


def _finite_real_scalar(value: object, *, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.isrealobj(array):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(array)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _finite_momentum(k: ArrayLike) -> FloatArray:
    raw = np.asarray(k)
    if raw.shape != (2,) or not np.isrealobj(raw):
        raise ValueError("k must be a length-2 finite real vector in 1/Angstrom")
    try:
        momentum = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a length-2 finite real vector in 1/Angstrom") from exc
    if not np.all(np.isfinite(momentum)):
        raise ValueError("k must be a length-2 finite real vector in 1/Angstrom")
    return momentum


def six_band_hamiltonian(
    k: ArrayLike,
    valley: int,
    Delta1: float,
    *,
    parameters: Vituri2024Parameters = VITURI2024_PARAMETERS,
) -> ComplexArray:
    """Return Eq. ``Ham6`` in eV in basis ``(A1,B3,B1,A2,B2,A3)``.

    Parameters
    ----------
    k:
        Local valley-centered Cartesian components ``(kx, ky)`` entering
        ``pi=tau*kx+i*ky``, in inverse Angstrom.  The source does not fully
        pin the absolute crystalline-axis convention for finite-q comparison.
    valley:
        Strict integer valley label ``+1`` or ``-1``.
    Delta1:
        Finite outer-layer potential in eV.
    parameters:
        Frozen tight-binding parameters.  No interaction parameters enter.
    """

    if not isinstance(parameters, Vituri2024Parameters):
        raise TypeError("parameters must be a Vituri2024Parameters instance")
    kx, ky = _finite_momentum(k)
    tau = _strict_valley(valley)
    d1 = _finite_real_scalar(Delta1, name="Delta1")

    p = tau * kx + 1j * ky
    pc = np.conjugate(p)
    v0 = parameters.velocity(parameters.gamma0)
    v3 = parameters.velocity(parameters.gamma3)
    v4 = parameters.velocity(parameters.gamma4)
    g1 = parameters.gamma1
    g2 = parameters.gamma2
    d2 = parameters.Delta2
    delta = parameters.delta

    return np.asarray(
        [
            [d1 + d2 + delta, 0.5 * g2, v0 * pc, v4 * pc, v3 * p, 0.0],
            [0.5 * g2, d2 - d1 + delta, 0.0, v3 * pc, v4 * p, v0 * p],
            [v0 * p, 0.0, d1 + d2, g1, v4 * pc, 0.0],
            [v4 * p, v3 * p, g1, -2.0 * d2, v0 * pc, v4 * pc],
            [v3 * pc, v4 * pc, v4 * p, v0 * p, -2.0 * d2, g1],
            [0.0, v0 * pc, 0.0, v4 * p, g1, d2 - d1],
        ],
        dtype=np.complex128,
    )


# Concise discoverable alias; both names have the same strict contract.
hamiltonian = six_band_hamiltonian


def c3_basis_operator(valley: int) -> ComplexArray:
    """Return the basis representation of a +120-degree momentum rotation.

    This follows algebraically from Eq. ``Ham6`` in the displayed basis.  It
    is an operator-covariance check, not a resolution of the paper's absolute
    crystalline-axis convention.
    """

    tau = _strict_valley(valley)
    angle = 2.0 * np.pi / 3.0
    phase = np.exp(1j * tau * angle)
    return np.diag([1.0, 1.0, phase, phase, np.conjugate(phase), np.conjugate(phase)]).astype(
        np.complex128
    )


def state_projector(state: ArrayLike) -> ComplexArray:
    """Return the normalized rank-one projector ``|state><state|``.

    The result is unchanged under any nonzero complex rescaling, including a
    phase-gauge change.  This is the supported gauge-independent object.
    """

    raw = np.asarray(state)
    if raw.shape != (6,):
        raise ValueError("state must have shape (6,)")
    try:
        vector = np.asarray(raw, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("state must be a finite complex vector") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError("state must be a finite complex vector")
    norm_squared = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("state must have nonzero finite norm")
    normalized = vector / np.sqrt(norm_squared)
    return np.outer(normalized, normalized.conjugate())


@dataclass(frozen=True, slots=True)
class ActiveBandEigensolution:
    """Locally nondegenerate third-lowest state with no imposed phase gauge.

    This does not certify physical isolation over any momentum/displacement
    domain; that domain and its minimum direct gaps remain unresolved.
    """

    energy: float
    eigenvector: ComplexArray
    projector: ComplexArray
    spectrum: FloatArray
    lower_gap: float
    upper_gap: float
    band_index_zero_based: int = ACTIVE_BAND_INDEX_ZERO_BASED


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    array.setflags(write=False)
    return array


def _require_local_third_band_nondegeneracy(
    energies: ArrayLike,
) -> tuple[float, float]:
    spectrum = np.asarray(energies, dtype=np.float64)
    if spectrum.shape != (6,) or not np.all(np.isfinite(spectrum)):
        raise ValueError("Vituri local spectrum must contain six finite energies")
    index = ACTIVE_BAND_INDEX_ZERO_BASED
    lower_gap = float(spectrum[index] - spectrum[index - 1])
    upper_gap = float(spectrum[index + 1] - spectrum[index])
    scale = max(float(np.max(np.abs(spectrum), initial=0.0)), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    if min(lower_gap, upper_gap) <= tolerance:
        raise RuntimeError(
            "third-lowest Vituri active band is not locally nondegenerate"
        )
    return lower_gap, upper_gap


def third_lowest_active_band(
    k: ArrayLike,
    valley: int,
    Delta1: float,
    *,
    parameters: Vituri2024Parameters = VITURI2024_PARAMETERS,
) -> ActiveBandEigensolution:
    """Diagonalize Eq. ``Ham6`` and return a locally nondegenerate state.

    This pointwise machine-scale check is not a physical band-isolation gate.
    ``numpy.linalg.eigh`` determines an arbitrary eigenvector phase.  The
    paper's contradictory B3/``psi_6`` gauge instruction is deliberately not
    applied.  Consumers requiring a gauge-independent quantity should use the
    returned projector.
    """

    matrix = six_band_hamiltonian(k, valley, Delta1, parameters=parameters)
    energies, vectors = np.linalg.eigh(matrix)
    index = ACTIVE_BAND_INDEX_ZERO_BASED
    lower_gap, upper_gap = _require_local_third_band_nondegeneracy(energies)
    vector = np.asarray(vectors[:, index], dtype=np.complex128).copy()
    projector = state_projector(vector)
    spectrum = np.asarray(energies, dtype=np.float64).copy()
    _readonly(vector)
    _readonly(projector)
    _readonly(spectrum)
    return ActiveBandEigensolution(
        energy=float(energies[index]),
        eigenvector=vector,
        projector=projector,
        spectrum=spectrum,
        lower_gap=lower_gap,
        upper_gap=upper_gap,
    )


def third_lowest_active_projector(
    k: ArrayLike,
    valley: int,
    Delta1: float,
    *,
    parameters: Vituri2024Parameters = VITURI2024_PARAMETERS,
) -> ComplexArray:
    """Return only the gauge-independent third-lowest-band projector."""

    return third_lowest_active_band(
        k, valley, Delta1, parameters=parameters
    ).projector


__all__ = [
    "ACTIVE_BAND_INDEX_ZERO_BASED",
    "ARXIV_IDENTIFIER",
    "ARXIV_PDF_SHA256",
    "ARXIV_SOURCE_SHA256",
    "ActiveBandEigensolution",
    "ApproximatePaperCheckpoint",
    "BASIS",
    "PDF_AUTHORITY_PATH",
    "SM_TEX_AUTHORITY_PATH",
    "SM_TEX_SHA256",
    "SPIN_STIFFNESS_CHECKPOINTS",
    "UNRESOLVED_AUTHORITY",
    "UnresolvedAuthorityItem",
    "UnresolvedVituriAuthorityError",
    "VITURI2024_PARAMETERS",
    "VITURI2024_SPEC",
    "Vituri2024Parameters",
    "Vituri2024Spec",
    "c3_basis_operator",
    "hamiltonian",
    "six_band_hamiltonian",
    "state_projector",
    "third_lowest_active_band",
    "third_lowest_active_projector",
]
