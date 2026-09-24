"""Fresh split-zero canonical Kane--Poisson fixed-point machinery.

This module implements the reusable numerical core only.  A caller supplies an
attested Kane eigensystem builder that re-diagonalizes the same parent Hamiltonian
for every input electrostatic potential.  The first production milestone is a
canonical neutral periodic slab; fixed-gate device electrostatics remain a separate,
blocked ensemble until their physical source data are resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from .normal_reference import (
    ChemicalPotentialRootTolerances,
    KaneChargeNeutralReference,
    KaneNormalReferenceContract,
    build_charge_neutral_kane_reference,
    kane_window_source_fingerprint,
    normal_reference_fingerprint,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
COULOMB_MEV_NM = 1439.96448
KANE_POISSON_SCHEMA_VERSION = 1
POTENTIAL_COUPLING_LABEL = "electron_energy_plus_U_identity_in_canonical_z_orbital_flattening"
CANONICAL_FLATTENING_LABEL = "flat_index=8*z_index+kane_orbital_index"
PLANE_WAVE_POTENTIAL_COUPLING_LABEL = (
    "electron_energy_plus_I8_kron_projected_U_pw"
)
PLANE_WAVE_FLATTENING_LABEL = (
    "flat_index=kane_orbital_index*(2*N+1)+harmonic_position"
)


def kane_poisson_array_sha256(values: np.ndarray) -> str:
    """Hash one exact numerical array with dtype and shape binding."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def canonical_parent_matrix_sequence_sha256(
    matrices: tuple[np.ndarray | sparse.spmatrix, ...] | list[np.ndarray | sparse.spmatrix],
) -> str:
    """Hash full-parent CSR matrices after an explicit ``1e-12 meV`` quantization."""

    if not matrices:
        raise ValueError("at least one full-parent matrix is required")
    digest = hashlib.sha256()
    digest.update(b"canonical_full_parent_matrix_sequence:v1")
    for matrix in matrices:
        csr = sparse.csr_matrix(matrix, dtype=np.complex128, copy=True)
        if csr.shape[0] != csr.shape[1]:
            raise ValueError("full-parent Hamiltonian matrices must be square")
        csr.sum_duplicates()
        csr.sort_indices()
        csr.data = (
            np.round(csr.data.real, decimals=12)
            + 1.0j * np.round(csr.data.imag, decimals=12)
        )
        csr.eliminate_zeros()
        digest.update(str(csr.shape).encode())
        digest.update(kane_poisson_array_sha256(csr.indptr.astype(np.int64)).encode())
        digest.update(kane_poisson_array_sha256(csr.indices.astype(np.int64)).encode())
        digest.update(kane_poisson_array_sha256(csr.data.astype(np.complex128)).encode())
    return digest.hexdigest()


def _validate_sha256(value: str, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return digest


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def kane_potential_operator_fingerprint(potential_mev: FloatArray) -> str:
    """Hash the exact canonical ``diag(repeat(U_z, 8))`` parent-space operator."""

    potential = np.asarray(potential_mev, dtype=np.float64)
    if potential.ndim != 1 or not np.all(np.isfinite(potential)):
        raise ValueError("potential_mev must be a finite one-dimensional array")
    digest = hashlib.sha256()
    digest.update(POTENTIAL_COUPLING_LABEL.encode())
    digest.update(CANONICAL_FLATTENING_LABEL.encode())
    digest.update(kane_poisson_array_sha256(np.repeat(potential, 8)).encode())
    return digest.hexdigest()


def plane_wave_potential_operator_fingerprint(
    potential_mev: FloatArray,
) -> str:
    """Bind sampled ``U(z)`` to the canonical ``I8 kron U_PW`` coupling."""

    potential = np.asarray(potential_mev, dtype=np.float64)
    if potential.ndim != 1 or not np.all(np.isfinite(potential)):
        raise ValueError("potential_mev must be a finite one-dimensional array")
    digest = hashlib.sha256()
    digest.update(PLANE_WAVE_POTENTIAL_COUPLING_LABEL.encode())
    digest.update(PLANE_WAVE_FLATTENING_LABEL.encode())
    digest.update(kane_poisson_array_sha256(potential).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class KaneParentSourceSpec:
    """Immutable parent/mesh/electrostatic identity shared by every iteration."""

    k_cart_nm_inv: FloatArray
    k_weights_nm2: FloatArray
    z_nm: FloatArray
    z_weights_nm: FloatArray
    epsilon_r: FloatArray
    parent_hilbert_dimension: int
    selected_band_indices: tuple[int, ...]
    candidate_eigenpair_count: int
    target_energy_mev: float
    window_selection_label: str
    static_parent_sha256: str
    material_profile_sha256: str
    kdotpy_source_sha256: str
    hamiltonian_options: tuple[tuple[str, str], ...]
    basis_flattening_label: str = CANONICAL_FLATTENING_LABEL
    potential_coupling_label: str = POTENTIAL_COUPLING_LABEL
    schema_version: int = KANE_POISSON_SCHEMA_VERSION

    def __post_init__(self) -> None:
        k_cart = _readonly_copy(self.k_cart_nm_inv, dtype=np.dtype(np.float64))
        k_weights = _readonly_copy(self.k_weights_nm2, dtype=np.dtype(np.float64))
        z = _readonly_copy(self.z_nm, dtype=np.dtype(np.float64))
        z_weights = _readonly_copy(self.z_weights_nm, dtype=np.dtype(np.float64))
        epsilon = _readonly_copy(self.epsilon_r, dtype=np.dtype(np.float64))
        object.__setattr__(self, "k_cart_nm_inv", k_cart)
        object.__setattr__(self, "k_weights_nm2", k_weights)
        object.__setattr__(self, "z_nm", z)
        object.__setattr__(self, "z_weights_nm", z_weights)
        object.__setattr__(self, "epsilon_r", epsilon)
        if k_cart.ndim != 2 or k_cart.shape[1] != 2:
            raise ValueError("k_cart_nm_inv must have shape (nk,2)")
        if k_weights.shape != (k_cart.shape[0],) or np.any(k_weights <= 0.0):
            raise ValueError("k_weights_nm2 must be positive with shape (nk,)")
        if z.ndim != 1 or z.size < 4 or z_weights.shape != z.shape or epsilon.shape != z.shape:
            raise ValueError("z, z_weights, and epsilon_r must be matching one-dimensional arrays")
        if not all(np.all(np.isfinite(array)) for array in (k_cart, k_weights, z, z_weights, epsilon)):
            raise ValueError("parent mesh arrays must be finite")
        if np.any(z_weights <= 0.0) or np.any(epsilon <= 0.0):
            raise ValueError("z weights and dielectric profile must be positive")
        parent_dimension = int(self.parent_hilbert_dimension)
        if parent_dimension != self.parent_hilbert_dimension or parent_dimension <= 0:
            raise ValueError("parent_hilbert_dimension must be a positive exact integer")
        if parent_dimension % 8 != 0:
            raise ValueError("parent_hilbert_dimension must contain eight Kane orbitals")
        labels = tuple(int(label) for label in self.selected_band_indices)
        if (
            not labels
            or len(set(labels)) != len(labels)
            or any(label == 0 for label in labels)
            or any(original != integer for original, integer in zip(self.selected_band_indices, labels))
        ):
            raise ValueError("selected_band_indices must be unique exact nonzero integers")
        object.__setattr__(self, "selected_band_indices", labels)
        if (
            int(self.candidate_eigenpair_count) != self.candidate_eigenpair_count
            or int(self.candidate_eigenpair_count) <= len(labels)
        ):
            raise ValueError(
                "candidate_eigenpair_count must be an integer strictly larger than the selected window"
            )
        if not np.isfinite(self.target_energy_mev):
            raise ValueError("target_energy_mev must be finite")
        if not str(self.window_selection_label).strip():
            raise ValueError("window_selection_label must be explicit")
        for value, label in (
            (self.static_parent_sha256, "static_parent_sha256"),
            (self.material_profile_sha256, "material_profile_sha256"),
            (self.kdotpy_source_sha256, "kdotpy_source_sha256"),
        ):
            _validate_sha256(value, label=label)
        options = tuple((str(key), str(value)) for key, value in self.hamiltonian_options)
        if not options or len({key for key, _value in options}) != len(options):
            raise ValueError("hamiltonian_options must contain unique typed keys")
        object.__setattr__(self, "hamiltonian_options", tuple(sorted(options)))
        representation = (
            self.basis_flattening_label,
            self.potential_coupling_label,
        )
        real_space_representation = (
            CANONICAL_FLATTENING_LABEL,
            POTENTIAL_COUPLING_LABEL,
        )
        plane_wave_representation = (
            PLANE_WAVE_FLATTENING_LABEL,
            PLANE_WAVE_POTENTIAL_COUPLING_LABEL,
        )
        if representation == real_space_representation:
            if parent_dimension != 8 * z.size:
                raise ValueError(
                    "real-space parent_hilbert_dimension must equal 8*nz"
                )
        elif representation != plane_wave_representation:
            raise ValueError(
                "basis flattening and electrostatic coupling must name one "
                "supported parent representation"
            )
        if int(self.schema_version) != KANE_POISSON_SCHEMA_VERSION:
            raise ValueError("unsupported Kane--Poisson source schema version")

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(f"KaneParentSourceSpec:v{self.schema_version}".encode())
        for array in (
            self.k_cart_nm_inv,
            self.k_weights_nm2,
            self.z_nm,
            self.z_weights_nm,
            self.epsilon_r,
        ):
            digest.update(kane_poisson_array_sha256(array).encode())
        digest.update(str(self.parent_hilbert_dimension).encode())
        digest.update(repr(self.selected_band_indices).encode())
        digest.update(str(self.candidate_eigenpair_count).encode())
        digest.update(repr(float(self.target_energy_mev)).encode())
        digest.update(self.window_selection_label.encode())
        digest.update(self.static_parent_sha256.encode())
        digest.update(self.material_profile_sha256.encode())
        digest.update(self.kdotpy_source_sha256.encode())
        digest.update(json.dumps(self.hamiltonian_options, separators=(",", ":")).encode())
        digest.update(self.basis_flattening_label.encode())
        digest.update(self.potential_coupling_label.encode())
        return digest.hexdigest()


@dataclass(frozen=True)
class KaneWindowAtPotential:
    """One selected Kane window obtained at an exact input potential."""

    hamiltonian_mev: ComplexArray
    micro_wavefunctions: ComplexArray
    selected_band_indices: tuple[int, ...]
    parent_hilbert_dimension: int
    calculation_split_mev: float
    energy_zero_label: str
    input_potential_sha256: str
    potential_operator_fingerprint: str
    parent_spec_fingerprint: str
    replayed_static_parent_sha256: str
    static_parent_replay_max_error_mev: float
    potential_operator_max_error_mev: float
    provenance_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hamiltonian_mev",
            _readonly_copy(self.hamiltonian_mev, dtype=np.dtype(np.complex128)),
        )
        object.__setattr__(
            self,
            "micro_wavefunctions",
            _readonly_copy(self.micro_wavefunctions, dtype=np.dtype(np.complex128)),
        )
        labels = tuple(int(label) for label in self.selected_band_indices)
        if any(original != integer for original, integer in zip(self.selected_band_indices, labels)):
            raise ValueError("selected_band_indices must contain exact integers")
        object.__setattr__(self, "selected_band_indices", labels)

    def validate(
        self,
        *,
        input_potential_mev: FloatArray,
        parent_spec: KaneParentSourceSpec,
    ) -> None:
        hamiltonian = np.asarray(self.hamiltonian_mev, dtype=np.complex128)
        wavefunctions = np.asarray(self.micro_wavefunctions, dtype=np.complex128)
        nk = parent_spec.k_cart_nm_inv.shape[0]
        nz = parent_spec.z_nm.size
        nband = len(self.selected_band_indices)
        if self.selected_band_indices != parent_spec.selected_band_indices:
            raise ValueError("selected Kane window policy changed relative to the parent spec")
        if not self.selected_band_indices or len(set(self.selected_band_indices)) != nband:
            raise ValueError("selected_band_indices must be nonempty and unique")
        if any(label == 0 for label in self.selected_band_indices):
            raise ValueError("selected Kane band labels must be nonzero")
        if hamiltonian.shape != (nband, nband, nk):
            raise ValueError("hamiltonian_mev must have shape (nband, nband, nk)")
        if wavefunctions.shape != (nk, nz, 8, nband):
            raise ValueError("micro_wavefunctions must have shape (nk, nz, 8, nband)")
        if not np.all(np.isfinite(hamiltonian)) or not np.all(np.isfinite(wavefunctions)):
            raise ValueError("Kane eigensystem arrays must be finite")
        if int(self.parent_hilbert_dimension) != parent_spec.parent_hilbert_dimension:
            raise ValueError("Kane window parent dimension differs from the immutable parent spec")
        if not np.isfinite(self.calculation_split_mev) or self.calculation_split_mev != 0.0:
            raise ValueError("fresh Kane--Poisson sources require calculation_split_mev == 0")
        if not str(self.energy_zero_label).strip():
            raise ValueError("energy_zero_label must be explicit")
        input_potential = np.asarray(input_potential_mev, dtype=np.float64)
        if input_potential.shape != (nz,):
            raise ValueError("input_potential_mev must have shape (nz,)")
        expected_potential_hash = kane_poisson_array_sha256(input_potential)
        if _validate_sha256(
            self.input_potential_sha256, label="input_potential_sha256"
        ) != expected_potential_hash:
            raise ValueError("Kane eigensystem is not bound to the supplied input potential")
        if parent_spec.potential_coupling_label == POTENTIAL_COUPLING_LABEL:
            expected_potential_fingerprint = kane_potential_operator_fingerprint(
                input_potential_mev
            )
        elif (
            parent_spec.potential_coupling_label
            == PLANE_WAVE_POTENTIAL_COUPLING_LABEL
        ):
            expected_potential_fingerprint = (
                plane_wave_potential_operator_fingerprint(input_potential_mev)
            )
        else:  # Defensive: KaneParentSourceSpec already rejects this.
            raise ValueError("unsupported electrostatic potential coupling")
        if _validate_sha256(
            self.potential_operator_fingerprint,
            label="potential_operator_fingerprint",
        ) != expected_potential_fingerprint:
            raise ValueError("Kane eigensystem does not attest the canonical +U parent operator")
        if _validate_sha256(
            self.parent_spec_fingerprint, label="parent_spec_fingerprint"
        ) != parent_spec.fingerprint:
            raise ValueError("Kane eigensystem parent identity differs from the solver parent spec")
        if (
            not np.isfinite(self.static_parent_replay_max_error_mev)
            or self.static_parent_replay_max_error_mev < 0.0
            or self.static_parent_replay_max_error_mev > 1e-10
        ):
            raise ValueError("actual H(U)-U replay does not reproduce the immutable static parent")
        if _validate_sha256(
            self.replayed_static_parent_sha256,
            label="replayed_static_parent_sha256",
        ) != parent_spec.static_parent_sha256:
            raise ValueError("verified static-parent identity differs from the immutable parent hash")
        if (
            not np.isfinite(self.potential_operator_max_error_mev)
            or self.potential_operator_max_error_mev < 0.0
            or self.potential_operator_max_error_mev > 1e-10
        ):
            raise ValueError("actual full-parent +U operator differs from the canonical coupling")
        _validate_sha256(self.provenance_fingerprint, label="provenance_fingerprint")

    @property
    def exact_archive_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(kane_poisson_array_sha256(self.hamiltonian_mev).encode())
        digest.update(kane_poisson_array_sha256(self.micro_wavefunctions).encode())
        digest.update(repr(self.selected_band_indices).encode())
        digest.update(str(self.parent_hilbert_dimension).encode())
        digest.update(repr(float(self.calculation_split_mev)).encode())
        digest.update(self.energy_zero_label.encode())
        digest.update(self.input_potential_sha256.encode())
        digest.update(self.potential_operator_fingerprint.encode())
        digest.update(self.parent_spec_fingerprint.encode())
        digest.update(self.replayed_static_parent_sha256.encode())
        digest.update(repr(float(self.static_parent_replay_max_error_mev)).encode())
        digest.update(repr(float(self.potential_operator_max_error_mev)).encode())
        digest.update(self.provenance_fingerprint.encode())
        return digest.hexdigest()


KaneWindowBuilder = Callable[[FloatArray], KaneWindowAtPotential]
IndependentKaneReplayBuilderFactory = Callable[[], KaneWindowBuilder]


@dataclass(frozen=True)
class CanonicalKanePoissonConfig:
    """Numerical contract for a canonical neutral periodic-slab fixed point."""

    temperature_K: float = 0.1
    maximum_iterations: int = 100
    mixing: float = 0.5
    fixed_point_tolerance_mev: float = 1e-5
    density_profile_tolerance_nm3: float = 1e-8
    gauss_residual_tolerance_mev_nm2: float = 1e-8
    integrated_charge_tolerance_nm2: float = 1e-10
    root_tolerances: ChemicalPotentialRootTolerances = ChemicalPotentialRootTolerances()

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        if int(self.maximum_iterations) < 1:
            raise ValueError("maximum_iterations must be positive")
        if not np.isfinite(self.mixing) or not 0.0 < self.mixing <= 1.0:
            raise ValueError("mixing must lie in (0,1]")
        tolerances = (
            self.fixed_point_tolerance_mev,
            self.density_profile_tolerance_nm3,
            self.gauss_residual_tolerance_mev_nm2,
            self.integrated_charge_tolerance_nm2,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in tolerances):
            raise ValueError("Kane--Poisson tolerances must be finite and positive")


@dataclass(frozen=True)
class PeriodicPoissonMap:
    electron_potential_mev: FloatArray
    gauss_residual_mev_nm2: float
    integrated_net_number_nm2: float
    removed_mean_net_number_nm3: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "electron_potential_mev",
            _readonly_copy(self.electron_potential_mev, dtype=np.dtype(np.float64)),
        )


@dataclass(frozen=True)
class KanePoissonIterationRecord:
    iteration: int
    input_potential_sha256: str
    source_array_fingerprint: str
    source_provenance_fingerprint: str
    mu_mev: float
    electron_density_nm2: float
    hole_density_nm2: float
    fixed_point_residual_mev: float
    density_profile_change_nm3: float | None
    gauss_residual_mev_nm2: float
    integrated_net_number_nm2: float
    removed_mean_net_number_nm3: float


@dataclass(frozen=True)
class CanonicalKanePoissonResult:
    config: CanonicalKanePoissonConfig
    parent_spec: KaneParentSourceSpec
    z_nm: FloatArray
    z_weights_nm: FloatArray
    epsilon_r: FloatArray
    k_weights_nm2: FloatArray
    potential_mev: FloatArray
    electron_density_nm3: FloatArray
    hole_density_nm3: FloatArray
    reference: KaneChargeNeutralReference
    final_eigensystem: KaneWindowAtPotential
    history: tuple[KanePoissonIterationRecord, ...]
    fixed_point_residual_mev: float
    gauss_residual_mev_nm2: float
    converged: bool
    replay_verified: bool

    def __post_init__(self) -> None:
        for name in (
            "z_nm",
            "z_weights_nm",
            "epsilon_r",
            "k_weights_nm2",
            "potential_mev",
            "electron_density_nm3",
            "hole_density_nm3",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_copy(getattr(self, name), dtype=np.dtype(np.float64)),
            )
        validate_canonical_kane_poisson_result(self, require_certified=True)

    @property
    def iterations(self) -> int:
        return len(self.history)

    @property
    def credibility_blockers(self) -> tuple[str, ...]:
        return (
            "canonical_periodic_window_root_is_not_a_fixed_gate_device_reference",
            *self.reference.contract.credibility_blockers(),
        )

    @property
    def claim_scope(self) -> str:
        return "canonical_periodic_window_root_not_fixed_gate"


def periodic_variable_dielectric_poisson_map(
    z_nm: FloatArray,
    electron_density_nm3: FloatArray,
    hole_density_nm3: FloatArray,
    epsilon_r: FloatArray,
    *,
    integrated_charge_tolerance_nm2: float = 1e-10,
) -> PeriodicPoissonMap:
    """Solve ``d_z(epsilon d_z U)=4*pi*C*(n_h-n_e)`` in zero-mean gauge."""

    z = np.asarray(z_nm, dtype=np.float64)
    electron = np.asarray(electron_density_nm3, dtype=np.float64)
    hole = np.asarray(hole_density_nm3, dtype=np.float64)
    epsilon = np.asarray(epsilon_r, dtype=np.float64)
    if z.ndim != 1 or z.size < 4:
        raise ValueError("periodic Poisson requires a one-dimensional z grid with >=4 points")
    if not (z.shape == electron.shape == hole.shape == epsilon.shape):
        raise ValueError("z, density, and dielectric arrays must have matching shapes")
    if not all(np.all(np.isfinite(array)) for array in (z, electron, hole, epsilon)):
        raise ValueError("periodic Poisson inputs must be finite")
    spacing = np.diff(z)
    if spacing[0] <= 0.0 or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError("periodic Poisson requires a strictly increasing uniform z grid")
    if np.any(epsilon <= 0.0):
        raise ValueError("dielectric profile must be positive")
    if not np.isfinite(integrated_charge_tolerance_nm2) or integrated_charge_tolerance_nm2 <= 0.0:
        raise ValueError("integrated_charge_tolerance_nm2 must be positive")

    dz = float(spacing[0])
    net_number = hole - electron
    integrated_net = float(np.sum(net_number) * dz)
    if abs(integrated_net) > integrated_charge_tolerance_nm2:
        raise ValueError(
            "periodic Poisson source is not neutral: "
            f"{integrated_net:.3e} nm^-2"
        )
    removed_mean = float(np.mean(net_number))
    net_number = net_number - removed_mean
    epsilon_plus = 2.0 * epsilon * np.roll(epsilon, -1) / (
        epsilon + np.roll(epsilon, -1)
    )
    epsilon_minus = np.roll(epsilon_plus, 1)
    n = z.size
    operator = np.zeros((n, n), dtype=np.float64)
    for index in range(n):
        operator[index, index] = -(
            epsilon_plus[index] + epsilon_minus[index]
        ) / dz**2
        operator[index, (index + 1) % n] = epsilon_plus[index] / dz**2
        operator[index, (index - 1) % n] = epsilon_minus[index] / dz**2
    source = 4.0 * np.pi * COULOMB_MEV_NM * net_number
    gauge_operator = operator.copy()
    gauge_source = source.copy()
    gauge_operator[-1, :] = 1.0 / n
    gauge_source[-1] = 0.0
    potential = np.linalg.solve(gauge_operator, gauge_source)
    potential -= np.mean(potential)
    gauss_residual = float(np.max(np.abs(operator @ potential - source)))
    return PeriodicPoissonMap(
        electron_potential_mev=potential,
        gauss_residual_mev_nm2=gauss_residual,
        integrated_net_number_nm2=integrated_net,
        removed_mean_net_number_nm3=removed_mean,
    )


def kane_orbital_transfer_density_integrand_components(
    micro_wavefunctions: ComplexArray,
    density: ComplexArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return occupied-electron, occupied-valence, and available-valence terms.

    All arrays have shape ``(nk,nz)`` and units ``nm^-1``.  The Note-6 hole
    integrand is ``available_valence - occupied_valence``.  Exposing this
    decomposition is necessary when a selected W4 window is reference
    subtracted: changes in its available-valence projector must not be hidden
    inside an apparent UV improvement.
    """

    phi = np.asarray(micro_wavefunctions, dtype=np.complex128)
    rho = np.asarray(density, dtype=np.complex128)
    if phi.ndim != 4 or phi.shape[2] != 8:
        raise ValueError("micro_wavefunctions must have shape (nk,nz,8,nband)")
    nk, _nz, _, nband = phi.shape
    if rho.shape != (nband, nband, nk):
        raise ValueError("density must have shape (nband,nband,nk)")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(rho)):
        raise ValueError("density-integrand inputs must be finite")
    hermiticity = float(np.max(np.abs(rho - np.swapaxes(rho.conj(), 0, 1))))
    if hermiticity > 1e-9:
        raise ValueError("ket-oriented density matrix must be Hermitian")
    for ik in range(nk):
        eigenvalues = np.linalg.eigvalsh(rho[:, :, ik])
        if np.min(eigenvalues) < -1e-9 or np.max(eigenvalues) > 1.0 + 1e-9:
            raise ValueError("density eigenvalues must lie in [0,1]")

    electron_local = np.einsum(
        "kzma,kzmb->kzab",
        phi[:, :, :2, :].conj(),
        phi[:, :, :2, :],
        optimize=True,
    )
    valence_local = np.einsum(
        "kzma,kzmb->kzab",
        phi[:, :, 2:, :].conj(),
        phi[:, :, 2:, :],
        optimize=True,
    )
    density_k = np.moveaxis(rho, 2, 0)
    identity = np.eye(nband, dtype=np.complex128)[None, :, :]
    electron_occupied = np.einsum(
        "kzab,kba->kz", electron_local, density_k, optimize=True
    )
    valence_occupied = np.einsum(
        "kzab,kba->kz", valence_local, density_k, optimize=True
    )
    valence_available = np.einsum(
        "kzab,kba->kz", valence_local, identity, optimize=True
    )
    maximum_imaginary = max(
        float(np.max(np.abs(electron_occupied.imag))),
        float(np.max(np.abs(valence_occupied.imag))),
        float(np.max(np.abs(valence_available.imag))),
    )
    if maximum_imaginary > 1e-9:
        raise ValueError("orbital-transfer density components are not real")
    return (
        np.asarray(electron_occupied.real, dtype=np.float64),
        np.asarray(valence_occupied.real, dtype=np.float64),
        np.asarray(valence_available.real, dtype=np.float64),
    )


def kane_orbital_transfer_density_integrands(
    micro_wavefunctions: ComplexArray,
    density: ComplexArray,
) -> tuple[FloatArray, FloatArray]:
    """Return per-k local Note-6 transfer integrands before radial integration."""

    electron, valence_occupied, valence_available = (
        kane_orbital_transfer_density_integrand_components(
            micro_wavefunctions,
            density,
        )
    )
    return electron, valence_available - valence_occupied


def kane_orbital_transfer_density_profiles(
    micro_wavefunctions: ComplexArray,
    density: ComplexArray,
    k_weights_nm2: FloatArray,
    z_weights_nm: FloatArray,
    *,
    expected_electron_density_nm2: float | None = None,
    expected_hole_density_nm2: float | None = None,
    profile_integral_tolerance_nm2: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Evaluate local Note-6 profiles for ket-oriented ``P_ab=<c_b^† c_a>``."""

    electron_kz, hole_kz = kane_orbital_transfer_density_integrands(
        micro_wavefunctions,
        density,
    )
    weights = np.asarray(k_weights_nm2, dtype=np.float64)
    z_weights = np.asarray(z_weights_nm, dtype=np.float64)
    nk, nz = electron_kz.shape
    if weights.shape != (nk,) or z_weights.shape != (nz,):
        raise ValueError("k and z quadrature weights have incompatible shapes")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(z_weights)):
        raise ValueError("quadrature weights must be finite")
    if np.any(weights <= 0.0) or np.any(z_weights <= 0.0):
        raise ValueError("quadrature weights must be positive")
    if not np.isfinite(profile_integral_tolerance_nm2) or profile_integral_tolerance_nm2 <= 0.0:
        raise ValueError("profile_integral_tolerance_nm2 must be positive")
    electron_real = np.einsum("k,kz->z", weights, electron_kz, optimize=True)
    hole_real = np.einsum("k,kz->z", weights, hole_kz, optimize=True)
    electron_integral = float(np.dot(z_weights, electron_real))
    hole_integral = float(np.dot(z_weights, hole_real))
    if expected_electron_density_nm2 is not None and abs(
        electron_integral - float(expected_electron_density_nm2)
    ) > profile_integral_tolerance_nm2:
        raise ValueError("electron profile does not reproduce the reference areal density")
    if expected_hole_density_nm2 is not None and abs(
        hole_integral - float(expected_hole_density_nm2)
    ) > profile_integral_tolerance_nm2:
        raise ValueError("hole profile does not reproduce the reference areal density")
    return electron_real, hole_real


def _contract_for_source(
    source: KaneWindowAtPotential,
    *,
    source_fingerprint: str,
    poisson_residual_mev: float | None,
    poisson_residual_tolerance_mev: float,
) -> KaneNormalReferenceContract:
    return KaneNormalReferenceContract(
        source_fingerprint=source_fingerprint,
        selected_band_indices=tuple(int(label) for label in source.selected_band_indices),
        parent_hilbert_dimension=int(source.parent_hilbert_dimension),
        calculation_split_mev=float(source.calculation_split_mev),
        energy_zero_label=str(source.energy_zero_label),
        electrostatic_ensemble="canonical_neutral_slab",
        fixed_gate_source_complete=False,
        band_window_converged=False,
        momentum_window_converged=False,
        z_mesh_converged=False,
        poisson_residual_mev=poisson_residual_mev,
        poisson_residual_tolerance_mev=float(poisson_residual_tolerance_mev),
        radial_only=True,
    )


def _evaluate_source(
    source: KaneWindowAtPotential,
    *,
    input_potential_mev: FloatArray,
    parent_spec: KaneParentSourceSpec,
    temperature_K: float,
    root_tolerances: ChemicalPotentialRootTolerances,
    poisson_residual_tolerance_mev: float,
) -> tuple[KaneChargeNeutralReference, FloatArray, FloatArray, str]:
    source.validate(
        input_potential_mev=input_potential_mev,
        parent_spec=parent_spec,
    )
    source_fingerprint = kane_window_source_fingerprint(
        source.hamiltonian_mev,
        parent_spec.k_weights_nm2,
        source.micro_wavefunctions,
        parent_spec.z_weights_nm,
    )
    contract = _contract_for_source(
        source,
        source_fingerprint=source_fingerprint,
        poisson_residual_mev=None,
        poisson_residual_tolerance_mev=poisson_residual_tolerance_mev,
    )
    reference = build_charge_neutral_kane_reference(
        source.hamiltonian_mev,
        parent_spec.k_weights_nm2,
        source.micro_wavefunctions,
        parent_spec.z_weights_nm,
        contract=contract,
        temperature_K=temperature_K,
        tolerances=root_tolerances,
    )
    electron, hole = kane_orbital_transfer_density_profiles(
        source.micro_wavefunctions,
        reference.density,
        parent_spec.k_weights_nm2,
        parent_spec.z_weights_nm,
        expected_electron_density_nm2=reference.electron_density_nm2,
        expected_hole_density_nm2=reference.hole_density_nm2,
    )
    return reference, electron, hole, source_fingerprint


def _window_replay_subspace_error(
    first: ComplexArray,
    second: ComplexArray,
    z_weights_nm: FloatArray,
) -> float:
    overlap = np.einsum(
        "kzma,z,kzmb->kab",
        np.asarray(first).conj(),
        np.asarray(z_weights_nm),
        np.asarray(second),
        optimize=True,
    )
    error = 0.0
    for ik in range(overlap.shape[0]):
        singular_values = np.linalg.svd(overlap[ik], compute_uv=False)
        error = max(error, float(np.max(np.abs(singular_values - 1.0))))
    return error


def _window_replay_covariance_errors(
    first_source: KaneWindowAtPotential,
    second_source: KaneWindowAtPotential,
    first_reference: KaneChargeNeutralReference,
    second_reference: KaneChargeNeutralReference,
    z_weights_nm: FloatArray,
) -> tuple[float, float]:
    """Return polar-unitary Hamiltonian and density-matrix covariance errors."""

    overlap = np.einsum(
        "kzma,z,kzmb->kab",
        first_source.micro_wavefunctions.conj(),
        np.asarray(z_weights_nm),
        second_source.micro_wavefunctions,
        optimize=True,
    )
    hamiltonian_error = 0.0
    density_error = 0.0
    for ik, weighted_overlap in enumerate(overlap):
        left, _singular_values, right_adjoint = np.linalg.svd(
            weighted_overlap, full_matrices=False
        )
        polar_unitary = left @ right_adjoint
        expected_hamiltonian = (
            polar_unitary.conj().T
            @ first_source.hamiltonian_mev[:, :, ik]
            @ polar_unitary
        )
        expected_density = (
            polar_unitary.conj().T
            @ first_reference.density[:, :, ik]
            @ polar_unitary
        )
        hamiltonian_error = max(
            hamiltonian_error,
            float(
                np.max(
                    np.abs(
                        second_source.hamiltonian_mev[:, :, ik]
                        - expected_hamiltonian
                    )
                )
            ),
        )
        density_error = max(
            density_error,
            float(
                np.max(
                    np.abs(
                        second_reference.density[:, :, ik] - expected_density
                    )
                )
            ),
        )
    return hamiltonian_error, density_error


def _window_replay_metadata(source: KaneWindowAtPotential) -> tuple[object, ...]:
    return (
        source.selected_band_indices,
        source.parent_hilbert_dimension,
        source.calculation_split_mev,
        source.energy_zero_label,
        source.input_potential_sha256,
        source.parent_spec_fingerprint,
        source.potential_operator_fingerprint,
        source.replayed_static_parent_sha256,
        source.static_parent_replay_max_error_mev,
        source.potential_operator_max_error_mev,
    )


def _assert_cold_kane_replay_builder(builder: object) -> None:
    """Fail closed when a supposedly independent replay builder is warmed."""

    for method_name in ("verify_cold_start", "assert_cold_start"):
        verifier = getattr(builder, method_name, None)
        if callable(verifier):
            outcome = verifier()
            if outcome is False:
                raise ValueError("independent canonical replay builder is not cold")
            break
    for attribute in (
        "last_diagnostics",
        "_previous_anchor_frame",
        "_previous_iteration_frames",
        "_previous_successful_potential_mev",
    ):
        if hasattr(builder, attribute) and getattr(builder, attribute) is not None:
            raise ValueError(
                "independent canonical replay builder is warmed: "
                f"{attribute} is populated"
            )
    for attribute in ("diagnostics_history", "calls", "pair_resolved_history"):
        history = getattr(builder, attribute, None)
        if history is not None:
            try:
                history_length = len(history)
            except TypeError as exc:
                raise TypeError(
                    f"independent canonical replay builder {attribute} must be sized"
                ) from exc
            if history_length != 0:
                raise ValueError(
                    "independent canonical replay builder is warmed: "
                    f"{attribute} is nonempty"
                )


def _new_independent_kane_replay_builder(
    factory: IndependentKaneReplayBuilderFactory,
    *,
    parent_spec: KaneParentSourceSpec,
    forbidden_builders: tuple[object, ...],
) -> KaneWindowBuilder:
    candidate = factory()
    if not callable(candidate):
        raise TypeError(
            "independent_replay_builder_factory must return a callable builder"
        )
    candidate_objects = (candidate, getattr(candidate, "raw_builder", None))
    forbidden_objects = tuple(
        item
        for forbidden in forbidden_builders
        for item in (forbidden, getattr(forbidden, "raw_builder", None))
        if item is not None
    )
    if any(
        item is forbidden
        for item in candidate_objects
        if item is not None
        for forbidden in forbidden_objects
    ):
        raise ValueError(
            "independent_replay_builder_factory must return a fresh builder "
            "distinct from the iteration and other replay builders"
        )
    candidate_parent_spec = getattr(candidate, "parent_spec", None)
    if not isinstance(candidate_parent_spec, KaneParentSourceSpec):
        raise TypeError(
            "each independent canonical replay builder must expose typed parent_spec"
        )
    if candidate_parent_spec.fingerprint != parent_spec.fingerprint:
        raise ValueError(
            "independent canonical replay builder does not have the solver parent spec"
        )
    _assert_cold_kane_replay_builder(candidate)
    return candidate


def solve_canonical_split_zero_kane_poisson(
    builder: KaneWindowBuilder,
    *,
    independent_replay_builder_factory: IndependentKaneReplayBuilderFactory,
    parent_spec: KaneParentSourceSpec,
    initial_potential_mev: FloatArray,
    config: CanonicalKanePoissonConfig | None = None,
) -> CanonicalKanePoissonResult:
    """Solve and two-cold-builder replay a canonical periodic fixed point.

    Every potential iterate rebuilds the Kane window and solves exactly one
    common chemical potential from ``n_e(mu, U) = n_h(mu, U)``.  This API does
    not accept separate electron/hole chemical potentials or an externally
    imposed pair density.
    """

    if not callable(builder):
        raise TypeError("builder must be callable")
    if not callable(independent_replay_builder_factory):
        raise TypeError("independent_replay_builder_factory must be callable")

    solver_config = CanonicalKanePoissonConfig() if config is None else config
    z = np.asarray(parent_spec.z_nm, dtype=np.float64)
    z_weights = np.asarray(parent_spec.z_weights_nm, dtype=np.float64)
    epsilon = np.asarray(parent_spec.epsilon_r, dtype=np.float64)
    k_weights = np.asarray(parent_spec.k_weights_nm2, dtype=np.float64)
    potential = np.asarray(initial_potential_mev, dtype=np.float64).copy()
    if z.ndim != 1 or z_weights.shape != z.shape or epsilon.shape != z.shape:
        raise ValueError("z, z_weights, and epsilon_r must be matching one-dimensional arrays")
    if potential.shape != z.shape or not np.all(np.isfinite(potential)):
        raise ValueError("initial_potential_mev must be finite with shape (nz,)")
    if np.any(z_weights <= 0.0) or np.any(epsilon <= 0.0):
        raise ValueError("z weights and dielectric profile must be positive")
    spacing = np.diff(z)
    if spacing[0] <= 0.0 or not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError(
            "canonical periodic Kane--Poisson currently requires a strictly increasing uniform z grid"
        )
    if not np.allclose(z_weights, spacing[0], rtol=0.0, atol=1e-12):
        raise ValueError("canonical periodic Kane--Poisson requires uniform periodic z weights")
    if k_weights.ndim != 1 or np.any(k_weights <= 0.0) or not np.all(np.isfinite(k_weights)):
        raise ValueError("k_weights_nm2 must be a finite positive one-dimensional array")
    potential -= np.mean(potential)

    history: list[KanePoissonIterationRecord] = []
    previous_electron: FloatArray | None = None
    previous_hole: FloatArray | None = None
    last_source: KaneWindowAtPotential | None = None
    last_source_fingerprint: str | None = None
    last_reference: KaneChargeNeutralReference | None = None
    last_electron: FloatArray | None = None
    last_hole: FloatArray | None = None
    final_potential: FloatArray | None = None
    converged = False

    for iteration in range(1, solver_config.maximum_iterations + 1):
        source = builder(potential.copy())
        reference, electron, hole, source_fingerprint = _evaluate_source(
            source,
            input_potential_mev=potential,
            parent_spec=parent_spec,
            temperature_K=solver_config.temperature_K,
            root_tolerances=solver_config.root_tolerances,
            poisson_residual_tolerance_mev=solver_config.fixed_point_tolerance_mev,
        )
        poisson = periodic_variable_dielectric_poisson_map(
            z,
            electron,
            hole,
            epsilon,
            integrated_charge_tolerance_nm2=solver_config.integrated_charge_tolerance_nm2,
        )
        if poisson.gauss_residual_mev_nm2 > solver_config.gauss_residual_tolerance_mev_nm2:
            raise ValueError(
                "periodic Poisson Gauss residual exceeds tolerance: "
                f"{poisson.gauss_residual_mev_nm2:.3e} meV nm^-2"
            )
        fixed_point_residual = float(
            np.max(np.abs(poisson.electron_potential_mev - potential))
        )
        if previous_electron is None:
            density_change = None
        else:
            density_change = float(
                max(
                    np.max(np.abs(electron - previous_electron)),
                    np.max(np.abs(hole - previous_hole)),
                )
            )
        history.append(
            KanePoissonIterationRecord(
                iteration=iteration,
                input_potential_sha256=kane_poisson_array_sha256(potential),
                source_array_fingerprint=source_fingerprint,
                source_provenance_fingerprint=source.provenance_fingerprint,
                mu_mev=reference.mu_mev,
                electron_density_nm2=reference.electron_density_nm2,
                hole_density_nm2=reference.hole_density_nm2,
                fixed_point_residual_mev=fixed_point_residual,
                density_profile_change_nm3=density_change,
                gauss_residual_mev_nm2=poisson.gauss_residual_mev_nm2,
                integrated_net_number_nm2=poisson.integrated_net_number_nm2,
                removed_mean_net_number_nm3=poisson.removed_mean_net_number_nm3,
            )
        )
        last_source = source
        last_source_fingerprint = source_fingerprint
        last_reference = reference
        last_electron = electron
        last_hole = hole
        density_converged = (
            fixed_point_residual <= solver_config.fixed_point_tolerance_mev
            if density_change is None
            else density_change <= solver_config.density_profile_tolerance_nm3
        )
        if (
            fixed_point_residual <= solver_config.fixed_point_tolerance_mev
            and density_converged
        ):
            # The current input potential is the point certified by the
            # fixed-point residual. Do not silently promote its Poisson image:
            # F(F(U)) need not satisfy the same tolerance even when F(U)-U does.
            final_potential = potential.copy()
            converged = True
            break
        previous_electron = electron
        previous_hole = hole
        potential = (
            potential
            + solver_config.mixing * (poisson.electron_potential_mev - potential)
        )
        potential -= np.mean(potential)

    if (
        not converged
        or final_potential is None
        or last_source is None
        or last_source_fingerprint is None
        or last_reference is None
        or last_electron is None
        or last_hole is None
    ):
        final_residual = history[-1].fixed_point_residual_mev if history else np.inf
        raise RuntimeError(
            "canonical split-zero Kane--Poisson did not converge: "
            f"iterations={len(history)}, residual={final_residual:.3e} meV"
        )

    # Allocate and cold-check both builders at the exact certified final
    # iteration potential before either is called. This prevents one
    # replay from warming the state used by the other.
    first_replay_builder = _new_independent_kane_replay_builder(
        independent_replay_builder_factory,
        parent_spec=parent_spec,
        forbidden_builders=(builder,),
    )
    second_replay_builder = _new_independent_kane_replay_builder(
        independent_replay_builder_factory,
        parent_spec=parent_spec,
        forbidden_builders=(builder, first_replay_builder),
    )

    first_source = first_replay_builder(final_potential.copy())
    first_reference, first_electron, first_hole, _first_fingerprint = _evaluate_source(
        first_source,
        input_potential_mev=final_potential,
        parent_spec=parent_spec,
        temperature_K=solver_config.temperature_K,
        root_tolerances=solver_config.root_tolerances,
        poisson_residual_tolerance_mev=solver_config.fixed_point_tolerance_mev,
    )
    first_poisson = periodic_variable_dielectric_poisson_map(
        z,
        first_electron,
        first_hole,
        epsilon,
        integrated_charge_tolerance_nm2=solver_config.integrated_charge_tolerance_nm2,
    )
    first_residual = float(
        np.max(np.abs(first_poisson.electron_potential_mev - final_potential))
    )
    first_density_change = float(
        max(
            np.max(np.abs(first_electron - last_electron)),
            np.max(np.abs(first_hole - last_hole)),
        )
    )
    if _window_replay_metadata(last_source) != _window_replay_metadata(first_source):
        raise ValueError("warm-to-cold Kane replay changed typed source metadata")
    warm_spectrum = np.linalg.eigvalsh(
        np.moveaxis(last_source.hamiltonian_mev, 2, 0)
    )
    first_spectrum = np.linalg.eigvalsh(
        np.moveaxis(first_source.hamiltonian_mev, 2, 0)
    )
    warm_spectrum_error = float(np.max(np.abs(warm_spectrum - first_spectrum)))
    warm_subspace_error = _window_replay_subspace_error(
        last_source.micro_wavefunctions,
        first_source.micro_wavefunctions,
        z_weights,
    )
    warm_hamiltonian_error, warm_density_error = _window_replay_covariance_errors(
        last_source,
        first_source,
        last_reference,
        first_reference,
        z_weights,
    )
    warm_mu_error = abs(last_reference.mu_mev - first_reference.mu_mev)
    if (
        warm_spectrum_error > 1e-9
        or warm_subspace_error > 1e-8
        or warm_hamiltonian_error > 1e-8
        or warm_density_error > 1e-8
    ):
        raise ValueError(
            "warm-to-cold Kane replay changed spectrum/subspace/H/density covariance: "
            f"spectrum={warm_spectrum_error:.3e} meV, "
            f"subspace={warm_subspace_error:.3e}, "
            f"Hcov={warm_hamiltonian_error:.3e} meV, "
            f"Pcov={warm_density_error:.3e}"
        )
    if warm_mu_error > 1e-9:
        raise ValueError(
            "warm-to-cold Kane replay changed the neutrality-root chemical potential"
        )
    if first_residual > solver_config.fixed_point_tolerance_mev:
        raise ValueError(
            "first cold final-potential rebuild exceeds fixed-point tolerance"
        )
    if first_density_change > solver_config.density_profile_tolerance_nm3:
        raise ValueError(
            "first cold final-potential rebuild exceeds density-profile tolerance"
        )
    if first_poisson.gauss_residual_mev_nm2 > solver_config.gauss_residual_tolerance_mev_nm2:
        raise ValueError("first cold final-potential rebuild exceeds Gauss tolerance")

    replay_source = second_replay_builder(final_potential.copy())
    replay_reference, replay_electron, replay_hole, _replay_fingerprint = _evaluate_source(
        replay_source,
        input_potential_mev=final_potential,
        parent_spec=parent_spec,
        temperature_K=solver_config.temperature_K,
        root_tolerances=solver_config.root_tolerances,
        poisson_residual_tolerance_mev=solver_config.fixed_point_tolerance_mev,
    )
    if _window_replay_metadata(first_source) != _window_replay_metadata(replay_source):
        raise ValueError("independent Kane replay changed typed source metadata")

    first_spectrum = np.linalg.eigvalsh(
        np.moveaxis(first_source.hamiltonian_mev, 2, 0)
    )
    replay_spectrum = np.linalg.eigvalsh(
        np.moveaxis(replay_source.hamiltonian_mev, 2, 0)
    )
    spectrum_error = float(np.max(np.abs(first_spectrum - replay_spectrum)))
    subspace_error = _window_replay_subspace_error(
        first_source.micro_wavefunctions,
        replay_source.micro_wavefunctions,
        z_weights,
    )
    hamiltonian_covariance_error, density_covariance_error = (
        _window_replay_covariance_errors(
            first_source,
            replay_source,
            first_reference,
            replay_reference,
            z_weights,
        )
    )
    profile_error = float(
        max(
            np.max(np.abs(first_electron - replay_electron)),
            np.max(np.abs(first_hole - replay_hole)),
        )
    )
    mu_error = abs(first_reference.mu_mev - replay_reference.mu_mev)
    if (
        spectrum_error > 1e-9
        or subspace_error > 1e-8
        or hamiltonian_covariance_error > 1e-8
        or density_covariance_error > 1e-8
    ):
        raise ValueError(
            "independent Kane replay changed spectrum/subspace/H/density covariance: "
            f"spectrum={spectrum_error:.3e} meV, subspace={subspace_error:.3e}, "
            f"Hcov={hamiltonian_covariance_error:.3e} meV, "
            f"Pcov={density_covariance_error:.3e}"
        )
    if profile_error > solver_config.density_profile_tolerance_nm3 or mu_error > 1e-9:
        raise ValueError(
            "independent Kane replay changed mu or orbital-transfer profiles: "
            f"mu={mu_error:.3e} meV, profile={profile_error:.3e} nm^-3"
        )

    replay_poisson = periodic_variable_dielectric_poisson_map(
        z,
        replay_electron,
        replay_hole,
        epsilon,
        integrated_charge_tolerance_nm2=solver_config.integrated_charge_tolerance_nm2,
    )
    replay_residual = float(
        np.max(np.abs(replay_poisson.electron_potential_mev - final_potential))
    )
    poisson_map_error = float(
        np.max(
            np.abs(
                first_poisson.electron_potential_mev
                - replay_poisson.electron_potential_mev
            )
        )
    )
    if replay_residual > solver_config.fixed_point_tolerance_mev:
        raise ValueError("final Kane--Poisson replay exceeds fixed-point tolerance")
    if poisson_map_error > solver_config.fixed_point_tolerance_mev:
        raise ValueError("independent Kane replay changed the periodic Poisson map")
    if replay_poisson.gauss_residual_mev_nm2 > solver_config.gauss_residual_tolerance_mev_nm2:
        raise ValueError("final Kane--Poisson replay exceeds Gauss tolerance")

    final_contract = replace(
        replay_reference.contract,
        poisson_residual_mev=replay_residual,
        poisson_residual_tolerance_mev=solver_config.fixed_point_tolerance_mev,
    )
    replay_reference = replace(
        replay_reference,
        contract=final_contract,
        density=_readonly_copy(replay_reference.density, dtype=np.dtype(np.complex128)),
        energies_mev=_readonly_copy(replay_reference.energies_mev, dtype=np.dtype(np.float64)),
    )
    return CanonicalKanePoissonResult(
        config=solver_config,
        parent_spec=parent_spec,
        z_nm=z.copy(),
        z_weights_nm=z_weights.copy(),
        epsilon_r=epsilon.copy(),
        k_weights_nm2=k_weights.copy(),
        potential_mev=final_potential.copy(),
        electron_density_nm3=replay_electron.copy(),
        hole_density_nm3=replay_hole.copy(),
        reference=replay_reference,
        final_eigensystem=replay_source,
        history=tuple(history),
        fixed_point_residual_mev=replay_residual,
        gauss_residual_mev_nm2=replay_poisson.gauss_residual_mev_nm2,
        converged=True,
        replay_verified=True,
    )


def validate_canonical_kane_poisson_result(
    result: CanonicalKanePoissonResult,
    *,
    require_certified: bool = True,
) -> None:
    """Recompute the physical and numerical closure of one canonical result."""

    parent = result.parent_spec
    for result_name, parent_name in (
        ("z_nm", "z_nm"),
        ("z_weights_nm", "z_weights_nm"),
        ("epsilon_r", "epsilon_r"),
        ("k_weights_nm2", "k_weights_nm2"),
    ):
        if not np.array_equal(getattr(result, result_name), getattr(parent, parent_name)):
            raise ValueError(f"canonical result {result_name} changed from parent spec")
    source = result.final_eigensystem
    source.validate(input_potential_mev=result.potential_mev, parent_spec=parent)
    contract = result.reference.contract
    if (
        contract.selected_band_indices != source.selected_band_indices
        or contract.parent_hilbert_dimension != source.parent_hilbert_dimension
        or contract.calculation_split_mev != source.calculation_split_mev
        or contract.energy_zero_label != source.energy_zero_label
        or contract.electrostatic_ensemble != "canonical_neutral_slab"
    ):
        raise ValueError("canonical reference contract changed from final source")
    source_fingerprint = kane_window_source_fingerprint(
        source.hamiltonian_mev,
        parent.k_weights_nm2,
        source.micro_wavefunctions,
        parent.z_weights_nm,
    )
    if contract.source_fingerprint != source_fingerprint:
        raise ValueError("canonical reference source fingerprint mismatch")
    recomputed_reference = build_charge_neutral_kane_reference(
        source.hamiltonian_mev,
        parent.k_weights_nm2,
        source.micro_wavefunctions,
        parent.z_weights_nm,
        contract=contract,
        temperature_K=result.reference.temperature_K,
        tolerances=result.reference.tolerances,
    )
    if normal_reference_fingerprint(recomputed_reference) != normal_reference_fingerprint(
        result.reference
    ):
        raise ValueError("canonical reference failed exact typed reconstruction")
    electron, hole = kane_orbital_transfer_density_profiles(
        source.micro_wavefunctions,
        result.reference.density,
        parent.k_weights_nm2,
        parent.z_weights_nm,
        expected_electron_density_nm2=result.reference.electron_density_nm2,
        expected_hole_density_nm2=result.reference.hole_density_nm2,
    )
    profile_scale = max(
        1.0,
        float(np.max(np.abs(electron))),
        float(np.max(np.abs(hole))),
    )
    profile_atol = 64.0 * np.finfo(np.float64).eps * profile_scale
    if not np.allclose(
        result.electron_density_nm3, electron, rtol=0.0, atol=profile_atol
    ) or not np.allclose(
        result.hole_density_nm3, hole, rtol=0.0, atol=profile_atol
    ):
        raise ValueError("canonical archived carrier profiles changed")
    poisson = periodic_variable_dielectric_poisson_map(
        result.z_nm,
        electron,
        hole,
        result.epsilon_r,
        integrated_charge_tolerance_nm2=(
            result.config.integrated_charge_tolerance_nm2
        ),
    )
    fixed_point_residual = float(
        np.max(np.abs(poisson.electron_potential_mev - result.potential_mev))
    )
    scalar_atol = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        abs(fixed_point_residual),
        abs(poisson.gauss_residual_mev_nm2),
    )
    if not np.isclose(
        result.fixed_point_residual_mev,
        fixed_point_residual,
        rtol=0.0,
        atol=scalar_atol,
    ):
        raise ValueError("canonical fixed-point residual changed")
    if not np.isclose(
        result.gauss_residual_mev_nm2,
        poisson.gauss_residual_mev_nm2,
        rtol=0.0,
        atol=scalar_atol,
    ):
        raise ValueError("canonical Gauss residual changed")
    if contract.poisson_residual_mev is None or not np.isclose(
        contract.poisson_residual_mev,
        fixed_point_residual,
        rtol=0.0,
        atol=scalar_atol,
    ):
        raise ValueError("canonical reference Poisson residual changed")
    if require_certified and (
        result.converged is not True
        or result.replay_verified is not True
        or not result.reference.numerically_certified
        or fixed_point_residual > result.config.fixed_point_tolerance_mev
        or poisson.gauss_residual_mev_nm2
        > result.config.gauss_residual_tolerance_mev_nm2
    ):
        raise ValueError("canonical Kane--Poisson result is not certified")


def canonical_kane_poisson_fingerprint(result: CanonicalKanePoissonResult) -> str:
    """Fingerprint the validated complete Kane--Poisson state archive/checkpoint."""

    validate_canonical_kane_poisson_result(result, require_certified=True)
    digest = hashlib.sha256()
    digest.update(f"CanonicalKanePoissonResult:v{KANE_POISSON_SCHEMA_VERSION}".encode())
    digest.update(repr(result.config).encode())
    digest.update(result.parent_spec.fingerprint.encode())
    digest.update(normal_reference_fingerprint(result.reference).encode())
    digest.update(result.final_eigensystem.exact_archive_fingerprint.encode())
    for values in (
        result.z_nm,
        result.z_weights_nm,
        result.epsilon_r,
        result.k_weights_nm2,
        result.potential_mev,
        result.electron_density_nm3,
        result.hole_density_nm3,
        result.reference.density,
        result.reference.energies_mev,
        result.final_eigensystem.hamiltonian_mev,
        result.final_eigensystem.micro_wavefunctions,
    ):
        digest.update(kane_poisson_array_sha256(np.asarray(values)).encode())
    digest.update(result.claim_scope.encode())
    digest.update(repr(float(result.fixed_point_residual_mev)).encode())
    digest.update(repr(float(result.gauss_residual_mev_nm2)).encode())
    digest.update(repr(bool(result.converged)).encode())
    digest.update(repr(bool(result.replay_verified)).encode())
    digest.update(repr(result.history).encode())
    return digest.hexdigest()
