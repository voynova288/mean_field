"""Reusable time-dependent Hartree-Fock / RPA helpers.

The tensor convention used here is part of the public TDHF core contract:
``V[a, b, c, d]`` is the coefficient of both equivalent monomials

    c_b^† c_a^† c_c c_d  ==  c_a^† c_b^† c_d c_c.

Therefore the TDHF matrices are assembled as

    A[p h, p' h'] = (E[p] - E[h]) δ[p,p']δ[h,h']
                    + V[p,h',h,p'] - V[p,h',p',h]
    B[p h, p' h'] = V[p,p',h,h'] - V[p,p',h',h]

where every ``(p, h)`` pair must already belong to a fixed collective-momentum
sector.  Production system adapters should provide an on-demand matrix-element
callable instead of materializing the full four-index tensor.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import linalg as scipy_linalg

TDHF_TWO_BODY_CONVENTION = (
    "V[a,b,c,d] is the coefficient of c_b^† c_a^† c_c c_d "
    "(equivalently c_a^† c_b^† c_d c_c); pass un-antisymmetrized "
    "density-density matrix elements."
)

FlavorChannel = Literal[
    "intraflavor",
    "intervalley",
    "interspin",
    "inter_spin_valley",
]

TwoBodyMatrixInput = (
    np.ndarray
    | Mapping[tuple[int, int, int, int], complex]
    | Callable[[int, int, int, int], complex]
)


@dataclass(frozen=True)
class SpinValleyFlavor:
    """Minimal flavor tag for spin/valley TDHF sector classification."""

    spin: Hashable
    valley: Hashable


@dataclass(frozen=True)
class ParticleHolePair:
    """One particle-hole basis element inside a fixed collective-momentum sector.

    ``particle`` must label an unoccupied HF orbital and ``hole`` an occupied HF
    orbital.  Momentum/flavor metadata is optional and only used by helper
    routines for sector construction and classification.
    """

    particle: int
    hole: int
    particle_momentum: Hashable | None = None
    hole_momentum: Hashable | None = None
    particle_flavor: Any | None = None
    hole_flavor: Any | None = None


@dataclass(frozen=True)
class TDHFStructureResiduals:
    """Structure-check residuals for TDHF/RPA matrices."""

    a_hermitian: float
    b_symmetric: float
    particle_hole_symmetry: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return (
            self.a_hermitian <= self.tolerance
            and self.b_symmetric <= self.tolerance
            and self.particle_hole_symmetry <= self.tolerance
        )


@dataclass(frozen=True)
class TDHFMatrices:
    """Dense TDHF matrices for one already-filtered ph sector."""

    pairs: tuple[ParticleHolePair, ...]
    A: np.ndarray
    B: np.ndarray
    L: np.ndarray
    structure: TDHFStructureResiduals


@dataclass(frozen=True)
class TDHFSpectrum:
    """Positive-metric TDHF modes returned by :func:`solve_tdhf_liouvillian`.

    Arrays are mode-major: ``X[mode, pair]`` and ``Y[mode, pair]``.
    ``raw_eigenvalues`` contains the complete non-Hermitian spectrum for
    stability and +/- pairing diagnostics; ``eigenvalues`` contains only the
    selected positive branch.
    """

    eigenvalues: np.ndarray
    energies: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    eta_norms: np.ndarray
    residuals: np.ndarray
    selected_indices: np.ndarray
    raw_eigenvalues: np.ndarray
    pairing_residual: float

    @property
    def amplitudes(self) -> np.ndarray:
        return np.concatenate([self.X, self.Y], axis=1)


@dataclass(frozen=True)
class SingleFlavorSimplificationStatus:
    """Whether the conduction-only fully polarized TDHF shortcut may be used."""

    allowed: bool
    reason: str


def _as_complex_scalar(value: complex | np.number) -> complex:
    return complex(np.asarray(value).item())


def two_body_matrix_element(
    interaction: TwoBodyMatrixInput,
    a: int,
    b: int,
    c: int,
    d: int,
) -> complex:
    """Return ``V[a,b,c,d]`` from a dense tensor, sparse mapping, or callable."""

    if callable(interaction):
        return complex(interaction(a, b, c, d))
    if isinstance(interaction, Mapping):
        return complex(interaction.get((a, b, c, d), 0.0))
    array = np.asarray(interaction)
    if array.ndim != 4:
        raise ValueError("dense TDHF interaction tensor must have four axes")
    return _as_complex_scalar(array[a, b, c, d])


def transform_dense_two_body_to_hf_basis(
    orbital_interaction: np.ndarray,
    hf_coefficients: np.ndarray,
) -> np.ndarray:
    """Transform a small dense orbital-basis tensor to the HF basis.

    This helper is intended for smoke tests and debugging.  Production TDHF
    adapters should usually provide an on-demand callable because the dense
    four-index tensor is too large for realistic moire grids.

    ``hf_coefficients[alpha, i]`` implements
    ``d_alpha^† = sum_i hf_coefficients[alpha, i] c_i^†``.
    """

    interaction = np.asarray(orbital_interaction, dtype=np.complex128)
    coeffs = np.asarray(hf_coefficients, dtype=np.complex128)
    if interaction.ndim != 4:
        raise ValueError("orbital_interaction must be a dense four-index tensor")
    if coeffs.ndim != 2:
        raise ValueError("hf_coefficients must have shape (n_hf, n_orbital)")
    if interaction.shape != (coeffs.shape[1],) * 4:
        raise ValueError(
            "orbital_interaction axes must match the orbital axis of hf_coefficients"
        )
    return np.einsum(
        "ijkl,ai,bj,ck,dl->abcd",
        interaction,
        np.conj(coeffs),
        np.conj(coeffs),
        coeffs,
        coeffs,
        optimize=True,
    )


def build_all_particle_hole_pairs(
    occupied: Sequence[int],
    unoccupied: Sequence[int],
    *,
    flavors: Sequence[Any] | Mapping[int, Any] | None = None,
    momenta: Sequence[Hashable] | Mapping[int, Hashable] | None = None,
) -> tuple[ParticleHolePair, ...]:
    """Build all unoccupied x occupied pairs for toy/no-translation models.

    Do not use this helper for production translationally invariant TDHF unless
    the input lists have already been filtered to one fixed collective momentum
    sector.  For moire momentum-sector construction use
    :func:`build_momentum_sector_particle_hole_pairs`.
    """

    return tuple(
        ParticleHolePair(
            particle=int(particle),
            hole=int(hole),
            particle_momentum=_lookup_optional(momenta, int(particle)),
            hole_momentum=_lookup_optional(momenta, int(hole)),
            particle_flavor=_lookup_optional(flavors, int(particle)),
            hole_flavor=_lookup_optional(flavors, int(hole)),
        )
        for hole in occupied
        for particle in unoccupied
    )


def build_momentum_sector_particle_hole_pairs(
    occupied_by_momentum: Mapping[Hashable, Sequence[int]],
    unoccupied_by_momentum: Mapping[Hashable, Sequence[int]],
    momentum_transfer: Hashable,
    add_momentum: Callable[[Hashable, Hashable], Hashable],
    *,
    flavors: Sequence[Any] | Mapping[int, Any] | None = None,
) -> tuple[ParticleHolePair, ...]:
    """Build ph pairs ``(k+q, particle; k, hole)`` for one momentum sector.

    ``add_momentum(k, q)`` must include the system adapter's periodic-gauge
    mapping back to the discrete mBZ grid.  Missing target momenta simply
    contribute no pairs, making non-grid q points explicit instead of silently
    nearest-neighbor sampled.
    """

    pairs: list[ParticleHolePair] = []
    for hole_momentum, holes in occupied_by_momentum.items():
        particle_momentum = add_momentum(hole_momentum, momentum_transfer)
        particles = unoccupied_by_momentum.get(particle_momentum, ())
        for hole in holes:
            for particle in particles:
                pairs.append(
                    ParticleHolePair(
                        particle=int(particle),
                        hole=int(hole),
                        particle_momentum=particle_momentum,
                        hole_momentum=hole_momentum,
                        particle_flavor=_lookup_optional(flavors, int(particle)),
                        hole_flavor=_lookup_optional(flavors, int(hole)),
                    )
                )
    return tuple(pairs)


def _lookup_optional(
    values: Sequence[Any] | Mapping[int, Any] | None,
    index: int,
) -> Any | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        return values.get(index)
    return values[index]


def assemble_tdhf_liouvillian(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Assemble ``L = [[A, B], [-B*, -A*]]`` as an ordinary matrix."""

    a_matrix = np.asarray(A, dtype=np.complex128)
    b_matrix = np.asarray(B, dtype=np.complex128)
    if a_matrix.shape != b_matrix.shape or a_matrix.ndim != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
        raise ValueError("A and B must be square matrices with the same shape")
    return np.block(
        [
            [a_matrix, b_matrix],
            [-np.conj(b_matrix), -np.conj(a_matrix)],
        ]
    )


def build_tdhf_matrices(
    energies: Sequence[float] | np.ndarray,
    pairs: Sequence[ParticleHolePair | tuple[int, int]],
    interaction: TwoBodyMatrixInput,
    *,
    include_direct_terms: bool = True,
    include_exchange_terms: bool = True,
    include_b_terms: bool = True,
    structure_tolerance: float = 1.0e-6,
    raise_on_structure_error: bool = False,
) -> TDHFMatrices:
    """Construct dense ``A``, ``B`` and ``L`` for one ph sector.

    ``pairs`` must already be the fixed-q particle-hole basis.  The one-body
    Hamiltonian is not used; the only single-particle input is the converged HF
    energy array.
    """

    energy_array = np.asarray(energies, dtype=float)
    if energy_array.ndim != 1:
        raise ValueError("TDHF energies must be a one-dimensional HF spectrum")
    ph_pairs = tuple(_coerce_pair(pair) for pair in pairs)
    n_pairs = len(ph_pairs)
    A = np.zeros((n_pairs, n_pairs), dtype=np.complex128)
    B = np.zeros((n_pairs, n_pairs), dtype=np.complex128)

    for row, pair in enumerate(ph_pairs):
        p = pair.particle
        h = pair.hole
        if p < 0 or h < 0 or p >= energy_array.size or h >= energy_array.size:
            raise IndexError("particle/hole index is outside the energy array")
        for col, other in enumerate(ph_pairs):
            pp = other.particle
            hp = other.hole
            if pp < 0 or hp < 0 or pp >= energy_array.size or hp >= energy_array.size:
                raise IndexError("particle/hole index is outside the energy array")
            if p == pp and h == hp:
                A[row, col] += energy_array[p] - energy_array[h]
            if include_direct_terms:
                A[row, col] += two_body_matrix_element(interaction, p, hp, h, pp)
                if include_b_terms:
                    B[row, col] += two_body_matrix_element(interaction, p, pp, h, hp)
            if include_exchange_terms:
                A[row, col] -= two_body_matrix_element(interaction, p, hp, pp, h)
                if include_b_terms:
                    B[row, col] -= two_body_matrix_element(interaction, p, pp, hp, h)

    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(
        A,
        B,
        L,
        tolerance=structure_tolerance,
        raise_on_fail=raise_on_structure_error,
    )
    return TDHFMatrices(pairs=ph_pairs, A=A, B=B, L=L, structure=structure)


def _coerce_pair(pair: ParticleHolePair | tuple[int, int]) -> ParticleHolePair:
    if isinstance(pair, ParticleHolePair):
        return pair
    if len(pair) != 2:
        raise ValueError("tuple particle-hole pairs must be (particle, hole)")
    return ParticleHolePair(particle=int(pair[0]), hole=int(pair[1]))


def validate_tdhf_structures(
    A: np.ndarray,
    B: np.ndarray,
    L: np.ndarray | None = None,
    *,
    tolerance: float = 1.0e-6,
    raise_on_fail: bool = False,
) -> TDHFStructureResiduals:
    """Check ``A=A†``, ``B=Bᵀ`` and ``L=-sigma_x L* sigma_x``."""

    a_matrix = np.asarray(A, dtype=np.complex128)
    b_matrix = np.asarray(B, dtype=np.complex128)
    if a_matrix.shape != b_matrix.shape or a_matrix.ndim != 2 or a_matrix.shape[0] != a_matrix.shape[1]:
        raise ValueError("A and B must be square matrices with the same shape")
    l_matrix = assemble_tdhf_liouvillian(a_matrix, b_matrix) if L is None else np.asarray(L, dtype=np.complex128)
    n_pairs = a_matrix.shape[0]
    if l_matrix.shape != (2 * n_pairs, 2 * n_pairs):
        raise ValueError("L must have shape (2*n_pairs, 2*n_pairs)")

    a_residual = _max_abs(a_matrix - np.conj(a_matrix.T))
    b_residual = _max_abs(b_matrix - b_matrix.T)
    block_residuals = (
        _max_abs(l_matrix[:n_pairs, :n_pairs] + np.conj(l_matrix[n_pairs:, n_pairs:])),
        _max_abs(l_matrix[:n_pairs, n_pairs:] + np.conj(l_matrix[n_pairs:, :n_pairs])),
        _max_abs(l_matrix[n_pairs:, :n_pairs] + np.conj(l_matrix[:n_pairs, n_pairs:])),
        _max_abs(l_matrix[n_pairs:, n_pairs:] + np.conj(l_matrix[:n_pairs, :n_pairs])),
    )
    ph_residual = max(block_residuals)
    residuals = TDHFStructureResiduals(
        a_hermitian=a_residual,
        b_symmetric=b_residual,
        particle_hole_symmetry=ph_residual,
        tolerance=float(tolerance),
    )
    if raise_on_fail and not residuals.ok:
        raise ValueError(
            "TDHF structure residuals exceed tolerance: "
            f"A Hermitian={a_residual:.3e}, "
            f"B symmetric={b_residual:.3e}, "
            f"particle-hole={ph_residual:.3e}, tolerance={tolerance:.3e}"
        )
    return residuals


def _max_abs(array: np.ndarray) -> float:
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def eta_metric(n_pairs: int) -> np.ndarray:
    """Return the diagonal indefinite TDHF metric ``diag(+1, -1)``."""

    if n_pairs < 0:
        raise ValueError("n_pairs must be non-negative")
    return np.concatenate([np.ones(n_pairs), -np.ones(n_pairs)])


def eta_inner(left: np.ndarray, right: np.ndarray, n_pairs: int) -> complex:
    """Compute ``left† eta right`` without materializing eta."""

    lvec = np.asarray(left, dtype=np.complex128)
    rvec = np.asarray(right, dtype=np.complex128)
    if lvec.shape != rvec.shape or lvec.ndim != 1 or lvec.size != 2 * n_pairs:
        raise ValueError("eta_inner expects vectors of shape (2*n_pairs,)")
    return np.vdot(lvec[:n_pairs], rvec[:n_pairs]) - np.vdot(lvec[n_pairs:], rvec[n_pairs:])


def tdhf_metric_gram(vectors: np.ndarray, n_pairs: int | None = None) -> np.ndarray:
    """Return the eta-Gram matrix for mode-major vectors."""

    matrix = np.asarray(vectors, dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("vectors must have shape (n_modes, 2*n_pairs)")
    if n_pairs is None:
        if matrix.shape[1] % 2:
            raise ValueError("cannot infer n_pairs from an odd vector dimension")
        n_pairs = matrix.shape[1] // 2
    gram = np.empty((matrix.shape[0], matrix.shape[0]), dtype=np.complex128)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            gram[i, j] = eta_inner(matrix[i], matrix[j], n_pairs)
    return gram


def eigenvalue_pairing_residual(eigenvalues: np.ndarray) -> float:
    """Max distance from each eigenvalue to its particle-hole partner ``-lambda*``."""

    values = np.asarray(eigenvalues, dtype=np.complex128).reshape(-1)
    if values.size == 0:
        return 0.0
    distances = []
    for value in values:
        distances.append(float(np.min(np.abs(values + np.conj(value)))))
    return max(distances)


def solve_tdhf_matrices(
    matrices: TDHFMatrices,
    **kwargs: Any,
) -> TDHFSpectrum:
    """Solve a :class:`TDHFMatrices` object."""

    return solve_tdhf_liouvillian(matrices.L, n_pairs=len(matrices.pairs), **kwargs)


def solve_tdhf_liouvillian(
    L: np.ndarray,
    *,
    n_pairs: int | None = None,
    energy_tol: float = 1.0e-10,
    imag_tol: float = 1.0e-8,
    norm_tol: float = 1.0e-10,
    degeneracy_tol: float = 1.0e-8,
    include_zero_modes: bool = False,
    include_complex_modes: bool = False,
) -> TDHFSpectrum:
    """Diagonalize TDHF ``L`` and return positive-metric modes.

    This is an ordinary non-Hermitian eigenproblem, not ``eig(A, B)``.  The
    selected modes are eta-orthonormalized within near-degenerate eigenspaces.
    Modes with non-positive eta norm are treated as the conjugate branch and are
    not returned.  Set ``include_complex_modes=True`` only for diagnostics of
    unstable finite-q sectors; the returned ``energies`` are still the real
    parts of the selected eigenvalues.
    """

    matrix = np.asarray(L, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 2:
        raise ValueError("TDHF L must be a square matrix with even dimension")
    inferred_pairs = matrix.shape[0] // 2
    if n_pairs is None:
        n_pairs = inferred_pairs
    if n_pairs != inferred_pairs:
        raise ValueError("n_pairs does not match the Liouvillian dimension")

    raw_eigenvalues, raw_eigenvectors = scipy_linalg.eig(matrix)
    real_parts = np.real(raw_eigenvalues)
    imag_parts = np.imag(raw_eigenvalues)
    real_mask = real_parts >= (-energy_tol if include_zero_modes else energy_tol)
    imag_mask = np.ones_like(real_mask, dtype=bool) if include_complex_modes else np.abs(imag_parts) <= imag_tol
    candidate_indices = np.nonzero(real_mask & imag_mask)[0]
    if candidate_indices.size:
        order = np.lexsort((np.abs(imag_parts[candidate_indices]), real_parts[candidate_indices]))
        candidate_indices = candidate_indices[order]

    selected_vectors: list[np.ndarray] = []
    selected_eigenvalues: list[complex] = []
    selected_indices: list[int] = []
    residuals: list[float] = []

    for group in _degenerate_index_groups(raw_eigenvalues, candidate_indices, degeneracy_tol):
        vectors, group_eigenvalues, source_indices = _positive_metric_basis_for_group(
            raw_eigenvectors[:, group],
            raw_eigenvalues[group],
            group,
            n_pairs,
            norm_tol,
        )
        for vector, eigenvalue, source_index in zip(vectors, group_eigenvalues, source_indices, strict=True):
            selected_vectors.append(vector)
            selected_eigenvalues.append(eigenvalue)
            selected_indices.append(int(source_index))
            residuals.append(float(np.linalg.norm(matrix @ vector - eigenvalue * vector)))

    if selected_vectors:
        modes = np.vstack(selected_vectors)
        eigenvalues = np.asarray(selected_eigenvalues, dtype=np.complex128)
        sort_order = np.lexsort((np.abs(np.imag(eigenvalues)), np.real(eigenvalues)))
        modes = modes[sort_order]
        eigenvalues = eigenvalues[sort_order]
        selected_index_array = np.asarray(selected_indices, dtype=int)[sort_order]
        residual_array = np.asarray(residuals, dtype=float)[sort_order]
    else:
        modes = np.empty((0, 2 * n_pairs), dtype=np.complex128)
        eigenvalues = np.empty((0,), dtype=np.complex128)
        selected_index_array = np.empty((0,), dtype=int)
        residual_array = np.empty((0,), dtype=float)

    eta_norms = np.asarray(
        [float(np.real(eta_inner(mode, mode, n_pairs))) for mode in modes],
        dtype=float,
    )
    return TDHFSpectrum(
        eigenvalues=eigenvalues,
        energies=np.real(eigenvalues),
        X=modes[:, :n_pairs].copy(),
        Y=modes[:, n_pairs:].copy(),
        eta_norms=eta_norms,
        residuals=residual_array,
        selected_indices=selected_index_array,
        raw_eigenvalues=raw_eigenvalues,
        pairing_residual=eigenvalue_pairing_residual(raw_eigenvalues),
    )


def _degenerate_index_groups(
    eigenvalues: np.ndarray,
    indices: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        previous = groups[-1][-1]
        if abs(eigenvalues[index] - eigenvalues[previous]) <= tolerance:
            groups[-1].append(int(index))
        else:
            groups.append([int(index)])
    return [np.asarray(group, dtype=int) for group in groups]


def _positive_metric_basis_for_group(
    raw_vectors: np.ndarray,
    raw_eigenvalues: np.ndarray,
    source_indices: np.ndarray,
    n_pairs: int,
    norm_tol: float,
) -> tuple[list[np.ndarray], list[complex], list[int]]:
    group_size = raw_vectors.shape[1]
    metric_gram = np.empty((group_size, group_size), dtype=np.complex128)
    for i in range(group_size):
        for j in range(group_size):
            metric_gram[i, j] = eta_inner(raw_vectors[:, i], raw_vectors[:, j], n_pairs)
    metric_gram = 0.5 * (metric_gram + np.conj(metric_gram.T))
    metric_evals, metric_vecs = scipy_linalg.eigh(metric_gram)
    order = np.argsort(metric_evals)[::-1]

    vectors: list[np.ndarray] = []
    eigenvalues: list[complex] = []
    indices: list[int] = []
    representative_eigenvalue = complex(np.mean(raw_eigenvalues))
    representative_index = int(source_indices[0])
    for metric_index in order:
        metric_eval = float(np.real(metric_evals[metric_index]))
        if metric_eval <= norm_tol:
            continue
        vector = raw_vectors @ metric_vecs[:, metric_index]
        vector = vector / np.sqrt(metric_eval)
        vector = _fix_mode_phase(vector)
        vectors.append(vector)
        eigenvalues.append(representative_eigenvalue)
        indices.append(representative_index)
    return vectors, eigenvalues, indices


def _fix_mode_phase(vector: np.ndarray) -> np.ndarray:
    if vector.size == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) == 0.0:
        return vector
    phase = np.exp(-1j * np.angle(vector[pivot]))
    return vector * phase


def flavor_quantum_numbers(flavor: Any) -> SpinValleyFlavor:
    """Coerce common flavor tags to ``SpinValleyFlavor``.

    Accepted forms are ``SpinValleyFlavor``, objects with ``spin``/``valley``
    attributes, mappings with ``"spin"``/``"valley"`` keys, and two-tuples
    ``(spin, valley)``.
    """

    if isinstance(flavor, SpinValleyFlavor):
        return flavor
    if isinstance(flavor, Mapping):
        return SpinValleyFlavor(spin=flavor["spin"], valley=flavor["valley"])
    if hasattr(flavor, "spin") and hasattr(flavor, "valley"):
        return SpinValleyFlavor(spin=getattr(flavor, "spin"), valley=getattr(flavor, "valley"))
    if isinstance(flavor, tuple) and len(flavor) == 2:
        return SpinValleyFlavor(spin=flavor[0], valley=flavor[1])
    raise ValueError(
        "flavor tags must provide spin and valley, e.g. SpinValleyFlavor or (spin, valley)"
    )


def classify_flavor_channel(
    particle_flavor: Any,
    hole_flavor: Any,
) -> FlavorChannel:
    """Classify a ph pair as intra/intervalley/interspin/inter-spin-valley."""

    particle = flavor_quantum_numbers(particle_flavor)
    hole = flavor_quantum_numbers(hole_flavor)
    same_spin = particle.spin == hole.spin
    same_valley = particle.valley == hole.valley
    if same_spin and same_valley:
        return "intraflavor"
    if same_spin and not same_valley:
        return "intervalley"
    if not same_spin and same_valley:
        return "interspin"
    return "inter_spin_valley"


def split_pair_indices_by_flavor_channel(
    pairs: Sequence[ParticleHolePair],
) -> dict[FlavorChannel, np.ndarray]:
    """Return pair indices grouped by flavor charge channel."""

    groups: dict[FlavorChannel, list[int]] = {
        "intraflavor": [],
        "intervalley": [],
        "interspin": [],
        "inter_spin_valley": [],
    }
    for index, pair in enumerate(pairs):
        if pair.particle_flavor is None or pair.hole_flavor is None:
            raise ValueError("all pairs must carry particle_flavor and hole_flavor metadata")
        groups[classify_flavor_channel(pair.particle_flavor, pair.hole_flavor)].append(index)
    return {key: np.asarray(value, dtype=int) for key, value in groups.items()}


def restrict_tdhf_matrices(
    matrices: TDHFMatrices,
    pair_indices: Sequence[int] | np.ndarray,
) -> TDHFMatrices:
    """Restrict dense TDHF matrices to a subset of ph-pair indices."""

    indices = np.asarray(pair_indices, dtype=int)
    pairs = tuple(matrices.pairs[int(index)] for index in indices)
    A = matrices.A[np.ix_(indices, indices)].copy()
    B = matrices.B[np.ix_(indices, indices)].copy()
    L = assemble_tdhf_liouvillian(A, B)
    structure = validate_tdhf_structures(A, B, L, tolerance=matrices.structure.tolerance)
    return TDHFMatrices(pairs=pairs, A=A, B=B, L=L, structure=structure)


def check_single_flavor_simplification(
    *,
    active_space_has_valence: bool,
    occupied_flavor_counts: Mapping[Any, int],
    polarized_flavor: Any,
) -> SingleFlavorSimplificationStatus:
    """Check whether the conduction-only fully polarized shortcut is legal.

    The shortcut corresponds to the plan's special case: no valence bands in the
    active space, exactly one doped spin-valley flavor has occupied active
    states, and every other active flavor has zero occupied states.  Only in this
    case may intervalley/interspin blocks be built with ``B=0`` and without the
    direct A1 term.
    """

    if active_space_has_valence:
        return SingleFlavorSimplificationStatus(False, "active space contains valence bands")
    if polarized_flavor not in occupied_flavor_counts:
        return SingleFlavorSimplificationStatus(False, "polarized flavor is absent")
    if occupied_flavor_counts[polarized_flavor] <= 0:
        return SingleFlavorSimplificationStatus(False, "polarized flavor has no occupied active states")
    extra_occupied = {
        flavor: count
        for flavor, count in occupied_flavor_counts.items()
        if flavor != polarized_flavor and count != 0
    }
    if extra_occupied:
        return SingleFlavorSimplificationStatus(
            False,
            f"non-polarized flavors have occupied active states: {extra_occupied}",
        )
    return SingleFlavorSimplificationStatus(True, "conduction-only fully polarized active space")
