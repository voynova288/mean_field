from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os

import numpy as np
from scipy.linalg import eigh
from scipy.special import expit, xlogy

from .density import ket_projector_to_stored_orientation, stored_orientation_to_ket_projector


def find_chemical_potential(energies: np.ndarray, filling_fraction: float) -> float:
    flattened = np.sort(np.ravel(energies))
    occupancies = np.arange(1, flattened.size + 1, dtype=float) / float(flattened.size)
    idx = 0
    while idx < flattened.size - 1 and filling_fraction > occupancies[idx]:
        idx += 1
    if idx < flattened.size - 1:
        return float((flattened[idx + 1] + flattened[idx]) / 2.0)
    return float(flattened[idx])


def occupied_state_linear_indices(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    flattened = np.ravel(np.asarray(energies, dtype=float), order="F")
    if total_occupied <= 0:
        return np.empty(0, dtype=int)
    if total_occupied >= flattened.size:
        return np.arange(flattened.size, dtype=int)
    # Match Julia's column-major `sortperm` tie-breaking for near-degenerate occupancies.
    return np.argsort(flattened, kind="stable")[:total_occupied]


def occupied_state_mask(energies: np.ndarray, total_occupied: int) -> np.ndarray:
    occupied = occupied_state_linear_indices(energies, total_occupied)
    mask = np.zeros(energies.size, dtype=bool)
    mask[occupied] = True
    return mask.reshape(energies.shape, order="F")


@dataclass(frozen=True)
class GlobalOccupationResult:
    """One-chemical-potential occupation result over all momentum blocks."""

    projector_ket: np.ndarray
    energies: np.ndarray
    chemical_potential: float
    occupied_count: int
    fermi_gap: float


@dataclass(frozen=True)
class FixedRankOccupationResult:
    """Zero-temperature projector with the same occupied rank in every K block."""

    projector_ket: np.ndarray
    energies: np.ndarray
    chemical_potential: float
    occupied_per_k: int
    minimum_direct_gap: float
    indirect_gap: float


@dataclass(frozen=True)
class FermionicDensityDiagnostics:
    """Weighted particle number, entropy, and spectrum of a ket density."""

    particle_number: float
    entropy_dimensionless: float
    eigenvalue_min: float
    eigenvalue_max: float
    hermiticity_residual: float
    k_weights: np.ndarray


def _validate_outer_eigensolver_workers(eigensolver_workers: int) -> None:
    """Fail closed on invalid workers or nested numerical threading."""

    if type(eigensolver_workers) is not int or eigensolver_workers <= 0:
        raise TypeError("eigensolver_workers must be an exact positive integer")
    if eigensolver_workers == 1:
        return
    required = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
    invalid = {
        name: os.environ.get(name)
        for name in required
        if os.environ.get(name) != "1"
    }
    if invalid:
        raise RuntimeError(
            "outer eigensolver parallelism requires explicit one-thread "
            f"BLAS/OpenMP binding; invalid environment: {invalid}"
        )
    for name in ("MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        value = os.environ.get(name)
        if value not in (None, "1"):
            raise RuntimeError(
                "outer eigensolver parallelism forbids nested numerical "
                f"threads; {name}={value!r}"
            )


def fermionic_density_diagnostics(
    density_ket: np.ndarray,
    *,
    k_weights: np.ndarray | None = None,
    spectrum_tolerance: float = 1.0e-10,
    eigensolver_workers: int = 1,
) -> FermionicDensityDiagnostics:
    """Evaluate finite-temperature density diagnostics with explicit weights."""

    density = np.asarray(density_ket, dtype=np.complex128)
    if (
        density.ndim != 3
        or density.shape[1] != density.shape[2]
        or density.shape[0] == 0
        or density.shape[1] == 0
        or not np.all(np.isfinite(density))
    ):
        raise ValueError("density_ket must be finite and nonempty with shape (nk,nb,nb)")
    tolerance = float(spectrum_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("spectrum_tolerance must be finite and nonnegative")
    nk = density.shape[0]
    if k_weights is None:
        weights = np.ones(nk, dtype=float)
    else:
        weights = np.asarray(k_weights, dtype=float)
        if weights.shape != (nk,):
            raise ValueError(f"k_weights must have shape {(nk,)}, got {weights.shape}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("k_weights must be finite and nonnegative")
        weights = weights.copy()
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("k_weights must contain positive total weight")

    scale = max(1.0, float(np.max(np.abs(density))))
    hermiticity_residual = float(
        np.max(np.abs(density - density.conj().transpose(0, 2, 1)))
    )
    if hermiticity_residual > 64.0 * np.finfo(float).eps * scale:
        raise ValueError("density_ket must be Hermitian before thermodynamic diagnostics")

    _validate_outer_eigensolver_workers(eigensolver_workers)
    nb = density.shape[1]
    eigenvalues_by_k = np.empty((nk, nb), dtype=float)

    def diagonalize(k: int) -> tuple[int, np.ndarray]:
        # Retain the legacy batched-gufunc LAPACK path even for one K block;
        # ``eigvalsh(density[k])`` can differ by one ulp from the historical
        # ``eigvalsh(density)`` result on the same matrix.
        return k, np.linalg.eigvalsh(density[k : k + 1])[0]

    if eigensolver_workers == 1:
        results = map(diagonalize, range(nk))
        for k, values in results:
            eigenvalues_by_k[k] = values
    else:
        with ThreadPoolExecutor(
            max_workers=min(eigensolver_workers, nk)
        ) as executor:
            for k, values in executor.map(diagonalize, range(nk)):
                eigenvalues_by_k[k] = values

    # Keep the historical non-contiguous transpose view: its strides determine
    # the exact optimized-einsum reduction path used for N and entropy.
    eigenvalues = eigenvalues_by_k.T
    eigenvalue_min = float(np.min(eigenvalues))
    eigenvalue_max = float(np.max(eigenvalues))
    if eigenvalue_min < -tolerance or eigenvalue_max > 1.0 + tolerance:
        raise ValueError(
            "density_ket spectrum lies outside [0,1] beyond spectrum_tolerance"
        )
    clipped = np.clip(eigenvalues, 0.0, 1.0)
    entropy_terms = xlogy(clipped, clipped) + xlogy(1.0 - clipped, 1.0 - clipped)
    return FermionicDensityDiagnostics(
        particle_number=float(np.einsum("k,bk->", weights, clipped, optimize=True)),
        entropy_dimensionless=-float(
            np.einsum("k,bk->", weights, entropy_terms, optimize=True)
        ),
        eigenvalue_min=eigenvalue_min,
        eigenvalue_max=eigenvalue_max,
        hermiticity_residual=hermiticity_residual,
        k_weights=weights,
    )


def helmholtz_free_energy_ev(
    internal_energy_ev: float,
    *,
    kbt_ev: float,
    entropy_dimensionless: float,
) -> float:
    """Return ``F = E - k_B T S/k_B`` in eV with explicit finite values."""

    energy = float(internal_energy_ev)
    temperature = float(kbt_ev)
    entropy = float(entropy_dimensionless)
    if not math.isfinite(energy):
        raise ValueError("internal_energy_ev must be finite")
    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("kbt_ev must be finite and nonnegative")
    if not math.isfinite(entropy) or entropy < 0.0:
        raise ValueError("entropy_dimensionless must be finite and nonnegative")
    return float(energy - temperature * entropy)


@dataclass(frozen=True)
class GlobalFermiOccupationResult:
    """Finite-temperature one-chemical-potential occupation result.

    ``k_weights`` and ``target_particle_number`` use the same explicit
    normalization: the solved constraint is
    ``sum_k weight[k] * sum_band occupation[band,k] = target``.
    ``entropy_dimensionless`` is the correspondingly weighted fermionic
    entropy with Boltzmann's constant omitted.
    """

    density_ket: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    chemical_potential: float
    particle_number: float
    target_particle_number: float
    particle_residual: float
    entropy_dimensionless: float
    k_weights: np.ndarray
    kbt_ev: float

    def helmholtz_free_energy_ev(self, internal_energy_ev: float) -> float:
        """Combine a consistently normalized internal energy with this entropy."""

        return helmholtz_free_energy_ev(
            internal_energy_ev,
            kbt_ev=self.kbt_ev,
            entropy_dimensionless=self.entropy_dimensionless,
        )


def _diagonalize_hermitian_blocks(
    hamiltonian: np.ndarray, *, eigensolver_workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize independent K blocks with deterministic result placement."""

    _validate_outer_eigensolver_workers(eigensolver_workers)
    nk, nb, _ = hamiltonian.shape
    energies = np.empty((nb, nk), dtype=float)
    vectors = np.empty((nk, nb, nb), dtype=np.complex128)

    def diagonalize(k: int) -> tuple[int, np.ndarray, np.ndarray]:
        values, columns = np.linalg.eigh(hamiltonian[k])
        return k, values, columns

    if eigensolver_workers == 1:
        results = map(diagonalize, range(nk))
        for k, values, columns in results:
            energies[:, k] = values
            vectors[k] = columns
    else:
        with ThreadPoolExecutor(
            max_workers=min(eigensolver_workers, nk)
        ) as executor:
            for k, values, columns in executor.map(diagonalize, range(nk)):
                energies[:, k] = values
                vectors[k] = columns
    return energies, vectors


def fixed_rank_occupations(
    hamiltonian_ket: np.ndarray,
    *,
    occupied_per_k: int,
    direct_gap_tolerance: float,
    eigensolver_workers: int = 1,
) -> FixedRankOccupationResult:
    """Build a fixed-rank projector independently in every momentum block.

    The rank boundary must remain directly open at every represented K point.
    The indirect gap is reported but is not used to replace the declared
    pointwise-rank occupation contract.
    """

    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[1] != hamiltonian.shape[2]
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("hamiltonian_ket must be finite with shape (nk,nb,nb)")
    nk, nb, _ = hamiltonian.shape
    if type(occupied_per_k) is not int or not (0 < occupied_per_k < nb):
        raise ValueError("occupied_per_k must be an exact integer inside the basis")
    tolerance = float(direct_gap_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("direct_gap_tolerance must be finite and positive")
    scale = max(1.0, float(np.max(np.abs(hamiltonian), initial=0.0)))
    residual = float(
        np.max(
            np.abs(hamiltonian - hamiltonian.conj().transpose(0, 2, 1)),
            initial=0.0,
        )
    )
    if residual > 64.0 * np.finfo(float).eps * scale:
        raise ValueError("hamiltonian_ket must be Hermitian before occupation")
    energies, vectors = _diagonalize_hermitian_blocks(
        hamiltonian, eigensolver_workers=eigensolver_workers
    )
    occupied_columns = vectors[:, :, :occupied_per_k]
    projector = occupied_columns @ occupied_columns.conj().transpose(0, 2, 1)
    direct_gaps = energies[occupied_per_k] - energies[occupied_per_k - 1]
    minimum_direct_gap = float(np.min(direct_gaps))
    if minimum_direct_gap <= tolerance:
        raise RuntimeError(
            "closed fixed-rank occupation boundary: "
            f"minimum direct gap={minimum_direct_gap:.16e} eV"
        )
    highest_occupied = float(np.max(energies[occupied_per_k - 1]))
    lowest_empty = float(np.min(energies[occupied_per_k]))
    return FixedRankOccupationResult(
        projector_ket=projector,
        energies=energies,
        chemical_potential=0.5 * (highest_occupied + lowest_empty),
        occupied_per_k=occupied_per_k,
        minimum_direct_gap=minimum_direct_gap,
        indirect_gap=lowest_empty - highest_occupied,
    )


def global_fermi_occupations(
    hamiltonian_ket: np.ndarray,
    *,
    target_particle_number: float,
    kbt_ev: float,
    k_weights: np.ndarray | None = None,
    particle_tolerance: float = 1.0e-12,
    max_bisection_iterations: int = 512,
    eigensolver_workers: int = 1,
) -> GlobalFermiOccupationResult:
    """Build a fractional Fermi density using one chemical potential.

    This routine only solves the occupation problem. Omitted ``k_weights``
    means unit weights, not weights normalized to sum to one. The absolute
    particle residual must not exceed ``particle_tolerance * max(1, capacity)``.
    The caller remains responsible for using the same k-point normalization
    in its energy, entropy, and interaction functionals. Empty/full targets
    return the extended-real chemical potentials ``-inf``/``+inf``.
    """

    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[1] != hamiltonian.shape[2]
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("hamiltonian_ket must be finite with shape (nk,nb,nb)")
    scale = max(1.0, float(np.max(np.abs(hamiltonian), initial=0.0)))
    hermitian_residual = float(
        np.max(
            np.abs(hamiltonian - hamiltonian.conj().transpose(0, 2, 1)),
            initial=0.0,
        )
    )
    if hermitian_residual > 64.0 * np.finfo(float).eps * scale:
        raise ValueError("hamiltonian_ket must be Hermitian before occupation")

    temperature = float(kbt_ev)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("kbt_ev must be finite and strictly positive")
    number_tolerance = float(particle_tolerance)
    if not math.isfinite(number_tolerance) or number_tolerance <= 0.0:
        raise ValueError("particle_tolerance must be finite and strictly positive")
    if type(max_bisection_iterations) is not int or max_bisection_iterations <= 0:
        raise ValueError("max_bisection_iterations must be a positive integer")

    nk, nb, _ = hamiltonian.shape
    if nk == 0 or nb == 0:
        raise ValueError("hamiltonian_ket must contain at least one k point and band")
    if k_weights is None:
        weights = np.ones(nk, dtype=float)
    else:
        weights = np.asarray(k_weights, dtype=float)
        if weights.shape != (nk,):
            raise ValueError(f"k_weights must have shape {(nk,)}, got {weights.shape}")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("k_weights must be finite and nonnegative")
        weights = weights.copy()
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("k_weights must contain positive total weight")
    capacity = float(nb * total_weight)
    target = float(target_particle_number)
    if not math.isfinite(target) or target < 0.0 or target > capacity:
        raise ValueError(
            "target_particle_number is outside the weighted one-particle space"
        )

    number_tolerance_absolute = number_tolerance * max(1.0, capacity)
    energies, vectors = _diagonalize_hermitian_blocks(
        hamiltonian, eigensolver_workers=eigensolver_workers
    )

    if target == 0.0:
        chemical_potential = float("-inf")
        occupations = np.zeros_like(energies)
    elif target == capacity:
        chemical_potential = float("inf")
        occupations = np.ones_like(energies)
    else:
        state_energies = energies.T.reshape(-1)
        state_weights = np.repeat(weights, nb)
        positive = state_weights > 0.0
        order = np.argsort(state_energies[positive], kind="stable")
        sorted_energies = state_energies[positive][order]
        sorted_weights = state_weights[positive][order]
        crossing = int(np.searchsorted(np.cumsum(sorted_weights), target, side="left"))
        reference_energy = float(sorted_energies[min(crossing, sorted_energies.size - 1)])
        with np.errstate(over="ignore", invalid="raise", divide="ignore"):
            scaled_energies = (energies - reference_energy) / temperature

        def occupations_at_scaled_mu(scaled_mu: float) -> np.ndarray:
            return np.asarray(expit(scaled_mu - scaled_energies), dtype=float)

        def particle_number_at_scaled_mu(scaled_mu: float) -> float:
            return float(
                np.einsum(
                    "k,bk->", weights, occupations_at_scaled_mu(scaled_mu), optimize=True
                )
            )

        bound = 64.0
        for _ in range(16):
            lower, upper = -bound, bound
            if (
                particle_number_at_scaled_mu(lower) <= target
                and particle_number_at_scaled_mu(upper) >= target
            ):
                break
            bound *= 2.0
        else:
            raise RuntimeError("failed to bracket the scaled chemical potential")

        best_residual = float("inf")
        best_scaled_mu = 0.0
        best_occupations = occupations_at_scaled_mu(best_scaled_mu)
        for _ in range(max_bisection_iterations):
            scaled_mu = lower + 0.5 * (upper - lower)
            trial_occupations = occupations_at_scaled_mu(scaled_mu)
            number = float(
                np.einsum("k,bk->", weights, trial_occupations, optimize=True)
            )
            residual = number - target
            if abs(residual) < abs(best_residual):
                best_residual = residual
                best_scaled_mu = scaled_mu
                best_occupations = trial_occupations
            if abs(residual) <= number_tolerance_absolute:
                break
            if residual < 0.0:
                next_lower, next_upper = scaled_mu, upper
            else:
                next_lower, next_upper = lower, scaled_mu
            if next_lower == lower and next_upper == upper:
                break
            lower, upper = next_lower, next_upper
        occupations = best_occupations
        chemical_potential = float(reference_energy + temperature * best_scaled_mu)

    particle_number = float(
        np.einsum("k,bk->", weights, occupations, optimize=True)
    )
    particle_residual = float(particle_number - target)
    if abs(particle_residual) > number_tolerance_absolute:
        raise RuntimeError(
            "finite-temperature occupation did not meet the particle-number "
            f"tolerance: residual={particle_residual!r}"
        )

    density = np.empty_like(hamiltonian)

    def assemble_density(k: int) -> tuple[int, np.ndarray]:
        columns = vectors[k]
        block = (columns * occupations[:, k][None, :]) @ columns.conj().T
        return k, block

    if eigensolver_workers == 1:
        density_results = map(assemble_density, range(nk))
        for k, block in density_results:
            density[k] = block
    else:
        with ThreadPoolExecutor(
            max_workers=min(eigensolver_workers, nk)
        ) as executor:
            for k, block in executor.map(assemble_density, range(nk)):
                density[k] = block
    entropy_terms = xlogy(occupations, occupations) + xlogy(
        1.0 - occupations, 1.0 - occupations
    )
    entropy = -float(np.einsum("k,bk->", weights, entropy_terms, optimize=True))
    return GlobalFermiOccupationResult(
        density_ket=density,
        energies=energies,
        occupations=occupations,
        chemical_potential=float(chemical_potential),
        particle_number=particle_number,
        target_particle_number=target,
        particle_residual=particle_residual,
        entropy_dimensionless=entropy,
        k_weights=weights,
        kbt_ev=temperature,
    )


def global_canonical_occupations(
    hamiltonian_ket: np.ndarray,
    *,
    total_occupied: int,
    fermi_degeneracy_tolerance: float,
    eigensolver_workers: int = 1,
) -> GlobalOccupationResult:
    """Fill globally lowest states, failing closed on an ambiguous Fermi tie.

    The Hamiltonian and returned projector use ``(nk,nb,nb)`` ket blocks.
    Energies use the established band-major ``(nb,nk)`` HF layout.
    """

    hamiltonian = np.asarray(hamiltonian_ket, dtype=np.complex128)
    if (
        hamiltonian.ndim != 3
        or hamiltonian.shape[1] != hamiltonian.shape[2]
        or not np.all(np.isfinite(hamiltonian))
    ):
        raise ValueError("hamiltonian_ket must be finite with shape (nk,nb,nb)")
    scale = max(1.0, float(np.max(np.abs(hamiltonian), initial=0.0)))
    hermitian_residual = float(
        np.max(
            np.abs(hamiltonian - hamiltonian.conj().transpose(0, 2, 1)),
            initial=0.0,
        )
    )
    if hermitian_residual > 64.0 * np.finfo(float).eps * scale:
        raise ValueError("hamiltonian_ket must be Hermitian before occupation")
    tolerance = float(fermi_degeneracy_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Fermi degeneracy tolerance must be finite and nonnegative")
    nk, nb, _ = hamiltonian.shape
    if type(total_occupied) is not int:
        raise TypeError("total_occupied must be an exact integer")
    if total_occupied < 0 or total_occupied > nk * nb:
        raise ValueError("total_occupied is outside the global one-particle space")

    energies, vectors = _diagonalize_hermitian_blocks(
        hamiltonian, eigensolver_workers=eigensolver_workers
    )
    records: list[tuple[float, int, int]] = []
    for k in range(nk):
        records.extend((float(energies[band, k]), k, band) for band in range(nb))
    records.sort(key=lambda item: (item[0], item[1], item[2]))

    if total_occupied == 0:
        chemical_potential, gap = float("-inf"), float("inf")
    elif total_occupied == nk * nb:
        chemical_potential, gap = float("inf"), float("inf")
    else:
        lower = records[total_occupied - 1][0]
        upper = records[total_occupied][0]
        gap = float(upper - lower)
        if gap <= tolerance:
            raise ValueError(
                "global Fermi boundary is degenerate within the explicit tolerance; "
                "canonical occupation is not unique"
            )
        chemical_potential = 0.5 * (lower + upper)

    occupied_bands_by_k: list[list[int]] = [[] for _ in range(nk)]
    for _energy, k, band in records[:total_occupied]:
        occupied_bands_by_k[k].append(band)
    projector = np.zeros_like(hamiltonian)

    def assemble_projector(k: int) -> tuple[int, np.ndarray]:
        occupied = occupied_bands_by_k[k]
        if not occupied:
            return k, np.zeros((nb, nb), dtype=np.complex128)
        columns = vectors[k][:, occupied]
        return k, columns @ columns.conj().T

    if eigensolver_workers == 1:
        projector_results = map(assemble_projector, range(nk))
        for k, block in projector_results:
            projector[k] = block
    else:
        with ThreadPoolExecutor(
            max_workers=min(eigensolver_workers, nk)
        ) as executor:
            for k, block in executor.map(assemble_projector, range(nk)):
                projector[k] = block
    return GlobalOccupationResult(
        projector_ket=projector,
        energies=energies,
        chemical_potential=float(chemical_potential),
        occupied_count=total_occupied,
        fermi_gap=float(gap),
    )


def calculate_norm_convergence(updated_density: np.ndarray, previous_density: np.ndarray) -> float:
    numerator = float(np.linalg.norm(previous_density - updated_density))
    denominator = float(np.linalg.norm(updated_density))
    if denominator < 1e-15:
        return 0.0 if numerator < 1e-15 else float("inf")
    return numerator / denominator


def _reference_diagonal_array(reference_diagonal: np.ndarray | float | None, nb: int) -> np.ndarray:
    if reference_diagonal is None:
        return np.zeros((nb,), dtype=float)
    values = np.asarray(reference_diagonal, dtype=float)
    if values.ndim == 0:
        return np.full((nb,), float(values), dtype=float)
    if values.shape != (nb,):
        raise ValueError(f"Expected reference_diagonal shape {(nb,)}, got {values.shape}")
    return values


def flat_sector_indices(n_spin: int, n_eta: int, nb: int, ispin: int, ieta: int) -> np.ndarray:
    layout = np.arange(int(n_spin) * int(n_eta) * int(nb), dtype=int).reshape(
        (int(n_spin), int(n_eta), int(nb)),
        order="F",
    )
    return np.asarray(layout[int(ispin), int(ieta), :], dtype=int)


def flatten_sector_blocks(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=np.complex128)
    if arr.ndim != 5:
        raise ValueError(f"Expected blocks shape (n_spin, n_eta, nb, nb, nk), got {arr.shape}")
    n_spin, n_eta, nb, nb_col, nk = arr.shape
    if nb_col != nb:
        raise ValueError(f"Expected square sector blocks, got {arr.shape}")
    flat = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)
    k_indices = np.arange(nk)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            flat[np.ix_(idx, idx, k_indices)] = arr[ispin, ieta]
    return flat


def unflatten_sector_blocks(flat: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    arr = np.asarray(flat, dtype=np.complex128)
    expected = int(n_spin) * int(n_eta) * int(nb)
    if arr.ndim != 3 or arr.shape[0] != expected or arr.shape[1] != expected:
        raise ValueError(f"Expected flat blocks shape {(expected, expected, 'nk')}, got {arr.shape}")
    nk = arr.shape[2]
    blocks = np.zeros((int(n_spin), int(n_eta), int(nb), int(nb), nk), dtype=np.complex128)
    k_indices = np.arange(nk)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            blocks[ispin, ieta] = arr[np.ix_(idx, idx, k_indices)]
    return blocks


def unflatten_sector_energies(flat_energies: np.ndarray, *, n_spin: int, n_eta: int, nb: int) -> np.ndarray:
    arr = np.asarray(flat_energies, dtype=float)
    expected = int(n_spin) * int(n_eta) * int(nb)
    if arr.ndim != 2 or arr.shape[0] != expected:
        raise ValueError(f"Expected flat energy shape {(expected, 'nk')}, got {arr.shape}")
    nk = arr.shape[1]
    energies = np.zeros((int(n_spin), int(n_eta), int(nb), nk), dtype=float)
    for ispin in range(int(n_spin)):
        for ieta in range(int(n_eta)):
            idx = flat_sector_indices(n_spin, n_eta, nb, ispin, ieta)
            energies[ispin, ieta] = arr[idx, :]
    return energies


def density_from_fixed_sector_occupations(
    hamiltonian_blocks: np.ndarray,
    occupation_counts: np.ndarray,
    reference_diagonal: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = np.asarray(hamiltonian_blocks, dtype=np.complex128)
    if blocks.ndim != 5:
        raise ValueError(f"Expected hamiltonian_blocks shape (n_spin, n_eta, nb, nb, nk), got {blocks.shape}")
    n_spin, n_eta, nb, nb_col, nk = blocks.shape
    if nb_col != nb:
        raise ValueError(f"Expected square sector blocks, got {blocks.shape}")
    counts = np.asarray(occupation_counts, dtype=int)
    if counts.shape != (n_spin, n_eta):
        raise ValueError(f"Expected occupation_counts shape {(n_spin, n_eta)}, got {counts.shape}")
    ref = _reference_diagonal_array(reference_diagonal, nb)
    density = np.zeros_like(blocks, dtype=np.complex128)
    energies = np.zeros((n_spin, n_eta, nb, nk), dtype=float)
    for ispin in range(n_spin):
        for ieta in range(n_eta):
            n_occ = int(counts[ispin, ieta])
            if n_occ < 0 or n_occ > nb:
                raise ValueError(f"occupation count {n_occ} is outside [0, {nb}]")
            for ik in range(nk):
                evals, evecs = eigh(blocks[ispin, ieta, :, :, ik])
                energies[ispin, ieta, :, ik] = evals
                if n_occ:
                    projector = evecs[:, :n_occ] @ evecs[:, :n_occ].conjugate().T
                else:
                    projector = np.zeros((nb, nb), dtype=np.complex128)
                density[ispin, ieta, :, :, ik] = projector - np.diag(ref)
    return density, energies


def conventional_projector_to_stored(projector: np.ndarray) -> np.ndarray:
    return ket_projector_to_stored_orientation(projector)


def stored_projector_to_conventional(stored: np.ndarray) -> np.ndarray:
    return stored_orientation_to_ket_projector(stored)


def random_unitary_from_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    sampled = rng.standard_normal((int(dim), int(dim))) + 1j * rng.standard_normal((int(dim), int(dim)))
    hermitian = sampled + sampled.conjugate().T
    _, vecs = np.linalg.eigh(hermitian)
    return np.asarray(vecs, dtype=np.complex128)


def apply_random_projector_rotation(
    density: np.ndarray,
    *,
    reference_density: np.ndarray,
    alpha: float,
    seed: int,
) -> None:
    arr = np.asarray(density, dtype=np.complex128)
    reference = np.asarray(reference_density, dtype=np.complex128)
    if arr.shape != reference.shape or arr.ndim != 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            "density and reference_density must have matching shape (nt, nt, nk); "
            f"got {arr.shape} and {reference.shape}"
        )
    rng = np.random.default_rng(seed)
    nt = arr.shape[0]
    for ik in range(arr.shape[2]):
        unitary = random_unitary_from_hermitian(nt, rng)
        projector = arr[:, :, ik] + reference[:, :, ik]
        rotated_density = unitary.conjugate().T @ projector @ unitary - reference[:, :, ik]
        arr[:, :, ik] = (1.0 - float(alpha)) * arr[:, :, ik] + float(alpha) * rotated_density
