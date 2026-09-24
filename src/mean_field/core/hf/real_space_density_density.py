"""Periodic density--density Hartree--Fock for translational ``Q=0`` states.

The reusable input is an orbital-resolved lattice interaction ``U_ab(q)``.
Physical systems remain responsible for constructing that interaction from a
screening model and orbital density vertices.  Densities use the repository
stored convention ``D[a,b,k] = <c_a^dagger c_b> - D_ref[a,b,k]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import fft


@dataclass(frozen=True)
class DensityDensityInteractionResult:
    """Hartree/Fock action and matching quadratic interaction energies."""

    hartree_h: np.ndarray
    fock_h: np.ndarray
    interaction_h: np.ndarray
    hartree_energy: float
    fock_energy: float

    @property
    def interaction_energy(self) -> float:
        return float(self.hartree_energy + self.fock_energy)


@dataclass(frozen=True)
class PeriodicDensityDensityKernel:
    """Precomputed periodic orbital kernel for a block-repeated spin basis.

    ``fock_kernel_q[a,b,i,j]`` is the lattice Fourier interaction in eV for
    spatial orbitals ``a,b`` and periodic transfer index ``(i,j)``.  The full
    one-particle basis is ordered as contiguous equal spatial blocks, e.g.
    ``[all orbitals spin-up, all orbitals spin-down]``.  The interaction is
    spin independent and is broadcast over every pair of spin blocks.

    ``hartree_kernel[a,b]`` is the eV interaction at exact zero transfer after
    the physical system has applied its declared macroscopic-background rule.
    It is intentionally separate from the Fock zero-cell average.
    """

    mesh_shape: tuple[int, int]
    fock_kernel_q: np.ndarray
    hartree_kernel: np.ndarray
    spin_blocks: int = 1
    require_neutral_density: bool = False
    neutrality_tolerance: float = 1.0e-10
    validation_tolerance: float = 2.0e-11
    _fock_kernel_fft: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n1, n2 = (int(self.mesh_shape[0]), int(self.mesh_shape[1]))
        if n1 <= 0 or n2 <= 0:
            raise ValueError("mesh_shape entries must be positive")
        if type(self.spin_blocks) is not int or self.spin_blocks <= 0:
            raise TypeError("spin_blocks must be an exact positive integer")
        if type(self.require_neutral_density) is not bool:
            raise TypeError("require_neutral_density must be an exact Boolean")
        neutrality_tolerance = float(self.neutrality_tolerance)
        if not np.isfinite(neutrality_tolerance) or neutrality_tolerance < 0.0:
            raise ValueError("neutrality_tolerance must be finite and nonnegative")
        tolerance = float(self.validation_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("validation_tolerance must be finite and positive")

        kernel = np.asarray(self.fock_kernel_q, dtype=np.complex128)
        hartree = np.asarray(self.hartree_kernel, dtype=np.complex128)
        if kernel.ndim != 4 or kernel.shape[2:] != (n1, n2):
            raise ValueError(
                "fock_kernel_q must have shape (n_spatial,n_spatial,n1,n2)"
            )
        n_spatial = int(kernel.shape[0])
        if kernel.shape[1] != n_spatial or hartree.shape != (n_spatial, n_spatial):
            raise ValueError("spatial interaction matrices must be square and consistent")
        if n_spatial <= 0 or not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(hartree)):
            raise ValueError("interaction kernels must be finite and nonempty")

        scale = max(1.0, float(np.max(np.abs(kernel))), float(np.max(np.abs(hartree))))
        matrix_residual = float(
            np.max(np.abs(kernel - kernel.conj().swapaxes(0, 1)), initial=0.0)
        )
        inverse_i = (-np.arange(n1)) % n1
        inverse_j = (-np.arange(n2)) % n2
        inverse_kernel = kernel[:, :, inverse_i][:, :, :, inverse_j]
        inversion_residual = float(
            np.max(np.abs(inverse_kernel - kernel.conj()), initial=0.0)
        )
        hartree_imaginary = float(np.max(np.abs(hartree.imag), initial=0.0))
        hartree_symmetry = float(
            np.max(np.abs(hartree.real - hartree.real.T), initial=0.0)
        )
        limit = tolerance * scale
        if matrix_residual > limit:
            raise ValueError(
                f"fock_kernel_q is not orbital-Hermitian: residual={matrix_residual:.6e}"
            )
        if inversion_residual > limit:
            raise ValueError(
                f"fock_kernel_q violates U(-q)=U(q)*: residual={inversion_residual:.6e}"
            )
        if hartree_imaginary > limit or hartree_symmetry > limit:
            raise ValueError(
                "hartree_kernel must be real symmetric within validation_tolerance"
            )

        normalized_kernel = np.ascontiguousarray(kernel)
        normalized_hartree = np.ascontiguousarray(
            0.5 * (hartree.real + hartree.real.T), dtype=np.float64
        )
        normalized_kernel.setflags(write=False)
        normalized_hartree.setflags(write=False)
        kernel_fft = np.ascontiguousarray(
            fft.fftn(normalized_kernel, axes=(-2, -1), workers=1)
        )
        kernel_fft.setflags(write=False)
        object.__setattr__(self, "mesh_shape", (n1, n2))
        object.__setattr__(self, "fock_kernel_q", normalized_kernel)
        object.__setattr__(self, "hartree_kernel", normalized_hartree)
        object.__setattr__(self, "_fock_kernel_fft", kernel_fft)

    @property
    def n_spatial(self) -> int:
        return int(self.fock_kernel_q.shape[0])

    @property
    def n_basis(self) -> int:
        return int(self.spin_blocks * self.n_spatial)

    @property
    def nk(self) -> int:
        return int(self.mesh_shape[0] * self.mesh_shape[1])

    @property
    def fock_kernel_fft(self) -> np.ndarray:
        return self._fock_kernel_fft

    def validate_density(self, density_stored: np.ndarray) -> np.ndarray:
        density = np.asarray(density_stored, dtype=np.complex128)
        expected = (self.n_basis, self.n_basis, self.nk)
        if density.shape != expected:
            raise ValueError(f"density_stored must have shape {expected}, got {density.shape}")
        if not np.all(np.isfinite(density)):
            raise ValueError("density_stored must be finite")
        scale = max(1.0, float(np.max(np.abs(density), initial=0.0)))
        residual = float(
            np.max(
                np.abs(density - density.conj().swapaxes(0, 1)), initial=0.0
            )
        )
        if residual > self.validation_tolerance * scale:
            raise ValueError(
                f"density_stored must be Hermitian at every k: residual={residual:.6e}"
            )
        return density

    def average_basis_charges(self, density_stored: np.ndarray) -> np.ndarray:
        """Return average reference-relative charges and enforce background policy."""

        density = self.validate_density(density_stored)
        diagonal_density = np.einsum("aak->a", density, optimize=True) / self.nk
        scale = max(1.0, float(np.max(np.abs(diagonal_density), initial=0.0)))
        if float(np.max(np.abs(diagonal_density.imag), initial=0.0)) > (
            self.validation_tolerance * scale
        ):
            raise ValueError("density diagonal average is not real within tolerance")
        charges = np.ascontiguousarray(diagonal_density.real)
        total_charge = float(np.sum(charges))
        if self.require_neutral_density and abs(total_charge) > float(
            self.neutrality_tolerance
        ):
            raise ValueError(
                "macroscopic Hartree G=0 removal requires a neutral reference-relative "
                f"density; total charge={total_charge:.16e}"
            )
        return charges

    def apply(
        self,
        density_stored: np.ndarray,
        *,
        fft_workers: int = 1,
    ) -> DensityDensityInteractionResult:
        """Apply Hartree plus exchange using cached transfer-kernel FFTs."""

        density = self.validate_density(density_stored)
        if type(fft_workers) is not int or fft_workers <= 0:
            raise TypeError("fft_workers must be an exact positive integer")
        n1, n2 = self.mesh_shape
        ns = self.n_spatial
        nspin = self.spin_blocks

        # Orbital-major planes keep each small reciprocal mesh contiguous.
        # ``overwrite_x=True`` below is permitted only on a detached work
        # buffer.  ``ascontiguousarray`` alone may return a view of the caller's
        # already-contiguous density and would silently corrupt SCF state.
        # Exchange uses the conventional ket density
        # ``rho[a,b]=<c_b^dagger c_a>``.  The repository stores
        # ``P[a,b]=<c_a^dagger c_b>``, hence the leading-axis transpose.
        density_ket = density.swapaxes(0, 1)
        density_planes = np.array(
            density_ket.reshape(nspin, ns, nspin, ns, n1, n2),
            dtype=np.complex128,
            order="C",
            copy=True,
        )
        transformed = fft.fftn(
            density_planes,
            axes=(-2, -1),
            workers=fft_workers,
            overwrite_x=True,
        )
        transformed *= self.fock_kernel_fft[None, :, None, :, :, :]
        fock_planes = fft.ifftn(
            transformed,
            axes=(-2, -1),
            workers=fft_workers,
            overwrite_x=True,
        )
        fock_h = np.ascontiguousarray(-fock_planes.reshape(density.shape) / self.nk)

        basis_charge = self.average_basis_charges(density)
        spatial_charge = basis_charge.reshape(nspin, ns).sum(axis=0)
        spatial_potential = self.hartree_kernel @ spatial_charge
        hartree_h = np.zeros_like(density)
        diagonal_indices = np.arange(self.n_basis)
        hartree_h[diagonal_indices, diagonal_indices, :] = np.tile(
            spatial_potential, nspin
        )[:, None]

        interaction_h = hartree_h + fock_h
        interaction_scale = max(
            1.0, float(np.max(np.abs(interaction_h), initial=0.0))
        )
        interaction_residual = float(
            np.max(
                np.abs(interaction_h - interaction_h.conj().swapaxes(0, 1)),
                initial=0.0,
            )
        )
        if interaction_residual > self.validation_tolerance * interaction_scale:
            raise RuntimeError(
                "density-density HF action is not Hermitian: "
                f"residual={interaction_residual:.6e}"
            )
        hartree_contraction = np.einsum(
            "abk,abk->", density, hartree_h, optimize=True
        ) / self.nk
        fock_contraction = np.einsum(
            "abk,abk->", density, fock_h, optimize=True
        ) / self.nk
        contraction_scale = max(
            1.0, abs(complex(hartree_contraction)), abs(complex(fock_contraction))
        )
        if max(abs(hartree_contraction.imag), abs(fock_contraction.imag)) > (
            self.validation_tolerance * contraction_scale
        ):
            raise RuntimeError("interaction energy contraction is not real")
        hartree_energy = 0.5 * float(hartree_contraction.real)
        fock_energy = 0.5 * float(fock_contraction.real)
        return DensityDensityInteractionResult(
            hartree_h=hartree_h,
            fock_h=fock_h,
            interaction_h=interaction_h,
            hartree_energy=hartree_energy,
            fock_energy=fock_energy,
        )

    def real_space_fock_kernel(self, *, fft_workers: int = 1) -> np.ndarray:
        """Return the periodic inverse transform ``V_ab(R)`` in eV."""

        if type(fft_workers) is not int or fft_workers <= 0:
            raise TypeError("fft_workers must be an exact positive integer")
        return np.asarray(
            fft.ifftn(
                self.fock_kernel_q,
                axes=(-2, -1),
                workers=fft_workers,
            ),
            dtype=np.complex128,
        )


def reference_relative_density_density_energy(
    h0: np.ndarray,
    density_stored: np.ndarray,
    result: DensityDensityInteractionResult,
) -> float:
    """Return the ODA-compatible reference-relative energy per primitive cell."""

    density = np.asarray(density_stored, dtype=np.complex128)
    bare = np.asarray(h0, dtype=np.complex128)
    if density.shape != bare.shape or result.interaction_h.shape != density.shape:
        raise ValueError("h0, density, and interaction action shapes must match")
    one_body = float(
        np.einsum("abk,abk->", density, bare, optimize=True).real
        / density.shape[2]
    )
    return float(one_body + result.interaction_energy)


def direct_periodic_density_density_elements(
    kernel: PeriodicDensityDensityKernel,
    density_stored: np.ndarray,
    targets: Iterable[tuple[int, int, int, int]],
) -> np.ndarray:
    """Literal selected-element oracle for the periodic Hartree--Fock action."""

    density = kernel.validate_density(density_stored)
    n1, n2 = kernel.mesh_shape
    ns = kernel.n_spatial
    nspin = kernel.spin_blocks
    basis_charge = kernel.average_basis_charges(density)
    spatial_charge = basis_charge.reshape(nspin, ns).sum(axis=0)
    spatial_potential = kernel.hartree_kernel @ spatial_charge
    values: list[complex] = []
    for target_i, target_j, full_a, full_b in targets:
        ti, tj, a, b = int(target_i), int(target_j), int(full_a), int(full_b)
        if not (0 <= ti < n1 and 0 <= tj < n2):
            raise IndexError("target reciprocal-grid index is out of range")
        if not (0 <= a < kernel.n_basis and 0 <= b < kernel.n_basis):
            raise IndexError("target basis index is out of range")
        spatial_a, spatial_b = a % ns, b % ns
        total = 0.0j
        for source_i in range(n1):
            for source_j in range(n2):
                delta_i = (ti - source_i) % n1
                delta_j = (tj - source_j) % n2
                source_k = source_i * n2 + source_j
                total += (
                    kernel.fock_kernel_q[
                        spatial_a, spatial_b, delta_i, delta_j
                    ]
                    * density[b, a, source_k]
                )
        value = -total / kernel.nk
        if a == b:
            value += spatial_potential[spatial_a]
        values.append(value)
    return np.asarray(values, dtype=np.complex128)


__all__ = [
    "DensityDensityInteractionResult",
    "PeriodicDensityDensityKernel",
    "direct_periodic_density_density_elements",
    "reference_relative_density_density_energy",
]
