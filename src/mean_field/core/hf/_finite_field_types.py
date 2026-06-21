from __future__ import annotations

from ._finite_field_shared import *  # noqa: F401,F403

@dataclass(frozen=True)
class MagneticOverlapData:
    """Projected density-overlap matrices for finite-B HF.

    ``overlaps[(m, n)]`` stores ``Λ_G`` with ``G = m*g1 + (n/q)*g2`` and shape
    ``(nt, nk_target, nt, nk_source)``.  For the full magnetic-BZ calculation
    ``nk_target == nk_source == q*nq**2``.  For the tL-symmetric reduced IKS
    calculation, overlaps are still stored on the full strip orbit with
    ``nk_full = q*nq**2`` while the density lives on ``nq**2`` reduced momenta.
    """

    shifts: tuple[tuple[int, int], ...]
    gvecs: Array
    overlaps: Mapping[tuple[int, int], Array]

    def __post_init__(self) -> None:
        shifts = tuple((int(m), int(n)) for m, n in self.shifts)
        gvecs = np.asarray(self.gvecs, dtype=np.complex128)
        if gvecs.shape != (len(shifts),):
            raise ValueError(f"Expected gvecs shape {(len(shifts),)}, got {gvecs.shape}")
        for shift in shifts:
            if shift not in self.overlaps:
                raise ValueError(f"Missing overlap block for shift {shift}")
            block = np.asarray(self.overlaps[shift], dtype=np.complex128)
            if block.ndim != 4 or block.shape[0] != block.shape[2]:
                raise ValueError(f"Overlap {shift} must have shape (nt,nk_t,nt,nk_s), got {block.shape}")
        object.__setattr__(self, "shifts", shifts)
        object.__setattr__(self, "gvecs", gvecs)

    @classmethod
    def from_overlap_mapping(
        cls,
        overlaps: Mapping[tuple[int, int], Array],
        *,
        g1: complex,
        g2: complex,
        q: int,
        shell_ng: int | None = None,
    ) -> "MagneticOverlapData":
        shifts = tuple(sorted((int(m), int(n)) for m, n in overlaps))
        if shell_ng is not None:
            shifts = tuple(shift for shift in shifts if in_hex_shell(shift[0], shift[1], g1=g1, g2=g2, q=q, shell_ng=shell_ng))
        gvecs = np.asarray([magnetic_reciprocal_vector(m, n, g1=g1, g2=g2, q=q) for m, n in shifts], dtype=np.complex128)
        return cls(shifts=shifts, gvecs=gvecs, overlaps={shift: np.asarray(overlaps[shift], dtype=np.complex128) for shift in shifts})

@dataclass(frozen=True)
class FiniteFieldHartreeFockInputs:
    """Fully assembled finite-B HF inputs for one flux point.

    This is a small no-I/O bundle: workflow code may decide how to cache or
    persist these arrays, while the system layer owns the physics conventions
    and array ordering needed by the HF kernel.
    """

    state: "FiniteFieldHartreeFockState"
    overlap_data: MagneticOverlapData
    k_vectors: Array
    normalization_count: int

    def __post_init__(self) -> None:
        kvec = np.asarray(self.k_vectors, dtype=np.complex128)
        if kvec.shape != (self.state.nk,):
            raise ValueError(f"Expected k_vectors shape {(self.state.nk,)}, got {kvec.shape}")
        if int(self.normalization_count) <= 0:
            raise ValueError("normalization_count must be positive")
        object.__setattr__(self, "k_vectors", kvec)
        object.__setattr__(self, "normalization_count", int(self.normalization_count))

@dataclass(frozen=True)
class FiniteFieldTLSymmetricHartreeFockInputs:
    """Assembled reduced tL-symmetric / IKS finite-B HF inputs."""

    state: "FiniteFieldHartreeFockState"
    overlap_data: MagneticOverlapData
    full_k_vectors: Array
    normalization_count: int

    def __post_init__(self) -> None:
        full_k = np.asarray(self.full_k_vectors, dtype=np.complex128)
        expected = self.state.q * self.state.nk
        if not self.state.reduced_translation:
            raise ValueError("tL-symmetric inputs require a reduced_translation=True state")
        if full_k.shape != (expected,):
            raise ValueError(f"Expected full_k_vectors shape {(expected,)}, got {full_k.shape}")
        if int(self.normalization_count) <= 0:
            raise ValueError("normalization_count must be positive")
        object.__setattr__(self, "full_k_vectors", full_k)
        object.__setattr__(self, "normalization_count", int(self.normalization_count))

FiniteFieldHartreeFockInputBundle = FiniteFieldHartreeFockInputs | FiniteFieldTLSymmetricHartreeFockInputs
"""Either full magnetic-BZ or reduced tL-symmetric finite-B HF input bundle."""


@dataclass(frozen=True)
class FiniteFieldHartreeFockSummary:
    """Small no-I/O summary of a finite-B HF state/run.

    ``single_particle_gap`` is the occupied/unoccupied HF eigenvalue gap in the
    stored finite system; it is not a many-body charge gap.
    """

    filling: float
    energy_per_muc: float
    mu: float
    single_particle_gap: float
    final_raw_norm: float
    iterations: int
    converged: bool
    exit_reason: str

@dataclass
class FiniteFieldHartreeFockState:
    """Finite-B HF state in the stored-projector convention of the Julia code."""

    h0: Array
    sigma_z: Array
    density: Array
    hamiltonian: Array
    energies: Array
    sigma_ztauz: Array
    nu: float
    flux: MagneticFlux
    nq: int
    v0: float
    mu: float = float("nan")
    precision: float = 1e-5
    n_eta: int = 2
    n_spin: int = 2
    n_band: int = 2
    reduced_translation: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.h0 = np.asarray(self.h0, dtype=np.complex128)
        self.sigma_z = np.asarray(self.sigma_z, dtype=np.complex128)
        self.density = np.asarray(self.density, dtype=np.complex128)
        self.hamiltonian = np.asarray(self.hamiltonian, dtype=np.complex128)
        self.energies = np.asarray(self.energies, dtype=float)
        self.sigma_ztauz = np.asarray(self.sigma_ztauz, dtype=float)
        if self.h0.ndim != 3 or self.h0.shape[0] != self.h0.shape[1]:
            raise ValueError(f"Expected h0 shape (nt,nt,nk), got {self.h0.shape}")
        if self.density.shape != self.h0.shape or self.hamiltonian.shape != self.h0.shape:
            raise ValueError("density and hamiltonian must match h0 shape")
        if self.sigma_z.shape != self.h0.shape:
            raise ValueError("sigma_z must match h0 shape")
        if self.energies.shape != (self.nt, self.nk):
            raise ValueError(f"Expected energies shape {(self.nt, self.nk)}, got {self.energies.shape}")
        if self.sigma_ztauz.shape != (self.nt, self.nk):
            raise ValueError(f"Expected sigma_ztauz shape {(self.nt, self.nk)}, got {self.sigma_ztauz.shape}")

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def q(self) -> int:
        return int(self.flux.q)

    @property
    def p(self) -> int:
        return int(self.flux.p)

    @property
    def subbands_per_flavor(self) -> int:
        return int(self.n_band * self.q)

    @classmethod
    def from_h0(
        cls,
        h0: Array,
        *,
        sigma_z: Array | None = None,
        nu: float,
        flux: MagneticFlux,
        nq: int,
        v0: float,
        precision: float = 1e-5,
        reduced_translation: bool = False,
    ) -> "FiniteFieldHartreeFockState":
        h0_arr = np.asarray(h0, dtype=np.complex128)
        nt, _, nk = h0_arr.shape
        sigma = np.zeros_like(h0_arr) if sigma_z is None else np.asarray(sigma_z, dtype=np.complex128)
        return cls(
            h0=h0_arr,
            sigma_z=sigma,
            density=np.zeros_like(h0_arr),
            hamiltonian=h0_arr.copy(),
            energies=np.zeros((nt, nk), dtype=float),
            sigma_ztauz=np.zeros((nt, nk), dtype=float),
            nu=float(nu),
            flux=flux,
            nq=int(nq),
            v0=float(v0),
            precision=float(precision),
            reduced_translation=bool(reduced_translation),
        )

__all__ = [name for name in globals() if not name.startswith('__')]
