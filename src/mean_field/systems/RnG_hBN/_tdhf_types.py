from __future__ import annotations

from ._tdhf_shared import *  # noqa: F401,F403
from ._tdhf_support import _reject_zero_literal_q0_fock_env

@dataclass(frozen=True)
class RLGhBNTDHFMomentumShift:
    """Discrete TDHF momentum transfer on the saved mBZ mesh."""

    shift: tuple[int, int]
    mesh_shape: tuple[int, int]

    @property
    def frac(self) -> tuple[float, float]:
        return (float(self.shift[0]) / float(self.mesh_shape[0]), float(self.shift[1]) / float(self.mesh_shape[1]))


@dataclass(frozen=True)
class RLGhBNTDHFOrbitals:
    """HF orbitals and occupation mask with a stable global-index convention.

    ``eigenvectors[basis_index, hf_index, k]`` is the unitary returned by the
    per-k HF diagonalization.  The global TDHF index is
    ``local_hf_index + nt * k_index``, matching Fortran flattening of
    ``energies[local, k]``.
    """

    energies: np.ndarray
    eigenvectors: np.ndarray
    occupied_mask: np.ndarray
    mu: float
    n_spin: int
    n_eta: int
    n_band: int

    @property
    def nt(self) -> int:
        return int(self.energies.shape[0])

    @property
    def nk(self) -> int:
        return int(self.energies.shape[1])

    @property
    def global_energies(self) -> np.ndarray:
        return np.asarray(self.energies, dtype=float).reshape(-1, order="F")

    def global_index(self, local_index: int, k_index: int) -> int:
        local = int(local_index)
        ik = int(k_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        if ik < 0 or ik >= self.nk:
            raise IndexError(f"k_index={ik} outside [0, {self.nk})")
        return local + self.nt * ik

    def decode_global_index(self, global_index: int) -> tuple[int, int]:
        index = int(global_index)
        if index < 0 or index >= self.nt * self.nk:
            raise IndexError(f"global_index={index} outside [0, {self.nt * self.nk})")
        return index % self.nt, index // self.nt

    def flavor_tag(self, local_index: int) -> SpinValleyFlavor:
        local = int(local_index)
        if local < 0 or local >= self.nt:
            raise IndexError(f"local_index={local} outside [0, {self.nt})")
        ispin = local % self.n_spin
        ieta = (local // self.n_spin) % self.n_eta
        # RLG/hBN uses two valleys ordered as K, K'.  Keep integer valley labels
        # system-local to avoid imposing a plotting/string convention here.
        valley = 1 if ieta == 0 else -1 if self.n_eta == 2 else ieta
        return SpinValleyFlavor(spin=ispin, valley=valley)


@dataclass(frozen=True)
class RLGhBNTDHFInteraction:
    """On-demand RLG/hBN HF-basis two-body matrix element for TDHF.

    The returned value follows the generic core convention ``V[a,b,c,d]`` as the
    coefficient of ``c_b† c_a† c_c c_d``.  It is assembled from layer-resolved
    form factors as

    ``sum_Q,l,l' F_l(a,c; Q) conj(F_l'(d,b; Q)) V_ll'(Q) / (N_k Omega)``.

    This is deliberately callable-based; materializing the full four-index
    tensor is only suitable for very small smoke tests.
    """

    basis_data: RLGhBNProjectedBasisData
    overlap_blocks: RLGhBNLayerOverlapBlockSet
    orbitals: RLGhBNTDHFOrbitals
    beta: float = 1.0
    momentum_policy: MomentumPolicy = "strict"
    momentum_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if self.basis_data.nt != self.orbitals.nt or self.basis_data.nk != self.orbitals.nk:
            raise ValueError(
                "basis_data and orbitals dimensions differ: "
                f"basis nt/nk=({self.basis_data.nt}, {self.basis_data.nk}), "
                f"orbital nt/nk=({self.orbitals.nt}, {self.orbitals.nk})"
            )
        if self.momentum_policy not in {"strict", "mod_integer"}:
            raise ValueError(f"Unsupported momentum_policy={self.momentum_policy!r}")
        _reject_zero_literal_q0_fock_env()

    @property
    def scale(self) -> float:
        return float(self.beta) * float(self.basis_data.v0) / float(self.basis_data.nk)

    def __call__(self, a: int, b: int, c: int, d: int) -> complex:
        return self.matrix_element(a, b, c, d)

    def matrix_element(self, a: int, b: int, c: int, d: int) -> complex:
        a_local, a_k = self.orbitals.decode_global_index(a)
        b_local, b_k = self.orbitals.decode_global_index(b)
        c_local, c_k = self.orbitals.decode_global_index(c)
        d_local, d_k = self.orbitals.decode_global_index(d)
        if not self._momentum_conserved(a_k, b_k, c_k, d_k):
            return 0.0 + 0.0j

        total = 0.0 + 0.0j
        for shift in self.overlap_blocks.shifts:
            layer_overlap = self.overlap_blocks.layer_overlaps[shift]
            fock_kernel = self.overlap_blocks.fock_layer_coulomb[shift]
            if layer_overlap.shape[2] != self.basis_data.nk or layer_overlap.shape[4] != self.basis_data.nk:
                raise ValueError(f"Layer overlap for shift {shift} is incompatible with basis nk={self.basis_data.nk}")
            for target_layer in range(layer_overlap.shape[0]):
                left = self._hf_form_factor(
                    layer_overlap[target_layer],
                    a_local,
                    a_k,
                    c_local,
                    c_k,
                )
                if left == 0.0:
                    continue
                for source_layer in range(layer_overlap.shape[0]):
                    right = self._hf_form_factor(
                        layer_overlap[source_layer],
                        d_local,
                        d_k,
                        b_local,
                        b_k,
                    )
                    if right == 0.0:
                        continue
                    total += (
                        self.scale
                        * complex(fock_kernel[a_k, c_k, target_layer, source_layer])
                        * left
                        * np.conj(right)
                    )
        return complex(total)

    def _hf_form_factor(
        self,
        overlap: np.ndarray,
        target_hf: int,
        target_k: int,
        source_hf: int,
        source_k: int,
    ) -> complex:
        target_vec = self.orbitals.eigenvectors[:, int(target_hf), int(target_k)]
        source_vec = self.orbitals.eigenvectors[:, int(source_hf), int(source_k)]
        block = overlap[:, int(target_k), :, int(source_k)]
        return complex(np.vdot(target_vec, block @ source_vec))

    def _momentum_conserved(self, a_k: int, b_k: int, c_k: int, d_k: int) -> bool:
        frac = np.asarray(self.basis_data.k_grid_frac, dtype=float)
        if frac.shape[0] != self.basis_data.nk or frac.shape[1] != 2:
            raise ValueError(f"Expected k_grid_frac shape (nk, 2), got {frac.shape}")
        # The two form factors use transfers k_c-k_a and k_b-k_d with the same G.
        residual = (frac[int(c_k)] - frac[int(a_k)]) - (frac[int(b_k)] - frac[int(d_k)])
        if self.momentum_policy == "mod_integer":
            residual = residual - np.rint(residual)
        return bool(np.max(np.abs(residual)) <= float(self.momentum_tolerance))

__all__ = [name for name in globals() if not name.startswith('__')]
