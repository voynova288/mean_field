from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import math
from typing import Literal

import numpy as np

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
)
from mean_field.core.hf.overlap import (
    HFOverlapBlockSet,
    ProjectedWavefunctionBasis,
    compute_density_overlap_trace_from_diagonal,
    contract_fock_action_from_overlap,
    diagonal_overlap_blocks,
    shift_wavefunction_grid,
    validate_projected_basis_compatibility,
)
from mean_field.core.hf.problem import (
    HartreeFockKernel,
    HartreeFockProblem,
    run_hartree_fock_problem,
)
from mean_field.core.hf.interaction import (
    build_projected_interaction_hamiltonian,
    build_projected_target_hamiltonian,
)
from mean_field.core.hf.occupations import conventional_projector_to_stored
from mean_field.core.hf.coulomb import (
    real_space_cell_area_nm2_from_reciprocal,
    screened_coulomb,
)

from .domains import HTQGDomain, HTQGExplicitDisplacements, canonical_domain_key, domain_displacements
from .hamiltonian import centered_band_indices, diagonalize_hamiltonian
from .lattice import HTQGLattice, build_htqg_lattice, build_moire_k_grid, hex_shell_indices
from .params import DEFAULT_THETA_DEG, HTQGParams

SPIN_LABELS: tuple[str, str] = ("up", "down")
VALLEY_SEQUENCE: tuple[int, int] = (1, -1)
VALLEY_LABELS: dict[int, str] = {1: "K", -1: "Kprime"}


@dataclass(frozen=True)
class HTQGInteractionSettings:
    """Projected long-range Coulomb settings for HTQG two-flat-band HF.

    ``epsilon_r`` is the dielectric constant.  ``d_sc_nm`` gives a double-gate
    screening length in the standard continuum convention
    ``V(q)=2π(e²/4πϵ0)/(ϵ|q|) tanh(|q| d_sc)``.  The current runner uses this
    as a screened-Coulomb pilot; changing the gate geometry is a physical input,
    not a postprocessing option.
    """

    epsilon_r: float = 10.0
    d_sc_nm: float = 25.0
    g_shells: int = 2
    include_hartree: bool = True
    include_fock: bool = True
    hartree_reference: Literal["charge_neutral", "none"] = "charge_neutral"
    fock_density: Literal["absolute", "reference_subtracted"] = "absolute"
    finite_zero_limit: bool = True


@dataclass(frozen=True)
class HTQGProjectedHFConfig:
    theta_deg: float = DEFAULT_THETA_DEG
    n_shells: int = 6
    mesh_size: int = 7
    active_band_count: int = 2
    domain: str = "alpha_beta_gamma"
    filling: int = 0
    params: HTQGParams = field(default_factory=lambda: HTQGParams.realistic(kappa=0.6))
    interaction: HTQGInteractionSettings = field(default_factory=HTQGInteractionSettings)
    precision: float = 1.0e-7
    max_iter: int = 80
    mixing: float = 0.5
    use_oda: bool = False
    active_basis: Literal["auto", "energy", "sublattice_chern"] = "energy"
    frac_shift: tuple[float, float] | None = None
    explicit_displacements: HTQGExplicitDisplacements | None = None


@dataclass(frozen=True)
class HTQGStateLabel:
    index: int
    spin: str
    valley: int
    band_position: int
    band_index: int

    @property
    def valley_label(self) -> str:
        return VALLEY_LABELS.get(int(self.valley), f"valley{self.valley}")

    def to_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "spin": self.spin,
            "valley": int(self.valley),
            "valley_label": self.valley_label,
            "band_position": int(self.band_position),
            "band_index": int(self.band_index),
        }


@dataclass(frozen=True)
class _HTQGEmbedding:
    grid_shape: tuple[int, int]
    local_basis_size: int
    basis_indices: np.ndarray  # (N_G, 8)
    origin: tuple[int, int]


@dataclass(frozen=True)
class HTQGProjectedHFData:
    lattice: HTQGLattice
    domain: HTQGDomain
    d12: complex
    d34: complex
    displacement_provenance: Mapping[str, object]
    config: HTQGProjectedHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    band_indices: tuple[int, ...]
    labels: tuple[HTQGStateLabel, ...]
    h0: np.ndarray
    wavefunctions: np.ndarray  # ProjectedWavefunctionBasis layout: (basis, band, valley, k)
    reference_density: np.ndarray
    n_occupied_per_k: int
    moire_area_nm2: float
    shifts: tuple[tuple[int, int], ...]
    shift_gvecs: np.ndarray
    embedding: _HTQGEmbedding

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def n_band(self) -> int:
        return int(len(self.band_indices))

    @property
    def v0(self) -> float:
        return 1.0 / float(self.moire_area_nm2)


@dataclass
class HTQGHartreeFockState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    mu: float = float("nan")
    precision: float = 1.0e-7
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class HTQGProjectedHFTargetData:
    kvec: np.ndarray
    h0: np.ndarray
    wavefunctions: np.ndarray
    band_indices: tuple[int, ...] | None = None

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])






@dataclass(frozen=True)
class HTQGProjectedHFTargetInteractionContext:
    """Branch-independent target form factors bound to one source/target pair."""

    identity: tuple[object, ...]
    identity_sha256: str
    target_overlap_blocks: HFOverlapBlockSet
    target_source_overlap_blocks: HFOverlapBlockSet


@dataclass(frozen=True)
class HTQGProjectedHFTargetHamiltonianAssembly:
    """Raw and Hermitian target-HF assembly with auditable components."""

    h0: np.ndarray
    hartree: np.ndarray
    fock: np.ndarray
    raw: np.ndarray
    hermitian: np.ndarray
    hermitian_correction: np.ndarray

@dataclass(frozen=True)
class HTQGProjectedHFTargetHamiltonianAction:
    """Component-wise target-HF action without a dense interaction operator."""

    h0: np.ndarray
    hartree: np.ndarray
    fock: np.ndarray
    raw: np.ndarray


@dataclass(frozen=True)
class HTQGProjectedHFTargetHamiltonianBatchAction:
    """Actions with shape ``(n_batch, target.nt, n_vector, target.nk)``."""

    h0: np.ndarray
    hartree: np.ndarray
    fock: np.ndarray
    raw: np.ndarray

@dataclass(frozen=True)
class HTQGHartreeFockResult:
    data: HTQGProjectedHFData
    run: HartreeFockRun
    init_mode: str
    seed: int
    energy_components: Mapping[str, float]

    @property
    def state(self) -> HTQGHartreeFockState:
        return self.run.state  # type: ignore[return-value]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "domain": self.data.domain.to_dict(),
            "resolved_displacements": {
                "d12_nm": [float(self.data.d12.real), float(self.data.d12.imag)],
                "d34_nm": [float(self.data.d34.real), float(self.data.d34.imag)],
                "provenance": dict(self.data.displacement_provenance),
            },
            "filling": int(self.data.config.filling),
            "init_mode": self.init_mode,
            "seed": int(self.seed),
            "converged": bool(self.run.converged),
            "exit_reason": self.run.exit_reason,
            "iterations": int(self.run.iterations),
            "final_error": None if self.run.iter_err.size == 0 else float(self.run.iter_err[-1]),
            "mu_ev": float(self.state.mu),
            "energy_components_ev_per_cell": dict(self.energy_components),
            "occupation_by_label": occupation_by_label(self.data, self.state.density),
            "grid_gap_ev": gap_estimate(self.state.energies, self.data.n_occupied_per_k),
        }


def validate_htqg_projected_hf_config(config: HTQGProjectedHFConfig) -> None:
    if int(config.n_shells) < 0:
        raise ValueError("n_shells must be non-negative")
    if int(config.mesh_size) <= 0:
        raise ValueError("mesh_size must be positive")
    active_band_count = int(config.active_band_count)
    if active_band_count <= 0:
        raise ValueError("active_band_count must be positive")
    if active_band_count % 2 != 0:
        raise ValueError("active_band_count must be even so charge neutrality is unambiguous")
    if int(config.filling) != config.filling:
        raise ValueError(f"filling must be integer, got {config.filling}")
    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * (active_band_count // 2)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * active_band_count
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(
            f"HTQG filling gives invalid occupied states per k: n_occ={n_occupied_per_k}, nt={nt}, "
            f"active_band_count={active_band_count}, filling={config.filling}"
        )
    if config.interaction.epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive")
    if config.interaction.d_sc_nm < 0.0:
        raise ValueError("d_sc_nm must be non-negative")
    if int(config.interaction.g_shells) < 0:
        raise ValueError("g_shells must be non-negative")
    if config.interaction.hartree_reference not in {"charge_neutral", "none"}:
        raise ValueError(f"Unsupported hartree_reference={config.interaction.hartree_reference!r}")
    if config.interaction.fock_density not in {"absolute", "reference_subtracted"}:
        raise ValueError(f"Unsupported fock_density={config.interaction.fock_density!r}")
    if not (0.0 < float(config.mixing) <= 1.0):
        raise ValueError("mixing must lie in (0, 1]")
    if config.active_basis not in {"auto", "energy", "sublattice_chern"}:
        raise ValueError(f"Unsupported active_basis={config.active_basis!r}")
    explicit = config.explicit_displacements
    if explicit is not None:
        if not isinstance(explicit, HTQGExplicitDisplacements):
            raise TypeError("explicit_displacements must be HTQGExplicitDisplacements or None")
        for name, xy in (("d12_nm_xy", explicit.d12_nm_xy), ("d34_nm_xy", explicit.d34_nm_xy)):
            if len(xy) != 2 or not all(math.isfinite(float(value)) for value in xy):
                raise ValueError(f"{name} must contain exactly two finite Cartesian-nm values")
        if not str(explicit.label).strip():
            raise ValueError("explicit displacement label must be nonempty")


def _sha256_json(name: str, value: object) -> str:
    payload = json.dumps(
        {"name": str(name), "value": value},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(name: str, value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot receipt object array {name}")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_fields(name: str, fields: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    for key in sorted(fields):
        digest.update(b"\0")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"=")
        digest.update(str(fields[key]).encode("ascii"))
    return digest.hexdigest()










def htqg_moire_area_nm2(lattice: HTQGLattice) -> float:
    return real_space_cell_area_nm2_from_reciprocal(lattice.b_m1, lattice.b_m2)


def _embedding(lattice: HTQGLattice) -> _HTQGEmbedding:
    indices = np.asarray(lattice.g_indices, dtype=int)
    min1 = int(np.min(indices[:, 0]))
    min2 = int(np.min(indices[:, 1]))
    max1 = int(np.max(indices[:, 0]))
    max2 = int(np.max(indices[:, 1]))
    nx = max1 - min1 + 1
    ny = max2 - min2 + 1
    local = 8
    basis_indices = np.zeros((lattice.n_g, local), dtype=int)
    for ig, (n1, n2) in enumerate(indices):
        ix = int(n1) - min1
        iy = int(n2) - min2
        for alpha in range(local):
            basis_indices[ig, alpha] = alpha + local * (ix + nx * iy)
    return _HTQGEmbedding(grid_shape=(nx, ny), local_basis_size=local, basis_indices=basis_indices, origin=(min1, min2))


def _shift_table(lattice: HTQGLattice, g_shells: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    """Return a C3/C6-closed reciprocal-transfer table for projected Coulomb terms.

    The moiré reciprocal coordinates transform under C3 as
    ``(m, n) -> (-m-n, m)``.  A square cutoff ``|m|,|n|<=g_shells`` is not
    closed under this map and therefore injects an artificial nematic field into
    otherwise C3-symmetric HTQG HF calculations.  Use the same triangular-shell
    convention as the continuum plane-wave basis: ``max(|m|, |n|, |m+n|) <= g``.
    """

    shifts_arr = hex_shell_indices(int(g_shells))
    shifts: list[tuple[int, int]] = []
    gvecs: list[complex] = []
    for m, n in np.asarray(shifts_arr, dtype=int):
        shifts.append((int(m), int(n)))
        gvecs.append(complex(int(m) * lattice.b_m1 + int(n) * lattice.b_m2))
    return tuple(shifts), np.asarray(gvecs, dtype=np.complex128)


def _label_index(spin_index: int, valley_index: int, band_position: int) -> int:
    return int(spin_index + len(SPIN_LABELS) * (valley_index + len(VALLEY_SEQUENCE) * band_position))


def _sublattice_sigma_z(lattice: HTQGLattice) -> np.ndarray:
    pattern = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=float)
    return np.diag(np.tile(pattern, lattice.n_g)).astype(np.complex128)


def _resolve_active_basis(config: HTQGProjectedHFConfig, domain_key: str) -> str:
    # Keep auto conservative until the Type-I sublattice/Chern active-basis HF
    # convention has a separate validation gate.  The explicit
    # ``sublattice_chern`` option is available for diagnostics only.
    if config.active_basis == "auto":
        return "energy"
    return str(config.active_basis)


def _build_labels(band_indices: tuple[int, ...]) -> tuple[HTQGStateLabel, ...]:
    labels: list[HTQGStateLabel] = []
    for iband, band_index in enumerate(band_indices):
        for ivalley, valley in enumerate(VALLEY_SEQUENCE):
            for ispin, spin in enumerate(SPIN_LABELS):
                labels.append(
                    HTQGStateLabel(
                        index=_label_index(ispin, ivalley, iband),
                        spin=spin,
                        valley=int(valley),
                        band_position=int(iband),
                        band_index=int(band_index),
                    )
                )
    return tuple(sorted(labels, key=lambda item: item.index))


def _projected_basis(data: HTQGProjectedHFData, wavefunctions: np.ndarray | None = None, *, name: str) -> ProjectedWavefunctionBasis:
    wf = data.wavefunctions if wavefunctions is None else np.asarray(wavefunctions, dtype=np.complex128)
    return ProjectedWavefunctionBasis(
        wf,
        data.embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=data.embedding.local_basis_size,
        name=name,
        boundary_mode="zero_fill",
    )


def _valley_transfer_sign(valley: int) -> int:
    """Return reciprocal-transfer sign for the stored valley embedding.

    The K' wavefunctions are represented by time reversal, ``H_K' (k) =
    H_K(-k)^*``.  In this embedding a density transfer labelled by moire
    reciprocal index ``(m, n)`` shifts the K' plane-wave coefficients in the
    opposite direction from K.  Treating both valleys with the same shift is a
    C3-breaking convention error in the projected interaction.
    """

    return 1 if int(valley) == 1 else -1


def _calculate_htqg_projected_overlap_between(
    target: ProjectedWavefunctionBasis,
    source: ProjectedWavefunctionBasis,
    m: int,
    n: int,
) -> np.ndarray:
    """HTQG projected overlap with the correct valley-dependent transfer sign."""

    validate_projected_basis_compatibility(target, source)
    nx, ny = target.grid_shape
    target_band_k = target.n_band * target.nk
    source_band_k = source.n_band * source.nk
    overlap_blocks = np.zeros(
        (
            target.n_spin,
            target.n_flavor,
            target_band_k,
            target.n_spin,
            target.n_flavor,
            source_band_k,
        ),
        dtype=np.complex128,
        order="F",
    )

    for iflavor, valley in enumerate(VALLEY_SEQUENCE):
        sign = _valley_transfer_sign(int(valley))
        ul = target.wavefunctions[:, :, iflavor, :].reshape(target.basis_dimension, target_band_k, order="F")
        ur = source.wavefunctions[:, :, iflavor, :].reshape(source.basis_dimension, source_band_k, order="F")
        shifted = shift_wavefunction_grid(
            ur.reshape(source.local_basis_size, nx, ny, source_band_k, order="F"),
            -sign * int(m),
            -sign * int(n),
            boundary_mode=target.boundary_mode,
            grid_axes=(1, 2),
        ).reshape(source.basis_dimension, source_band_k, order="F")
        lambda_kp = ul.conj().T @ shifted
        for ispin in range(target.n_spin):
            overlap_blocks[ispin, iflavor, :, ispin, iflavor, :] = lambda_kp

    return overlap_blocks.reshape((target.nt, target.nk, source.nt, source.nk), order="F")


def build_htqg_projected_hf_data(config: HTQGProjectedHFConfig) -> HTQGProjectedHFData:
    validate_htqg_projected_hf_config(config)
    lattice = build_htqg_lattice(
        float(config.theta_deg),
        n_shells=int(config.n_shells),
        graphene_lattice_constant_nm=config.params.graphene_lattice_constant_nm,
    )
    domain = domain_displacements(lattice, config.domain)
    if config.explicit_displacements is None:
        d12 = complex(domain.d12)
        d34 = complex(domain.d34)
        displacement_provenance: dict[str, object] = {
            "kind": "named_domain",
            "domain": str(domain.key),
        }
    else:
        d12 = complex(config.explicit_displacements.d12)
        d34 = complex(config.explicit_displacements.d34)
        displacement_provenance = {
            "kind": "explicit",
            **config.explicit_displacements.to_dict(),
            "reference_domain": str(domain.key),
        }
    band_indices = tuple(int(index) for index in centered_band_indices(lattice.matrix_dim, int(config.active_band_count)))
    labels = _build_labels(band_indices)
    mesh = int(config.mesh_size)
    # ``build_moire_k_grid`` interprets frac_shift in units of one mesh step:
    # k_i = (i + frac_shift[0]) / mesh * b1 + ... .  Therefore the standard
    # half-cell shifted mesh is (0.5, 0.5), not (0.5/mesh, 0.5/mesh).
    frac_shift = config.frac_shift if config.frac_shift is not None else (0.5, 0.5)
    k_grid_frac, kvec_grid = build_moire_k_grid(lattice, mesh, endpoint=False, frac_shift=frac_shift)
    kvec = np.asarray(kvec_grid, dtype=np.complex128).reshape(-1)
    nk = int(kvec.size)
    nt = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * len(band_indices)
    emb = _embedding(lattice)
    basis_dim = emb.local_basis_size * emb.grid_shape[0] * emb.grid_shape[1]
    core_wavefunctions = np.zeros((basis_dim, len(band_indices), len(VALLEY_SEQUENCE), nk), dtype=np.complex128)
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    active_basis = _resolve_active_basis(config, domain.key)
    sigma_z = _sublattice_sigma_z(lattice) if active_basis == "sublattice_chern" else None
    for ivalley, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(kvec):
            evals, evecs = diagonalize_hamiltonian(
                complex(kval),
                lattice,
                config.params,
                domain=domain,
                valley=int(valley),
                d12=d12,
                d34=d34,
                band_indices=band_indices,
                return_eigenvectors=True,
            )
            if evecs is None:
                raise RuntimeError("Expected HTQG central-band eigenvectors")
            energy_diag = np.diag(np.asarray(evals, dtype=float)).astype(np.complex128)
            if sigma_z is None:
                active_vectors = np.asarray(evecs, dtype=np.complex128)
                local_h0 = energy_diag
            else:
                projected_sigma = np.asarray(evecs, dtype=np.complex128).conjugate().T @ sigma_z @ np.asarray(evecs, dtype=np.complex128)
                _sigma_evals, sigma_basis = np.linalg.eigh(projected_sigma)
                active_vectors = np.asarray(evecs, dtype=np.complex128) @ sigma_basis
                local_h0 = sigma_basis.conjugate().T @ energy_diag @ sigma_basis
            vec_grid = np.asarray(active_vectors, dtype=np.complex128).reshape(lattice.n_g, 8, len(band_indices))
            for iband in range(len(band_indices)):
                for jband in range(len(band_indices)):
                    for ispin in range(len(SPIN_LABELS)):
                        row = _label_index(ispin, ivalley, iband)
                        col = _label_index(ispin, ivalley, jband)
                        h0[row, col, ik] = local_h0[iband, jband]
                for alpha in range(8):
                    core_wavefunctions[emb.basis_indices[:, alpha], iband, ivalley, ik] = vec_grid[:, alpha, iband]
    reference_density = np.zeros((nt, nt, nk), dtype=np.complex128)
    neutral_occupied_per_k = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * (len(band_indices) // 2)
    for ik in range(nk):
        _vals, vecs = np.linalg.eigh(h0[:, :, ik])
        projector = vecs[:, :neutral_occupied_per_k] @ vecs[:, :neutral_occupied_per_k].conjugate().T
        reference_density[:, :, ik] = conventional_projector_to_stored(projector)
    n_occupied_per_k = neutral_occupied_per_k + int(config.filling)
    if n_occupied_per_k < 0 or n_occupied_per_k > nt:
        raise ValueError(f"Invalid occupied states per k: {n_occupied_per_k} for nt={nt}")
    shifts, shift_gvecs = _shift_table(lattice, config.interaction.g_shells)
    return HTQGProjectedHFData(
        lattice=lattice,
        domain=domain,
        d12=d12,
        d34=d34,
        displacement_provenance=displacement_provenance,
        config=config,
        k_grid_frac=np.asarray(k_grid_frac, dtype=float),
        kvec=kvec,
        band_indices=band_indices,
        labels=labels,
        h0=h0,
        wavefunctions=core_wavefunctions,
        reference_density=reference_density,
        n_occupied_per_k=int(n_occupied_per_k),
        moire_area_nm2=htqg_moire_area_nm2(lattice),
        shifts=shifts,
        shift_gvecs=shift_gvecs,
        embedding=emb,
    )


def _screening_values(data: HTQGProjectedHFData, gvec: complex) -> tuple[float, np.ndarray]:
    settings = data.config.interaction
    hartree = float(
        screened_coulomb(
            abs(complex(gvec)),
            epsilon_r=float(settings.epsilon_r),
            d_sc_nm=float(settings.d_sc_nm),
            finite_zero_limit=bool(settings.finite_zero_limit),
        )
    )
    qvals = data.kvec[None, :] - data.kvec[:, None] + complex(gvec)
    fock = np.asarray(
        screened_coulomb(
            qvals,
            epsilon_r=float(settings.epsilon_r),
            d_sc_nm=float(settings.d_sc_nm),
            finite_zero_limit=bool(settings.finite_zero_limit),
        ),
        dtype=float,
    )
    return hartree, fock


def _build_htqg_overlap_blocks_from_basis(
    data: HTQGProjectedHFData,
    basis: ProjectedWavefunctionBasis,
) -> HFOverlapBlockSet:
    if (basis.nt, basis.nk) != (data.nt, data.nk):
        raise ValueError(
            "Source projected basis does not match the HTQG HF data: "
            f"expected {(data.nt, data.nk)}, got {(basis.nt, basis.nk)}"
        )
    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    fock_screening: dict[tuple[int, int], np.ndarray] = {}
    for shift, gvec in zip(data.shifts, data.shift_gvecs, strict=True):
        block = _calculate_htqg_projected_overlap_between(
            basis,
            basis,
            int(shift[0]),
            int(shift[1]),
        )
        overlaps[shift] = block
        diagonal[shift] = diagonal_overlap_blocks(block, nt=data.nt, nk=data.nk)
        hartree, fock = _screening_values(data, complex(gvec))
        hartree_screening[shift] = hartree
        fock_screening[shift] = fock
    return HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=overlaps,
        diagonal_overlaps=diagonal,
        hartree_screening=hartree_screening,
        fock_screening=fock_screening,
    )

def build_htqg_overlap_blocks(data: HTQGProjectedHFData) -> HFOverlapBlockSet:
    basis = _projected_basis(data, name="htqg-source-grid")
    return _build_htqg_overlap_blocks_from_basis(data, basis)


def _split_overlap_blocks(overlap_blocks: HFOverlapBlockSet) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet]:
    hartree = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening=overlap_blocks.hartree_screening,
        fock_screening={},
    )
    fock = HFOverlapBlockSet(
        shifts=overlap_blocks.shifts,
        gvecs=overlap_blocks.gvecs,
        overlaps=overlap_blocks.overlaps,
        diagonal_overlaps=overlap_blocks.diagonal_overlaps,
        hartree_screening={},
        fock_screening=overlap_blocks.fock_screening,
    )
    return hartree, fock


def _hartree_density(data: HTQGProjectedHFData, density: np.ndarray) -> np.ndarray:
    if data.config.interaction.hartree_reference == "charge_neutral":
        return np.asarray(density, dtype=np.complex128) - data.reference_density
    return np.asarray(density, dtype=np.complex128)


def _fock_density(data: HTQGProjectedHFData, density: np.ndarray) -> np.ndarray:
    if data.config.interaction.fock_density == "reference_subtracted":
        return np.asarray(density, dtype=np.complex128) - data.reference_density
    return np.asarray(density, dtype=np.complex128)


def _build_htqg_interaction_components_from_effective_densities(
    data: HTQGProjectedHFData,
    *,
    hartree_density: np.ndarray,
    fock_density: np.ndarray,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    blocks = build_htqg_overlap_blocks(data) if overlap_blocks is None else overlap_blocks
    hartree_blocks, fock_blocks = _split_overlap_blocks(blocks)
    components: dict[str, np.ndarray] = {}
    if data.config.interaction.include_hartree:
        components["hartree"] = build_projected_interaction_hamiltonian(
            np.asarray(hartree_density, dtype=np.complex128),
            hartree_blocks,
            v0=data.v0,
            beta=1.0,
        )
    else:
        components["hartree"] = np.zeros_like(data.h0)
    if data.config.interaction.include_fock:
        components["fock"] = build_projected_interaction_hamiltonian(
            np.asarray(fock_density, dtype=np.complex128),
            fock_blocks,
            v0=data.v0,
            beta=1.0,
        )
    else:
        components["fock"] = np.zeros_like(data.h0)
    components["total"] = components["hartree"] + components["fock"]
    for value in components.values():
        for ik in range(value.shape[2]):
            value[:, :, ik] = 0.5 * (value[:, :, ik] + value[:, :, ik].conjugate().T)
    return components


def build_htqg_interaction_components(
    data: HTQGProjectedHFData,
    density: np.ndarray,
    *,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    return _build_htqg_interaction_components_from_effective_densities(
        data,
        hartree_density=_hartree_density(data, density),
        fock_density=_fock_density(data, density),
        overlap_blocks=overlap_blocks,
    )


def build_htqg_delta_interaction_components(
    data: HTQGProjectedHFData,
    delta_density: np.ndarray,
    *,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, np.ndarray]:
    """Linear HF response to a density update, with no reference offset.

    The SCF interaction is linear in the effective Hartree/Fock densities, but
    neutral-background and optional Fock reference subtractions are constant
    offsets. ODA therefore needs H[delta_P], not H[delta_P - P_ref].
    """
    delta = np.asarray(delta_density, dtype=np.complex128)
    return _build_htqg_interaction_components_from_effective_densities(
        data,
        hartree_density=delta,
        fock_density=delta,
        overlap_blocks=overlap_blocks,
    )


def _stored_inner(left: np.ndarray, right: np.ndarray, nk: int) -> float:
    return float((np.einsum("abk,abk->", left, right, optimize=True) / float(nk)).real)


def htqg_energy_components(
    data: HTQGProjectedHFData,
    density: np.ndarray,
    *,
    interaction_components: Mapping[str, np.ndarray] | None = None,
    overlap_blocks: HFOverlapBlockSet | None = None,
) -> dict[str, float]:
    comps = build_htqg_interaction_components(data, density, overlap_blocks=overlap_blocks) if interaction_components is None else interaction_components
    one_body = _stored_inner(data.h0, density, data.nk)
    hartree = 0.5 * _stored_inner(comps["hartree"], _hartree_density(data, density), data.nk)
    fock = 0.5 * _stored_inner(comps["fock"], _fock_density(data, density), data.nk)
    return {
        "one_body": one_body,
        "hartree": hartree,
        "fock": fock,
        "total": one_body + hartree + fock,
    }


def _active_flavor_order(mode: str, *, seed: int) -> list[tuple[int, int]]:
    all_flavors = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (spin, valley): up-K, down-K, up-K', down-K'
    mode = mode.strip().lower().replace("-", "_")
    if mode in {"flavor", "sp", "spin_valley"}:
        return all_flavors
    if mode in {"spin_up", "spin"}:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    if mode in {"spin_down"}:
        return [(1, 0), (1, 1), (0, 0), (0, 1)]
    if mode in {"valley_k", "vp_k"}:
        return [(0, 0), (1, 0), (0, 1), (1, 1)]
    if mode in {"valley_kprime", "vp_kprime"}:
        return [(0, 1), (1, 1), (0, 0), (1, 0)]
    if mode in {"balanced", "bm", "noninteracting"}:
        return all_flavors
    if mode.startswith("random"):
        rng = np.random.default_rng(seed)
        order = all_flavors.copy()
        rng.shuffle(order)
        return order
    raise ValueError(
        f"Unsupported HTQG init_mode={mode!r}; use bm, flavor, spin_up, spin_down, valley_k, valley_kprime, balanced, or random."
    )


def initialize_htqg_density(data: HTQGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    mode = init_mode.strip().lower().replace("-", "_")
    nt, nk = data.nt, data.nk
    filling = int(data.config.filling)
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    if mode in {"bm", "noninteracting"}:
        for ik in range(nk):
            evals = np.real(np.diag(data.h0[:, :, ik]))
            projector = np.zeros((nt, nt), dtype=np.complex128)
            for idx in np.argsort(evals, kind="stable")[: data.n_occupied_per_k]:
                projector[int(idx), int(idx)] = 1.0
            density[:, :, ik] = conventional_projector_to_stored(projector)
        return density

    order = _active_flavor_order(mode, seed=seed)
    count = abs(filling)
    for ik in range(nk):
        projector = np.diag(np.real(np.diag(data.reference_density[:, :, ik]))).astype(np.complex128)
        neutral_band_count = data.n_band // 2
        band_position = neutral_band_count if filling >= 0 else neutral_band_count - 1
        sign = 1.0 if filling >= 0 else -1.0
        for spin_index, valley_index in order[:count]:
            idx = _label_index(spin_index, valley_index, band_position)
            projector[idx, idx] += sign
        density[:, :, ik] = conventional_projector_to_stored(projector)
    return density


class HTQGInitializer:
    def __init__(self, data: HTQGProjectedHFData):
        self.data = data

    def __call__(self, state: HTQGHartreeFockState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = initialize_htqg_density(self.data, init_mode=init_mode, seed=seed)
        state.diagnostics.update(_numeric_order_parameters(self.data, state.density))


class HTQGDensityBuilder:
    def __init__(self, data: HTQGProjectedHFData):
        self.data = data

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occ_mask = htqg_density_from_hamiltonian(hamiltonian, self.data.n_occupied_per_k)
        observables = {"occupation_mask": occ_mask}
        observables.update(_numeric_order_parameters(self.data, density))
        return DensityUpdateResult(density=density, energies=energies, mu=mu, observables=observables)


def htqg_density_from_hamiltonian(hamiltonian: np.ndarray, n_occupied_per_k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    nt, nt_rhs, nk = hamiltonian.shape
    if nt != nt_rhs:
        raise ValueError(f"Expected square Hamiltonian blocks, got {hamiltonian.shape}")
    nocc = int(n_occupied_per_k)
    if nocc < 0 or nocc > nt:
        raise ValueError(f"Invalid occupied count per k {nocc} for nt={nt}")
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    energies = np.zeros((nt, nk), dtype=float)
    occ_mask = np.zeros((nt, nk), dtype=bool)
    for ik in range(nk):
        vals, vecs = np.linalg.eigh(hamiltonian[:, :, ik])
        energies[:, ik] = vals
        if nocc:
            occupied = vecs[:, :nocc]
            projector = occupied @ occupied.conjugate().T
            density[:, :, ik] = conventional_projector_to_stored(projector)
            occ_mask[:nocc, ik] = True
    if nocc <= 0 or nocc >= nt:
        mu = float(np.mean(energies))
    else:
        mu = 0.5 * (float(np.max(energies[:nocc, :])) + float(np.min(energies[nocc:, :])))
    return density, energies, float(mu), occ_mask


def _hermitize_blocks(blocks: np.ndarray) -> None:
    for ik in range(blocks.shape[2]):
        blocks[:, :, ik] = 0.5 * (blocks[:, :, ik] + blocks[:, :, ik].conjugate().T)



def build_htqg_hf_problem(
    data: HTQGProjectedHFData,
    overlap_blocks: HFOverlapBlockSet,
    *,
    step_callback=None,
) -> HartreeFockProblem:
    interaction_builder = lambda density: build_htqg_interaction_components(data, density, overlap_blocks=overlap_blocks)["total"]

    def energy_functional(_interaction_h: np.ndarray, _h0: np.ndarray, density: np.ndarray) -> float:
        return htqg_energy_components(data, density, overlap_blocks=overlap_blocks)["total"]

    if bool(data.config.use_oda):
        oda_parameterizer = None
        oda_delta_interaction_builder = lambda delta: build_htqg_delta_interaction_components(
            data,
            delta,
            overlap_blocks=overlap_blocks,
        )["total"]
        convergence_rule = "raw"
    else:
        oda_parameterizer = lambda _state, _delta: float(data.config.mixing)
        oda_delta_interaction_builder = None
        convergence_rule = "raw"

    kernel = HartreeFockKernel(
        interaction_builder=interaction_builder,
        density_builder=HTQGDensityBuilder(data),
        energy_functional=energy_functional,
        oda_parameterizer=oda_parameterizer,
        oda_delta_interaction_builder=oda_delta_interaction_builder,
        hamiltonian_postprocessor=_hermitize_blocks,
        step_callback=step_callback,
        convergence_rule=convergence_rule,
    )
    return HartreeFockProblem(initializer=HTQGInitializer(data), kernel=kernel)


def build_htqg_hf_state(data: HTQGProjectedHFData) -> HTQGHartreeFockState:
    return HTQGHartreeFockState(
        h0=np.asarray(data.h0, dtype=np.complex128).copy(),
        density=np.zeros_like(data.h0, dtype=np.complex128),
        hamiltonian=np.asarray(data.h0, dtype=np.complex128).copy(),
        energies=np.zeros((data.nt, data.nk), dtype=float),
        precision=float(data.config.precision),
    )


def run_htqg_projected_hf(
    data: HTQGProjectedHFData,
    *,
    init_mode: str = "bm",
    seed: int = 1,
    overlap_blocks: HFOverlapBlockSet | None = None,
    step_callback=None,
) -> HTQGHartreeFockResult:
    blocks = build_htqg_overlap_blocks(data) if overlap_blocks is None else overlap_blocks
    state = build_htqg_hf_state(data)
    problem = build_htqg_hf_problem(data, blocks, step_callback=step_callback)
    run = run_hartree_fock_problem(
        state,
        problem,
        init_mode=init_mode,
        seed=int(seed),
        max_iter=int(data.config.max_iter),
        oda_stall_threshold=0.0,
    )
    comps = htqg_energy_components(data, state.density, overlap_blocks=blocks)
    return HTQGHartreeFockResult(data=data, run=run, init_mode=init_mode, seed=int(seed), energy_components=comps)



def build_htqg_target_data(
    data: HTQGProjectedHFData,
    kvec: np.ndarray,
    *,
    band_indices: Sequence[int] | None = None,
) -> HTQGProjectedHFTargetData:
    """Build target-point data in the source or an explicitly requested window.

    A different target window changes only the basis in which the fixed source
    density is evaluated; it does not rerun HF or claim active-space
    self-consistency.  The rectangular target/source contraction is handled by
    the generic projected-HF target builder.
    """

    target_k = np.asarray(kvec, dtype=np.complex128).reshape(-1)
    target_bands = (
        tuple(int(index) for index in data.band_indices)
        if band_indices is None
        else tuple(int(index) for index in band_indices)
    )
    if not target_bands or len(set(target_bands)) != len(target_bands):
        raise ValueError(f"target band_indices must be nonempty and unique, got {target_bands}")
    if tuple(sorted(target_bands)) != target_bands:
        raise ValueError(f"target band_indices must be strictly increasing, got {target_bands}")
    if target_bands[0] < 0 or target_bands[-1] >= int(data.lattice.matrix_dim):
        raise ValueError(
            f"target band_indices {target_bands} escape matrix dimension {data.lattice.matrix_dim}"
        )

    nk = int(target_k.size)
    n_target_band = len(target_bands)
    nt_target = len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * n_target_band
    basis_dim = data.wavefunctions.shape[0]
    target_wavefunctions = np.zeros(
        (basis_dim, n_target_band, len(VALLEY_SEQUENCE), nk),
        dtype=np.complex128,
    )
    target_h0 = np.zeros((nt_target, nt_target, nk), dtype=np.complex128)
    for ivalley, valley in enumerate(VALLEY_SEQUENCE):
        for ik, kval in enumerate(target_k):
            evals, evecs = diagonalize_hamiltonian(
                complex(kval),
                data.lattice,
                data.config.params,
                domain=data.domain,
                valley=int(valley),
                d12=data.d12,
                d34=data.d34,
                band_indices=target_bands,
                return_eigenvectors=True,
            )
            if evecs is None:
                raise RuntimeError("Expected HTQG target eigenvectors")
            vec_grid = np.asarray(evecs, dtype=np.complex128).reshape(
                data.lattice.n_g,
                8,
                n_target_band,
            )
            for iband, energy in enumerate(evals):
                for ispin in range(len(SPIN_LABELS)):
                    idx = _label_index(ispin, ivalley, iband)
                    target_h0[idx, idx, ik] = float(energy)
                for alpha in range(8):
                    target_wavefunctions[
                        data.embedding.basis_indices[:, alpha],
                        iband,
                        ivalley,
                        ik,
                    ] = vec_grid[:, alpha, iband]
    return HTQGProjectedHFTargetData(
        kvec=target_k,
        h0=target_h0,
        wavefunctions=target_wavefunctions,
        band_indices=target_bands,
    )




def _target_overlap_blocks(data: HTQGProjectedHFData, target: HTQGProjectedHFTargetData) -> tuple[HFOverlapBlockSet, HFOverlapBlockSet]:
    source_basis = _projected_basis(data, name="htqg-source-grid")
    target_basis = ProjectedWavefunctionBasis(
        target.wavefunctions,
        data.embedding.grid_shape,
        n_spin=len(SPIN_LABELS),
        local_basis_size=data.embedding.local_basis_size,
        name="htqg-target-path",
        boundary_mode="zero_fill",
    )
    target_overlaps: dict[tuple[int, int], np.ndarray] = {}
    target_diagonal: dict[tuple[int, int], np.ndarray] = {}
    target_source_overlaps: dict[tuple[int, int], np.ndarray] = {}
    target_source_fock: dict[tuple[int, int], np.ndarray] = {}
    hartree_screening: dict[tuple[int, int], float] = {}
    for shift, gvec in zip(data.shifts, data.shift_gvecs, strict=True):
        target_block = _calculate_htqg_projected_overlap_between(target_basis, target_basis, int(shift[0]), int(shift[1]))
        target_overlaps[shift] = target_block
        target_diagonal[shift] = diagonal_overlap_blocks(
            target_block,
            nt=target.nt,
            nk=target.nk,
        )
        ts_block = _calculate_htqg_projected_overlap_between(target_basis, source_basis, int(shift[0]), int(shift[1]))
        target_source_overlaps[shift] = ts_block
        settings = data.config.interaction
        hartree_screening[shift] = float(
            screened_coulomb(
                abs(complex(gvec)),
                epsilon_r=float(settings.epsilon_r),
                d_sc_nm=float(settings.d_sc_nm),
                finite_zero_limit=bool(settings.finite_zero_limit),
            )
        )
        qvals = data.kvec[None, :] - target.kvec[:, None] + complex(gvec)
        target_source_fock[shift] = np.asarray(
            screened_coulomb(
                qvals,
                epsilon_r=float(settings.epsilon_r),
                d_sc_nm=float(settings.d_sc_nm),
                finite_zero_limit=bool(settings.finite_zero_limit),
            ),
            dtype=float,
        )
    target_blocks = HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=target_overlaps,
        diagonal_overlaps=target_diagonal,
        hartree_screening=hartree_screening,
        fock_screening={},
    )
    target_source_blocks = HFOverlapBlockSet(
        shifts=data.shifts,
        gvecs=np.asarray(data.shift_gvecs, dtype=np.complex128),
        overlaps=target_source_overlaps,
        diagonal_overlaps={},
        hartree_screening={},
        fock_screening=target_source_fock,
    )
    return target_blocks, target_source_blocks


def _overlap_block_set_sha256(
    blocks: HFOverlapBlockSet,
    *,
    owner_sha256: str,
    cache_name: str,
) -> str:
    fields: dict[str, object] = {
        "owner": owner_sha256,
        "shifts": _sha256_array(
            f"{cache_name}.shifts",
            np.asarray(blocks.shifts, dtype=np.int64),
        ),
        "gvecs": _sha256_array(f"{cache_name}.gvecs", blocks.gvecs),
    }
    for mapping_name, mapping in (
        ("overlaps", blocks.overlaps),
        ("diagonal_overlaps", blocks.diagonal_overlaps),
        ("hartree_screening", blocks.hartree_screening),
        ("fock_screening", blocks.fock_screening),
    ):
        fields[f"{mapping_name}.keys"] = _sha256_array(
            f"{cache_name}.{mapping_name}.keys",
            np.asarray(tuple(sorted(mapping)), dtype=np.int64).reshape(-1, 2),
        )
        for shift in blocks.shifts:
            if shift in mapping:
                fields[f"{mapping_name}[{shift[0]},{shift[1]}]"] = _sha256_array(
                    f"{cache_name}.{mapping_name}[{shift[0]},{shift[1]}]",
                    np.asarray(mapping[shift]),
                )
    return _sha256_fields(f"{cache_name}-v1", fields)


def _target_interaction_context_sha256(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    target_blocks: HFOverlapBlockSet,
    target_source_blocks: HFOverlapBlockSet,
) -> str:
    source_fields = {
        "config": _sha256_json("config", asdict(data.config)),
        "k_grid_frac": _sha256_array("k_grid_frac", data.k_grid_frac),
        "kvec": _sha256_array("kvec", data.kvec),
        "h0": _sha256_array("h0", data.h0),
        "wavefunctions": _sha256_array("wavefunctions", data.wavefunctions),
        "reference_density": _sha256_array(
            "reference_density",
            data.reference_density,
        ),
        "band_indices": _sha256_array(
            "band_indices",
            np.asarray(data.band_indices, dtype=np.int64),
        ),
    }
    source_gauge_sha256 = _sha256_fields(
        "htqg-source-gauge-receipt-v1",
        source_fields,
    )
    target_fields = {
        "kvec": _sha256_array("target.kvec", target.kvec),
        "h0": _sha256_array("target.h0", target.h0),
        "wavefunctions": _sha256_array("target.wavefunctions", target.wavefunctions),
        "band_indices": _sha256_array(
            "target.band_indices",
            np.asarray(
                () if target.band_indices is None else target.band_indices,
                dtype=np.int64,
            ),
        ),
    }
    target_payload_sha256 = _sha256_fields(
        "htqg-projected-hf-target-payload-v1",
        target_fields,
    )
    owner_sha256 = _sha256_fields(
        "htqg-target-interaction-owner-v1",
        {
            "source_gauge": source_gauge_sha256,
            "target_payload": target_payload_sha256,
        },
    )
    return _sha256_fields(
        "htqg-target-interaction-context-v1",
        {
            "owner": owner_sha256,
            "target_overlap_blocks": _overlap_block_set_sha256(
                target_blocks,
                owner_sha256=owner_sha256,
                cache_name="htqg-target-overlap-cache",
            ),
            "target_source_overlap_blocks": _overlap_block_set_sha256(
                target_source_blocks,
                owner_sha256=owner_sha256,
                cache_name="htqg-target-source-overlap-cache",
            ),
        },
    )


def _target_interaction_identity(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
) -> tuple[object, ...]:
    settings = data.config.interaction
    return (
        int(data.nt),
        int(data.nk),
        int(target.nt),
        int(target.nk),
        tuple(data.shifts),
        tuple(complex(value) for value in np.asarray(data.shift_gvecs).reshape(-1)),
        tuple(complex(value) for value in np.asarray(data.kvec).reshape(-1)),
        tuple(complex(value) for value in np.asarray(target.kvec).reshape(-1)),
        complex(data.d12),
        complex(data.d34),
        None if target.band_indices is None else tuple(int(value) for value in target.band_indices),
        float(data.v0),
        float(settings.epsilon_r),
        float(settings.d_sc_nm),
        int(settings.g_shells),
        bool(settings.include_hartree),
        bool(settings.include_fock),
        str(settings.hartree_reference),
        str(settings.fock_density),
        bool(settings.finite_zero_limit),
    )


def prepare_htqg_hf_target_interaction(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
) -> HTQGProjectedHFTargetInteractionContext:
    """Precompute complete target and target/source form-factor caches."""

    target_blocks, target_source_blocks = _target_overlap_blocks(data, target)
    return HTQGProjectedHFTargetInteractionContext(
        identity=_target_interaction_identity(data, target),
        identity_sha256=_target_interaction_context_sha256(
            data,
            target,
            target_blocks,
            target_source_blocks,
        ),
        target_overlap_blocks=target_blocks,
        target_source_overlap_blocks=target_source_blocks,
    )


def assemble_htqg_hf_target_hamiltonian(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    target_interaction_context: HTQGProjectedHFTargetInteractionContext | None = None,
    use_numba: bool | None = None,
) -> HTQGProjectedHFTargetHamiltonianAssembly:
    """Assemble a fixed-density target operator without hiding raw residuals."""

    source_blocks = (
        build_htqg_overlap_blocks(data)
        if source_overlap_blocks is None
        else source_overlap_blocks
    )
    context = (
        prepare_htqg_hf_target_interaction(data, target)
        if target_interaction_context is None
        else target_interaction_context
    )
    if context.identity != _target_interaction_identity(data, target):
        raise ValueError(
            "Target interaction context does not belong to this source/target/configuration."
        )
    if target_interaction_context is not None:
        expected_context_sha256 = _target_interaction_context_sha256(
            data,
            target,
            context.target_overlap_blocks,
            context.target_source_overlap_blocks,
        )
        if not hmac.compare_digest(str(context.identity_sha256), expected_context_sha256):
            raise ValueError(
                "Target interaction context payload hash does not match this source/target/configuration."
            )
    source_hartree, source_fock = _split_overlap_blocks(source_blocks)
    h0 = np.asarray(target.h0, dtype=np.complex128).copy()
    current = h0.copy()
    hartree = np.zeros_like(h0)
    fock = np.zeros_like(h0)
    if data.config.interaction.include_hartree:
        updated = build_projected_target_hamiltonian(
            current,
            _hartree_density(data, density),
            source_overlap_blocks=source_hartree,
            target_overlap_blocks=context.target_overlap_blocks,
            target_source_overlap_blocks=HFOverlapBlockSet(
                shifts=context.target_source_overlap_blocks.shifts,
                gvecs=context.target_source_overlap_blocks.gvecs,
                overlaps=context.target_source_overlap_blocks.overlaps,
                diagonal_overlaps={},
                hartree_screening={},
                fock_screening={},
            ),
            v0=data.v0,
            beta=1.0,
            use_numba=use_numba,
        )
        hartree = updated - current
        current = updated
    if data.config.interaction.include_fock:
        updated = build_projected_target_hamiltonian(
            current,
            _fock_density(data, density),
            source_overlap_blocks=source_fock,
            target_overlap_blocks=context.target_overlap_blocks,
            target_source_overlap_blocks=context.target_source_overlap_blocks,
            v0=data.v0,
            beta=1.0,
            use_numba=use_numba,
        )
        fock = updated - current
        current = updated
    raw = current
    hermitian = 0.5 * (raw + raw.swapaxes(0, 1).conj())
    return HTQGProjectedHFTargetHamiltonianAssembly(
        h0=h0,
        hartree=hartree,
        fock=fock,
        raw=raw,
        hermitian=hermitian,
        hermitian_correction=hermitian - raw,
    )


def _finite_numeric_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a real or complex numeric dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_htqg_target_action_inputs(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    density: np.ndarray,
    vectors: np.ndarray,
    *,
    use_numba: bool | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_htqg_projected_hf_config(data.config)
    if use_numba is not None and not isinstance(use_numba, (bool, np.bool_)):
        raise TypeError(f"use_numba must be bool or None, got {type(use_numba).__name__}")

    source_h0 = _finite_numeric_array("data.h0", data.h0)
    if source_h0.ndim != 3 or source_h0.shape[0] != source_h0.shape[1]:
        raise ValueError(f"Expected data.h0 shape (nt, nt, nk), got {source_h0.shape}")
    nt_source, _, nk_source = source_h0.shape
    source_kvec = _finite_numeric_array("data.kvec", data.kvec)
    if source_kvec.shape != (nk_source,):
        raise ValueError(f"Expected data.kvec shape {(nk_source,)}, got {source_kvec.shape}")
    source_wavefunctions = _finite_numeric_array("data.wavefunctions", data.wavefunctions)
    if source_wavefunctions.ndim != 4:
        raise ValueError(
            "Expected data.wavefunctions shape (basis, band, flavor, k), "
            f"got {source_wavefunctions.shape}"
        )
    expected_source_basis = (
        int(data.embedding.local_basis_size)
        * int(data.embedding.grid_shape[0])
        * int(data.embedding.grid_shape[1])
    )
    if source_wavefunctions.shape[0] != expected_source_basis:
        raise ValueError(
            f"Expected source wavefunction basis dimension {expected_source_basis}, "
            f"got {source_wavefunctions.shape[0]}"
        )
    if source_wavefunctions.shape[2:] != (len(VALLEY_SEQUENCE), nk_source):
        raise ValueError(
            "Source wavefunction flavor/k axes do not match the HTQG source: "
            f"got {source_wavefunctions.shape[2:]}"
        )
    expected_source_nt = (
        len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * source_wavefunctions.shape[1]
    )
    if nt_source != expected_source_nt or len(data.band_indices) != source_wavefunctions.shape[1]:
        raise ValueError(
            "Source h0, projected wavefunctions, and band_indices do not describe "
            "the same HTQG projected basis"
        )
    reference_density = _finite_numeric_array(
        "data.reference_density",
        data.reference_density,
    )
    if reference_density.shape != (nt_source, nt_source, nk_source):
        raise ValueError(
            f"Expected data.reference_density shape {(nt_source, nt_source, nk_source)}, "
            f"got {reference_density.shape}"
        )

    target_h0 = _finite_numeric_array("target.h0", target.h0)
    if target_h0.ndim != 3 or target_h0.shape[0] != target_h0.shape[1]:
        raise ValueError(f"Expected target.h0 shape (nt, nt, nk), got {target_h0.shape}")
    nt_target, _, nk_target = target_h0.shape
    target_kvec = _finite_numeric_array("target.kvec", target.kvec)
    if target_kvec.shape != (nk_target,):
        raise ValueError(f"Expected target.kvec shape {(nk_target,)}, got {target_kvec.shape}")
    target_wavefunctions = _finite_numeric_array(
        "target.wavefunctions",
        target.wavefunctions,
    )
    if target_wavefunctions.ndim != 4:
        raise ValueError(
            "Expected target.wavefunctions shape (basis, band, flavor, k), "
            f"got {target_wavefunctions.shape}"
        )
    if target_wavefunctions.shape[0] != source_wavefunctions.shape[0]:
        raise ValueError(
            "Target/source wavefunctions must share the same embedded parent basis dimension"
        )
    if target_wavefunctions.shape[2:] != (len(VALLEY_SEQUENCE), nk_target):
        raise ValueError(
            "Target wavefunction flavor/k axes do not match target.h0: "
            f"got {target_wavefunctions.shape[2:]}"
        )
    expected_target_nt = (
        len(SPIN_LABELS) * len(VALLEY_SEQUENCE) * target_wavefunctions.shape[1]
    )
    if nt_target != expected_target_nt:
        raise ValueError(
            "Target h0 and projected wavefunctions do not describe the same HTQG projected basis"
        )
    if target.band_indices is not None and len(target.band_indices) != target_wavefunctions.shape[1]:
        raise ValueError(
            "target.band_indices and target.wavefunctions do not describe the same target window"
        )

    density_array = _finite_numeric_array("density", density)
    if density_array.shape != (nt_source, nt_source, nk_source):
        raise ValueError(
            f"Expected density shape {(nt_source, nt_source, nk_source)}, "
            f"got {density_array.shape}"
        )
    vectors_array = _finite_numeric_array("vectors", vectors)
    if (
        vectors_array.ndim != 3
        or vectors_array.shape[0] != nt_target
        or vectors_array.shape[2] != nk_target
    ):
        raise ValueError(
            f"Expected vectors shape ({nt_target}, n_vector, {nk_target}), "
            f"got {vectors_array.shape}"
        )

    shift_gvecs = _finite_numeric_array("data.shift_gvecs", data.shift_gvecs)
    if shift_gvecs.shape != (len(data.shifts),):
        raise ValueError(
            f"Expected one data.shift_gvecs entry per shift, got {shift_gvecs.shape} "
            f"for {len(data.shifts)} shifts"
        )
    if len(set(data.shifts)) != len(data.shifts):
        raise ValueError("data.shifts must be unique and explicitly ordered")
    settings = data.config.interaction
    for setting_name, setting_value in (
        ("epsilon_r", settings.epsilon_r),
        ("d_sc_nm", settings.d_sc_nm),
    ):
        if not np.isfinite(float(setting_value)):
            raise ValueError(
                f"data.config.interaction.{setting_name} must be finite, got {setting_value}"
            )
    if not np.isfinite(float(data.v0)) or float(data.v0) <= 0.0:
        raise ValueError(f"data.v0 must be finite and positive, got {data.v0}")

    return (
        np.asarray(density_array, dtype=np.complex128),
        np.asarray(vectors_array, dtype=np.complex128),
        np.asarray(target_h0, dtype=np.complex128),
        np.asarray(target_wavefunctions, dtype=np.complex128),
    )


def _validate_htqg_target_batch_action_inputs(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    densities: np.ndarray,
    vectors: np.ndarray,
    *,
    use_numba: bool | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    densities_array = _finite_numeric_array("densities", densities)
    expected_density_shape = (data.nt, data.nt, data.nk)
    if densities_array.ndim != 4:
        raise ValueError(
            "Expected densities shape "
            f"(n_batch, {data.nt}, {data.nt}, {data.nk}), got {densities_array.shape}"
        )
    if densities_array.shape[0] == 0:
        raise ValueError("densities batch must be nonempty")
    if densities_array.shape[1:] != expected_density_shape:
        raise ValueError(
            f"Expected densities shape (n_batch, {data.nt}, {data.nt}, {data.nk}), "
            f"got {densities_array.shape}"
        )

    _, vectors_array, target_h0, target_wavefunctions = (
        _validate_htqg_target_action_inputs(
            data,
            target,
            densities_array[0],
            vectors,
            use_numba=use_numba,
        )
    )
    return (
        np.asarray(densities_array, dtype=np.complex128),
        vectors_array,
        target_h0,
        target_wavefunctions,
    )


def _validate_htqg_source_overlap_blocks_for_action(
    data: HTQGProjectedHFData,
    blocks: HFOverlapBlockSet,
) -> None:
    if blocks.shifts != data.shifts or not np.array_equal(
        np.asarray(blocks.gvecs, dtype=np.complex128),
        np.asarray(data.shift_gvecs, dtype=np.complex128),
    ):
        raise ValueError(
            "source_overlap_blocks does not belong to this HTQG source/configuration"
        )
    if set(blocks.overlaps) != set(data.shifts):
        raise ValueError("source_overlap_blocks overlap keys must exactly match data.shifts")
    for mapping_name, mapping in (
        ("diagonal_overlaps", blocks.diagonal_overlaps),
        ("hartree_screening", blocks.hartree_screening),
        ("fock_screening", blocks.fock_screening),
    ):
        if not set(mapping).issubset(data.shifts):
            raise ValueError(f"source_overlap_blocks.{mapping_name} has an unknown shift")

    expected_overlap_shape = (data.nt, data.nk, data.nt, data.nk)
    for shift in data.shifts:
        overlap = _finite_numeric_array(
            f"source_overlap_blocks.overlaps[{shift}]",
            blocks.overlaps[shift],
        )
        if overlap.shape != expected_overlap_shape:
            raise ValueError(
                f"Expected source overlap shape {expected_overlap_shape}, "
                f"got {overlap.shape} for shift {shift}"
            )
        diagonal = blocks.diagonal_overlaps.get(shift)
        if diagonal is not None:
            diagonal_array = _finite_numeric_array(
                f"source_overlap_blocks.diagonal_overlaps[{shift}]",
                diagonal,
            )
            if diagonal_array.shape != (data.nt, data.nt, data.nk):
                raise ValueError(
                    f"Expected source diagonal overlap shape {(data.nt, data.nt, data.nk)}, "
                    f"got {diagonal_array.shape} for shift {shift}"
                )
        hartree_kernel = blocks.hartree_screening.get(shift)
        if hartree_kernel is not None:
            kernel_array = _finite_numeric_array(
                f"source_overlap_blocks.hartree_screening[{shift}]",
                np.asarray(hartree_kernel),
            )
            if kernel_array.shape != ():
                raise ValueError(f"Hartree screening for shift {shift} must be scalar")
        fock_kernel = blocks.fock_screening.get(shift)
        if fock_kernel is not None:
            kernel_array = _finite_numeric_array(
                f"source_overlap_blocks.fock_screening[{shift}]",
                fock_kernel,
            )
            if kernel_array.shape != (data.nk, data.nk):
                raise ValueError(
                    f"Expected source Fock screening shape {(data.nk, data.nk)}, "
                    f"got {kernel_array.shape} for shift {shift}"
                )






def apply_htqg_hf_target_hamiltonian(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    density: np.ndarray,
    vectors: np.ndarray,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    use_numba: bool | None = None,
) -> HTQGProjectedHFTargetHamiltonianAction:
    """Apply fixed-density target HF one target point at a time.

    The source form factors and Hartree source traces are reused, while target
    form factors are restricted to one target point.  No target-square HF
    Hamiltonian or all-target overlap block is assembled.
    """

    density_array, vectors_array, target_h0, target_wavefunctions = (
        _validate_htqg_target_action_inputs(
            data,
            target,
            density,
            vectors,
            use_numba=use_numba,
        )
    )
    source_basis = _projected_basis(data, name="htqg-source-grid")
    source_blocks = (
        _build_htqg_overlap_blocks_from_basis(data, source_basis)
        if source_overlap_blocks is None
        else source_overlap_blocks
    )
    _validate_htqg_source_overlap_blocks_for_action(data, source_blocks)

    scale = float(data.v0) / data.nk
    hartree_density = _hartree_density(data, density_array)
    fock_density = _fock_density(data, density_array)
    hartree_source_traces: dict[tuple[int, int], complex] = {}
    if data.config.interaction.include_hartree:
        for shift in source_blocks.shifts:
            if source_blocks.hartree_screening.get(shift) is None:
                continue
            source_diagonal = source_blocks.diagonal_overlaps.get(shift)
            if source_diagonal is None:
                raise ValueError(f"Missing source diagonal overlap for active Hartree shift {shift}")
            hartree_source_traces[shift] = compute_density_overlap_trace_from_diagonal(
                hartree_density,
                source_diagonal,
                use_numba=use_numba,
            )

    h0_action = np.einsum("abk,brk->ark", target_h0, vectors_array, optimize=True)
    hartree_action = np.zeros_like(h0_action)
    fock_action = np.zeros_like(h0_action)
    settings = data.config.interaction
    for ik in range(target.nk):
        target_point_basis = ProjectedWavefunctionBasis(
            target_wavefunctions[:, :, :, ik : ik + 1],
            data.embedding.grid_shape,
            n_spin=len(SPIN_LABELS),
            local_basis_size=data.embedding.local_basis_size,
            name=f"htqg-target-point-{ik}",
            boundary_mode="zero_fill",
        )
        point_vectors = vectors_array[:, :, ik : ik + 1]
        for shift, gvec in zip(source_blocks.shifts, source_blocks.gvecs, strict=True):
            if shift in hartree_source_traces:
                target_overlap = _calculate_htqg_projected_overlap_between(
                    target_point_basis,
                    target_point_basis,
                    int(shift[0]),
                    int(shift[1]),
                )
                target_diagonal = target_overlap[:, 0, :, 0]
                hartree_coefficient = (
                    scale
                    * float(source_blocks.hartree_screening[shift])
                    * hartree_source_traces[shift]
                )
                if hartree_coefficient != 0.0:
                    hartree_action[:, :, ik] += (
                        hartree_coefficient * (target_diagonal @ vectors_array[:, :, ik])
                    )

            if settings.include_fock:
                target_source_overlap = _calculate_htqg_projected_overlap_between(
                    target_point_basis,
                    source_basis,
                    int(shift[0]),
                    int(shift[1]),
                )
                qvals = data.kvec[None, :] - target.kvec[ik] + complex(gvec)
                fock_screening = np.asarray(
                    screened_coulomb(
                        qvals,
                        epsilon_r=float(settings.epsilon_r),
                        d_sc_nm=float(settings.d_sc_nm),
                        finite_zero_limit=bool(settings.finite_zero_limit),
                    ),
                    dtype=float,
                )
                fock_action[:, :, ik : ik + 1] -= contract_fock_action_from_overlap(
                    target_source_overlap,
                    fock_density,
                    scale * fock_screening,
                    point_vectors,
                )

    raw_action = h0_action + hartree_action
    raw_action += fock_action
    return HTQGProjectedHFTargetHamiltonianAction(
        h0=h0_action,
        hartree=hartree_action,
        fock=fock_action,
        raw=raw_action,
    )


def apply_htqg_hf_target_hamiltonian_batch(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    densities: np.ndarray,
    vectors: np.ndarray,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    use_numba: bool | None = None,
) -> HTQGProjectedHFTargetHamiltonianBatchAction:
    """Apply several fixed densities while reusing each target point/shift overlap."""

    densities_array, vectors_array, target_h0, target_wavefunctions = (
        _validate_htqg_target_batch_action_inputs(
            data,
            target,
            densities,
            vectors,
            use_numba=use_numba,
        )
    )
    source_basis = _projected_basis(data, name="htqg-source-grid")
    source_blocks = (
        _build_htqg_overlap_blocks_from_basis(data, source_basis)
        if source_overlap_blocks is None
        else source_overlap_blocks
    )
    _validate_htqg_source_overlap_blocks_for_action(data, source_blocks)

    n_batch = int(densities_array.shape[0])
    scale = float(data.v0) / data.nk
    hartree_densities = _hartree_density(data, densities_array)
    fock_densities = _fock_density(data, densities_array)
    hartree_source_traces: dict[tuple[int, int], np.ndarray] = {}
    if data.config.interaction.include_hartree:
        for shift in source_blocks.shifts:
            if source_blocks.hartree_screening.get(shift) is None:
                continue
            source_diagonal = source_blocks.diagonal_overlaps.get(shift)
            if source_diagonal is None:
                raise ValueError(f"Missing source diagonal overlap for active Hartree shift {shift}")
            traces = np.empty(n_batch, dtype=np.complex128)
            for ibatch in range(n_batch):
                traces[ibatch] = compute_density_overlap_trace_from_diagonal(
                    hartree_densities[ibatch],
                    source_diagonal,
                    use_numba=use_numba,
                )
            hartree_source_traces[shift] = traces

    single_h0_action = np.einsum(
        "abk,brk->ark",
        target_h0,
        vectors_array,
        optimize=True,
    )
    h0_action = np.repeat(single_h0_action[None, ...], n_batch, axis=0)
    hartree_action = np.zeros_like(h0_action)
    fock_action = np.zeros_like(h0_action)
    settings = data.config.interaction
    for ik in range(target.nk):
        target_point_basis = ProjectedWavefunctionBasis(
            target_wavefunctions[:, :, :, ik : ik + 1],
            data.embedding.grid_shape,
            n_spin=len(SPIN_LABELS),
            local_basis_size=data.embedding.local_basis_size,
            name=f"htqg-target-point-{ik}",
            boundary_mode="zero_fill",
        )
        point_vectors = vectors_array[:, :, ik : ik + 1]
        for shift, gvec in zip(source_blocks.shifts, source_blocks.gvecs, strict=True):
            if shift in hartree_source_traces:
                target_overlap = _calculate_htqg_projected_overlap_between(
                    target_point_basis,
                    target_point_basis,
                    int(shift[0]),
                    int(shift[1]),
                )
                target_diagonal = target_overlap[:, 0, :, 0]
                hartree_coefficients = (
                    scale
                    * float(source_blocks.hartree_screening[shift])
                    * hartree_source_traces[shift]
                )
                if np.any(hartree_coefficients != 0.0):
                    target_vector_action = target_diagonal @ vectors_array[:, :, ik]
                    for ibatch, coefficient in enumerate(hartree_coefficients):
                        if coefficient != 0.0:
                            hartree_action[ibatch, :, :, ik] += (
                                coefficient * target_vector_action
                            )

            if settings.include_fock:
                target_source_overlap = _calculate_htqg_projected_overlap_between(
                    target_point_basis,
                    source_basis,
                    int(shift[0]),
                    int(shift[1]),
                )
                qvals = data.kvec[None, :] - target.kvec[ik] + complex(gvec)
                fock_screening = np.asarray(
                    screened_coulomb(
                        qvals,
                        epsilon_r=float(settings.epsilon_r),
                        d_sc_nm=float(settings.d_sc_nm),
                        finite_zero_limit=bool(settings.finite_zero_limit),
                    ),
                    dtype=float,
                )
                for ibatch in range(n_batch):
                    fock_action[ibatch, :, :, ik : ik + 1] -= (
                        contract_fock_action_from_overlap(
                            target_source_overlap,
                            fock_densities[ibatch],
                            scale * fock_screening,
                            point_vectors,
                        )
                    )

    raw_action = h0_action + hartree_action
    raw_action += fock_action
    return HTQGProjectedHFTargetHamiltonianBatchAction(
        h0=h0_action,
        hartree=hartree_action,
        fock=fock_action,
        raw=raw_action,
    )





def build_htqg_hf_target_hamiltonian(
    data: HTQGProjectedHFData,
    target: HTQGProjectedHFTargetData,
    density: np.ndarray,
    *,
    source_overlap_blocks: HFOverlapBlockSet | None = None,
    target_interaction_context: HTQGProjectedHFTargetInteractionContext | None = None,
    use_numba: bool | None = None,
) -> np.ndarray:
    """Evaluate the fixed HF density on an arbitrary direct target k path."""

    return assemble_htqg_hf_target_hamiltonian(
        data,
        target,
        density,
        source_overlap_blocks=source_overlap_blocks,
        target_interaction_context=target_interaction_context,
        use_numba=use_numba,
    ).hermitian


def diagonalize_htqg_hf_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    energies = np.zeros((hamiltonian.shape[0], hamiltonian.shape[2]), dtype=float)
    for ik in range(hamiltonian.shape[2]):
        energies[:, ik] = np.linalg.eigvalsh(hamiltonian[:, :, ik])
    return energies


def gap_estimate(energies: np.ndarray, n_occupied_per_k: int) -> float | None:
    arr = np.asarray(energies, dtype=float)
    nocc = int(n_occupied_per_k)
    if nocc <= 0 or nocc >= arr.shape[0]:
        return None
    return float(np.min(arr[nocc:, :]) - np.max(arr[:nocc, :]))


def occupation_by_label(data: HTQGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    diag = np.real(np.diagonal(np.asarray(density), axis1=0, axis2=1).T)
    out: dict[str, float] = {}
    for label in data.labels:
        key = f"{label.spin}_{label.valley_label}_band{label.band_position}"
        out[key] = float(np.mean(diag[label.index, :]))
    return out


def _numeric_order_parameters(data: HTQGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    occ = occupation_by_label(data, density)
    spin_up = sum(value for key, value in occ.items() if key.startswith("up_"))
    spin_down = sum(value for key, value in occ.items() if key.startswith("down_"))
    valley_k = sum(value for key, value in occ.items() if "_K_" in key)
    valley_kp = sum(value for key, value in occ.items() if "_Kprime_" in key)
    return {
        "spin_polarization": float(spin_up - spin_down),
        "valley_polarization": float(valley_k - valley_kp),
        "total_occupation": float(spin_up + spin_down),
    }


__all__ = [
    "HTQGInteractionSettings",
    "HTQGProjectedHFConfig",
    "HTQGProjectedHFData",
    "HTQGProjectedHFTargetData",
    "HTQGProjectedHFTargetInteractionContext",
    "HTQGProjectedHFTargetHamiltonianAssembly",
    "HTQGProjectedHFTargetHamiltonianAction",
    "HTQGProjectedHFTargetHamiltonianBatchAction",
    "HTQGHartreeFockResult",
    "apply_htqg_hf_target_hamiltonian",
    "apply_htqg_hf_target_hamiltonian_batch",
    "assemble_htqg_hf_target_hamiltonian",
    "build_htqg_delta_interaction_components",
    "build_htqg_hf_target_hamiltonian",
    "build_htqg_overlap_blocks",
    "build_htqg_projected_hf_data",
    "build_htqg_target_data",
    "prepare_htqg_hf_target_interaction",
    "diagonalize_htqg_hf_hamiltonian",
    "gap_estimate",
    "occupation_by_label",
    "run_htqg_projected_hf",
]
