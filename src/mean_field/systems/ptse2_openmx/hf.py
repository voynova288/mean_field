"""PtSe2 adapters from physical-Q form factors to the generic HF core."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

from mean_field.core.hf.engine import (
    DensityUpdateResult,
    HartreeFockRun,
    HartreeFockStepResult,
)
from mean_field.core.hf.overlap import HFOverlapBlockSet
from mean_field.core.hf.density import ket_projector_to_stored_orientation
from mean_field.core.hf.interaction import run_projected_hartree_fock

from .screening import PtSe2DoubleGateScreening
from .source import ANGSTROM_TO_BOHR, OpenMXSource


BOHR_INV_TO_NM_INV = 10.0 / (1.0 / ANGSTROM_TO_BOHR)


@dataclass(frozen=True)
class PtSe2HFOverlapCache:
    blocks: HFOverlapBlockSet
    pair_masks: dict[tuple[int, int], np.ndarray]
    channel_q_indices: dict[tuple[int, int], np.ndarray]
    folding_carries: np.ndarray
    reduced_steps: np.ndarray
    physical_q_numerators: np.ndarray
    physical_q_bohr_inv: np.ndarray
    source_cache: str
    source_summary: str
    cache_sha256: str
    run_id: str
    active_rank: int
    input_cache_schema: str
    eigensystem_manifest_sha256: str | None
    hf_authorized: bool


def _readonly_copy(values: np.ndarray, *, dtype=None) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def build_ptse2_hf_overlap_cache(
    cache_path: str | Path,
    summary_path: str | Path,
    source: OpenMXSource,
    screening: PtSe2DoubleGateScreening,
    *,
    require_hf_authorized: bool = True,
    expected_eigensystem_manifest_sha256: str | None = None,
) -> PtSe2HFOverlapCache:
    """Build a masked ``HFOverlapBlockSet`` from one projected-Q cache.

    Pair masks remain system-owned audit data.  Invalid pair/key entries are
    exact zero in both the overlap and Fock kernel before the generic HF core
    sees the block set.
    """

    cache_file = Path(cache_path).resolve()
    summary_file = Path(summary_path).resolve()
    summary = json.loads(summary_file.read_text())
    cache_sha256 = _sha256_file(cache_file)
    expected_cache_sha256 = summary.get("outputs", {}).get(
        "projected_cache_sha256"
    )
    if expected_cache_sha256 != cache_sha256:
        raise ValueError("projected cache hash differs from its summary")
    traced_manifest_sha256: str | None = None
    if summary.get("input_cache_schema") == "ptse2_full_q_control_variate_cache/v1":
        control_summary_path = cache_file.parent / "control_variate_summary.json"
        control_summary = json.loads(control_summary_path.read_text())
        if (
            control_summary.get("outputs", {}).get("cache_sha256")
            != summary.get("inputs", {}).get("raw_cache_sha256")
        ):
            raise ValueError("final cache does not bind the control-variate cache")
        traced_manifest_sha256 = control_summary.get("inputs", {}).get(
            "manifest_sha256"
        )
    if (
        expected_eigensystem_manifest_sha256 is not None
        and traced_manifest_sha256 != expected_eigensystem_manifest_sha256
    ):
        raise ValueError("form-factor chain and eigensystem manifest hashes differ")
    hf_authorized = bool(summary.get("hf_authorized", False)) and bool(
        summary.get("qualification", {}).get("hf_authorized", False)
    )
    if require_hf_authorized and not hf_authorized:
        raise ValueError("form-factor cache is not HF-authorized")

    with np.load(cache_file, allow_pickle=False) as payload:
        schema = str(payload["schema"])
        if schema not in {
            "ptse2_full_q_reverse_c3_projected_cache/v1",
            "ptse2_full_q_reverse_c3_projected_cache/v2",
        }:
            raise ValueError(f"unsupported projected-cache schema {schema!r}")
        run_id = str(payload["run_id"])
        active_rank_recorded = int(payload["active_rank"])
        input_cache_schema = str(payload["input_cache_schema"])
        local_fields = np.asarray(payload["local_fields"], dtype=np.int64)
        admitted_mask = np.asarray(payload["admitted_mask"], dtype=bool)
        q_numerators = np.asarray(
            payload["physical_q_numerators"], dtype=np.int64
        )
        q_bohr = np.asarray(payload["physical_q_bohr_inv"], dtype=np.float64)
        channel_q = np.asarray(payload["channel_q_indices"], dtype=np.int64)
        reduced_steps = np.asarray(payload["reduced_steps"], dtype=np.int64)
        folding_carries = np.asarray(
            payload["folding_carries"], dtype=np.int64
        )
        overlaps_array = np.asarray(
            payload["projected_overlaps"], dtype=np.complex128
        )

    if summary.get("run_id") != run_id:
        raise ValueError("projected cache/summary run identity mismatch")
    if summary.get("input_cache_schema") != input_cache_schema:
        raise ValueError("projected cache/summary parent-cache schema mismatch")
    if local_fields.ndim != 2 or local_fields.shape[1] != 3:
        raise ValueError("local fields must have shape (n_local, 3)")
    if np.any(local_fields[:, 2] != 0):
        raise ValueError("the first PtSe2 HF bridge supports in-plane local fields only")
    n_local, active_rank, nk, active_rank_rhs, nk_rhs = overlaps_array.shape
    if active_rank != active_rank_rhs or nk != nk_rhs:
        raise ValueError("projected overlap blocks must be square")
    if active_rank_recorded != active_rank:
        raise ValueError("projected cache active-rank record differs from its arrays")
    if admitted_mask.shape != (n_local, nk, nk):
        raise ValueError("pair-mask shape mismatch")
    if channel_q.shape != admitted_mask.shape:
        raise ValueError("channel-Q table shape mismatch")
    if q_numerators.ndim != 2 or q_numerators.shape[1] != 3:
        raise ValueError("physical-Q numerators must have shape (n_q, 3)")
    if q_bohr.shape != q_numerators.shape:
        raise ValueError("physical-Q numerator/Cartesian table shape mismatch")
    mesh = int(round(np.sqrt(nk)))
    if mesh * mesh != nk:
        raise ValueError("projected overlap cache is not one square k mesh")
    expected_q_bohr = (
        q_numerators.astype(np.float64) / float(mesh)
    ) @ source.reciprocal_bohr
    if not np.allclose(q_bohr, expected_q_bohr, rtol=0.0, atol=1.0e-12):
        raise ValueError("physical-Q numerator and Cartesian tables disagree")
    admitted_q = channel_q[admitted_mask]
    if admitted_q.size and (
        int(np.min(admitted_q)) < 0 or int(np.max(admitted_q)) >= q_numerators.shape[0]
    ):
        raise ValueError("admitted channel-Q index is outside the physical-Q table")
    if reduced_steps.shape != (nk, nk, 3) or folding_carries.shape != (
        nk,
        nk,
        3,
    ):
        raise ValueError("pair transfer-table shape mismatch")

    shifts = tuple((int(row[0]), int(row[1])) for row in local_fields)
    if len(set(shifts)) != len(shifts):
        raise ValueError("local-field keys are not unique")
    local_cart_bohr = local_fields @ source.reciprocal_bohr
    local_cart_nm = local_cart_bohr * BOHR_INV_TO_NM_INV
    gvecs = _readonly_copy(
        local_cart_nm[:, 0] + 1j * local_cart_nm[:, 1],
        dtype=np.complex128,
    )
    q_nm_inv = np.linalg.norm(q_bohr, axis=1) * BOHR_INV_TO_NM_INV
    w_by_q = np.asarray(
        screening.interaction_ev_nm2(q_nm_inv), dtype=np.float64
    )

    overlaps: dict[tuple[int, int], np.ndarray] = {}
    diagonal: dict[tuple[int, int], np.ndarray] = {}
    hartree: dict[tuple[int, int], float] = {}
    fock: dict[tuple[int, int], np.ndarray] = {}
    pair_masks: dict[tuple[int, int], np.ndarray] = {}
    channel_tables: dict[tuple[int, int], np.ndarray] = {}
    for local_index, shift in enumerate(shifts):
        mask = admitted_mask[local_index]
        block = np.array(overlaps_array[local_index], copy=True)
        block *= mask[None, :, None, :]
        overlaps[shift] = _readonly_copy(block)
        diagonal_block = np.empty(
            (active_rank, active_rank, nk), dtype=np.complex128
        )
        for k_index in range(nk):
            diagonal_block[:, :, k_index] = block[
                :, k_index, :, k_index
            ]
        diagonal[shift] = _readonly_copy(diagonal_block)

        kernel = np.zeros((nk, nk), dtype=np.float64)
        q_indices = channel_q[local_index]
        kernel[mask] = w_by_q[q_indices[mask]]
        fock[shift] = _readonly_copy(kernel)
        pair_masks[shift] = _readonly_copy(mask, dtype=bool)
        channel_tables[shift] = _readonly_copy(q_indices, dtype=np.int64)

        diagonal_mask = np.diag(mask)
        if np.all(diagonal_mask):
            diagonal_q = np.diag(q_indices)
            if np.unique(diagonal_q).size != 1:
                raise ValueError("Hartree diagonal does not use one physical Q")
            hartree[shift] = float(w_by_q[int(diagonal_q[0])])
        elif np.any(diagonal_mask):
            raise ValueError("partially admitted Hartree diagonal is unsupported")

    blocks = HFOverlapBlockSet(
        shifts=shifts,
        gvecs=gvecs,
        overlaps=overlaps,
        diagonal_overlaps=diagonal,
        hartree_screening=hartree,
        fock_screening=fock,
    )
    return PtSe2HFOverlapCache(
        blocks=blocks,
        pair_masks=pair_masks,
        channel_q_indices=channel_tables,
        folding_carries=_readonly_copy(folding_carries, dtype=np.int64),
        reduced_steps=_readonly_copy(reduced_steps, dtype=np.int64),
        physical_q_numerators=_readonly_copy(q_numerators, dtype=np.int64),
        physical_q_bohr_inv=_readonly_copy(q_bohr, dtype=np.float64),
        source_cache=str(cache_file),
        source_summary=str(summary_file),
        cache_sha256=cache_sha256,
        run_id=run_id,
        active_rank=active_rank,
        input_cache_schema=input_cache_schema,
        eigensystem_manifest_sha256=traced_manifest_sha256,
        hf_authorized=hf_authorized,
    )


@dataclass(frozen=True)
class PtSe2HFConfig:
    """Fixed-filling top-valence PtSe2 projected-HF inputs."""

    filling_nu: int
    dielectric_constant: float
    gate_distance_from_plane_nm: float = 20.0
    active_rank: int = 8
    occupied_below_ev: float = -4.498290056263158
    precision: float = 1.0e-7
    max_iter: int = 300
    init_mode: str = "bare"
    seed: int = 1
    random_rotation_strength: float = 0.08
    occupation_degeneracy_tolerance_ev: float = 1.0e-10
    degenerate_occupation_policy: str = "fail"
    oda_mode: str = "default"
    convergence_rule: str = "raw"
    use_numba: bool = True
    require_hf_authorized: bool = True

    def __post_init__(self) -> None:
        if int(self.active_rank) <= 0:
            raise ValueError("active_rank must be positive")
        if int(self.filling_nu) != self.filling_nu:
            raise ValueError("primitive-cell filling_nu must be an integer")
        if not -int(self.active_rank) <= int(self.filling_nu) <= 0:
            raise ValueError("valence-only filling_nu must lie in [-active_rank, 0]")
        if not np.isfinite(self.dielectric_constant) or self.dielectric_constant <= 0.0:
            raise ValueError("dielectric_constant must be finite and positive")
        if (
            not np.isfinite(self.gate_distance_from_plane_nm)
            or self.gate_distance_from_plane_nm <= 0.0
        ):
            raise ValueError("gate_distance_from_plane_nm must be finite and positive")
        if not np.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be finite and positive")
        if int(self.max_iter) <= 0:
            raise ValueError("max_iter must be positive")
        if (
            not np.isfinite(self.occupation_degeneracy_tolerance_ev)
            or self.occupation_degeneracy_tolerance_ev < 0.0
        ):
            raise ValueError(
                "occupation_degeneracy_tolerance_ev must be finite and nonnegative"
            )
        if self.oda_mode not in {"default", "direct"}:
            raise ValueError("oda_mode must be 'default' or 'direct'")
        if self.degenerate_occupation_policy not in {"fail", "equal_ensemble"}:
            raise ValueError(
                "degenerate_occupation_policy must be 'fail' or 'equal_ensemble'"
            )
        if self.init_mode not in {"bare", "random_coherent"}:
            raise ValueError("init_mode must be 'bare' or 'random_coherent'")
        if not 0.0 <= float(self.random_rotation_strength) <= 1.0:
            raise ValueError("random_rotation_strength must lie in [0, 1]")
        if self.convergence_rule not in {"raw", "mixed"}:
            raise ValueError("convergence_rule must be 'raw' or 'mixed'")


@dataclass(frozen=True)
class PtSe2ProjectedHFData:
    config: PtSe2HFConfig
    source: OpenMXSource
    k_fractional: np.ndarray
    active_energies_ev: np.ndarray
    h0: np.ndarray
    initial_density: np.ndarray
    overlap_cache: PtSe2HFOverlapCache
    eigensystem_root: str
    eigensystem_manifest_sha256: str
    edge_summary_sha256: str
    area_nm2: float


@dataclass
class PtSe2HFState:
    h0: np.ndarray
    density: np.ndarray
    hamiltonian: np.ndarray
    energies: np.ndarray
    precision: float
    v0: float
    mu: float = float("nan")
    diagnostics: dict[str, float] = field(default_factory=dict)

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ptse2_density_from_fixed_filling(
    hamiltonian: np.ndarray,
    filling_nu: int,
    *,
    degeneracy_tolerance_ev: float = 1.0e-10,
    degenerate_occupation_policy: str = "fail",
) -> DensityUpdateResult:
    """Return stored ``D=P-I`` at one integer primitive-cell filling.

    Occupations are selected globally over the half-open k mesh.  A boundary
    multiplet either fails closed or receives one explicitly declared equal
    zero-temperature ensemble occupation.  The ket projector is transposed
    exactly once into Mean_Field's stored orientation.
    """

    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim != 3 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must have shape (active_rank, active_rank, nk)")
    active_rank, _, nk = h.shape
    nu = int(filling_nu)
    if nu != filling_nu or not -active_rank <= nu <= 0:
        raise ValueError("filling_nu must be one integer in [-active_rank, 0]")
    tolerance = float(degeneracy_tolerance_ev)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("degeneracy_tolerance_ev must be finite and nonnegative")
    if degenerate_occupation_policy not in {"fail", "equal_ensemble"}:
        raise ValueError(
            "degenerate_occupation_policy must be 'fail' or 'equal_ensemble'"
        )
    total_occupied = (active_rank + nu) * nk
    energies = np.empty((active_rank, nk), dtype=np.float64)
    eigenvectors = np.empty_like(h)
    for k_index in range(nk):
        block = 0.5 * (h[:, :, k_index] + h[:, :, k_index].conj().T)
        energies[:, k_index], eigenvectors[:, :, k_index] = np.linalg.eigh(block)
    flattened_unsorted = energies.ravel(order="F")
    order = np.argsort(flattened_unsorted, kind="stable")
    flattened = flattened_unsorted[order]
    weights_flat = np.zeros(flattened_unsorted.size, dtype=np.float64)
    boundary_multiplet_size = 0
    boundary_occupation_fraction = 0.0
    ensemble_used = False
    if total_occupied == 0:
        mu = float(np.nextafter(flattened[0], -np.inf))
        occupation_gap = float("inf")
    elif total_occupied == flattened.size:
        weights_flat.fill(1.0)
        mu = float(np.nextafter(flattened[-1], np.inf))
        occupation_gap = float("inf")
    else:
        occupied_edge = float(flattened[total_occupied - 1])
        empty_edge = float(flattened[total_occupied])
        mu = 0.5 * (occupied_edge + empty_edge)
        occupation_gap = empty_edge - occupied_edge
        if occupation_gap <= tolerance:
            if degenerate_occupation_policy == "fail":
                raise ValueError(
                    "fixed-filling occupation boundary is degenerate within "
                    f"{tolerance:.3e} eV: gap={occupation_gap:.3e} eV"
                )
            left = total_occupied - 1
            while left > 0 and flattened[left] - flattened[left - 1] <= tolerance:
                left -= 1
            right = total_occupied
            while (
                right + 1 < flattened.size
                and flattened[right + 1] - flattened[right] <= tolerance
            ):
                right += 1
            boundary_multiplet_size = right - left + 1
            boundary_occupation_fraction = (
                total_occupied - left
            ) / boundary_multiplet_size
            weights_flat[order[:left]] = 1.0
            weights_flat[order[left : right + 1]] = boundary_occupation_fraction
            mu = float(np.mean(flattened[left : right + 1]))
            ensemble_used = True
        else:
            weights_flat[order[:total_occupied]] = 1.0
    occupation_weights = weights_flat.reshape(energies.shape, order="F")
    identity = np.eye(active_rank, dtype=np.complex128)
    stored_delta = np.empty_like(h)
    if total_occupied == active_rank * nk:
        stored_delta.fill(0.0)
    elif total_occupied == 0:
        stored_delta[:] = -identity[:, :, None]
    else:
        for k_index in range(nk):
            vectors = eigenvectors[:, :, k_index]
            ket_projector = (
                vectors * occupation_weights[:, k_index][None, :]
            ) @ vectors.conj().T
            stored_delta[:, :, k_index] = ket_projector_to_stored_orientation(
                ket_projector - identity
            )
    filling_measured = float(
        sum(np.trace(stored_delta[:, :, k]).real for k in range(nk)) / nk
    )
    return DensityUpdateResult(
        density=stored_delta,
        energies=energies,
        mu=mu,
        observables={
            "filling_nu": filling_measured,
            "total_occupied": float(total_occupied),
            "occupation_gap_ev": occupation_gap,
            "boundary_multiplet_size": float(boundary_multiplet_size),
            "boundary_occupation_fraction": boundary_occupation_fraction,
            "degenerate_ensemble_used": float(ensemble_used),
        },
    )


def _randomly_rotate_initial_density(
    density: np.ndarray,
    *,
    seed: int,
    strength: float,
) -> np.ndarray:
    stored_delta = np.asarray(density, dtype=np.complex128)
    active_rank, _, nk = stored_delta.shape
    if np.max(np.abs(stored_delta), initial=0.0) == 0.0:
        return np.zeros_like(stored_delta)
    rng = np.random.default_rng(int(seed))
    sampled = rng.normal(size=(active_rank, active_rank)) + 1j * rng.normal(
        size=(active_rank, active_rank)
    )
    generator = 0.5 * (sampled + sampled.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    spectral_radius = max(float(np.max(np.abs(eigenvalues))), 1.0e-300)
    phases = np.exp(1j * float(strength) * eigenvalues / spectral_radius)
    unitary = (eigenvectors * phases[None, :]) @ eigenvectors.conj().T
    identity = np.eye(active_rank, dtype=np.complex128)
    rotated = np.empty_like(stored_delta)
    for k_index in range(nk):
        ket_projector = stored_delta[:, :, k_index].T + identity
        ket_rotated = unitary @ ket_projector @ unitary.conj().T
        rotated[:, :, k_index] = ket_projector_to_stored_orientation(
            ket_rotated - identity
        )
    return rotated


def _hermitize_blocks(blocks: np.ndarray) -> None:
    for k_index in range(blocks.shape[2]):
        blocks[:, :, k_index] = 0.5 * (
            blocks[:, :, k_index] + blocks[:, :, k_index].conj().T
        )


def build_ptse2_projected_hf_data(
    config: PtSe2HFConfig,
    *,
    eigensystem_root: str | Path,
    cache_path: str | Path,
    cache_summary_path: str | Path,
    source_input_path: str | Path | None = None,
) -> PtSe2ProjectedHFData:
    root = Path(eigensystem_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha256 = _sha256_file(manifest_path)
    source_path = (
        Path(manifest["source_input"]).resolve()
        if source_input_path is None
        else Path(source_input_path).resolve()
    )
    source = OpenMXSource.from_file(source_path)
    if manifest.get("source_input_sha256") != source.input_sha256:
        raise ValueError("eigensystem/source input hash mismatch")
    mesh = int(manifest["mesh"])
    if mesh < 12:
        raise ValueError("PtSe2 HF requires a half-open mesh of at least 12x12")
    if manifest.get("mesh_shape") != [mesh, mesh]:
        raise ValueError("eigensystem manifest does not declare one square mesh")
    expected_mesh_definition = (
        f"half_open_fractional_ij: k=(i/{mesh},j/{mesh},0), "
        f"i,j=0,...,{mesh - 1}; flat index=i*{mesh}+j"
    )
    if manifest.get("mesh_definition") != expected_mesh_definition:
        raise ValueError("eigensystem mesh is not the required half-open chart")
    nk = mesh * mesh
    active_rank = int(config.active_rank)
    active_energies = np.empty((active_rank, nk), dtype=np.float64)
    run_id = str(manifest["run_id"])
    if abs(float(manifest["target_energy_ev"]) - config.occupied_below_ev) > 1.0e-12:
        raise ValueError("active-window cutoff differs from the eigensystem target")
    edge_summary_path = root / "edge_summary.json"
    edge_summary = json.loads(edge_summary_path.read_text())
    edge_qualification = edge_summary.get("qualification", {})
    if (
        edge_summary.get("edge_run_id") != run_id
        or edge_summary.get("edge_manifest_sha256") != manifest_sha256
        or abs(
            float(edge_summary.get("parent_chemical_potential_ev"))
            - config.occupied_below_ev
        )
        > 1.0e-12
        or not edge_qualification.get("fixed_rank_window_passed")
        or not edge_qualification.get("conduction_isolation_passed")
    ):
        raise ValueError("eigensystem edge summary does not qualify the top-8 window")
    for index in range(nk):
        npz_path = root / "points" / f"k_{index:04d}.npz"
        json_path = root / "points" / f"k_{index:04d}.json"
        receipt = json.loads(json_path.read_text())
        if (
            receipt.get("run_id") != run_id
            or receipt.get("manifest_sha256") != manifest_sha256
            or not receipt.get("qualification", {}).get("passed")
            or receipt.get("npz_sha256") != _sha256_file(npz_path)
        ):
            raise ValueError(f"unqualified eigensystem point {index}")
        expected_k = np.asarray(
            [index // mesh / mesh, index % mesh / mesh, 0.0], dtype=np.float64
        )
        if not np.array_equal(
            np.asarray(receipt.get("k_fractional"), dtype=np.float64), expected_k
        ):
            raise ValueError(f"point receipt k coordinate mismatch at {index}")
        with np.load(npz_path, allow_pickle=False) as point:
            if str(point["run_id"]) != run_id or int(point["k_index"]) != index:
                raise ValueError(f"eigensystem point identity mismatch at {index}")
            if not np.array_equal(
                np.asarray(point["k_fractional"], dtype=np.float64), expected_k
            ):
                raise ValueError(f"point payload k coordinate mismatch at {index}")
            energies = np.asarray(point["energies_ev"], dtype=np.float64)
        occupied = energies[energies < float(config.occupied_below_ev)]
        if occupied.size < active_rank:
            raise ValueError(f"point {index} has too few occupied edge states")
        active_energies[:, index] = occupied[-active_rank:]
    h0 = np.zeros((active_rank, active_rank, nk), dtype=np.complex128)
    diagonal_indices = np.arange(active_rank)
    h0[diagonal_indices, diagonal_indices, :] = active_energies
    initial_density = ptse2_density_from_fixed_filling(
        h0,
        config.filling_nu,
        degeneracy_tolerance_ev=config.occupation_degeneracy_tolerance_ev,
        degenerate_occupation_policy=config.degenerate_occupation_policy,
    ).density
    screening = PtSe2DoubleGateScreening(
        epsilon_top=float(config.dielectric_constant),
        epsilon_bottom=float(config.dielectric_constant),
        d_top_nm=float(config.gate_distance_from_plane_nm),
        d_bottom_nm=float(config.gate_distance_from_plane_nm),
    )
    overlap_cache = build_ptse2_hf_overlap_cache(
        cache_path,
        cache_summary_path,
        source,
        screening,
        require_hf_authorized=bool(config.require_hf_authorized),
        expected_eigensystem_manifest_sha256=manifest_sha256,
    )
    one_overlap = next(iter(overlap_cache.blocks.overlaps.values()))
    if overlap_cache.run_id != run_id:
        raise ValueError("eigensystem and form-factor cache run identities differ")
    if overlap_cache.active_rank != active_rank:
        raise ValueError("eigensystem and form-factor cache active ranks differ")
    if one_overlap.shape != (active_rank, nk, active_rank, nk):
        raise ValueError("active eigensystem and form-factor cache shapes differ")
    area_nm2 = float(
        np.linalg.norm(
            np.cross(source.lattice_angstrom[0], source.lattice_angstrom[1])
        )
        / 100.0
    )
    k_fractional = np.asarray(
        [(i / mesh, j / mesh, 0.0) for i in range(mesh) for j in range(mesh)],
        dtype=np.float64,
    )
    return PtSe2ProjectedHFData(
        config=config,
        source=source,
        k_fractional=k_fractional,
        active_energies_ev=active_energies,
        h0=h0,
        initial_density=initial_density,
        overlap_cache=overlap_cache,
        eigensystem_root=str(root),
        eigensystem_manifest_sha256=manifest_sha256,
        edge_summary_sha256=_sha256_file(edge_summary_path),
        area_nm2=area_nm2,
    )


def build_ptse2_hf_state(data: PtSe2ProjectedHFData) -> PtSe2HFState:
    return PtSe2HFState(
        h0=np.array(data.h0, copy=True),
        density=np.array(data.initial_density, copy=True),
        hamiltonian=np.array(data.h0, copy=True),
        energies=np.array(data.active_energies_ev, copy=True),
        precision=float(data.config.precision),
        v0=1.0 / float(data.area_nm2),
    )


def run_ptse2_projected_hf(
    data: PtSe2ProjectedHFData,
    *,
    initial_density: np.ndarray | None = None,
    step_callback: Callable[[PtSe2HFState, HartreeFockStepResult], None] | None = None,
    final_state_callback: Callable[[PtSe2HFState, DensityUpdateResult], None] | None = None,
) -> HartreeFockRun:
    state = build_ptse2_hf_state(data)
    provided_density = None
    if initial_density is not None:
        provided_density = np.asarray(initial_density, dtype=np.complex128).copy()
        if provided_density.shape != state.density.shape:
            raise ValueError("provided PtSe2 initial density has the wrong shape")
        identity = np.eye(data.config.active_rank, dtype=np.complex128)
        measured_filling = 0.0
        for k_index in range(state.nk):
            block = provided_density[:, :, k_index]
            if np.linalg.norm(block - block.conj().T) > 1.0e-10:
                raise ValueError("provided PtSe2 initial density is not Hermitian")
            projector_eigenvalues = np.linalg.eigvalsh(block + identity)
            if (
                projector_eigenvalues[0] < -1.0e-10
                or projector_eigenvalues[-1] > 1.0 + 1.0e-10
            ):
                raise ValueError("provided PtSe2 initial projector is outside [0, 1]")
            measured_filling += float(np.trace(block).real)
        measured_filling /= state.nk
        if abs(measured_filling - data.config.filling_nu) > 1.0e-10:
            raise ValueError("provided PtSe2 initial density has the wrong filling")

    def initializer(target: PtSe2HFState, *, init_mode: str, seed: int) -> None:
        if provided_density is not None:
            target.density[:, :, :] = provided_density
        elif init_mode == "bare":
            target.density[:, :, :] = data.initial_density
        elif init_mode == "random_coherent":
            target.density[:, :, :] = _randomly_rotate_initial_density(
                data.initial_density,
                seed=seed,
                strength=float(data.config.random_rotation_strength),
            )
        else:  # guarded by PtSe2HFConfig, retained fail-closed at the callback boundary
            raise ValueError(f"unsupported PtSe2 init_mode={init_mode!r}")
        _hermitize_blocks(target.density)

    def density_builder(hamiltonian: np.ndarray) -> DensityUpdateResult:
        return ptse2_density_from_fixed_filling(
            hamiltonian,
            data.config.filling_nu,
            degeneracy_tolerance_ev=data.config.occupation_degeneracy_tolerance_ev,
            degenerate_occupation_policy=data.config.degenerate_occupation_policy,
        )

    return run_projected_hartree_fock(
        state,
        initializer=initializer,
        density_builder=density_builder,
        overlap_blocks=data.overlap_cache.blocks,
        init_mode=data.config.init_mode,
        seed=int(data.config.seed),
        v0=state.v0,
        oda_parameterizer=(
            "default" if data.config.oda_mode == "default" else None
        ),
        hamiltonian_postprocessor=_hermitize_blocks,
        density_postprocessor=_hermitize_blocks,
        step_callback=step_callback,
        final_state_callback=final_state_callback,
        convergence_rule=data.config.convergence_rule,
        max_iter=int(data.config.max_iter),
        use_numba=bool(data.config.use_numba),
    )
