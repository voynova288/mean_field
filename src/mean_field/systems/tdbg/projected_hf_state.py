from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import numpy as np

from ...core.hf import (
    DensityConvention,
    DensityUpdateResult,
    conventional_projector_to_stored,
    density_to_stored_delta,
    stored_projector_to_conventional,
)
from .projected_hf_config import SPIN_LABELS, TDBGProjectedHFConfig, VALLEY_LABELS

if TYPE_CHECKING:
    from ...core.hf import HartreeFockRun
    from .model import TDBGModel
    from .params import TDBGParameters

@dataclass(frozen=True)
class TDBGStateLabel:
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
class TDBGProjectedHFData:
    model: TDBGModel
    config: TDBGProjectedHFConfig
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    band_indices: tuple[int, ...]
    labels: tuple[TDBGStateLabel, ...]
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, nk, n_q, 4)
    reference_density: np.ndarray
    n_occupied_per_k: int
    lower_band_count: int
    moire_area_nm2: float
    shifts: tuple[tuple[int, int], ...]
    shift_gvecs: np.ndarray
    shift_srcmaps: tuple[np.ndarray, ...]
    valley_params: Mapping[int, TDBGParameters] | None = None

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])

    @property
    def n_band(self) -> int:
        return int(len(self.band_indices))


@dataclass
class TDBGProjectedHFState:
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
class TDBGProjectedHFTargetData:
    kvec: np.ndarray
    h0: np.ndarray
    wavefunctions: np.ndarray  # (nt, n_target, n_q, 4)

    @property
    def nt(self) -> int:
        return int(self.h0.shape[0])

    @property
    def nk(self) -> int:
        return int(self.h0.shape[2])


@dataclass(frozen=True)
class TDBGProjectedHFResult:
    run: HartreeFockRun
    data: TDBGProjectedHFData
    init_mode: str
    seed: int
    order_parameters: dict[str, object]
    energy_components: dict[str, float]
    hamiltonian_components: Mapping[str, np.ndarray] | None = None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "init_mode": self.init_mode,
            "seed": int(self.seed),
            "converged": bool(self.run.converged),
            "exit_reason": self.run.exit_reason,
            "iterations": int(self.run.iterations),
            "final_error": float(self.run.state.diagnostics.get("final_raw_norm", np.nan)),
            "hf_energy_ev": float(self.run.state.diagnostics.get("hf_energy", np.nan)),
            "order_parameters": self.order_parameters,
            "energy_components_ev": self.energy_components,
        }

def _conventional_projector_to_stored(projector: np.ndarray) -> np.ndarray:
    return conventional_projector_to_stored(projector)


def _stored_to_conventional(stored: np.ndarray) -> np.ndarray:
    return stored_projector_to_conventional(stored)


def _first_conduction_indices(data: TDBGProjectedHFData) -> list[int]:
    if data.n_band == 1:
        position = 0
    else:
        position = data.lower_band_count
    return [label.index for label in data.labels if label.band_position == position]

def _active_filling_indices(data: TDBGProjectedHFData) -> list[int]:
    filling = int(data.config.filling)
    if data.n_band == 1:
        position = 0
    elif filling >= 0:
        position = data.lower_band_count
    else:
        if data.lower_band_count <= 0:
            raise ValueError("Negative TDBG filling requires at least one valence band in the projected window")
        position = data.lower_band_count - 1
    return [label.index for label in data.labels if label.band_position == position]

def _reference_projector(data: TDBGProjectedHFData) -> np.ndarray:
    projector = np.zeros((data.nt, data.nt), dtype=np.complex128)
    for label in data.labels:
        if label.band_position < data.lower_band_count:
            projector[label.index, label.index] = 1.0
    return projector

def initialize_tdbg_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Return an absolute occupied projector in the core stored convention.

    The initializer supports positive and negative fillings relative to the
    charge-neutral reference. Positive fillings add projectors in the first
    conduction band; negative fillings remove hole projectors from the highest
    valence band. The unrestricted density builder then refills the lowest HF
    eigenstates at the configured occupation count.
    """

    mode = init_mode.strip().lower().replace("-", "_")
    nk = data.nk
    nt = data.nt
    filling = int(data.config.filling)
    count = abs(filling)
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    active_labels = [data.labels[idx] for idx in _active_filling_indices(data)]
    rng = np.random.default_rng(seed)

    def apply_active_projectors(projector: np.ndarray, projectors: list[np.ndarray]) -> np.ndarray:
        if len(projectors) != count:
            raise ValueError(f"init_mode={init_mode!r} produced {len(projectors)} projectors for filling {filling}")
        for active_projector in projectors:
            if filling >= 0:
                projector += active_projector
            else:
                projector -= active_projector
        return projector

    def basis_projectors(indices: list[int]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for idx in indices:
            vec = np.zeros(nt, dtype=np.complex128)
            vec[int(idx)] = 1.0
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def coherent_projectors(phase: complex, *, k_weight: float = 0.5) -> list[np.ndarray]:
        k_weight = float(k_weight)
        if not 0.0 < k_weight < 1.0:
            raise ValueError(f"IVC valley weight must be in (0, 1), got {k_weight}")
        kp_weight = 1.0 - k_weight
        out: list[np.ndarray] = []
        for spin in SPIN_LABELS:
            states = [label.index for label in active_labels if label.spin == spin]
            if len(states) != 2:
                raise ValueError("IVC initializer requires exactly two valley states per spin in the active filling band")
            vec = np.zeros(nt, dtype=np.complex128)
            vec[states[0]] = math.sqrt(k_weight)
            vec[states[1]] = complex(phase) * math.sqrt(kp_weight)
            out.append(np.outer(vec, vec.conjugate()))
        return out

    def parse_ivc_weight_token(token: str) -> float:
        if not token:
            raise ValueError(f"Biased IVC initializer {init_mode!r} must include a valley weight, e.g. ivc_k85")
        value = float(token.replace("p", "."))
        if value > 1.0:
            value /= 100.0
        if not 0.0 < value < 1.0:
            raise ValueError(f"Biased IVC valley weight must be in (0, 1), got {value} from {init_mode!r}")
        return value

    def biased_coherent_projectors_from_mode() -> list[np.ndarray]:
        phase: complex = 1.0j if mode.endswith("_odd") else 1.0
        base = mode[:-4] if mode.endswith("_odd") else mode
        if base.startswith("ivc_kprime"):
            kp_weight = parse_ivc_weight_token(base[len("ivc_kprime") :])
            return coherent_projectors(phase, k_weight=1.0 - kp_weight)
        if base.startswith("ivc_k"):
            k_weight = parse_ivc_weight_token(base[len("ivc_k") :])
            return coherent_projectors(phase, k_weight=k_weight)
        raise ValueError(f"Unsupported biased IVC initializer {init_mode!r}")

    def random_projectors() -> list[np.ndarray]:
        states = [label.index for label in active_labels]
        if count > len(states):
            raise ValueError(f"Cannot choose {count} active projectors from {len(states)} states")
        z = rng.standard_normal((len(states), len(states))) + 1j * rng.standard_normal((len(states), len(states)))
        herm = z + z.conjugate().T
        _, vecs = np.linalg.eigh(herm)
        out: list[np.ndarray] = []
        for col in range(count):
            vec = np.zeros(nt, dtype=np.complex128)
            vec[np.asarray(states, dtype=int)] = vecs[:, col]
            out.append(np.outer(vec, vec.conjugate()))
        return out

    for ik in range(nk):
        projector = _reference_projector(data)
        if filling == 0:
            pass
        elif mode in {"sp", "sp_up"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "up"]))
        elif mode == "sp_down":
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if label.spin == "down"]))
        elif mode in {"vp", "vp_k"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == 1]))
        elif mode in {"vp_kprime", "vp_kp"}:
            projector = apply_active_projectors(projector, basis_projectors([label.index for label in active_labels if int(label.valley) == -1]))
        elif mode in {"ivc", "ivc_even"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0))
        elif mode in {"ivc_odd", "kivc"}:
            projector = apply_active_projectors(projector, coherent_projectors(1.0j))
        elif mode.startswith("ivc_k"):
            projector = apply_active_projectors(projector, biased_coherent_projectors_from_mode())
        elif mode in {"random", "random_flavor"}:
            projector = apply_active_projectors(projector, random_projectors())
        elif mode in {"bm", "noninteracting"}:
            projector = np.zeros((nt, nt), dtype=np.complex128)
            evals = np.real(np.diag(data.h0[:, :, ik]))
            for idx in np.argsort(evals, kind="stable")[: data.n_occupied_per_k]:
                projector[int(idx), int(idx)] = 1.0
        else:
            raise ValueError(
                f"Unsupported TDBG projected-HF init_mode={init_mode!r}. "
                "Use sp, sp_down, vp_k, vp_kprime, ivc_even, ivc_odd, ivc_k85, ivc_kprime85, random, or bm."
            )
        density[:, :, ik] = _conventional_projector_to_stored(projector)
    return density

def initialize_tdbg_nu2_density(data: TDBGProjectedHFData, *, init_mode: str, seed: int = 1) -> np.ndarray:
    """Backward-compatible alias for the generic TDBG filling initializer."""

    return initialize_tdbg_density(data, init_mode=init_mode, seed=seed)

class TDBGProjectedHFInitializer:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, state: TDBGProjectedHFState, *, init_mode: str, seed: int) -> None:
        state.density[:, :, :] = initialize_tdbg_density(self.data, init_mode=init_mode, seed=seed)
        state.diagnostics.update(_numeric_order_parameters(self.data, state.density))


class TDBGProjectedHFDensityBuilder:
    def __init__(self, data: TDBGProjectedHFData):
        self.data = data

    def __call__(self, hamiltonian: np.ndarray) -> DensityUpdateResult:
        density, energies, mu, occ_mask = tdbg_density_from_hamiltonian(hamiltonian, self.data.n_occupied_per_k)
        observables = {"occupation_mask": occ_mask}
        observables.update(_numeric_order_parameters(self.data, density))
        return DensityUpdateResult(density=density, energies=energies, mu=mu, observables=observables)


def tdbg_density_from_hamiltonian(hamiltonian: np.ndarray, n_occupied_per_k: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
            density[:, :, ik] = _conventional_projector_to_stored(projector)
            occ_mask[:nocc, ik] = True
    if nocc <= 0 or nocc >= nt:
        mu = float(np.mean(energies))
    else:
        mu = 0.5 * (float(np.max(energies[:nocc, :])) + float(np.min(energies[nocc:, :])))
    return density, energies, float(mu), occ_mask

def _reference_subtracted_tdbg_density(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    return density_to_stored_delta(
        density,
        DensityConvention.PROJECTOR,
        reference=data.reference_density,
        reference_policy="require",
    )


def _hartree_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.hartree_reference == "none":
        return density
    if settings.hartree_reference == "charge_neutral":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Hartree reference policy: {settings.hartree_reference!r}")


def _fock_density_for_policy(data: TDBGProjectedHFData, density: np.ndarray) -> np.ndarray:
    settings = data.config.interaction
    if settings.fock_density == "absolute":
        return density
    if settings.fock_density == "reference_subtracted":
        return _reference_subtracted_tdbg_density(data, density)
    raise ValueError(f"Unsupported TDBG Fock density policy: {settings.fock_density!r}")

def _numeric_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, float]:
    occ = np.zeros(data.nt, dtype=float)
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        occ += np.real(np.diag(projector))
    occ /= float(data.nk)
    spin_pol = 0.0
    valley_pol = 0.0
    active_spin_pol = 0.0
    active_valley_pol = 0.0
    active_indices = set(_active_filling_indices(data))
    for label in data.labels:
        value = float(occ[label.index])
        spin_sign = 1.0 if label.spin == "up" else -1.0
        valley_sign = 1.0 if int(label.valley) == 1 else -1.0
        spin_pol += spin_sign * value
        valley_pol += valley_sign * value
        if label.index in active_indices:
            active_spin_pol += spin_sign * value
            active_valley_pol += valley_sign * value
    ivc = 0.0
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        for spin in SPIN_LABELS:
            k_idx = [label.index for label in data.labels if label.spin == spin and int(label.valley) == 1 and label.index in active_indices]
            kp_idx = [label.index for label in data.labels if label.spin == spin and int(label.valley) == -1 and label.index in active_indices]
            if k_idx and kp_idx:
                ivc += float(np.linalg.norm(projector[np.ix_(k_idx, kp_idx)]))
    ivc /= float(data.nk)
    return {
        "spin_polarization": float(spin_pol),
        "valley_polarization": float(valley_pol),
        "active_spin_polarization": float(active_spin_pol),
        "active_valley_polarization": float(active_valley_pol),
        # Backward-compatible names for the original nu=+2 conduction-band workflow.
        "cb_spin_polarization": float(active_spin_pol),
        "cb_valley_polarization": float(active_valley_pol),
        "ivc_amplitude": float(ivc),
    }

def tdbg_order_parameters(data: TDBGProjectedHFData, density: np.ndarray) -> dict[str, object]:
    numeric = _numeric_order_parameters(data, density)
    occ_by_label: list[dict[str, object]] = []
    occ = np.zeros(data.nt, dtype=float)
    for ik in range(data.nk):
        projector = _stored_to_conventional(density[:, :, ik])
        occ += np.real(np.diag(projector))
    occ /= float(data.nk)
    for label in data.labels:
        item = label.to_dict()
        item["occupation"] = float(occ[label.index])
        occ_by_label.append(item)
    classification = "mixed"
    if abs(numeric["ivc_amplitude"]) > 0.15:
        classification = "IVC_or_valley_coherent"
    elif abs(numeric["cb_valley_polarization"]) > 1.2:
        classification = "VP_K" if numeric["cb_valley_polarization"] > 0 else "VP_Kprime"
    elif abs(numeric["cb_spin_polarization"]) > 1.2:
        classification = "SP_up" if numeric["cb_spin_polarization"] > 0 else "SP_down"
    return {**numeric, "classification": classification, "occupations": occ_by_label}

__all__ = [
    "TDBGProjectedHFData",
    "TDBGProjectedHFInitializer",
    "TDBGProjectedHFDensityBuilder",
    "TDBGProjectedHFResult",
    "TDBGProjectedHFState",
    "TDBGProjectedHFTargetData",
    "TDBGStateLabel",
    "_active_filling_indices",
    "_conventional_projector_to_stored",
    "_first_conduction_indices",
    "_fock_density_for_policy",
    "_hartree_density_for_policy",
    "_numeric_order_parameters",
    "_reference_projector",
    "_reference_subtracted_tdbg_density",
    "_stored_to_conventional",
    "initialize_tdbg_density",
    "initialize_tdbg_nu2_density",
    "tdbg_density_from_hamiltonian",
    "tdbg_order_parameters",
]
