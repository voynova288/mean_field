from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from .lattice import TDBGLattice, strain_twist_matrix
from .params import TDBGParameters, VALID_STACKINGS, VALID_VALLEYS


MOIRE_CHANNELS = (0, 1, 2)


@dataclass(frozen=True)
class MoireCouplingEntry:
    source_index: int
    target_index: int
    channel: int


def _validate_valley(valley: int) -> int:
    valley = int(valley)
    if valley not in VALID_VALLEYS:
        raise ValueError(f"Expected valley in {VALID_VALLEYS}, got {valley}")
    return valley


def _validate_stacking(stacking: str) -> str:
    if stacking not in VALID_STACKINGS:
        raise ValueError(f"Expected stacking in {VALID_STACKINGS}, got {stacking}")
    return stacking


def _dirac_non_dimer(k_plus: complex, k_minus: complex, onsite: float, params: TDBGParameters) -> np.ndarray:
    return np.asarray(
        [
            [onsite, -params.vf * k_minus],
            [-params.vf * k_plus, params.delta_prime + onsite],
        ],
        dtype=np.complex128,
    )


def _dirac_dimer(k_plus: complex, k_minus: complex, onsite: float, params: TDBGParameters) -> np.ndarray:
    return np.asarray(
        [
            [params.delta_prime + onsite, -params.vf * k_minus],
            [-params.vf * k_plus, onsite],
        ],
        dtype=np.complex128,
    )


def blg_interlayer(k_plus: complex, k_minus: complex, params: TDBGParameters) -> np.ndarray:
    return np.asarray(
        [
            [params.v4 * k_plus, params.gamma1],
            [params.v3 * k_minus, params.v4 * k_plus],
        ],
        dtype=np.complex128,
    )


def build_bilayer_block(
    k_plus: complex,
    k_minus: complex,
    params: TDBGParameters,
    *,
    upper_layer_potential: float,
    lower_layer_potential: float,
    stacking_order: str = "AB",
) -> np.ndarray:
    if stacking_order == "AB":
        upper = _dirac_non_dimer(k_plus, k_minus, upper_layer_potential, params)
        lower = _dirac_dimer(k_plus, k_minus, lower_layer_potential, params)
        coupling_upper_lower = blg_interlayer(k_plus, k_minus, params).conjugate().T
        coupling_lower_upper = blg_interlayer(k_plus, k_minus, params)
    elif stacking_order == "BA":
        upper = _dirac_dimer(k_plus, k_minus, upper_layer_potential, params)
        lower = _dirac_non_dimer(k_plus, k_minus, lower_layer_potential, params)
        coupling_upper_lower = blg_interlayer(k_plus, k_minus, params)
        coupling_lower_upper = blg_interlayer(k_plus, k_minus, params).conjugate().T
    else:
        raise ValueError(f"Unsupported local bilayer stacking order: {stacking_order}")

    block = np.zeros((4, 4), dtype=np.complex128)
    block[0:2, 0:2] = upper
    block[2:4, 2:4] = lower
    block[0:2, 2:4] = coupling_upper_lower
    block[2:4, 0:2] = coupling_lower_upper
    return block


def moire_coupling_matrix(channel: int, params: TDBGParameters, valley: int) -> np.ndarray:
    valley = _validate_valley(valley)
    if int(channel) not in MOIRE_CHANNELS:
        raise ValueError(f"Unsupported moire channel: {channel}")

    omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))
    if int(channel) == 0:
        return np.asarray([[params.u, params.u_prime], [params.u_prime, params.u]], dtype=np.complex128)
    if int(channel) == 1:
        return np.asarray(
            [
                [params.u, params.u_prime * omega ** (-valley)],
                [params.u_prime * omega ** (valley), params.u],
            ],
            dtype=np.complex128,
        )
    return np.asarray(
        [
            [params.u, params.u_prime * omega ** (valley)],
            [params.u_prime * omega ** (-valley), params.u],
        ],
        dtype=np.complex128,
    )


def _site_frame_momentum(
    k_tilde: complex,
    site_index: int,
    lattice: TDBGLattice,
    *,
    valley: int,
) -> tuple[complex, complex]:
    valley = _validate_valley(valley)
    sector = int(round(float(lattice.q_sites[site_index, 2])))
    twist_sign = -1.0 if sector == 0 else 1.0
    deformation = strain_twist_matrix(
        twist_sign * valley * lattice.theta_rad / 2.0,
        lattice.phi_rad,
        twist_sign * valley * lattice.epsilon / 2.0,
        lattice.poisson_ratio,
    )
    strain_tensor = 0.5 * (deformation + deformation.T)
    gauge_shift = valley * lattice.gauge_connection_nm_inv * np.asarray(
        [strain_tensor[0, 0] - strain_tensor[1, 1], -2.0 * strain_tensor[0, 1]],
        dtype=float,
    )
    momentum_lab = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    kj = (np.eye(2, dtype=float) + deformation) @ (momentum_lab + lattice.q_sites[site_index, :2] + gauge_shift)
    k_minus = complex(valley * kj[0], -kj[1])
    k_plus = complex(valley * kj[0], kj[1])
    return k_plus, k_minus


def build_site_block(
    k_tilde: complex,
    site_index: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    _validate_stacking(params.stacking)
    sector = int(round(float(lattice.q_sites[site_index, 2])))
    k_plus, k_minus = _site_frame_momentum(k_tilde, site_index, lattice, valley=valley)

    if sector == 0:
        return build_bilayer_block(
            k_plus,
            k_minus,
            params,
            upper_layer_potential=1.5 * params.Delta,
            lower_layer_potential=0.5 * params.Delta,
            stacking_order="AB",
        )

    lower_bilayer_stacking = "AB" if params.stacking == "AB-AB" else "BA"
    return build_bilayer_block(
        k_plus,
        k_minus,
        params,
        upper_layer_potential=-0.5 * params.Delta,
        lower_layer_potential=-1.5 * params.Delta,
        stacking_order=lower_bilayer_stacking,
    )


def build_coupling_table(lattice: TDBGLattice) -> tuple[MoireCouplingEntry, ...]:
    entries: list[MoireCouplingEntry] = []
    for source_index, neighbors in enumerate(lattice.q_neighbors):
        source_sector = int(round(float(lattice.q_sites[source_index, 2])))
        if source_sector != 1:
            continue
        for target_index, channel in neighbors:
            target_sector = int(round(float(lattice.q_sites[target_index, 2])))
            if target_sector != 0:
                continue
            entries.append(
                MoireCouplingEntry(
                    source_index=int(source_index),
                    target_index=int(target_index),
                    channel=int(channel),
                )
            )
    return tuple(entries)


def build_hamiltonian(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    dim = int(lattice.matrix_dim)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    for site_index in range(lattice.n_q):
        sl = slice(4 * site_index, 4 * (site_index + 1))
        hamiltonian[sl, sl] = build_site_block(k_tilde, site_index, lattice, params, valley=valley)

    for entry in build_coupling_table(lattice):
        source_slice = slice(4 * entry.source_index, 4 * entry.source_index + 2)
        target_slice = slice(4 * entry.target_index + 2, 4 * entry.target_index + 4)
        coupling = moire_coupling_matrix(entry.channel, params, valley)
        hamiltonian[target_slice, source_slice] += coupling
        hamiltonian[source_slice, target_slice] += coupling.conjugate().T

    return hamiltonian


def diagonalize_hamiltonian(
    k_tilde: complex,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    n_bands: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    hamiltonian = build_hamiltonian(k_tilde, lattice, params, valley=valley)
    if n_bands is None or int(n_bands) >= hamiltonian.shape[0]:
        evals, evecs = eigh(hamiltonian)
    else:
        evals, evecs = eigh(hamiltonian, subset_by_index=[0, int(n_bands) - 1])
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)
