from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .params import GRAPHENE_LATTICE_CONSTANT_NM, TMBGParameters, hopping_to_velocity


VALID_VALLEYS = (-1, 1)
MOIRE_CHANNELS = ("0", "+", "-")


@dataclass(frozen=True)
class CrossCheckCouplingEntry:
    channel: str
    middle_index: int
    top_index: int


def _validate_valley(valley: int) -> int:
    valley = int(valley)
    if valley not in VALID_VALLEYS:
        raise ValueError(f"Expected valley in {VALID_VALLEYS}, got {valley}")
    return valley


def _complex_key(value: complex, *, digits: int = 12) -> tuple[float, float]:
    return (round(float(value.real), digits), round(float(value.imag), digits))


def _resolve_params(params: TMBGParameters | Mapping[str, float]) -> dict[str, Any]:
    if isinstance(params, TMBGParameters):
        return {
            "graphene_lattice_constant_nm": float(params.graphene_lattice_constant_nm),
            "t0": float(params.t0),
            "t1": float(params.t1),
            "t3": float(params.t3),
            "t4": float(params.t4),
            "delta": float(params.delta),
            "omega": float(params.omega),
            "omega_prime": float(params.omega_prime),
            "interlayer_potential": float(params.interlayer_potential),
            "staggered_potential": float(params.staggered_potential),
            "vf": float(params.vf),
            "v3": float(params.v3),
            "v4": float(params.v4),
            "blg_stacking": str(params.blg_stacking),
            "bernal_convention": str(params.bernal_convention),
        }

    resolved: dict[str, Any] = dict(params)
    for key in (
        "graphene_lattice_constant_nm",
        "t0",
        "t1",
        "t3",
        "t4",
        "delta",
        "omega",
        "omega_prime",
        "interlayer_potential",
        "staggered_potential",
    ):
        if key in resolved:
            resolved[key] = float(resolved[key])
    a_nm = float(resolved.get("graphene_lattice_constant_nm", GRAPHENE_LATTICE_CONSTANT_NM))
    resolved.setdefault("vf", hopping_to_velocity(float(resolved["t0"]), a_nm))
    resolved.setdefault("v3", hopping_to_velocity(float(resolved["t3"]), a_nm))
    resolved.setdefault("v4", hopping_to_velocity(float(resolved["t4"]), a_nm))
    resolved.setdefault("blg_stacking", "AB")
    resolved.setdefault("bernal_convention", "park")
    return resolved


def _rotated_complex_momentum(kvec: complex, phi: float) -> complex:
    return complex(kvec) * complex(math.cos(phi), -math.sin(phi))


def _valley_pi(kvec: complex, phi: float, valley: int) -> tuple[complex, complex]:
    q = _rotated_complex_momentum(kvec, phi)
    if _validate_valley(valley) == 1:
        return complex(q), complex(q.conjugate())
    return complex(-q.conjugate()), complex(-q)


def generate_g_vectors(
    g_m1: complex,
    g_m2: complex,
    *,
    n_shells: int,
    g_cutoff: float | None = None,
) -> np.ndarray:
    if int(n_shells) < 0:
        raise ValueError(f"Expected non-negative n_shells, got {n_shells}")

    resolved_cutoff = float((int(n_shells) + 0.5) * abs(g_m1) if g_cutoff is None else g_cutoff)
    index_limit = int(n_shells) + 2
    entries: list[tuple[float, int, int, complex]] = []
    for n1 in range(-index_limit, index_limit + 1):
        for n2 in range(-index_limit, index_limit + 1):
            gvec = complex(n1 * g_m1 + n2 * g_m2)
            if abs(gvec) <= resolved_cutoff + 1.0e-12:
                entries.append((abs(gvec), n1, n2, gvec))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.asarray([gvec for _, _, _, gvec in entries], dtype=np.complex128)


def build_coupling_table(
    g_vectors: np.ndarray,
    *,
    q0: complex,
    q_plus: complex,
    q_minus: complex,
    valley: int = 1,
) -> tuple[CrossCheckCouplingEntry, ...]:
    valley = _validate_valley(valley)
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    q_vectors = {"0": complex(q0), "+": complex(q_plus), "-": complex(q_minus)}
    mapping = {_complex_key(complex(gvec)): idx for idx, gvec in enumerate(g_vectors)}

    entries: list[CrossCheckCouplingEntry] = []
    for middle_index, g_middle in enumerate(g_vectors):
        for channel in MOIRE_CHANNELS:
            shift = complex(valley * (q_vectors[channel] - q_vectors["0"]))
            top_index = mapping.get(_complex_key(complex(g_middle + shift)))
            if top_index is None:
                continue
            entries.append(
                CrossCheckCouplingEntry(
                    channel=str(channel),
                    middle_index=int(middle_index),
                    top_index=int(top_index),
                )
            )
    return tuple(entries)


def dirac_block(kvec: complex, *, phi: float, vf: float, valley: int) -> np.ndarray:
    pi, pi_dag = _valley_pi(kvec, phi, valley)
    return np.asarray([[0.0, vf * pi_dag], [vf * pi, 0.0]], dtype=np.complex128)


def blg_interlayer(
    kvec: complex,
    *,
    phi: float,
    t1: float,
    v3: float,
    v4: float,
    valley: int,
    blg_stacking: str = "AB",
    bernal_convention: str = "park",
) -> np.ndarray:
    pi, pi_dag = _valley_pi(kvec, phi, valley)
    if bernal_convention == "polshyn2020":
        return np.asarray(
            [
                [-v4 * pi, t1],
                [-v3 * pi_dag, -v4 * pi],
            ],
            dtype=np.complex128,
        )
    if blg_stacking == "AB":
        return np.asarray(
            [
                [-v4 * pi_dag, -v3 * pi],
                [t1, -v4 * pi_dag],
            ],
            dtype=np.complex128,
        )
    return np.asarray(
        [
            [-v4 * pi_dag, t1],
            [-v3 * pi, -v4 * pi_dag],
        ],
        dtype=np.complex128,
    )


def tmbg_diagonal_block(
    k_tilde: complex,
    gvec: complex,
    *,
    q0: complex,
    theta_rad: float,
    params: TMBGParameters | Mapping[str, float],
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    resolved = _resolve_params(params)
    k_bottom = complex(k_tilde + gvec)
    k_top = complex(k_tilde + gvec + valley * q0)
    phi_bottom = -theta_rad / 2.0
    phi_top = theta_rad / 2.0

    h_bottom = dirac_block(k_bottom, phi=phi_bottom, vf=resolved["vf"], valley=valley)
    h_middle = dirac_block(k_bottom, phi=phi_bottom, vf=resolved["vf"], valley=valley)
    h_top = dirac_block(k_top, phi=phi_top, vf=resolved["vf"], valley=valley)
    t_blg = blg_interlayer(
        k_bottom,
        phi=phi_bottom,
        t1=resolved["t1"],
        v3=resolved["v3"],
        v4=resolved["v4"],
        valley=valley,
        blg_stacking=str(resolved["blg_stacking"]),
        bernal_convention=str(resolved["bernal_convention"]),
    )

    block = np.zeros((6, 6), dtype=np.complex128)
    block[0:2, 0:2] = h_bottom
    block[2:4, 2:4] = h_middle
    block[4:6, 4:6] = h_top
    block[0:2, 2:4] = t_blg
    block[2:4, 0:2] = t_blg.conjugate().T

    if resolved["bernal_convention"] == "polshyn2020":
        block[0, 0] += resolved["delta"]
        block[3, 3] -= resolved["delta"]
    elif resolved["blg_stacking"] == "AB":
        block[1, 1] += resolved["delta"]
        block[2, 2] += resolved["delta"]
    else:
        block[0, 0] += resolved["delta"]
        block[3, 3] += resolved["delta"]
    block[0:2, 0:2] += -resolved["interlayer_potential"] * np.eye(2, dtype=np.complex128)
    block[4:6, 4:6] += resolved["interlayer_potential"] * np.eye(2, dtype=np.complex128)
    block[4:6, 4:6] += resolved["staggered_potential"] * np.asarray(
        [[1.0, 0.0], [0.0, -1.0]],
        dtype=np.complex128,
    )
    return block


def moire_coupling_matrix(channel: str, *, omega: float, omega_prime: float, valley: int = 1) -> np.ndarray:
    valley = _validate_valley(valley)
    phase = complex(math.cos(2.0 * math.pi * valley / 3.0), math.sin(2.0 * math.pi * valley / 3.0))
    if channel == "0":
        return np.asarray([[omega_prime, omega], [omega, omega_prime]], dtype=np.complex128)
    if channel == "+":
        return np.asarray(
            [
                [omega_prime, omega * phase.conjugate()],
                [omega * phase, omega_prime],
            ],
            dtype=np.complex128,
        )
    if channel == "-":
        return np.asarray(
            [
                [omega_prime, omega * phase],
                [omega * phase.conjugate(), omega_prime],
            ],
            dtype=np.complex128,
        )
    raise ValueError(f"Unsupported moire coupling channel: {channel}")


def build_hamiltonian_tmbg(
    k_tilde: complex,
    g_vectors: np.ndarray,
    coupling_table: tuple[CrossCheckCouplingEntry, ...],
    q0: complex,
    theta_rad: float,
    params: TMBGParameters | Mapping[str, float],
    *,
    valley: int = 1,
) -> np.ndarray:
    valley = _validate_valley(valley)
    resolved = _resolve_params(params)
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)

    dim = 6 * int(g_vectors.size)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)
    for ig, gvec in enumerate(g_vectors):
        sl = slice(6 * ig, 6 * (ig + 1))
        hamiltonian[sl, sl] = tmbg_diagonal_block(
            k_tilde,
            complex(gvec),
            q0=q0,
            theta_rad=theta_rad,
            params=resolved,
            valley=valley,
        )

    for entry in coupling_table:
        middle_slice = slice(6 * int(entry.middle_index) + 2, 6 * int(entry.middle_index) + 4)
        top_slice = slice(6 * int(entry.top_index) + 4, 6 * int(entry.top_index) + 6)
        coupling = moire_coupling_matrix(
            entry.channel,
            omega=resolved["omega"],
            omega_prime=resolved["omega_prime"],
            valley=valley,
        )
        hamiltonian[middle_slice, top_slice] += coupling
        hamiltonian[top_slice, middle_slice] += coupling.conjugate().T

    return hamiltonian
