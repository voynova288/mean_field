"""Mao-2025 hTG system adapter and convention helpers.

This module owns the Mao-specific hTG Hamiltonian decorations, parameters,
stacking phase conversion, and dH/dk validation helpers.  It is not a claim that
Mao 2025 figures have been reproduced; reproduction status remains documented
in the analysis workspace audit notes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from mean_field.systems.htg.hamiltonian import layer_rotation_angle
from mean_field.systems.htg.lattice import HTGLattice
from mean_field.systems.htg.model import HTGModel
from mean_field.systems.htg.params import GRAPHENE_LATTICE_CONSTANT_NM, HBAR_EV_S, HTGParams

CARBON_BOND_NM = GRAPHENE_LATTICE_CONSTANT_NM / math.sqrt(3.0)
MAO_HBAR_VF_K_EV = 9.905
MAO_W1_EV = 0.110
MAO_SUBLATTICE_MASS_EV = 0.030
MAO_REALISTIC_CORRUGATION = 0.8


def graphene_k_mag_nm_inv(a_nm: float = GRAPHENE_LATTICE_CONSTANT_NM) -> float:
    """Return |K| = 4*pi/(3a) in nm^{-1} for the graphene convention used here."""

    return 4.0 * math.pi / (3.0 * float(a_nm))


def mao_vf_ev_nm(a_nm: float = GRAPHENE_LATTICE_CONSTANT_NM) -> float:
    """Return hbar*v_F in eV*nm from Mao's hbar*v_F*|K| = 9.905 eV."""

    return MAO_HBAR_VF_K_EV / graphene_k_mag_nm_inv(a_nm)


def vf_ev_nm_to_m_per_s(vf_ev_nm: float) -> float:
    """Convert hbar*v_F from eV*nm to v_F in m/s."""

    return float(vf_ev_nm) / (HBAR_EV_S * 1.0e9)


def eta_mev_to_ev(eta_mev: float) -> float:
    return float(eta_mev) * 1.0e-3


@dataclass(frozen=True)
class MaoHTGConfig:
    """Parameters for Mao et al. hTTG shift-current benchmarks."""

    theta_deg: float = 1.95
    stacking: str = "ABA"
    corrugation_r: float = MAO_REALISTIC_CORRUGATION
    n_shells: int = 2
    mass_ev: float = MAO_SUBLATTICE_MASS_EV
    w1_ev: float = MAO_W1_EV
    hbar_vf_k_ev: float = MAO_HBAR_VF_K_EV
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    valley: int = 1
    zeta_rad: float | None = 0.0
    domain: str = "h"

    @property
    def vf_ev_nm(self) -> float:
        if self.hbar_vf_k_ev == MAO_HBAR_VF_K_EV and self.graphene_lattice_constant_nm == GRAPHENE_LATTICE_CONSTANT_NM:
            return mao_vf_ev_nm(self.graphene_lattice_constant_nm)
        k_mag = 4.0 * math.pi / (3.0 * float(self.graphene_lattice_constant_nm))
        return float(self.hbar_vf_k_ev) / k_mag


def stacking_phase_pair(stacking: str) -> tuple[float, float]:
    """Return (phi1, phi2) for the top-middle interface in Mao Eq. (13)."""

    key = str(stacking).upper()
    if key == "ABA":
        return 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0
    if key == "AAA":
        return 0.0, 0.0
    raise ValueError(f"Unsupported stacking {stacking!r}; expected 'ABA' or 'AAA'")


def phase_displacement(lattice: HTGLattice, phi1: float, phi2: float, *, valley: int = 1) -> complex:
    """Convert Mao relative channel phases to a layer displacement.

    The existing HTG Hamiltonian multiplies channel j by exp(i*valley*q_j.d).
    Mao's V_{phi1,phi2} fixes phases relative to channel 0, so we solve

        valley*(q_1-q_0).d = phi1,
        valley*(q_2-q_0).d = phi2.

    Any common channel phase is a layer-gauge choice and is not fixed here.
    """

    valley = int(valley)
    if valley not in (-1, 1):
        raise ValueError(f"valley must be +/-1, got {valley}")
    q0, q1, q2 = [complex(value) for value in lattice.q_vectors]
    rows = np.asarray(
        [
            [float((q1 - q0).real), float((q1 - q0).imag)],
            [float((q2 - q0).real), float((q2 - q0).imag)],
        ],
        dtype=float,
    )
    rhs = np.asarray([float(phi1) / valley, float(phi2) / valley], dtype=float)
    dx, dy = np.linalg.solve(rows, rhs)
    return complex(float(dx), float(dy))


def stacking_displacements(
    lattice: HTGLattice,
    stacking: str,
    *,
    valley: int = 1,
    domain: str = "h",
) -> tuple[complex, complex]:
    """Return (d_top, d_bot) implementing ABA/AAA phases of Mao Eq. (13).

    ``domain='hbar'`` flips the stacking-domain phases.  This is useful as a
    convention diagnostic because opposite helical domains rotate/sign-change
    the C3 shift-current tensor while leaving the coarse band structure nearly
    unchanged.
    """

    phi1, phi2 = stacking_phase_pair(stacking)
    if str(domain).lower() in {"hbar", "anti-h", "anti_h"}:
        phi1, phi2 = -phi1, -phi2
    elif str(domain).lower() != "h":
        raise ValueError(f"Unsupported stacking domain {domain!r}; expected 'h' or 'hbar'.")
    d_top = phase_displacement(lattice, phi1, phi2, valley=valley)
    d_bot = phase_displacement(lattice, -phi1, -phi2, valley=valley)
    return d_top, d_bot


def make_mao_model(config: MaoHTGConfig) -> HTGModel:
    """Build an HTGModel using Mao's v_F, w1, corrugation r, and cutoff."""

    params = HTGParams(
        graphene_lattice_constant_nm=float(config.graphene_lattice_constant_nm),
        fermi_velocity_m_per_s=vf_ev_nm_to_m_per_s(config.vf_ev_nm),
        w_ev=float(config.w1_ev),
        kappa=float(config.corrugation_r),
        zeta_rad=config.zeta_rad,
        model_name="mao2025_shift_current",
    )
    return HTGModel.from_config(float(config.theta_deg), n_shells=int(config.n_shells), params=params)


def _orbital_slice(g_index: int, layer: int) -> slice:
    start = 6 * int(g_index) + 2 * (int(layer) - 1)
    return slice(start, start + 2)


def add_sublattice_mass(hamiltonian: np.ndarray, lattice: HTGLattice, mass_ev: float) -> np.ndarray:
    """Add m sigma_z on every layer and G block."""

    out = np.array(hamiltonian, dtype=np.complex128, copy=True)
    mass = float(mass_ev)
    if mass == 0.0:
        return out
    for ig in range(lattice.n_g):
        for layer in (1, 2, 3):
            sl = _orbital_slice(ig, layer)
            out[sl.start, sl.start] += mass
            out[sl.start + 1, sl.start + 1] -= mass
    return out


def build_mao_hamiltonian(
    k_tilde: complex,
    model: HTGModel,
    config: MaoHTGConfig,
    *,
    d_top: complex | None = None,
    d_bot: complex | None = None,
) -> np.ndarray:
    """HTG Hamiltonian plus Mao's hBN mass term."""

    if d_top is None or d_bot is None:
        default_top, default_bot = stacking_displacements(
            model.lattice,
            config.stacking,
            valley=config.valley,
            domain=config.domain,
        )
        d_top = default_top if d_top is None else d_top
        d_bot = default_bot if d_bot is None else d_bot
    h0 = model.build_hamiltonian(complex(k_tilde), valley=int(config.valley), d_top=d_top, d_bot=d_bot)
    return add_sublattice_mass(h0, model.lattice, float(config.mass_ev))


def _dirac_dhdk_block(axis: int, angle_rad: float, vf_ev_nm: float, valley: int) -> np.ndarray:
    """Analytic derivative of the existing HTG Dirac block with respect to k_x/k_y."""

    rot = complex(math.cos(angle_rad), -math.sin(angle_rad))
    vf = float(vf_ev_nm)
    valley = int(valley)
    if valley == 1:
        if axis == 0:
            return np.asarray([[0.0, vf * rot.conjugate()], [vf * rot, 0.0]], dtype=np.complex128)
        if axis == 1:
            return np.asarray([[0.0, -1.0j * vf * rot.conjugate()], [1.0j * vf * rot, 0.0]], dtype=np.complex128)
    elif valley == -1:
        if axis == 0:
            return np.asarray([[0.0, -vf * rot], [-vf * rot.conjugate(), 0.0]], dtype=np.complex128)
        if axis == 1:
            return np.asarray([[0.0, -1.0j * vf * rot], [1.0j * vf * rot.conjugate(), 0.0]], dtype=np.complex128)
    else:
        raise ValueError(f"valley must be +/-1, got {valley}")
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def analytic_dhdk(model: HTGModel, config: MaoHTGConfig) -> tuple[np.ndarray, np.ndarray]:
    """Analytic dH/dk matrices in the current HTG code convention.

    Interlayer couplings and the sublattice mass are k independent.  Only the
    layer Dirac blocks contribute.  This must still be checked against finite
    differences before using the matrices in a production shift-current run.
    """

    dim = model.matrix_dim
    out = [np.zeros((dim, dim), dtype=np.complex128), np.zeros((dim, dim), dtype=np.complex128)]
    for ig in range(model.lattice.n_g):
        for layer in (1, 2, 3):
            angle = layer_rotation_angle(model.lattice, model.params, layer)
            sl = _orbital_slice(ig, layer)
            for axis in (0, 1):
                out[axis][sl, sl] = _dirac_dhdk_block(axis, angle, model.params.vf_ev_nm, int(config.valley))
    return out[0], out[1]


def finite_difference_dhdk(
    k_tilde: complex,
    model: HTGModel,
    config: MaoHTGConfig,
    *,
    step_nm_inv: float = 1.0e-6,
    d_top: complex | None = None,
    d_bot: complex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Central finite-difference dH/dk for convention validation."""

    step = float(step_nm_inv)
    if step <= 0.0:
        raise ValueError(f"step_nm_inv must be positive, got {step_nm_inv}")
    k = complex(k_tilde)
    hx_p = build_mao_hamiltonian(k + step, model, config, d_top=d_top, d_bot=d_bot)
    hx_m = build_mao_hamiltonian(k - step, model, config, d_top=d_top, d_bot=d_bot)
    hy_p = build_mao_hamiltonian(k + 1.0j * step, model, config, d_top=d_top, d_bot=d_bot)
    hy_m = build_mao_hamiltonian(k - 1.0j * step, model, config, d_top=d_top, d_bot=d_bot)
    return (hx_p - hx_m) / (2.0 * step), (hy_p - hy_m) / (2.0 * step)


@dataclass(frozen=True)
class DHdkValidationResult:
    max_abs_x_ev_nm: float
    max_abs_y_ev_nm: float
    max_abs_ev_nm: float
    finite_step_nm_inv: float
    k_tilde: complex


def validate_analytic_dhdk(
    k_tilde: complex,
    model: HTGModel,
    config: MaoHTGConfig,
    *,
    step_nm_inv: float = 1.0e-6,
    d_top: complex | None = None,
    d_bot: complex | None = None,
) -> DHdkValidationResult:
    analytic = analytic_dhdk(model, config)
    numeric = finite_difference_dhdk(
        k_tilde,
        model,
        config,
        step_nm_inv=step_nm_inv,
        d_top=d_top,
        d_bot=d_bot,
    )
    err_x = float(np.max(np.abs(analytic[0] - numeric[0])))
    err_y = float(np.max(np.abs(analytic[1] - numeric[1])))
    return DHdkValidationResult(
        max_abs_x_ev_nm=err_x,
        max_abs_y_ev_nm=err_y,
        max_abs_ev_nm=max(err_x, err_y),
        finite_step_nm_inv=float(step_nm_inv),
        k_tilde=complex(k_tilde),
    )


def central_band_indices(matrix_dim: int, count: int = 8) -> tuple[int, ...]:
    """Contiguous central-band window useful for smoke plots/diagnostics."""

    count = int(count)
    if count <= 0 or count > int(matrix_dim):
        raise ValueError(f"Invalid count={count} for matrix_dim={matrix_dim}")
    center = int(matrix_dim) // 2
    lower = max(0, center - count // 2)
    upper = min(int(matrix_dim), lower + count)
    lower = max(0, upper - count)
    return tuple(range(lower, upper))

__all__ = [
    "MAO_W1_EV",
    "MAO_SUBLATTICE_MASS_EV",
    "MAO_REALISTIC_CORRUGATION",
    "MAO_HBAR_VF_K_EV",
    "GRAPHENE_LATTICE_CONSTANT_NM",
    "CARBON_BOND_NM",
    "graphene_k_mag_nm_inv",
    "mao_vf_ev_nm",
    "vf_ev_nm_to_m_per_s",
    "eta_mev_to_ev",
    "MaoHTGConfig",
    "stacking_phase_pair",
    "phase_displacement",
    "stacking_displacements",
    "make_mao_model",
    "add_sublattice_mass",
    "build_mao_hamiltonian",
    "analytic_dhdk",
    "finite_difference_dhdk",
    "DHdkValidationResult",
    "validate_analytic_dhdk",
    "central_band_indices",
]
