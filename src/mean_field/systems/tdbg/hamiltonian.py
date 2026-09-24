from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from mean_field.core.validation import validate_valley as _validate_valley

from .lattice import TDBGLattice, strain_twist_matrix
from .params import TDBGParameters, VALID_STACKINGS, VALID_VALLEYS


MOIRE_CHANNELS = (0, 1, 2)


@dataclass(frozen=True)
class MoireCouplingEntry:
    source_index: int
    target_index: int
    channel: int



def _validate_stacking(stacking: str) -> str:
    if stacking not in VALID_STACKINGS:
        raise ValueError(f"Expected stacking in {VALID_STACKINGS}, got {stacking}")
    return stacking


def _top_bottom_bilayer_stackings(stacking: str) -> tuple[str, str]:
    """Return physical ``(top, bottom)`` Bernal stacking orders.

    TDBG q-sector 0 carries the positive/top-bilayer layer potentials and
    q-sector 1 the negative/bottom-bilayer potentials.  Keeping both AB-BA
    domains explicit matters for orientation-sensitive rank-3 responses even
    when their zero-field scalar spectra are symmetry equivalent.
    """

    resolved = _validate_stacking(stacking)
    if resolved == "AB-AB":
        return "AB", "AB"
    if resolved == "AB-BA":
        return "AB", "BA"
    return "BA", "AB"


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


def _uses_exact_bernal_structure_factor(params: TDBGParameters) -> bool:
    return str(params.model_name) == "liu_dai2020_eq16_exact_f"


def _bernal_structure_factor_data(
    k_plus: complex,
    k_minus: complex,
    params: TDBGParameters,
    *,
    valley: int,
) -> tuple[complex, np.ndarray, np.ndarray]:
    """Return Liu-Dai Eq.(16) ``f``, local gradient, and local Hessian.

    The paper uses the full monolayer structure factor in the Bernal interlayer
    block. Its paper K valley is code valley ``-1``; therefore the absolute
    graphene corner entering ``f`` is ``K_x = -valley * 4π/(3a)``. The
    expansion obeys ``f = (sqrt(3)a/2) k_minus + O(k^2)``, so the exact block
    reduces to the existing linearized ``gamma3/gamma4`` block at small q.
    Gradient and Hessian axes are local Cartesian ``(qx, qy)``.
    """

    resolved_valley = _validate_valley(valley)
    qx = (complex(k_plus) + complex(k_minus)) / (2.0 * resolved_valley)
    qy = (complex(k_plus) - complex(k_minus)) / (2.0j)
    a = float(params.graphene_lattice_constant_nm)
    kx = -resolved_valley * 4.0 * math.pi / (3.0 * a) + qx
    sqrt3 = math.sqrt(3.0)
    term_1 = np.exp(-1.0j * sqrt3 * a * qy / 3.0)
    term_2 = np.exp(1.0j * (a * kx / 2.0 + sqrt3 * a * qy / 6.0))
    term_3 = np.exp(1.0j * (-a * kx / 2.0 + sqrt3 * a * qy / 6.0))
    f = term_1 + term_2 + term_3
    gradient = np.asarray(
        [
            1.0j * a * (term_2 - term_3) / 2.0,
            -1.0j * sqrt3 * a * term_1 / 3.0 + 1.0j * sqrt3 * a * (term_2 + term_3) / 6.0,
        ],
        dtype=np.complex128,
    )
    hessian = np.asarray(
        [
            [-a * a * (term_2 + term_3) / 4.0, sqrt3 * a * a * (-term_2 + term_3) / 12.0],
            [sqrt3 * a * a * (-term_2 + term_3) / 12.0, -a * a * term_1 / 3.0 - a * a * (term_2 + term_3) / 12.0],
        ],
        dtype=np.complex128,
    )
    return complex(f), gradient, hessian


def blg_interlayer(
    k_plus: complex,
    k_minus: complex,
    params: TDBGParameters,
    *,
    valley: int = 1,
) -> np.ndarray:
    if _uses_exact_bernal_structure_factor(params):
        f, _gradient, _hessian = _bernal_structure_factor_data(k_plus, k_minus, params, valley=valley)
        return np.asarray(
            [
                [params.gamma4 * np.conjugate(f), params.gamma1],
                [params.gamma3 * f, params.gamma4 * np.conjugate(f)],
            ],
            dtype=np.complex128,
        )
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
    valley: int = 1,
) -> np.ndarray:
    if stacking_order == "AB":
        upper = _dirac_non_dimer(k_plus, k_minus, upper_layer_potential, params)
        lower = _dirac_dimer(k_plus, k_minus, lower_layer_potential, params)
        coupling_upper_lower = blg_interlayer(k_plus, k_minus, params, valley=valley).conjugate().T
        coupling_lower_upper = blg_interlayer(k_plus, k_minus, params, valley=valley)
    elif stacking_order == "BA":
        upper = _dirac_dimer(k_plus, k_minus, upper_layer_potential, params)
        lower = _dirac_non_dimer(k_plus, k_minus, lower_layer_potential, params)
        coupling_upper_lower = blg_interlayer(k_plus, k_minus, params, valley=valley)
        coupling_lower_upper = blg_interlayer(k_plus, k_minus, params, valley=valley).conjugate().T
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


def _site_frame_jacobian(site_index: int, lattice: TDBGLattice, *, valley: int) -> np.ndarray:
    """Return d(site-frame momentum)/d(lab k) for one q site.

    Rows are site-frame Cartesian components and columns are lab-frame
    ``(kx, ky)`` axes.  The Hamiltonian is affine in lab ``k_tilde`` for fixed
    lattice/strain parameters, so this Jacobian is k independent.
    """

    valley = _validate_valley(valley)
    sector = int(round(float(lattice.q_sites[site_index, 2])))
    twist_sign = -1.0 if sector == 0 else 1.0
    deformation = strain_twist_matrix(
        twist_sign * valley * lattice.theta_rad / 2.0,
        lattice.phi_rad,
        twist_sign * valley * lattice.epsilon / 2.0,
        lattice.poisson_ratio,
    )
    return np.eye(2, dtype=float) + deformation

def _site_frame_momentum(
    k_tilde: complex,
    site_index: int,
    lattice: TDBGLattice,
    *,
    valley: int,
) -> tuple[complex, complex]:
    valley = _validate_valley(valley)
    deformation = _site_frame_jacobian(site_index, lattice, valley=valley) - np.eye(2, dtype=float)
    strain_tensor = 0.5 * (deformation + deformation.T)
    gauge_shift = valley * lattice.gauge_connection_nm_inv * np.asarray(
        [strain_tensor[0, 0] - strain_tensor[1, 1], -2.0 * strain_tensor[0, 1]],
        dtype=float,
    )
    momentum_lab = np.asarray([float(complex(k_tilde).real), float(complex(k_tilde).imag)], dtype=float)
    kj = _site_frame_jacobian(site_index, lattice, valley=valley) @ (momentum_lab + lattice.q_sites[site_index, :2] + gauge_shift)
    k_minus = complex(valley * kj[0], -kj[1])
    k_plus = complex(valley * kj[0], kj[1])
    return k_plus, k_minus


def _site_frame_k_derivatives(site_index: int, lattice: TDBGLattice, *, valley: int) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Return ``((dk_plus/dkx, dk_minus/dkx), (dk_plus/dky, dk_minus/dky))``."""

    valley = _validate_valley(valley)
    jac = _site_frame_jacobian(site_index, lattice, valley=valley)
    out: list[tuple[complex, complex]] = []
    for axis in (0, 1):
        dkj = jac[:, axis]
        dk_plus = complex(valley * dkj[0], dkj[1])
        dk_minus = complex(valley * dkj[0], -dkj[1])
        out.append((dk_plus, dk_minus))
    return out[0], out[1]

def _dirac_block_derivative(dk_plus: complex, dk_minus: complex, params: TDBGParameters) -> np.ndarray:
    return np.asarray(
        [
            [0.0, -params.vf * dk_minus],
            [-params.vf * dk_plus, 0.0],
        ],
        dtype=np.complex128,
    )

def _local_cartesian_direction(dk_plus: complex, dk_minus: complex, *, valley: int) -> np.ndarray:
    resolved_valley = _validate_valley(valley)
    return np.asarray(
        [
            (complex(dk_plus) + complex(dk_minus)) / (2.0 * resolved_valley),
            (complex(dk_plus) - complex(dk_minus)) / (2.0j),
        ],
        dtype=np.complex128,
    )


def _blg_interlayer_derivative(
    k_plus: complex,
    k_minus: complex,
    dk_plus: complex,
    dk_minus: complex,
    params: TDBGParameters,
    *,
    valley: int,
) -> np.ndarray:
    if _uses_exact_bernal_structure_factor(params):
        _f, gradient, _hessian = _bernal_structure_factor_data(k_plus, k_minus, params, valley=valley)
        direction = _local_cartesian_direction(dk_plus, dk_minus, valley=valley)
        df = complex(gradient @ direction)
        return np.asarray(
            [
                [params.gamma4 * np.conjugate(df), 0.0],
                [params.gamma3 * df, params.gamma4 * np.conjugate(df)],
            ],
            dtype=np.complex128,
        )
    return np.asarray(
        [
            [params.v4 * dk_plus, 0.0],
            [params.v3 * dk_minus, params.v4 * dk_plus],
        ],
        dtype=np.complex128,
    )


def _blg_interlayer_second_derivative(
    k_plus: complex,
    k_minus: complex,
    dk_plus_a: complex,
    dk_minus_a: complex,
    dk_plus_b: complex,
    dk_minus_b: complex,
    params: TDBGParameters,
    *,
    valley: int,
) -> np.ndarray:
    if not _uses_exact_bernal_structure_factor(params):
        return np.zeros((2, 2), dtype=np.complex128)
    _f, _gradient, hessian = _bernal_structure_factor_data(k_plus, k_minus, params, valley=valley)
    direction_a = _local_cartesian_direction(dk_plus_a, dk_minus_a, valley=valley)
    direction_b = _local_cartesian_direction(dk_plus_b, dk_minus_b, valley=valley)
    d2f = complex(direction_a @ hessian @ direction_b)
    return np.asarray(
        [
            [params.gamma4 * np.conjugate(d2f), 0.0],
            [params.gamma3 * d2f, params.gamma4 * np.conjugate(d2f)],
        ],
        dtype=np.complex128,
    )


def _assemble_bilayer_interlayer_derivative(inter: np.ndarray, *, stacking_order: str) -> np.ndarray:
    block = np.zeros((4, 4), dtype=np.complex128)
    if stacking_order == "AB":
        block[0:2, 2:4] = inter.conjugate().T
        block[2:4, 0:2] = inter
    elif stacking_order == "BA":
        block[0:2, 2:4] = inter
        block[2:4, 0:2] = inter.conjugate().T
    else:
        raise ValueError(f"Unsupported local bilayer stacking order: {stacking_order}")
    return block

def build_bilayer_block_derivative(
    dk_plus: complex,
    dk_minus: complex,
    params: TDBGParameters,
    *,
    stacking_order: str = "AB",
    k_plus: complex = 0.0j,
    k_minus: complex = 0.0j,
    valley: int = 1,
) -> np.ndarray:
    """Analytic lab-axis derivative of a local bilayer block."""

    upper = _dirac_block_derivative(dk_plus, dk_minus, params)
    lower = _dirac_block_derivative(dk_plus, dk_minus, params)
    inter = _blg_interlayer_derivative(k_plus, k_minus, dk_plus, dk_minus, params, valley=valley)
    block = _assemble_bilayer_interlayer_derivative(inter, stacking_order=stacking_order)
    block[0:2, 0:2] = upper
    block[2:4, 2:4] = lower
    return block


def build_bilayer_block_second_derivative(
    k_plus: complex,
    k_minus: complex,
    dk_plus_a: complex,
    dk_minus_a: complex,
    dk_plus_b: complex,
    dk_minus_b: complex,
    params: TDBGParameters,
    *,
    stacking_order: str = "AB",
    valley: int = 1,
) -> np.ndarray:
    """Analytic second lab-axis derivative of a local bilayer block."""

    inter = _blg_interlayer_second_derivative(
        k_plus,
        k_minus,
        dk_plus_a,
        dk_minus_a,
        dk_plus_b,
        dk_minus_b,
        params,
        valley=valley,
    )
    return _assemble_bilayer_interlayer_derivative(inter, stacking_order=stacking_order)

def build_site_block_derivative(
    site_index: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    axis: int,
    k_tilde: complex = 0.0j,
) -> np.ndarray:
    """Analytic ``dH_site/dk_axis`` in eV nm for lab-frame k axes."""

    valley = _validate_valley(valley)
    _validate_stacking(params.stacking)
    if int(axis) not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    sector = int(round(float(lattice.q_sites[site_index, 2])))
    k_plus, k_minus = _site_frame_momentum(k_tilde, site_index, lattice, valley=valley)
    dk_plus, dk_minus = _site_frame_k_derivatives(site_index, lattice, valley=valley)[int(axis)]
    top_stacking, bottom_stacking = _top_bottom_bilayer_stackings(params.stacking)
    stacking_order = top_stacking if sector == 0 else bottom_stacking
    return build_bilayer_block_derivative(
        dk_plus,
        dk_minus,
        params,
        stacking_order=stacking_order,
        k_plus=k_plus,
        k_minus=k_minus,
        valley=valley,
    )


def build_site_block_second_derivative(
    site_index: int,
    lattice: TDBGLattice,
    params: TDBGParameters,
    *,
    valley: int = 1,
    axis_a: int,
    axis_b: int,
    k_tilde: complex = 0.0j,
) -> np.ndarray:
    """Analytic ``d²H_site/dk_axis_a dk_axis_b`` in eV nm²."""

    valley = _validate_valley(valley)
    _validate_stacking(params.stacking)
    if int(axis_a) not in (0, 1) or int(axis_b) not in (0, 1):
        raise ValueError(f"axes must be 0 or 1, got {(axis_a, axis_b)}")
    sector = int(round(float(lattice.q_sites[site_index, 2])))
    k_plus, k_minus = _site_frame_momentum(k_tilde, site_index, lattice, valley=valley)
    derivatives = _site_frame_k_derivatives(site_index, lattice, valley=valley)
    dk_plus_a, dk_minus_a = derivatives[int(axis_a)]
    dk_plus_b, dk_minus_b = derivatives[int(axis_b)]
    top_stacking, bottom_stacking = _top_bottom_bilayer_stackings(params.stacking)
    stacking_order = top_stacking if sector == 0 else bottom_stacking
    return build_bilayer_block_second_derivative(
        k_plus,
        k_minus,
        dk_plus_a,
        dk_minus_a,
        dk_plus_b,
        dk_minus_b,
        params,
        stacking_order=stacking_order,
        valley=valley,
    )

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

    top_stacking, bottom_stacking = _top_bottom_bilayer_stackings(params.stacking)
    if sector == 0:
        return build_bilayer_block(
            k_plus,
            k_minus,
            params,
            upper_layer_potential=1.5 * params.Delta,
            lower_layer_potential=0.5 * params.Delta,
            stacking_order=top_stacking,
            valley=valley,
        )

    return build_bilayer_block(
        k_plus,
        k_minus,
        params,
        upper_layer_potential=-0.5 * params.Delta,
        lower_layer_potential=-1.5 * params.Delta,
        stacking_order=bottom_stacking,
        valley=valley,
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


def build_hamiltonian_dhdk(
    lattice: TDBGLattice,
    params: TDBGParameters,
    valley: int = 1,
    *,
    k_tilde: complex = 0.0j,
) -> np.ndarray:
    """Return analytic ``dH/d(kx,ky)`` in eV nm for lab-frame k axes.

    The default TDBG model is affine in ``k_tilde``. The Liu-Dai exact-Eq.(16)
    profile retains the full Bernal structure factor, making this derivative
    k dependent; callers of that profile must pass the evaluated ``k_tilde``.
    """

    valley = _validate_valley(valley)
    dim = int(lattice.matrix_dim)
    out = np.zeros((2, dim, dim), dtype=np.complex128)
    for axis in (0, 1):
        for site_index in range(lattice.n_q):
            sl = slice(4 * site_index, 4 * (site_index + 1))
            out[axis, sl, sl] = build_site_block_derivative(
                site_index,
                lattice,
                params,
                valley=valley,
                axis=axis,
                k_tilde=k_tilde,
            )
    return out


def build_hamiltonian_d2hdk2(
    lattice: TDBGLattice,
    params: TDBGParameters | None = None,
    valley: int = 1,
    *,
    k_tilde: complex = 0.0j,
) -> np.ndarray:
    """Return analytic ``d²H/dk²`` in eV nm².

    It is identically zero for the affine default model and nonzero only for
    the Liu-Dai exact-Eq.(16) Bernal structure-factor profile.
    """

    dim = int(lattice.matrix_dim)
    out = np.zeros((2, 2, dim, dim), dtype=np.complex128)
    if params is None or not _uses_exact_bernal_structure_factor(params):
        return out
    resolved_valley = _validate_valley(valley)
    for axis_a in (0, 1):
        for axis_b in (0, 1):
            for site_index in range(lattice.n_q):
                sl = slice(4 * site_index, 4 * (site_index + 1))
                out[axis_a, axis_b, sl, sl] = build_site_block_second_derivative(
                    site_index,
                    lattice,
                    params,
                    valley=resolved_valley,
                    axis_a=axis_a,
                    axis_b=axis_b,
                    k_tilde=k_tilde,
                )
    return out

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
