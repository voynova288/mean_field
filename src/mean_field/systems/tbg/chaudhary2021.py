from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh

from analysis.response_derivative_gauge import HamiltonianGaugeData, hamiltonian_gauge_data
from analysis.shift_current import (
    JOYA_EQ7_GEOMETRIC_CONVENTION,
    PairTransitionKernel,
    ShiftCurrentComponent,
    ShiftCurrentConvention,
    ShiftCurrentTensors,
    component_kernel_from_gauge_pair,
    precompute_shift_current_tensors,
)
from mean_field.systems.atmg.lattice import ATMGLattice, build_atmg_lattice, build_moire_k_grid
from mean_field.systems.atmg.tbg import TBGCouplingEntry, build_coupling_table, moire_coupling_matrix
from mean_field.systems.tbg.params import TBGParameters
from mean_field.systems.tbg.zero_field.model import (
    _construct_diagonal_block as _b0_construct_diagonal_block,
    _generate_gvec as _b0_generate_gvec,
    _generate_t12 as _b0_generate_t12,
    _generate_t12_zero_fill as _b0_generate_t12_zero_fill,
)

from mean_field.systems.atmg.params import GRAPHENE_LATTICE_CONSTANT_NM


@dataclass(frozen=True)
class ChaudharyTBGConfig:
    """Single-particle TBG parameters used by Chaudhary-Lewandowski-Refael.

    The paper uses ``hbar*v/a = 2.1354 eV`` with graphene lattice constant
    ``a=0.246 nm``, ``u0 = 90 meV`` for AB/BA tunnelling, and ``u = 0.4 u0``
    for AA tunnelling near the magic angle.  The layer sublattice offsets are
    ``Delta_l sigma_z`` and are allowed to differ so that the D3 -> C3 symmetry
    lowering can be tested.
    """

    theta_deg: float = 0.8
    n_shells: int = 3
    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    kinetic_ev: float = 2.1354  # hbar*v/a in eV, paper Eq. (3) paragraph.
    w_ab_ev: float = 0.090
    w_aa_ratio: float = 0.4
    delta1_ev: float = 0.005
    delta2_ev: float = 0.005
    valley: int = 1
    dirac_sign: float = -1.0

    @property
    def vf_ev_nm(self) -> float:
        return float(self.kinetic_ev) * float(self.graphene_lattice_constant_nm)

    @property
    def w_aa_ev(self) -> float:
        return float(self.w_aa_ratio) * float(self.w_ab_ev)


def make_chau_lattice(config: ChaudharyTBGConfig) -> ATMGLattice:
    return build_atmg_lattice(
        float(config.theta_deg),
        n_shells=int(config.n_shells),
        graphene_lattice_constant_nm=float(config.graphene_lattice_constant_nm),
    )


def _validate_valley(valley: int) -> int:
    valley = int(valley)
    if valley not in (-1, 1):
        raise ValueError(f"valley must be +/-1, got {valley}")
    return valley


def _rotated_complex(kvec: complex, angle_rad: float) -> complex:
    return complex(kvec) * complex(math.cos(angle_rad), -math.sin(angle_rad))


def _valley_dirac_block(kvec: complex, *, angle_rad: float, vf_ev_nm: float, valley: int, sign: float) -> np.ndarray:
    q = _rotated_complex(kvec, angle_rad)
    valley = _validate_valley(valley)
    if valley == 1:
        pi = q
        pi_dag = q.conjugate()
    else:
        pi = -q.conjugate()
        pi_dag = -q
    return float(sign) * float(vf_ev_nm) * np.asarray(
        [[0.0, pi_dag], [pi, 0.0]],
        dtype=np.complex128,
    )


def _dirac_dhdk_block(axis: int, *, angle_rad: float, vf_ev_nm: float, valley: int, sign: float) -> np.ndarray:
    rot = complex(math.cos(angle_rad), -math.sin(angle_rad))
    vf = float(sign) * float(vf_ev_nm)
    valley = _validate_valley(valley)
    if valley == 1:
        if axis == 0:
            return np.asarray([[0.0, vf * rot.conjugate()], [vf * rot, 0.0]], dtype=np.complex128)
        if axis == 1:
            return np.asarray([[0.0, -1.0j * vf * rot.conjugate()], [1.0j * vf * rot, 0.0]], dtype=np.complex128)
    else:
        if axis == 0:
            return np.asarray([[0.0, -vf * rot], [-vf * rot.conjugate(), 0.0]], dtype=np.complex128)
        if axis == 1:
            return np.asarray([[0.0, -1.0j * vf * rot], [1.0j * vf * rot.conjugate(), 0.0]], dtype=np.complex128)
    raise ValueError(f"axis must be 0 or 1, got {axis}")


def _layer_angle(config: ChaudharyTBGConfig, layer: int) -> float:
    # Paper convention: layers 1/2 rotated by +/- theta/2.
    half = 0.5 * float(config.theta_deg) * math.pi / 180.0
    if int(layer) == 1:
        return half
    if int(layer) == 2:
        return -half
    raise ValueError(f"layer must be 1 or 2, got {layer}")


def _orbital_slice(g_index: int, layer: int) -> slice:
    start = 4 * int(g_index) + (0 if int(layer) == 1 else 2)
    return slice(start, start + 2)


def build_chau_hamiltonian(
    k_tilde: complex,
    lattice: ATMGLattice,
    config: ChaudharyTBGConfig,
    *,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> np.ndarray:
    """Continuum TBG Hamiltonian for one valley and one spin.

    Basis per moire reciprocal vector is ``(A1,B1,A2,B2)``.  The second layer
    uses the same momentum-gauge offset as the existing local TBG builder.
    """

    valley = _validate_valley(config.valley)
    dim = 4 * int(lattice.n_g)
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    for ig, gvec in enumerate(lattice.g_vectors):
        sl1 = _orbital_slice(ig, 1)
        sl2 = _orbital_slice(ig, 2)
        k1 = complex(k_tilde + gvec)
        k2 = complex(k_tilde + gvec + valley * lattice.q0)
        hamiltonian[sl1, sl1] = _valley_dirac_block(
            k1,
            angle_rad=_layer_angle(config, 1),
            vf_ev_nm=config.vf_ev_nm,
            valley=valley,
            sign=config.dirac_sign,
        )
        hamiltonian[sl2, sl2] = _valley_dirac_block(
            k2,
            angle_rad=_layer_angle(config, 2),
            vf_ev_nm=config.vf_ev_nm,
            valley=valley,
            sign=config.dirac_sign,
        )
        hamiltonian[sl1.start, sl1.start] += float(config.delta1_ev)
        hamiltonian[sl1.start + 1, sl1.start + 1] -= float(config.delta1_ev)
        hamiltonian[sl2.start, sl2.start] += float(config.delta2_ev)
        hamiltonian[sl2.start + 1, sl2.start + 1] -= float(config.delta2_ev)

    resolved_table = coupling_table
    if resolved_table is None:
        resolved_table = build_coupling_table(lattice.g_vectors, lattice.q_vectors, valley=valley)
    for entry in resolved_table:
        sl1 = _orbital_slice(entry.odd_index, 1)
        sl2 = _orbital_slice(entry.even_index, 2)
        coupling = moire_coupling_matrix(
            entry.channel,
            w_ab=float(config.w_ab_ev),
            w_aa=float(config.w_aa_ev),
            valley=valley,
        )
        hamiltonian[sl1, sl2] += coupling
        hamiltonian[sl2, sl1] += coupling.conjugate().T

    return hamiltonian


def analytic_dhdk(lattice: ATMGLattice, config: ChaudharyTBGConfig) -> tuple[np.ndarray, np.ndarray]:
    dim = 4 * int(lattice.n_g)
    out = [np.zeros((dim, dim), dtype=np.complex128), np.zeros((dim, dim), dtype=np.complex128)]
    for ig in range(lattice.n_g):
        for layer in (1, 2):
            sl = _orbital_slice(ig, layer)
            for axis in (0, 1):
                out[axis][sl, sl] = _dirac_dhdk_block(
                    axis,
                    angle_rad=_layer_angle(config, layer),
                    vf_ev_nm=config.vf_ev_nm,
                    valley=config.valley,
                    sign=config.dirac_sign,
                )
    return out[0], out[1]


def finite_difference_dhdk(
    k_tilde: complex,
    lattice: ATMGLattice,
    config: ChaudharyTBGConfig,
    *,
    step_nm_inv: float = 1.0e-6,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    step = float(step_nm_inv)
    if step <= 0.0:
        raise ValueError(f"step_nm_inv must be positive, got {step_nm_inv}")
    k = complex(k_tilde)
    hx_p = build_chau_hamiltonian(k + step, lattice, config, coupling_table=coupling_table)
    hx_m = build_chau_hamiltonian(k - step, lattice, config, coupling_table=coupling_table)
    hy_p = build_chau_hamiltonian(k + 1.0j * step, lattice, config, coupling_table=coupling_table)
    hy_m = build_chau_hamiltonian(k - 1.0j * step, lattice, config, coupling_table=coupling_table)
    return (hx_p - hx_m) / (2.0 * step), (hy_p - hy_m) / (2.0 * step)


@dataclass(frozen=True)
class DHdkValidationResult:
    max_abs_x_ev_nm: float
    max_abs_y_ev_nm: float
    max_abs_ev_nm: float
    finite_step_nm_inv: float


def validate_analytic_dhdk(
    k_tilde: complex,
    lattice: ATMGLattice,
    config: ChaudharyTBGConfig,
    *,
    step_nm_inv: float = 1.0e-6,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> DHdkValidationResult:
    analytic = analytic_dhdk(lattice, config)
    numeric = finite_difference_dhdk(
        k_tilde,
        lattice,
        config,
        step_nm_inv=step_nm_inv,
        coupling_table=coupling_table,
    )
    err_x = float(np.max(np.abs(analytic[0] - numeric[0])))
    err_y = float(np.max(np.abs(analytic[1] - numeric[1])))
    return DHdkValidationResult(err_x, err_y, max(err_x, err_y), float(step_nm_inv))


def centered_flat_indices(matrix_dim: int) -> tuple[int, int]:
    center = int(matrix_dim) // 2
    return center - 1, center


def fd_transition_pairs(
    matrix_dim: int,
    *,
    n_fd_bands_each_side: int = 10,
    mode: str = "same_side",
) -> tuple[tuple[int, int], ...]:
    """Return positive-energy flat--dispersive direct-transition pairs.

    ``mode='same_side'`` is the Chaudhary Fig. 2 convention: transitions
    between the lower dispersive bands and the valence flat band, plus
    transitions between the conduction flat band and the upper dispersive
    bands.  These transitions are Pauli blocked at charge neutrality and turn
    on when the Fermi level lies between a flat band and a dispersive band,
    matching the paper's discussion of Fig. 2(e).

    ``mode='cross_gap'`` selects lower-dispersive -> conduction-flat and
    valence-flat -> upper-dispersive transitions.  ``mode='all'`` is the
    earlier diagnostic union of both choices; it overcounts the paper's FD
    panel and was the source of a spurious large FD signal at neutrality.
    """

    v_flat, c_flat = centered_flat_indices(matrix_dim)
    lower = tuple(range(max(0, v_flat - int(n_fd_bands_each_side)), v_flat))
    upper = tuple(range(c_flat + 1, min(int(matrix_dim), c_flat + 1 + int(n_fd_bands_each_side))))
    key = str(mode).lower().replace("-", "_")
    pairs: set[tuple[int, int]] = set()
    if key in {"same_side", "paper", "fig2"}:
        pairs.update((int(d), int(v_flat)) for d in lower)
        pairs.update((int(c_flat), int(d)) for d in upper)
    elif key == "cross_gap":
        pairs.update((int(d), int(c_flat)) for d in lower)
        pairs.update((int(v_flat), int(d)) for d in upper)
    elif key == "all":
        flat = {v_flat, c_flat}
        for f in flat:
            for d in lower + upper:
                n, m = sorted((int(f), int(d)))
                if n != m:
                    pairs.add((n, m))
    else:
        raise ValueError(f"Unsupported FD pair mode {mode!r}; expected same_side, cross_gap, or all")
    return tuple(sorted(pairs))


def flat_filling_to_mu(
    flat_energies_ev: np.ndarray,
    filling: float,
    *,
    degeneracy: float = 4.0,
) -> float:
    """Approximate T=0 chemical potential for a target flat-band filling nu.

    ``nu`` is the conventional TBG filling relative to charge neutrality.  With
    spin/valley degeneracy ``g`` the per-flavor number of occupied flat states
    per k point is ``1 + nu/g``.  The input must contain both central flat-band
    energies for a uniform k sample and one valley/spin flavor.
    """

    energies = np.sort(np.asarray(flat_energies_ev, dtype=float).reshape(-1))
    if energies.size == 0:
        raise ValueError("flat_energies_ev is empty")
    n_k = energies.size / 2.0
    target_per_k = 1.0 + float(filling) / float(degeneracy)
    target_count = target_per_k * n_k
    if target_count <= 0.0:
        return float(energies[0] - 1.0e-3)
    if target_count >= energies.size:
        return float(energies[-1] + 1.0e-3)
    idx = int(math.floor(target_count))
    if abs(target_count - idx) < 1.0e-12 and 0 < idx < energies.size:
        return float(0.5 * (energies[idx - 1] + energies[idx]))
    idx = min(max(idx, 0), energies.size - 1)
    return float(energies[idx])


def sample_flat_energies_for_mu(
    lattice: ATMGLattice,
    config: ChaudharyTBGConfig,
    *,
    mesh_size: int,
    coupling_table: tuple[TBGCouplingEntry, ...] | None = None,
) -> np.ndarray:
    from scipy.linalg import eigvalsh

    _frac, k_grid = build_moire_k_grid(lattice, int(mesh_size), endpoint=False)
    dim = 4 * int(lattice.n_g)
    v_flat, c_flat = centered_flat_indices(dim)
    out = []
    for k in k_grid.reshape(-1):
        evals = eigvalsh(build_chau_hamiltonian(complex(k), lattice, config, coupling_table=coupling_table))
        out.append([float(evals[v_flat]), float(evals[c_flat])])
    return np.asarray(out, dtype=float)


def make_b0_parameters(config: ChaudharyTBGConfig) -> TBGParameters:
    """Return the repository's earlier b0 BM parameters for Chaudhary's model.

    This is the convention used by the existing noninteracting/HF TBG code in
    ``mean_field.systems.tbg.zero_field``.  Its momenta are dimensionless
    (graphene lattice constant set to one) and energies are in meV.
    """

    return TBGParameters.from_degrees(
        float(config.theta_deg),
        vf=1000.0 * float(config.kinetic_ev),
        w0=1000.0 * float(config.w_aa_ev),
        w1=1000.0 * float(config.w_ab_ev),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )


def b0_mbz_area_nm_inv_sq(params: TBGParameters, config: ChaudharyTBGConfig) -> float:
    area_dimensionless = abs(float(params.g1.real * params.g2.imag - params.g1.imag * params.g2.real))
    return area_dimensionless / (float(config.graphene_lattice_constant_nm) ** 2)


def b0_moire_length_nm(params: TBGParameters, config: ChaudharyTBGConfig) -> float:
    # Same expression as the older b0 parameterization, but restored to nm.
    theta = float(params.dtheta_rad)
    return float(config.graphene_lattice_constant_nm) / (2.0 * math.sin(theta / 2.0))


def _add_b0_layer_sublattice_offsets_mev(hamiltonian_mev: np.ndarray, *, lg: int, delta1_mev: float, delta2_mev: float) -> np.ndarray:
    out = np.array(hamiltonian_mev, dtype=np.complex128, copy=True)
    for ig in range(int(lg) * int(lg)):
        base = 4 * ig
        out[base, base] += float(delta1_mev)
        out[base + 1, base + 1] -= float(delta1_mev)
        out[base + 2, base + 2] += float(delta2_mev)
        out[base + 3, base + 3] -= float(delta2_mev)
    return out


def build_chau_b0_hamiltonian(
    k_dimless: complex,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
) -> np.ndarray:
    """Build the earlier b0 BM Hamiltonian plus Chaudhary sublattice offsets.

    Parameters
    ----------
    k_dimless:
        Momentum in the old b0 convention (graphene lattice constant set to
        one), not in nm^{-1}.
    Returns
    -------
    np.ndarray
        Hamiltonian in eV.
    """

    zeta = _validate_valley(config.valley)
    resolved_gvec = _b0_generate_gvec(params, int(lg)) if gvec is None else np.asarray(gvec, dtype=np.complex128)
    if tunnel is None:
        tunnel_builder = _b0_generate_t12 if bool(periodic_g_grid) else _b0_generate_t12_zero_fill
        tunnel = tunnel_builder(params, int(lg), zeta)
    h_mev = _b0_construct_diagonal_block(params, resolved_gvec, int(lg), complex(k_dimless), zeta, bool(sigma_rotation))
    h_mev = h_mev + np.asarray(tunnel, dtype=np.complex128)
    h_mev = _add_b0_layer_sublattice_offsets_mev(
        h_mev,
        lg=int(lg),
        delta1_mev=1000.0 * float(config.delta1_ev),
        delta2_mev=1000.0 * float(config.delta2_ev),
    )
    return h_mev * 1.0e-3


def finite_difference_b0_dhdk(
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    step_dimless: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """dH/dk in eV nm for the old b0 model.

    The finite difference is taken with respect to dimensionless b0 momentum
    and then multiplied by the graphene lattice constant to convert to a
    derivative with respect to physical momentum in nm^{-1}.
    """

    step = float(step_dimless)
    if step <= 0.0:
        raise ValueError(f"step_dimless must be positive, got {step_dimless}")
    gvec = _b0_generate_gvec(params, int(lg))
    tunnel_builder = _b0_generate_t12 if bool(periodic_g_grid) else _b0_generate_t12_zero_fill
    tunnel = tunnel_builder(params, int(lg), _validate_valley(config.valley))
    hx_p = build_chau_b0_hamiltonian(step + 0.0j, params, config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    hx_m = build_chau_b0_hamiltonian(-step + 0.0j, params, config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    hy_p = build_chau_b0_hamiltonian(1.0j * step, params, config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    hy_m = build_chau_b0_hamiltonian(-1.0j * step, params, config, lg=lg, sigma_rotation=sigma_rotation, periodic_g_grid=periodic_g_grid, gvec=gvec, tunnel=tunnel)
    scale_to_nm = float(config.graphene_lattice_constant_nm)
    return (hx_p - hx_m) / (2.0 * step) * scale_to_nm, (hy_p - hy_m) / (2.0 * step) * scale_to_nm


@dataclass(frozen=True)
class B0ShiftCurrentPoint:
    """Old b0 TBG point data prepared for the generic shift-current API."""

    k_dimless: complex
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    dhdk_ev_nm: np.ndarray
    gauge_data: HamiltonianGaugeData


def b0_shift_current_point_data(
    k_dimless: complex,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    step_dimless: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-8,
    gvec: np.ndarray | None = None,
    tunnel: np.ndarray | None = None,
) -> B0ShiftCurrentPoint:
    """Diagonalize the old b0 model and prepare generic shift-current data."""

    resolved_gvec = _b0_generate_gvec(params, int(lg)) if gvec is None else np.asarray(gvec, dtype=np.complex128)
    if tunnel is None:
        tunnel_builder = _b0_generate_t12 if bool(periodic_g_grid) else _b0_generate_t12_zero_fill
        tunnel = tunnel_builder(params, int(lg), _validate_valley(config.valley))
    hmat = build_chau_b0_hamiltonian(
        complex(k_dimless),
        params,
        config,
        lg=int(lg),
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        gvec=resolved_gvec,
        tunnel=tunnel,
    )
    evals, evecs = eigh(hmat)
    dhdk = np.stack(
        finite_difference_b0_dhdk(
            params,
            config,
            lg=int(lg),
            sigma_rotation=sigma_rotation,
            periodic_g_grid=periodic_g_grid,
            step_dimless=step_dimless,
        ),
        axis=0,
    )
    gauge = hamiltonian_gauge_data(evals, evecs, dhdk, denominator_cutoff=float(denominator_cutoff_ev))
    return B0ShiftCurrentPoint(
        k_dimless=complex(k_dimless),
        energies_ev=np.asarray(evals, dtype=float),
        eigenvectors=np.asarray(evecs, dtype=np.complex128),
        dhdk_ev_nm=dhdk,
        gauge_data=gauge,
    )


def b0_shift_current_tensors_at_k(
    k_dimless: complex,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    step_dimless: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-8,
    principal_value_eta_ev: float | None = None,
    mu_ev: float = 0.0,
    temperature_k: float = 0.0,
) -> ShiftCurrentTensors:
    """Return full generic shift-current tensors for a tiny b0 point."""

    point = b0_shift_current_point_data(
        k_dimless,
        params,
        config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        step_dimless=step_dimless,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    return precompute_shift_current_tensors(
        point.energies_ev,
        point.eigenvectors,
        point.dhdk_ev_nm,
        mu_ev=mu_ev,
        temperature_k=temperature_k,
        denominator_cutoff_ev=denominator_cutoff_ev,
        principal_value_eta_ev=principal_value_eta_ev,
    )


def b0_component_kernel_at_k(
    k_dimless: complex,
    params: TBGParameters,
    config: ChaudharyTBGConfig,
    initial_band: int,
    final_band: int,
    component: ShiftCurrentComponent | tuple[int, int, int] | str,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    periodic_g_grid: bool = False,
    step_dimless: float = 1.0e-6,
    denominator_cutoff_ev: float = 1.0e-8,
    principal_value_eta_ev: float | None = None,
    convention: ShiftCurrentConvention = JOYA_EQ7_GEOMETRIC_CONVENTION,
) -> PairTransitionKernel:
    """Return a selected-pair b0 kernel using the generic shift-current API."""

    point = b0_shift_current_point_data(
        k_dimless,
        params,
        config,
        lg=lg,
        sigma_rotation=sigma_rotation,
        periodic_g_grid=periodic_g_grid,
        step_dimless=step_dimless,
        denominator_cutoff_ev=denominator_cutoff_ev,
    )
    return component_kernel_from_gauge_pair(
        point.gauge_data.velocity_h,
        point.gauge_data.energies,
        point.gauge_data.berry_connection,
        int(initial_band),
        int(final_band),
        component,
        denominator_cutoff_ev=denominator_cutoff_ev,
        second_velocity_h=None,
        principal_value_eta_ev=principal_value_eta_ev,
        convention=convention,
    )


def b0_fig2_kpath(params: TBGParameters, points_per_segment: int):
    from mean_field.systems.tbg.zero_field.path import build_kpath_from_nodes, select_adjacent_m_point

    # Chaudhary Fig. 2(a) is plotted through kappa' -> gamma -> kappa -> mu -> kappa.
    # The old b0 convention has two inequivalent moire corners params.kb_point and params.kt;
    # the adjacent M point is selected using the existing path helper.
    mu = select_adjacent_m_point(params)
    return build_kpath_from_nodes(
        (params.kb_point, params.gamma_point, params.kt, mu, params.kt),
        ("kappa'", "gamma", "kappa", "mu", "kappa"),
        int(points_per_segment),
    )


def config_summary(config: ChaudharyTBGConfig, lattice: ATMGLattice | None = None, *, b0_params: TBGParameters | None = None, lg: int | None = None) -> dict[str, object]:
    base = {
        "theta_deg": float(config.theta_deg),
        "n_shells": int(config.n_shells),
        "graphene_lattice_constant_nm": float(config.graphene_lattice_constant_nm),
        "vf_ev_nm": float(config.vf_ev_nm),
        "kinetic_ev_hbar_v_over_a": float(config.kinetic_ev),
        "w_ab_ev": float(config.w_ab_ev),
        "w_aa_ev": float(config.w_aa_ev),
        "w_aa_ratio": float(config.w_aa_ratio),
        "delta1_ev": float(config.delta1_ev),
        "delta2_ev": float(config.delta2_ev),
        "valley": int(config.valley),
        "dirac_sign": float(config.dirac_sign),
    }
    if lattice is not None:
        base.update(
            {
                "matrix_dim": int(4 * lattice.n_g),
                "n_g": int(lattice.n_g),
                "moire_length_nm": float(lattice.l_m),
                "mbz_area_nm_inv_sq": float(lattice.mbz_area),
            }
        )
    if b0_params is not None:
        resolved_lg = int(lg) if lg is not None else None
        base.update(
            {
                "model_convention": "previous_b0_zero_field",
                "lg": resolved_lg,
                "matrix_dim": None if resolved_lg is None else int(4 * resolved_lg * resolved_lg),
                "n_g": None if resolved_lg is None else int(resolved_lg * resolved_lg),
                "moire_length_nm": b0_moire_length_nm(b0_params, config),
                "mbz_area_nm_inv_sq": b0_mbz_area_nm_inv_sq(b0_params, config),
                "b0_g1": [float(b0_params.g1.real), float(b0_params.g1.imag)],
                "b0_g2": [float(b0_params.g2.real), float(b0_params.g2.imag)],
                "b0_kt": [float(b0_params.kt.real), float(b0_params.kt.imag)],
                "b0_kb_point": [float(b0_params.kb_point.real), float(b0_params.kb_point.imag)],
            }
        )
    return base

__all__ = [
    "B0ShiftCurrentPoint",
    "ChaudharyTBGConfig",
    "DHdkValidationResult",
    "analytic_dhdk",
    "b0_component_kernel_at_k",
    "b0_fig2_kpath",
    "b0_mbz_area_nm_inv_sq",
    "b0_moire_length_nm",
    "b0_shift_current_point_data",
    "b0_shift_current_tensors_at_k",
    "build_chau_b0_hamiltonian",
    "build_chau_hamiltonian",
    "centered_flat_indices",
    "config_summary",
    "fd_transition_pairs",
    "finite_difference_b0_dhdk",
    "finite_difference_dhdk",
    "flat_filling_to_mu",
    "make_b0_parameters",
    "make_chau_lattice",
    "sample_flat_energies_for_mu",
    "validate_analytic_dhdk",
]
