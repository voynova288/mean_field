from __future__ import annotations

from ._spectrum_shared import *  # noqa: F401,F403
from ._spectrum_params import *  # noqa: F401,F403
from ._spectrum_ll import *  # noqa: F401,F403
from ._spectrum_hamiltonian import *  # noqa: F401,F403

def compute_coulomb_overlap(result: MagneticSpectrumResult, m: int, n: int) -> Array:
    """Compute one projected density-overlap matrix ``Λ_(m,n)``.

    This is a direct, readable implementation with the same layer/K-point
    convention as the author production routine ``computeCoulombOverlap_v2``.
    Use :func:`compute_coulomb_overlap_fast` for the symmetry-reduced version.
    """

    params = result.params
    p, q, nq = result.p, result.q, result.nq
    n_h = result.n_h
    inner = result.inner_dim
    n_sub = 2 * q
    theta_strain = np.angle(params.a2) - np.pi / 2.0
    tmp = np.zeros((n_sub, q, nq, nq, n_sub, q, nq, nq), dtype=np.complex128)
    lambda_psi = np.zeros((inner, inner), dtype=np.complex128)
    lambda_psi_view = lambda_psi.reshape((n_h, p, 2, n_h, p, 2), order="F")
    identity_sublattice = np.eye(2, dtype=np.complex128)

    for ip2 in range(nq):
        for ik2 in range(nq):
            for ip1 in range(nq):
                for rp1 in range(q):
                    for ik1 in range(nq):
                        for rk1 in range(q):
                            lambda_psi[:, :] = 0.0
                            k1 = result.lattice_k1[ik1 + rk1 * nq]
                            p1 = result.lattice_k1[ip1 + rp1 * nq]
                            for rk2 in range(p):
                                rp2 = (rk2 + int(n)) % p
                                s = -((rk2 + int(n) - rp2) // p)
                                k2 = result.lattice_k2[ik2 + rk2 * nq]
                                p2 = result.lattice_k2[ip2 + rp2 * nq]
                                k20 = result.lattice_k2[ik2]
                                p20 = result.lattice_k2[ip2]
                                q_lab = (p1 - k1 + int(m)) * params.g1 + (p20 - k20 + int(n) / q) * params.g2
                                q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
                                for layer in range(2):
                                    if result.valley == "K":
                                        kl = params.kb_point if layer == 0 else params.kt
                                    else:
                                        kl = params.kt if layer == 0 else params.kb_point
                                    theta_l = (2 * (layer + 1) - 3) * params.dtheta_rad / 2.0
                                    k2l = projector_para(params.g2, params.a2) * k2 - projector_para(kl, params.a2)
                                    expfactor = (
                                        np.exp(1j * 2.0 * np.pi * s * (p1 - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                                        * np.exp(1j * s * s / 2.0 * projector_para(result.q_phi, params.a1) * abs(params.a1))
                                        * np.exp(-1j * s * projector_norm(kl, params.a2) * projector_norm(params.a1, params.a2))
                                        * np.exp(1j * q_rot.real * k2l * result.l_b**2)
                                        * np.exp(1j * q_rot.real * q_rot.imag * result.l_b**2 / 2.0)
                                    )
                                    lambda_psi_view[:, rk2, layer, :, rp2, layer] = tll_matrix(
                                        identity_sublattice,
                                        q_rot,
                                        n_landau=result.n_landau,
                                        n_h=n_h,
                                        l_b=result.l_b,
                                        theta0=theta_strain,
                                        theta1=theta_l,
                                        theta2=theta_l,
                                        sigma_rotation=result.sigma_rotation,
                                        valley=result.valley,
                                    ) * expfactor
                            left = result.vec[:, :, rk1, ik1, ik2]
                            right = result.vec[:, :, rp1, ip1, ip2]
                            tmp[:, rk1, ik1, ik2, :, rp1, ip1, ip2] = left.conj().T @ lambda_psi @ right
    return tmp.reshape((n_sub * q * nq * nq, n_sub * q * nq * nq), order="F")


def _overlap_slice(
    result: MagneticSpectrumResult,
    m: int,
    n: int,
    *,
    ik2: int,
    ip2: int,
    ik1: int,
    ip1: int,
    rk1: int,
    rp1: int,
    tll_cache: dict[tuple[int, int, int], Array] | None = None,
) -> Array:
    params = result.params
    p, q, nq = result.p, result.q, result.nq
    n_h = result.n_h
    inner = result.inner_dim
    ik2_i = int(ik2)
    ip2_i = int(ip2)
    ik1_i = int(ik1)
    ip1_i = int(ip1)
    rk1_i = int(rk1)
    rp1_i = int(rp1)
    m_i = int(m)
    n_i = int(n)
    theta_strain = np.angle(params.a2) - np.pi / 2.0
    lambda_psi = np.zeros((inner, inner), dtype=np.complex128)
    lambda_psi_view = lambda_psi.reshape((n_h, p, 2, n_h, p, 2), order="F")
    identity_sublattice = np.eye(2, dtype=np.complex128)
    k1 = result.lattice_k1[ik1_i + rk1_i * nq]
    p1 = result.lattice_k1[ip1_i + rp1_i * nq]
    k20 = result.lattice_k2[ik2_i]
    p20 = result.lattice_k2[ip2_i]
    q_lab = (p1 - k1 + m_i) * params.g1 + (p20 - k20 + n_i / q) * params.g2
    q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
    # The lattice mesh is an arithmetic progression, so q0/mesh_shift cancel
    # from the momentum transfer.  Cache the expensive LL translation matrix by
    # this exact integer transfer and by layer-specific theta arguments.
    delta1_num = (ip1_i + rp1_i * nq) - (ik1_i + rk1_i * nq) + m_i * nq * q
    delta2_num = (ip2_i - ik2_i) + n_i * nq
    for rk2 in range(p):
        rp2 = (rk2 + n_i) % p
        s = -((rk2 + n_i - rp2) // p)
        k2 = result.lattice_k2[ik2_i + rk2 * nq]
        p2 = result.lattice_k2[ip2_i + rp2 * nq]
        for layer in range(2):
            if result.valley == "K":
                kl = params.kb_point if layer == 0 else params.kt
            else:
                kl = params.kt if layer == 0 else params.kb_point
            theta_l = (2 * (layer + 1) - 3) * params.dtheta_rad / 2.0
            k2l = projector_para(params.g2, params.a2) * k2 - projector_para(kl, params.a2)
            expfactor = (
                np.exp(1j * 2.0 * np.pi * s * (p1 - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                * np.exp(1j * s * s / 2.0 * projector_para(result.q_phi, params.a1) * abs(params.a1))
                * np.exp(-1j * s * projector_norm(kl, params.a2) * projector_norm(params.a1, params.a2))
                * np.exp(1j * q_rot.real * k2l * result.l_b**2)
                * np.exp(1j * q_rot.real * q_rot.imag * result.l_b**2 / 2.0)
            )
            cache_key = (delta1_num, delta2_num, layer)
            if tll_cache is None:
                tll = tll_matrix(
                    identity_sublattice,
                    q_rot,
                    n_landau=result.n_landau,
                    n_h=n_h,
                    l_b=result.l_b,
                    theta0=theta_strain,
                    theta1=theta_l,
                    theta2=theta_l,
                    sigma_rotation=result.sigma_rotation,
                    valley=result.valley,
                )
            else:
                tll = tll_cache.get(cache_key)
                if tll is None:
                    tll = tll_matrix(
                        identity_sublattice,
                        q_rot,
                        n_landau=result.n_landau,
                        n_h=n_h,
                        l_b=result.l_b,
                        theta0=theta_strain,
                        theta1=theta_l,
                        theta2=theta_l,
                        sigma_rotation=result.sigma_rotation,
                        valley=result.valley,
                    )
                    tll_cache[cache_key] = tll
            lambda_psi_view[:, rk2, layer, :, rp2, layer] = tll * expfactor
    left = result.vec[:, :, rk1_i, ik1_i, ik2_i]
    right = result.vec[:, :, rp1_i, ip1_i, ip2_i]
    return left.conj().T @ lambda_psi @ right


def compute_coulomb_overlap_fast(result: MagneticSpectrumResult, m: int, n: int) -> Array:
    """Symmetry-reduced port of author ``computeCoulombOverlap_v2``.

    The result has the same flattened shape as :func:`compute_coulomb_overlap`,
    but only explicitly computes the first row/column of magnetic-translation
    strip blocks and reconstructs the rest with the author phase factors.
    """

    p, q, nq = result.p, result.q, result.nq
    n_sub = 2 * q
    tmp = np.zeros((n_sub, q, nq, nq, n_sub, q, nq, nq), dtype=np.complex128)
    ips = np.asarray([(rp * p) % q for rp in range(q)], dtype=int)
    tll_cache: dict[tuple[int, int, int], Array] = {}
    for ip2 in range(nq):
        for ik2 in range(nq):
            for ip1 in range(nq):
                for rp1 in range(q):
                    for ik1 in range(nq):
                        tmp[:, 0, ik1, ik2, :, rp1, ip1, ip2] = _overlap_slice(
                            result, m, n, ik2=ik2, ip2=ip2, ik1=ik1, ip1=ip1, rk1=0, rp1=rp1, tll_cache=tll_cache
                        )
            for ip1 in range(nq):
                for ik1 in range(nq):
                    for rk1 in range(1, q):
                        tmp[:, rk1, ik1, ik2, :, 0, ip1, ip2] = _overlap_slice(
                            result, m, n, ik2=ik2, ip2=ip2, ik1=ik1, ip1=ip1, rk1=rk1, rp1=0, tll_cache=tll_cache
                        )

            for rp1 in range(1, q):
                for rk1 in range(1, q):
                    lhs_rk = int(ips[rk1])
                    lhs_rp = int(ips[rp1])
                    if ips[rp1] > ips[rk1]:
                        source_rp = int(ips[rp1] - ips[rk1])
                        tmp[:, lhs_rk, :, ik2, :, lhs_rp, :, ip2] = tmp[:, 0, :, ik2, :, source_rp, :, ip2] * np.exp(
                            -1j * 2.0 * np.pi * rk1 * int(n) / q
                        )
                    else:
                        source_rk = int(ips[rk1] - ips[rp1])
                        tmp[:, lhs_rk, :, ik2, :, lhs_rp, :, ip2] = tmp[:, source_rk, :, ik2, :, 0, :, ip2] * np.exp(
                            -1j * 2.0 * np.pi * rp1 * int(n) / q
                        )
    return tmp.reshape((n_sub * q * nq * nq, n_sub * q * nq * nq), order="F")

__all__ = [name for name in globals() if not name.startswith('__')]
