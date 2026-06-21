from __future__ import annotations

from ._spectrum_shared import *  # noqa: F401,F403
from ._spectrum_params import *  # noqa: F401,F403
from ._spectrum_ll import *  # noqa: F401,F403

def magnetic_lattice_coordinates(nq: int, q: int, *, valley: Valley = "K", q0: complex = 0.0 + 0.0j, mesh_shift: float = 0.0) -> tuple[Array, Array]:
    """Return author-code lattice coordinates for ``lk=nq*q``."""

    lk = int(nq) * int(q)
    k1 = np.arange(lk, dtype=float) / float(lk) + float(mesh_shift)
    k2 = np.arange(lk, dtype=float) / float(lk) + float(mesh_shift)
    if valley == "Kprime":
        k1 = k1 + float(np.real(q0)) / float(lk)
        k2 = k2 + float(np.imag(q0)) / float(lk)
    return k1, k2


def qjs_for_valley(valley: Valley) -> tuple[complex, complex, complex]:
    return (0.0 + 0.0j, 0.0 + 1.0j, 1.0 + 1.0j) if valley == "K" else (0.0 + 0.0j, 0.0 - 1.0j, -1.0 - 1.0j)


def construct_ll_hamiltonian(
    params: FiniteFieldBMParameters,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nq: int,
    valley: Valley = "K",
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
    mesh_shift: float = 0.0,
) -> tuple[Array, Array, Array, Array, float, complex]:
    """Construct the LL Hamiltonian and ``Σz`` operator before projection."""

    p, q = int(flux.p), int(flux.q)
    n_h = int(n_landau) * 2 - 1
    k1, k2 = magnetic_lattice_coordinates(nq, q, valley=valley, q0=q0, mesh_shift=mesh_shift)
    q_phi = 2.0 * np.pi / abs(params.a2) ** 2 * params.a2 * flux.ratio
    l_b = float(np.sqrt(q / (2.0 * np.pi * abs(p)) * params.area))
    h = np.zeros((n_h, p, 2, n_h, p, 2, int(nq), int(nq)), dtype=np.complex128)

    eps_b = params.vf / l_b
    for ih in range(n_h):
        n, gamma = in_gamma(ih)
        for ip in range(p):
            h[ih, ip, 0, ih, ip, 0, :, :] += gamma * np.sqrt(2.0 * n) * eps_b
            h[ih, ip, 1, ih, ip, 1, :, :] += gamma * np.sqrt(2.0 * n) * eps_b

    if hbn:
        sign_hbn = 1.0 if valley == "K" else -1.0
        for ip in range(p):
            h[0, ip, 0, 0, ip, 0, :, :] += -params.delta * sign_hbn
            for n in range(1, int(n_landau)):
                for ig1 in range(2):
                    gamma1 = 2 * (ig1 + 1) - 3
                    ih1 = (n - 1) * 2 + ig1 + 1
                    for ig2 in range(2):
                        gamma2 = 2 * (ig2 + 1) - 3
                        ih2 = (n - 1) * 2 + ig2 + 1
                        h[ih1, ip, 0, ih2, ip, 0, :, :] += -(1.0 - gamma1 * gamma2) / 2.0 * params.delta * sign_hbn

    if include_strain:
        def_pot = params.deformation_potential * (params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0
        for ih in range(n_h):
            for ip in range(p):
                h[ih, ip, 0, ih, ip, 0, :, :] -= def_pot
                h[ih, ip, 1, ih, ip, 1, :, :] += def_pot

    theta_strain = np.angle(params.a2) - np.pi / 2.0
    tunnels = (params.t0, params.t1, params.t2)
    for qj, tunnel in zip(qjs_for_valley(valley), tunnels, strict=True):
        q_lab = qj.real * params.g1 + qj.imag * params.g2
        q_lab = q_lab + (params.kb_point - params.kt if valley == "K" else params.kt - params.kb_point)
        q_rot = projector_norm(q_lab, params.a2) + 1j * projector_para(q_lab, params.a2)
        tmat = tll_matrix(
            tunnel if valley == "K" else np.conj(tunnel),
            q_rot,
            n_landau=int(n_landau),
            n_h=n_h,
            l_b=l_b,
            theta0=theta_strain,
            theta1=-params.dtheta_rad / 2.0,
            theta2=params.dtheta_rad / 2.0,
            sigma_rotation=sigma_rotation,
            valley=valley,
        )
        kl = params.kb_point if valley == "K" else -params.kb_point
        kr = params.kt if valley == "K" else -params.kt
        for ik2 in range(int(nq)):
            for r1 in range(p):
                k2l = projector_para(params.g2, params.a2) * k2[r1 * int(nq) + ik2] - projector_para(kl, params.a2)
                raw_r2 = r1 + q * int(qj.imag)
                r2 = raw_r2 % p
                s = -((raw_r2 - r2) // p)
                p2 = k2[r2 * int(nq) + ik2]
                for ik1 in range(int(nq)):
                    expfactor = (
                        np.exp(1j * 2.0 * np.pi * s * (k1[ik1] - p2 * projector_para(params.a1, params.a2) / abs(params.a2)))
                        * np.exp(1j * s * s / 2.0 * projector_para(q_phi, params.a1) * abs(params.a1))
                        * np.exp(-1j * s * projector_norm(kr, params.a2) * projector_norm(params.a1, params.a2))
                        * np.exp(1j * q_rot.real * k2l * l_b**2)
                        * np.exp(1j * q_rot.real * q_rot.imag * l_b**2 / 2.0)
                    )
                    h[:, r1, 0, :, r2, 1, ik1, ik2] += expfactor * tmat

    sigma_z = construct_sigma_z_ll(n_landau=int(n_landau), p=p, valley=valley)
    return h, sigma_z, k1, k2, l_b, q_phi


def construct_sigma_z_ll(*, n_landau: int, p: int, valley: Valley = "K") -> Array:
    n_h = int(n_landau) * 2 - 1
    sigma = np.zeros((n_h, int(p), 2, n_h, int(p), 2), dtype=np.complex128)
    sign = 1.0 if valley == "K" else -1.0
    for ip in range(int(p)):
        sigma[0, ip, 0, 0, ip, 0] = -sign
        sigma[0, ip, 1, 0, ip, 1] = -sign
        for n in range(1, int(n_landau)):
            for ig1 in range(2):
                gamma1 = 2 * (ig1 + 1) - 3
                ih1 = (n - 1) * 2 + ig1 + 1
                for ig2 in range(2):
                    gamma2 = 2 * (ig2 + 1) - 3
                    ih2 = (n - 1) * 2 + ig2 + 1
                    value = -(1.0 - gamma1 * gamma2) / 2.0 * sign
                    sigma[ih1, ip, 0, ih2, ip, 0] = value
                    sigma[ih1, ip, 1, ih2, ip, 1] = value
    return sigma


def _matrix_from_ll_tensor(tensor: Array, i1: int, i2: int) -> Array:
    block = np.asarray(tensor[..., int(i1), int(i2)], dtype=np.complex128)
    dim = int(np.prod(block.shape[:3]))
    return block.reshape((dim, dim), order="F")


def _hermitian_from_upper(mat: Array) -> Array:
    """Return the Hermitian matrix represented by the upper triangle of ``mat``.

    Julia's ``Hermitian(H, :U)`` keeps the diagonal once and mirrors only the
    strict upper triangle.  Do the same here; adding the diagonal twice would
    double the LL kinetic energies.
    """

    return np.triu(mat) + np.triu(mat, 1).conj().T


def generate_magnetic_translation_orbit(vec: Array, *, q: int, p: int, nq: int) -> None:
    """Fill the ``r=1..q-1`` magnetic-translation orbit in-place."""

    base = vec[:, :, 0, :, :].reshape((-1, int(p), 2, 2 * int(q), int(nq), int(nq)), order="F")
    rk2 = np.arange(int(p), dtype=float)
    orbit_positions = magnetic_r_orbit_positions(p, q)
    inverse_orbit = np.empty(int(q), dtype=int)
    inverse_orbit[orbit_positions] = np.arange(int(q), dtype=int)
    for r1 in range(1, int(q)):
        phase = np.exp(-1j * 2.0 * np.pi * int(inverse_orbit[r1]) * (rk2 / float(q)))
        shifted = base * phase.reshape((1, int(p), 1, 1, 1, 1))
        vec[:, :, r1, :, :] = shifted.reshape((vec.shape[0], 2 * int(q), int(nq), int(nq)), order="F")


def compute_magnetic_spectrum(
    params: FiniteFieldBMParameters,
    *,
    flux: MagneticFlux,
    n_landau: int,
    nq: int,
    valley: Valley = "K",
    sigma_rotation: bool = False,
    hbn: bool = False,
    include_strain: bool = True,
    q0: complex = 0.0 + 0.0j,
    mesh_shift: float = 0.0,
) -> MagneticSpectrumResult:
    """Construct and diagonalize the central finite-B Hofstadter subbands."""

    h, sigma_z_ll, k1, k2, l_b, q_phi = construct_ll_hamiltonian(
        params,
        flux=flux,
        n_landau=n_landau,
        nq=nq,
        valley=valley,
        sigma_rotation=sigma_rotation,
        hbn=hbn,
        include_strain=include_strain,
        q0=q0,
        mesh_shift=mesh_shift,
    )
    p, q = flux.p, flux.q
    n_h = int(n_landau) * 2 - 1
    inner = 2 * n_h * p
    n_sub = 2 * q
    spectrum = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    vec = np.zeros((inner, n_sub, q, int(nq), int(nq)), dtype=np.complex128)
    p_sigma_z = np.zeros((n_sub, n_sub, int(nq), int(nq)), dtype=np.complex128)
    sigma_z_eigs = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    sigma_z_energy = np.zeros((n_sub, int(nq), int(nq)), dtype=float)
    sigma_mat = sigma_z_ll.reshape((inner, inner), order="F")
    start = n_h * p - q
    stop = n_h * p + q - 1
    if start < 0 or stop >= inner:
        raise ValueError(f"Central subband window [{start},{stop}] outside inner dimension {inner}; increase n_landau")

    for i2 in range(int(nq)):
        for i1 in range(int(nq)):
            hmat_upper = _matrix_from_ll_tensor(h, i1, i2)
            hmat = _hermitian_from_upper(hmat_upper)
            vals, evecs = eigh(hmat, subset_by_index=(start, stop), check_finite=False)
            order = np.argsort(vals.real, kind="stable")
            vals = vals[order].real
            evecs = evecs[:, order]
            spectrum[:, i1, i2] = vals
            vec[:, :, 0, i1, i2] = evecs
            psigma = evecs.conj().T @ sigma_mat @ evecs
            p_sigma_z[:, :, i1, i2] = psigma
            sigma_z_energy[:, i1, i2] = np.diag(psigma).real
            sigma_z_eigs[:, i1, i2] = np.linalg.eigvalsh((psigma + psigma.conj().T) / 2.0).real
    generate_magnetic_translation_orbit(vec, q=q, p=p, nq=int(nq))
    return MagneticSpectrumResult(
        params=params,
        flux=flux,
        valley=valley,
        n_landau=int(n_landau),
        n_h=n_h,
        nq=int(nq),
        sigma_rotation=bool(sigma_rotation),
        l_b=float(l_b),
        q_phi=complex(q_phi),
        lattice_k1=np.asarray(k1, dtype=float),
        lattice_k2=np.asarray(k2, dtype=float),
        hamiltonian_ll=h,
        sigma_z_ll=sigma_z_ll,
        spectrum=spectrum,
        vec=vec,
        p_sigma_z=p_sigma_z,
        sigma_z_eigenvalues=sigma_z_eigs,
        sigma_z_energy_diag=sigma_z_energy,
    )

__all__ = [name for name in globals() if not name.startswith('__')]
