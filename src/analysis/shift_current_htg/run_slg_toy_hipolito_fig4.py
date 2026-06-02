from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .response import fermi_occupation, precompute_response_tensors, velocity_matrices
from .slg_toy import (
    GappedSLGParams,
    _nearest_reciprocal_vectors,
    d2hdk,
    dhdk,
    diagonalize,
    hex_bz_vertices,
)

FIG4_COMPONENTS = {
    "sigma^{y;yy}": (1, 1, 1, +1.0),
    "-sigma^{y;xx}": (1, 0, 0, -1.0),
    "-sigma^{x;xy}": (0, 0, 1, -1.0),
    "-sigma^{x;yx}": (0, 1, 0, -1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hipolito/Pedersen/Pereira PRB 94, 045434 Fig. 4 reproduction for "
            "the gapped-graphene photoconductivity benchmark.  The calculation "
            "implements the two-band Eq. (25b) interband-intraband term; for the "
            "neutral TR-symmetric two-band monolayer, the paper states the ee/ei/ii "
            "terms vanish for the Fig. 4 tensor components."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/shift_current_htg/crossref_hipolito2016_fig4_reproduction"))
    parser.add_argument("--gamma-mev", type=float, default=1.0, help="Hipolito broadening Gamma in meV")
    parser.add_argument("--delta-ev", type=float, default=0.2, help="Gap Delta in eV")
    parser.add_argument("--hopping-ev", type=float, default=3.0, help="gamma0 in eV")
    parser.add_argument("--temperature-k", type=float, default=1.0)
    parser.add_argument("--mu-ev", type=float, default=0.0)
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=0.8)
    parser.add_argument("--n-energy", type=int, default=321)
    parser.add_argument("--patch-radius-nm-inv", type=float, default=1.4)
    parser.add_argument("--mesh-radial", type=int, default=180)
    parser.add_argument("--mesh-angular", type=int, default=120)
    parser.add_argument("--radial-power", type=float, default=2.0, help="r=R*s^p clusters quadrature near K/K'")
    parser.add_argument("--fd-step-nm-inv", type=float, default=1.0e-5)
    parser.add_argument(
        "--derivative-method",
        choices=("analytic", "finite-difference"),
        default="analytic",
        help="Use analytic covariant derivative from response.py, or the older finite-difference derivative diagnostic.",
    )
    parser.add_argument(
        "--normalization",
        choices=("eq25", "eq31"),
        default="eq31",
        help=(
            "eq25 uses the direct prefactor g*gamma0/a from Eq. (25b) in our k units. "
            "eq31 additionally fixes the overall convention to Hipolito's analytic "
            "K-point threshold Re sigma/sigma2=-1/(4 Delta/gamma0)."
        ),
    )
    parser.add_argument("--eq31-reference-offset-ev", type=float, default=0.03)
    parser.add_argument("--reference-page", type=Path, default=Path("tmp/pdfs/hipolito2016/render/page-07.png"))
    return parser.parse_args()


def align_to_reference(evecs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.array(evecs, dtype=np.complex128, copy=True)
    for band in range(out.shape[1]):
        overlap = np.vdot(reference[:, band], out[:, band])
        if abs(overlap) > 1.0e-14:
            out[:, band] *= np.conj(overlap) / abs(overlap)
    return out


def diagonalize_aligned(k_xy: np.ndarray, params: GappedSLGParams, reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = diagonalize(k_xy, params)
    if reference is not None:
        evecs = align_to_reference(evecs, reference)
    D = velocity_matrices(evecs, dhdk(k_xy, params))
    return evals, evecs, D


def in_first_hex_bz(k_xy: np.ndarray, nearest_g: np.ndarray) -> bool:
    bounds = 0.5 * np.sum(nearest_g * nearest_g, axis=1)
    return bool(np.all(np.asarray(k_xy, dtype=float) @ nearest_g.T <= bounds + 1.0e-10))


def k_corner_patch_grid(
    params: GappedSLGParams,
    *,
    radius: float,
    n_radial: int,
    n_angular: int,
    radial_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = hex_bz_vertices(params)
    nearest_g = _nearest_reciprocal_vectors(params)
    radius = float(radius)
    p = float(radial_power)
    if radius <= 0.0 or p <= 0.0:
        raise ValueError("radius and radial_power must be positive")
    points: list[np.ndarray] = []
    weights: list[float] = []
    dtheta = 2.0 * math.pi / float(n_angular)
    ds = 1.0 / float(n_radial)
    for vertex in vertices:
        for ir in range(int(n_radial)):
            s = (ir + 0.5) * ds
            rr = radius * s**p
            dr_ds = radius * p * s ** (p - 1.0)
            dr = dr_ds * ds
            for it in range(int(n_angular)):
                theta = (it + 0.5) * dtheta
                k_xy = vertex + rr * np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
                if not in_first_hex_bz(k_xy, nearest_g):
                    continue
                points.append(np.asarray(k_xy, dtype=float))
                weights.append(float(rr * dr * dtheta))
    if not points:
        raise RuntimeError("K/K' patch quadrature selected no k points")
    return np.asarray(points, dtype=float), np.asarray(weights, dtype=float)


def f_factor(
    k_xy: np.ndarray,
    *,
    params: GappedSLGParams,
    reference: np.ndarray,
    alpha: int,
    m: int,
    n: int,
    photon_energies_ev: np.ndarray,
    gamma_ev: float,
    mu_ev: float,
    temperature_k: float,
) -> np.ndarray:
    evals, _evecs, D = diagonalize_aligned(k_xy, params, reference)
    occ = fermi_occupation(evals, mu_ev=mu_ev, temperature_k=temperature_k)
    delta_mn = float(evals[m] - evals[n])
    if abs(delta_mn) < 1.0e-14:
        return np.zeros_like(photon_energies_ev, dtype=np.complex128)
    # Hipolito notation: f_nm = f_n - f_m.
    f_nm = float(occ[n] - occ[m])
    return D[alpha, m, n] * f_nm / delta_mn / (photon_energies_ev - delta_mn + 1.0j * gamma_ev)


def hipolito_eq25b_raw_component_finite_difference(
    photon_energies_ev: np.ndarray,
    *,
    params: GappedSLGParams,
    component: tuple[int, int, int],
    gamma_ev: float,
    mu_ev: float,
    temperature_k: float,
    k_points: np.ndarray,
    k_weights: np.ndarray,
    fd_step: float,
) -> np.ndarray:
    """Older diagnostic implementation of the covariant derivative.

    This is intentionally retained as a regression test: with Gamma=1 meV, the
    resonant denominator amplifies tiny finite-difference/gauge-alignment errors
    into visible wiggles.  The production path below uses the analytic
    generalized derivative instead.
    """

    lam, alpha, beta = component
    unit = np.zeros(2, dtype=float)
    unit[beta] = 1.0
    out = np.zeros_like(photon_energies_ev, dtype=np.complex128)
    for k_xy, weight in zip(k_points, k_weights, strict=True):
        evals, evecs, D = diagonalize_aligned(k_xy, params, None)
        occ = fermi_occupation(evals, mu_ev=mu_ev, temperature_k=temperature_k)
        for m in range(evals.size):
            for n in range(evals.size):
                if m == n:
                    continue
                delta_mn = float(evals[m] - evals[n])
                f_nm = float(occ[n] - occ[m])
                if abs(delta_mn) < 1.0e-14 or abs(f_nm) < 1.0e-14:
                    continue
                fp = f_factor(
                    k_xy + float(fd_step) * unit,
                    params=params,
                    reference=evecs,
                    alpha=alpha,
                    m=m,
                    n=n,
                    photon_energies_ev=photon_energies_ev,
                    gamma_ev=gamma_ev,
                    mu_ev=mu_ev,
                    temperature_k=temperature_k,
                )
                fm = f_factor(
                    k_xy - float(fd_step) * unit,
                    params=params,
                    reference=evecs,
                    alpha=alpha,
                    m=m,
                    n=n,
                    photon_energies_ev=photon_energies_ev,
                    gamma_ev=gamma_ev,
                    mu_ev=mu_ev,
                    temperature_k=temperature_k,
                )
                cov_derivative = (fp - fm) / (2.0 * float(fd_step))
                out += (
                    float(weight)
                    / (2.0 * math.pi) ** 2
                    * (-D[lam, n, m])
                    / (-delta_mn + 2.0j * float(gamma_ev))
                    * cov_derivative
                )
    return out


def hipolito_eq25b_raw_component_analytic(
    photon_energies_ev: np.ndarray,
    *,
    params: GappedSLGParams,
    component: tuple[int, int, int],
    gamma_ev: float,
    mu_ev: float,
    temperature_k: float,
    k_points: np.ndarray,
    k_weights: np.ndarray,
) -> np.ndarray:
    """Eq. (25b) with analytic gauge-covariant derivative.

    For

        F_mn = D^alpha_mn f_nm / Delta_mn / (E - Delta_mn + i Gamma),

    and constant occupation in the insulating gap,

        (F_mn)_;beta = f_nm [ i r^alpha_{mn;beta}/den
            + i r^alpha_mn (partial_beta Delta_mn)/den^2 ].

    The required ``r_;`` is exactly the W-corrected generalized derivative in
    ``response.py``.  This removes the finite-difference wiggles that were
    visible in the first Fig. 4 reproduction attempt.
    """

    lam, alpha, beta = component
    out = np.zeros_like(photon_energies_ev, dtype=np.complex128)
    for k_xy, weight in zip(k_points, k_weights, strict=True):
        evals, evecs = diagonalize(k_xy, params)
        tensors = precompute_response_tensors(
            evals,
            evecs,
            dhdk(k_xy, params),
            d2hdk=d2hdk(k_xy, params),
            mu_ev=mu_ev,
            temperature_k=temperature_k,
            denominator_cutoff_ev=1.0e-10,
        )
        D = tensors.D
        r = tensors.r
        r_covariant = tensors.r_covariant
        occ = tensors.occupations
        for m in range(evals.size):
            for n in range(evals.size):
                if m == n:
                    continue
                delta_mn = float(evals[m] - evals[n])
                f_nm = float(occ[n] - occ[m])
                if abs(delta_mn) < 1.0e-14 or abs(f_nm) < 1.0e-14:
                    continue
                den = photon_energies_ev - delta_mn + 1.0j * float(gamma_ev)
                partial_beta_delta = D[beta, m, m] - D[beta, n, n]
                cov_derivative = f_nm * (
                    1.0j * r_covariant[beta, alpha, m, n] / den
                    + 1.0j * r[alpha, m, n] * partial_beta_delta / (den * den)
                )
                out += (
                    float(weight)
                    / (2.0 * math.pi) ** 2
                    * (-D[lam, n, m])
                    / (-delta_mn + 2.0j * float(gamma_ev))
                    * cov_derivative
                )
    return out


def crop_reference_fig4(reference_page: Path, output: Path) -> bool:
    if not reference_page.exists():
        return False
    image = Image.open(reference_page)
    # Crop the upper-left Fig. 4 panel from the rendered page-07.png.  The crop
    # is included only as a visual reference; numerical evidence comes from the
    # generated data/summary.
    width, height = image.size
    left = int(0.08 * width)
    upper = int(0.07 * height)
    right = int(0.49 * width)
    lower = int(0.36 * height)
    crop = image.crop((left, upper, right, lower))
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    gamma_ev = float(args.gamma_mev) * 1.0e-3
    delta_ev = float(args.delta_ev)
    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=0.5 * delta_ev)
    k_points, k_weights = k_corner_patch_grid(
        params,
        radius=float(args.patch_radius_nm_inv),
        n_radial=int(args.mesh_radial),
        n_angular=int(args.mesh_angular),
        radial_power=float(args.radial_power),
    )

    raw: dict[str, np.ndarray] = {}
    for label, (lam, alpha, beta, _sign) in FIG4_COMPONENTS.items():
        if str(args.derivative_method) == "analytic":
            raw[label] = hipolito_eq25b_raw_component_analytic(
                photon,
                params=params,
                component=(lam, alpha, beta),
                gamma_ev=gamma_ev,
                mu_ev=float(args.mu_ev),
                temperature_k=float(args.temperature_k),
                k_points=k_points,
                k_weights=k_weights,
            )
        else:
            raw[label] = hipolito_eq25b_raw_component_finite_difference(
                photon,
                params=params,
                component=(lam, alpha, beta),
                gamma_ev=gamma_ev,
                mu_ev=float(args.mu_ev),
                temperature_k=float(args.temperature_k),
                k_points=k_points,
                k_weights=k_weights,
                fd_step=float(args.fd_step_nm_inv),
            )

    # Eq. (25b) has a prefactor g*gamma0/a after replacing velocities by
    # D=hbar*v.  The minus sign fixes the current/component convention of the
    # present A/B basis to Hipolito's plotted sigma_222 convention.
    spin_g = 2.0
    eq25_scale = -spin_g * float(params.hopping_ev) / float(params.bond_nm)
    scale = eq25_scale
    normalization_note = "direct Eq.(25b) prefactor in current k/basis convention"
    if str(args.normalization) == "eq31":
        # Hipolito Eq. (31) uses Delta in units of gamma0.  We use it only to
        # fix the remaining global convention factor between the present TB
        # basis/component convention and Hipolito's plotted sigma_222/sigma2.
        reference_energy = delta_ev + float(args.eq31_reference_offset_ev)
        idx = int(np.argmin(np.abs(photon - reference_energy)))
        delta_dimensionless = delta_ev / float(params.hopping_ev)
        target = -1.0 / (4.0 * delta_dimensionless)
        scaled_reference = (eq25_scale * raw["sigma^{y;yy}"])[idx].real
        if abs(scaled_reference) > 1.0e-14:
            scale = eq25_scale * target / scaled_reference
        normalization_note = (
            "Eq.(25b) shape with global convention fixed by Hipolito Eq.(31) "
            f"at E=Delta+{float(args.eq31_reference_offset_ev):g} eV"
        )

    curves = {label: sign * scale * raw[label] for label, (*_abc, sign) in FIG4_COMPONENTS.items()}
    reference = curves["sigma^{y;yy}"]
    overlap_error = float(max(np.max(np.abs(curves[label] - reference)) for label in curves))

    data_path = output_dir / "hipolito2016_fig4_reproduction_data.npz"
    np.savez(
        data_path,
        photon_energies_ev=photon,
        **{f"raw_{i}": values for i, values in enumerate(raw.values())},
        **{f"curve_{i}": values for i, values in enumerate(curves.values())},
        labels=np.asarray(list(curves.keys()), dtype=object),
    )

    reference_crop = output_dir / "hipolito2016_fig4_reference_crop.png"
    have_crop = crop_reference_fig4(args.reference_page, reference_crop)

    fig, axes = plt.subplots(1, 2 if have_crop else 1, figsize=(10.5 if have_crop else 5.3, 4.0), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    ax.plot(photon, reference.real, lw=1.8, color="#003c4c", label="Re")
    ax.plot(photon, reference.imag, lw=1.4, color="#e41a1c", label="Im")
    delta_dimensionless = delta_ev / float(params.hopping_ev)
    eq31_level = -1.0 / (4.0 * delta_dimensionless)
    ax.plot(
        [delta_ev, min(float(args.emax), delta_ev + 0.18)],
        [eq31_level, eq31_level],
        color="#003c4c",
        ls="--",
        lw=1.0,
        alpha=0.8,
        label="Eq.(31) threshold",
    )
    ax.axvline(delta_ev, color="0.35", ls="--", lw=1.0, label=r"$\Delta$")
    ax.axhline(0.0, color="0.5", lw=0.7)
    ax.set_xlim(float(args.emin), float(args.emax))
    ax.set_xlabel(r"$\hbar\omega$ [eV]")
    ax.set_ylabel(r"$\sigma^{(2)}_{dc}/\sigma_2$")
    ax.set_title("Local reproduction of Hipolito Fig. 4")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.legend(frameon=False, fontsize=7)

    if have_crop:
        ax_ref = axes[1]
        ax_ref.imshow(Image.open(reference_crop))
        ax_ref.axis("off")
        ax_ref.set_title("Hipolito 2016 Fig. 4 crop")

    fig.savefig(output_dir / "hipolito2016_fig4_reproduction.png", dpi=180)
    fig.savefig(output_dir / "hipolito2016_fig4_reproduction.pdf")
    plt.close(fig)

    summary = {
        "reference": "Hipolito, Pedersen, Pereira, PRB 94, 045434 (2016), Fig. 4",
        "method": "two-band Eq.(25b) interband-intraband term; ee term vanishes for two bands and ei/ii vanish for neutral TR-symmetric monolayer components as stated in the paper",
        "derivative_method": str(args.derivative_method),
        "normalization": str(args.normalization),
        "normalization_note": normalization_note,
        "eq25_scale_before_eq31_convention_fix": float(eq25_scale),
        "final_scale_applied_to_raw_integral": float(scale),
        "parameters": {
            "gamma0_ev": float(params.hopping_ev),
            "Delta_ev": float(delta_ev),
            "mass_ev": float(params.mass_ev),
            "Gamma_mev": float(args.gamma_mev),
            "mu_ev": float(args.mu_ev),
            "temperature_k": float(args.temperature_k),
            "patch_radius_nm_inv": float(args.patch_radius_nm_inv),
            "mesh_radial": int(args.mesh_radial),
            "mesh_angular": int(args.mesh_angular),
            "radial_power": float(args.radial_power),
            "fd_step_nm_inv": float(args.fd_step_nm_inv),
            "n_k_points": int(k_points.shape[0]),
            "patch_area_nm_inv_sq": float(np.sum(k_weights)),
        },
        "component_overlap_max_abs_sigma_over_sigma2": overlap_error,
        "outputs": {
            "png": str(output_dir / "hipolito2016_fig4_reproduction.png"),
            "pdf": str(output_dir / "hipolito2016_fig4_reproduction.pdf"),
            "data": str(data_path),
            "reference_crop": str(reference_crop) if have_crop else None,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_dir / "hipolito2016_fig4_reproduction.png")


if __name__ == "__main__":
    main()
