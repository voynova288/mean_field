from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .constants import SHIFT_CURRENT_PREFAC_UA_NM_PER_V2, eta_mev_to_ev
from .response import (
    add_transitions_to_integral,
    fermi_occupation,
    lorentzian_delta,
    parse_component,
    positive_transition_terms,
    precompute_response_tensors,
    second_derivative_matrices,
    sigma_from_integral,
    velocity_matrices,
)
from .slg_toy import GappedSLGParams, d2hdk, dhdk, diagonalize, hex_bz_grid

DEFAULT_COMPONENTS = ("x;xx", "x;xy", "x;yx", "x;yy", "y;xx", "y;xy", "y;yx", "y;yy")


def rotate_c3(k_xy: np.ndarray, multiple: int) -> np.ndarray:
    theta = 2.0 * math.pi * int(multiple) / 3.0
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([c * k_xy[0] - s * k_xy[1], s * k_xy[0] + c * k_xy[1]], dtype=float)


def wannier90_internal_imn(
    energies_ev: np.ndarray,
    evecs: np.ndarray,
    k_xy: np.ndarray,
    params: GappedSLGParams,
    *,
    sc_eta_ev: float,
) -> np.ndarray:
    """Return the Wannier90/WannierBerri internal shift-current integrand.

    This is a direct two-dimensional transcription of the internal-term part of
    official reference implementations:

    - Wannier90 `src/postw90/berry.F90::berry_get_sc_klist`, equations around
      `gen_r_nm` and `I_nm` for shift current.
    - WannierBerri `wannierberri/calculators/dynamic.py::ShiftCurrentFormula`.

    For the present orthogonal two-band nearest-neighbor SLG model in the TB
    phase convention, the external position-operator matrix `AA_R` is zero after
    subtracting Wannier centers, so this reference expression should match the
    local gauge-free `response.py` implementation.  It is an audit, not an
    alternative fitted plotting path.

    Output shape is `Imn[n, m, a, b, c]`, i.e. the real integrand multiplying
    the occupation difference and the spectral delta function.
    """

    D = velocity_matrices(evecs, dhdk(k_xy, params))  # [axis, n, m]
    W = second_derivative_matrices(evecs, d2hdk(k_xy, params))  # [axis_a, axis_b, n, m]

    # Reference-code index convention: V[n,m,a], del2E[n,m,c,a].
    V = np.transpose(D, (1, 2, 0))
    del2e = np.transpose(W, (2, 3, 0, 1))

    d_eig = energies_ev[:, None] - energies_ev[None, :]
    inv = np.zeros_like(d_eig, dtype=float)
    mask = np.abs(d_eig) > 1.0e-10
    inv[mask] = 1.0 / d_eig[mask]

    d_h_no_eta = -V * inv[:, :, None]
    eta = float(sc_eta_ev)
    d_h_pvalue = -V * (d_eig / (d_eig * d_eig + eta * eta))[:, :, None]

    # WannierBerri variable names: sum_HD and DV_bit.  The two terms are the
    # compact vectorized form of the corresponding Wannier90 loops after the
    # diagonal subtractions are made explicit.
    sum_hd = (
        np.einsum("nlc,lma->nmca", V, d_h_pvalue, optimize=True)
        - np.einsum("nnc,nma->nmca", V, d_h_pvalue, optimize=True)
        - np.einsum("nla,lmc->nmca", d_h_pvalue, V, optimize=True)
        + np.einsum("nma,mmc->nmca", d_h_pvalue, V, optimize=True)
    )
    dv_bit = (
        np.einsum("nmc,nna->nmca", d_h_no_eta, V, optimize=True)
        - np.einsum("nmc,mma->nmca", d_h_no_eta, V, optimize=True)
        + np.einsum("nma,nnc->nmac", d_h_no_eta, V, optimize=True)
        - np.einsum("nma,mmc->nmac", d_h_no_eta, V, optimize=True)
    )
    a_gen_der = 1.0j * (del2e + sum_hd + dv_bit) * inv[:, :, None, None]
    a_h = 1.0j * d_h_no_eta

    imn = -np.einsum("nmca,mnb->nmabc", a_gen_der, a_h, optimize=True).imag
    return imn + np.swapaxes(imn, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit SLG toy response.py against the official Wannier90/WannierBerri shift-current formula."
    )
    parser.add_argument("--mesh-size", type=int, default=60)
    parser.add_argument("--eta-mev", type=float, default=50.0, help="spectral Lorentzian broadening")
    parser.add_argument("--sc-eta-mev", type=float, default=40.0, help="postw90 sc_eta principal-value regularizer")
    parser.add_argument("--emin", type=float, default=0.0)
    parser.add_argument("--emax", type=float, default=8.0)
    parser.add_argument("--n-energy", type=int, default=201)
    parser.add_argument("--mass-ev", type=float, default=1.5)
    parser.add_argument("--hopping-ev", type=float, default=2.73)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/shift_current_slg_toy_reference_formula_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = GappedSLGParams(hopping_ev=float(args.hopping_ev), mass_ev=float(args.mass_ev))
    photon = np.linspace(float(args.emin), float(args.emax), int(args.n_energy), dtype=float)
    eta_ev = eta_mev_to_ev(float(args.eta_mev))
    sc_eta_ev = eta_mev_to_ev(float(args.sc_eta_mev))
    parsed = {name: parse_component(name) for name in DEFAULT_COMPONENTS}

    ours_integral = {name: np.zeros_like(photon, dtype=np.complex128) for name in parsed}
    ref_integral = {name: np.zeros_like(photon, dtype=float) for name in parsed}

    k_points, k_weights = hex_bz_grid(int(args.mesh_size), params)
    for k_xy, weight in zip(k_points, k_weights, strict=True):
        for multiple in (0, 1, 2):
            k_rot = rotate_c3(k_xy, multiple)
            evals, evecs = diagonalize(k_rot, params)

            tensors = precompute_response_tensors(
                evals,
                evecs,
                dhdk(k_rot, params),
                d2hdk=d2hdk(k_rot, params),
                denominator_cutoff_ev=1.0e-10,
            )
            for name, component in parsed.items():
                transitions, weights = positive_transition_terms(tensors, component)
                add_transitions_to_integral(
                    ours_integral[name],
                    photon,
                    transitions,
                    weights,
                    k_weight_nm_inv_sq=float(weight) / 3.0,
                    eta_ev=eta_ev,
                )

            imn = wannier90_internal_imn(evals, evecs, k_rot, params, sc_eta_ev=sc_eta_ev)
            occupations = fermi_occupation(evals, mu_ev=0.0, temperature_k=0.0)
            for name, (a, b, c) in parsed.items():
                for n in range(evals.size):
                    for m in range(evals.size):
                        transition_ev = float(evals[m] - evals[n])
                        if transition_ev <= 0.0:
                            continue
                        occ_diff = float(occupations[n] - occupations[m])
                        if abs(occ_diff) < 1.0e-14:
                            continue
                        ref_integral[name] += (
                            float(weight)
                            / 3.0
                            / (2.0 * np.pi) ** 2
                            * occ_diff
                            * imn[n, m, a, b, c]
                            * lorentzian_delta(photon, transition_ev, eta_ev)
                        )

    ours = {name: sigma_from_integral(values) for name, values in ours_integral.items()}
    ref = {name: SHIFT_CURRENT_PREFAC_UA_NM_PER_V2 * values for name, values in ref_integral.items()}

    comparisons: dict[str, dict[str, float]] = {}
    for name in parsed:
        diff = ours[name] - ref[name]
        peak_idx = int(np.argmax(np.abs(ours[name])))
        ref_peak_idx = int(np.argmax(np.abs(ref[name])))
        comparisons[name] = {
            "max_abs_ours_uA_nm_per_V2": float(np.max(np.abs(ours[name]))),
            "energy_at_max_abs_ours_ev": float(photon[peak_idx]),
            "value_at_ours_peak_uA_nm_per_V2": float(ours[name][peak_idx]),
            "max_abs_reference_uA_nm_per_V2": float(np.max(np.abs(ref[name]))),
            "energy_at_max_abs_reference_ev": float(photon[ref_peak_idx]),
            "value_at_reference_peak_uA_nm_per_V2": float(ref[name][ref_peak_idx]),
            "max_abs_difference_uA_nm_per_V2": float(np.max(np.abs(diff))),
        }

    summary = {
        "purpose": "formula audit only; no visual fitting or post-hoc scaling",
        "official_reference_sources": [
            "reference/upstream/wannier90/src/postw90/berry.F90::berry_get_sc_klist",
            "reference/upstream/wannier-berri/wannierberri/calculators/dynamic.py::ShiftCurrentFormula",
        ],
        "params": {
            "mesh_size_before_c3_orbit_average": int(args.mesh_size),
            "eta_mev": float(args.eta_mev),
            "sc_eta_mev": float(args.sc_eta_mev),
            "mass_ev": params.mass_ev,
            "hopping_ev": params.hopping_ev,
            "bond_nm": params.bond_nm,
        },
        "comparisons": comparisons,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "slg_toy_reference_formula_audit.npz",
        photon_energies_ev=photon,
        **{f"ours_{name.replace(';', '_')}": values for name, values in ours.items()},
        **{f"reference_{name.replace(';', '_')}": values for name, values in ref.items()},
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
