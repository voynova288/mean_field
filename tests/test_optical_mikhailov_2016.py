from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.optical.benchmarks.mikhailov_2016 import (
    mikhailov2016_thg_xxxx_scattering,
)


PASSOS_FIG12_CHECKPOINTS = (
    Path(__file__).parents[1]
    / "src/analysis/optical/benchmarks/data/passos2018_fig12_source_checkpoints.json"
)


def test_mikhailov_scattering_poles_and_sector_sum() -> None:
    omega = np.array([0.45, 0.8, 1.35])
    gamma = 0.11
    result = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=gamma,
    )

    expected_s30 = 3.0j / (
        (omega + 1.0j * gamma)
        * (2.0 * omega + 1.0j * gamma)
        * (3.0 * omega + 1.0j * gamma)
    )
    adiabatic_s30 = 3.0j / (
        (omega + 1.0j * gamma)
        * (2.0 * omega + 2.0j * gamma)
        * (3.0 * omega + 3.0j * gamma)
    )
    assert np.allclose(result.s_30, expected_s30, rtol=2.0e-14, atol=2.0e-14)
    assert not np.allclose(result.s_30, adiabatic_s30, rtol=1.0e-3, atol=1.0e-3)
    assert np.allclose(
        result.total,
        result.s_30 + result.s_21 + result.s_12 + result.s_03,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    for value in (
        result.omega_over_fermi_energy,
        result.s_30,
        result.s_21,
        result.s_12,
        result.s_03,
        result.total,
    ):
        assert not value.flags.writeable


def test_mikhailov_closed_forms_match_defining_integrals_away_from_narrow_poles() -> None:
    omega = np.array([0.45, 0.65, 0.82, 1.25])
    closed = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=0.11,
        integral_method="closed_form",
    )
    integrated = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=0.11,
        integral_method="gauss_legendre",
        quadrature_order=1024,
    )
    for name in ("s_21", "s_12", "s_03", "total"):
        assert np.allclose(
            getattr(closed, name),
            getattr(integrated, name),
            rtol=2.0e-9,
            atol=2.0e-9,
        )


@pytest.mark.parametrize("gamma", [0.03, 0.003])
def test_mikhailov_low_frequency_uses_converged_defining_integral_limit(
    gamma: float,
) -> None:
    omega = np.array([1.0e-4, 0.02, 0.08])
    automatic = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=gamma,
        integral_method="closed_form",
        quadrature_order=512,
    )
    integrated = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=gamma,
        integral_method="gauss_legendre",
        quadrature_order=1024,
    )
    assert automatic.coincident_argument_fallback_count == omega.size
    assert automatic.quadrature_order == 1024
    assert automatic.evaluation_method == "closed_form_with_converged_quadrature_limits"
    assert automatic.quadrature_max_absolute_change > 0.0
    assert automatic.quadrature_max_relative_change >= 0.0
    np.testing.assert_allclose(
        automatic.total,
        integrated.total,
        rtol=2.0e-12,
        atol=2.0e-8,
    )


def test_mikhailov_hybrid_boundary_is_continuous_and_matches_integrals() -> None:
    omega = np.array([0.099999, 0.100001])
    hybrid = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=0.03,
        integral_method="closed_form",
        quadrature_order=512,
    )
    integrated = mikhailov2016_thg_xxxx_scattering(
        omega,
        gamma_over_fermi_energy=0.03,
        integral_method="gauss_legendre",
        quadrature_order=1024,
    )
    assert hybrid.coincident_argument_fallback_count == 1
    np.testing.assert_allclose(hybrid.total, integrated.total, rtol=2.0e-9, atol=2.0e-9)


def test_mikhailov_passos_fig12_source_vector_checkpoints() -> None:
    # Official arXiv source panels E1/E2/E3, calibrated only from printed ticks.
    fixture = json.loads(PASSOS_FIG12_CHECKPOINTS.read_text(encoding="utf-8"))
    assert fixture["source_vector_npz_sha256"] == (
        "3405af8f3f2984c75ac8d597c82d5d58526ad6d14755e5d0316accdb87fb8eb1"
    )
    assert fixture["source_metadata_sha256"] == (
        "e09b0fc5acd8d4e175199eeeeb7ed65cb73a304b4981a31d1fdd4a657e560ffc"
    )
    mu_over_t = fixture["parameters"]["mu_over_t"]
    for checkpoint in fixture["checkpoints"]:
        photon_ratio = checkpoint["photon_energy_over_t"]
        gamma_over_t = checkpoint["gamma_over_t"]
        source_value = complex(checkpoint["source_re"], checkpoint["source_im"])
        computed = mikhailov2016_thg_xxxx_scattering(
            np.array([photon_ratio / mu_over_t]),
            gamma_over_fermi_energy=gamma_over_t / mu_over_t,
        ).total[0]
        assert computed == pytest.approx(
            source_value,
            abs=checkpoint["absolute_tolerance"],
        )


def test_mikhailov_input_validation() -> None:
    with pytest.raises(ValueError, match="nonempty 1D"):
        mikhailov2016_thg_xxxx_scattering(
            np.array(0.5),
            gamma_over_fermi_energy=0.1,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        mikhailov2016_thg_xxxx_scattering(
            np.array([0.0]),
            gamma_over_fermi_energy=0.1,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        mikhailov2016_thg_xxxx_scattering(
            np.array([0.5]),
            gamma_over_fermi_energy=0.0,
        )
    with pytest.raises(ValueError, match="integral_method"):
        mikhailov2016_thg_xxxx_scattering(
            np.array([0.5]),
            gamma_over_fermi_energy=0.1,
            integral_method="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="real scalar, not bool"):
        mikhailov2016_thg_xxxx_scattering(
            np.array([0.5]),
            gamma_over_fermi_energy=True,
        )
    with pytest.raises(TypeError, match="integer, not bool"):
        mikhailov2016_thg_xxxx_scattering(
            np.array([0.5]),
            gamma_over_fermi_energy=0.1,
            quadrature_order=True,
        )
