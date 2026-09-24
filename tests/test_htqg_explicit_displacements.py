from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mean_field.systems.htqg import HTQGExplicitDisplacements
import mean_field.systems.htqg.hf as htqg_hf
from mean_field.systems.htqg.hf import (
    HTQGInteractionSettings,
    HTQGProjectedHFConfig,
    assemble_htqg_hf_target_hamiltonian,
    build_htqg_projected_hf_data,
    build_htqg_target_data,
    prepare_htqg_hf_target_interaction,
    validate_htqg_projected_hf_config,
)
from mean_field.systems.htqg.optical import noninteracting_optical_derivative_bundle
from mean_field.systems.htqg.params import HTQGParams


def _config(domain: str, explicit: HTQGExplicitDisplacements | None = None) -> HTQGProjectedHFConfig:
    return HTQGProjectedHFConfig(
        theta_deg=2.25,
        n_shells=1,
        mesh_size=2,
        active_band_count=4,
        domain=domain,
        filling=0,
        params=HTQGParams.realistic(kappa=0.6),
        interaction=HTQGInteractionSettings(g_shells=0),
        active_basis="energy",
        frac_shift=(0.0, 0.0),
        explicit_displacements=explicit,
    )


@pytest.mark.parametrize(
    ("domain", "d34_sign"),
    (("alpha_beta_alpha", -1.0), ("alpha_beta_gamma", 1.0)),
)
def test_explicit_displacements_reproduce_named_endpoint_h0(domain: str, d34_sign: float) -> None:
    named = build_htqg_projected_hf_data(_config(domain))
    explicit = HTQGExplicitDisplacements.from_complex(
        named.lattice.d_ba,
        d34_sign * named.lattice.d_ba,
        label=f"endpoint_{domain}",
    )
    replay = build_htqg_projected_hf_data(_config("alpha_beta_alpha", explicit))

    assert replay.d12 == pytest.approx(named.d12, abs=0.0)
    assert replay.d34 == pytest.approx(named.d34, abs=0.0)
    assert replay.displacement_provenance["kind"] == "explicit"
    np.testing.assert_allclose(replay.h0, named.h0, atol=1.0e-13, rtol=0.0)


def test_asymmetric_explicit_displacements_reach_every_target_stencil_point(monkeypatch) -> None:
    base = build_htqg_projected_hf_data(_config("alpha_beta_alpha"))
    scale = abs(base.lattice.d_ba)
    explicit = HTQGExplicitDisplacements.from_complex(
        0.37 * base.lattice.d_ba + 0.19j * scale,
        -0.22 * base.lattice.d_ba + 0.31j * scale,
        label="asymmetric_non_domain_point",
    )
    data = build_htqg_projected_hf_data(_config("alpha_beta_alpha", explicit))

    calls: list[tuple[complex, complex]] = []
    original = htqg_hf.diagonalize_hamiltonian

    def recording_diagonalize(*args, **kwargs):
        calls.append((complex(kwargs["d12"]), complex(kwargs["d34"])))
        return original(*args, **kwargs)

    monkeypatch.setattr(htqg_hf, "diagonalize_hamiltonian", recording_diagonalize)
    point = noninteracting_optical_derivative_bundle(
        data,
        complex(data.kvec[0]),
        step_nm_inv=1.0e-4,
    )

    assert len(calls) == 2 * 9
    assert all(d12 == data.d12 and d34 == data.d34 for d12, d34 in calls)
    assert data.d12 != base.domain.d12
    assert data.d34 != base.domain.d34
    assert point.band_indices == data.band_indices
    np.testing.assert_allclose(point.hamiltonian, data.h0[:, :, 0], atol=1.0e-12, rtol=0.0)
    assert np.isfinite(point.dhdk).all()
    assert np.isfinite(point.d2hdk).all()
    assert point.min_basis_overlap_singular_value > 0.0
    tau = np.diag(np.asarray([label.valley for label in data.labels], dtype=float))
    operators = (point.hamiltonian, *point.dhdk, *point.d2hdk.reshape(-1, data.nt, data.nt))
    assert max(float(np.max(np.abs(tau @ operator - operator @ tau))) for operator in operators) <= 1.0e-13


def test_explicit_displacements_are_canonical_json_native_values() -> None:
    explicit = HTQGExplicitDisplacements(
        d12_nm_xy=[np.float32(1.25), np.float64(-0.5)],  # type: ignore[arg-type]
        d34_nm_xy=np.asarray([0.75, 0.125]),  # type: ignore[arg-type]
        label=123,  # type: ignore[arg-type]
    )
    assert explicit.d12_nm_xy == (1.25, -0.5)
    assert explicit.d34_nm_xy == (0.75, 0.125)
    assert explicit.label == "123"
    hash(explicit)


def test_target_context_identity_binds_displacements_and_payload_hash() -> None:
    data = build_htqg_projected_hf_data(_config("alpha_beta_alpha"))
    target = build_htqg_target_data(data, np.asarray([complex(data.kvec[0])]))
    context = prepare_htqg_hf_target_interaction(data, target)
    density = np.zeros_like(data.h0)

    changed_displacement = replace(data, d34=0.123 + 0.456j)
    with pytest.raises(ValueError, match="does not belong"):
        assemble_htqg_hf_target_hamiltonian(
            changed_displacement,
            target,
            density,
            target_interaction_context=context,
        )

    bad_hash = replace(context, identity_sha256="0" * 64)
    with pytest.raises(ValueError, match="payload hash"):
        assemble_htqg_hf_target_hamiltonian(
            data,
            target,
            density,
            target_interaction_context=bad_hash,
        )


def test_invalid_explicit_displacements_fail_closed() -> None:
    good = HTQGExplicitDisplacements.from_complex(1.0 + 2.0j, 3.0 + 4.0j, label="good")
    validate_htqg_projected_hf_config(_config("alpha_beta_alpha", good))

    with pytest.raises(ValueError, match="exactly two real"):
        HTQGExplicitDisplacements(d12_nm_xy=(1.0,), d34_nm_xy=(3.0, 4.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly two finite"):
        validate_htqg_projected_hf_config(
            _config(
                "alpha_beta_alpha",
                HTQGExplicitDisplacements(d12_nm_xy=(np.inf, 0.0), d34_nm_xy=(0.0, 0.0)),
            )
        )
    with pytest.raises(ValueError, match="label must be nonempty"):
        validate_htqg_projected_hf_config(
            _config(
                "alpha_beta_alpha",
                replace(good, label="  "),
            )
        )
