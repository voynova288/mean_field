from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mean_field.core.hf import DensityConvention, density_to_stored_delta
from mean_field.systems.tdbg import TDBGInteractionSettings, TDBGProjectedHFConfig
from mean_field.systems.tdbg.projected_hf import (
    _fock_density_for_policy,
    _hartree_density_for_policy,
    _reference_subtracted_tdbg_density,
)


def _policy_data(reference_density: np.ndarray, settings: TDBGInteractionSettings) -> SimpleNamespace:
    return SimpleNamespace(
        config=TDBGProjectedHFConfig(interaction=settings),
        reference_density=reference_density,
    )


def test_tdbg_reference_subtracted_density_matches_core_helper() -> None:
    rng = np.random.default_rng(7)
    density = rng.standard_normal((3, 3, 2)) + 1j * rng.standard_normal((3, 3, 2))
    reference = rng.standard_normal((3, 3, 2)) + 1j * rng.standard_normal((3, 3, 2))
    data = _policy_data(reference, TDBGInteractionSettings())

    old = density - reference
    direct = density_to_stored_delta(
        density,
        DensityConvention.PROJECTOR,
        reference=reference,
        reference_policy="require",
    )
    migrated = _reference_subtracted_tdbg_density(data, density)

    np.testing.assert_array_equal(direct, old)
    np.testing.assert_array_equal(migrated, old)


def test_tdbg_hartree_and_fock_density_policies_use_reference_subtraction() -> None:
    density = np.asarray(
        [
            [[1.0 + 0.0j, 0.1 + 0.2j], [0.3 - 0.4j, 0.5 + 0.0j]],
            [[0.3 + 0.4j, 0.6 + 0.0j], [0.7 + 0.0j, 0.8 - 0.1j]],
        ],
        dtype=np.complex128,
    )
    reference = 0.25 * np.ones_like(density)
    absolute_data = _policy_data(
        reference,
        TDBGInteractionSettings(hartree_reference="none", fock_density="absolute"),
    )
    subtracted_data = _policy_data(
        reference,
        TDBGInteractionSettings(hartree_reference="charge_neutral", fock_density="reference_subtracted"),
    )

    np.testing.assert_array_equal(_hartree_density_for_policy(absolute_data, density), density)
    np.testing.assert_array_equal(_fock_density_for_policy(absolute_data, density), density)
    np.testing.assert_array_equal(_hartree_density_for_policy(subtracted_data, density), density - reference)
    np.testing.assert_array_equal(_fock_density_for_policy(subtracted_data, density), density - reference)
