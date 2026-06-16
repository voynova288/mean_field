from __future__ import annotations

import numpy as np

from mean_field.core.hf.density import (
    DensityBundle,
    DensityConvention,
    ReferenceDensity,
    average_reference_density,
    density_to_projector,
    density_to_stored_delta,
    stored_density_to_projector,
)
from mean_field.core.hf.occupations import conventional_projector_to_stored, stored_projector_to_conventional


def test_stored_delta_round_trip_to_projector_and_back() -> None:
    projector = np.zeros((2, 2, 1), dtype=np.complex128)
    projector[0, 0, 0] = 1.0
    reference = ReferenceDensity.average(2, 1)
    stored_delta = density_to_stored_delta(projector, DensityConvention.PROJECTOR, reference=reference)

    recovered = density_to_projector(stored_delta, DensityConvention.STORED_DELTA, reference=reference)

    np.testing.assert_allclose(recovered, projector)
    np.testing.assert_allclose(DensityBundle(stored_delta, "stored_delta", reference).as_projector(), projector)


def test_half_shifted_is_average_reference_special_case() -> None:
    projector = np.zeros((3, 3, 2), dtype=np.complex128)
    projector[0, 0, :] = 1.0
    half_shifted = projector - average_reference_density(3, 2)

    recovered = density_to_projector(half_shifted, "half_shifted")

    np.testing.assert_allclose(recovered, projector)


def test_archive_projector_orientation_matches_legacy_helpers() -> None:
    stored_projector = np.zeros((2, 2, 1), dtype=np.complex128)
    stored_projector[0, 1, 0] = 2.0 + 3.0j
    ket_projector = stored_density_to_projector(stored_projector, np.zeros_like(stored_projector), convention="ket")

    np.testing.assert_allclose(ket_projector, stored_projector_to_conventional(stored_projector))
    np.testing.assert_allclose(conventional_projector_to_stored(ket_projector), stored_projector)
