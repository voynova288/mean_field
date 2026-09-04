"""Tests for generic paired-sector zero-temperature Hessian plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from mean_field.core import hf
from mean_field.core.hf.zero_temperature_sector_stability import (
    OrbitalTransitionLane,
    PairedOrbitalTransitionFrame,
    PairedSectorOrbitalHessian,
)


def _lane(label: str, particle: int, hole: int, particle_energy: float, hole_energy: float):
    return OrbitalTransitionLane(
        label=label,
        particle_orbital_ids=np.asarray([particle], dtype=np.int64),
        hole_orbital_ids=np.asarray([hole], dtype=np.int64),
        particle_energies_ev=np.asarray([particle_energy], dtype=np.float64),
        hole_energies_ev=np.asarray([hole_energy], dtype=np.float64),
    )


def test_paired_real_action_has_factor_two_and_conjugate_lane_coupling() -> None:
    frame = PairedOrbitalTransitionFrame(
        _lane("q", 3, 0, 2.0, 0.5),
        _lane("minus_q", 2, 1, 3.0, 1.0),
    )

    def interaction(first, second):
        return 0.4 * first + 0.3 * second.conj(), 0.4 * second + 0.3 * first.conj()

    hessian = PairedSectorOrbitalHessian(frame, interaction)
    first = np.asarray([0.2 + 0.7j])
    second = np.asarray([-0.4 + 0.1j])
    vector = frame.pack_complex(first, second)
    actual = hessian.matvec(vector)
    expected_first = 1.5 * first + interaction(first, second)[0]
    expected_second = 2.0 * second + interaction(first, second)[1]
    expected = frame.pack_complex(expected_first, expected_second, factor=2.0)
    assert np.array_equal(actual, expected)
    with pytest.raises(AttributeError):
        hessian._interaction_response = lambda first, second: (first, second)
    assert not hasattr(hessian, "linear_operator")
    diagnostic = hessian.check_bilinear_symmetry(seed=17, probe_count=8)
    assert diagnostic.all_evaluated_pairs_symmetric
    assert diagnostic.outcome == "sampled_pairs_symmetric_not_proof"


def test_bilinear_probe_detects_asymmetric_callback() -> None:
    frame = PairedOrbitalTransitionFrame(
        _lane("q", 1, 0, 1.0, 0.0),
        _lane("minus_q", 3, 2, 1.0, 0.0),
    )
    hessian = PairedSectorOrbitalHessian(
        frame,
        lambda first, second: (2.0 * second, np.zeros_like(second)),
    )
    diagnostic = hessian.check_bilinear_symmetry(
        seed=19, probe_count=8, atol=0.0, rtol=0.0
    )
    assert diagnostic.asymmetry_detected


def test_lane_arrays_are_byte_immutable_and_duplicate_pairs_reject() -> None:
    lane = _lane("q", 1, 0, 2.0, 1.0)
    assert np.array_equal(lane.one_body_gaps_ev, [1.0])
    for array in (
        lane.particle_orbital_ids,
        lane.hole_orbital_ids,
        lane.particle_energies_ev,
        lane.hole_energies_ev,
        lane.one_body_gaps_ev,
    ):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            array.setflags(write=True)

    with pytest.raises(ValueError, match="duplicate"):
        OrbitalTransitionLane(
            label="bad",
            particle_orbital_ids=np.asarray([1, 1], dtype=np.int64),
            hole_orbital_ids=np.asarray([0, 0], dtype=np.int64),
            particle_energies_ev=np.asarray([2.0, 2.0]),
            hole_energies_ev=np.asarray([1.0, 1.0]),
        )


def test_zero_dimension_and_callback_validation_are_fail_closed() -> None:
    empty = OrbitalTransitionLane(
        label="empty",
        particle_orbital_ids=np.asarray([], dtype=np.int64),
        hole_orbital_ids=np.asarray([], dtype=np.int64),
        particle_energies_ev=np.asarray([], dtype=np.float64),
        hole_energies_ev=np.asarray([], dtype=np.float64),
    )
    frame = PairedOrbitalTransitionFrame(empty, empty)
    hessian = PairedSectorOrbitalHessian(
        frame, lambda first, second: (first.copy(), second.copy())
    )
    assert hessian.real_dimension == 0
    assert hessian.check_bilinear_symmetry(seed=1, probe_count=2).inconclusive

    nonempty = PairedOrbitalTransitionFrame(
        _lane("q", 1, 0, 1.0, 0.0), empty
    )
    bad = PairedSectorOrbitalHessian(
        nonempty, lambda first, second: np.zeros_like(first)
    )
    with pytest.raises(TypeError, match="exact pair"):
        bad.matvec(np.zeros(2))


def test_public_core_exports() -> None:
    for name in (
        "OrbitalTransitionLane",
        "PairedInteractionResponse",
        "PairedOrbitalTransitionFrame",
        "PairedSectorOrbitalHessian",
    ):
        assert hasattr(hf, name)
        assert name in hf.__all__
