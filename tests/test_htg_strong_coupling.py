from __future__ import annotations

import numpy as np

from mean_field.systems.htg import classify_htg_strong_coupling_state


def _density_from_occupied_flavors(occupied_bands: tuple[tuple[int, int, tuple[int, ...]], ...], *, nk: int = 3) -> np.ndarray:
    n_spin = 2
    n_eta = 2
    n_band = 2
    nt = n_spin * n_eta * n_band
    idx = np.arange(nt, dtype=int).reshape((n_spin, n_eta, n_band), order="F")
    density = np.zeros((nt, nt, nk), dtype=np.complex128)
    for ik in range(nk):
        density[:, :, ik] = -0.5 * np.eye(nt, dtype=np.complex128)
    for ispin, ieta, bands in occupied_bands:
        for iband in bands:
            density[int(idx[ispin, ieta, iband]), int(idx[ispin, ieta, iband]), :] = 0.5
    return density


def test_strong_coupling_classifier_labels_fb_d2b2_state() -> None:
    density = _density_from_occupied_flavors(
        (
            (0, 0, (0, 1)),
            (0, 1, (0, 1)),
            (1, 0, (1,)),
            (1, 1, (1,)),
        )
    )

    classification = classify_htg_strong_coupling_state(density)

    assert classification.family == "FB"
    assert classification.class_label == "[D2 B2]"
    assert classification.flavor_occupation_pattern == (2, 2, 1, 1)
    assert np.isclose(classification.nu_z, -2.0)


def test_strong_coupling_classifier_labels_fi_d3_state() -> None:
    density = _density_from_occupied_flavors(
        (
            (0, 0, (0, 1)),
            (0, 1, (0, 1)),
            (1, 0, (0, 1)),
        )
    )

    classification = classify_htg_strong_coupling_state(density)

    assert classification.family == "FI"
    assert classification.class_label == "[D3]"
    assert classification.flavor_occupation_pattern == (2, 2, 2, 0)
    assert classification.n_empty == 1
