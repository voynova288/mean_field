from __future__ import annotations

import numpy as np

from analysis.order_parameters import (
    StateLabel,
    analyze_tdbg_order_parameters,
    finite_field_valley_spin_order_parameters,
    folded_translation_order_parameters,
)
from mean_field.core.hf.finite_field import calculate_valley_spin_order_parameters
from mean_field.systems.tdbg.projected_hf_config import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedWindow
from mean_field.systems.tdbg.projected_hf_state import (
    TDBGProjectedHFData,
    TDBGStateLabel,
    _conventional_projector_to_stored,
    _numeric_order_parameters,
    tdbg_order_parameters,
)
from mean_field.systems.tmbg.polshyn_supercell import translation_order_parameters


def _toy_tdbg_data() -> TDBGProjectedHFData:
    labels = (
        TDBGStateLabel(0, "up", 1, 0, 10),
        TDBGStateLabel(1, "up", -1, 0, 10),
        TDBGStateLabel(2, "down", 1, 0, 10),
        TDBGStateLabel(3, "down", -1, 0, 10),
    )
    nt = len(labels)
    nk = 2
    h0 = np.zeros((nt, nt, nk), dtype=np.complex128)
    return TDBGProjectedHFData(
        model=object(),  # type: ignore[arg-type]
        config=TDBGProjectedHFConfig(
            window=TDBGProjectedWindow("toy", band_indices=(10,)),
            filling=2,
            interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        ),
        k_grid_frac=np.zeros((nk, 2), dtype=float),
        kvec=np.zeros(nk, dtype=np.complex128),
        band_indices=(10,),
        labels=labels,
        h0=h0,
        wavefunctions=np.zeros((nt, nk, 1, 4), dtype=np.complex128),
        reference_density=np.zeros_like(h0),
        n_occupied_per_k=2,
        lower_band_count=0,
        moire_area_nm2=1.0,
        shifts=(),
        shift_gvecs=np.zeros(0, dtype=np.complex128),
        shift_srcmaps=(),
    )


def test_tdbg_order_parameter_wrapper_matches_common_adapter() -> None:
    data = _toy_tdbg_data()
    projector = np.zeros((data.nt, data.nt, data.nk), dtype=np.complex128)
    for ik in range(data.nk):
        projector[0, 0, ik] = 1.0
        projector[1, 1, ik] = 1.0
        projector[0, 1, ik] = 0.25
        projector[1, 0, ik] = 0.25
    density = np.zeros_like(projector)
    for ik in range(data.nk):
        density[:, :, ik] = _conventional_projector_to_stored(projector[:, :, ik])

    labels = tuple(
        StateLabel(
            index=label.index,
            spin=label.spin,
            valley=label.valley,
            band=label.band_index,
            active=True,
            metadata=label.to_dict(),
        )
        for label in data.labels
    )
    common = analyze_tdbg_order_parameters(projector, labels)
    numeric = _numeric_order_parameters(data, density)
    wrapped = tdbg_order_parameters(data, density)

    assert numeric["spin_polarization"] == common.scalars["spin_polarization"]
    assert numeric["valley_polarization"] == common.scalars["valley_polarization"]
    assert numeric["ivc_amplitude"] == common.scalars["ivc_amplitude"]
    assert wrapped["classification"] == common.classification
    assert len(wrapped["occupations"]) == data.nt


def test_polshyn_translation_wrapper_matches_common_helper() -> None:
    density = np.zeros((2, 2, 4, 4, 3), dtype=np.complex128)
    density[0, 0, 0, 1, :] = [0.1, 0.2, 0.3]
    density[0, 0, 2, 3, :] = [0.4, 0.0, 0.2]
    kwargs = {"projected_indices": (5, 6), "target_band_index": 5, "spin_index": 0, "valley_index": 0}
    wrapped = translation_order_parameters(density, **kwargs)
    common = folded_translation_order_parameters(density, **kwargs)
    for key, value in common.items():
        np.testing.assert_allclose(wrapped[key], value)


def test_finite_field_valley_spin_wrapper_matches_common_helper() -> None:
    rng = np.random.default_rng(123)
    dim = 16
    nk = 2
    h = np.zeros((dim, dim, nk), dtype=np.complex128)
    energies = np.zeros((dim, nk), dtype=float)
    for ik in range(nk):
        a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        h[:, :, ik] = a + a.conjugate().T
        energies[:, ik] = np.linalg.eigvalsh(h[:, :, ik])
    mu = float(np.median(energies))
    wrapped = calculate_valley_spin_order_parameters(h, energies, mu, q=2, n_eta=2, n_spin=2, n_band=2)
    common = finite_field_valley_spin_order_parameters(h, energies, mu, q=2, n_eta=2, n_spin=2, n_band=2)
    assert wrapped.keys() == common.keys()
    for key in wrapped:
        np.testing.assert_allclose(wrapped[key], common[key])
