from __future__ import annotations

import pytest

from analysis.topology import compute_lattice_topology
from mean_field.systems.tdbg import TDBGModel, TDBGParameters


def test_tdbg_sewn_state_fhs_reproduces_liu2022_single_valley_conduction_chern() -> None:
    """AB-BA high-D TDBG needs moire-BZ sewing to recover the paper C_v=1 band."""

    model = TDBGModel.from_config(
        1.38,
        cut=2.0,
        params=TDBGParameters.full(stacking="AB-BA", Delta=0.09),
    )
    conduction_band = model.matrix_dim // 2
    chern_k = compute_lattice_topology(model.fhs_state_on_grid(5, conduction_band, valley=1, n_bands=conduction_band + 1))
    chern_kprime = compute_lattice_topology(model.fhs_state_on_grid(5, conduction_band, valley=-1, n_bands=conduction_band + 1))

    assert chern_k.chern_number == pytest.approx(1.0, abs=1.0e-8)
    assert chern_kprime.chern_number == pytest.approx(-1.0, abs=1.0e-8)
    assert chern_k.min_link_magnitude > 0.1
    assert chern_k.index_metadata["metadata"]["boundary_sewing"] is True


def test_tdbg_unsewn_state_records_that_boundary_sewing_was_disabled() -> None:
    model = TDBGModel.from_config(
        1.38,
        cut=2.0,
        params=TDBGParameters.full(stacking="AB-BA", Delta=0.09),
    )
    conduction_band = model.matrix_dim // 2
    unsewn = compute_lattice_topology(
        model.fhs_state_on_grid(
            5,
            conduction_band,
            valley=1,
            n_bands=conduction_band + 1,
            boundary_sewing=False,
        )
    )

    assert unsewn.index_metadata["metadata"]["boundary_sewing"] is False
