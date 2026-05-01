from __future__ import annotations

import numpy as np

from mean_field.devtools.run_atmg_fig3_band_plot import build_khalaf_fig3_path
from mean_field.systems.atmg import ATMGModel, ATMGParameters


def test_khalaf_fig3_path_uses_lattice_high_symmetry_points() -> None:
    params = ATMGParameters.chiral(3, 1.35)
    model = ATMGModel.from_config(3, params.theta_deg, n_shells=1, params=params)
    lattice = model.lattice

    path = build_khalaf_fig3_path(model, points_per_segment=4)
    node_kvec = np.asarray([node.kvec for node in path.nodes], dtype=np.complex128)

    assert path.labels == ("K'", "K", r"$\Gamma$", r"$\Gamma'$", "K'")
    assert np.allclose(
        node_kvec,
        np.asarray(
            [
                -lattice.q0,
                0.0 + 0.0j,
                -lattice.q0 - lattice.q_plus,
                -lattice.q0 - lattice.q_plus + lattice.g_m1,
                -lattice.q0 + lattice.g_m1,
            ],
            dtype=np.complex128,
        ),
    )
