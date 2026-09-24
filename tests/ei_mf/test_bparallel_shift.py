from __future__ import annotations

import numpy as np

from ei_mf.bfield import delta_k_parallel_nm_inv


def test_bparallel_shift_uses_hbar_not_h() -> None:
    assert np.isclose(delta_k_parallel_nm_inv(35.0, 10.0), 0.53, rtol=0.05)
