from __future__ import annotations

import numpy as np
import pytest

from mean_field.devtools.run_rlg_hbn_paper_hf import _load_archive_density, _preflight_run_specs


def _config_for_specs(run_specs: list[dict[str, object]]) -> dict[str, object]:
    return {
        "run_specs": run_specs,
        "init_modes": tuple(str(spec["init_mode"]) for spec in run_specs),
        "seeds": tuple(int(spec["seed"]) for spec in run_specs),
        "active_valence_bands": 3,
        "active_conduction_bands": 3,
        "k_mesh_size": 18,
    }


def test_rlg_hbn_paper_hf_archive_density_loader_uses_core_shape_validation(tmp_path) -> None:
    path = tmp_path / "hf_run_state.npz"
    density = np.zeros((2, 2, 3), dtype=np.complex128)
    np.savez_compressed(
        path,
        density=density,
        hamiltonian=np.zeros_like(density),
        iter_energy_mev=np.asarray([3.0, 2.0]),
        iter_err=np.asarray([1.0, 0.5]),
        iter_oda=np.asarray([0.1, 0.2]),
    )

    loaded, trace = _load_archive_density(path, (2, 2, 3))
    np.testing.assert_array_equal(loaded, density)
    assert trace["iteration"] == [1, 2]
    assert trace["energy_mev"] == [3.0, 2.0]

    with pytest.raises(ValueError, match="does not match current basis"):
        _load_archive_density(path, (2, 2, 4))


def test_rlg_hbn_paper_hf_preflight_accepts_supported_init_modes() -> None:
    payload = _preflight_run_specs(
        _config_for_specs(
            [
                {"init_mode": "flavor", "seed": 1},
                {"init_mode": "bm", "seed": 1},
                {"init_mode": "perturbed", "seed": 2},
            ]
        )
    )

    assert payload["status"] == "ok"
    assert [row["normalized_init_mode"] for row in payload["run_specs"]] == [
        "flavor",
        "bm",
        "perturbed",
    ]


def test_rlg_hbn_paper_hf_preflight_rejects_unsupported_init_mode_before_setup() -> None:
    with pytest.raises(ValueError, match="vp:1") as exc_info:
        _preflight_run_specs(_config_for_specs([{"init_mode": "vp", "seed": 1}]))

    message = str(exc_info.value)
    assert "before expensive setup" in message
    assert "Unsupported RLG/hBN HF init mode: vp" in message
