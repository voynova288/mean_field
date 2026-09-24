from __future__ import annotations

import pytest

import mean_field.systems.tdbg.projected_hf_config as projected_hf_config



def test_tdbg_projected_hf_config_valley_ud_mapping_and_validation() -> None:
    assert projected_hf_config.tdbg_delta_from_paper_ud_for_valley(
        0.09,
        1,
        convention="minus_xi_ud_over3",
    ) == pytest.approx(-0.03)
    assert projected_hf_config.tdbg_delta_from_paper_ud_for_valley(
        0.09,
        -1,
        convention="minus_xi_ud_over3",
    ) == pytest.approx(0.03)

    cfg = projected_hf_config.TDBGProjectedHFConfig(mesh_size=1, max_iter=1)
    projected_hf_config.validate_tdbg_projected_hf_config(cfg)

    with pytest.raises(ValueError, match="Unsupported paper_ud_convention"):
        projected_hf_config.validate_tdbg_projected_hf_config(
            projected_hf_config.TDBGProjectedHFConfig(paper_ud_convention="bad")  # type: ignore[arg-type]
        )
