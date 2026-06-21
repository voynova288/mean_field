from __future__ import annotations

from mean_field.systems.tdbg import projected_hf, projected_hf_reports


def test_tdbg_projected_hf_reports_split_preserves_legacy_facade() -> None:
    assert projected_hf.tdbg_hf_grid_band_summary is projected_hf_reports.tdbg_hf_grid_band_summary
    assert projected_hf.liu2022_default_projected_hf_config is projected_hf_reports.liu2022_default_projected_hf_config
    assert projected_hf.liu2022_projected_hf_metadata is projected_hf_reports.liu2022_projected_hf_metadata


def test_tdbg_projected_hf_report_helpers_keep_metadata_contract() -> None:
    config = projected_hf_reports.liu2022_default_projected_hf_config(
        mesh_size=1,
        cut=1.0,
        include_intersite=False,
        include_onsite=True,
        filling=2,
        max_iter=1,
    )

    metadata = projected_hf_reports.liu2022_projected_hf_metadata(config)

    assert config.mesh_size == 1
    assert config.cut == 1.0
    assert config.interaction.include_intersite is False
    assert config.interaction.include_onsite is True
    assert metadata["theta_deg"] == 1.38
    assert metadata["paper_ud_ev"] == 0.09
    assert metadata["density_convention"] == "core stored projector P[a,b,k]=rho_conventional[b,a,k]"
    assert metadata["workflow"].startswith("self-consistent projected HF")
    assert set(metadata["code_delta_by_valley_ev"]) == {"K", "Kprime"}
