from __future__ import annotations

import numpy as np
import pytest

from mean_field.api import HFConfig, component_group_records, component_groups, compute_bands, make_model, model_record, run_hf


def test_public_api_imports_and_htg_band_smoke() -> None:
    model = make_model("htg", theta_deg=1.5, n_shells=0)
    bundle = compute_bands(model, n_bands=2, points_per_segment=2)

    assert bundle.energies.shape[1] == 2
    assert np.asarray(bundle.k).ndim == 1
    assert bundle.convention.energy_unit == "meV"


def test_public_model_record_and_component_group_contract() -> None:
    model = make_model("htg", theta_deg=1.5, n_shells=0)
    record = model_record(model, system_name="htg")

    assert record.system_name == "htg"
    assert "theta_deg" in record.lattice
    assert component_groups(model) == ()


def test_public_make_model_htqg_pilot_band_contract() -> None:
    model = make_model("htqg", theta_deg=2.25, n_shells=0, domain="abg")
    record = model_record(model, system_name="htqg")

    assert record.system_name == "htqg"
    assert record.lattice["model_name"] == "fujimoto2025_convention_locked"
    assert record.lattice["domain"] == "alpha_beta_gamma"
    assert model.matrix_dim == 8

    groups = component_groups(model)
    assert [group.name for group in groups] == [
        "layer_0",
        "layer_1",
        "layer_2",
        "layer_3",
        "sublattice_A",
        "sublattice_B",
    ]
    assert [group.indices.tolist() for group in groups[:4]] == [[0, 1], [2, 3], [4, 5], [6, 7]]

    path_bundle = compute_bands(model, n_bands=4, points_per_segment=1)
    assert path_bundle.energies.shape == (5, 4)
    assert path_bundle.basis_metadata["component_groups"][0] == {"name": "layer_0", "indices": [0, 1]}

    grid_bundle = compute_bands(model, grid_mesh=2, n_bands=4)
    assert grid_bundle.energies.shape == (2, 2, 4)


def test_public_make_model_tbg_zero_field_bm_band_contract() -> None:
    model = make_model("tbg", variant="zero_field_bm", theta_deg=1.2, lg=1)
    record = model_record(model, system_name="tbg")

    assert record.system_name == "tbg"
    assert record.lattice["model_name"] == "zero_field_bm"
    assert model.matrix_dim == 4
    assert [group.name for group in component_groups(model)] == ["layer_bottom", "layer_top"]
    path_bundle = compute_bands(model, n_bands=2, points_per_segment=1)
    assert path_bundle.energies.shape == (4, 2)
    assert path_bundle.basis_metadata["component_groups"] == [
        {"name": "layer_bottom", "indices": [0, 1]},
        {"name": "layer_top", "indices": [2, 3]},
    ]
    grid_bundle = compute_bands(model, grid_mesh=2, n_bands=2)
    assert grid_bundle.energies.shape == (2, 2, 2)
    with pytest.raises(NotImplementedError, match="central two bands"):
        compute_bands(model, n_bands=4, points_per_segment=1)
    with pytest.raises(NotImplementedError, match="zero_field_bm"):
        make_model("tbg", variant="finite_field")


def test_rlg_hbn_model_declares_layer_component_groups() -> None:
    model = make_model("rlg_hbn", layer_count=3, shell_count=1)
    groups = component_groups(model)

    assert [group.name for group in groups] == ["layer_0", "layer_1", "layer_2"]
    assert [group.indices.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5]]
    assert component_group_records(model) == (
        {"name": "layer_0", "indices": [0, 1]},
        {"name": "layer_1", "indices": [2, 3]},
        {"name": "layer_2", "indices": [4, 5]},
    )


def test_tmbg_model_declares_layer_component_groups() -> None:
    model = make_model("tmbg", n_shells=0)
    groups = component_groups(model)

    assert [group.name for group in groups] == ["layer_bottom", "layer_middle", "layer_top"]
    assert [group.indices.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5]]


def test_tdbg_model_declares_q_site_sector_layer_component_groups() -> None:
    model = make_model("tdbg", cut=1.0)
    groups = component_groups(model)

    assert [group.name for group in groups] == [
        "sector_0",
        "sector_1",
        "layer_0",
        "layer_1",
        "layer_2",
        "layer_3",
        "sublattice_A",
        "sublattice_B",
    ]
    sector_0 = groups[0].indices.tolist()
    sector_1 = groups[1].indices.tolist()
    assert sector_0 and sector_1
    assert sorted(sector_0 + sector_1) == list(range(model.matrix_dim))
    assert groups[2].indices.tolist() == [idx for idx in sector_0 if idx % 4 in (0, 1)]
    assert groups[3].indices.tolist() == [idx for idx in sector_0 if idx % 4 in (2, 3)]
    assert groups[4].indices.tolist() == [idx for idx in sector_1 if idx % 4 in (0, 1)]
    assert groups[5].indices.tolist() == [idx for idx in sector_1 if idx % 4 in (2, 3)]
    records = component_group_records(model)
    assert records[0]["index_space"] == "tdbg_full_hamiltonian_basis"
    assert "q-sites" in records[0]["description"]


def test_atmg_model_declares_layer_component_groups() -> None:
    model = make_model("atmg", n_layers=4, n_shells=0)
    groups = component_groups(model)

    assert [group.name for group in groups] == ["layer_0", "layer_1", "layer_2", "layer_3"]
    assert [group.indices.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_compute_bands_includes_declared_component_group_metadata() -> None:
    model = make_model("atmg", n_layers=3, n_shells=0)
    bundle = compute_bands(model, n_bands=2, points_per_segment=1)

    assert bundle.basis_metadata["component_groups"] == [
        {"name": "layer_0", "indices": [0, 1]},
        {"name": "layer_1", "indices": [2, 3]},
        {"name": "layer_2", "indices": [4, 5]},
    ]


def test_public_run_hf_fails_explicitly_for_system_without_adapter() -> None:
    cfg = HFConfig(filling=0.0, mesh=(2, 2))
    model = make_model("htqg", theta_deg=2.25, n_shells=0, domain="abg")

    with pytest.raises(NotImplementedError, match=r"no run_hf\(config\) adapter"):
        run_hf(model, cfg)
