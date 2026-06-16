from __future__ import annotations

import numpy as np

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


def test_public_run_hf_fails_explicitly_until_system_adapter_exists() -> None:
    cfg = HFConfig(filling=0.0, mesh=(2, 2))
    model = make_model("htg", theta_deg=1.5, n_shells=0)

    try:
        run_hf(model, cfg)
    except NotImplementedError as exc:
        assert "run_hf(config) adapter" in str(exc)
    else:  # pragma: no cover - future adapters may implement this
        raise AssertionError("HTG unexpectedly exposed the public run_hf adapter; update this contract test")
