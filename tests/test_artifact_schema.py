from __future__ import annotations

import json

import numpy as np

from mean_field.api import ArtifactManifest, ConventionBundle, HFConfig, HFResult, HFState, ModelRecord, load_result, required_artifact_files


def test_required_artifact_schema_names_are_stable() -> None:
    required = required_artifact_files()

    assert required == (
        "manifest.json",
        "model.json",
        "config.yaml",
        "conventions.json",
        "environment.json",
        "validation.json",
        "observables.json",
    )


def test_convention_bundle_serializes_density_axis_contract() -> None:
    payload = ConventionBundle().to_dict()

    assert payload["energy_unit"] == "meV"
    assert payload["momentum_unit"] == "nm^-1"
    assert payload["density_convention"] == "stored_delta"
    assert payload["density_axis_order"] == "abk"
    assert payload["valley_labels"] == {"K": 1, "Kprime": -1}


def test_hf_result_save_writes_public_manifest_files(tmp_path) -> None:
    model = ModelRecord(system_name="toy")
    cfg = HFConfig(filling=0.0, mesh=(1, 1))
    state = HFState(density=np.zeros((1, 1, 1), dtype=np.complex128))
    result = HFResult(model=model, config=cfg, state=state, observables={"gap_mev": 1.0})

    manifest_path = result.save(tmp_path)
    loaded = load_result(tmp_path)

    assert manifest_path == tmp_path / "manifest.json"
    assert json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))["system_name"] == "toy"
    assert json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8"))["mesh"] == [1, 1]
    assert loaded.manifest["root"] == str(tmp_path)
