from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mean_field import cli
from mean_field.api import HFConfig
from mean_field.systems.tdbg import TDBGProjectedHFConfig


def _tdbg_cli_config(output_dir: Path | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "workflow": "tdbg.projected_hf.explicit_config",
        "system": "tdbg",
        "tdbg_projected_hf": {
            "theta_deg": 1.38,
            "cut": 1.0,
            "mesh_size": 1,
            "paper_ud_ev": 0.09,
            "paper_ud_convention": "minus_xi_ud_over3",
            "stacking": "AB-BA",
            "window": {"name": "two_flat", "band_indices": None},
            "filling": 2,
            "interaction": {
                "include_intersite": False,
                "include_onsite": False,
                "hubbard_u_ev": 0.5,
                "epsilon_r": 10.0,
                "kappa_nm_inv": 0.05,
                "g_shells": None,
                "hartree_reference": "charge_neutral",
                "fock_density": "absolute",
                "onsite_valley_policy": "valley_diagonal",
                "drop_g0_hartree": False,
            },
            "precision": 1.0e-7,
            "max_iter": 1,
            "mix_fallback": None,
            "frac_shift": None,
            "orbital_zeeman_b_t": 0.0,
            "orbital_zeeman_delta_k_nm_inv": 1.0e-5,
        },
        "run": {"init_mode": "sp", "seed": 11},
    }
    if output_dir is not None:
        payload["result"] = {"output_dir": str(output_dir)}
    return payload


def test_cli_tdbg_projected_hf_dry_run_validates_config_without_compute(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tdbg_projected_hf.json"
    config_path.write_text(json.dumps(_tdbg_cli_config(tmp_path / "out")), encoding="utf-8")
    monkeypatch.setattr(cli, "run_hf", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_hf called")))
    monkeypatch.setattr(
        cli,
        "_ensure_not_running_compute_on_login_node",
        lambda workload_name: (_ for _ in ()).throw(AssertionError("compute guard called")),
    )

    rc = cli.main(["tdbg", "projected-hf", str(config_path), "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "workflow=tdbg.projected_hf.explicit_config" in out
    assert "theta_deg=1.38" in out
    assert "mesh_size=1" in out
    assert "init_mode=sp" in out
    assert f"output_dir={tmp_path / 'out'}" in out


def test_cli_tdbg_projected_hf_dispatches_public_adapter_and_saves_result(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "tdbg_projected_hf.json"
    config_path.write_text(json.dumps(_tdbg_cli_config()), encoding="utf-8")
    output_dir = tmp_path / "run"
    called: dict[str, object] = {}

    monkeypatch.setattr(cli, "_ensure_not_running_compute_on_login_node", lambda workload_name: called.update({"guard": workload_name}))

    def fake_make_model(system_name: str, **kwargs: object) -> SimpleNamespace:
        called["model_system"] = system_name
        called["model_kwargs"] = kwargs
        return SimpleNamespace(system_name=system_name)

    class FakeResult:
        def save(self, root: Path) -> Path:
            called["save_root"] = root
            root.mkdir(parents=True, exist_ok=True)
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            return manifest

    def fake_run_hf(model: object, hf_config: HFConfig, **kwargs: object) -> FakeResult:
        called["model"] = model
        called["hf_config"] = hf_config
        called["run_kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr(cli, "make_model", fake_make_model)
    monkeypatch.setattr(cli, "run_hf", fake_run_hf)

    rc = cli.main(["tdbg", "projected-hf", str(config_path), "--output-dir", str(output_dir)])

    assert rc == 0
    assert called["guard"] == "TDBG projected HF"
    assert called["model_system"] == "tdbg"
    assert called["model_kwargs"]["theta_deg"] == 1.38  # type: ignore[index]
    assert called["model_kwargs"]["cut"] == 1.0  # type: ignore[index]
    hf_config = called["hf_config"]
    assert isinstance(hf_config, HFConfig)
    assert hf_config.mesh == (1, 1)
    assert hf_config.density_convention == "projector"
    run_kwargs = called["run_kwargs"]
    assert isinstance(run_kwargs["tdbg_config"], TDBGProjectedHFConfig)  # type: ignore[index]
    assert run_kwargs["init_mode"] == "sp"  # type: ignore[index]
    assert run_kwargs["seed"] == 11  # type: ignore[index]
    assert called["save_root"] == output_dir
    assert f"manifest={output_dir / 'manifest.json'}" in capsys.readouterr().out


def test_cli_tdbg_projected_hf_rejects_out_of_scope_system(tmp_path: Path) -> None:
    payload = _tdbg_cli_config()
    payload["system"] = "htqg"
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        cli.main(["tdbg", "projected-hf", str(config_path), "--dry-run"])
    except ValueError as exc:
        assert "system='tdbg'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("out-of-scope system was accepted")
