from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mean_field.api import HFConfig, HFResult, make_model, run_hf
from mean_field.core.contracts import HFRunResult as ContractHFRunResult
from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.tdbg import TDBGInteractionSettings, TDBGProjectedHFConfig, TDBGProjectedWindow


def _tiny_tdbg_config() -> TDBGProjectedHFConfig:
    return TDBGProjectedHFConfig(
        theta_deg=1.38,
        cut=1.0,
        mesh_size=1,
        paper_ud_ev=0.09,
        paper_ud_convention="minus_xi_ud_over3",
        window=TDBGProjectedWindow("two_flat"),
        filling=2,
        interaction=TDBGInteractionSettings(include_intersite=False, include_onsite=False),
        precision=1.0e-7,
        max_iter=1,
    )


def test_public_run_hf_tbg_bm_requires_explicit_system_workflow() -> None:
    model = make_model("tbg", variant="zero_field_bm", theta_deg=1.2, lg=1)
    cfg = HFConfig(filling=0, mesh=(1, 1), max_iter=1)

    with pytest.raises(NotImplementedError, match=r"no run_hf\(config\) adapter"):
        run_hf(model, cfg)


def test_public_run_hf_tdbg_requires_explicit_projected_config() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")

    with pytest.raises(NotImplementedError, match="explicit tdbg_config"):
        run_hf(model, cfg)


def test_public_run_hf_tdbg_explicit_config_dispatches_without_guessing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")
    tdbg_cfg = _tiny_tdbg_config()
    calls: dict[str, object] = {}

    def fake_build_data(config: TDBGProjectedHFConfig) -> SimpleNamespace:
        calls["build_config"] = config
        return SimpleNamespace(config=config)

    class FakeTDBGResult:
        def to_summary_dict(self) -> dict[str, object]:
            return {
                "init_mode": "sp",
                "seed": 7,
                "converged": False,
                "exit_reason": "max_iter",
                "iterations": 1,
            }

    def fake_run(data: SimpleNamespace, *, init_mode: str, seed: int = 1) -> FakeTDBGResult:
        calls["run_data"] = data
        calls["init_mode"] = init_mode
        calls["seed"] = seed
        return FakeTDBGResult()

    monkeypatch.setattr(tdbg_system, "build_tdbg_projected_hf_data", fake_build_data)
    monkeypatch.setattr(tdbg_system, "run_tdbg_projected_hf", fake_run)

    result = run_hf(model, cfg, tdbg_config=tdbg_cfg, init_mode="sp", seed=7)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "tdbg"
    assert result.state.to_summary_dict()["iterations"] == 1
    assert result.observables["init_mode"] == "sp"
    assert calls["build_config"] is tdbg_cfg
    assert calls["init_mode"] == "sp"
    assert calls["seed"] == 7
    assert result.artifacts is not None
    assert result.artifacts.metadata["workflow"] == "tdbg.projected_hf.explicit_config"
    assert result.artifacts.conventions.to_dict()["energy_unit"] == "eV"  # type: ignore[union-attr]
    assert result.artifacts.conventions.to_dict()["density_convention"] == "projector"  # type: ignore[union-attr]
    assert result.canonical_run_result is None


def test_public_run_hf_tdbg_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7, density_convention="projector")

    result = run_hf(model, cfg, tdbg_config=_tiny_tdbg_config(), init_mode="sp", seed=7)

    assert isinstance(result, HFResult)
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.canonical_run_result.final_state.density.reference.scheme == "CN"
    np.testing.assert_allclose(
        result.canonical_run_result.final_state.density.projector,
        result.state.run.state.density,
    )
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_tdbg_rejects_mismatched_generic_config() -> None:
    model = make_model("tdbg", theta_deg=1.38, cut=1.0)
    cfg = HFConfig(filling=2, mesh=(1, 1), max_iter=1, precision=1.0e-7)

    with pytest.raises(ValueError, match="density_convention='projector'"):
        run_hf(model, cfg, tdbg_config=_tiny_tdbg_config(), init_mode="sp")
