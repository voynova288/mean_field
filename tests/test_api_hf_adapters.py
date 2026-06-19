from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mean_field.api.hf as hf_api
from mean_field.api import HFConfig, HFResult, make_model, run_hf
from mean_field.api.hf import get_hf_adapter_info, list_hf_adapters, resolve_hf_adapter
from mean_field.core.contracts import HFRunResult as ContractHFRunResult
from mean_field.systems import tdbg as tdbg_system
from mean_field.systems.htg import HTGRunHFConfig, HTGSupercellRunHFConfig, InteractionParams
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


def test_public_hf_adapter_registry_exposes_post_run_converters_without_run_dispatch() -> None:
    adapters = {info.name: info for info in list_hf_adapters()}
    expected = {
        "tdbg_projected_hf_result_to_hf_run_result",
        "tdbg_explicit_projected_run_hf",
        "htg_hf_run_to_hf_run_result",
        "htg_hf_run_to_hf_result",
        "htg_explicit_primitive_run_hf",
        "htg_supercell_hf_run_to_hf_run_result",
        "htg_supercell_hf_run_to_hf_result",
        "htg_explicit_supercell_run_hf",
        "tbg_zero_field_hf_run_to_hf_run_result",
        "b0_hf_benchmark_run_to_hf_run_result",
        "rlg_hbn_hf_run_to_hf_run_result",
        "polshyn_wang_hf_bundle_to_hf_run_result",
    }
    run_adapters = {
        "tdbg_explicit_projected_run_hf",
        "htg_explicit_primitive_run_hf",
        "htg_explicit_supercell_run_hf",
    }

    assert expected <= set(adapters)
    for name in run_adapters:
        assert adapters[name].adapter_type == "run_hf"
        assert adapters[name].supports_run_hf_config is True
        assert adapters[name].requires_explicit_inputs
        assert adapters[name].run_hf_config_reason
    for name in expected - run_adapters:
        assert adapters[name].adapter_type in {"canonical_hf_run_result", "hf_result"}
        assert adapters[name].supports_run_hf_config is False
        assert ":" in adapters[name].import_path
        assert adapters[name].requires_explicit_inputs
        assert adapters[name].run_hf_config_reason


def test_public_hf_adapter_registry_filters_and_resolves_existing_helpers() -> None:
    htg_supercell = {info.name for info in list_hf_adapters(system_name="htg_supercell")}
    assert htg_supercell == {
        "htg_supercell_hf_run_to_hf_run_result",
        "htg_supercell_hf_run_to_hf_result",
        "htg_explicit_supercell_run_hf",
    }
    canonical = {info.name for info in list_hf_adapters(adapter_type="canonical_hf_run_result")}
    assert "tdbg_explicit_projected_run_hf" not in canonical
    assert "polshyn_wang_hf_bundle_to_hf_run_result" in canonical

    adapter = resolve_hf_adapter("htg_supercell_hf_run_to_hf_run_result")
    assert adapter.__name__ == "htg_supercell_hf_run_to_hf_run_result"
    assert adapter.__module__ == "mean_field.systems.htg.supercell_contracts"
    assert get_hf_adapter_info("tdbg_explicit_projected_run_hf").supports_run_hf_config is True
    assert get_hf_adapter_info("htg_explicit_primitive_run_hf").supports_run_hf_config is True
    assert "HTGRunHFConfig" in get_hf_adapter_info("htg_explicit_primitive_run_hf").run_hf_config_reason
    assert "htg_supercell_hf_run_to_hf_result" in hf_api.__all__

    with pytest.raises(KeyError, match="Unknown HF adapter"):
        get_hf_adapter_info("not_a_registered_hf_adapter")


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


def test_public_run_hf_htg_requires_explicit_system_config() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    cfg = HFConfig(
        filling=3.0,
        mesh=(1, 1),
        max_iter=1,
        density_convention="stored_delta",
        epsilon_r=8.0,
        dsc_nm=25.0,
    )

    with pytest.raises(NotImplementedError, match="explicit htg_config"):
        run_hf(model, cfg)


def test_public_run_hf_htg_primitive_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    interaction = InteractionParams(n_k=1, g_shells=0)
    cfg = HFConfig(
        filling=3.0,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.d_sc_nm,
    )
    htg_cfg = HTGRunHFConfig(
        nu=3.0,
        mesh_size=1,
        interaction=interaction,
        init_mode="bm",
        seed=2,
        max_iter=1,
        precision=1.0e-6,
        g_shells=0,
        use_numba=False,
    )

    result = run_hf(model, cfg, htg_config=htg_cfg)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "htg"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 2
    assert result.observables["public_run_hf_adapter"].endswith("run_htg_hf_config_adapter")
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


def test_public_run_hf_htg_supercell_explicit_config_attaches_canonical_contract_result() -> None:
    model = make_model("htg", theta_deg=1.8, n_shells=0)
    interaction = InteractionParams(n_k=1, g_shells=0)
    cfg = HFConfig(
        filling=3.5,
        mesh=(1, 1),
        max_iter=1,
        precision=1.0e-6,
        density_convention="stored_delta",
        epsilon_r=interaction.epsilon_r,
        dsc_nm=interaction.d_sc_nm,
    )
    htg_supercell_cfg = HTGSupercellRunHFConfig(
        primitive_nu=3.5,
        mesh_size=1,
        interaction=interaction,
        init_mode="bm",
        seed=1,
        max_iter=1,
        precision=1.0e-6,
        g_shells=0,
        use_numba=False,
    )

    result = run_hf(model, cfg, htg_supercell_config=htg_supercell_cfg)

    assert isinstance(result, HFResult)
    assert result.model.system_name == "htg_supercell"
    assert isinstance(result.canonical_run_result, ContractHFRunResult)
    assert result.state.seed == 1
    assert result.observables["supercell_area_ratio"] == 2
    assert result.observables["public_run_hf_adapter"].endswith("run_htg_supercell_hf_config_adapter")
    assert result.canonical_run_result.final_state.density.reference.metadata["raw_density_convention"] == "stored_delta"
    assert result.canonical_run_result.final_state.hamiltonian.metadata["supports_crpa"] is False


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
