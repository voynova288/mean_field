from __future__ import annotations

import pytest

from mean_field.api import (
    CRPAConfig,
    TDHFConfig,
    compute_crpa,
    get_crpa_adapter_info,
    get_tdhf_adapter_info,
    list_crpa_adapters,
    list_tdhf_adapters,
    resolve_crpa_adapter,
    resolve_tdhf_adapter,
    run_tdhf,
)


class _DummyCRPA:
    def compute_crpa(self, config: CRPAConfig, **kwargs: object) -> tuple[CRPAConfig, dict[str, object]]:
        return config, dict(kwargs)


class _DummyTDHF:
    def run_tdhf(self, config: TDHFConfig, **kwargs: object) -> tuple[TDHFConfig, dict[str, object]]:
        return config, dict(kwargs)


def test_crpa_registry_exposes_tbg_workflow_adapter() -> None:
    names = [item.name for item in list_crpa_adapters()]
    assert "tbg_workflow" in names
    info = get_crpa_adapter_info("tbg_workflow")
    assert info.system_name == "tbg"
    assert callable(resolve_crpa_adapter("tbg_workflow"))


def test_crpa_facade_preserves_object_hook_and_requires_explicit_adapter() -> None:
    cfg = CRPAConfig(q_mesh=3)
    assert compute_crpa(_DummyCRPA(), cfg, marker="ok") == (cfg, {"marker": "ok"})
    with pytest.raises(NotImplementedError, match="adapter='tbg_workflow'"):
        compute_crpa(object(), cfg)
    with pytest.raises(ValueError, match="params and theta_deg"):
        compute_crpa(object(), cfg, adapter="tbg_workflow")


def test_tdhf_registry_exposes_rlg_hbn_adapters() -> None:
    names = [item.name for item in list_tdhf_adapters(system_name="rlg_hbn")]
    assert names == ["rlg_hbn_q0", "rlg_hbn_finite_q"]
    assert get_tdhf_adapter_info("rlg_hbn_q0").system_name == "rlg_hbn"
    assert callable(resolve_tdhf_adapter("rlg_hbn_q0"))


def test_tdhf_facade_preserves_object_hook_and_validates_rlg_hbn_inputs() -> None:
    cfg = TDHFConfig(q_sector="q0")
    assert run_tdhf(_DummyTDHF(), cfg, marker="ok") == (cfg, {"marker": "ok"})
    with pytest.raises(NotImplementedError, match="registered adapter"):
        run_tdhf(object(), cfg)
    with pytest.raises(ValueError, match="canonical_hf"):
        run_tdhf(object(), cfg, adapter="rlg_hbn_q0")
