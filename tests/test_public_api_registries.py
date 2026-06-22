from __future__ import annotations

import pytest

from mean_field.api import (
    TDHFConfig,
    get_model_adapter_info,
    get_tdhf_adapter_info,
    list_model_adapters,
    list_tdhf_adapters,
    resolve_model_adapter,
    resolve_tdhf_adapter,
    run_tdhf,
)


class _DummyTDHF:
    def run_tdhf(self, config: TDHFConfig, **kwargs: object) -> tuple[TDHFConfig, dict[str, object]]:
        return config, dict(kwargs)


def test_model_registry_preserves_public_aliases() -> None:
    names = [item.name for item in list_model_adapters()]
    assert names == ["htg", "htqg", "rlg_hbn", "tbg", "tdbg", "tmbg", "atmg"]
    assert get_model_adapter_info("helical-trilayer-graphene").name == "htg"
    assert get_model_adapter_info("rng-hbn").name == "rlg_hbn"
    assert callable(resolve_model_adapter("twisted_bilayer_graphene"))


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
