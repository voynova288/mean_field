from __future__ import annotations

from mean_field.api import (
    get_model_adapter_info,
    list_model_adapters,
    resolve_model_adapter,
)


def test_model_registry_preserves_public_aliases() -> None:
    names = [item.name for item in list_model_adapters()]
    assert names == ["htg", "rlg_hbn", "tbg", "tdbg", "tmbg"]
    assert get_model_adapter_info("helical-trilayer-graphene").name == "htg"
    assert get_model_adapter_info("rng-hbn").name == "rlg_hbn"
    assert callable(resolve_model_adapter("twisted_bilayer_graphene"))
