from __future__ import annotations

import importlib

import pytest

RESTORED_THIN_WRAPPERS = (
    "mean_field.systems.tmbg.topology",
    "mean_field.systems.tdbg.topology",
    "mean_field.systems.RnG_hBN.topology",
    "mean_field.systems.htg.topology",
)

@pytest.mark.parametrize("module_name", RESTORED_THIN_WRAPPERS)
def test_restored_thin_system_topology_wrappers_expose_common_entrypoints(module_name: str) -> None:
    module = importlib.import_module(module_name)

    for name in (
        "compute_topology_from_eigenvectors",
        "compute_topology_from_grid_result",
        "compute_topology_on_grid",
    ):
        assert callable(getattr(module, name))
