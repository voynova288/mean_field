from __future__ import annotations

from types import SimpleNamespace

from mean_field.systems.RnG_hBN.model import RLGhBNModel
from mean_field.systems.htg.model import HTGModel
from mean_field.systems.tdbg.model import TDBGModel
from mean_field.systems.tmbg.model import TMBGModel


def _patch_state_builder(monkeypatch, module_path: str, calls: dict[str, object], sentinel: object) -> None:
    module = __import__(module_path, fromlist=["fhs_state_on_grid"])

    def fake(mesh_size, lattice, params, band_indices, **kwargs):
        calls.update({"mesh_size": mesh_size, "lattice": lattice, "params": params, "band_indices": band_indices, "kwargs": kwargs})
        return sentinel

    monkeypatch.setattr(module, "fhs_state_on_grid", fake)


def test_tmbg_model_fhs_state_on_grid_delegates_to_state_builder(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sentinel = object()
    _patch_state_builder(monkeypatch, "mean_field.systems.tmbg.topology", calls, sentinel)
    lattice, params = object(), object()

    result = TMBGModel(lattice, params).fhs_state_on_grid(7, 3, valley=-1, endpoint=False, frac_shift=(0.25, 0.5))

    assert result is sentinel
    assert calls == {"mesh_size": 7, "lattice": lattice, "params": params, "band_indices": 3, "kwargs": {"valley": -1, "endpoint": False, "frac_shift": (0.25, 0.5)}}


def test_tdbg_model_fhs_state_on_grid_uses_params_valley_by_default(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sentinel = object()
    _patch_state_builder(monkeypatch, "mean_field.systems.tdbg.topology", calls, sentinel)
    lattice, params = object(), SimpleNamespace(valley=-1)

    result = TDBGModel(lattice, params).fhs_state_on_grid(9, 1, boundary_sewing=False)

    assert result is sentinel
    assert calls["kwargs"] == {"boundary_sewing": False, "valley": -1}
    TDBGModel(lattice, params).fhs_state_on_grid(9, 1, valley=1)
    assert calls["kwargs"] == {"valley": 1}


def test_rlg_hbn_model_fhs_state_on_grid_delegates_builder_specific_kwargs(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sentinel = object()
    _patch_state_builder(monkeypatch, "mean_field.systems.RnG_hBN.topology", calls, sentinel)
    lattice, params = object(), object()

    result = RLGhBNModel(lattice, params).fhs_state_on_grid(11, 4, use_boundary_sewing=False, paper_orientation=True)

    assert result is sentinel
    assert calls == {"mesh_size": 11, "lattice": lattice, "params": params, "band_indices": 4, "kwargs": {"use_boundary_sewing": False, "paper_orientation": True}}


def test_htg_model_fhs_state_on_grid_delegates_to_state_builder(monkeypatch) -> None:
    calls: dict[str, object] = {}
    sentinel = object()
    _patch_state_builder(monkeypatch, "mean_field.systems.htg.topology", calls, sentinel)
    lattice, params = object(), object()

    result = HTGModel(lattice, params).fhs_state_on_grid(13, (20, 21), d_top=1.0j, d_bot=-0.5j, boundary_sewing=False)

    assert result is sentinel
    assert calls == {"mesh_size": 13, "lattice": lattice, "params": params, "band_indices": (20, 21), "kwargs": {"d_top": 1.0j, "d_bot": -0.5j, "boundary_sewing": False}}
