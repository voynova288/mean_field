from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

import mean_field.core.plotting as plotting
from mean_field.core.lattice import KPath
from mean_field.core.plotting.bands import kpath_node_ticks, write_kpath_band_tsv, write_kpath_nodes_tsv


def test_core_plotting_root_is_an_empty_namespace() -> None:
    init_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "mean_field"
        / "core"
        / "plotting"
        / "__init__.py"
    )
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    assert len(tree.body) == 2
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)
    assert isinstance(tree.body[1], ast.Assign)
    assert len(tree.body[1].targets) == 1
    assert isinstance(tree.body[1].targets[0], ast.Name)
    assert tree.body[1].targets[0].id == "__all__"
    assert ast.literal_eval(tree.body[1].value) == ()
    assert plotting.__all__ == ()
    assert not hasattr(plotting, "plot_band_columns")


def _path() -> KPath:
    return KPath(
        kvec=np.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
        kdist=np.asarray([0.0, 1.0], dtype=float),
        labels=("Gamma", "K"),
        node_indices=(1, 2),
    )


def test_core_plotting_band_namespace_exposes_path_helpers(tmp_path) -> None:
    path = _path()
    ticks, labels = kpath_node_ticks(path, label_map={"Gamma": "Γ"})
    assert ticks == [0.0, 1.0]
    assert labels == ["Γ", "K"]

    band_path = tmp_path / "bands.tsv"
    nodes_path = tmp_path / "nodes.tsv"
    write_kpath_band_tsv(
        band_path,
        kdist=path.kdist,
        energies=np.asarray([[0.0, 1.0], [0.5, 1.5]], dtype=float),
        band_labels=("b0", "b1"),
    )
    write_kpath_nodes_tsv(nodes_path, path)

    assert band_path.read_text(encoding="utf-8").splitlines()[0] == "k_dist\tb0\tb1"
    assert nodes_path.read_text(encoding="utf-8").splitlines()[0] == "label\tindex\tk_dist\tkx\tky"
