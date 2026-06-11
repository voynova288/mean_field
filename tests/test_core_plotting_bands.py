from __future__ import annotations

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.core.plotting.bands import kpath_node_ticks, write_kpath_band_tsv, write_kpath_nodes_tsv


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
