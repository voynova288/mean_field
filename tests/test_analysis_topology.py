from __future__ import annotations

import numpy as np

from analysis.topology import (
    WavefunctionIndex,
    compute_lattice_topology,
    compute_lattice_topology_for_state_groups,
    split_state_indices_by_direct_gaps,
)


def _qiwuzhang_wavefunctions(mesh: int, mass: float) -> tuple[np.ndarray, np.ndarray]:
    wavefunctions = np.empty((mesh, mesh, 2, 2), dtype=np.complex128)
    energies = np.empty((2, mesh, mesh), dtype=float)
    for ix in range(mesh):
        kx = 2.0 * np.pi * ix / mesh
        for iy in range(mesh):
            ky = 2.0 * np.pi * iy / mesh
            dz = mass + np.cos(kx) + np.cos(ky)
            hamiltonian = np.asarray(
                [
                    [dz, np.sin(kx) - 1j * np.sin(ky)],
                    [np.sin(kx) + 1j * np.sin(ky), -dz],
                ],
                dtype=np.complex128,
            )
            vals, vecs = np.linalg.eigh(hamiltonian)
            energies[:, ix, iy] = vals
            wavefunctions[ix, iy] = vecs
    return wavefunctions, energies


def test_fhs_chern_for_qiwuzhang_single_bands_and_subspace() -> None:
    wavefunctions, _energies = _qiwuzhang_wavefunctions(mesh=21, mass=1.0)

    lower = compute_lattice_topology(
        wavefunctions,
        0,
        index=WavefunctionIndex(indices=(0,), role="band", labels=("lower",), system="qiwuzhang"),
    )
    upper = compute_lattice_topology(wavefunctions, 1)
    full_subspace = compute_lattice_topology(wavefunctions, (0, 1), link_method="determinant")

    assert lower.rounded_chern_number == 1
    assert lower.is_nearly_integer
    assert upper.rounded_chern_number == -1
    assert np.isclose(lower.chern_number + upper.chern_number, 0.0, atol=1.0e-12)
    assert full_subspace.rounded_chern_number == 0
    assert full_subspace.is_nearly_integer
    assert lower.min_link_magnitude > 0.9


def test_fhs_chern_distinguishes_trivial_and_topological_mass_regions() -> None:
    topological, _ = _qiwuzhang_wavefunctions(mesh=21, mass=-1.0)
    trivial, _ = _qiwuzhang_wavefunctions(mesh=21, mass=3.0)

    assert compute_lattice_topology(topological, 0).rounded_chern_number == -1
    assert compute_lattice_topology(trivial, 0).rounded_chern_number == 0


def test_gap_grouping_and_group_topology_api() -> None:
    wavefunctions, energies = _qiwuzhang_wavefunctions(mesh=15, mass=1.0)
    groups = split_state_indices_by_direct_gaps(energies, (0, 1), min_gap=0.5)

    assert groups == ((0,), (1,))

    results = compute_lattice_topology_for_state_groups(
        wavefunctions,
        groups,
        base_index=WavefunctionIndex(indices=(0, 1), labels=("lower", "upper"), system="qiwuzhang"),
    )

    assert tuple(result.rounded_chern_number for result in results) == (1, -1)
    assert all(result.is_nearly_integer for result in results)
