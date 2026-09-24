from __future__ import annotations

import numpy as np

from ei_mf.kane_band_input import (
    read_kdotpy_e1_h1,
    read_kdotpy_e1_h1_subband_overlap,
    write_stageB_band_csv,
)


def _point(k: float, energies: list[float], gamma6: list[float], gamma8h: list[float]) -> str:
    return f"""
    <momentum k="{k}">
      <energies>{' '.join(map(str, energies))}</energies>
      <bandindex>-2 -1 1 2</bandindex>
      <observable q="gamma6">{' '.join(map(str, gamma6))}</observable>
      <observable q="gamma8h">{' '.join(map(str, gamma8h))}</observable>
    </momentum>
    """


def test_kdotpy_orbital_projection_diabatizes_character_swap(tmp_path) -> None:
    # At low k the E1 pair is energy ordered below H1; at high k the orbital
    # characters swap energy order, as at an avoided E1/H1 crossing.
    xml = "<datafile><dispersion>" + "".join(
        [
            _point(0.00, [-2.1, -1.9, 1.9, 2.1], [0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.9, 0.9]),
            _point(0.01, [-1.1, -0.9, 0.9, 1.1], [0.8, 0.8, 0.2, 0.2], [0.2, 0.2, 0.8, 0.8]),
            _point(0.02, [-0.2, -0.1, 0.1, 0.2], [0.3, 0.3, 0.7, 0.7], [0.7, 0.7, 0.3, 0.3]),
            _point(0.03, [-1.1, -0.9, 0.9, 1.1], [0.1, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.1]),
        ]
    ) + "</dispersion></datafile>"
    path = tmp_path / "kane.xml"
    path.write_text(xml)
    data = read_kdotpy_e1_h1(path)
    assert np.allclose(data.electron_energy_mev, [-1.6, -0.6, 0.06, 0.8])
    assert np.allclose(data.valence_energy_mev, [1.6, 0.6, -0.06, -0.8])
    legacy = read_kdotpy_e1_h1(path, tracking_mode="max_character")
    assert np.allclose(legacy.electron_energy_mev, [-2.0, -1.0, 0.15, 1.0])
    assert np.allclose(legacy.valence_energy_mev, [2.0, 1.0, -0.15, -1.0])
    eps_a, eps_b = data.aligned_stageB_bands(0.01)
    assert np.isclose(eps_a[1], 0.0)
    assert np.isclose(eps_b[1], 0.0)
    out = tmp_path / "bands.csv"
    write_stageB_band_csv(out, data, k_cross_nm_inv=0.01)
    saved = np.genfromtxt(out, delimiter=",", names=True)
    assert "epsilon_a_meV" in saved.dtype.names
    assert "epsilon_b_meV" in saved.dtype.names


def test_subband_overlap_projection_recovers_frozen_basis_diagonal(tmp_path) -> None:
    indices = (-2, -1, 1, 2)
    k_values = (0.0, 0.01, 0.02, 0.03)
    angles = (0.0, 0.2, 0.4, 0.6)
    energies = np.array([-1.0, -1.0, 3.0, 3.0])
    points = []
    for k, theta in zip(k_values, angles):
        points.append(
            f'<momentum k="{k}"><energies>{" ".join(map(str, energies))}</energies>'
            '<bandindex>-2 -1 1 2</bandindex></momentum>'
        )
        c, s = np.cos(theta), np.sin(theta)
        components = {
            -2: (c, 0.0, s, 0.0),
            -1: (0.0, c, 0.0, s),
            1: (-s, 0.0, c, 0.0),
            2: (0.0, -s, 0.0, c),
        }
        for index in indices:
            filename = tmp_path / f"wfstoy_{k:.3f}_0.{index}.csv"
            ep, em, hp, hm = components[index]
            filename.write_text(
                'z,"Γ6,+1/2","Γ6,+1/2","Γ6,-1/2","Γ6,-1/2",'
                '"Γ8,+3/2","Γ8,+3/2","Γ8,-3/2","Γ8,-3/2"\n'
                'nm,Re ψ_i,Im ψ_i,Re ψ_i,Im ψ_i,Re ψ_i,Im ψ_i,Re ψ_i,Im ψ_i\n'
                f"0,{ep},0,{em},0,{hp},0,{hm},0\n"
                "1,0,0,0,0,0,0,0,0\n"
            )
    xml_path = tmp_path / "overlap.xml"
    xml_path.write_text("<datafile><dispersion>" + "".join(points) + "</dispersion></datafile>")
    data = read_kdotpy_e1_h1_subband_overlap(
        xml_path, tmp_path, output_id="toy"
    )
    expected_e = -np.cos(angles) ** 2 + 3.0 * np.sin(angles) ** 2
    expected_h = -np.sin(angles) ** 2 + 3.0 * np.cos(angles) ** 2
    assert np.allclose(data.electron_energy_mev, expected_e)
    assert np.allclose(data.valence_energy_mev, expected_h)
    assert np.allclose(data.electron_gamma6_weight, 1.0)
    assert np.allclose(data.hole_gamma8h_weight, 1.0)
