from __future__ import annotations

import csv
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ei_mf.kane_momentum_profiles import build_fixed_reference_momentum_profiles

from ei_mf.wavefunction_form_factor import (
    KaneLayerProfiles,
    KaneMomentumLayerProfiles,
    _pair_projector_profiles_from_quartet,
)


def test_delta_layer_form_factors_reduce_to_point_layers() -> None:
    z = np.arange(-2.0, 13.0, 1.0)
    rho_e = np.zeros_like(z)
    rho_h = np.zeros_like(z)
    rho_e[np.where(z == 0.0)[0][0]] = 1.0
    rho_h[np.where(z == 10.0)[0][0]] = 1.0
    profiles = KaneLayerProfiles(z, rho_e, rho_h)
    q = np.array([0.0, 0.05, 0.1, 0.2])
    Faa, Fbb, Fab = profiles.form_factors(q)
    assert np.allclose(Faa, 1.0)
    assert np.allclose(Fbb, 1.0)
    assert np.allclose(Fab, np.exp(-10.0 * q))
    moments = profiles.moments()
    assert np.isclose(moments["centroid_separation_nm"], 10.0)


def test_profile_npz_roundtrip(tmp_path) -> None:
    z = np.linspace(-2.0, 2.0, 9)
    rho_e = np.exp(-0.5 * ((z + 0.5) / 0.4) ** 2)
    rho_h = np.exp(-0.5 * ((z - 0.5) / 0.4) ** 2)
    dz = z[1] - z[0]
    rho_e /= np.sum(rho_e) * dz
    rho_h /= np.sum(rho_h) * dz
    profiles = KaneLayerProfiles(z, rho_e, rho_h, ("a.csv", "b.csv"))
    path = tmp_path / "profiles.npz"
    profiles.save_npz(path)
    loaded = KaneLayerProfiles.from_npz(path)
    assert np.allclose(loaded.z_nm, z)
    assert np.allclose(loaded.electron_density_nm_inv, rho_e)
    assert np.allclose(loaded.hole_density_nm_inv, rho_h)
    assert loaded.source_files == ("a.csv", "b.csv")


def test_pair_projector_profiles_are_u4_covariant() -> None:
    z_weights = np.ones(4)
    frame = np.zeros((4, 8, 4), dtype=np.complex128)
    frame[0, 0, 0] = 1.0
    frame[1, 1, 1] = 1.0
    frame[2, 2, 2] = 1.0
    frame[3, 5, 3] = 1.0
    eig, _electron_pair, _hole_pair, rho_e, rho_h = (
        _pair_projector_profiles_from_quartet(frame, z_weights)
    )

    rng = np.random.default_rng(19)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary, _r = np.linalg.qr(raw)
    rotated = np.einsum("zma,ab->zmb", frame, unitary)
    eig_rot, _electron_rot, _hole_rot, rho_e_rot, rho_h_rot = (
        _pair_projector_profiles_from_quartet(rotated, z_weights)
    )
    assert np.allclose(eig, [1.0, 1.0, 0.0, 0.0], atol=1e-13)
    assert np.allclose(eig_rot, eig, atol=1e-13)
    assert np.allclose(rho_e_rot, rho_e, atol=1e-13)
    assert np.allclose(rho_h_rot, rho_h, atol=1e-13)


def test_momentum_profiles_interpolate_positive_normalized_and_roundtrip(tmp_path) -> None:
    k = np.array([0.0, 0.1, 0.2])
    z = np.linspace(-2.0, 2.0, 9)
    electron = np.stack(
        [np.exp(-0.5 * ((z + shift) / 0.5) ** 2) for shift in (0.8, 0.5, 0.2)]
    )
    hole = np.stack(
        [np.exp(-0.5 * ((z - shift) / 0.6) ** 2) for shift in (0.8, 0.5, 0.2)]
    )
    dz = z[1] - z[0]
    electron /= np.sum(electron, axis=1)[:, None] * dz
    hole /= np.sum(hole, axis=1)[:, None] * dz
    profiles = KaneMomentumLayerProfiles(
        k,
        z,
        electron,
        hole,
        np.tile(np.array([0.8, 0.7, 0.1, 0.0]), (3, 1)),
        np.ones(2),
        np.ones(2),
        np.ones(2),
        "source.xml",
        "wavefunctions",
        "output",
        (-2, -1, 1, 2),
    )
    profiles.validate()
    target = np.array([0.025, 0.075, 0.125, 0.175])
    electron_i, hole_i = profiles.densities_at(target)
    assert np.all(electron_i >= 0.0)
    assert np.all(hole_i >= 0.0)
    assert np.allclose(np.sum(electron_i, axis=1) * dz, 1.0, atol=1e-12)
    assert np.allclose(np.sum(hole_i, axis=1) * dz, 1.0, atol=1e-12)

    path = tmp_path / "momentum_profiles.npz"
    profiles.save_npz(path)
    loaded = KaneMomentumLayerProfiles.from_npz(path)
    assert np.array_equal(loaded.k_nm_inv, profiles.k_nm_inv)
    assert np.allclose(loaded.electron_density_nm_inv, electron)
    assert np.allclose(loaded.hole_density_nm_inv, hole)
    assert loaded.active_band_indices == (-2, -1, 1, 2)


def _write_kdotpy_state(path, z: np.ndarray, amplitudes: np.ndarray) -> None:
    labels = ["z"]
    units = ["nm"]
    orbital_labels = (
        "Γ6,+1/2",
        "Γ6,-1/2",
        "Γ8,+3/2",
        "Γ8,+1/2",
        "Γ8,-1/2",
        "Γ8,-3/2",
        "Γ7,+1/2",
        "Γ7,-1/2",
    )
    for label in orbital_labels:
        labels.extend((label, label))
        units.extend(("Re ψ_i", "Im ψ_i"))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        writer.writerow(units)
        for iz, z_value in enumerate(z):
            row = [z_value]
            for value in amplitudes[iz]:
                row.extend((value.real, value.imag))
            writer.writerow(row)


def test_fixed_reference_builder_repairs_only_degenerate_pairs_and_tracks_u4(tmp_path) -> None:
    z = np.arange(4.0)
    reference = np.zeros((4, 8, 4), dtype=np.complex128)
    reference[0, 0, 0] = 1.0
    reference[0, 0, 1] = 0.2
    reference[1, 1, 1] = np.sqrt(0.96)
    reference[2, 2, 2] = 1.0
    reference[2, 2, 3] = 0.1
    reference[3, 5, 3] = np.sqrt(0.99)
    # The source repair at Gamma produces an orthonormal basis for each fixed
    # pair span. Use those spans to generate arbitrary full-quartet gauges at
    # later k points.
    electron, _r = np.linalg.qr(reference[:, :, :2].reshape(-1, 2))
    hole, _r = np.linalg.qr(reference[:, :, 2:].reshape(-1, 2))
    orthogonal = np.concatenate(
        (electron.reshape(4, 8, 2), hole.reshape(4, 8, 2)), axis=2
    )
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary, _r = np.linalg.qr(raw)
    frames = (reference, np.einsum("zma,ab->zmb", orthogonal, unitary), orthogonal[:, :, [1, 0, 3, 2]])
    k_values = (0.0, 0.1, 0.2)
    indices = (13, 14, 15, 16)
    root = ET.Element("root")
    dispersion = ET.SubElement(root, "dispersion")
    for ik, (k_value, frame) in enumerate(zip(k_values, frames)):
        point = ET.SubElement(dispersion, "momentum", k=str(k_value))
        ET.SubElement(point, "bandindex").text = "13 14 15 16"
        ET.SubElement(point, "energies").text = (
            "-1 -1 1 1" if ik == 0 else "0 0 0 0"
        )
        for column, band_index in enumerate(indices):
            _write_kdotpy_state(
                tmp_path / f"wfsdemo_{k_value:.3f}_0.{band_index}.csv",
                z,
                frame[:, :, column],
            )
    xml_path = tmp_path / "outputdemo.xml"
    ET.ElementTree(root).write(xml_path)
    profiles = build_fixed_reference_momentum_profiles(
        xml_path,
        tmp_path,
        output_id="demo",
        active_band_indices=indices,
    )
    assert profiles.pair_construction == "fixed_reference_polar_lowdin"
    assert np.allclose(profiles.fixed_reference_principal_cos2, 1.0, atol=2e-13)
    assert np.max(profiles.raw_quartet_orthonormality_error) > 0.1
    assert np.allclose(
        profiles.electron_density_nm_inv,
        profiles.electron_density_nm_inv[0],
        atol=2e-13,
    )
    assert np.allclose(
        profiles.hole_density_nm_inv,
        profiles.hole_density_nm_inv[0],
        atol=2e-13,
    )

    # The same nonorthogonal pair must fail closed if its energies are not
    # exactly degenerate.
    first = dispersion.findall("momentum")[0]
    first.find("energies").text = "-1 -0.5 1 1"
    ET.ElementTree(root).write(xml_path)
    with pytest.raises(ValueError, match="nonorthogonal nondegenerate source pair"):
        build_fixed_reference_momentum_profiles(
            xml_path,
            tmp_path,
            output_id="demo",
            active_band_indices=indices,
        )
