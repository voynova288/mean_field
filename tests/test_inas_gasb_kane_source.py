from __future__ import annotations

import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mean_field.systems.inas_gasb.conventions import E1H1BasisSpec
from mean_field.systems.inas_gasb.kane_source import (
    CANONICAL_KANE_ORBITALS,
    _active_time_reversal_unitary_from_frame,
    _canonicalize_gamma_e1h1_frame,
    _repair_exactly_degenerate_pairs,
    _sha256,
    _wavefunction_manifest_sha256,
    _weighted_gram,
    _xml_declares_split_zero,
    load_split_zero_kdotpy_radial_kane4_bundle,
    read_kdotpy_wavefunction_csv,
)


def _canonical_frame() -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(2, dtype=float)
    frame = np.zeros((2, 8, 4), dtype=np.complex128)
    frame[0, 0, 0] = 1.0
    frame[0, 1, 1] = 1.0
    frame[0, 2, 2] = 1.0
    frame[0, 5, 3] = 1.0
    return frame, weights


def _random_unitary(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    unitary, _ = np.linalg.qr(matrix)
    return unitary


def test_exactly_degenerate_pair_repair_is_pair_local() -> None:
    frame, weights = _canonical_frame()
    mixed = frame.copy()
    mixed[:, :, 1] = 0.25 * frame[:, :, 0] + frame[:, :, 1]
    mixed[:, :, 1] /= np.sqrt(1.0 + 0.25**2)
    mixed[:, :, 3] = -0.15j * frame[:, :, 2] + frame[:, :, 3]
    mixed[:, :, 3] /= np.sqrt(1.0 + 0.15**2)

    repaired, diagnostics = _repair_exactly_degenerate_pairs(
        mixed,
        np.asarray((1.0, 1.0, 2.0, 2.0)),
        weights,
    )

    assert diagnostics["raw_quartet_orthonormality_error"] > 0.1
    assert np.max(np.abs(_weighted_gram(repaired, weights) - np.eye(4))) < 1e-12
    assert np.max(np.abs(_weighted_gram(repaired[:, :, :2], weights) - np.eye(2))) < 1e-12
    assert np.max(np.abs(_weighted_gram(repaired[:, :, 2:], weights) - np.eye(2))) < 1e-12


def test_nonorthogonal_nondegenerate_pair_is_rejected() -> None:
    frame, weights = _canonical_frame()
    frame[:, :, 1] = 0.2 * frame[:, :, 0] + frame[:, :, 1]
    frame[:, :, 1] /= np.sqrt(1.04)
    with pytest.raises(ValueError, match="nonorthogonal nondegenerate"):
        _repair_exactly_degenerate_pairs(
            frame,
            np.asarray((1.0, 1.01, 2.0, 2.0)),
            weights,
        )


def test_gamma_canonicalization_is_pair_u2_invariant() -> None:
    canonical, weights = _canonical_frame()
    rotated = canonical.copy()
    rotated[:, :, :2] = np.einsum(
        "zma,ab->zmb", canonical[:, :, :2], _random_unitary(2), optimize=True
    )
    rotated[:, :, 2:] = np.einsum(
        "zma,ab->zmb", canonical[:, :, 2:], _random_unitary(3), optimize=True
    )
    energies = np.asarray((1.0, 1.0, 2.0, 2.0))

    fixed_reference, _, diagnostics = _canonicalize_gamma_e1h1_frame(
        canonical, energies, weights
    )
    fixed_rotated, _, _ = _canonicalize_gamma_e1h1_frame(
        rotated, energies, weights
    )

    assert np.max(np.abs(fixed_reference - fixed_rotated)) < 1e-12
    assert diagnostics["gamma_projected_jz_e1_plus"] == pytest.approx(0.5)
    assert diagnostics["gamma_projected_jz_e1_minus"] == pytest.approx(-0.5)
    assert diagnostics["gamma_projected_jz_h1_plus"] == pytest.approx(1.5)
    assert diagnostics["gamma_projected_jz_h1_minus"] == pytest.approx(-1.5)
    expected_tr = np.kron(np.eye(2), np.asarray(((0.0, -1.0), (1.0, 0.0))))
    active_tr = _active_time_reversal_unitary_from_frame(fixed_rotated, weights)
    assert np.max(np.abs(active_tr - expected_tr)) < 1e-12


def test_xml_split_zero_requires_explicit_record() -> None:
    split_zero = ET.fromstring(
        "<datafile><cmdargs><parsegroup>split 0</parsegroup></cmdargs></datafile>"
    )
    split_nonzero = ET.fromstring(
        "<datafile><cmdargs><parsegroup>split 0.1</parsegroup></cmdargs></datafile>"
    )
    contradictory = ET.fromstring(
        "<datafile><cmdargs><parsegroup>split 0</parsegroup>"
        "<argv>kdotpy split 0.1</argv></cmdargs></datafile>"
    )
    missing = ET.fromstring("<datafile><cmdargs /></datafile>")
    assert _xml_declares_split_zero(split_zero)
    assert not _xml_declares_split_zero(split_nonzero)
    assert not _xml_declares_split_zero(contradictory)
    assert not _xml_declares_split_zero(missing)


def test_wavefunction_csv_schema_units_and_complex_column_order_are_checked(
    tmp_path: Path,
) -> None:
    wrong_units = tmp_path / "wrong_units.csv"
    wrong_units.write_text(
        'z,"Γ6,+1/2","Γ6,+1/2"\n'
        'angstrom,Re ψ_i,Im ψ_i\n'
        '0,1,0\n1,0,0\n'
    )
    with pytest.raises(ValueError, match="declare z in nm"):
        read_kdotpy_wavefunction_csv(wrong_units)

    wrong_order = tmp_path / "wrong_order.csv"
    wrong_order.write_text(
        'z,"Γ6,+1/2","Γ6,+1/2"\n'
        'nm,Im ψ_i,Re ψ_i\n'
        '0,1,0\n1,0,0\n'
    )
    with pytest.raises(ValueError, match="declared Re then Im"):
        read_kdotpy_wavefunction_csv(wrong_order)


def test_strict_loader_rejects_nonfinite_policy_and_noncanonical_partition() -> None:
    with pytest.raises(ValueError, match="radial_tr_h0_tolerance_mev"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            "missing.xml",
            "missing",
            output_id="missing",
            potential_source="synthetic",
            radial_tr_h0_tolerance_mev=np.nan,
        )
    with pytest.raises(ValueError, match="canonical labels and E1"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            "missing.xml",
            "missing",
            output_id="missing",
            potential_source="synthetic",
            basis=E1H1BasisSpec(
                electron_indices=(0, 2),
                hole_indices=(1, 3),
            ),
        )


def _write_state_csv(path: Path, components: dict[str, complex]) -> None:
    labels = ["z"]
    units = ["nm"]
    first_row: list[float] = [0.0]
    second_row: list[float] = [1.0]
    for orbital, amplitude in components.items():
        labels.extend((orbital, orbital))
        units.extend(("Re ψ_i", "Im ψ_i"))
        first_row.extend((float(amplitude.real), float(amplitude.imag)))
        second_row.extend((0.0, 0.0))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(labels)
        writer.writerow(units)
        writer.writerow(first_row)
        writer.writerow(second_row)


def _write_synthetic_source(
    root: Path,
    *,
    break_finite_k_time_reversal: bool = False,
) -> tuple[Path, list[Path], str]:
    output_id = "synthetic_split0"
    orbitals = (
        CANONICAL_KANE_ORBITALS[0],
        CANONICAL_KANE_ORBITALS[1],
        CANONICAL_KANE_ORBITALS[2],
        CANONICAL_KANE_ORBITALS[5],
    )
    datafile = ET.Element("datafile")
    cmdargs = ET.SubElement(datafile, "cmdargs")
    ET.SubElement(cmdargs, "parsegroup").text = "split 0"
    dispersion = ET.SubElement(datafile, "dispersion")
    paths: list[Path] = []
    for k in (0.0, 0.1, 0.2, 0.3):
        point = ET.SubElement(dispersion, "momentum", k=str(k))
        ET.SubElement(point, "energies").text = (
            f"{1.0 + k} {1.0 + k} {2.0 - k} {2.0 - k}"
        )
        ET.SubElement(point, "bandindex").text = "13 14 15 16"
        components = [{orbital: 1.0 + 0.0j} for orbital in orbitals]
        if break_finite_k_time_reversal and k == 0.1:
            cosine, sine = 0.8, 0.6
            components[0] = {
                CANONICAL_KANE_ORBITALS[0]: cosine,
                CANONICAL_KANE_ORBITALS[2]: sine,
            }
            components[2] = {
                CANONICAL_KANE_ORBITALS[0]: -sine,
                CANONICAL_KANE_ORBITALS[2]: cosine,
            }
        for band, state_components in zip((13, 14, 15, 16), components):
            path = root / f"wfs{output_id}_{k:.3f}_0.{band}.csv"
            _write_state_csv(path, state_components)
            paths.append(path)
    xml_path = root / "source.xml"
    ET.ElementTree(datafile).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path, paths, output_id


def test_strict_loader_requires_source_hash_attestations(tmp_path: Path) -> None:
    xml_path, paths, output_id = _write_synthetic_source(tmp_path)
    with pytest.raises(ValueError, match="requires expected_xml_sha256"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            xml_path,
            tmp_path,
            output_id=output_id,
            potential_source="synthetic",
        )
    with pytest.raises(
        ValueError, match="requires expected_wavefunction_manifest_sha256"
    ):
        load_split_zero_kdotpy_radial_kane4_bundle(
            xml_path,
            tmp_path,
            output_id=output_id,
            potential_source="synthetic",
            expected_xml_sha256=_sha256(xml_path),
        )
    assert paths


def test_strict_loader_rejects_nonfinite_xml_momentum_and_energy(
    tmp_path: Path,
) -> None:
    momentum_root = tmp_path / "momentum"
    momentum_root.mkdir()
    xml_path, paths, output_id = _write_synthetic_source(momentum_root)
    tree = ET.parse(xml_path)
    tree.getroot().findall("./dispersion/momentum")[0].attrib["k"] = "nan"
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="non-finite momentum"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            xml_path,
            momentum_root,
            output_id=output_id,
            potential_source="synthetic",
            expected_xml_sha256=_sha256(xml_path),
            expected_wavefunction_manifest_sha256=_wavefunction_manifest_sha256(
                paths, momentum_root
            ),
        )

    energy_root = tmp_path / "energy"
    energy_root.mkdir()
    xml_path, paths, output_id = _write_synthetic_source(energy_root)
    tree = ET.parse(xml_path)
    tree.getroot().findall("./dispersion/momentum")[0].find("energies").text = (
        "nan 1 2 2"
    )
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    with pytest.raises(ValueError, match="non-finite energy"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            xml_path,
            energy_root,
            output_id=output_id,
            potential_source="synthetic",
            expected_xml_sha256=_sha256(xml_path),
            expected_wavefunction_manifest_sha256=_wavefunction_manifest_sha256(
                paths, energy_root
            ),
        )


def test_strict_split_zero_loader_excludes_gamma_and_replays_spectrum(tmp_path: Path) -> None:
    xml_path, paths, output_id = _write_synthetic_source(tmp_path)
    manifest = _wavefunction_manifest_sha256(paths, tmp_path)

    bundle, diagnostics = load_split_zero_kdotpy_radial_kane4_bundle(
        xml_path,
        tmp_path,
        output_id=output_id,
        potential_source="synthetic",
        expected_xml_sha256=_sha256(xml_path),
        expected_wavefunction_manifest_sha256=manifest,
    )

    assert bundle.nk == 3
    assert np.array_equal(bundle.k_cart_nm_inv[:, 0], np.asarray((0.1, 0.2, 0.3)))
    assert np.all(bundle.k_cart_nm_inv[:, 0] > 0.0)
    assert np.sum(bundle.weights_nm2) == pytest.approx(0.3**2 / (4.0 * np.pi))
    assert diagnostics["max_spectrum_replay_error_mev"] < 1e-12
    assert diagnostics["global_min_completeness"] > 1.0 - 1e-12
    assert diagnostics["frame_radial_time_reversal_error_nm_minus_half"] < 1e-12
    assert bundle.provenance["gamma_used_as_reference_only"] is True
    assert bundle.provenance["gamma_excluded_from_interaction_nodes"] is True
    assert bundle.provenance["finite_k_time_reversal_sewing_verified"] is True


def test_strict_split_zero_loader_rejects_finite_k_tr_breaking_source(
    tmp_path: Path,
) -> None:
    xml_path, paths, output_id = _write_synthetic_source(
        tmp_path,
        break_finite_k_time_reversal=True,
    )
    manifest = _wavefunction_manifest_sha256(paths, tmp_path)

    with pytest.raises(ValueError, match="fails axial time-reversal sewing"):
        load_split_zero_kdotpy_radial_kane4_bundle(
            xml_path,
            tmp_path,
            output_id=output_id,
            potential_source="synthetic",
            expected_xml_sha256=_sha256(xml_path),
            expected_wavefunction_manifest_sha256=manifest,
        )
