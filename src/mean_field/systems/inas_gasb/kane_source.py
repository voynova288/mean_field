"""File-backed kdotpy wavefunction adapter for Kane4 bundle construction.

This module has no runtime kdotpy dependency.  It reads kdotpy XML energies
and complex wavefunction CSVs, normalizes the full eight-orbital amplitudes,
and applies the fixed-reference polar lift. The maintained loader requires
source-attested ``split=0`` data and rejects historical nonzero-split inputs.
"""

from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .conventions import (
    E1H1BasisSpec,
    KANE8_JZ,
    kane8_time_reversal_unitary,
    spinful_time_reversal_errors,
)
from .kane4_bundle import (
    Kane4Bundle,
    lift_active_frame_to_reference,
)


CANONICAL_KANE_ORBITALS = (
    "Γ6,+1/2",
    "Γ6,-1/2",
    "Γ8,+3/2",
    "Γ8,+1/2",
    "Γ8,-1/2",
    "Γ8,-3/2",
    "Γ7,+1/2",
    "Γ7,-1/2",
)


_CANONICAL_ACTIVE_TR = np.kron(
    np.eye(2, dtype=np.complex128),
    np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.complex128),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wavefunction_manifest_sha256(paths: list[Path], root: Path) -> str:
    """Hash relative names and contents of one complete wavefunction source."""

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value.relative_to(root))):
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _xml_declares_split_zero(root: ET.Element) -> bool:
    """Require every recorded kdotpy ``split`` declaration to equal zero."""

    declarations: list[float] = []
    for group in root.findall("./cmdargs/parsegroup"):
        tokens = (group.text or "").split()
        if tokens and tokens[0] == "split":
            if len(tokens) != 2:
                return False
            try:
                declarations.append(float(tokens[1]))
            except ValueError:
                return False
    for source in (
        root.findtext("./cmdargs/argv", default=""),
        root.findtext("./info/cmdargs", default=""),
    ):
        tokens = source.split()
        for index, token in enumerate(tokens):
            if token != "split":
                continue
            if index + 1 >= len(tokens):
                return False
            try:
                declarations.append(float(tokens[index + 1]))
            except ValueError:
                return False
    return bool(declarations) and all(
        np.isfinite(value) and value == 0.0 for value in declarations
    )


def _require_finite_nonnegative(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _weighted_gram(frame: np.ndarray, z_weights_nm: np.ndarray) -> np.ndarray:
    states = np.asarray(frame, dtype=np.complex128)
    return np.einsum(
        "zma,z,zmb->ab",
        states.conj(),
        np.asarray(z_weights_nm, dtype=float),
        states,
        optimize=True,
    )


def _weighted_pair_qr(block: np.ndarray, z_weights_nm: np.ndarray) -> np.ndarray:
    """Orthonormalize one rank-two source span without mixing E1 and H1."""

    pair = np.asarray(block, dtype=np.complex128)
    weights = np.asarray(z_weights_nm, dtype=float)
    if pair.ndim != 3 or pair.shape[-1] != 2 or weights.shape != (pair.shape[0],):
        raise ValueError("pair block and z weights have incompatible shapes")
    weighted = (np.sqrt(weights)[:, None, None] * pair).reshape(-1, 2)
    orthonormal, triangular = np.linalg.qr(weighted)
    if float(np.min(np.abs(np.diag(triangular)))) <= 1e-12:
        raise ValueError("degenerate source pair is rank deficient")
    return orthonormal.reshape(pair.shape) / np.sqrt(weights)[:, None, None]


def _repair_exactly_degenerate_pairs(
    frame: np.ndarray,
    energies_mev: np.ndarray,
    z_weights_nm: np.ndarray,
    *,
    exact_degeneracy_tolerance_mev: float = 1e-7,
    orthonormality_tolerance: float = 1e-8,
    maximum_cross_pair_gram: float = 1e-7,
) -> tuple[np.ndarray, dict[str, float]]:
    """Repair only nonorthogonal, exactly degenerate E1/H1 source pairs."""

    exact_degeneracy_tolerance_mev = _require_finite_nonnegative(
        "exact_degeneracy_tolerance_mev", exact_degeneracy_tolerance_mev
    )
    orthonormality_tolerance = _require_finite_nonnegative(
        "orthonormality_tolerance", orthonormality_tolerance
    )
    maximum_cross_pair_gram = _require_finite_nonnegative(
        "maximum_cross_pair_gram", maximum_cross_pair_gram
    )
    repaired = np.asarray(frame, dtype=np.complex128).copy()
    energies = np.asarray(energies_mev, dtype=float)
    weights = np.asarray(z_weights_nm, dtype=float)
    if repaired.ndim != 3 or repaired.shape[-1] != 4 or energies.shape != (4,):
        raise ValueError("source quartet must have four states and four energies")
    raw_gram = _weighted_gram(repaired, weights)
    cross_pair = float(np.max(np.abs(raw_gram[:2, 2:])))
    if cross_pair > maximum_cross_pair_gram:
        raise ValueError(f"raw E1/H1 pair spans overlap: {cross_pair:.3e}")
    maximum_pair_error = 0.0
    maximum_repaired_pair_error = 0.0
    maximum_pair_spread = 0.0
    for start in (0, 2):
        block = repaired[:, :, start : start + 2]
        pair_error = float(np.max(np.abs(_weighted_gram(block, weights) - np.eye(2))))
        spread = float(np.ptp(energies[start : start + 2]))
        maximum_pair_error = max(maximum_pair_error, pair_error)
        maximum_pair_spread = max(maximum_pair_spread, spread)
        if pair_error > orthonormality_tolerance:
            if spread > exact_degeneracy_tolerance_mev:
                raise ValueError(
                    "nonorthogonal nondegenerate source pair: "
                    f"spread={spread:.3e} meV, error={pair_error:.3e}"
                )
            repaired[:, :, start : start + 2] = _weighted_pair_qr(block, weights)
        repaired_error = float(
            np.max(
                np.abs(
                    _weighted_gram(repaired[:, :, start : start + 2], weights)
                    - np.eye(2)
                )
            )
        )
        maximum_repaired_pair_error = max(maximum_repaired_pair_error, repaired_error)
    repaired_gram = _weighted_gram(repaired, weights)
    repaired_error = float(np.max(np.abs(repaired_gram - np.eye(4))))
    if repaired_error > 2.0 * orthonormality_tolerance:
        raise ValueError(f"pairwise-repaired quartet is not orthonormal: {repaired_error:.3e}")
    return repaired, {
        "raw_quartet_orthonormality_error": float(
            np.max(np.abs(raw_gram - np.eye(4)))
        ),
        "maximum_raw_pair_orthonormality_error": maximum_pair_error,
        "maximum_repaired_pair_orthonormality_error": maximum_repaired_pair_error,
        "maximum_cross_pair_gram": cross_pair,
        "maximum_pair_energy_spread_mev": maximum_pair_spread,
    }


def _apply_micro_time_reversal(state: np.ndarray) -> np.ndarray:
    return np.einsum(
        "mn,zn->zm",
        kane8_time_reversal_unitary(),
        np.asarray(state, dtype=np.complex128).conj(),
        optimize=True,
    )


def _active_time_reversal_unitary_from_frame(
    frame: np.ndarray,
    z_weights_nm: np.ndarray,
) -> np.ndarray:
    """Project the microscopic antiunitary ``Theta`` into one active frame."""

    states = np.asarray(frame, dtype=np.complex128)
    theta_states = np.einsum(
        "mn,zna->zma",
        kane8_time_reversal_unitary(),
        states.conj(),
        optimize=True,
    )
    return np.einsum(
        "zma,z,zmb->ab",
        states.conj(),
        np.asarray(z_weights_nm, dtype=float),
        theta_states,
        optimize=True,
    )


def _phase_fix_state(state: np.ndarray, z_weights_nm: np.ndarray) -> np.ndarray:
    values = np.asarray(state, dtype=np.complex128).copy()
    weighted_magnitude = np.sqrt(np.asarray(z_weights_nm, dtype=float))[:, None] * np.abs(values)
    index = np.unravel_index(int(np.argmax(weighted_magnitude)), values.shape)
    anchor = values[index]
    if abs(anchor) <= 1e-14:
        raise ValueError("cannot phase-fix a zero microscopic state")
    values *= np.exp(-1j * np.angle(anchor))
    if values[index].real < 0.0:
        values *= -1.0
    return values


def _canonicalize_gamma_e1h1_frame(
    frame: np.ndarray,
    energies_mev: np.ndarray,
    z_weights_nm: np.ndarray,
    *,
    exact_degeneracy_tolerance_mev: float = 1e-7,
    tr_span_tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Order Gamma pairs by projected Jz and fix canonical spinful-TR phases."""

    exact_degeneracy_tolerance_mev = _require_finite_nonnegative(
        "exact_degeneracy_tolerance_mev", exact_degeneracy_tolerance_mev
    )
    tr_span_tolerance = _require_finite_nonnegative(
        "tr_span_tolerance", tr_span_tolerance
    )
    states = np.asarray(frame, dtype=np.complex128)
    energies = np.asarray(energies_mev, dtype=float)
    weights = np.asarray(z_weights_nm, dtype=float)
    if states.ndim != 3 or states.shape[1:] != (8, 4) or energies.shape != (4,):
        raise ValueError("Gamma source frame must have shape (nz, 8, 4)")
    canonical_blocks: list[np.ndarray] = []
    canonical_energies: list[float] = []
    projected_jz_values: list[float] = []
    tr_span_residuals: list[float] = []
    for start in (0, 2):
        pair = states[:, :, start : start + 2]
        pair_energies = energies[start : start + 2]
        spread = float(np.ptp(pair_energies))
        if spread > exact_degeneracy_tolerance_mev:
            raise ValueError(
                f"Gamma E1/H1 source pair is not exactly degenerate: {spread:.3e} meV"
            )
        jz_projected = np.einsum(
            "zma,z,m,zmb->ab",
            pair.conj(),
            weights,
            KANE8_JZ,
            pair,
            optimize=True,
        )
        jz_projected = 0.5 * (jz_projected + jz_projected.conj().T)
        jz_values, jz_vectors = np.linalg.eigh(jz_projected)
        plus = np.einsum(
            "zma,a->zm",
            pair,
            jz_vectors[:, int(np.argmax(jz_values))],
            optimize=True,
        )
        plus = _phase_fix_state(plus, weights)
        minus_target = _apply_micro_time_reversal(plus)
        coefficients = np.einsum(
            "zma,z,zm->a",
            pair.conj(),
            weights,
            minus_target,
            optimize=True,
        )
        minus = np.einsum("zma,a->zm", pair, coefficients, optimize=True)
        residual = float(
            np.sqrt(
                np.einsum(
                    "z,zm,zm->",
                    weights,
                    (minus_target - minus).conj(),
                    minus_target - minus,
                    optimize=True,
                ).real
            )
        )
        if residual > tr_span_tolerance:
            raise ValueError(
                f"Gamma Kramers partner leaves its source pair span: {residual:.3e}"
            )
        norm = float(
            np.einsum("z,zm,zm->", weights, minus.conj(), minus, optimize=True).real
        )
        minus /= np.sqrt(norm)
        block = np.stack((plus, minus), axis=-1)
        block_error = float(np.max(np.abs(_weighted_gram(block, weights) - np.eye(2))))
        if block_error > 1e-8:
            raise ValueError(f"canonical Gamma Kramers pair failed: {block_error:.3e}")
        canonical_blocks.append(block)
        canonical_energies.extend((float(np.mean(pair_energies)),) * 2)
        projected_jz_values.extend((float(np.max(jz_values)), float(np.min(jz_values))))
        tr_span_residuals.append(residual)
    canonical = np.concatenate(canonical_blocks, axis=-1)
    active_tr = _active_time_reversal_unitary_from_frame(canonical, weights)
    canonical_tr_error = float(np.max(np.abs(active_tr - _CANONICAL_ACTIVE_TR)))
    tr_unitarity_error, tr_square_error = spinful_time_reversal_errors(active_tr)
    if max(canonical_tr_error, tr_unitarity_error, tr_square_error) > 1e-8:
        raise ValueError(
            "canonical Gamma frame has invalid active time reversal: "
            f"canonical={canonical_tr_error:.3e}, unitarity={tr_unitarity_error:.3e}, "
            f"square={tr_square_error:.3e}"
        )
    return canonical, np.asarray(canonical_energies, dtype=float), {
        "gamma_projected_jz_e1_plus": projected_jz_values[0],
        "gamma_projected_jz_e1_minus": projected_jz_values[1],
        "gamma_projected_jz_h1_plus": projected_jz_values[2],
        "gamma_projected_jz_h1_minus": projected_jz_values[3],
        "gamma_max_tr_span_residual": max(tr_span_residuals, default=0.0),
        "gamma_canonical_active_tr_error": canonical_tr_error,
        "gamma_active_tr_unitarity_error": tr_unitarity_error,
        "gamma_active_tr_square_error": tr_square_error,
    }


def _uniform_z_weights(z_nm: np.ndarray) -> np.ndarray:
    z = np.asarray(z_nm, dtype=float)
    if z.ndim != 1 or z.size < 2 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z grid must be strictly increasing")
    spacing = np.diff(z)
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-10):
        raise ValueError("kdotpy wavefunction CSV z grid must be uniform")
    return np.full(z.shape, float(spacing[0]), dtype=float)


def read_kdotpy_wavefunction_csv(
    path: str | Path,
    *,
    orbital_labels: tuple[str, ...] = CANONICAL_KANE_ORBITALS,
    minimum_present_orbitals: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one kdotpy state as normalized ``(nz, norb)`` amplitudes."""

    source = Path(path)
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        labels = next(reader)
        units = next(reader)
    data = np.loadtxt(source, delimiter=",", skiprows=2)
    if data.ndim != 2 or data.shape[1] != len(labels) or (len(labels) - 1) % 2:
        raise ValueError(f"unexpected kdotpy wavefunction CSV layout: {source}")
    if len(units) != len(labels) or labels[0].strip() != "z" or units[0].strip() != "nm":
        raise ValueError(f"wavefunction CSV must declare z in nm: {source}")
    for column in range(1, len(labels), 2):
        if labels[column] != labels[column + 1]:
            raise ValueError(f"wavefunction real/imaginary orbital labels do not match: {source}")
        if not units[column].strip().startswith("Re") or not units[column + 1].strip().startswith("Im"):
            raise ValueError(f"wavefunction columns must be declared Re then Im: {source}")
    z = np.asarray(data[:, 0], dtype=float)
    weights = _uniform_z_weights(z)
    amplitudes = np.zeros((z.size, len(orbital_labels)), dtype=np.complex128)
    index = {label: i for i, label in enumerate(orbital_labels)}
    found: set[str] = set()
    for column in range(1, len(labels), 2):
        label = labels[column]
        if label not in index:
            raise ValueError(f"unknown Kane orbital label {label!r}: {source}")
        if label in found:
            raise ValueError(f"duplicate Kane orbital label {label!r}: {source}")
        found.add(label)
        amplitudes[:, index[label]] = data[:, column] + 1j * data[:, column + 1]
    if len(found) < int(minimum_present_orbitals):
        raise ValueError(
            f"wavefunction CSV contains only {len(found)} orbital components; "
            f"minimum is {minimum_present_orbitals}: {source}"
        )
    norm = float(np.einsum("z,zm,zm->", weights, amplitudes.conj(), amplitudes, optimize=True).real)
    if not np.isfinite(norm) or norm <= 1e-14:
        raise ValueError(f"invalid kdotpy wavefunction norm {norm}: {source}")
    amplitudes /= np.sqrt(norm)
    return z, weights, amplitudes


def _radial_endpoint_weights(k_nm_inv: np.ndarray) -> np.ndarray:
    """Positive annular weights for a radial endpoint grid on ``[0,kmax]``."""

    k = np.asarray(k_nm_inv, dtype=float)
    if k.ndim != 1 or k.size < 3 or k[0] < 0.0 or np.any(np.diff(k) <= 0.0):
        raise ValueError("k must be a strictly increasing radial endpoint grid")
    edges = np.empty(k.size + 1, dtype=float)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (k[:-1] + k[1:])
    edges[-1] = k[-1]
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("radial endpoint grid does not define positive annular cells")
    return (edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * np.pi)




def load_split_zero_kdotpy_radial_kane4_bundle(
    xml_path: str | Path,
    wavefunction_directory: str | Path,
    *,
    output_id: str,
    active_band_indices: tuple[int, int, int, int] = (13, 14, 15, 16),
    potential_source: str,
    expected_xml_sha256: str | None = None,
    expected_wavefunction_manifest_sha256: str | None = None,
    source_validation_path: str | Path | None = None,
    expected_source_validation_sha256: str | None = None,
    exact_degeneracy_tolerance_mev: float = 1e-7,
    maximum_cross_pair_gram: float = 1e-7,
    radial_tr_h0_tolerance_mev: float = 1e-8,
    radial_tr_frame_tolerance_nm_minus_half: float = 1e-8,
    gamma_jz_tolerance: float = 1e-6,
    minimum_gamma6_character_gap: float = 1e-6,
    basis: E1H1BasisSpec | None = None,
) -> tuple[Kane4Bundle, dict[str, Any]]:
    """Build a canonical positive-radial Kane4 bundle from strict split-zero data.

    Gamma is used only to define ``(E1+,E1-,H1+,H1-)`` by projected Jz and
    canonical spinful time reversal.  It is then excluded from the interaction
    quadrature.  At every source momentum, nonorthogonal vectors may be repaired
    only inside an exactly degenerate rank-two E1 or H1 source pair.  The complete
    quartet is transported by the fixed-reference polar/Löwdin lift.
    """

    spec = E1H1BasisSpec() if basis is None else basis
    if (
        spec.labels != ("E1+", "E1-", "H1+", "H1-")
        or spec.electron_indices != (0, 1)
        or spec.hole_indices != (2, 3)
    ):
        raise ValueError(
            "strict split-zero loader requires canonical labels and E1[:2]/H1[2:] partition"
        )
    exact_degeneracy_tolerance_mev = _require_finite_nonnegative(
        "exact_degeneracy_tolerance_mev", exact_degeneracy_tolerance_mev
    )
    maximum_cross_pair_gram = _require_finite_nonnegative(
        "maximum_cross_pair_gram", maximum_cross_pair_gram
    )
    radial_tr_h0_tolerance_mev = _require_finite_nonnegative(
        "radial_tr_h0_tolerance_mev", radial_tr_h0_tolerance_mev
    )
    radial_tr_frame_tolerance_nm_minus_half = _require_finite_nonnegative(
        "radial_tr_frame_tolerance_nm_minus_half",
        radial_tr_frame_tolerance_nm_minus_half,
    )
    gamma_jz_tolerance = _require_finite_nonnegative(
        "gamma_jz_tolerance", gamma_jz_tolerance
    )
    minimum_gamma6_character_gap = _require_finite_nonnegative(
        "minimum_gamma6_character_gap", minimum_gamma6_character_gap
    )
    if len(active_band_indices) != spec.dimension:
        raise ValueError("active_band_indices must contain one E1/H1 quartet")
    xml_source = Path(xml_path)
    wf_dir = Path(wavefunction_directory)
    xml_hash = _sha256(xml_source)
    if expected_xml_sha256 is None:
        raise ValueError("strict source loading requires expected_xml_sha256")
    if expected_xml_sha256 is not None and xml_hash != expected_xml_sha256.lower():
        raise ValueError(
            "source XML hash mismatch: "
            f"expected={expected_xml_sha256.lower()}, actual={xml_hash}"
        )

    root = ET.parse(xml_source).getroot()
    if not _xml_declares_split_zero(root):
        raise ValueError("kdotpy XML does not explicitly attest split 0")
    points = root.findall("./dispersion/momentum")
    if len(points) < 4:
        raise ValueError("strict radial source requires Gamma and at least three positive momenta")

    expected_paths: list[Path] = []
    for point in points:
        k = float(point.attrib["k"])
        if not np.isfinite(k):
            raise ValueError("kdotpy XML contains a non-finite momentum")
        indices = np.fromstring(point.findtext("bandindex", default=""), sep=" ", dtype=int)
        energies_all = np.fromstring(
            point.findtext("energies", default=""), sep=" ", dtype=float
        )
        if indices.size == 0 or energies_all.shape != indices.shape:
            raise ValueError("kdotpy XML band indices and energies have incompatible shapes")
        if not np.all(np.isfinite(energies_all)):
            raise ValueError(f"kdotpy XML contains a non-finite energy at k={k}")
        for band_index in active_band_indices:
            if np.count_nonzero(indices == band_index) != 1:
                raise ValueError(f"band index {band_index} is not unique at k={k}")
            expected_paths.append(wf_dir / f"wfs{output_id}_{k:.3f}_0.{band_index}.csv")
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"strict wavefunction source is incomplete: {missing[0]}")
    manifest_hash = _wavefunction_manifest_sha256(expected_paths, wf_dir)
    if expected_wavefunction_manifest_sha256 is None:
        raise ValueError(
            "strict source loading requires expected_wavefunction_manifest_sha256"
        )
    if (
        expected_wavefunction_manifest_sha256 is not None
        and manifest_hash != expected_wavefunction_manifest_sha256.lower()
    ):
        raise ValueError(
            "wavefunction manifest hash mismatch: "
            f"expected={expected_wavefunction_manifest_sha256.lower()}, actual={manifest_hash}"
        )

    validation_hash: str | None = None
    validation_payload: dict[str, Any] | None = None
    if source_validation_path is not None:
        validation_source = Path(source_validation_path)
        validation_hash = _sha256(validation_source)
        if expected_source_validation_sha256 is None:
            raise ValueError(
                "attested source_validation_path requires expected_source_validation_sha256"
            )
        if (
            expected_source_validation_sha256 is not None
            and validation_hash != expected_source_validation_sha256.lower()
        ):
            raise ValueError(
                "source validation hash mismatch: "
                f"expected={expected_source_validation_sha256.lower()}, actual={validation_hash}"
            )
        validation_payload = json.loads(validation_source.read_text())
        if validation_payload.get("integrity", {}).get("split") != 0:
            raise ValueError("source validation receipt does not attest split zero")
        if tuple(validation_payload.get("band_indices", ())) != tuple(active_band_indices):
            raise ValueError("source validation receipt band indices do not match")
        if int(validation_payload.get("n_k", -1)) != len(points):
            raise ValueError("source validation receipt momentum count does not match XML")

    records: list[tuple[float, np.ndarray, np.ndarray, dict[str, float]]] = []
    z_ref: np.ndarray | None = None
    z_weights_ref: np.ndarray | None = None
    for point in points:
        k = float(point.attrib["k"])
        energies_all = np.fromstring(
            point.findtext("energies", default=""), sep=" ", dtype=float
        )
        indices = np.fromstring(
            point.findtext("bandindex", default=""), sep=" ", dtype=int
        )
        if not np.isfinite(k) or not np.all(np.isfinite(energies_all)):
            raise ValueError("strict source contains non-finite momentum or energy")
        energies: list[float] = []
        states: list[np.ndarray] = []
        for band_index in active_band_indices:
            source_index = int(np.flatnonzero(indices == band_index)[0])
            path = wf_dir / f"wfs{output_id}_{k:.3f}_0.{band_index}.csv"
            z, z_weights, amplitudes = read_kdotpy_wavefunction_csv(path)
            if z_ref is None:
                z_ref = z
                z_weights_ref = z_weights
            elif not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
                raise ValueError("wavefunction CSVs use inconsistent z grids")
            elif not np.allclose(z_weights, z_weights_ref, rtol=0.0, atol=1e-12):
                raise ValueError("wavefunction CSVs use inconsistent z weights")
            energies.append(float(energies_all[source_index]))
            states.append(amplitudes)
        assert z_weights_ref is not None
        frame, repair = _repair_exactly_degenerate_pairs(
            np.stack(states, axis=-1),
            np.asarray(energies, dtype=float),
            z_weights_ref,
            exact_degeneracy_tolerance_mev=exact_degeneracy_tolerance_mev,
            maximum_cross_pair_gram=maximum_cross_pair_gram,
        )
        records.append((k, np.asarray(energies, dtype=float), frame, repair))

    records.sort(key=lambda item: item[0])
    k_all = np.asarray([item[0] for item in records], dtype=float)
    if abs(float(k_all[0])) > 1e-12 or np.any(np.diff(k_all) <= 0.0):
        raise ValueError("strict radial source must start at Gamma and increase uniquely")
    assert z_ref is not None and z_weights_ref is not None
    gamma_reference, gamma_energies, gamma_diagnostics = _canonicalize_gamma_e1h1_frame(
        records[0][2],
        records[0][1],
        z_weights_ref,
        exact_degeneracy_tolerance_mev=exact_degeneracy_tolerance_mev,
    )
    observed_gamma_jz = np.asarray(
        (
            gamma_diagnostics["gamma_projected_jz_e1_plus"],
            gamma_diagnostics["gamma_projected_jz_e1_minus"],
            gamma_diagnostics["gamma_projected_jz_h1_plus"],
            gamma_diagnostics["gamma_projected_jz_h1_minus"],
        ),
        dtype=float,
    )
    expected_gamma_jz = np.asarray((0.5, -0.5, 1.5, -1.5), dtype=float)
    gamma_jz_error = float(np.max(np.abs(observed_gamma_jz - expected_gamma_jz)))
    if gamma_jz_error > gamma_jz_tolerance:
        raise ValueError(
            "Gamma source pairs do not establish canonical E1/H1 Jz identity: "
            f"error={gamma_jz_error:.3e}, tolerance={gamma_jz_tolerance:.3e}"
        )
    gamma6_projector = np.zeros(8, dtype=float)
    gamma6_projector[:2] = 1.0
    gamma6_weights = np.einsum(
        "zma,z,m,zma->a",
        gamma_reference.conj(),
        z_weights_ref,
        gamma6_projector,
        gamma_reference,
        optimize=True,
    ).real
    gamma6_character_gap = float(
        np.min(gamma6_weights[:2]) - np.max(gamma6_weights[2:])
    )
    if gamma6_character_gap < minimum_gamma6_character_gap:
        raise ValueError(
            "Gamma source pairs do not establish E1 Gamma6 character over H1: "
            f"gap={gamma6_character_gap:.3e}, "
            f"minimum={minimum_gamma6_character_gap:.3e}"
        )
    active_tr = _active_time_reversal_unitary_from_frame(
        gamma_reference, z_weights_ref
    )
    gamma_lift = lift_active_frame_to_reference(
        gamma_reference,
        gamma_reference,
        gamma_energies,
        z_weights_ref,
    )
    gamma_spectrum_error = float(
        np.max(
            np.abs(
                np.linalg.eigvalsh(gamma_lift.h0_mev)
                - np.sort(records[0][1])
            )
        )
    )

    lifted_frames: list[np.ndarray] = []
    h0_values: list[np.ndarray] = []
    completeness: list[float] = []
    spectrum_errors: list[float] = []
    source_energies: list[np.ndarray] = []
    for _k, energies, frame, _repair in records[1:]:
        lifted = lift_active_frame_to_reference(
            gamma_reference,
            frame,
            energies,
            z_weights_ref,
        )
        lifted_frames.append(lifted.micro_wavefunctions)
        h0_values.append(lifted.h0_mev)
        completeness.append(lifted.min_completeness)
        spectrum_errors.append(lifted.spectrum_error_mev)
        source_energies.append(energies)

    k_positive = k_all[1:]
    bundle = Kane4Bundle(
        k_cart_nm_inv=np.column_stack((k_positive, np.zeros_like(k_positive))),
        weights_nm2=_radial_endpoint_weights(k_positive),
        z_nm=z_ref,
        z_weights_nm=z_weights_ref,
        h0_mev=np.stack(h0_values, axis=-1),
        micro_wavefunctions=np.stack(lifted_frames, axis=0),
        time_reversal_unitary=active_tr,
        basis=spec,
        provenance={
            "classification": "strict_split_zero_canonical_radial_kane4",
            "source_xml": str(xml_source),
            "source_xml_sha256": xml_hash,
            "wavefunction_directory": str(wf_dir),
            "wavefunction_manifest_sha256": manifest_hash,
            "wavefunction_csv_count": len(expected_paths),
            "source_validation": (
                None if source_validation_path is None else str(source_validation_path)
            ),
            "source_validation_sha256": validation_hash,
            "output_id": output_id,
            "active_band_indices": list(active_band_indices),
            "calculation_split_mev": 0.0,
            "potential_source": str(potential_source),
            "basis_construction": (
                "Gamma projected-Jz ordering + canonical spinful TR; "
                "exact-degenerate pair repair; full-quartet fixed-reference polar/Lowdin"
            ),
            "gamma_used_as_reference_only": True,
            "gamma_excluded_from_interaction_nodes": True,
            "gamma_identity_verification": {
                "jz_error": gamma_jz_error,
                "jz_tolerance": float(gamma_jz_tolerance),
                "gamma6_character_gap": gamma6_character_gap,
                "minimum_gamma6_character_gap": float(
                    minimum_gamma6_character_gap
                ),
            },
            "radial_only": True,
            "finite_k_time_reversal_sewing_verified": False,
            "source_attestation_required": True,
        },
    )
    bundle.validate()

    adjacent_continuity = []
    for left, right in zip(lifted_frames[:-1], lifted_frames[1:]):
        overlap = np.einsum(
            "zma,z,zmb->ab",
            left.conj(),
            z_weights_ref,
            right,
            optimize=True,
        )
        adjacent_continuity.append(float(np.min(np.linalg.svd(overlap, compute_uv=False))))
    source_energy_array = np.stack(source_energies, axis=0)
    replayed_eigenvalues = np.stack(
        [np.linalg.eigvalsh(matrix) for matrix in h0_values], axis=0
    )
    replay_error = float(
        np.max(
            np.abs(
                replayed_eigenvalues
                - np.sort(source_energy_array, axis=1)
            )
        )
    )
    repair_diagnostics = [item[3] for item in records]
    from .axial import axial_radial_time_reversal_residuals

    radial_tr = axial_radial_time_reversal_residuals(bundle)
    h0_tr_error = max(
        radial_tr["h0_radial_time_reversal_error_mev"],
        radial_tr["h0_radial_projector_error_mev"],
    )
    frame_tr_error = radial_tr["frame_radial_time_reversal_error_nm_minus_half"]
    if h0_tr_error > radial_tr_h0_tolerance_mev:
        raise ValueError(
            "strict radial Hamiltonian fails axial time-reversal sewing: "
            f"error={h0_tr_error:.3e} meV, tolerance={radial_tr_h0_tolerance_mev:.3e} meV"
        )
    if frame_tr_error > radial_tr_frame_tolerance_nm_minus_half:
        raise ValueError(
            "strict radial frame fails axial time-reversal sewing: "
            f"error={frame_tr_error:.3e} nm^-1/2, "
            f"tolerance={radial_tr_frame_tolerance_nm_minus_half:.3e} nm^-1/2"
        )
    bundle = replace(
        bundle,
        provenance={
            **bundle.provenance,
            "finite_k_time_reversal_sewing_verified": True,
            "finite_k_time_reversal_verification": {
                "h0_tolerance_mev": float(radial_tr_h0_tolerance_mev),
                "frame_tolerance_nm_minus_half": float(
                    radial_tr_frame_tolerance_nm_minus_half
                ),
                **radial_tr,
            },
        },
    )
    bundle_diagnostics = bundle.validate()
    diagnostics: dict[str, Any] = {
        **bundle_diagnostics,
        **gamma_diagnostics,
        **radial_tr,
        "source_xml_sha256": xml_hash,
        "wavefunction_manifest_sha256": manifest_hash,
        "source_validation_sha256": validation_hash,
        "source_point_count_including_gamma": len(records),
        "positive_radial_node_count": bundle.nk,
        "gamma_spectrum_replay_error_mev": gamma_spectrum_error,
        "max_spectrum_replay_error_mev": replay_error,
        "global_min_completeness": float(np.min(completeness)),
        "minimum_adjacent_quartet_continuity_smin": float(
            np.min(adjacent_continuity)
        ),
        "maximum_raw_quartet_orthonormality_error": max(
            item["raw_quartet_orthonormality_error"] for item in repair_diagnostics
        ),
        "maximum_repaired_pair_orthonormality_error": max(
            item["maximum_repaired_pair_orthonormality_error"]
            for item in repair_diagnostics
        ),
        "maximum_cross_pair_gram": max(
            item["maximum_cross_pair_gram"] for item in repair_diagnostics
        ),
        "gamma6_weight_e1_state_min": float(np.min(gamma6_weights[:2])),
        "gamma6_weight_e1_pair_total": float(np.sum(gamma6_weights[:2])),
        "gamma6_weight_h1_state_max": float(np.max(gamma6_weights[2:])),
        "gamma6_weight_h1_pair_total": float(np.sum(gamma6_weights[2:])),
        "gamma6_character_gap": gamma6_character_gap,
        "gamma_jz_error": gamma_jz_error,
        "radial_weight_sum_nm2": float(np.sum(bundle.weights_nm2)),
        "expected_radial_disk_weight_nm2": float(k_positive[-1] ** 2 / (4.0 * np.pi)),
    }
    return bundle, diagnostics
