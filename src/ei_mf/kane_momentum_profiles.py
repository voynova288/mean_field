"""Fixed-reference E1/H1 radial profile builder for kdotpy Kane quartets."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from mean_field.systems.inas_gasb.kane4_bundle import lift_active_frame_to_reference

from .wavefunction_form_factor import (
    KaneMomentumLayerProfiles,
    _CANONICAL_KANE_ORBITALS,
    _integration_weights,
    _pair_projector_profiles_from_quartet,
    read_wavefunction_components,
)

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _weighted_pair_qr(block: ComplexArray, z_weights_nm: Array) -> ComplexArray:
    weighted = (np.sqrt(z_weights_nm)[:, None, None] * block).reshape(-1, 2)
    orthonormal, _r = np.linalg.qr(weighted)
    return orthonormal.reshape(block.shape) / np.sqrt(z_weights_nm)[:, None, None]


def build_fixed_reference_momentum_profiles(
    xml_path: str | Path,
    wavefunction_directory: str | Path,
    *,
    output_id: str,
    active_band_indices: tuple[int, int, int, int] = (13, 14, 15, 16),
    maximum_cross_pair_gram: float = 1e-7,
) -> KaneMomentumLayerProfiles:
    """Build Γ-anchored pair profiles by a rank-four polar/Löwdin lift.

    The source ordering is used only at Γ to define E1=(first pair) and
    H1=(second pair). At nonzero k, the complete active quartet is aligned to
    that reference by its polar isometry, so permutations and U(4) gauge choices
    inside the active source span do not set pair identity.
    """

    xml_source = Path(xml_path)
    wf_dir = Path(wavefunction_directory)
    points = ET.parse(xml_source).getroot().findall("./dispersion/momentum")
    if len(points) < 3 or len(active_band_indices) != 4:
        raise ValueError("fixed-reference source needs a radial rank-four quartet")
    orbital_index = {label: i for i, label in enumerate(_CANONICAL_KANE_ORBITALS)}
    raw_records: list[tuple[float, Array, ComplexArray, Array, float]] = []
    z_ref: Array | None = None
    weights_ref: Array | None = None

    for point in points:
        k = float(point.attrib["k"])
        indices = np.fromstring(
            point.findtext("bandindex", default=""), sep=" ", dtype=int
        )
        energies_all = np.fromstring(
            point.findtext("energies", default=""), sep=" ", dtype=np.float64
        )
        energies: list[float] = []
        states: list[ComplexArray] = []
        for band_index in active_band_indices:
            matches = np.flatnonzero(indices == band_index)
            if matches.size != 1:
                raise ValueError(f"band index {band_index} is not unique at k={k}")
            path = wf_dir / f"wfs{output_id}_{k:.3f}_0.{band_index}.csv"
            z, components = read_wavefunction_components(path)
            if z_ref is None:
                z_ref = z
                weights_ref = _integration_weights(z)
            elif not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
                raise ValueError("wavefunction files use inconsistent z grids")
            amplitudes = np.zeros(
                (z.size, len(_CANONICAL_KANE_ORBITALS)), dtype=np.complex128
            )
            for label, component in components.items():
                if label not in orbital_index:
                    raise ValueError(f"unknown Kane orbital {label!r}: {path}")
                amplitudes[:, orbital_index[label]] = component
            assert weights_ref is not None
            norm = float(
                np.einsum(
                    "z,zm,zm->",
                    weights_ref,
                    amplitudes.conj(),
                    amplitudes,
                    optimize=True,
                ).real
            )
            if not np.isfinite(norm) or norm <= 1e-14:
                raise ValueError(f"invalid wavefunction norm {norm}: {path}")
            states.append(amplitudes / np.sqrt(norm))
            energies.append(float(energies_all[matches[0]]))

        assert weights_ref is not None
        frame = np.stack(states, axis=-1)
        gram_raw = np.einsum(
            "zma,z,zmb->ab", frame.conj(), weights_ref, frame, optimize=True
        )
        raw_error = float(np.max(np.abs(gram_raw - np.eye(4))))
        raw_eigenvalues = np.linalg.eigvalsh(gram_raw).real
        cross_pair = float(np.max(np.abs(gram_raw[:2, 2:])))
        if cross_pair > maximum_cross_pair_gram:
            raise ValueError(
                f"raw E1/H1 pair spans overlap at k={k}: {cross_pair:.3e}"
            )
        energy_array = np.asarray(energies, dtype=np.float64)
        for start in (0, 2):
            block = frame[:, :, start : start + 2]
            block_gram = np.einsum(
                "zma,z,zmb->ab", block.conj(), weights_ref, block, optimize=True
            )
            error = float(np.max(np.abs(block_gram - np.eye(2))))
            if error > 1e-8:
                spread = float(np.ptp(energy_array[start : start + 2]))
                if spread > 1e-7:
                    raise ValueError(
                        "nonorthogonal nondegenerate source pair at "
                        f"k={k}: spread={spread:.3e} meV, error={error:.3e}"
                    )
                frame[:, :, start : start + 2] = _weighted_pair_qr(
                    block, weights_ref
                )
        gram = np.einsum(
            "zma,z,zmb->ab", frame.conj(), weights_ref, frame, optimize=True
        )
        if float(np.max(np.abs(gram - np.eye(4)))) > 2e-8:
            raise ValueError(f"pairwise-orthonormalized quartet failed at k={k}")
        raw_records.append((k, energy_array, frame, raw_eigenvalues, raw_error))

    raw_records.sort(key=lambda item: item[0])
    k_values = np.asarray([item[0] for item in raw_records], dtype=np.float64)
    if np.any(np.diff(k_values) <= 0.0) or abs(float(k_values[0])) > 1e-12:
        raise ValueError("fixed-reference radial grid must start at zero and increase")
    assert z_ref is not None and weights_ref is not None
    reference = raw_records[0][2]
    lifted_records: list[tuple[ComplexArray, ComplexArray, ComplexArray, Array, Array, Array, Array]] = []
    for _k, energies, frame, _gram_eigenvalues, _raw_error in raw_records:
        lifted = lift_active_frame_to_reference(reference, frame, energies, weights_ref)
        micro = lifted.micro_wavefunctions
        electron_pair = micro[:, :, :2]
        hole_pair = micro[:, :, 2:]
        rho_e = np.sum(np.abs(electron_pair) ** 2, axis=(1, 2))
        rho_h = np.sum(np.abs(hole_pair) ** 2, axis=(1, 2))
        rho_e /= float(np.dot(weights_ref, rho_e))
        rho_h /= float(np.dot(weights_ref, rho_h))
        character, _le, _lh, _lrho_e, _lrho_h = (
            _pair_projector_profiles_from_quartet(micro, weights_ref)
        )
        lifted_records.append(
            (
                micro,
                electron_pair,
                hole_pair,
                rho_e,
                rho_h,
                character,
                lifted.principal_cos2,
            )
        )

    def continuity(left: ComplexArray, right: ComplexArray) -> float:
        overlap = np.einsum(
            "zma,z,zmb->ab", left.conj(), weights_ref, right, optimize=True
        )
        return float(np.min(np.linalg.svd(overlap, compute_uv=False)))

    result = KaneMomentumLayerProfiles(
        k_nm_inv=k_values,
        z_nm=z_ref,
        electron_density_nm_inv=np.stack([item[3] for item in lifted_records]),
        hole_density_nm_inv=np.stack([item[4] for item in lifted_records]),
        gamma6_character_eigenvalues=np.stack([item[5] for item in lifted_records]),
        quartet_continuity_smin=np.asarray(
            [continuity(a[0], b[0]) for a, b in zip(lifted_records[:-1], lifted_records[1:])]
        ),
        electron_pair_continuity_smin=np.asarray(
            [continuity(a[1], b[1]) for a, b in zip(lifted_records[:-1], lifted_records[1:])]
        ),
        hole_pair_continuity_smin=np.asarray(
            [continuity(a[2], b[2]) for a, b in zip(lifted_records[:-1], lifted_records[1:])]
        ),
        source_xml=str(xml_source),
        wavefunction_directory=str(wf_dir),
        output_id=str(output_id),
        active_band_indices=tuple(int(x) for x in active_band_indices),
        raw_quartet_gram_eigenvalues=np.stack([item[3] for item in raw_records]),
        raw_quartet_orthonormality_error=np.asarray([item[4] for item in raw_records]),
        pair_construction="fixed_reference_polar_lowdin",
        fixed_reference_principal_cos2=np.stack([item[6] for item in lifted_records]),
    )
    result.validate()
    return result
