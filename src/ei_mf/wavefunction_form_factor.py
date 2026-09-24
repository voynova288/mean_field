"""Finite-width Coulomb form factors from kdotpy Kane wavefunctions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def _integration_weights(z_nm: Array) -> Array:
    z = np.asarray(z_nm, dtype=np.float64)
    if z.ndim != 1 or z.size < 3 or np.any(np.diff(z) <= 0.0):
        raise ValueError("z grid must be one-dimensional and strictly increasing")
    edges = np.empty(z.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    return np.diff(edges)


def read_wavefunction_components(path: str | Path) -> tuple[Array, dict[str, NDArray[np.complex128]]]:
    """Read complex orbital components from a kdotpy wavefunction CSV."""

    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        labels = next(reader)
        next(reader)  # units / Re-Im labels
    data = np.loadtxt(path, delimiter=",", skiprows=2)
    z = np.asarray(data[:, 0], dtype=np.float64)
    orbital: dict[str, NDArray[np.complex128]] = {}
    if (len(labels) - 1) % 2 != 0 or data.shape[1] != len(labels):
        raise ValueError(f"unexpected kdotpy wavefunction CSV layout: {path}")
    for col in range(1, len(labels), 2):
        label = labels[col]
        component = np.asarray(data[:, col] + 1j * data[:, col + 1], dtype=np.complex128)
        if label in orbital:
            raise ValueError(f"duplicate orbital component {label!r}: {path}")
        orbital[label] = component
    return z, orbital


def read_orbital_densities(path: str | Path) -> tuple[Array, dict[str, Array]]:
    z, components = read_wavefunction_components(path)
    return z, {label: np.abs(component) ** 2 for label, component in components.items()}


@dataclass(frozen=True)
class KaneLayerProfiles:
    z_nm: Array
    electron_density_nm_inv: Array
    hole_density_nm_inv: Array
    source_files: tuple[str, ...] = ()

    @classmethod
    def from_active_quartet_csv(cls, paths: list[str | Path]) -> "KaneLayerProfiles":
        """Build Gamma6-electron and Gamma8-heavy-hole projector densities."""

        if len(paths) != 4:
            raise ValueError("exactly four active E1/H1 wavefunction files are required")
        z_ref: Array | None = None
        rho_e: Array | None = None
        rho_h: Array | None = None
        for path in paths:
            z, orbitals = read_orbital_densities(path)
            if z_ref is None:
                z_ref = z
                rho_e = np.zeros_like(z)
                rho_h = np.zeros_like(z)
            elif not np.allclose(z, z_ref, rtol=0.0, atol=1e-10):
                raise ValueError("wavefunction files use inconsistent z grids")
            assert rho_e is not None and rho_h is not None
            for label, density in orbitals.items():
                if label.startswith("Γ6,"):
                    rho_e += density
                elif label in {"Γ8,+3/2", "Γ8,-3/2"}:
                    rho_h += density
        assert z_ref is not None and rho_e is not None and rho_h is not None
        weights = _integration_weights(z_ref)
        norm_e = float(np.sum(weights * rho_e))
        norm_h = float(np.sum(weights * rho_h))
        if norm_e <= 1e-12 or norm_h <= 1e-12:
            raise ValueError("active quartet has insufficient Gamma6 or heavy-hole density")
        return cls(
            z_ref,
            rho_e / norm_e,
            rho_h / norm_h,
            tuple(str(Path(path)) for path in paths),
        )

    def validate(self) -> None:
        w = _integration_weights(self.z_nm)
        for name, density in (
            ("electron", self.electron_density_nm_inv),
            ("hole", self.hole_density_nm_inv),
        ):
            if density.shape != self.z_nm.shape or np.any(density < 0.0):
                raise ValueError(f"invalid {name} density")
            if not np.isclose(np.sum(w * density), 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(f"{name} density is not normalized")

    def moments(self) -> dict[str, float]:
        self.validate()
        w = _integration_weights(self.z_nm)
        ze = float(np.sum(w * self.electron_density_nm_inv * self.z_nm))
        zh = float(np.sum(w * self.hole_density_nm_inv * self.z_nm))
        sigma_e = float(np.sqrt(np.sum(w * self.electron_density_nm_inv * (self.z_nm - ze) ** 2)))
        sigma_h = float(np.sqrt(np.sum(w * self.hole_density_nm_inv * (self.z_nm - zh) ** 2)))
        return {
            "electron_mean_z_nm": ze,
            "hole_mean_z_nm": zh,
            "centroid_separation_nm": abs(zh - ze),
            "electron_rms_width_nm": sigma_e,
            "hole_rms_width_nm": sigma_h,
        }

    def form_factors(self, q_nm_inv: Array) -> tuple[Array, Array, Array]:
        """Return ``F_aa``, ``F_bb``, and ``F_ab`` for nonnegative q."""

        self.validate()
        q = np.asarray(q_nm_inv, dtype=np.float64)
        if q.ndim != 1 or np.any(q < 0.0):
            raise ValueError("q must be a one-dimensional nonnegative array")
        w = _integration_weights(self.z_nm)
        pe = w * self.electron_density_nm_inv
        ph = w * self.hole_density_nm_inv
        distance = np.abs(self.z_nm[:, None] - self.z_nm[None, :])
        Faa = np.empty_like(q)
        Fbb = np.empty_like(q)
        Fab = np.empty_like(q)
        for iq, qval in enumerate(q):
            kernel = np.exp(-qval * distance)
            Faa[iq] = pe @ kernel @ pe
            Fbb[iq] = ph @ kernel @ ph
            Fab[iq] = pe @ kernel @ ph
        return Faa, Fbb, Fab

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            z_nm=self.z_nm,
            electron_density_nm_inv=self.electron_density_nm_inv,
            hole_density_nm_inv=self.hole_density_nm_inv,
            source_files=np.asarray(self.source_files),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "KaneLayerProfiles":
        with np.load(path) as data:
            result = cls(
                np.asarray(data["z_nm"], dtype=np.float64),
                np.asarray(data["electron_density_nm_inv"], dtype=np.float64),
                np.asarray(data["hole_density_nm_inv"], dtype=np.float64),
                tuple(str(x) for x in data.get("source_files", np.array([], dtype=str))),
            )
        result.validate()
        return result


_CANONICAL_KANE_ORBITALS = (
    "Γ6,+1/2",
    "Γ6,-1/2",
    "Γ8,+3/2",
    "Γ8,+1/2",
    "Γ8,-1/2",
    "Γ8,-3/2",
    "Γ7,+1/2",
    "Γ7,-1/2",
)


def _pair_projector_profiles_from_quartet(
    frame: NDArray[np.complex128], z_weights_nm: Array
) -> tuple[Array, NDArray[np.complex128], NDArray[np.complex128], Array, Array]:
    """Split one orthonormal quartet by Γ6 character and return pair traces."""

    gamma6_character = np.einsum(
        "zma,z,zmb->ab",
        frame[:, :2, :].conj(),
        z_weights_nm,
        frame[:, :2, :],
        optimize=True,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(gamma6_character)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order].real, dtype=np.float64)
    rotated = np.einsum(
        "zma,ab->zmb", frame, eigenvectors[:, order], optimize=True
    )
    electron_pair = rotated[:, :, :2]
    hole_pair = rotated[:, :, 2:]
    rho_e = np.sum(np.abs(electron_pair) ** 2, axis=(1, 2))
    rho_h = np.sum(np.abs(hole_pair) ** 2, axis=(1, 2))
    rho_e /= float(np.dot(z_weights_nm, rho_e))
    rho_h /= float(np.dot(z_weights_nm, rho_h))
    return eigenvalues, electron_pair, hole_pair, rho_e, rho_h


@dataclass(frozen=True)
class KaneMomentumLayerProfiles:
    """Gauge-invariant E1/H1 pair densities resolved on a radial k grid.

    The returned local densities are traces over rank-two E1/H1 projectors,
    so they are invariant under U(2) rotations inside either selected pair.
    The preferred source path uses a fixed Γ-point E1/H1 reference and a
    polar/Löwdin lift of the complete active quartet; local Γ6-character
    splitting remains available only as a diagnostic construction.
    """

    k_nm_inv: Array
    z_nm: Array
    electron_density_nm_inv: Array  # (nk, nz), normalized independently at each k
    hole_density_nm_inv: Array  # (nk, nz), normalized independently at each k
    gamma6_character_eigenvalues: Array  # (nk, 4), descending
    quartet_continuity_smin: Array  # (nk-1,)
    electron_pair_continuity_smin: Array  # (nk-1,)
    hole_pair_continuity_smin: Array  # (nk-1,)
    source_xml: str = ""
    wavefunction_directory: str = ""
    output_id: str = ""
    active_band_indices: tuple[int, ...] = ()
    raw_quartet_gram_eigenvalues: Array | None = None
    raw_quartet_orthonormality_error: Array | None = None
    pair_construction: str = "gamma6_character_local"
    fixed_reference_principal_cos2: Array | None = None

    @classmethod
    def from_kdotpy_radial_export(
        cls,
        xml_path: str | Path,
        wavefunction_directory: str | Path,
        *,
        output_id: str,
        active_band_indices: tuple[int, int, int, int] = (-2, -1, 1, 2),
        minimum_gamma6_character_gap: float = 1e-3,
    ) -> "KaneMomentumLayerProfiles":
        """Build pair-projector profiles from complex kdotpy quartet CSVs."""

        xml_source = Path(xml_path)
        wf_dir = Path(wavefunction_directory)
        points = ET.parse(xml_source).getroot().findall("./dispersion/momentum")
        if len(points) < 3:
            raise ValueError("kdotpy XML needs at least three radial momentum points")
        if len(active_band_indices) != 4:
            raise ValueError("active_band_indices must contain one rank-four quartet")

        orbital_index = {label: i for i, label in enumerate(_CANONICAL_KANE_ORBITALS)}
        records: list[tuple[float, Array, Array, Array, float, Array, Array, Array, Array]] = []
        z_ref: Array | None = None
        weights_ref: Array | None = None
        for point in points:
            k = float(point.attrib["k"])
            indices = np.fromstring(
                point.findtext("bandindex", default=""), sep=" ", dtype=int
            )
            states: list[NDArray[np.complex128]] = []
            for band_index in active_band_indices:
                if np.count_nonzero(indices == band_index) != 1:
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

            assert weights_ref is not None
            frame = np.stack(states, axis=-1)
            gram = np.einsum(
                "zma,z,zmb->ab", frame.conj(), weights_ref, frame, optimize=True
            )
            raw_orth_error = float(np.max(np.abs(gram - np.eye(4))))
            gram_eigenvalues, gram_eigenvectors = np.linalg.eigh(gram)
            if float(np.min(gram_eigenvalues)) <= 1e-8:
                raise ValueError(
                    f"rank-deficient exported quartet at k={k}: "
                    f"min Gram eigenvalue={float(np.min(gram_eigenvalues)):.3e}"
                )
            inverse_sqrt = (
                gram_eigenvectors
                @ np.diag(1.0 / np.sqrt(gram_eigenvalues))
                @ gram_eigenvectors.conj().T
            )
            frame = np.einsum("zma,ab->zmb", frame, inverse_sqrt, optimize=True)
            orthonormal_gram = np.einsum(
                "zma,z,zmb->ab", frame.conj(), weights_ref, frame, optimize=True
            )
            orth_error = float(np.max(np.abs(orthonormal_gram - np.eye(4))))
            if orth_error > 2e-10:
                raise ValueError(
                    f"Löwdin-orthonormalized quartet failed at k={k}: error={orth_error:.3e}"
                )
            eigenvalues, electron_pair, hole_pair, rho_e, rho_h = (
                _pair_projector_profiles_from_quartet(frame, weights_ref)
            )
            records.append(
                (
                    k,
                    frame,
                    electron_pair,
                    hole_pair,
                    raw_orth_error,
                    eigenvalues,
                    rho_e,
                    rho_h,
                    np.asarray(gram_eigenvalues, dtype=np.float64),
                )
            )

        records.sort(key=lambda item: item[0])
        k_values = np.asarray([item[0] for item in records], dtype=np.float64)
        if np.any(np.diff(k_values) <= 0.0):
            raise ValueError("radial k values must be strictly increasing")
        assert z_ref is not None and weights_ref is not None

        quartet_smin: list[float] = []
        electron_smin: list[float] = []
        hole_smin: list[float] = []
        for left, right in zip(records[:-1], records[1:]):
            for target, left_frame, right_frame in (
                (quartet_smin, left[1], right[1]),
                (electron_smin, left[2], right[2]),
                (hole_smin, left[3], right[3]),
            ):
                overlap = np.einsum(
                    "zma,z,zmb->ab",
                    left_frame.conj(),
                    weights_ref,
                    right_frame,
                    optimize=True,
                )
                target.append(float(np.min(np.linalg.svd(overlap, compute_uv=False))))

        character = np.stack([item[5] for item in records])
        character_gap = character[:, 1] - character[:, 2]
        if float(np.min(character_gap)) < float(minimum_gamma6_character_gap):
            raise ValueError(
                "E1/H1 Γ6-character gap closes: "
                f"min={float(np.min(character_gap)):.6g}"
            )
        result = cls(
            k_values,
            z_ref,
            np.stack([item[6] for item in records]),
            np.stack([item[7] for item in records]),
            character,
            np.asarray(quartet_smin),
            np.asarray(electron_smin),
            np.asarray(hole_smin),
            str(xml_source),
            str(wf_dir),
            str(output_id),
            tuple(int(x) for x in active_band_indices),
            np.stack([item[8] for item in records]),
            np.asarray([item[4] for item in records], dtype=np.float64),
        )
        result.validate()
        return result

    def validate(self) -> None:
        k = np.asarray(self.k_nm_inv, dtype=np.float64)
        z = np.asarray(self.z_nm, dtype=np.float64)
        nk, nz = k.size, z.size
        if nk < 3 or np.any(np.diff(k) <= 0.0):
            raise ValueError("k grid must be strictly increasing")
        if nz < 3 or np.any(np.diff(z) <= 0.0):
            raise ValueError("z grid must be strictly increasing")
        if self.electron_density_nm_inv.shape != (nk, nz) or self.hole_density_nm_inv.shape != (nk, nz):
            raise ValueError("momentum-profile density shape mismatch")
        if self.gamma6_character_eigenvalues.shape != (nk, 4):
            raise ValueError("Γ6-character eigenvalue shape mismatch")
        for values in (
            self.quartet_continuity_smin,
            self.electron_pair_continuity_smin,
            self.hole_pair_continuity_smin,
        ):
            if values.shape != (nk - 1,) or np.any(values < -1e-12) or np.any(values > 1.0 + 1e-8):
                raise ValueError("invalid adjacent-k continuity singular values")
        weights = _integration_weights(z)
        for density in (self.electron_density_nm_inv, self.hole_density_nm_inv):
            if np.any(density < 0.0) or not np.all(np.isfinite(density)):
                raise ValueError("invalid momentum-resolved density")
            norms = density @ weights
            if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-8):
                raise ValueError("momentum-resolved profiles are not normalized")
        if np.any(np.diff(self.gamma6_character_eigenvalues, axis=1) > 1e-10):
            raise ValueError("Γ6-character eigenvalues are not descending")
        if self.raw_quartet_gram_eigenvalues is not None:
            if self.raw_quartet_gram_eigenvalues.shape != (nk, 4) or np.any(
                self.raw_quartet_gram_eigenvalues <= 0.0
            ):
                raise ValueError("invalid raw-quartet Gram eigenvalues")
        if self.raw_quartet_orthonormality_error is not None:
            if self.raw_quartet_orthonormality_error.shape != (nk,) or np.any(
                self.raw_quartet_orthonormality_error < 0.0
            ):
                raise ValueError("invalid raw-quartet orthonormality errors")
        if not self.pair_construction:
            raise ValueError("pair construction must be identified")
        if self.fixed_reference_principal_cos2 is not None:
            if self.fixed_reference_principal_cos2.shape != (nk, 4) or np.any(
                self.fixed_reference_principal_cos2 <= 0.0
            ) or np.any(self.fixed_reference_principal_cos2 > 1.0 + 1e-8):
                raise ValueError("invalid fixed-reference principal cosines")

    @property
    def minimum_gamma6_character_gap(self) -> float:
        return float(np.min(self.gamma6_character_eigenvalues[:, 1] - self.gamma6_character_eigenvalues[:, 2]))

    def densities_at(self, k_nm_inv: Array) -> tuple[Array, Array]:
        """Linearly interpolate positive pair-projector densities and renormalize."""

        target = np.asarray(k_nm_inv, dtype=np.float64)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("target k grid must be a nonempty vector")
        if target.min() < self.k_nm_inv[0] or target.max() > self.k_nm_inv[-1]:
            raise ValueError("target k grid extends outside profile source range")
        electron = np.column_stack(
            [np.interp(target, self.k_nm_inv, self.electron_density_nm_inv[:, iz]) for iz in range(self.z_nm.size)]
        )
        hole = np.column_stack(
            [np.interp(target, self.k_nm_inv, self.hole_density_nm_inv[:, iz]) for iz in range(self.z_nm.size)]
        )
        weights = _integration_weights(self.z_nm)
        electron /= electron @ weights[:, None]
        hole /= hole @ weights[:, None]
        return electron, hole

    def frozen_profile(self, k0_nm_inv: float) -> KaneLayerProfiles:
        electron, hole = self.densities_at(np.asarray([float(k0_nm_inv)]))
        return KaneLayerProfiles(
            self.z_nm,
            electron[0],
            hole[0],
            (self.source_xml, f"pair-projector profile frozen at k={float(k0_nm_inv):.12g} nm^-1"),
        )

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            k_nm_inv=self.k_nm_inv,
            z_nm=self.z_nm,
            electron_density_nm_inv=self.electron_density_nm_inv,
            hole_density_nm_inv=self.hole_density_nm_inv,
            gamma6_character_eigenvalues=self.gamma6_character_eigenvalues,
            quartet_continuity_smin=self.quartet_continuity_smin,
            electron_pair_continuity_smin=self.electron_pair_continuity_smin,
            hole_pair_continuity_smin=self.hole_pair_continuity_smin,
            source_xml=np.asarray(self.source_xml),
            wavefunction_directory=np.asarray(self.wavefunction_directory),
            output_id=np.asarray(self.output_id),
            active_band_indices=np.asarray(self.active_band_indices, dtype=np.int64),
            raw_quartet_gram_eigenvalues=(
                np.empty((0, 4), dtype=np.float64)
                if self.raw_quartet_gram_eigenvalues is None
                else self.raw_quartet_gram_eigenvalues
            ),
            raw_quartet_orthonormality_error=(
                np.empty((0,), dtype=np.float64)
                if self.raw_quartet_orthonormality_error is None
                else self.raw_quartet_orthonormality_error
            ),
            pair_construction=np.asarray(self.pair_construction),
            fixed_reference_principal_cos2=(
                np.empty((0, 4), dtype=np.float64)
                if self.fixed_reference_principal_cos2 is None
                else self.fixed_reference_principal_cos2
            ),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "KaneMomentumLayerProfiles":
        with np.load(path, allow_pickle=False) as data:
            result = cls(
                np.asarray(data["k_nm_inv"], dtype=np.float64),
                np.asarray(data["z_nm"], dtype=np.float64),
                np.asarray(data["electron_density_nm_inv"], dtype=np.float64),
                np.asarray(data["hole_density_nm_inv"], dtype=np.float64),
                np.asarray(data["gamma6_character_eigenvalues"], dtype=np.float64),
                np.asarray(data["quartet_continuity_smin"], dtype=np.float64),
                np.asarray(data["electron_pair_continuity_smin"], dtype=np.float64),
                np.asarray(data["hole_pair_continuity_smin"], dtype=np.float64),
                str(data["source_xml"]),
                str(data["wavefunction_directory"]),
                str(data["output_id"]),
                tuple(int(x) for x in np.asarray(data["active_band_indices"], dtype=np.int64)),
                (
                    None
                    if np.asarray(data.get("raw_quartet_gram_eigenvalues", np.empty((0, 4)))).size == 0
                    else np.asarray(data["raw_quartet_gram_eigenvalues"], dtype=np.float64)
                ),
                (
                    None
                    if np.asarray(data.get("raw_quartet_orthonormality_error", np.empty((0,)))).size == 0
                    else np.asarray(data["raw_quartet_orthonormality_error"], dtype=np.float64)
                ),
                str(data.get("pair_construction", np.asarray("gamma6_character_local"))),
                (
                    None
                    if np.asarray(data.get("fixed_reference_principal_cos2", np.empty((0, 4)))).size == 0
                    else np.asarray(data["fixed_reference_principal_cos2"], dtype=np.float64)
                ),
            )
        result.validate()
        return result
