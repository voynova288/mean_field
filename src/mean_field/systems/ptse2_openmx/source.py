"""OpenMX source parsing and PAO-index contracts for PtSe2.

The parser intentionally preserves the original OpenMX atom order.  It does not
reuse geometry-sorted MoireKP tables because H/S rows and columns are indexed in
that original order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import scipy.sparse

# OpenMX 3.9 hard-codes BohrR=0.529177249 Angstrom in openmx_common.h.
# Use its reciprocal rather than a newer CODATA value so phase/Q contracts
# reproduce OpenMX's own internal Cartesian coordinates exactly.
ANGSTROM_TO_BOHR = 1.0 / 0.529177249
_L_BY_SYMBOL = {"s": 0, "p": 1, "d": 2, "f": 3}


@dataclass(frozen=True)
class OpenMXBasisSpec:
    """Selected OpenMX radial multiplicities for one species."""

    species: str
    orbital_label: str
    radial_multiplicities: tuple[int, ...]

    @classmethod
    def parse(cls, species: str, orbital_label: str) -> "OpenMXBasisSpec":
        try:
            selected = orbital_label.rsplit("-", 1)[1]
        except IndexError as exc:
            raise ValueError(f"invalid OpenMX orbital label: {orbital_label!r}") from exc
        matches = re.findall(r"([spdf])(\d+)", selected.lower())
        if not matches or "".join(f"{s}{n}" for s, n in matches) != selected.lower():
            raise ValueError(f"unsupported OpenMX basis suffix: {selected!r}")
        angular_momenta = [_L_BY_SYMBOL[symbol] for symbol, _ in matches]
        if len(set(angular_momenta)) != len(angular_momenta):
            raise ValueError(f"duplicate angular-momentum suffix in {selected!r}")
        if angular_momenta != sorted(angular_momenta):
            raise ValueError(f"basis suffix must be ordered by angular momentum: {selected!r}")
        max_l = max(angular_momenta)
        multiplicities = [0] * (max_l + 1)
        for symbol, count_text in matches:
            multiplicities[_L_BY_SYMBOL[symbol]] = int(count_text)
        return cls(
            species=species,
            orbital_label=orbital_label,
            radial_multiplicities=tuple(multiplicities),
        )

    @property
    def scalar_orbital_count(self) -> int:
        return sum((2 * l + 1) * count for l, count in enumerate(self.radial_multiplicities))

    @property
    def radial_function_count(self) -> int:
        return sum(self.radial_multiplicities)

    def channels(self) -> tuple[tuple[int, int, int], ...]:
        """Return OpenMX channels in l -> radial multiplicity -> real harmonic order."""

        return tuple(
            (l, radial, harmonic)
            for l, count in enumerate(self.radial_multiplicities)
            for radial in range(count)
            for harmonic in range(2 * l + 1)
        )


@dataclass(frozen=True)
class OpenMXFourierSourceManifest:
    """Hash-bound source contract for an OpenMX ``.ftpao`` artifact."""

    source_id: str
    openmx_version: str
    basis_specs: tuple[OpenMXBasisSpec, ...]
    radial_transform_path: Path
    radial_transform_sha256: str
    energy_cutoff_ry: float
    momentum_grid_size: int
    radial_grid_size: int
    padding_values_per_function: int
    pao_sha256: dict[str, str]
    vps_sha256: dict[str, str]
    manifest_path: Path

    @classmethod
    def from_file(cls, path: str | Path) -> "OpenMXFourierSourceManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != "ptse2_openmx_fourier_source/v1":
            raise ValueError("unsupported PtSe2 OpenMX Fourier manifest schema")
        radial = payload["radial_transform"]
        basis_specs = tuple(
            OpenMXBasisSpec.parse(item["species"], item["orbital_label"])
            for item in payload["basis"]
        )
        radial_path = (manifest_path.parent / radial["file"]).resolve()
        actual_hash = _sha256_file(radial_path)
        expected_hash = str(radial["sha256"])
        if actual_hash != expected_hash:
            raise ValueError(
                f".ftpao hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        return cls(
            source_id=str(payload["source_id"]),
            openmx_version=str(payload["openmx_version"]),
            basis_specs=basis_specs,
            radial_transform_path=radial_path,
            radial_transform_sha256=expected_hash,
            energy_cutoff_ry=float(radial["energy_cutoff_ry"]),
            momentum_grid_size=int(radial["momentum_grid_size"]),
            radial_grid_size=int(radial["radial_grid_size"]),
            padding_values_per_function=int(radial["padding_values_per_function"]),
            pao_sha256={str(key): str(value) for key, value in payload["pao"].items()},
            vps_sha256={str(key): str(value) for key, value in payload["vps"].items()},
            manifest_path=manifest_path,
        )

    def validate_source_basis(self, source: "OpenMXSource") -> None:
        if self.basis_specs != source.basis_specs:
            raise ValueError("Fourier manifest basis does not match OpenMX source basis")


@dataclass(frozen=True)
class OpenMXAtom:
    index: int
    species: str
    position_angstrom: tuple[float, float, float]


@dataclass(frozen=True)
class ScalarPAO:
    index: int
    atom_index: int
    species: str
    l: int
    radial: int
    harmonic: int


@dataclass(frozen=True)
class SpinorPAO:
    index: int
    scalar_index: int
    spin: int


@dataclass(frozen=True)
class OpenMXSource:
    """Structure and exact PAO ordering parsed from one OpenMX input."""

    lattice_angstrom: np.ndarray
    atoms: tuple[OpenMXAtom, ...]
    basis_specs: tuple[OpenMXBasisSpec, ...]
    spin_mode: str
    spin_orbit_coupling: bool
    input_sha256: str

    @classmethod
    def from_file(cls, path: str | Path) -> "OpenMXSource":
        return cls.from_text(Path(path).read_text())

    @classmethod
    def from_text(cls, text: str) -> "OpenMXSource":
        lattice_unit = _keyword_value(text, "Atoms.UnitVectors.Unit").lower()
        atom_unit = _keyword_value(text, "Atoms.SpeciesAndCoordinates.Unit").lower()
        if lattice_unit != "ang" or atom_unit != "ang":
            raise ValueError(
                "PtSe2 OpenMX adapter currently requires Ang lattice and coordinates, "
                f"got {lattice_unit!r}/{atom_unit!r}"
            )

        lattice_lines = _section_lines(text, "Atoms.UnitVectors")
        if len(lattice_lines) != 3:
            raise ValueError(f"expected three lattice vectors, got {len(lattice_lines)}")
        lattice = np.asarray(
            [[float(value) for value in line.split()[:3]] for line in lattice_lines],
            dtype=np.float64,
        )

        species_lines = _section_lines(text, "Definition.of.Atomic.Species")
        basis_specs: list[OpenMXBasisSpec] = []
        for line in species_lines:
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"malformed species line: {line!r}")
            basis_specs.append(OpenMXBasisSpec.parse(fields[0], fields[1]))
        if len({spec.species for spec in basis_specs}) != len(basis_specs):
            raise ValueError("duplicate species declarations")

        atom_lines = _section_lines(text, "Atoms.SpeciesAndCoordinates")
        declared_atoms = int(_keyword_value(text, "Atoms.Number"))
        if len(atom_lines) != declared_atoms:
            raise ValueError(
                f"Atoms.Number={declared_atoms} but coordinate block has {len(atom_lines)} rows"
            )
        known_species = {spec.species for spec in basis_specs}
        atoms: list[OpenMXAtom] = []
        for expected_index, line in enumerate(atom_lines, start=1):
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"malformed atom line: {line!r}")
            index = int(fields[0])
            species = fields[1]
            if index != expected_index:
                raise ValueError(
                    f"atom indices must preserve sequential OpenMX order: "
                    f"expected {expected_index}, got {index}"
                )
            if species not in known_species:
                raise ValueError(f"atom uses undeclared species {species!r}")
            atoms.append(
                OpenMXAtom(
                    index=index - 1,
                    species=species,
                    position_angstrom=tuple(float(value) for value in fields[2:5]),
                )
            )

        declared_species = int(_keyword_value(text, "Species.Number"))
        if declared_species != len(basis_specs):
            raise ValueError(
                f"Species.Number={declared_species} but species block has "
                f"{len(basis_specs)} rows"
            )
        spin_mode = _keyword_value(text, "scf.SpinPolarization").upper()
        if spin_mode != "NC":
            raise ValueError(
                "the PtSe2 reciprocal adapter supports only OpenMX NC ordering, "
                f"got {spin_mode!r}"
            )
        spin_orbit_text = _keyword_value(text, "scf.SpinOrbit.Coupling").lower()
        if spin_orbit_text not in {"on", "off"}:
            raise ValueError(f"invalid scf.SpinOrbit.Coupling value {spin_orbit_text!r}")
        if spin_orbit_text != "on":
            raise ValueError("the PtSe2 source contract requires spin-orbit coupling on")
        return cls(
            lattice_angstrom=lattice,
            atoms=tuple(atoms),
            basis_specs=tuple(basis_specs),
            spin_mode=spin_mode,
            spin_orbit_coupling=True,
            input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    @property
    def structure_digest(self) -> str:
        payload = {
            "lattice_angstrom": self.lattice_angstrom.tolist(),
            "atoms": [
                {
                    "index": atom.index,
                    "species": atom.species,
                    "position_angstrom": list(atom.position_angstrom),
                }
                for atom in self.atoms
            ],
            "basis": [
                {
                    "species": spec.species,
                    "orbital_label": spec.orbital_label,
                    "radial_multiplicities": list(spec.radial_multiplicities),
                }
                for spec in self.basis_specs
            ],
            "spin_mode": self.spin_mode,
            "spin_orbit_coupling": self.spin_orbit_coupling,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def basis_by_species(self) -> dict[str, OpenMXBasisSpec]:
        return {spec.species: spec for spec in self.basis_specs}

    @property
    def lattice_bohr(self) -> np.ndarray:
        return self.lattice_angstrom * ANGSTROM_TO_BOHR

    @property
    def reciprocal_bohr(self) -> np.ndarray:
        """Reciprocal row vectors satisfying a_i dot b_j = 2 pi delta_ij."""

        return 2.0 * np.pi * np.linalg.inv(self.lattice_bohr).T

    @property
    def reciprocal_angstrom(self) -> np.ndarray:
        return 2.0 * np.pi * np.linalg.inv(self.lattice_angstrom).T

    @property
    def volume_bohr3(self) -> float:
        return float(abs(np.linalg.det(self.lattice_bohr)))

    @property
    def spin_blocks(self) -> int:
        return 2

    def scalar_pao_map(self) -> tuple[ScalarPAO, ...]:
        basis_by_species = self.basis_by_species
        orbitals: list[ScalarPAO] = []
        for atom in self.atoms:
            for l, radial, harmonic in basis_by_species[atom.species].channels():
                orbitals.append(
                    ScalarPAO(
                        index=len(orbitals),
                        atom_index=atom.index,
                        species=atom.species,
                        l=l,
                        radial=radial,
                        harmonic=harmonic,
                    )
                )
        return tuple(orbitals)

    def spinor_pao_map(self) -> tuple[SpinorPAO, ...]:
        scalar_count = len(self.scalar_pao_map())
        return tuple(
            SpinorPAO(
                index=spin * scalar_count + scalar_index,
                scalar_index=scalar_index,
                spin=spin,
            )
            for spin in range(self.spin_blocks)
            for scalar_index in range(scalar_count)
        )

    @property
    def scalar_pao_count(self) -> int:
        return sum(
            self.basis_by_species[atom.species].scalar_orbital_count
            for atom in self.atoms
        )

    @property
    def spinor_pao_count(self) -> int:
        return self.spin_blocks * self.scalar_pao_count

    def spinor_centers_angstrom(self) -> np.ndarray:
        scalar = np.asarray(
            [
                self.atoms[orbital.atom_index].position_angstrom
                for orbital in self.scalar_pao_map()
            ],
            dtype=np.float64,
        )
        return np.concatenate([scalar] * self.spin_blocks, axis=0)

    def scalar_indices_by_species(self, species: str) -> np.ndarray:
        return np.asarray(
            [orbital.index for orbital in self.scalar_pao_map() if orbital.species == species],
            dtype=np.int64,
        )


def load_sparse_realspace_npz(
    path: str | Path,
) -> dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load OpenMX sparse real-space blocks without densifying them."""

    blocks: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as payload:
        prefixes = sorted(key[: -len("_val")] for key in payload.files if key.endswith("_val"))
        for prefix in prefixes:
            cell = tuple(int(value) for value in re.findall(r"-?\d+", prefix))
            if len(cell) != 3:
                raise ValueError(f"cannot parse real-space cell from {prefix!r}")
            blocks[cell] = (
                np.asarray(payload[f"{prefix}_row"], dtype=np.int64),
                np.asarray(payload[f"{prefix}_col"], dtype=np.int64),
                np.asarray(payload[f"{prefix}_val"], dtype=np.complex128),
            )
    return blocks


def assemble_super_gauge_matrix(
    blocks: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    source: OpenMXSource,
    k_fractional: Iterable[float],
) -> scipy.sparse.csr_matrix:
    """Assemble H(k) or S(k) in the physical OpenMX-order super gauge."""

    k = tuple(k_fractional)
    return assemble_super_gauge_transfer_matrix(blocks, source, k, k)


def assemble_super_gauge_transfer_matrix(
    blocks: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    source: OpenMXSource,
    k_target_fractional: Iterable[float],
    k_source_fractional: Iterable[float],
) -> scipy.sparse.csr_matrix:
    """Assemble real-space PAO blocks with target/source super-gauge phases."""

    k_target = np.asarray(tuple(k_target_fractional), dtype=np.float64)
    k_source = np.asarray(tuple(k_source_fractional), dtype=np.float64)
    if k_target.shape != (3,) or k_source.shape != (3,):
        raise ValueError("target/source k points must contain three components")
    target_cart = k_target @ source.reciprocal_angstrom
    source_cart = k_source @ source.reciprocal_angstrom
    centers = source.spinor_centers_angstrom()
    dimension = source.spinor_pao_count

    if not blocks:
        raise ValueError("at least one real-space sparse block is required")
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for cell, (row, col, value) in blocks.items():
        if row.shape != col.shape or row.shape != value.shape:
            raise ValueError(f"inconsistent sparse arrays for cell {cell}")
        if row.size and (
            row.min() < 0
            or col.min() < 0
            or row.max() >= dimension
            or col.max() >= dimension
        ):
            raise ValueError(f"sparse index is outside spinor PAO dimension in cell {cell}")
        r_cart = np.asarray(cell, dtype=np.float64) @ source.lattice_angstrom
        phase = np.exp(
            1j * float(source_cart @ r_cart)
            - 1j * (centers[row] @ target_cart)
            + 1j * (centers[col] @ source_cart)
        )
        rows.append(row)
        cols.append(col)
        values.append(value * phase)

    return scipy.sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dimension, dimension),
        dtype=np.complex128,
    ).tocsr()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _keyword_value(text: str, keyword: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(keyword)}\s+([^#\s]+)", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing OpenMX keyword {keyword!r}")
    return match.group(1)


def _section_lines(text: str, name: str) -> list[str]:
    pattern = re.compile(
        rf"^\s*<{re.escape(name)}\s*$\n(.*?)^\s*{re.escape(name)}>\s*$",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"missing OpenMX section {name!r}")
    lines: list[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines
