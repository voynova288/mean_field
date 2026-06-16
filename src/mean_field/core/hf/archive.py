from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..io.artifacts import NpzArtifactSummary, read_npz_scalar, summarize_npz_artifact
from .density import (
    average_reference_density as _average_reference_density,
    stored_density_to_projector as _stored_density_to_projector,
)

HF_ARCHIVE_READER_VERSION = "hf_state_archive_reader_v1"
ReferencePolicy = Literal["require", "average", "none"]
ProjectorConvention = Literal["stored", "ket"]


@dataclass(frozen=True)
class HFArchiveSummary:
    path: Path
    artifact: NpzArtifactSummary
    density_shape: tuple[int, int, int] | None
    hamiltonian_shape: tuple[int, int, int] | None
    h0_shape: tuple[int, int, int] | None
    energy_shape: tuple[int, ...] | None
    k_grid_shape: tuple[int, ...] | None
    n_spin: int | None = None
    n_eta: int | None = None
    n_band: int | None = None
    active_valence_bands: int | None = None
    has_reference_density: bool = False
    cache_dir: str | None = None
    cache_key_basis: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def nt(self) -> int | None:
        return None if self.density_shape is None else int(self.density_shape[0])

    @property
    def nk(self) -> int | None:
        return None if self.density_shape is None else int(self.density_shape[2])

    def to_dict(self) -> dict[str, object]:
        return {
            "reader_version": HF_ARCHIVE_READER_VERSION,
            "path": str(self.path),
            "keys": list(self.artifact.keys),
            "density_shape": None if self.density_shape is None else list(self.density_shape),
            "hamiltonian_shape": None if self.hamiltonian_shape is None else list(self.hamiltonian_shape),
            "h0_shape": None if self.h0_shape is None else list(self.h0_shape),
            "energy_shape": None if self.energy_shape is None else list(self.energy_shape),
            "k_grid_shape": None if self.k_grid_shape is None else list(self.k_grid_shape),
            "n_spin": self.n_spin,
            "n_eta": self.n_eta,
            "n_band": self.n_band,
            "active_valence_bands": self.active_valence_bands,
            "has_reference_density": bool(self.has_reference_density),
            "cache_dir": self.cache_dir,
            "cache_key_basis": self.cache_key_basis,
            "metadata": dict(self.metadata),
        }


def _shape(summary: NpzArtifactSummary, key: str) -> tuple[int, ...] | None:
    try:
        return summary.array(key).shape
    except KeyError:
        return None


def _int_or_none(value: object | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def summarize_hf_state_archive(path: str | Path) -> HFArchiveSummary:
    archive_path = Path(path)
    artifact = summarize_npz_artifact(archive_path)
    with np.load(archive_path, allow_pickle=False) as payload:
        metadata = {
            "nu": read_npz_scalar(payload, "nu"),
            "scheme": read_npz_scalar(payload, "scheme"),
            "basis_periodic_gauge": read_npz_scalar(payload, "basis_periodic_gauge"),
            "zero_literal_q0_fock": read_npz_scalar(payload, "zero_literal_q0_fock"),
            "iteration": read_npz_scalar(payload, "iteration"),
        }
        return HFArchiveSummary(
            path=archive_path,
            artifact=artifact,
            density_shape=_shape(artifact, "density"),  # type: ignore[arg-type]
            hamiltonian_shape=_shape(artifact, "hamiltonian"),  # type: ignore[arg-type]
            h0_shape=_shape(artifact, "h0"),  # type: ignore[arg-type]
            energy_shape=_shape(artifact, "energies_mev"),
            k_grid_shape=_shape(artifact, "k_grid_frac"),
            n_spin=_int_or_none(read_npz_scalar(payload, "n_spin")),
            n_eta=_int_or_none(read_npz_scalar(payload, "n_eta")),
            n_band=_int_or_none(read_npz_scalar(payload, "n_band")),
            active_valence_bands=_int_or_none(read_npz_scalar(payload, "active_valence_bands")),
            has_reference_density="reference_density" in artifact.keys,
            cache_dir=_str_or_none(read_npz_scalar(payload, "cache_dir")),
            cache_key_basis=_str_or_none(read_npz_scalar(payload, "cache_key_basis")),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )


def validate_hf_archive_shapes(summary: HFArchiveSummary) -> None:
    density_shape = summary.density_shape
    if density_shape is None:
        raise ValueError(f"HF archive {summary.path} has no density array")
    if len(density_shape) != 3 or density_shape[0] != density_shape[1]:
        raise ValueError(f"Expected density shape (nt, nt, nk), got {density_shape}")
    for name, shape in (
        ("hamiltonian", summary.hamiltonian_shape),
        ("h0", summary.h0_shape),
    ):
        if shape is not None and shape != density_shape:
            raise ValueError(f"{name} shape {shape} does not match density shape {density_shape}")
    if summary.n_spin is not None and summary.n_eta is not None and summary.n_band is not None:
        expected_nt = int(summary.n_spin) * int(summary.n_eta) * int(summary.n_band)
        if expected_nt != density_shape[0]:
            raise ValueError(
                f"n_spin*n_eta*n_band={expected_nt} does not match density nt={density_shape[0]}"
            )


def average_reference_density(nt: int, nk: int, *, value: float = 0.5) -> np.ndarray:
    return _average_reference_density(nt, nk, value=value)


def stored_density_to_projector(
    density: np.ndarray,
    reference_density: np.ndarray | None = None,
    *,
    reference_policy: ReferencePolicy = "average",
    convention: ProjectorConvention = "ket",
) -> np.ndarray:
    """Convert a saved HF density/delta archive array into a projector-like array.

    Historical RnG/hBN archives store the density matrix in the ``P_ab=<c_a†c_b>``
    convention, often as a delta from a reference.  For ket-space wavefunction
    contractions, the corresponding matrix is the per-k transpose.  Use
    ``convention='stored'`` to keep the archive orientation.
    """

    return _stored_density_to_projector(
        density,
        reference_density,
        reference_policy=reference_policy,
        convention=convention,
    )


def load_projector_from_hf_archive(
    path: str | Path,
    *,
    reference_policy: ReferencePolicy = "average",
    convention: ProjectorConvention = "ket",
) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as payload:
        density = np.asarray(payload["density"], dtype=np.complex128)
        reference = np.asarray(payload["reference_density"], dtype=np.complex128) if "reference_density" in payload else None
    return stored_density_to_projector(
        density,
        reference,
        reference_policy=reference_policy,
        convention=convention,
    )


__all__ = [
    "HF_ARCHIVE_READER_VERSION",
    "HFArchiveSummary",
    "ProjectorConvention",
    "ReferencePolicy",
    "average_reference_density",
    "load_projector_from_hf_archive",
    "stored_density_to_projector",
    "summarize_hf_state_archive",
    "validate_hf_archive_shapes",
]
