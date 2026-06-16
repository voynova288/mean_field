from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

DensityAxisOrder = Literal["abk"]
ReferencePolicy = Literal["require", "average", "none"]
ProjectorOrientation = Literal["stored", "ket"]


class DensityConvention(str, Enum):
    """Canonical HF density conventions.

    ``PROJECTOR`` is the physical occupied-state projector ``P`` in the stored
    matrix orientation ``P_ab = <c_a^† c_b>``.
    ``STORED_DELTA`` is the archive/SCF delta ``D = P - P_ref``.
    ``HALF_SHIFTED`` is the common special case ``D = P - 1/2 I``.
    """

    PROJECTOR = "projector"
    STORED_DELTA = "stored_delta"
    HALF_SHIFTED = "half_shifted"


def _resolve_density_convention(convention: DensityConvention | str) -> DensityConvention:
    if isinstance(convention, DensityConvention):
        return convention
    return DensityConvention(str(convention))


@dataclass(frozen=True)
class ReferenceDensity:
    """Reference density used to convert a stored delta into a projector."""

    data: np.ndarray
    convention: str = "explicit"
    axis_order: DensityAxisOrder = "abk"

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.complex128)
        validate_density_array(arr, axis_order=self.axis_order)
        object.__setattr__(self, "data", arr)

    @classmethod
    def average(cls, nt: int, nk: int, *, value: float = 0.5) -> "ReferenceDensity":
        return cls(average_reference_density(nt, nk, value=value), convention=f"average:{float(value):.12g}")

    @classmethod
    def zeros(cls, nt: int, nk: int) -> "ReferenceDensity":
        return cls(np.zeros((int(nt), int(nt), int(nk)), dtype=np.complex128), convention="zero")


@dataclass(frozen=True)
class DensityBundle:
    """Density array plus its convention and reference metadata."""

    data: np.ndarray
    convention: DensityConvention | str
    reference: ReferenceDensity | None = None
    axis_order: DensityAxisOrder = "abk"

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.complex128)
        validate_density_array(arr, axis_order=self.axis_order)
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "convention", _resolve_density_convention(self.convention))
        if self.reference is not None and self.reference.data.shape != arr.shape:
            raise ValueError(f"reference shape {self.reference.data.shape} does not match density {arr.shape}")

    def as_projector(self, *, orientation: ProjectorOrientation = "stored") -> np.ndarray:
        projector = density_to_projector(self.data, self.convention, reference=self.reference)
        if orientation == "stored":
            return projector
        if orientation == "ket":
            return stored_orientation_to_ket_projector(projector)
        raise ValueError(f"Unsupported projector orientation={orientation!r}")

    def as_stored_delta(self, reference: ReferenceDensity | None = None) -> np.ndarray:
        ref = reference if reference is not None else self.reference
        return density_to_stored_delta(self.data, self.convention, reference=ref)


def validate_density_array(array: np.ndarray, *, axis_order: DensityAxisOrder = "abk") -> None:
    if axis_order != "abk":
        raise ValueError(f"Unsupported density axis_order={axis_order!r}; expected 'abk'")
    if array.ndim != 3 or array.shape[0] != array.shape[1]:
        raise ValueError(f"Expected density shape (nt, nt, nk) for axis_order='abk', got {array.shape}")


def average_reference_density(nt: int, nk: int, *, value: float = 0.5) -> np.ndarray:
    if int(nt) <= 0 or int(nk) <= 0:
        raise ValueError(f"nt and nk must be positive, got nt={nt}, nk={nk}")
    eye = np.eye(int(nt), dtype=np.complex128)[:, :, None]
    return float(value) * eye * np.ones((1, 1, int(nk)), dtype=np.complex128)


def resolve_reference_density(
    shape: tuple[int, int, int],
    reference: ReferenceDensity | np.ndarray | None = None,
    *,
    reference_policy: ReferencePolicy = "average",
) -> ReferenceDensity:
    nt, _, nk = shape
    if reference is None:
        if reference_policy == "require":
            raise ValueError("reference_density is required by reference_policy='require'")
        if reference_policy == "average":
            return ReferenceDensity.average(nt, nk)
        if reference_policy == "none":
            return ReferenceDensity.zeros(nt, nk)
        raise ValueError(f"Unsupported reference_policy={reference_policy!r}")
    if isinstance(reference, ReferenceDensity):
        resolved = reference
    else:
        resolved = ReferenceDensity(np.asarray(reference, dtype=np.complex128))
    if resolved.data.shape != shape:
        raise ValueError(f"reference density shape {resolved.data.shape} does not match density shape {shape}")
    return resolved


def stored_orientation_to_ket_projector(projector: np.ndarray) -> np.ndarray:
    arr = np.asarray(projector, dtype=np.complex128)
    if arr.ndim < 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected projector with square leading matrix axes, got {arr.shape}")
    return np.swapaxes(arr, 0, 1).copy()


def ket_projector_to_stored_orientation(projector: np.ndarray) -> np.ndarray:
    return stored_orientation_to_ket_projector(projector)


def density_to_projector(
    density: np.ndarray,
    convention: DensityConvention | str,
    *,
    reference: ReferenceDensity | np.ndarray | None = None,
    reference_policy: ReferencePolicy = "average",
) -> np.ndarray:
    arr = np.asarray(density, dtype=np.complex128)
    validate_density_array(arr)
    resolved = _resolve_density_convention(convention)
    if resolved == DensityConvention.PROJECTOR:
        return arr.copy()
    if resolved == DensityConvention.HALF_SHIFTED:
        return arr + average_reference_density(arr.shape[0], arr.shape[2])
    if resolved == DensityConvention.STORED_DELTA:
        ref = resolve_reference_density(arr.shape, reference, reference_policy=reference_policy)
        return arr + ref.data
    raise ValueError(f"Unsupported density convention={convention!r}")


def density_to_stored_delta(
    density: np.ndarray,
    convention: DensityConvention | str,
    *,
    reference: ReferenceDensity | np.ndarray | None = None,
    reference_policy: ReferencePolicy = "average",
) -> np.ndarray:
    arr = np.asarray(density, dtype=np.complex128)
    validate_density_array(arr)
    resolved = _resolve_density_convention(convention)
    if resolved == DensityConvention.STORED_DELTA:
        return arr.copy()
    if resolved == DensityConvention.HALF_SHIFTED:
        return arr.copy()
    if resolved == DensityConvention.PROJECTOR:
        ref = resolve_reference_density(arr.shape, reference, reference_policy=reference_policy)
        return arr - ref.data
    raise ValueError(f"Unsupported density convention={convention!r}")


def stored_density_to_projector(
    density: np.ndarray,
    reference_density: np.ndarray | ReferenceDensity | None = None,
    *,
    reference_policy: ReferencePolicy = "average",
    convention: ProjectorOrientation = "ket",
) -> np.ndarray:
    projector = density_to_projector(
        density,
        DensityConvention.STORED_DELTA,
        reference=reference_density,
        reference_policy=reference_policy,
    )
    if convention == "stored":
        return projector
    if convention == "ket":
        return stored_orientation_to_ket_projector(projector)
    raise ValueError(f"Unsupported projector convention={convention!r}")


__all__ = [
    "DensityAxisOrder",
    "DensityBundle",
    "DensityConvention",
    "ProjectorOrientation",
    "ReferenceDensity",
    "ReferencePolicy",
    "average_reference_density",
    "density_to_projector",
    "density_to_stored_delta",
    "ket_projector_to_stored_orientation",
    "resolve_reference_density",
    "stored_density_to_projector",
    "stored_orientation_to_ket_projector",
    "validate_density_array",
]
