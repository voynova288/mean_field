"""Cell-centered WS cubature and FFT backend for local four-pocket TiSe2.

This module is a system-local numerical regulator for the expanded local basis
``(v_Gamma, c_L1, c_L2, c_L3)`` and its 28 fixed-representative ``G=0``
routes.  It is **not** a full-material Brillouin-zone model and it does not
supply the missing three-dimensional eight-sector closure.  In particular,
physical representative differences are used literally: no transfer is
reduced modulo a reciprocal lattice vector.

The mesh is a cell-centered equal-weight WS cubature normalized exactly to
the volume of one hexagonal Wigner--Seitz prism.  The prism
geometry and volume are exact consequences of the bound TiSe2 geometry; the
uniform cubature cells and equal-volume-sphere Coulomb self cell are regulator
approximations, not exact midpoint-cell quadrature.  The exchange backend
embeds the hexagon in its smallest rectangular label box, zeros points outside
the hexagonal mask, and uses zero-padded FFTs only as an exact algorithm for
the resulting *linear* convolution.  It does not impose periodic convolution
boundary conditions and never constructs an ``nk x nk`` production array.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fftn, ifftn, next_fast_len

from .folded_hf import (
    FoldedTiSe2Geometry,
    physical_momentum_routes,
    physical_q_vectors,
)
from .hf import BOLTZMANN_EV_PER_K, COULOMB_EV_ANGSTROM
from .local_full_hf import (
    ELECTRONS_PER_LOCAL_MOMENTUM,
    LocalFullSpectrum,
    diagonalize_local_full_finite_temperature,
    local_full_h0,
)
from .monney import MonneyTiSe2Parameters

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

LOCAL_DIMENSION = 4
_TWO_PI_CUBED = (2.0 * np.pi) ** 3


def _exact_nonnegative_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be an exact nonnegative integer")
    return value


def _positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class LocalWSMesh:
    """Cell-centered equal-weight WS cubature normalized exactly to WS volume.

    With ``d=2*M+1`` and ``dz=2*L+1``, integer labels obey
    ``max(|m|,|n|,|m+n|)<=M`` and ``|l|<=L``.  Their Cartesian points are

    ``x=Qparallel*m/d``, ``y=Qparallel*(m+2*n)/(sqrt(3)*d)``,
    ``z=Qz*l/dz``.

    ``B_Ainv`` is retained as a compatibility field name, but means
    ``Qparallel=|Q1_xy|``; it is not a primitive reciprocal-vector magnitude.
    ``Qz_Ainv`` means the reduced-zone z translation ``Q1_z``.  Both are
    derived from :func:`physical_q_vectors` for ``geometry`` and cannot be set
    independently.  The strict label bisectors and the WS prism geometry and
    volume are exact.  Treating all represented cubature cells as uniform is
    a regulator approximation.  Weights are equal and normalized exactly to
    the WS volume, including the conventional ``(2*pi)^-3`` factor.
    """

    points_Ainv: FloatArray
    weights_Ainv3: FloatArray
    labels: IntArray
    M: int
    L: int
    geometry: FoldedTiSe2Geometry
    q_vectors_Ainv: FloatArray
    B_Ainv: float
    Qz_Ainv: float
    area_RBZ_Ainv2: float
    volume_RBZ_Ainv3: float
    cell_volume_Ainv3: float
    box_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        M = _exact_nonnegative_integer(self.M, name="M")
        L = _exact_nonnegative_integer(self.L, name="L")
        if not isinstance(self.geometry, FoldedTiSe2Geometry):
            raise TypeError("geometry must be a FoldedTiSe2Geometry")
        q_vectors = _validate_q_vectors(self.q_vectors_Ainv)
        expected_q_vectors = _validate_q_vectors(physical_q_vectors(self.geometry))
        if not np.array_equal(q_vectors, expected_q_vectors):
            raise ValueError("q_vectors_Ainv must be derived exactly from geometry")
        expected_B = float(np.linalg.norm(expected_q_vectors[1, :2]))
        expected_Qz = float(expected_q_vectors[1, 2])
        B = _positive_finite(self.B_Ainv, name="B_Ainv (Qparallel)")
        Qz = _positive_finite(self.Qz_Ainv, name="Qz_Ainv (reduced z translation)")
        if B != expected_B or Qz != expected_Qz:
            raise ValueError(
                "B_Ainv/Qz_Ainv must equal Qparallel=|Q1_xy| and Q1_z "
                "derived from geometry"
            )
        points = np.asarray(self.points_Ainv, dtype=np.float64)
        weights = np.asarray(self.weights_Ainv3, dtype=np.float64)
        labels = np.asarray(self.labels, dtype=np.int64)
        mask = np.asarray(self.box_mask, dtype=bool)
        expected_nxy = 3 * M * (M + 1) + 1
        expected_nk = expected_nxy * (2 * L + 1)
        if points.shape != (expected_nk, 3):
            raise ValueError(f"points_Ainv must have shape ({expected_nk},3)")
        if weights.shape != (expected_nk,):
            raise ValueError(f"weights_Ainv3 must have shape ({expected_nk},)")
        if labels.shape != (expected_nk, 3):
            raise ValueError(f"labels must have shape ({expected_nk},3)")
        if mask.shape != self.box_shape:
            raise ValueError(f"box_mask must have shape {self.box_shape}")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)):
            raise ValueError("mesh points and weights must be finite")
        if np.any(weights <= 0.0) or not np.all(weights == weights[0]):
            raise ValueError("WS quadrature weights must be positive and uniform")
        if np.unique(labels, axis=0).shape[0] != expected_nk:
            raise ValueError("WS labels must be unique")
        m, n, ell = labels.T
        expected_points = np.column_stack(
            (
                B * m / self.d,
                B * (m + 2 * n) / (sqrt(3.0) * self.d),
                Qz * ell / self.dz,
            )
        ).astype(np.float64, copy=False)
        if not np.array_equal(points, expected_points):
            raise ValueError(
                "points_Ainv must equal the exact geometry-bound label-to-point map"
            )
        if not np.all(
            (2 * np.abs(m) < self.d)
            & (2 * np.abs(n) < self.d)
            & (2 * np.abs(m + n) < self.d)
            & (2 * np.abs(ell) < self.dz)
        ):
            raise ValueError("labels must lie strictly inside all WS midpoint bisectors")
        indices = labels + np.asarray((M, M, L), dtype=np.int64)
        if not np.all(mask[indices[:, 0], indices[:, 1], indices[:, 2]]):
            raise ValueError("every label must select a true box-mask point")
        if int(np.count_nonzero(mask)) != expected_nk:
            raise ValueError("box_mask must contain exactly the WS label inventory")

        expected_area = sqrt(3.0) * B**2 / 2.0
        expected_volume = expected_area * Qz
        expected_cell = expected_volume / expected_nk
        scale = max(1.0, expected_volume)
        tolerance = 32.0 * np.finfo(float).eps * scale
        if abs(float(self.area_RBZ_Ainv2) - expected_area) > tolerance:
            raise ValueError("area_RBZ_Ainv2 is inconsistent with B_Ainv")
        if abs(float(self.volume_RBZ_Ainv3) - expected_volume) > tolerance:
            raise ValueError("volume_RBZ_Ainv3 is inconsistent with the WS prism")
        if abs(float(self.cell_volume_Ainv3) - expected_cell) > tolerance:
            raise ValueError("cell_volume_Ainv3 must equal exact prism volume/nk")
        expected_weight = expected_cell / _TWO_PI_CUBED
        if not np.all(weights == expected_weight):
            raise ValueError(
                "weights must be equal and normalized exactly to the WS volume"
            )

        label_set = {tuple(int(x) for x in row) for row in labels}
        inverted = {(-m0, -n0, -l0) for m0, n0, l0 in label_set}
        rotated = {(-n0, m0 + n0, l0) for m0, n0, l0 in label_set}
        if inverted != label_set:
            raise ValueError("WS labels must be exactly inversion closed")
        if rotated != label_set:
            raise ValueError("WS labels must be exactly C6 closed")

        object.__setattr__(self, "points_Ainv", points)
        object.__setattr__(self, "weights_Ainv3", weights)
        object.__setattr__(self, "q_vectors_Ainv", q_vectors)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "box_mask", mask)
        object.__setattr__(self, "B_Ainv", B)
        object.__setattr__(self, "Qz_Ainv", Qz)
        object.__setattr__(self, "area_RBZ_Ainv2", expected_area)
        object.__setattr__(self, "volume_RBZ_Ainv3", expected_volume)
        object.__setattr__(self, "cell_volume_Ainv3", expected_cell)

    @property
    def Qparallel_Ainv(self) -> float:
        """Return ``|Q1_xy|``; ``B_Ainv`` is its compatibility name."""

        return self.B_Ainv

    @property
    def d(self) -> int:
        return 2 * self.M + 1

    @property
    def dz(self) -> int:
        return 2 * self.L + 1

    @property
    def box_shape(self) -> tuple[int, int, int]:
        return (self.d, self.d, self.dz)

    @property
    def nk(self) -> int:
        return int(self.labels.shape[0])

    @property
    def normalized_weights(self) -> FloatArray:
        return np.full(self.nk, 1.0 / self.nk, dtype=np.float64)

    @property
    def gamma_index(self) -> int:
        hits = np.flatnonzero(np.all(self.labels == 0, axis=1))
        if hits.size != 1:
            raise ValueError("WS mesh must contain exactly one zero label")
        return int(hits[0])

    @property
    def box_indices(self) -> IntArray:
        return np.asarray(
            self.labels + np.asarray((self.M, self.M, self.L), dtype=np.int64),
            dtype=np.int64,
        )


def build_local_ws_mesh(
    *,
    M: int,
    L: int,
    geometry: FoldedTiSe2Geometry | None = None,
) -> LocalWSMesh:
    """Build geometry-bound cell-centered equal-weight WS cubature.

    The in-plane scale is ``Qparallel=|Q1_xy|`` and the z scale is the
    reduced-zone translation ``Q1_z``.  Neither is an independent production
    parameter.
    """

    M = _exact_nonnegative_integer(M, name="M")
    L = _exact_nonnegative_integer(L, name="L")
    lattice = FoldedTiSe2Geometry() if geometry is None else geometry
    if not isinstance(lattice, FoldedTiSe2Geometry):
        raise TypeError("geometry must be a FoldedTiSe2Geometry")
    q_vectors = _validate_q_vectors(physical_q_vectors(lattice))
    B = float(np.linalg.norm(q_vectors[1, :2]))
    Qz = float(q_vectors[1, 2])
    d = 2 * M + 1
    dz = 2 * L + 1

    labels = np.asarray(
        [
            (m, n, ell)
            for ell in range(-L, L + 1)
            for m in range(-M, M + 1)
            for n in range(-M, M + 1)
            if max(abs(m), abs(n), abs(m + n)) <= M
        ],
        dtype=np.int64,
    )
    order = np.lexsort((labels[:, 1], labels[:, 0], labels[:, 2]))
    labels = labels[order]
    m, n, ell = labels.T
    points = np.column_stack(
        (
            B * m / d,
            B * (m + 2 * n) / (sqrt(3.0) * d),
            Qz * ell / dz,
        )
    ).astype(np.float64, copy=False)

    area = sqrt(3.0) * B**2 / 2.0
    volume = area * Qz
    cell_volume = volume / labels.shape[0]
    weights = np.full(labels.shape[0], cell_volume / _TWO_PI_CUBED, dtype=np.float64)
    mask = np.zeros((d, d, dz), dtype=bool)
    indices = labels + np.asarray((M, M, L), dtype=np.int64)
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return LocalWSMesh(
        points_Ainv=points,
        weights_Ainv3=weights,
        labels=labels,
        M=M,
        L=L,
        geometry=lattice,
        q_vectors_Ainv=q_vectors,
        B_Ainv=B,
        Qz_Ainv=Qz,
        area_RBZ_Ainv2=area,
        volume_RBZ_Ainv3=volume,
        cell_volume_Ainv3=cell_volume,
        box_mask=mask,
    )


def _validate_fft_workers(fft_workers: int) -> int:
    if type(fft_workers) is not int or fft_workers <= 0:
        raise TypeError("fft_workers must be an exact positive integer")
    return fft_workers


def _validate_q_vectors(q_vectors_Ainv: np.ndarray) -> FloatArray:
    q_vectors = np.asarray(q_vectors_Ainv, dtype=np.float64)
    if q_vectors.shape != (LOCAL_DIMENSION, 3) or not np.all(np.isfinite(q_vectors)):
        raise ValueError("q_vectors_Ainv must be finite with shape (4,3)")
    if not np.array_equal(q_vectors[0], np.zeros(3, dtype=np.float64)):
        raise ValueError("the first fixed representative must be exactly zero")
    if np.unique(q_vectors, axis=0).shape[0] != LOCAL_DIMENSION:
        raise ValueError("fixed physical q representatives must be distinct")
    return q_vectors


def estimate_local_ws_fft_peak_bytes(
    mesh: LocalWSMesh,
    *,
    fft_shape: tuple[int, int, int] | None = None,
) -> int:
    """Conservatively estimate peak resident bytes without any ``nk**2`` term.

    The estimate retains all sixteen complex FFT kernels and budgets seven
    additional full sixteen-channel complex FFT inventories for source,
    transformed source, result, and backend workspaces.  It also budgets the
    signed-difference cubature and construction temporaries at deliberately
    inflated byte counts.  The estimate is checked before production arrays
    are allocated; it is conservative for this implementation, not a promise
    about unbounded external FFT-worker scratch allocation.
    """

    shape = (
        tuple(next_fast_len(2 * size - 1) for size in mesh.box_shape)
        if fft_shape is None
        else tuple(fft_shape)
    )
    if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
        raise TypeError("fft_shape must contain three exact positive integers")
    minimum = tuple(2 * size - 1 for size in mesh.box_shape)
    if any(actual < required for actual, required in zip(shape, minimum)):
        raise ValueError("fft_shape must support zero-padded linear convolution")
    nfft = prod(shape)
    ndifference = prod(minimum)
    complex_bytes = np.dtype(np.complex128).itemsize
    # 16 persistent kernels + 7 complete 16-channel action/build inventories.
    fft_inventories = 8 * LOCAL_DIMENSION**2 * nfft * complex_bytes
    # Label grids, vector displacements/transfers, masks, q2, and values.
    difference_work = 192 * ndifference
    fixed_overhead = 16384 + mesh.nk * (3 * 8 + 3 * 8 + 8 + 1)
    return int(fft_inventories + difference_work + fixed_overhead)

@dataclass(frozen=True)
class LocalWSFFTKernel:
    """Sixteen ordered fixed-representative linear-convolution kernels.

    ``ordered_kernel_fft_ev[s,v]`` is the FFT of the signed-displacement
    kernel ``w*V(delta_p+Q_s-Q_v)`` packed by ``delta % fft_shape``.  The
    transform lengths are at least ``2*box_shape-1`` and all displacement
    labels outside the exact difference hexagon are zero.  The only admitted
    Coulomb singularities are ``s==v`` at zero displacement, where an
    equal-volume-sphere self-cell *regulator approximation* replaces
    ``1/q^2``.  ``estimated_peak_bytes`` records the conservative preallocation
    estimate associated with ``fft_shape``.
    """

    mesh: LocalWSMesh
    q_vectors_Ainv: FloatArray
    epsilon_r: float
    self_cell_ev: float
    hartree_ev: FloatArray
    ordered_kernel_fft_ev: ComplexArray
    fft_shape: tuple[int, int, int]
    estimated_peak_bytes: int
    singular_inventory: tuple[tuple[int, int, int, int, int], ...]
    route_count: int = 28
    route_policy: str = "fixed_representative_g0"
    q0_hartree_policy: str = "neutral_background_global_canonical"

    def __post_init__(self) -> None:
        q_vectors = _validate_q_vectors(self.q_vectors_Ainv)
        if not np.array_equal(q_vectors, self.mesh.q_vectors_Ainv):
            raise ValueError("kernel and mesh q vectors must match")
        eps = _positive_finite(self.epsilon_r, name="epsilon_r")
        self_cell = _positive_finite(self.self_cell_ev, name="self_cell_ev")
        hartree = np.asarray(self.hartree_ev, dtype=np.float64)
        kernels = np.asarray(self.ordered_kernel_fft_ev, dtype=np.complex128)
        if len(self.fft_shape) != 3 or any(type(value) is not int for value in self.fft_shape):
            raise TypeError("fft_shape must contain three exact integers")
        minimum = tuple(2 * size - 1 for size in self.mesh.box_shape)
        if any(actual < required for actual, required in zip(self.fft_shape, minimum)):
            raise ValueError("fft_shape must support zero-padded linear convolution")
        expected_peak = estimate_local_ws_fft_peak_bytes(
            self.mesh, fft_shape=self.fft_shape
        )
        if type(self.estimated_peak_bytes) is not int:
            raise TypeError("estimated_peak_bytes must be an exact integer")
        if self.estimated_peak_bytes != expected_peak:
            raise ValueError("estimated_peak_bytes is inconsistent with fft_shape")
        expected = (LOCAL_DIMENSION, LOCAL_DIMENSION, *self.fft_shape)
        if kernels.shape != expected or not np.all(np.isfinite(kernels)):
            raise ValueError(f"ordered_kernel_fft_ev must be finite with shape {expected}")
        if hartree.shape != (LOCAL_DIMENSION, LOCAL_DIMENSION):
            raise ValueError("hartree_ev must have shape (4,4)")
        if not np.all(np.isfinite(hartree)) or np.any(hartree < 0.0):
            raise ValueError("Hartree coefficients must be finite and nonnegative")
        if not np.array_equal(np.diag(hartree), np.zeros(LOCAL_DIMENSION)):
            raise ValueError("only the q=0 Hartree diagonal must be removed")
        expected_singular = tuple((s, s, 0, 0, 0) for s in range(LOCAL_DIMENSION))
        if tuple(self.singular_inventory) != expected_singular:
            raise ValueError("singular inventory must contain only s=v, delta=0")
        if self.route_count != 28 or self.route_policy != "fixed_representative_g0":
            raise ValueError("Local WS FFT backend requires the exact 28-route G=0 inventory")
        if self.q0_hartree_policy != "neutral_background_global_canonical":
            raise ValueError("unsupported q=0 Hartree policy")
        object.__setattr__(self, "q_vectors_Ainv", q_vectors)
        object.__setattr__(self, "epsilon_r", eps)
        object.__setattr__(self, "self_cell_ev", self_cell)
        object.__setattr__(self, "hartree_ev", hartree)
        object.__setattr__(self, "ordered_kernel_fft_ev", kernels)


def build_local_ws_fft_kernel(
    mesh: LocalWSMesh,
    *,
    epsilon_r: float,
    geometry: FoldedTiSe2Geometry | None = None,
    fft_workers: int = 1,
    max_fft_memory_gb: float = 1.0,
) -> LocalWSFFTKernel:
    """Build the 16 direct-Q kernels after geometry and memory gates."""

    if not isinstance(mesh, LocalWSMesh):
        raise TypeError("mesh must be a LocalWSMesh")
    eps = _positive_finite(epsilon_r, name="epsilon_r")
    cap = _positive_finite(max_fft_memory_gb, name="max_fft_memory_gb")
    workers = _validate_fft_workers(fft_workers)
    lattice = mesh.geometry if geometry is None else geometry
    if not isinstance(lattice, FoldedTiSe2Geometry):
        raise TypeError("geometry must be a FoldedTiSe2Geometry")
    q_vectors = _validate_q_vectors(physical_q_vectors(lattice))
    if not np.array_equal(q_vectors, mesh.q_vectors_Ainv):
        raise ValueError("mesh and kernel geometry/q vectors do not match")
    rebound_B = float(np.linalg.norm(q_vectors[1, :2]))
    rebound_Qz = float(q_vectors[1, 2])
    if mesh.B_Ainv != rebound_B or mesh.Qz_Ainv != rebound_Qz:
        raise ValueError("mesh scales do not match geometry-bound Qparallel/Qz")
    routes = physical_momentum_routes()
    if sum(len(block) for row in routes for block in row) != 28:
        raise RuntimeError("unexpected fixed-representative route inventory")

    fft_shape = tuple(next_fast_len(2 * size - 1) for size in mesh.box_shape)
    estimated_peak = estimate_local_ws_fft_peak_bytes(mesh, fft_shape=fft_shape)
    if estimated_peak > int(cap * 1.0e9):
        raise MemoryError(
            "local WS FFT kernel exceeds the conservative peak-memory cap: "
            f"fft_shape={fft_shape}, estimate={estimated_peak} bytes"
        )
    kernels_fft = np.empty(
        (LOCAL_DIMENSION, LOCAL_DIMENSION, *fft_shape), dtype=np.complex128
    )
    delta_m = np.arange(-(mesh.d - 1), mesh.d, dtype=np.int64)
    delta_n = np.arange(-(mesh.d - 1), mesh.d, dtype=np.int64)
    delta_l = np.arange(-(mesh.dz - 1), mesh.dz, dtype=np.int64)
    dm, dn, dl = np.meshgrid(delta_m, delta_n, delta_l, indexing="ij")
    difference_hex = np.maximum.reduce((np.abs(dm), np.abs(dn), np.abs(dm + dn))) <= 2 * mesh.M
    displacement = np.empty((*dm.shape, 3), dtype=np.float64)
    displacement[..., 0] = mesh.B_Ainv * dm / mesh.d
    displacement[..., 1] = mesh.B_Ainv * (dm + 2 * dn) / (sqrt(3.0) * mesh.d)
    displacement[..., 2] = mesh.Qz_Ainv * dl / mesh.dz

    prefactor = 4.0 * np.pi * COULOMB_EV_ANGSTROM / eps
    weight = float(mesh.weights_Ainv3[0])
    cell_radius = float((3.0 * mesh.cell_volume_Ainv3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    self_cell = float(2.0 * COULOMB_EV_ANGSTROM * cell_radius / (np.pi * eps))
    singular_inventory: list[tuple[int, int, int, int, int]] = []
    scale2 = max(
        1.0,
        float(np.max(np.einsum("sd,sd->s", q_vectors, q_vectors, optimize=True))),
        mesh.B_Ainv**2,
        mesh.Qz_Ainv**2,
    )
    singular_tolerance = 64.0 * np.finfo(float).eps * scale2
    packed_index = np.ix_(
        delta_m % fft_shape[0],
        delta_n % fft_shape[1],
        delta_l % fft_shape[2],
    )

    for sector in range(LOCAL_DIMENSION):
        for density_col_sector in range(LOCAL_DIMENSION):
            transfer = displacement + (
                q_vectors[sector] - q_vectors[density_col_sector]
            )
            q2 = np.einsum("...d,...d->...", transfer, transfer, optimize=True)
            singular = difference_hex & (q2 <= singular_tolerance)
            hits = np.argwhere(singular)
            for hit in hits:
                i, j, k = (int(value) for value in hit)
                singular_inventory.append(
                    (
                        sector,
                        density_col_sector,
                        int(delta_m[i]),
                        int(delta_n[j]),
                        int(delta_l[k]),
                    )
                )
            if sector == density_col_sector:
                expected = hits.shape == (1, 3) and np.array_equal(
                    hits[0],
                    np.asarray((mesh.d - 1, mesh.d - 1, mesh.dz - 1)),
                )
            else:
                expected = hits.shape == (0, 3)
            if not expected:
                raise ValueError(
                    "shifted q=0 alias or missing self singularity: the only allowed "
                    "singularity is s=v at zero displacement"
                )

            values = np.zeros(q2.shape, dtype=np.float64)
            regular = difference_hex & ~singular
            values[regular] = weight * prefactor / q2[regular]
            values[singular] = self_cell
            padded = np.zeros(fft_shape, dtype=np.float64)
            padded[packed_index] = values
            kernels_fft[sector, density_col_sector] = fftn(
                padded, workers=workers
            )

    expected_inventory = [(s, s, 0, 0, 0) for s in range(LOCAL_DIMENSION)]
    if singular_inventory != expected_inventory:
        raise RuntimeError("noncanonical Coulomb singular inventory")

    hartree = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION), dtype=np.float64)
    for sector in range(LOCAL_DIMENSION):
        for target_sector in range(LOCAL_DIMENSION):
            if sector == target_sector:
                continue
            q = q_vectors[sector] - q_vectors[target_sector]
            q2 = float(np.dot(q, q))
            if q2 <= singular_tolerance:
                raise ValueError("distinct physical Q representatives alias at q=0")
            hartree[sector, target_sector] = prefactor / q2

    return LocalWSFFTKernel(
        mesh=mesh,
        q_vectors_Ainv=q_vectors,
        epsilon_r=eps,
        self_cell_ev=self_cell,
        hartree_ev=hartree,
        ordered_kernel_fft_ev=kernels_fft,
        fft_shape=fft_shape,
        estimated_peak_bytes=estimated_peak,
        singular_inventory=tuple(singular_inventory),
    )


@dataclass(frozen=True)
class LocalWSNormalState:
 """Geometry-bound normal-state inputs for a future nonlinear WS-FFT SCF.

 This adapter stops before nonlinear iteration. It binds one mesh to
 ``local_full_h0`` on those exact points, the existing finite-temperature
 diagonalizer and the same normalized weights, the physical Q vectors, and
 one FFT kernel. :meth:`interaction_callback` exposes the linear
 reference-subtracted action expected by an SCF driver.
 """

 mesh: LocalWSMesh
 h0_ev: ComplexArray
 reference: LocalFullSpectrum
 q_vectors_Ainv: FloatArray
 kernel: LocalWSFFTKernel
 params: MonneyTiSe2Parameters
 geometry: FoldedTiSe2Geometry
 temperature_K: float

 def __post_init__(self) -> None:
  if self.kernel.mesh is not self.mesh:
   raise ValueError("kernel and adapter must share one mesh object")
  if self.geometry != self.mesh.geometry:
   raise ValueError("adapter and mesh geometry must match")
  q_vectors = _validate_q_vectors(self.q_vectors_Ainv)
  expected_q = _validate_q_vectors(physical_q_vectors(self.geometry))
  if not np.array_equal(q_vectors, expected_q):
   raise ValueError("adapter q vectors must be derived exactly from geometry")
  if not np.array_equal(q_vectors, self.mesh.q_vectors_Ainv):
   raise ValueError("adapter and mesh q vectors must match")
  if not np.array_equal(q_vectors, self.kernel.q_vectors_Ainv):
   raise ValueError("adapter and kernel q vectors must match")
  temperature = _positive_finite(self.temperature_K, name="temperature_K")
  h0 = np.asarray(self.h0_ev, dtype=np.complex128)
  expected_h0 = local_full_h0(self.mesh.points_Ainv, params=self.params)
  if not np.array_equal(h0, expected_h0):
   raise ValueError("h0_ev must be local_full_h0 on the exact mesh points")
  expected_shape = (LOCAL_DIMENSION, LOCAL_DIMENSION, self.mesh.nk)
  if self.reference.projector_stored.shape != expected_shape:
   raise ValueError("reference projector does not match the WS mesh")
  if self.reference.energies_ev.shape != (LOCAL_DIMENSION, self.mesh.nk):
   raise ValueError("reference energies do not match the WS mesh")
  object.__setattr__(self, "h0_ev", h0)
  object.__setattr__(self, "q_vectors_Ainv", q_vectors)
  object.__setattr__(self, "temperature_K", temperature)

 @property
 def reference_projector_stored(self) -> ComplexArray:
  return self.reference.projector_stored

 @property
 def kbt_ev(self) -> float:
  return float(BOLTZMANN_EV_PER_K * self.temperature_K)

 def interaction_callback(
  self, density_delta_stored: np.ndarray, *, fft_workers: int = 1
 ) -> ComplexArray:
  """Apply the WS-FFT interaction to ``D=P-Pref`` on this mesh."""

  return local_ws_fft_interaction_action(
   density_delta_stored, kernel=self.kernel, fft_workers=fft_workers
  )

 def raw_finite_temperature_update(
  self,
  density_delta_stored: np.ndarray,
  *,
  fft_workers: int = 1,
  eigensolver_workers: int = 1,
 ) -> LocalFullSpectrum:
  """Return one unmixed finite-T update; this is not an SCF loop."""

  interaction = self.interaction_callback(
   density_delta_stored, fft_workers=fft_workers
  )
  return diagonalize_local_full_finite_temperature(
   self.h0_ev + interaction,
   kbt_ev=self.kbt_ev,
   k_weights=self.mesh.normalized_weights,
   electrons_per_local_momentum=ELECTRONS_PER_LOCAL_MOMENTUM,
   eigensolver_workers=eigensolver_workers,
  )


def build_local_ws_normal_state(
 *,
 M: int,
 L: int,
 epsilon_r: float,
 temperature_K: float = 65.0,
 params: MonneyTiSe2Parameters | None = None,
 geometry: FoldedTiSe2Geometry | None = None,
 fft_workers: int = 1,
 max_fft_memory_gb: float = 1.0,
) -> LocalWSNormalState:
 """Build the geometry-bound WS normal state without nonlinear SCF."""

 model = MonneyTiSe2Parameters() if params is None else params
 if not isinstance(model, MonneyTiSe2Parameters):
  raise TypeError("params must be MonneyTiSe2Parameters")
 lattice = FoldedTiSe2Geometry(c_angstrom=model.c_angstrom) if geometry is None else geometry
 if not isinstance(lattice, FoldedTiSe2Geometry):
  raise TypeError("geometry must be a FoldedTiSe2Geometry")
 if lattice.c_angstrom != model.c_angstrom:
  raise ValueError("geometry.c_angstrom must equal params.c_angstrom")
 temperature = _positive_finite(temperature_K, name="temperature_K")
 mesh = build_local_ws_mesh(M=M, L=L, geometry=lattice)
 h0 = local_full_h0(mesh.points_Ainv, params=model)
 reference = diagonalize_local_full_finite_temperature(
  h0,
  kbt_ev=BOLTZMANN_EV_PER_K * temperature,
  k_weights=mesh.normalized_weights,
  electrons_per_local_momentum=ELECTRONS_PER_LOCAL_MOMENTUM,
 )
 q_vectors = _validate_q_vectors(physical_q_vectors(lattice))
 kernel = build_local_ws_fft_kernel(
  mesh,
  epsilon_r=epsilon_r,
  geometry=lattice,
  fft_workers=fft_workers,
  max_fft_memory_gb=max_fft_memory_gb,
 )
 return LocalWSNormalState(
  mesh=mesh,
  h0_ev=h0,
  reference=reference,
  q_vectors_Ainv=q_vectors,
  kernel=kernel,
  params=model,
  geometry=lattice,
  temperature_K=temperature,
 )


def _validate_density(density_stored: np.ndarray, *, nk: int) -> ComplexArray:
    density = np.asarray(density_stored, dtype=np.complex128)
    expected = (LOCAL_DIMENSION, LOCAL_DIMENSION, nk)
    if density.shape != expected or not np.all(np.isfinite(density)):
        raise ValueError(f"density_stored must be finite with shape {expected}")
    return density


def _embed_scalar(values: np.ndarray, kernel: LocalWSFFTKernel) -> ComplexArray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != (kernel.mesh.nk,) or not np.all(np.isfinite(array)):
        raise ValueError(f"values must be finite with shape ({kernel.mesh.nk},)")
    embedded = np.zeros(kernel.fft_shape, dtype=np.complex128)
    indices = kernel.mesh.box_indices
    embedded[indices[:, 0], indices[:, 1], indices[:, 2]] = array
    return embedded


def _extract_scalar(values: np.ndarray, mesh: LocalWSMesh) -> ComplexArray:
    indices = mesh.box_indices
    return np.asarray(values[indices[:, 0], indices[:, 1], indices[:, 2]], dtype=np.complex128)


def local_ws_fft_ordered_kernel_action(
    values: np.ndarray,
    *,
    kernel: LocalWSFFTKernel,
    sector: int,
    density_col_sector: int,
    fft_workers: int = 1,
) -> ComplexArray:
    """Apply one ordered ``(s,v)`` weighted Coulomb matrix by linear FFT."""

    if type(sector) is not int or not 0 <= sector < LOCAL_DIMENSION:
        raise ValueError("sector must be an exact integer in [0,4)")
    if type(density_col_sector) is not int or not 0 <= density_col_sector < LOCAL_DIMENSION:
        raise ValueError("density_col_sector must be an exact integer in [0,4)")
    workers = _validate_fft_workers(fft_workers)
    source = _embed_scalar(values, kernel)
    transformed = fftn(source, workers=workers)
    result = ifftn(
        kernel.ordered_kernel_fft_ev[sector, density_col_sector] * transformed,
        workers=workers,
    )
    return _extract_scalar(result, kernel.mesh)


def local_ws_fft_fock_action(
    density_stored: np.ndarray,
    *,
    kernel: LocalWSFFTKernel,
    fft_workers: int = 1,
) -> ComplexArray:
    r"""Apply exchange to an arbitrary complex stored ``4x4`` matrix field.

    FFT worker parallelism is delegated directly to SciPy's FFT backend.  No
    target-block ``ThreadPoolExecutor`` is created, so worker pools are never
    nested around the 16 target blocks.
    """

    workers = _validate_fft_workers(fft_workers)
    density = _validate_density(density_stored, nk=kernel.mesh.nk)
    source = np.zeros((LOCAL_DIMENSION, LOCAL_DIMENSION, *kernel.fft_shape), dtype=np.complex128)
    indices = kernel.mesh.box_indices
    source[:, :, indices[:, 0], indices[:, 1], indices[:, 2]] = density
    source_fft = fftn(source, axes=(-3, -2, -1), workers=workers)

    sigma_fft = np.zeros_like(source_fft)
    routes = physical_momentum_routes()
    for sector in range(LOCAL_DIMENSION):
        for target_sector in range(LOCAL_DIMENSION):
            for density_row_sector, density_col_sector in routes[sector][target_sector]:
                sigma_fft[sector, target_sector] -= (
                    kernel.ordered_kernel_fft_ev[sector, density_col_sector]
                    * source_fft[density_row_sector, density_col_sector]
                )
    sigma_box = ifftn(sigma_fft, axes=(-3, -2, -1), workers=workers)
    return np.asarray(
        sigma_box[:, :, indices[:, 0], indices[:, 1], indices[:, 2]],
        dtype=np.complex128,
    )


def local_ws_fft_hartree_action(
    density_stored: np.ndarray,
    *,
    kernel: LocalWSFFTKernel,
) -> ComplexArray:
    """Apply the 28-route finite-Q Hartree map; remove only literal ``q=0``."""

    density = _validate_density(density_stored, nk=kernel.mesh.nk)
    sigma = np.zeros_like(density)
    routes = physical_momentum_routes()
    for sector in range(LOCAL_DIMENSION):
        for target_sector in range(LOCAL_DIMENSION):
            coefficient = kernel.hartree_ev[sector, target_sector]
            if coefficient == 0.0:
                continue
            value = 0.0j
            for density_row_sector, density_col_sector in routes[sector][target_sector]:
                value += np.einsum(
                    "p,p->",
                    density[density_row_sector, density_col_sector],
                    kernel.mesh.weights_Ainv3,
                    optimize=True,
                )
            sigma[sector, target_sector, :] = coefficient * value
    return sigma


def local_ws_fft_interaction_action(
    density_stored: np.ndarray,
    *,
    kernel: LocalWSFFTKernel,
    fft_workers: int = 1,
) -> ComplexArray:
    """Return finite-Q Hartree plus FFT Fock for arbitrary complex input."""

    return local_ws_fft_hartree_action(
        density_stored, kernel=kernel
    ) + local_ws_fft_fock_action(
        density_stored, kernel=kernel, fft_workers=fft_workers
    )


def local_ws_fft_interaction_energy(
    density_stored: np.ndarray,
    *,
    kernel: LocalWSFFTKernel,
    fft_workers: int = 1,
) -> float:
    """Return ``1/2 <D,L[D]>`` for a Hermitian stored density field."""

    density = _validate_density(density_stored, nk=kernel.mesh.nk)
    scale = max(1.0, float(np.max(np.abs(density))))
    residual = float(np.max(np.abs(density - density.conj().swapaxes(0, 1))))
    if residual > 128.0 * np.finfo(float).eps * scale:
        raise ValueError("interaction energy requires Hermitian density_stored")
    sigma = local_ws_fft_interaction_action(
        density, kernel=kernel, fft_workers=fft_workers
    )
    value = 0.5 * np.einsum(
        "abk,abk,k->", sigma, density, kernel.mesh.weights_Ainv3, optimize=True
    )
    if abs(value.imag) > 2.0e-11 * max(1.0, abs(value.real)):
        raise ValueError(f"interaction energy is not real: {value!r}")
    return float(value.real)


__all__ = [
    "LocalWSFFTKernel",
    "LocalWSMesh",
    "LocalWSNormalState",
    "build_local_ws_fft_kernel",
    "build_local_ws_mesh",
    "build_local_ws_normal_state",
    "estimate_local_ws_fft_peak_bytes",
    "local_ws_fft_fock_action",
    "local_ws_fft_hartree_action",
    "local_ws_fft_interaction_action",
    "local_ws_fft_interaction_energy",
    "local_ws_fft_ordered_kernel_action",
]
