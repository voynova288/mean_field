from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math

import numpy as np
from scipy.linalg import eigh

from ....core.bands import GridBandsResult, PathBandsResult
from ....core.hf import ComponentGroup
from ....core.lattice import KPath, LatticeGrid
from ..params import TBGParameters
from .interaction import TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1
from .path import build_b0_benchmark_kpath, build_fig6_kpath, build_gamma_m_k_gamma_kprime_kpath


TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION = (
    "dimensionless_b0_cartesian:k_code=a_graphene_nm*k_physical_nm_inv"
)
TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION = "cartesian_nm_and_nm^-1"

TBG_ZERO_FIELD_TORUS_MESH_SCHEMA = "mean_field.tbg.zero_field.half_open_torus_mesh"
TBG_ZERO_FIELD_TORUS_MESH_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_BM_GENERATION_SCHEMA = "mean_field.tbg.zero_field.bm_generation"
TBG_ZERO_FIELD_BM_GENERATION_SCHEMA_VERSION = 1
TBG_ZERO_FIELD_BM_SOLVER_SCHEMA = "mean_field.tbg.zero_field.bm_solver"
TBG_ZERO_FIELD_BM_SOLVER_SCHEMA_VERSION = 1


def _canonical_array_sha256(values: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.dtype("<i8")).tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TBGZeroFieldTorusMesh:
    """Exact half-open square torus in the zero-field BM flattening order.

    The mesh owns immutable copies of both arrays.  Construction is deliberately
    strict: callers cannot attach the type to an arbitrary square point cloud.
    """

    mesh_size: int
    g1: complex
    g2: complex
    k_grid_frac: np.ndarray
    kvec: np.ndarray
    schema: str = TBG_ZERO_FIELD_TORUS_MESH_SCHEMA
    schema_version: int = TBG_ZERO_FIELD_TORUS_MESH_SCHEMA_VERSION
    index_order: str = "F"
    fractional_domain: str = "[0,1)x[0,1)"

    def __post_init__(self) -> None:
        if self.schema != TBG_ZERO_FIELD_TORUS_MESH_SCHEMA:
            raise ValueError(f"Unsupported TBG zero-field torus mesh schema {self.schema!r}")
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, (int, np.integer))
            or int(self.schema_version) != TBG_ZERO_FIELD_TORUS_MESH_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported TBG zero-field torus mesh schema version {self.schema_version!r}")
        if self.index_order != "F":
            raise ValueError("TBG zero-field torus mesh index_order must be 'F'")
        if self.fractional_domain != "[0,1)x[0,1)":
            raise ValueError("TBG zero-field torus mesh must use the half-open [0,1)x[0,1) domain")
        if isinstance(self.mesh_size, (bool, np.bool_)) or not isinstance(
            self.mesh_size, (int, np.integer)
        ):
            raise ValueError(f"mesh_size must be a positive integer, got {self.mesh_size!r}")
        size = int(self.mesh_size)
        if size <= 0:
            raise ValueError(f"mesh_size must be a positive integer, got {self.mesh_size!r}")
        g1 = complex(self.g1)
        g2 = complex(self.g2)
        if not all(math.isfinite(value) for value in (g1.real, g1.imag, g2.real, g2.imag)):
            raise ValueError("TBG zero-field torus reciprocal vectors must be finite")
        if g1 == 0.0j or g2 == 0.0j:
            raise ValueError("TBG zero-field torus reciprocal vectors must be nonzero")

        # np.array(..., copy=True) is intentional: np.asarray/ascontiguousarray
        # can alias caller-owned storage and merely mark that storage read-only.
        frac = np.array(self.k_grid_frac, dtype=np.float64, order="C", copy=True)
        kvec = np.array(self.kvec, dtype=np.complex128, order="C", copy=True).reshape(-1)
        if frac.shape != (size * size, 2):
            raise ValueError(f"k_grid_frac must have shape {(size * size, 2)}, got {frac.shape}")
        if kvec.shape != (size * size,):
            raise ValueError(f"kvec must have shape {(size * size,)}, got {kvec.shape}")

        coordinate = np.arange(size, dtype=np.float64) / float(size)
        f1, f2 = np.meshgrid(coordinate, coordinate, indexing="ij")
        expected_frac = np.stack(
            [np.ravel(f1, order="F"), np.ravel(f2, order="F")],
            axis=1,
        )
        if not np.array_equal(frac, expected_frac):
            raise ValueError(
                "TBG zero-field torus coordinates must be exactly the Fortran-ordered "
                "(i/N,j/N), i,j=0,...,N-1 grid"
            )
        expected_kvec = expected_frac[:, 0] * g1 + expected_frac[:, 1] * g2
        if not np.array_equal(kvec, expected_kvec):
            raise ValueError(
                "TBG zero-field torus kvec must exactly equal f1*g1+f2*g2 in the "
                "carried Fortran-ordered fractional grid"
            )

        frac.setflags(write=False)
        kvec.setflags(write=False)
        object.__setattr__(self, "mesh_size", size)
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(self, "g1", g1)
        object.__setattr__(self, "g2", g2)
        object.__setattr__(self, "k_grid_frac", frac)
        object.__setattr__(self, "kvec", kvec)

    @property
    def nk(self) -> int:
        return int(self.mesh_size * self.mesh_size)

    @property
    def fractional_coordinates_sha256(self) -> str:
        return _canonical_array_sha256(self.k_grid_frac, dtype="<f8")

    @property
    def kvec_sha256(self) -> str:
        return _canonical_array_sha256(self.kvec, dtype="<c16")

    def _payload(self) -> dict[str, object]:
        return {
            "fractional_coordinates_sha256": self.fractional_coordinates_sha256,
            "fractional_domain": self.fractional_domain,
            "g1": [float(self.g1.real), float(self.g1.imag)],
            "g2": [float(self.g2.real), float(self.g2.imag)],
            "index_order": self.index_order,
            "kvec_sha256": self.kvec_sha256,
            "mesh_shape": [self.mesh_size, self.mesh_size],
            "point_count": self.nk,
            "schema": self.schema,
            "schema_version": self.schema_version,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload


def tbg_zero_field_bm_generation_fingerprint(
    params: TBGParameters,
    *,
    lg: int,
    periodic_g_grid: bool,
    sigma_rotation: bool,
    calculate_chern_operator: bool = True,
    torus_mesh_fingerprint: str,
) -> str:
    """Hash every independent BM input and the exact torus identity."""

    resolved_lg = int(lg)
    if resolved_lg <= 0:
        raise ValueError(f"BM generation lg must be positive, got {lg!r}")
    mesh_fingerprint = str(torus_mesh_fingerprint).strip().lower()
    if len(mesh_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in mesh_fingerprint
    ):
        raise ValueError("torus_mesh_fingerprint must be a SHA-256 hexadecimal digest")
    scalar_inputs = {
        "alpha": float(params.alpha),
        "beta_g": float(params.beta_g),
        "chemical_potential": float(params.chemical_potential),
        "deformation_potential": float(params.deformation_potential),
        "delta": float(params.delta),
        "dtheta_rad": float(params.dtheta_rad),
        "poisson": float(params.poisson),
        "strain": float(params.strain),
        "strain_angle_rad": float(params.strain_angle_rad),
        "vf": float(params.vf),
        "w0": float(params.w0),
        "w1": float(params.w1),
    }
    if not all(math.isfinite(value) for value in scalar_inputs.values()):
        raise ValueError("All independent TBGParameters BM generation inputs must be finite")
    payload: dict[str, object] = {
        "schema": TBG_ZERO_FIELD_BM_GENERATION_SCHEMA,
        "schema_version": TBG_ZERO_FIELD_BM_GENERATION_SCHEMA_VERSION,
        "params": {
            **scalar_inputs,
            "convention": str(params.convention),
        },
        "params_independent_fingerprint": params.independent_fingerprint,
        "lg": resolved_lg,
        "calculate_chern_operator": bool(calculate_chern_operator),
        "periodic_g_grid": bool(periodic_g_grid),
        "sigma_rotation": bool(sigma_rotation),
        "torus_mesh_fingerprint": mesh_fingerprint,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _validate_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return digest


class _BMSolutionSourceAttestationIssuer:
    """Private per-solve identity token bound to exactly one BMSolution object."""

    __slots__ = ("solution",)

    def __init__(self) -> None:
        self.solution: BMSolution | None = None


@dataclass(frozen=True)
class BMSolutionSourceAttestation:
    """Solver-issued identity for every array entering a typed BM source.

    The private issuer token intentionally prevents ordinary callers from
    relabelling hand-built arrays as solver output.  Live-array validation is
    repeated at every typed bundle boundary, so post-solve mutation is also
    fail-closed.
    """

    solver_entrypoint: str
    params_independent_fingerprint: str
    lg: int
    nlocal: int
    n_eta: int
    n_spin: int
    nb: int
    nk: int
    sigma_rotation: bool
    calculate_chern_operator: bool
    periodic_g_grid: bool
    torus_mesh_fingerprint: str | None
    hamiltonian_shape: tuple[int, ...]
    hamiltonian_sha256: str
    sigma_z_shape: tuple[int, ...]
    sigma_z_sha256: str
    uk_shape: tuple[int, ...]
    uk_sha256: str
    spectrum_shape: tuple[int, ...]
    spectrum_sha256: str
    gvec_shape: tuple[int, ...]
    gvec_sha256: str
    kvec_shape: tuple[int, ...]
    kvec_sha256: str
    _issuer: object = field(repr=False, compare=False)
    solver_schema: str = TBG_ZERO_FIELD_BM_SOLVER_SCHEMA
    solver_schema_version: int = TBG_ZERO_FIELD_BM_SOLVER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self._issuer, _BMSolutionSourceAttestationIssuer):
            raise ValueError(
                "BMSolutionSourceAttestation may be issued only by "
                "solve_bm_model or solve_bm_model_on_torus"
            )
        if self.solver_schema != TBG_ZERO_FIELD_BM_SOLVER_SCHEMA:
            raise ValueError(f"Unsupported BM solver schema {self.solver_schema!r}")
        if int(self.solver_schema_version) != TBG_ZERO_FIELD_BM_SOLVER_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported BM solver schema version {self.solver_schema_version!r}"
            )
        if self.solver_entrypoint not in (
            "solve_bm_model",
            "solve_bm_model_on_torus",
        ):
            raise ValueError(f"Unsupported BM solver entrypoint {self.solver_entrypoint!r}")
        dimensions = {
            name: int(getattr(self, name))
            for name in ("lg", "nlocal", "n_eta", "n_spin", "nb", "nk")
        }
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError("BM source-attestation dimensions must all be positive")
        for name, value in dimensions.items():
            object.__setattr__(self, name, value)
        for name in (
            "sigma_rotation",
            "calculate_chern_operator",
            "periodic_g_grid",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool in BM source attestation")
            object.__setattr__(self, name, bool(getattr(self, name)))
        object.__setattr__(
            self,
            "params_independent_fingerprint",
            _validate_sha256(
                self.params_independent_fingerprint,
                name="params_independent_fingerprint",
            ),
        )
        if self.torus_mesh_fingerprint is not None:
            object.__setattr__(
                self,
                "torus_mesh_fingerprint",
                _validate_sha256(
                    self.torus_mesh_fingerprint,
                    name="torus_mesh_fingerprint",
                ),
            )
        if (
            self.solver_entrypoint == "solve_bm_model_on_torus"
            and self.torus_mesh_fingerprint is None
        ):
            raise ValueError(
                "solve_bm_model_on_torus attestation requires a torus fingerprint"
            )
        if (
            self.solver_entrypoint == "solve_bm_model"
            and self.torus_mesh_fingerprint is not None
        ):
            raise ValueError(
                "solve_bm_model attestation cannot claim a torus fingerprint"
            )
        for name in (
            "hamiltonian_sha256",
            "sigma_z_sha256",
            "uk_sha256",
            "spectrum_sha256",
            "gvec_sha256",
            "kvec_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )
        shape_names = (
            "hamiltonian_shape",
            "sigma_z_shape",
            "uk_shape",
            "spectrum_shape",
            "gvec_shape",
            "kvec_shape",
        )
        for name in shape_names:
            shape = tuple(int(value) for value in getattr(self, name))
            if any(value < 0 for value in shape):
                raise ValueError(f"{name} entries must be non-negative")
            object.__setattr__(self, name, shape)
        dim = self.nlocal * self.lg * self.lg
        nt = self.n_spin * self.n_eta * self.nb
        expected_shapes = {
            "hamiltonian_shape": (dim, dim, self.n_eta, self.nk),
            "sigma_z_shape": (nt, nt, self.nk),
            "uk_shape": (dim, self.nb, self.n_eta, self.nk),
            "spectrum_shape": (self.nb, self.n_eta, self.nk),
            "gvec_shape": (self.lg * self.lg,),
            "kvec_shape": (self.nk,),
        }
        mismatched = [
            name for name, expected in expected_shapes.items()
            if getattr(self, name) != expected
        ]
        if mismatched:
            raise ValueError(
                "BM source-attestation array shapes do not match solver dimensions: "
                f"{sorted(mismatched)}"
            )
        object.__setattr__(self, "solver_schema_version", int(self.solver_schema_version))

    def _payload(self) -> dict[str, object]:
        return {
            "array_sources": {
                "gvec": {"shape": list(self.gvec_shape), "sha256": self.gvec_sha256},
                "hamiltonian": {
                    "shape": list(self.hamiltonian_shape),
                    "sha256": self.hamiltonian_sha256,
                },
                "kvec": {"shape": list(self.kvec_shape), "sha256": self.kvec_sha256},
                "sigma_z": {
                    "shape": list(self.sigma_z_shape),
                    "sha256": self.sigma_z_sha256,
                },
                "spectrum": {
                    "shape": list(self.spectrum_shape),
                    "sha256": self.spectrum_sha256,
                },
                "uk": {"shape": list(self.uk_shape), "sha256": self.uk_sha256},
            },
            "calculate_chern_operator": self.calculate_chern_operator,
            "dimensions": {
                "lg": self.lg,
                "n_eta": self.n_eta,
                "n_spin": self.n_spin,
                "nb": self.nb,
                "nk": self.nk,
                "nlocal": self.nlocal,
            },
            "params_independent_fingerprint": self.params_independent_fingerprint,
            "periodic_g_grid": self.periodic_g_grid,
            "sigma_rotation": self.sigma_rotation,
            "solver_entrypoint": self.solver_entrypoint,
            "solver_schema": self.solver_schema,
            "solver_schema_version": self.solver_schema_version,
            "torus_mesh_fingerprint": self.torus_mesh_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._payload(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        payload = self._payload()
        payload["fingerprint"] = self.fingerprint
        return payload

    def validate_solution(
        self,
        solution: "BMSolution",
        *,
        require_torus: bool = False,
    ) -> None:
        if solution.source_attestation is not self:
            raise ValueError("BMSolution does not carry this source attestation")
        issuer = self._issuer
        if (
            not isinstance(issuer, _BMSolutionSourceAttestationIssuer)
            or issuer.solution is not solution
        ):
            raise ValueError(
                "BMSolution source attestation was not issued for this live "
                "BMSolution object; hand-constructed clones are diagnostic-only"
            )
        if solution.params.independent_fingerprint != self.params_independent_fingerprint:
            raise ValueError(
                "BMSolution source attestation does not match live independent parameters"
            )
        live_mesh_fingerprint = (
            None if solution.torus_mesh is None else solution.torus_mesh.fingerprint
        )
        if require_torus and live_mesh_fingerprint is None:
            raise ValueError(
                "Typed TBG zero-field BM source requires solve_bm_model_on_torus"
            )
        if live_mesh_fingerprint != self.torus_mesh_fingerprint:
            raise ValueError(
                "BMSolution source attestation does not match the live torus mesh"
            )
        scalar_fields = (
            "lg",
            "nlocal",
            "n_eta",
            "n_spin",
            "nb",
            "nk",
            "sigma_rotation",
            "calculate_chern_operator",
            "periodic_g_grid",
        )
        mismatched_scalars = [
            name for name in scalar_fields
            if getattr(solution, name) != getattr(self, name)
        ]
        if mismatched_scalars:
            raise ValueError(
                "BMSolution source attestation does not match live dimensions/flags: "
                f"{sorted(mismatched_scalars)}"
            )
        live_arrays = {
            "hamiltonian": (solution.hamiltonian, "<c16"),
            "sigma_z": (solution.sigma_z, "<c16"),
            "uk": (solution.uk, "<c16"),
            "spectrum": (solution.spectrum, "<f8"),
            "gvec": (solution.gvec, "<c16"),
            "kvec": (solution.lattice_kvec, "<c16"),
        }
        mismatched_arrays: list[str] = []
        for name, (values, dtype) in live_arrays.items():
            if tuple(np.asarray(values).shape) != getattr(self, f"{name}_shape"):
                mismatched_arrays.append(f"{name}.shape")
            if _canonical_array_sha256(values, dtype=dtype) != getattr(
                self,
                f"{name}_sha256",
            ):
                mismatched_arrays.append(f"{name}.sha256")
        if mismatched_arrays:
            raise ValueError(
                "BMSolution source attestation does not match live solver arrays: "
                f"{sorted(mismatched_arrays)}"
            )


def _issue_bm_solution_source_attestation(
    *,
    solver_entrypoint: str,
    params: TBGParameters,
    lg: int,
    nlocal: int,
    n_eta: int,
    n_spin: int,
    nb: int,
    sigma_rotation: bool,
    calculate_chern_operator: bool,
    periodic_g_grid: bool,
    torus_mesh: TBGZeroFieldTorusMesh | None,
    hamiltonian: np.ndarray,
    sigma_z: np.ndarray,
    uk: np.ndarray,
    spectrum: np.ndarray,
    gvec: np.ndarray,
    lattice_kvec: np.ndarray,
) -> BMSolutionSourceAttestation:
    arrays = {
        "hamiltonian": (hamiltonian, "<c16"),
        "sigma_z": (sigma_z, "<c16"),
        "uk": (uk, "<c16"),
        "spectrum": (spectrum, "<f8"),
        "gvec": (gvec, "<c16"),
        "kvec": (lattice_kvec, "<c16"),
    }
    array_fields: dict[str, object] = {}
    for name, (values, dtype) in arrays.items():
        array_fields[f"{name}_shape"] = tuple(np.asarray(values).shape)
        array_fields[f"{name}_sha256"] = _canonical_array_sha256(values, dtype=dtype)
    return BMSolutionSourceAttestation(
        solver_entrypoint=solver_entrypoint,
        params_independent_fingerprint=params.independent_fingerprint,
        lg=int(lg),
        nlocal=int(nlocal),
        n_eta=int(n_eta),
        n_spin=int(n_spin),
        nb=int(nb),
        nk=int(np.asarray(lattice_kvec).size),
        sigma_rotation=bool(sigma_rotation),
        calculate_chern_operator=bool(calculate_chern_operator),
        periodic_g_grid=bool(periodic_g_grid),
        torus_mesh_fingerprint=(
            None if torus_mesh is None else torus_mesh.fingerprint
        ),
        _issuer=_BMSolutionSourceAttestationIssuer(),
        **array_fields,  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class BMSolution:
    params: TBGParameters
    lattice_kvec: np.ndarray
    lg: int
    nlocal: int
    n_eta: int
    n_spin: int
    nb: int
    hamiltonian: np.ndarray
    sigma_z: np.ndarray
    uk: np.ndarray
    spectrum: np.ndarray
    gvec: np.ndarray
    sigma_rotation: bool = True
    calculate_chern_operator: bool = True
    periodic_g_grid: bool = True
    torus_mesh: TBGZeroFieldTorusMesh | None = None
    source_attestation: BMSolutionSourceAttestation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma_rotation", bool(self.sigma_rotation))
        object.__setattr__(
            self,
            "calculate_chern_operator",
            bool(self.calculate_chern_operator),
        )
        object.__setattr__(self, "periodic_g_grid", bool(self.periodic_g_grid))
        if self.source_attestation is not None and not isinstance(
            self.source_attestation,
            BMSolutionSourceAttestation,
        ):
            raise TypeError(
                "source_attestation must be BMSolutionSourceAttestation or None"
            )
        if self.torus_mesh is None:
            return
        if not isinstance(self.torus_mesh, TBGZeroFieldTorusMesh):
            raise TypeError(
                "torus_mesh must be TBGZeroFieldTorusMesh or None, got "
                f"{type(self.torus_mesh).__name__}"
            )
        if self.torus_mesh.g1 != complex(self.params.g1) or self.torus_mesh.g2 != complex(self.params.g2):
            raise ValueError("BMSolution torus_mesh reciprocal vectors do not match params")
        lattice = np.asarray(self.lattice_kvec, dtype=np.complex128).reshape(-1)
        if not np.array_equal(lattice, self.torus_mesh.kvec):
            raise ValueError(
                "BMSolution lattice_kvec must exactly match the carried half-open torus mesh"
            )

    @property
    def generation_fingerprint(self) -> str:
        if self.torus_mesh is None:
            raise ValueError(
                "BM generation fingerprint requires the carried half-open torus mesh"
            )
        return tbg_zero_field_bm_generation_fingerprint(
            self.params,
            lg=self.lg,
            periodic_g_grid=self.periodic_g_grid,
            sigma_rotation=self.sigma_rotation,
            calculate_chern_operator=self.calculate_chern_operator,
            torus_mesh_fingerprint=self.torus_mesh.fingerprint,
        )

    @property
    def nk(self) -> int:
        return int(self.lattice_kvec.size)

    @property
    def nt(self) -> int:
        return self.n_eta * self.n_spin * self.nb

    def flattened_energies(self) -> np.ndarray:
        data = np.zeros((self.nt, self.nk), dtype=float)
        row = 0
        for ib in range(self.nb):
            for ieta in range(self.n_eta):
                for ispin in range(self.n_spin):
                    data[row, :] = self.spectrum[ib, ieta, :]
                    row += 1
        return data

    def validate_source_attestation(self, *, require_torus: bool = False) -> None:
        attestation = self.source_attestation
        if not isinstance(attestation, BMSolutionSourceAttestation):
            raise ValueError(
                "Typed TBG zero-field BM source requires a solver-issued "
                "BMSolutionSourceAttestation; hand-constructed and modified "
                "solutions are diagnostic-only"
            )
        attestation.validate_solution(self, require_torus=require_torus)

    def with_reference_uk(self, uk: np.ndarray) -> "BMSolution":
        """Return an explicitly diagnostic source with a substituted gauge frame."""

        uk = np.asarray(uk, dtype=np.complex128)
        if uk.shape != self.uk.shape:
            raise ValueError(f"Expected uk shape {self.uk.shape}, got {uk.shape}")
        sigma_z = build_sigma_z_from_uk(uk, lg=self.lg, n_spin=self.n_spin)
        return replace(
            self,
            uk=uk.copy(),
            sigma_z=sigma_z,
            source_attestation=None,
        )


def _complex_pair(value: complex) -> list[float]:
    z = complex(value)
    return [float(z.real), float(z.imag)]


def _b0_reciprocal_to_nm_inv(value: complex) -> complex:
    return complex(value) / TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1

def _b0_real_to_nm(value: complex) -> complex:
    return complex(value) * TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1

def _resolve_bm_valley_index(valley: int) -> int:
    value = int(valley)
    if value == 1:
        return 0
    if value == -1:
        return 1
    raise ValueError(f"TBG zero-field BM valley must be +1 or -1, got {valley}")


def _resolve_bm_band_count(n_bands: int | None) -> int:
    if n_bands is None:
        return 2
    count = int(n_bands)
    if count != 2:
        raise NotImplementedError("TBG zero-field BM public adapter currently exposes only the central two bands")
    return count


@dataclass(frozen=True)
class TBGZeroFieldBMModel:
    """Narrow public adapter for zero-field BM single-particle bands."""

    params: TBGParameters
    theta_deg: float
    lg: int = 9
    sigma_rotation: bool = True
    periodic_g_grid: bool = True

    @classmethod
    def from_config(
        cls,
        theta_deg: float,
        *,
        lg: int = 9,
        params: TBGParameters | None = None,
        sigma_rotation: bool = True,
        periodic_g_grid: bool = True,
    ) -> "TBGZeroFieldBMModel":
        resolved_params = params if params is not None else TBGParameters.from_degrees(theta_deg)
        return cls(
            params=resolved_params,
            theta_deg=float(theta_deg),
            lg=int(lg),
            sigma_rotation=bool(sigma_rotation),
            periodic_g_grid=bool(periodic_g_grid),
        )

    @property
    def matrix_dim(self) -> int:
        return int(4 * self.lg * self.lg)

    def lattice_summary(self) -> dict[str, object]:
        return {
            "theta_deg": float(self.theta_deg),
            "lg": int(self.lg),
            "coordinate_convention": TBG_ZERO_FIELD_B0_COORDINATE_CONVENTION,
            "physical_coordinate_convention": TBG_ZERO_FIELD_PHYSICAL_COORDINATE_CONVENTION,
            "graphene_a_nm": TBG_ZERO_FIELD_GRAPHENE_A_NM_SCHEMA_V1,
            "g1_b0_code": _complex_pair(self.params.g1),
            "g2_b0_code": _complex_pair(self.params.g2),
            "a1_b0_code": _complex_pair(self.params.a1),
            "a2_b0_code": _complex_pair(self.params.a2),
            "kt_b0_code": _complex_pair(self.params.kt),
            "kb_b0_code": _complex_pair(self.params.kb_point),
            "g1_nm_inv": _complex_pair(_b0_reciprocal_to_nm_inv(self.params.g1)),
            "g2_nm_inv": _complex_pair(_b0_reciprocal_to_nm_inv(self.params.g2)),
            "a1_nm": _complex_pair(_b0_real_to_nm(self.params.a1)),
            "a2_nm": _complex_pair(_b0_real_to_nm(self.params.a2)),
            "kt_nm_inv": _complex_pair(_b0_reciprocal_to_nm_inv(self.params.kt)),
            "kb_nm_inv": _complex_pair(_b0_reciprocal_to_nm_inv(self.params.kb_point)),
            "model_name": "zero_field_bm",
            "sigma_rotation": bool(self.sigma_rotation),
            "periodic_g_grid": bool(self.periodic_g_grid),
        }

    def component_groups(self) -> tuple[ComponentGroup, ...]:
        return (
            ComponentGroup("layer_bottom", np.asarray([0, 1], dtype=int)),
            ComponentGroup("layer_top", np.asarray([2, 3], dtype=int)),
        )

    def standard_kpath(self, *, points_per_segment: int = 120, path_kind: str = "fig6") -> KPath:
        kind = str(path_kind).strip().lower().replace("-", "_")
        if kind in {"fig6", "m_k_gamma_m"}:
            return build_fig6_kpath(self.params, int(points_per_segment))
        if kind in {"b0_benchmark", "benchmark"}:
            return build_b0_benchmark_kpath(self.params, int(points_per_segment))
        if kind in {"gamma_m_k_gamma_kprime", "gamma_m_k_gamma_kp"}:
            return build_gamma_m_k_gamma_kprime_kpath(self.params, int(points_per_segment))
        raise ValueError(f"Unsupported TBG zero-field BM path_kind={path_kind!r}")

    def _solve(self, kvec: np.ndarray) -> BMSolution:
        return solve_bm_model(
            self.params,
            np.asarray(kvec, dtype=np.complex128).reshape(-1),
            lg=int(self.lg),
            sigma_rotation=bool(self.sigma_rotation),
            calculate_chern_operator=False,
            periodic_g_grid=bool(self.periodic_g_grid),
        )

    def build_hamiltonian(self, k_tilde: complex, *, valley: int = 1) -> np.ndarray:
        solution = self._solve(np.asarray([complex(k_tilde)], dtype=np.complex128))
        return np.asarray(solution.hamiltonian[:, :, _resolve_bm_valley_index(valley), 0], dtype=np.complex128)

    def diagonalize(
        self,
        k_tilde: complex,
        *,
        valley: int = 1,
        n_bands: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        _resolve_bm_band_count(n_bands)
        solution = self._solve(np.asarray([complex(k_tilde)], dtype=np.complex128))
        valley_index = _resolve_bm_valley_index(valley)
        return (
            np.asarray(solution.spectrum[:, valley_index, 0], dtype=float),
            np.asarray(solution.uk[:, :, valley_index, 0], dtype=np.complex128),
        )

    def bands_along_path(
        self,
        path: KPath,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
    ) -> PathBandsResult:
        _resolve_bm_band_count(n_bands)
        solution = self._solve(np.asarray(path.kvec, dtype=np.complex128))
        valley_index = _resolve_bm_valley_index(valley)
        energies = np.asarray(solution.spectrum[:, valley_index, :], dtype=float).T
        eigenvectors = None
        if return_eigenvectors:
            eigenvectors = np.transpose(np.asarray(solution.uk[:, :, valley_index, :], dtype=np.complex128), (2, 0, 1))
        return PathBandsResult(
            path=path,
            energies=energies,
            eigenvectors=eigenvectors,
            band_indices=(self.matrix_dim // 2 - 1, self.matrix_dim // 2),
            metadata={"system": "tbg", "model": "zero_field_bm", "valley": int(valley), "lg": int(self.lg)},
        )

    def bands_on_grid(
        self,
        mesh_size: int,
        *,
        valley: int = 1,
        n_bands: int | None = None,
        return_eigenvectors: bool = False,
        endpoint: bool = False,
        frac_shift: tuple[float, float] = (0.0, 0.0),
    ) -> GridBandsResult:
        _resolve_bm_band_count(n_bands)
        mesh = int(mesh_size)
        if mesh <= 0:
            raise ValueError(f"mesh_size must be positive, got {mesh_size}")
        if endpoint:
            frac_1d = np.linspace(0.0, 1.0, mesh, dtype=float)
        else:
            frac_1d = (np.arange(mesh, dtype=float) + np.asarray(frac_shift, dtype=float)[0]) / float(mesh)
        frac_y = frac_1d if endpoint else (np.arange(mesh, dtype=float) + np.asarray(frac_shift, dtype=float)[1]) / float(mesh)
        f1, f2 = np.meshgrid(frac_1d, frac_y, indexing="ij")
        kvec = f1 * self.params.g1 + f2 * self.params.g2
        solution = self._solve(np.asarray(kvec, dtype=np.complex128).reshape(-1))
        valley_index = _resolve_bm_valley_index(valley)
        energies = np.asarray(solution.spectrum[:, valley_index, :], dtype=float).T.reshape(mesh, mesh, 2)
        eigenvectors = None
        if return_eigenvectors:
            eigenvectors = np.transpose(np.asarray(solution.uk[:, :, valley_index, :], dtype=np.complex128), (2, 0, 1)).reshape(
                mesh,
                mesh,
                self.matrix_dim,
                2,
            )
        return GridBandsResult(
            k_grid_frac=np.stack([f1, f2], axis=-1),
            kvec=np.asarray(kvec, dtype=np.complex128),
            energies=energies,
            eigenvectors=eigenvectors,
            band_indices=(self.matrix_dim // 2 - 1, self.matrix_dim // 2),
            metadata={"system": "tbg", "model": "zero_field_bm", "valley": int(valley), "lg": int(self.lg)},
        )


def dirac(k: complex, zeta: int, theta0: float = 0.0) -> np.ndarray:
    phase = np.exp(-1j * zeta * (np.angle(k) - theta0))
    scale = zeta * abs(k)
    return scale * np.asarray([[0.0, phase], [np.conj(phase), 0.0]], dtype=np.complex128)


def build_b0_uniform_lattice(params: TBGParameters, lk: int) -> LatticeGrid:
    """Build the legacy endpoint-inclusive ``(lk+1)^2`` B0 benchmark grid."""

    frac = np.arange(0, lk + 1, dtype=float) / float(lk)
    kvec = np.ravel(frac[:, None] * params.g1 + frac[None, :] * params.g2, order="F")
    return LatticeGrid(
        k1=frac.copy(),
        k2=frac.copy(),
        kvec=np.asarray(kvec, dtype=np.complex128),
        nk=int(kvec.size),
        lk=int(lk),
        flag_inv=True,
    )


def build_tbg_zero_field_half_open_torus_mesh(
    params: TBGParameters,
    mesh_size: int,
) -> TBGZeroFieldTorusMesh:
    """Build ``k=(i/N)g1+(j/N)g2`` for ``i,j=0,...,N-1``.

    Flattening uses Fortran order, matching the current BM/B0 indexing: the
    first fractional coordinate varies fastest.  Unlike
    :func:`build_b0_uniform_lattice`, this helper never includes either unit
    endpoint.
    """

    if isinstance(mesh_size, (bool, np.bool_)) or not isinstance(
        mesh_size, (int, np.integer)
    ):
        raise ValueError(f"mesh_size must be a positive integer, got {mesh_size!r}")
    size = int(mesh_size)
    if size <= 0:
        raise ValueError(f"mesh_size must be a positive integer, got {mesh_size!r}")
    frac = np.arange(size, dtype=np.float64) / float(size)
    f1, f2 = np.meshgrid(frac, frac, indexing="ij")
    coordinates = np.stack(
        [np.ravel(f1, order="F"), np.ravel(f2, order="F")],
        axis=1,
    )
    kvec = np.ravel(f1 * params.g1 + f2 * params.g2, order="F")
    return TBGZeroFieldTorusMesh(
        mesh_size=size,
        g1=params.g1,
        g2=params.g2,
        k_grid_frac=coordinates,
        kvec=np.asarray(kvec, dtype=np.complex128),
    )


def _generate_gvec(params: TBGParameters, lg: int) -> np.ndarray:
    coords = np.arange(-(lg // 2), lg // 2 + 1, dtype=int)
    return np.ravel(coords[:, None] * params.g1 + coords[None, :] * params.g2, order="F").astype(np.complex128)


def _generate_t12(params: TBGParameters, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(lg * lg).reshape(lg, lg, order="F")
    idx_nn1 = np.roll(idx, shift=(-zeta, zeta), axis=(0, 1))
    idx_nn2 = np.roll(idx, shift=(0, zeta), axis=(0, 1))
    idx_nn12 = np.roll(idx, shift=(-zeta, 0), axis=(0, 1))
    idx_nn1_flat = np.ravel(idx_nn1, order="F")
    idx_nn2_flat = np.ravel(idx_nn2, order="F")
    idx_nn12_flat = np.ravel(idx_nn12, order="F")

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    for ig in range(lg * lg):
        left = 4 * ig
        right1 = 4 * int(idx_nn1_flat[ig])
        right2 = 4 * int(idx_nn2_flat[ig])
        right0 = 4 * int(idx_nn12_flat[ig])

        t12[left + 2 : left + 4, right1 : right1 + 2] = t2
        t12[right1 : right1 + 2, left + 2 : left + 4] = t2
        t12[left + 2 : left + 4, right2 : right2 + 2] = t1
        t12[right2 : right2 + 2, left + 2 : left + 4] = t1
        t12[left + 2 : left + 4, right0 : right0 + 2] = t0
        t12[right0 : right0 + 2, left + 2 : left + 4] = t0

    return t12


def _generate_t12_zero_fill(params: TBGParameters, lg: int, zeta: int) -> np.ndarray:
    dim = 4 * lg * lg
    t12 = np.zeros((dim, dim), dtype=np.complex128)

    if zeta == 1:
        t0, t1, t2 = params.t0, params.t1, params.t2
    elif zeta == -1:
        t0, t1, t2 = params.t0, params.t2, params.t1
    else:
        raise ValueError(f"Unexpected valley label: {zeta}")

    def flat(ix: int, iy: int) -> int:
        return int(ix) + int(lg) * int(iy)

    def in_bounds(ix: int, iy: int) -> bool:
        return 0 <= int(ix) < int(lg) and 0 <= int(iy) < int(lg)

    for iy in range(lg):
        for ix in range(lg):
            here = flat(ix, iy)
            left = 4 * here
            neighbors = (
                (ix + zeta, iy - zeta, t2),
                (ix, iy - zeta, t1),
                (ix + zeta, iy, t0),
            )
            for nx, ny, tunnel in neighbors:
                if not in_bounds(nx, ny):
                    continue
                right = 4 * flat(nx, ny)
                t12[left + 2 : left + 4, right : right + 2] = tunnel
                t12[right : right + 2, left + 2 : left + 4] = tunnel

    return t12


def _construct_diagonal_block(params: TBGParameters, gvec: np.ndarray, lg: int, k: complex, zeta: int, sigma_rotation: bool) -> np.ndarray:
    dim = 4 * lg * lg
    h = np.zeros((dim, dim), dtype=np.complex128)
    sigma0 = np.eye(2, dtype=np.complex128)
    rotation = -params.dtheta_rad / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    div_u = float((params.strain_matrix[0, 0] + params.strain_matrix[1, 1]) / 2.0)

    for ig in range(lg * lg):
        qc = gvec[ig]
        if zeta == 1:
            kb = k - params.kb_point + qc
            kt = k - params.kt + qc
        elif zeta == -1:
            kb = k - params.kt + qc
            kt = k - params.kb_point + qc
        else:
            raise ValueError(f"Unexpected valley label: {zeta}")
        if sigma_rotation:
            k1 = (np.eye(2) + rotation - params.strain_matrix * params.alpha) @ np.asarray([kb.real, kb.imag], dtype=float)
            k2 = (np.eye(2) - rotation + params.strain_matrix * (1.0 - params.alpha)) @ np.asarray([kt.real, kt.imag], dtype=float)
        else:
            k1 = np.asarray([kb.real, kb.imag], dtype=float)
            k2 = np.asarray([kt.real, kt.imag], dtype=float)

        left = 4 * ig
        h[left : left + 2, left : left + 2] = params.vf * dirac(complex(k1[0], k1[1]), zeta, 0.0) - (params.deformation_potential * div_u) * sigma0
        h[left + 2 : left + 4, left + 2 : left + 4] = params.vf * dirac(complex(k2[0], k2[1]), zeta, 0.0) + (params.deformation_potential * div_u) * sigma0

    return h


def _c2t_operator(lg: int) -> np.ndarray:
    s0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    s1 = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    ig = np.eye(lg * lg, dtype=np.complex128)
    return np.kron(ig, np.kron(s0, s1))


def _sigma_z_operator(lg: int) -> np.ndarray:
    s0 = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    ig = np.eye(lg * lg, dtype=np.complex128)
    return np.kron(ig, np.kron(s0, sz))


def build_sigma_z_from_uk(uk: np.ndarray, *, lg: int, n_spin: int = 2) -> np.ndarray:
    uk = np.asarray(uk, dtype=np.complex128)
    if uk.ndim != 4:
        raise ValueError(f"Expected uk to have rank 4, got shape {uk.shape}")
    dim, nb, n_eta, nk = uk.shape
    sigma_z = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)
    sigma_z_local = _sigma_z_operator(lg)
    if sigma_z_local.shape != (dim, dim):
        raise ValueError(f"Expected local sigma_z shape {(dim, dim)}, got {sigma_z_local.shape}")

    for ik in range(nk):
        for ieta, zeta in enumerate((1, -1)):
            mat = uk[:, :, ieta, ik].conj().T @ sigma_z_local @ uk[:, :, ieta, ik] * zeta
            for ispin in range(n_spin):
                base = 2 * ieta + ispin
                rows = slice(base, n_spin * n_eta * nb, n_spin * n_eta)
                cols = slice(base, n_spin * n_eta * nb, n_spin * n_eta)
                sigma_z[rows, cols, ik] = mat
    return sigma_z


def _solve_bm_model(
    params: TBGParameters,
    lattice_kvec: np.ndarray,
    *,
    lg: int,
    sigma_rotation: bool,
    calculate_chern_operator: bool,
    periodic_g_grid: bool,
    torus_mesh: TBGZeroFieldTorusMesh | None,
    solver_entrypoint: str,
) -> BMSolution:
    lattice = np.asarray(lattice_kvec, dtype=np.complex128).reshape(-1)
    n_eta, n_spin, nb, nlocal = 2, 2, 2, 4
    nk = int(lattice.size)
    dim = nlocal * lg * lg
    gvec = _generate_gvec(params, lg)

    hamiltonian = np.zeros((dim, dim, n_eta, nk), dtype=np.complex128)
    spectrum = np.zeros((nb, n_eta, nk), dtype=float)
    uk = np.zeros((dim, nb, n_eta, nk), dtype=np.complex128)
    sigma_z = np.zeros((n_spin * n_eta * nb, n_spin * n_eta * nb, nk), dtype=np.complex128)

    c2t = _c2t_operator(lg)
    sigma_z_local = _sigma_z_operator(lg)
    tunnel_builder = _generate_t12 if periodic_g_grid else _generate_t12_zero_fill
    tunnel = {1: tunnel_builder(params, lg, 1), -1: tunnel_builder(params, lg, -1)}

    start = dim // 2 - 1
    stop = start + nb - 1

    for ieta, zeta in enumerate((1, -1)):
        valley_tunnel = tunnel[zeta]
        for ik, kval in enumerate(lattice):
            h0 = _construct_diagonal_block(params, gvec, lg, complex(kval), zeta, sigma_rotation)
            h = h0 + valley_tunnel - params.chemical_potential * np.eye(dim, dtype=np.complex128)
            hamiltonian[:, :, ieta, ik] = h
            evals, evecs = eigh(h, subset_by_index=[start, stop], driver="evr")
            evecs = evecs + c2t @ np.conj(evecs)
            norms = np.linalg.norm(evecs, axis=0)
            evecs = evecs / norms[None, :]
            spectrum[:, ieta, ik] = evals
            uk[:, :, ieta, ik] = evecs

    if calculate_chern_operator:
        sigma_z[:, :, :] = build_sigma_z_from_uk(uk, lg=lg, n_spin=n_spin)

    attestation = _issue_bm_solution_source_attestation(
        solver_entrypoint=solver_entrypoint,
        params=params,
        lg=lg,
        nlocal=nlocal,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        sigma_rotation=sigma_rotation,
        calculate_chern_operator=calculate_chern_operator,
        periodic_g_grid=periodic_g_grid,
        torus_mesh=torus_mesh,
        hamiltonian=hamiltonian,
        sigma_z=sigma_z,
        uk=uk,
        spectrum=spectrum,
        gvec=gvec,
        lattice_kvec=lattice,
    )
    solution = BMSolution(
        params=params,
        lattice_kvec=lattice,
        lg=lg,
        nlocal=nlocal,
        n_eta=n_eta,
        n_spin=n_spin,
        nb=nb,
        hamiltonian=hamiltonian,
        sigma_z=sigma_z,
        uk=uk,
        spectrum=spectrum,
        gvec=gvec,
        sigma_rotation=bool(sigma_rotation),
        calculate_chern_operator=bool(calculate_chern_operator),
        periodic_g_grid=bool(periodic_g_grid),
        torus_mesh=torus_mesh,
        source_attestation=attestation,
    )
    issuer = attestation._issuer
    if not isinstance(issuer, _BMSolutionSourceAttestationIssuer):
        raise RuntimeError("BM solver created an invalid source-attestation issuer")
    if issuer.solution is not None:
        raise RuntimeError("BM source-attestation issuer was already bound")
    issuer.solution = solution
    return solution


def solve_bm_model(
    params: TBGParameters,
    lattice_kvec: np.ndarray,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    calculate_chern_operator: bool = True,
    periodic_g_grid: bool = True,
) -> BMSolution:
    """Solve BM and issue an attestation for the exact returned arrays."""

    return _solve_bm_model(
        params,
        lattice_kvec,
        lg=lg,
        sigma_rotation=sigma_rotation,
        calculate_chern_operator=calculate_chern_operator,
        periodic_g_grid=periodic_g_grid,
        torus_mesh=None,
        solver_entrypoint="solve_bm_model",
    )


def solve_bm_model_on_torus(
    params: TBGParameters,
    mesh_size: int,
    *,
    lg: int = 9,
    sigma_rotation: bool = True,
    calculate_chern_operator: bool = True,
    periodic_g_grid: bool = True,
) -> BMSolution:
    """Solve BM on the canonical half-open torus and carry its exact identity."""

    mesh = build_tbg_zero_field_half_open_torus_mesh(params, mesh_size)
    return _solve_bm_model(
        params,
        mesh.kvec,
        lg=lg,
        sigma_rotation=sigma_rotation,
        calculate_chern_operator=calculate_chern_operator,
        periodic_g_grid=periodic_g_grid,
        torus_mesh=mesh,
        solver_entrypoint="solve_bm_model_on_torus",
    )
