from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np


TBG_PARAMETERS_INDEPENDENT_SCHEMA = "mean_field.tbg.parameters.independent"
TBG_PARAMETERS_INDEPENDENT_SCHEMA_VERSION = 1


def _complex_matrix(values: list[list[complex]]) -> np.ndarray:
    return np.asarray(values, dtype=np.complex128)


def _owned_read_only_array(values: np.ndarray, *, dtype: str) -> np.ndarray:
    array = np.array(values, dtype=np.dtype(dtype), order="C", copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TBGParameters:
    dtheta_rad: float
    convention: str = "b0"
    vf: float = 2482.0
    chemical_potential: float = 0.0
    w0: float = 77.0
    w1: float = 110.0
    delta: float = 0.0
    strain: float = 0.0
    strain_angle_rad: float = 0.0
    poisson: float = 0.16
    beta_g: float = 3.14
    alpha: float = 0.5
    deformation_potential: float = 0.0

    kb: float = field(init=False)
    g1: complex = field(init=False)
    g2: complex = field(init=False)
    a1: complex = field(init=False)
    a2: complex = field(init=False)
    theta12: float = field(init=False)
    gamma_point: complex = field(init=False)
    kt: complex = field(init=False)
    kb_point: complex = field(init=False)
    omega: complex = field(init=False)
    t0: np.ndarray = field(init=False, repr=False)
    t1: np.ndarray = field(init=False, repr=False)
    t2: np.ndarray = field(init=False, repr=False)
    exx: float = field(init=False)
    eyy: float = field(init=False)
    exy: float = field(init=False)
    gauge_shift: np.ndarray = field(init=False, repr=False)
    rotation_phi: np.ndarray = field(init=False, repr=False)
    strain_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        dtheta = float(self.dtheta_rad)
        phi = float(self.strain_angle_rad)
        strain = float(self.strain)
        poisson = float(self.poisson)
        convention = str(self.convention)
        if convention != "b0":
            raise ValueError(f"Unsupported parameter convention: {convention}")

        kb = 8.0 * math.pi / 3.0 * math.sin(dtheta / 2.0)
        g1 = math.sqrt(3.0) * kb + 0.0j
        g2 = math.sqrt(3.0) * kb * complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))
        a1 = 4.0 * math.pi / (3.0 * kb) * complex(math.cos(math.pi / 6.0), math.sin(math.pi / 6.0))
        a2 = 4.0 * math.pi / (3.0 * kb) * 1.0j
        theta12 = math.pi / 3.0
        gamma_point = 0.0 + 0.0j
        kt = kb / 2.0 * complex(math.cos(math.pi / 2.0), math.sin(math.pi / 2.0))
        kb_point = -kt

        omega = complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))
        t0 = _complex_matrix([[self.w0, self.w1], [self.w1, self.w0]])
        t1 = _complex_matrix([[self.w0, self.w1 * omega.conjugate()], [self.w1 * omega, self.w0]])
        t2 = _complex_matrix([[self.w0, self.w1 * omega], [self.w1 * omega.conjugate(), self.w0]])

        exx = -strain * math.cos(phi) ** 2 + poisson * strain * math.sin(phi) ** 2
        eyy = poisson * strain * math.cos(phi) ** 2 - strain * math.sin(phi) ** 2
        exy = (1.0 + poisson) * strain * math.cos(phi) * math.sin(phi)
        gauge_shift = (math.sqrt(3.0) * self.beta_g / 2.0) * np.asarray([exx - eyy, -2.0 * exy], dtype=float)
        rotation_phi = np.asarray(
            [[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]],
            dtype=float,
        )
        strain_matrix = rotation_phi.T @ np.asarray([[-strain, 0.0], [0.0, poisson * strain]], dtype=float) @ rotation_phi

        twist_generator = dtheta / 2.0 * np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=float)
        g1_cart = 4.0 * math.pi / math.sqrt(3.0) * np.asarray([0.0, -1.0], dtype=float)
        g2_cart = 4.0 * math.pi / math.sqrt(3.0) * np.asarray([math.sqrt(3.0) / 2.0, 0.5], dtype=float)
        tmp1 = (2.0 * twist_generator - strain_matrix) @ g1_cart
        tmp2 = (2.0 * twist_generator - strain_matrix) @ g2_cart
        g1 = complex(tmp1[0], tmp1[1])
        g2 = complex(tmp2[0], tmp2[1])

        area = abs(g1.real * g2.imag - g1.imag * g2.real)
        a1 = 2.0 * math.pi / area * complex(g2.imag, -g2.real)
        a2 = 2.0 * math.pi / area * complex(-g1.imag, g1.real)
        theta12 = math.atan2(a2.imag, a2.real) - math.atan2(a1.imag, a1.real)

        kt = kt + complex(gauge_shift[0], gauge_shift[1]) * self.alpha - complex(strain_matrix[0, 0], strain_matrix[1, 0]) * (4.0 * math.pi / 3.0) * self.alpha
        kb_point = kb_point - complex(gauge_shift[0], gauge_shift[1]) * (1.0 - self.alpha) + complex(strain_matrix[0, 0], strain_matrix[1, 0]) * (4.0 * math.pi / 3.0) * (1.0 - self.alpha)
        kt = kt - g1 / 2.0
        kb_point = kb_point + g1 / 2.0

        object.__setattr__(self, "kb", kb)
        object.__setattr__(self, "g1", complex(g1))
        object.__setattr__(self, "g2", complex(g2))
        object.__setattr__(self, "a1", complex(a1))
        object.__setattr__(self, "a2", complex(a2))
        object.__setattr__(self, "theta12", float(theta12))
        object.__setattr__(self, "gamma_point", complex(gamma_point))
        object.__setattr__(self, "kt", complex(kt))
        object.__setattr__(self, "kb_point", complex(kb_point))
        object.__setattr__(self, "omega", complex(omega))
        object.__setattr__(self, "t0", _owned_read_only_array(t0, dtype="<c16"))
        object.__setattr__(self, "t1", _owned_read_only_array(t1, dtype="<c16"))
        object.__setattr__(self, "t2", _owned_read_only_array(t2, dtype="<c16"))
        object.__setattr__(self, "exx", float(exx))
        object.__setattr__(self, "eyy", float(eyy))
        object.__setattr__(self, "exy", float(exy))
        object.__setattr__(
            self,
            "gauge_shift",
            _owned_read_only_array(gauge_shift, dtype="<f8"),
        )
        object.__setattr__(
            self,
            "rotation_phi",
            _owned_read_only_array(rotation_phi, dtype="<f8"),
        )
        object.__setattr__(
            self,
            "strain_matrix",
            _owned_read_only_array(strain_matrix, dtype="<f8"),
        )

    @classmethod
    def from_degrees(
        cls,
        theta_deg: float,
        *,
        convention: str = "b0",
        vf: float = 2482.0,
        w0: float = 77.0,
        w1: float = 110.0,
        delta: float = 0.0,
        strain: float = 0.0,
        strain_angle_deg: float = 0.0,
        alpha: float = 0.5,
        deformation_potential: float = 0.0,
    ) -> "TBGParameters":
        return cls(
            dtheta_rad=float(theta_deg) * math.pi / 180.0,
            convention=convention,
            vf=vf,
            w0=w0,
            w1=w1,
            delta=delta,
            strain=strain,
            strain_angle_rad=float(strain_angle_deg) * math.pi / 180.0,
            alpha=alpha,
            deformation_potential=deformation_potential,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        """Return every independent BM parameter input in JSON-native form."""

        return {
            "dtheta_rad": float(self.dtheta_rad),
            "convention": str(self.convention),
            "vf": float(self.vf),
            "chemical_potential": float(self.chemical_potential),
            "w0": float(self.w0),
            "w1": float(self.w1),
            "delta": float(self.delta),
            "strain": float(self.strain),
            "strain_angle_rad": float(self.strain_angle_rad),
            "poisson": float(self.poisson),
            "beta_g": float(self.beta_g),
            "alpha": float(self.alpha),
            "deformation_potential": float(self.deformation_potential),
        }

    @property
    def independent_fingerprint(self) -> str:
        """Hash the complete independent input set, excluding derived arrays."""

        payload = {
            "schema": TBG_PARAMETERS_INDEPENDENT_SCHEMA,
            "schema_version": TBG_PARAMETERS_INDEPENDENT_SCHEMA_VERSION,
            "params": self.to_summary_dict(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.to_summary_dict(),
            "kb": self.kb,
            "g1": self.g1,
            "g2": self.g2,
            "a1": self.a1,
            "a2": self.a2,
            "theta12": self.theta12,
            "kt": self.kt,
            "kb_point": self.kb_point,
        }
