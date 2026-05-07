from __future__ import annotations

from dataclasses import dataclass, field
import math


GRAPHENE_LATTICE_CONSTANT_NM = 0.246
DEFAULT_HBN_MISMATCH = 0.01673
DEFAULT_FERMI_VELOCITY_MEV_NM = 542.1
DEFAULT_REMOTE_VELOCITY_MEV_NM = 34.0
DEFAULT_T1_MEV = 355.16
DEFAULT_T2_MEV = -7.0
DEFAULT_ISP_MEV = 16.65
DEFAULT_LAYER_SPACING_NM = 0.333
VALID_LAYER_COUNTS = (3, 4, 5, 6, 7)
VALID_STACKING_CONFIGS = (0, 1)
VALID_VALLEYS = (-1, 1)


MOIRE_PARAMETER_TABLE: dict[tuple[int, int], tuple[float, float, float]] = {
    (3, 1): (0.0, 5.54, 16.55),
    (4, 1): (1.44, 6.91, 16.55),
    (5, 1): (1.50, 7.37, 16.55),
    (6, 1): (1.56, 7.80, 16.55),
    (7, 1): (1.47, 7.93, 16.55),
    (3, 0): (6.13, 5.95, -136.55),
    (4, 0): (7.16, 6.65, -136.55),
    (5, 0): (7.19, 7.49, -136.55),
    (6, 0): (7.12, 7.16, -136.55),
    (7, 0): (7.00, 7.37, -136.55),
}


def table_ii_moire_parameters(layer_count: int, xi: int) -> tuple[float, float, float]:
    key = (int(layer_count), int(xi))
    if key not in MOIRE_PARAMETER_TABLE:
        raise ValueError(
            f"Expected layer_count in {VALID_LAYER_COUNTS} and xi in {VALID_STACKING_CONFIGS}, "
            f"got layer_count={layer_count}, xi={xi}"
        )
    return MOIRE_PARAMETER_TABLE[key]


@dataclass(frozen=True)
class RLGhBNParams:
    layer_count: int = 5
    xi: int = 1
    displacement_field_mev: float = 0.0

    graphene_lattice_constant_nm: float = GRAPHENE_LATTICE_CONSTANT_NM
    hbn_lattice_mismatch: float = DEFAULT_HBN_MISMATCH
    fermi_velocity_mev_nm: float = DEFAULT_FERMI_VELOCITY_MEV_NM
    v3_mev_nm: float = DEFAULT_REMOTE_VELOCITY_MEV_NM
    v4_mev_nm: float = DEFAULT_REMOTE_VELOCITY_MEV_NM
    t1_mev: float = DEFAULT_T1_MEV
    t2_mev: float = DEFAULT_T2_MEV
    isp_mev: float = DEFAULT_ISP_MEV
    layer_spacing_nm: float = DEFAULT_LAYER_SPACING_NM

    moire_v0_mev: float | None = None
    moire_v1_mev: float | None = None
    moire_phase_deg: float | None = None

    moire_phase_rad: float = field(init=False)

    def __post_init__(self) -> None:
        layer_count = int(self.layer_count)
        xi = int(self.xi)
        if layer_count not in VALID_LAYER_COUNTS:
            raise ValueError(f"Expected layer_count in {VALID_LAYER_COUNTS}, got {self.layer_count}")
        if xi not in VALID_STACKING_CONFIGS:
            raise ValueError(f"Expected xi in {VALID_STACKING_CONFIGS}, got {self.xi}")

        table_v0, table_v1, table_phase_deg = table_ii_moire_parameters(layer_count, xi)
        v0 = table_v0 if self.moire_v0_mev is None else float(self.moire_v0_mev)
        v1 = table_v1 if self.moire_v1_mev is None else float(self.moire_v1_mev)
        phase_deg = table_phase_deg if self.moire_phase_deg is None else float(self.moire_phase_deg)

        object.__setattr__(self, "layer_count", layer_count)
        object.__setattr__(self, "xi", xi)
        object.__setattr__(self, "moire_v0_mev", float(v0))
        object.__setattr__(self, "moire_v1_mev", float(v1))
        object.__setattr__(self, "moire_phase_deg", float(phase_deg))
        object.__setattr__(self, "moire_phase_rad", float(phase_deg) * math.pi / 180.0)

    @classmethod
    def from_table(
        cls,
        *,
        layer_count: int = 5,
        xi: int = 1,
        displacement_field_mev: float = 0.0,
        isp_mev: float = DEFAULT_ISP_MEV,
    ) -> "RLGhBNParams":
        return cls(
            layer_count=layer_count,
            xi=xi,
            displacement_field_mev=displacement_field_mev,
            isp_mev=isp_mev,
        )

    @classmethod
    def without_moire(
        cls,
        *,
        layer_count: int = 5,
        xi: int = 1,
        displacement_field_mev: float = 0.0,
        isp_mev: float = DEFAULT_ISP_MEV,
    ) -> "RLGhBNParams":
        return cls(
            layer_count=layer_count,
            xi=xi,
            displacement_field_mev=displacement_field_mev,
            isp_mev=isp_mev,
            moire_v0_mev=0.0,
            moire_v1_mev=0.0,
        )

    @property
    def L(self) -> int:
        return int(self.layer_count)

    @property
    def internal_dim(self) -> int:
        return int(2 * self.layer_count)

    @property
    def valence_band_count_per_spin_valley_per_g_grid(self) -> int:
        return int(self.layer_count)

    @property
    def V0(self) -> float:
        return float(self.moire_v0_mev)

    @property
    def V1(self) -> float:
        return float(self.moire_v1_mev)

    @property
    def psi(self) -> float:
        return float(self.moire_phase_rad)

    @property
    def vF(self) -> float:
        return float(self.fermi_velocity_mev_nm)

    def to_summary_dict(self) -> dict[str, float | int]:
        return {
            "layer_count": int(self.layer_count),
            "xi": int(self.xi),
            "displacement_field_mev": float(self.displacement_field_mev),
            "graphene_lattice_constant_nm": float(self.graphene_lattice_constant_nm),
            "hbn_lattice_mismatch": float(self.hbn_lattice_mismatch),
            "fermi_velocity_mev_nm": float(self.fermi_velocity_mev_nm),
            "v3_mev_nm": float(self.v3_mev_nm),
            "v4_mev_nm": float(self.v4_mev_nm),
            "t1_mev": float(self.t1_mev),
            "t2_mev": float(self.t2_mev),
            "isp_mev": float(self.isp_mev),
            "layer_spacing_nm": float(self.layer_spacing_nm),
            "moire_v0_mev": float(self.moire_v0_mev),
            "moire_v1_mev": float(self.moire_v1_mev),
            "moire_phase_deg": float(self.moire_phase_deg),
        }


__all__ = [
    "DEFAULT_FERMI_VELOCITY_MEV_NM",
    "DEFAULT_HBN_MISMATCH",
    "DEFAULT_ISP_MEV",
    "DEFAULT_LAYER_SPACING_NM",
    "DEFAULT_REMOTE_VELOCITY_MEV_NM",
    "DEFAULT_T1_MEV",
    "DEFAULT_T2_MEV",
    "GRAPHENE_LATTICE_CONSTANT_NM",
    "MOIRE_PARAMETER_TABLE",
    "RLGhBNParams",
    "VALID_LAYER_COUNTS",
    "VALID_STACKING_CONFIGS",
    "VALID_VALLEYS",
    "table_ii_moire_parameters",
]
