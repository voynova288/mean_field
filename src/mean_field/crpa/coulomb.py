from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..systems.tbg.params import TBGParameters
from ..systems.tbg.zero_field.hf import coulomb_unit, screened_coulomb


@dataclass(frozen=True)
class CRPACoulombParams:
    epsilon_bn: float = 4.0
    ds_angstrom: float = 400.0
    graphene_lattice_angstrom: float = 2.46
    finite_zero_limit: bool = True
    zero_cutoff: float = 1.0e-6

    @property
    def screening_lm(self) -> float:
        # Existing B0 HF uses tanh(|q| * 2 * lm), with q in graphene-lattice units.
        return float(self.ds_angstrom / self.graphene_lattice_angstrom / 2.0)

    @property
    def ds_nm(self) -> float:
        return float(self.ds_angstrom / 10.0)


def coulomb_potential_mev(
    q: complex | np.ndarray,
    params: TBGParameters,
    coulomb: CRPACoulombParams,
) -> float | np.ndarray:
    """Return the Zhang double-gate Coulomb interaction in meV."""

    q_array = np.asarray(q, dtype=np.complex128)
    scalar = q_array.ndim == 0
    values = np.asarray(
        [
            screened_coulomb(
                complex(item),
                coulomb.screening_lm,
                relative_permittivity=float(coulomb.epsilon_bn),
                zero_cutoff=float(coulomb.zero_cutoff),
                finite_zero_limit=bool(coulomb.finite_zero_limit),
            )
            for item in q_array.reshape(-1)
        ],
        dtype=float,
    ).reshape(q_array.shape)
    out = coulomb_unit(params) * np.asarray(values, dtype=float)
    if scalar:
        return float(out.reshape(()))
    return out


def coulomb_potential_table_mev(
    q_tilde: complex,
    q_vectors: np.ndarray,
    params: TBGParameters,
    coulomb: CRPACoulombParams,
) -> np.ndarray:
    return np.asarray(coulomb_potential_mev(q_tilde + q_vectors, params, coulomb), dtype=float)
