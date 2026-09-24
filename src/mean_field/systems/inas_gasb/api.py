"""Stable API for retained InAs/GaSb modeling, bands, and mean field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .model import InAsGaSbBHZModel
from .kane4_bundle import Kane4Bundle
from .xue2018 import xue2018_standard_parameters
from .xue2018_hf import Xue2018HFResult, build_xue2018_hf_state, run_xue2018_hf
from .zeng2022 import Zeng2022Parameters


@dataclass(frozen=True)
class Kane4BandResult:
    """Exact noninteracting bands on a saved Kane4 source grid."""

    k_cart_nm_inv: np.ndarray
    energies_mev: np.ndarray
    eigenvectors: np.ndarray | None
    bundle_fingerprint: str


@dataclass(frozen=True)
class InAsGaSbHFConfig:
    """Explicit Xue-2018 Q=0 HF configuration.

    The cutoff and regulator policies are required API fields rather than
    hidden defaults inherited from historical scripts.
    """

    kmax_ab_inv: float
    points_per_axis: int
    init_mode: Literal["normal", "trsb_nematic", "trs_nematic", "qah", "random"]
    precision: float = 1.0e-8
    self_cell_policy: Literal["integrated", "omitted_diagnostic"] = "integrated"
    mesh_policy: Literal[
        "midpoint_cells", "inclusive_nodes_uniform_weight_diagnostic"
    ] = "midpoint_cells"
    q0_kernel_backend: Literal["dense", "toeplitz_fft"] = "dense"
    seed: int = 0
    seed_amplitude_ry: float = 0.2
    max_iter: int = 300
    max_oda_lambda: float | None = 0.5
    oda_stall_threshold: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.kmax_ab_inv) or self.kmax_ab_inv <= 0.0:
            raise ValueError("kmax_ab_inv must be finite and positive")
        if isinstance(self.points_per_axis, bool) or int(self.points_per_axis) != self.points_per_axis:
            raise ValueError("points_per_axis must be an odd positive integer")
        if int(self.points_per_axis) <= 0 or int(self.points_per_axis) % 2 != 1:
            raise ValueError("points_per_axis must be an odd positive integer")
        if not np.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("precision must be finite and positive")
        if isinstance(self.max_iter, bool) or int(self.max_iter) != self.max_iter or self.max_iter < 0:
            raise ValueError("max_iter must be a nonnegative integer")


def build_inas_gasb_model(
    *,
    variant: Literal["xue2018_q0_bhz", "zeng2022_folded_bhz"],
    eg_ry: float | None = None,
    hybridization_ab_ry: float | None = None,
    q_ab_inv: float | None = None,
    d_over_ab: float | None = None,
    mass_e_over_reduced: float | None = None,
    mass_h_over_reduced: float | None = None,
    slab_indices: tuple[int, ...] = (0,),
    path_extent_ab_inv: float | None = None,
    params: Zeng2022Parameters | None = None,
) -> InAsGaSbBHZModel:
    """Build a retained analytic model without inventing paper parameters."""

    scalar_parameters = (
        eg_ry,
        hybridization_ab_ry,
        q_ab_inv,
        d_over_ab,
        mass_e_over_reduced,
        mass_h_over_reduced,
    )
    if params is not None and any(value is not None for value in scalar_parameters):
        raise TypeError("pass params or scalar model parameters, not both")
    if variant == "xue2018_q0_bhz":
        if params is not None:
            expected = xue2018_standard_parameters(
                eg_ry=params.eg_ry,
                hybridization_ab_ry=params.hybridization_ab_ry,
            )
            if params != expected:
                raise ValueError("xue2018_q0_bhz requires the source Q=0 symmetric parameter policy")
            model_params = params
        else:
            if eg_ry is None or hybridization_ab_ry is None:
                raise TypeError("xue2018_q0_bhz requires eg_ry and hybridization_ab_ry")
            if q_ab_inv is not None:
                raise TypeError("xue2018_q0_bhz fixes q_ab_inv=0; do not override it")
            if any(
                value is not None
                for value in (
                    d_over_ab,
                    mass_e_over_reduced,
                    mass_h_over_reduced,
                )
            ):
                raise TypeError(
                    "xue2018_q0_bhz fixes d_over_ab=0.3 and symmetric mass ratios 2/2; "
                    "do not override the source policy"
                )
            model_params = xue2018_standard_parameters(
                eg_ry=float(eg_ry),
                hybridization_ab_ry=float(hybridization_ab_ry),
            )
        if tuple(slab_indices) != (0,):
            raise ValueError("xue2018_q0_bhz requires slab_indices=(0,)")
        return InAsGaSbBHZModel(
            params=model_params,
            slab_indices=(0,),
            path_extent_ab_inv=path_extent_ab_inv,
            variant=variant,
        )
    if variant != "zeng2022_folded_bhz":
        raise ValueError(f"unsupported InAs/GaSb model variant {variant!r}")
    if params is None:
        if eg_ry is None or hybridization_ab_ry is None or q_ab_inv is None:
            raise TypeError(
                "zeng2022_folded_bhz requires params or explicit eg_ry, "
                "hybridization_ab_ry, and q_ab_inv"
            )
        params = Zeng2022Parameters(
            eg_ry=float(eg_ry),
            hybridization_ab_ry=float(hybridization_ab_ry),
            q_ab_inv=float(q_ab_inv),
            d_over_ab=0.3 if d_over_ab is None else float(d_over_ab),
            mass_e_over_reduced=(
                2.0 if mass_e_over_reduced is None else float(mass_e_over_reduced)
            ),
            mass_h_over_reduced=(
                2.0 if mass_h_over_reduced is None else float(mass_h_over_reduced)
            ),
        )
    return InAsGaSbBHZModel(
        params=params,
        slab_indices=tuple(int(value) for value in slab_indices),
        path_extent_ab_inv=path_extent_ab_inv,
        variant=variant,
    )


def compute_kane4_bands(
    bundle: Kane4Bundle,
    *,
    return_eigenvectors: bool = False,
) -> Kane4BandResult:
    """Diagonalize the exact saved Kane4 grid without off-grid interpolation."""

    if not isinstance(bundle, Kane4Bundle):
        raise TypeError("bundle must be Kane4Bundle")
    bundle.validate()
    nk = bundle.nk
    energies = np.empty((nk, bundle.basis.dimension), dtype=float)
    vectors = (
        np.empty((nk, bundle.basis.dimension, bundle.basis.dimension), dtype=np.complex128)
        if return_eigenvectors
        else None
    )
    for ik in range(nk):
        values, frame = np.linalg.eigh(np.asarray(bundle.h0_mev[:, :, ik], dtype=np.complex128))
        energies[ik] = values
        if vectors is not None:
            vectors[ik] = frame
    return Kane4BandResult(
        k_cart_nm_inv=np.asarray(bundle.k_cart_nm_inv, dtype=float).copy(),
        energies_mev=energies,
        eigenvectors=vectors,
        bundle_fingerprint=bundle.fingerprint(),
    )


def run_inas_gasb_hf(
    model: InAsGaSbBHZModel,
    config: InAsGaSbHFConfig,
    *,
    initial_density_delta: np.ndarray | None = None,
) -> Xue2018HFResult:
    """Run the retained Xue-2018 Q=0 mean-field solver through a typed API."""

    if not isinstance(model, InAsGaSbBHZModel):
        raise TypeError("model must be InAsGaSbBHZModel")
    if model.variant != "xue2018_q0_bhz":
        raise NotImplementedError("the retained HF API currently supports xue2018_q0_bhz only")
    expected = xue2018_standard_parameters(
        eg_ry=model.params.eg_ry,
        hybridization_ab_ry=model.params.hybridization_ab_ry,
    )
    if model.params != expected or model.slab_indices != (0,):
        raise ValueError("model does not satisfy the Xue-2018 Q=0 HF parameter contract")
    state = build_xue2018_hf_state(
        eg_ry=model.params.eg_ry,
        hybridization_ab_ry=model.params.hybridization_ab_ry,
        kmax_ab_inv=config.kmax_ab_inv,
        points_per_axis=config.points_per_axis,
        precision=config.precision,
        self_cell_policy=config.self_cell_policy,
        mesh_policy=config.mesh_policy,
        q0_kernel_backend=config.q0_kernel_backend,
    )
    return run_xue2018_hf(
        state,
        init_mode=config.init_mode,
        seed=config.seed,
        seed_amplitude_ry=config.seed_amplitude_ry,
        max_iter=config.max_iter,
        max_oda_lambda=config.max_oda_lambda,
        oda_stall_threshold=config.oda_stall_threshold,
        initial_density_delta=initial_density_delta,
    )


__all__ = [
    "InAsGaSbBHZModel",
    "InAsGaSbHFConfig",
    "Kane4BandResult",
    "build_inas_gasb_model",
    "compute_kane4_bands",
    "run_inas_gasb_hf",
]
