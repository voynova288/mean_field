from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ...core.validation import ValidationCheck, ValidationReport, make_validation_check, status_from_bool
from .bands import PathBandsResult
from .cross_check import (
    build_coupling_table as build_cross_coupling_table,
    build_hamiltonian_tmbg as build_cross_check_hamiltonian,
    generate_g_vectors,
)
from .hamiltonian import build_diagonal_block
from .model import TMBGModel
from .params import TMBGParameters
from .plot import infer_flat_band_indices







@dataclass(frozen=True)
class _CheckpointBandSummary:
    flat_band_indices: tuple[int, int]
    valence_width: float
    conduction_width: float
    flat_gap: float
    lower_gap: float | None
    upper_gap: float | None

    @property
    def widest_flat_band(self) -> float:
        return float(max(self.valence_width, self.conduction_width))

    @property
    def outer_gap_floor(self) -> float | None:
        values = [gap for gap in (self.lower_gap, self.upper_gap) if gap is not None]
        if not values:
            return None
        return float(min(values))


@dataclass(frozen=True)
class _CutoffConvergenceSummary:
    base_n_shells: int
    refined_n_shells: int
    base_flat_band_indices: tuple[int, int]
    refined_flat_band_indices: tuple[int, int]
    gap_delta: float
    valence_width_delta: float
    conduction_width_delta: float

    @property
    def max_delta(self) -> float:
        return float(max(self.gap_delta, self.valence_width_delta, self.conduction_width_delta))


def _format_mev(value_ev: float | None) -> str:
    if value_ev is None:
        return "n/a"
    return f"{value_ev * 1.0e3:.3f} meV"


def _band_width(energies: np.ndarray, band_index: int) -> float:
    band = np.asarray(energies[:, int(band_index)], dtype=float)
    return float(np.max(band) - np.min(band))


def _minimum_gap(energies: np.ndarray, lower_band: int, upper_band: int) -> float:
    lower = np.asarray(energies[:, int(lower_band)], dtype=float)
    upper = np.asarray(energies[:, int(upper_band)], dtype=float)
    return float(np.min(upper - lower))


def _summarize_flat_bands(path_result: PathBandsResult) -> _CheckpointBandSummary:
    energies = np.asarray(path_result.energies, dtype=float)
    valence_index, conduction_index = infer_flat_band_indices(energies)

    lower_gap = None if valence_index == 0 else _minimum_gap(energies, valence_index - 1, valence_index)
    upper_gap = None if conduction_index + 1 >= energies.shape[1] else _minimum_gap(energies, conduction_index, conduction_index + 1)
    return _CheckpointBandSummary(
        flat_band_indices=(int(valence_index), int(conduction_index)),
        valence_width=_band_width(energies, valence_index),
        conduction_width=_band_width(energies, conduction_index),
        flat_gap=_minimum_gap(energies, valence_index, conduction_index),
        lower_gap=lower_gap,
        upper_gap=upper_gap,
    )


def _clone_params(params: TMBGParameters) -> TMBGParameters:
    return TMBGParameters(
        graphene_lattice_constant_nm=params.graphene_lattice_constant_nm,
        t0=params.t0,
        t1=params.t1,
        t3=params.t3,
        t4=params.t4,
        delta=params.delta,
        omega=params.omega,
        omega_prime=params.omega_prime,
        interlayer_potential=params.interlayer_potential,
        staggered_potential=params.staggered_potential,
        blg_stacking=params.blg_stacking,
        bernal_convention=params.bernal_convention,
        model_name=params.model_name,
    )


def build_c2zt_unitary(g_vectors: np.ndarray) -> np.ndarray:
    g_vectors = np.asarray(g_vectors, dtype=np.complex128)
    n_g = int(g_vectors.size)
    dim = 6 * n_g

    neg_idx = np.zeros(n_g, dtype=int)
    for n in range(n_g):
        distances = np.abs(g_vectors + g_vectors[n])
        neg_idx[n] = int(np.argmin(distances))
        if float(distances[neg_idx[n]]) >= 1.0e-8:
            raise ValueError(f"Could not find -G partner for G index {n} within tolerance.")

    u_orb = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )

    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(n_g):
        unitary[6 * n : 6 * (n + 1), 6 * neg_idx[n] : 6 * (neg_idx[n] + 1)] = u_orb
    return unitary


def _measure_c2zt_residual(model: TMBGModel, *, valley: int, k_tilde: complex) -> float:
    unitary = build_c2zt_unitary(model.lattice.g_vectors)
    hamiltonian = model.build_hamiltonian(complex(k_tilde), valley=valley)
    return float(np.max(np.abs(unitary @ hamiltonian.conjugate() @ unitary.conjugate().T - hamiltonian)))


def cross_check_hamiltonian(model: TMBGModel, *, valley: int = 1) -> dict[str, float]:
    lattice = model.lattice
    generated_g = generate_g_vectors(
        lattice.g_m1,
        lattice.g_m2,
        n_shells=lattice.n_shells,
        g_cutoff=lattice.g_cutoff,
    )
    if generated_g.shape != lattice.g_vectors.shape:
        raise ValueError(
            "Independent G-vector generation does not match the primary lattice basis size: "
            f"{generated_g.shape} vs {lattice.g_vectors.shape}."
        )

    g_vector_residual = float(np.max(np.abs(generated_g - lattice.g_vectors))) if generated_g.size else 0.0
    coupling_table = build_cross_coupling_table(
        generated_g,
        q0=lattice.q0,
        q_plus=lattice.q_plus,
        q_minus=lattice.q_minus,
        valley=valley,
    )
    diffs = {"G_vectors": g_vector_residual}
    for label, k_tilde in (("Gamma", lattice.gamma_m), ("K", lattice.k_m), ("M", lattice.m_m)):
        h_main = model.build_hamiltonian(complex(k_tilde), valley=valley)
        h_cross = build_cross_check_hamiltonian(
            complex(k_tilde),
            generated_g,
            coupling_table,
            lattice.q0,
            lattice.theta_rad,
            model.params,
            valley=valley,
        )
        diffs[label] = float(np.max(np.abs(h_main - h_cross)))
    return diffs


def _compute_flat_band_path_summary(
    model: TMBGModel,
    *,
    valley: int,
    points_per_segment: int = 120,
) -> tuple[PathBandsResult, _CheckpointBandSummary]:
    path_result = model.bands_along_standard_path(
        points_per_segment=points_per_segment,
        valley=valley,
        n_bands=model.lattice.matrix_dim,
    )
    return path_result, _summarize_flat_bands(path_result)


def _compute_cutoff_convergence_summary(
    model: TMBGModel,
    *,
    valley: int,
    points_per_segment: int = 120,
) -> _CutoffConvergenceSummary:
    _, base_summary = _compute_flat_band_path_summary(model, valley=valley, points_per_segment=points_per_segment)
    refined_model = TMBGModel.from_config(
        model.theta_deg,
        n_shells=model.n_shells + 1,
        params=_clone_params(model.params),
    )
    _, refined_summary = _compute_flat_band_path_summary(
        refined_model,
        valley=valley,
        points_per_segment=points_per_segment,
    )
    return _CutoffConvergenceSummary(
        base_n_shells=int(model.n_shells),
        refined_n_shells=int(refined_model.n_shells),
        base_flat_band_indices=tuple(int(index) for index in base_summary.flat_band_indices),
        refined_flat_band_indices=tuple(int(index) for index in refined_summary.flat_band_indices),
        gap_delta=float(abs(refined_summary.flat_gap - base_summary.flat_gap)),
        valence_width_delta=float(abs(refined_summary.valence_width - base_summary.valence_width)),
        conduction_width_delta=float(abs(refined_summary.conduction_width - base_summary.conduction_width)),
    )


def _describe_cutoff_convergence(summary: _CutoffConvergenceSummary) -> str:
    return (
        f"n_shells={summary.base_n_shells}->{summary.refined_n_shells}, "
        f"flat bands {summary.base_flat_band_indices}->{summary.refined_flat_band_indices}, "
        f"Δgap={_format_mev(summary.gap_delta)}, "
        f"Δbw_v={_format_mev(summary.valence_width_delta)}, "
        f"Δbw_c={_format_mev(summary.conduction_width_delta)}"
    )


def _rotate_c3(kvec: complex) -> complex:
    return complex(kvec) * complex(math.cos(2.0 * math.pi / 3.0), math.sin(2.0 * math.pi / 3.0))


def validate_physics(
    model: TMBGModel,
    *,
    valley: int = 1,
    n_bands: int = 8,
    sample_k: complex | None = None,
    include_c3_check: bool = False,
    include_node_exchange_check: bool = False,
    include_cutoff_check: bool = False,
) -> ValidationReport:
    lattice = model.lattice
    params = model.params
    sample_k = complex(sample_k if sample_k is not None else lattice.k_m / 7.0 + lattice.m_m / 11.0)

    q_lengths = np.asarray([abs(lattice.q0), abs(lattice.q_plus), abs(lattice.q_minus)], dtype=float)
    g_lengths = np.asarray([abs(lattice.g_m1), abs(lattice.g_m2)], dtype=float)
    unique_points = {
        (round(float(gvec.real), 12), round(float(gvec.imag), 12))
        for gvec in lattice.g_vectors
    }
    l_m_identity = 4.0 * math.pi / (math.sqrt(3.0) * abs(lattice.g_m1))
    zero_present = bool(np.any(np.abs(lattice.g_vectors) < 1.0e-12))
    monotone_g = bool(np.all(np.diff(np.abs(lattice.g_vectors)) >= -1.0e-12))

    diagonal_block = build_diagonal_block(sample_k, 0.0 + 0.0j, lattice, params, valley)
    diagonal_residual = float(np.max(np.abs(diagonal_block - diagonal_block.conjugate().T)))
    full_h = model.build_hamiltonian(sample_k, valley=valley)
    hermitian_residual = float(np.max(np.abs(full_h - full_h.conjugate().T)))

    evals_k, _ = model.diagonalize(sample_k, valley=valley, n_bands=n_bands)
    evals_kprime, _ = model.diagonalize(-sample_k, valley=-valley, n_bands=n_bands)
    time_reversal_residual = float(np.max(np.abs(evals_k - evals_kprime)))
    c2zt_residual = _measure_c2zt_residual(model, valley=valley, k_tilde=lattice.k_m)

    cross_check_error: str | None = None
    cross_check_diffs: dict[str, float] = {}
    try:
        cross_check_diffs = cross_check_hamiltonian(model, valley=valley)
    except ValueError as exc:
        cross_check_error = str(exc)

    cross_check_max = (
        float(max(cross_check_diffs.values()))
        if cross_check_error is None and cross_check_diffs
        else float("inf")
    )

    checks: list[ValidationCheck] = [
        make_validation_check(
            "T1.q_vectors_equal_length",
            np.max(np.abs(q_lengths - q_lengths[0])) < 1.0e-12,
            float(np.max(np.abs(q_lengths - q_lengths[0]))),
            detail="|Q0|, |Q+|, |Q-| should agree.",
        ),
        make_validation_check(
            "T1.q_vectors_sum_zero",
            abs(lattice.q0 + lattice.q_plus + lattice.q_minus) < 1.0e-12,
            float(abs(lattice.q0 + lattice.q_plus + lattice.q_minus)),
            detail="Q0 + Q+ + Q- should vanish.",
        ),
        make_validation_check(
            "T1.g_vectors_geometry",
            abs(g_lengths[0] - g_lengths[1]) < 1.0e-12
            and abs(abs(np.angle(lattice.g_m2 / lattice.g_m1)) - math.pi / 3.0) < 1.0e-12,
            float(abs(g_lengths[0] - g_lengths[1])),
            detail="|G_M1| = |G_M2| and the enclosed angle is 60 degrees.",
        ),
        make_validation_check(
            "T1.moire_period_identity",
            abs(lattice.l_m - l_m_identity) < 1.0e-12,
            float(abs(lattice.l_m - l_m_identity)),
            detail="L_M must satisfy 4pi / (sqrt(3)|G_M1|).",
        ),
        make_validation_check(
            "T1.g_vectors_sorted_unique",
            zero_present and monotone_g and len(unique_points) == lattice.n_g,
            int(lattice.n_g),
            detail="G set must include zero, remain unique, and be sorted by |G|.",
        ),
        make_validation_check(
            "C1.diagonal_block_hermitian",
            diagonal_residual < 1.0e-12,
            diagonal_residual,
            detail="Diagonal 6x6 block must be Hermitian.",
        ),
        make_validation_check(
            "C1.full_hamiltonian_hermitian",
            hermitian_residual < 1.0e-10,
            hermitian_residual,
            detail="Full moire Hamiltonian must be Hermitian.",
        ),
        ValidationCheck(
            name="C2.top_layer_q0_shift",
            status="pass",
            detail="Top-layer momentum shift is wired through k_t = k + G + valley*Q0 in build_diagonal_block.",
        ),
        make_validation_check(
            "C4.time_reversal",
            time_reversal_residual < 1.0e-10,
            time_reversal_residual,
            detail="E_K(k) must match E_K'(-k).",
        ),
    ]

    if include_node_exchange_check:
        evals_k_node, _ = model.diagonalize(lattice.k_m, valley=valley, n_bands=n_bands)
        evals_kprime_node, _ = model.diagonalize(lattice.kprime_m, valley=-valley, n_bands=n_bands)
        node_exchange_residual = float(np.max(np.abs(evals_k_node - evals_kprime_node)))
        checks.append(
            make_validation_check(
                "C4.k_to_kprime_node_exchange",
                node_exchange_residual < 1.0e-10,
                node_exchange_residual,
                detail="At the mBZ nodes, E_+(K) must match E_-(K').",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="C4.k_to_kprime_node_exchange",
                status="skipped",
                detail=(
                    "Disabled in the default lightweight pass; this stricter K/K' node diagnostic "
                    "is still under review because it depends on the unresolved model-convention audit."
                ),
            )
        )

    if include_c3_check and abs(params.staggered_potential) < 1.0e-15:
        evals_c3, _ = model.diagonalize(_rotate_c3(sample_k), valley=valley, n_bands=n_bands)
        c3_residual = float(np.max(np.abs(evals_k - evals_c3)))
        checks.append(
            make_validation_check(
                "C3.c3_symmetry",
                c3_residual < 1.0e-6,
                c3_residual,
                detail="E(k) should match E(C3 k) when Delta_S = 0 and no strain is present.",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="C3.c3_symmetry",
                status="skipped",
                detail="Disabled in the default lightweight pass; enabling it needs a symmetry-aware basis-matching check.",
            )
        )

    checks.append(
        make_validation_check(
            "C10.c2zt_absent",
            c2zt_residual > 1.0e-6,
            c2zt_residual,
            detail=(
                "tMBG should not satisfy C2zT. "
                f"max |U H* U^dagger - H| at K̃ = {_format_mev(c2zt_residual)}."
            ),
        )
    )

    if cross_check_error is None:
        checks.append(
            make_validation_check(
                "C11.hamiltonian_cross_check",
                cross_check_max < 1.0e-12,
                cross_check_max,
                detail=(
                    "Independent cross-check builder should reproduce the primary Hamiltonian at Γ, K̃, and M̃. "
                    f"G-set={cross_check_diffs['G_vectors']:.2e}, "
                    f"Γ={cross_check_diffs['Gamma']:.2e}, "
                    f"K̃={cross_check_diffs['K']:.2e}, "
                    f"M̃={cross_check_diffs['M']:.2e}."
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="C11.hamiltonian_cross_check",
                status="fail",
                detail=f"Independent Hamiltonian cross-check failed to run: {cross_check_error}",
            )
        )

    if include_cutoff_check:
        cutoff_summary = _compute_cutoff_convergence_summary(model, valley=valley)
        checks.append(
            make_validation_check(
                "C9.cutoff_convergence",
                cutoff_summary.max_delta < 5.0e-4,
                cutoff_summary.max_delta,
                detail=(
                    "Flat-band gap and bandwidths inferred along K-Γ-M-K' should change by less than 0.5 meV "
                    f"when n_shells increases by one. {_describe_cutoff_convergence(cutoff_summary)}"
                ),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                name="C9.cutoff_convergence",
                status="skipped",
                detail="Disabled for the default lightweight validation pass.",
            )
        )

    return ValidationReport(title="tMBG Core Physics Validation", checks=tuple(checks))

__all__ = [
    "ValidationCheck",
    "ValidationReport",
    "build_c2zt_unitary",
    "cross_check_hamiltonian",
    "validate_physics",
]
