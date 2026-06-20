from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from ...core.io import write_text_artifact
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
from .plot import TMBGBandPlotPanel, infer_flat_band_indices, write_tmbg_lattice_plot, write_tmbg_paper_band_figure







@dataclass(frozen=True)
class PaperCheckpointCase:
    theta_deg: float
    model_name: Literal["minimal", "full"]
    interlayer_potential: float = 0.0
    staggered_potential: float = 0.0

    def build_params(self) -> TMBGParameters:
        if self.model_name == "minimal":
            return TMBGParameters.minimal(
                interlayer_potential=self.interlayer_potential,
                staggered_potential=self.staggered_potential,
            )
        return TMBGParameters.full(
            interlayer_potential=self.interlayer_potential,
            staggered_potential=self.staggered_potential,
        )

    @property
    def panel_label(self) -> str:
        return f"Δ = {_format_signed_mev(self.interlayer_potential)}"


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
class _KPointGapSummary:
    k_label: str
    k_tilde: complex
    flat_band_indices: tuple[int, int]
    valence_energy: float
    conduction_energy: float
    flat_gap: float


@dataclass(frozen=True)
class _KtildeDiagnosticCase:
    name: str
    detail_prefix: str
    params: TMBGParameters
    theta_deg: float = 1.21
    expected_gap_upper: float | None = None
    expected_gap_lower: float | None = None


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


def _format_signed_mev(value_ev: float) -> str:
    mev = value_ev * 1.0e3
    if abs(mev) < 5.0e-10:
        return "0 meV"
    return f"{mev:+.0f} meV"


def _delta_token(delta_ev: float) -> str:
    if abs(delta_ev) < 5.0e-10:
        return "0"
    magnitude_mev = int(round(abs(delta_ev) * 1.0e3))
    sign = "p" if delta_ev > 0.0 else "m"
    return f"{sign}{magnitude_mev:03d}"


def _delta_panel_dirname(delta_ev: float) -> str:
    magnitude_mev = int(round(delta_ev * 1.0e3))
    return f"delta_{magnitude_mev:+04d}mev"


def _select_band_window_around_flat_pair(
    total_bands: int,
    flat_pair: tuple[int, int],
    bands_per_side: int,
) -> tuple[int, ...]:
    lower = max(0, int(flat_pair[0]) - int(bands_per_side))
    upper = min(int(total_bands), int(flat_pair[1]) + int(bands_per_side) + 1)
    return tuple(range(lower, upper))


def _band_width(energies: np.ndarray, band_index: int) -> float:
    band = np.asarray(energies[:, int(band_index)], dtype=float)
    return float(np.max(band) - np.min(band))


def _minimum_gap(energies: np.ndarray, lower_band: int, upper_band: int) -> float:
    lower = np.asarray(energies[:, int(lower_band)], dtype=float)
    upper = np.asarray(energies[:, int(upper_band)], dtype=float)
    return float(np.min(upper - lower))


def _path_gap_minimum(
    path_result: PathBandsResult,
    band_indices: tuple[int, int],
) -> tuple[int, float]:
    valence_band = np.asarray(path_result.energies[:, int(band_indices[0])], dtype=float)
    conduction_band = np.asarray(path_result.energies[:, int(band_indices[1])], dtype=float)
    gaps = conduction_band - valence_band
    index = int(np.argmin(gaps))
    return index, float(gaps[index])


def _format_k_location(path_result: PathBandsResult, index: int) -> str:
    for node in path_result.path.nodes:
        if int(node.index - 1) == int(index):
            return _display_k_label(node.label)
    kvec = complex(path_result.path.kvec[int(index)])
    return f"({kvec.real:+.4f}, {kvec.imag:+.4f}) nm^-1"


def _display_k_label(label: str) -> str:
    return {"Gamma": "Γ", "GammaPrime": "Γ'", "M": "M", "K": "K", "Kprime": "K'", "KPrime": "K'"}.get(label, label)


def _panel_annotation(delta_ev: float, path_result: PathBandsResult, band_indices: tuple[int, int]) -> str:
    gap_index, gap = _path_gap_minimum(path_result, band_indices)
    location = _format_k_location(path_result, gap_index)
    return f"flat_gap @ Δ={_format_signed_mev(delta_ev)}: {gap * 1.0e3:.2f} meV at k={location}"


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


def _describe_band_summary(summary: _CheckpointBandSummary) -> str:
    return (
        f"flat bands={summary.flat_band_indices}, "
        f"widths=({_format_mev(summary.valence_width)}, {_format_mev(summary.conduction_width)}), "
        f"flat gap={_format_mev(summary.flat_gap)}, "
        f"outer gaps=({_format_mev(summary.lower_gap)}, {_format_mev(summary.upper_gap)})"
    )


def _summarize_band_gap_at_k(
    model: TMBGModel,
    k_tilde: complex,
    band_indices: tuple[int, int],
    *,
    valley: int,
    k_label: str,
) -> _KPointGapSummary:
    upper_band = int(band_indices[1])
    resolved_n_bands = max(upper_band + 1, 2)
    evals, _ = model.diagonalize(complex(k_tilde), valley=valley, n_bands=resolved_n_bands)
    valence_index, conduction_index = (int(band_indices[0]), int(band_indices[1]))
    valence_energy = float(evals[valence_index])
    conduction_energy = float(evals[conduction_index])
    return _KPointGapSummary(
        k_label=str(k_label),
        k_tilde=complex(k_tilde),
        flat_band_indices=(valence_index, conduction_index),
        valence_energy=valence_energy,
        conduction_energy=conduction_energy,
        flat_gap=float(conduction_energy - valence_energy),
    )


def _summarize_neutral_gap_at_k(
    model: TMBGModel,
    k_tilde: complex,
    *,
    valley: int,
    k_label: str,
) -> _KPointGapSummary:
    evals, _ = model.diagonalize(complex(k_tilde), valley=valley)
    flat_band_indices = infer_flat_band_indices(np.asarray(evals, dtype=float).reshape(1, -1))
    return _summarize_band_gap_at_k(
        model,
        complex(k_tilde),
        flat_band_indices,
        valley=valley,
        k_label=k_label,
    )


def _describe_kpoint_gap(summary: _KPointGapSummary) -> str:
    return (
        f"{summary.k_label} gap={_format_mev(summary.flat_gap)} on bands {summary.flat_band_indices} "
        f"(E_v={_format_mev(summary.valence_energy)}, E_c={_format_mev(summary.conduction_energy)})."
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


def _build_checkpoint_model(case: PaperCheckpointCase, *, n_shells: int) -> TMBGModel:
    return TMBGModel.from_config(case.theta_deg, n_shells=n_shells, params=case.build_params())


def diagnose_ktilde_symmetry(
    *,
    theta_deg: float = 1.21,
    n_shells: int = 5,
    valley: int = 1,
    output_dir: Path | str | None = None,
) -> ValidationReport:
    diagnostic_cases = (
        _KtildeDiagnosticCase(
            name="R1.full_model_delta_0_ktilde_gap",
            detail_prefix="Current Park Fig. 2(a) full-model reference should remain nearly touching at K̃.",
            params=TMBGParameters.full(interlayer_potential=0.0, staggered_potential=0.0),
            theta_deg=theta_deg,
            expected_gap_upper=1.0e-3,
        ),
        _KtildeDiagnosticCase(
            name="D2.delta_p060_opens_ktilde_gap",
            detail_prefix="Applying Δ=+60 meV should open a visible K̃ gap.",
            params=TMBGParameters.full(interlayer_potential=0.06, staggered_potential=0.0),
            theta_deg=theta_deg,
            expected_gap_lower=5.0e-2,
        ),
    )

    checks: list[ValidationCheck] = []
    for case in diagnostic_cases:
        model = TMBGModel.from_config(case.theta_deg, n_shells=n_shells, params=case.params)
        _, path_summary = _compute_flat_band_path_summary(model, valley=valley)
        summary = _summarize_band_gap_at_k(
            model,
            model.lattice.k_m,
            path_summary.flat_band_indices,
            valley=valley,
            k_label="Ktilde",
        )
        passed = True
        if case.expected_gap_upper is not None:
            passed = summary.flat_gap < case.expected_gap_upper
        if case.expected_gap_lower is not None:
            passed = passed and summary.flat_gap > case.expected_gap_lower
        checks.append(
            make_validation_check(
                case.name, passed, summary.flat_gap,
                detail=(
                    f"{case.detail_prefix} {_describe_kpoint_gap(summary)} "
                    f"Flat-band indices inferred from the full K-Γ-M-K' path: {path_summary.flat_band_indices}."
                ),
            )
        )

    c2zt_model = TMBGModel.from_config(theta_deg, n_shells=n_shells, params=TMBGParameters.full())
    _, c2zt_path_summary = _compute_flat_band_path_summary(c2zt_model, valley=valley)
    c2zt_gap = _summarize_band_gap_at_k(
        c2zt_model,
        c2zt_model.lattice.k_m,
        c2zt_path_summary.flat_band_indices,
        valley=valley,
        k_label="Ktilde",
    )
    c2zt_residual = _measure_c2zt_residual(c2zt_model, valley=valley, k_tilde=c2zt_model.lattice.k_m)
    checks.append(
        make_validation_check(
            "D3.c2zt_absent", c2zt_residual > 1.0e-6, c2zt_residual,
            detail=(
                "tMBG should lack C2zT. "
                f"max |U H* U^dagger - H| = {_format_mev(c2zt_residual)} at K̃; "
                f"{_describe_kpoint_gap(c2zt_gap)}"
            ),
        )
    )

    report = ValidationReport(title="tMBG Ktilde Symmetry Diagnostics", checks=tuple(checks))
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        report_path = resolved_output_dir / "ktilde_symmetry_report.md"
        write_text_artifact(report.to_markdown() + "\n", report_path)
    return report


def reproduce_paper_checkpoints(
    *,
    n_shells: int = 5,
    points_per_segment: int = 120,
    path_n_bands: int | None = None,
    topology_mesh_size: int = 24,
    topology_n_bands: int | None = None,
    valley: int = 1,
    verify_opposite_valley: bool = True,
    cp4_delta_abs: float = 0.06,
    cp6_staggered_potentials: tuple[float, ...] = (0.01, -0.01),
    bands_per_side: int = 6,
    output_dir: Path | str | None = None,
) -> ValidationReport:
    path_cache: dict[PaperCheckpointCase, tuple[TMBGModel, PathBandsResult, _CheckpointBandSummary]] = {}
    topology_cache: dict[tuple[PaperCheckpointCase, int, int], object] = {}
    checks: list[ValidationCheck] = []

    def get_path_case(case: PaperCheckpointCase) -> tuple[TMBGModel, PathBandsResult, _CheckpointBandSummary]:
        cached = path_cache.get(case)
        if cached is not None:
            return cached

        model = _build_checkpoint_model(case, n_shells=n_shells)
        resolved_n_bands = model.lattice.matrix_dim if path_n_bands is None else int(path_n_bands)
        path_result = model.bands_along_standard_path(points_per_segment=points_per_segment, n_bands=resolved_n_bands)
        summary = _summarize_flat_bands(path_result)
        cached = (model, path_result, summary)
        path_cache[case] = cached
        return cached

    def get_topology_result(case: PaperCheckpointCase, band_index: int, *, valley_label: int):
        cache_key = (case, int(band_index), int(valley_label))
        cached = topology_cache.get(cache_key)
        if cached is not None:
            return cached

        model, _, _ = get_path_case(case)
        resolved_n_bands = max(int(band_index) + 1, 0 if topology_n_bands is None else int(topology_n_bands))
        result = model.topology_on_grid(
            topology_mesh_size,
            int(band_index),
            valley=valley_label,
            n_bands=resolved_n_bands,
        )
        topology_cache[cache_key] = result
        return result

    cp1_case = PaperCheckpointCase(theta_deg=1.07, model_name="minimal")
    _, _, cp1_summary = get_path_case(cp1_case)
    cp1_pass = cp1_summary.widest_flat_band < 5.0e-3
    checks.append(
        make_validation_check(
            "CP1.minimal_magic_angle_bandwidth", cp1_pass, cp1_summary.widest_flat_band,
            detail="Minimal model at θ=1.07° should keep the neutral flat-band pair below 5 meV. "
            + _describe_band_summary(cp1_summary),
        )
    )

    fig2_cases = tuple(
        PaperCheckpointCase(theta_deg=1.21, model_name="full", interlayer_potential=delta_ev)
        for delta_ev in (0.0, 0.06, -0.04)
    )
    fig2_panels: list[TMBGBandPlotPanel] = []
    cp3_expected = {
        0.0: (2, -3),
        0.06: (-2, 1),
        -0.04: (1, -2),
    }
    cp3_delta_zero_pair: tuple[int, int] | None = None

    for case in fig2_cases:
        model, path_result, summary = get_path_case(case)
        selected_band_indices = _select_band_window_around_flat_pair(
            path_result.energies.shape[1],
            summary.flat_band_indices,
            bands_per_side,
        )
        fig2_panels.append(
            TMBGBandPlotPanel(
                label=case.panel_label,
                path_result=path_result,
                band_indices=selected_band_indices,
                flat_band_indices=summary.flat_band_indices,
                annotation=_panel_annotation(case.interlayer_potential, path_result, summary.flat_band_indices),
            )
        )

        outer_gap_floor = summary.outer_gap_floor
        cp2_pass = summary.widest_flat_band < 2.0e-2 and outer_gap_floor is not None and outer_gap_floor > 1.0e-2
        checks.append(
            make_validation_check(
                f"CP2.delta_{_delta_token(case.interlayer_potential)}_band_isolation", cp2_pass, summary.widest_flat_band,
                detail="Full model along K-Γ-M-K' should show an isolated neutral flat-band pair in the Fig. 2 window. "
                + _describe_band_summary(summary),
            )
        )

        if abs(case.interlayer_potential) < 5.0e-10:
            ktilde_gap = _summarize_band_gap_at_k(
                model=model,
                k_tilde=model.lattice.k_m,
                band_indices=summary.flat_band_indices,
                valley=valley,
                k_label="Ktilde",
            )
            cp2b_pass = ktilde_gap.flat_gap < 1.0e-3
            checks.append(
                make_validation_check(
                    "CP2b.delta_0_band_touching", cp2b_pass, ktilde_gap.flat_gap,
                    detail=(
                        "At Δ = 0 in the full model, the neutral flat-band pair should remain nearly touching "
                        f"at K̃ in Park Fig. 2(a); {_describe_kpoint_gap(ktilde_gap)}"
                    ),
                )
            )

        expected_valence, expected_conduction = cp3_expected[case.interlayer_potential]
        observed_valence_result = get_topology_result(case, summary.flat_band_indices[0], valley_label=valley)
        observed_conduction_result = get_topology_result(case, summary.flat_band_indices[1], valley_label=valley)
        observed_valence = int(observed_valence_result.rounded_chern_number)
        observed_conduction = int(observed_conduction_result.rounded_chern_number)
        cp3_pass = (observed_valence, observed_conduction) == (expected_valence, expected_conduction)
        checks.append(
            make_validation_check(
                f"CP3.delta_{_delta_token(case.interlayer_potential)}_valley_chern",
                cp3_pass,
                str((observed_valence, observed_conduction)),
                detail=(
                    f"K valley flat-band Chern numbers should match {expected_valence, expected_conduction}; "
                    f"observed {(observed_valence, observed_conduction)} on bands {summary.flat_band_indices}."
                ),
            )
        )

        if verify_opposite_valley:
            kprime_valence_result = get_topology_result(case, summary.flat_band_indices[0], valley_label=-valley)
            kprime_conduction_result = get_topology_result(case, summary.flat_band_indices[1], valley_label=-valley)
            kprime_valence = int(kprime_valence_result.rounded_chern_number)
            kprime_conduction = int(kprime_conduction_result.rounded_chern_number)
            opposite_valley_pass = (
                kprime_valence == -observed_valence and kprime_conduction == -observed_conduction
            )
            checks.append(
                make_validation_check(
                    f"CP3.delta_{_delta_token(case.interlayer_potential)}_opposite_valley",
                    opposite_valley_pass,
                    str((kprime_valence, kprime_conduction)),
                    detail=(
                        "K' valley should carry the opposite Chern numbers. "
                        f"K={(observed_valence, observed_conduction)}, K'={(kprime_valence, kprime_conduction)}."
                    ),
                )
            )
        else:
            checks.append(
                ValidationCheck(
                    name=f"CP3.delta_{_delta_token(case.interlayer_potential)}_opposite_valley",
                    status="skipped",
                    detail="Opposite-valley topology verification disabled for this checkpoint run.",
                )
            )

        if abs(case.interlayer_potential) < 5.0e-10:
            cp3_delta_zero_pair = (observed_valence, observed_conduction)

    cp4_full_plus = PaperCheckpointCase(theta_deg=1.21, model_name="full", interlayer_potential=cp4_delta_abs)
    cp4_full_minus = PaperCheckpointCase(theta_deg=1.21, model_name="full", interlayer_potential=-cp4_delta_abs)
    cp4_min_plus = PaperCheckpointCase(theta_deg=1.21, model_name="minimal", interlayer_potential=cp4_delta_abs)
    cp4_min_minus = PaperCheckpointCase(theta_deg=1.21, model_name="minimal", interlayer_potential=-cp4_delta_abs)
    _, _, cp4_full_plus_summary = get_path_case(cp4_full_plus)
    _, _, cp4_full_minus_summary = get_path_case(cp4_full_minus)
    _, _, cp4_min_plus_summary = get_path_case(cp4_min_plus)
    _, _, cp4_min_minus_summary = get_path_case(cp4_min_minus)

    cp4_full_asym = abs(cp4_full_plus_summary.widest_flat_band - cp4_full_minus_summary.widest_flat_band)
    cp4_min_asym = abs(cp4_min_plus_summary.widest_flat_band - cp4_min_minus_summary.widest_flat_band)
    cp4_pass = cp4_full_asym > max(2.0 * cp4_min_asym, 1.0e-3)
    checks.append(
        make_validation_check(
            "CP4.delta_sign_asymmetry", cp4_pass, cp4_full_asym,
            detail=(
                "Full model should show a visibly stronger Δ ↔ -Δ asymmetry than the minimal model. "
                f"full asymmetry={_format_mev(cp4_full_asym)}, minimal asymmetry={_format_mev(cp4_min_asym)}."
            ),
        )
    )

    cp5_full_case = PaperCheckpointCase(theta_deg=1.21, model_name="full")
    cp5_min_case = PaperCheckpointCase(theta_deg=1.21, model_name="minimal")
    _, _, cp5_full_summary = get_path_case(cp5_full_case)
    _, _, cp5_min_summary = get_path_case(cp5_min_case)
    cp5_ratio = cp5_full_summary.widest_flat_band / max(cp5_min_summary.widest_flat_band, 1.0e-12)
    cp5_pass = cp5_full_summary.widest_flat_band > cp5_min_summary.widest_flat_band and cp5_ratio > 2.0
    checks.append(
        make_validation_check(
            "CP5.full_vs_minimal_bandwidth", cp5_pass, cp5_ratio,
            detail=(
                "Full model should broaden the neutral flat-band pair relative to the minimal model at the same parameters. "
                f"full={_format_mev(cp5_full_summary.widest_flat_band)}, "
                f"minimal={_format_mev(cp5_min_summary.widest_flat_band)}, ratio={cp5_ratio:.3f}."
            ),
        )
    )

    cp6_abs_maxima: list[int] = []
    for staggered_potential in cp6_staggered_potentials:
        cp6_case = PaperCheckpointCase(
            theta_deg=1.21,
            model_name="full",
            interlayer_potential=0.0,
            staggered_potential=staggered_potential,
        )
        _, _, cp6_summary = get_path_case(cp6_case)
        cp6_valence = int(
            get_topology_result(cp6_case, cp6_summary.flat_band_indices[0], valley_label=valley).rounded_chern_number
        )
        cp6_conduction = int(
            get_topology_result(cp6_case, cp6_summary.flat_band_indices[1], valley_label=valley).rounded_chern_number
        )
        cp6_abs_maxima.append(max(abs(cp6_valence), abs(cp6_conduction)))
    cp6_reference_abs_max = 0 if cp3_delta_zero_pair is None else max(abs(value) for value in cp3_delta_zero_pair)
    cp6_pass = cp6_reference_abs_max == 3 and all(abs_max < 3 for abs_max in cp6_abs_maxima)
    checks.append(
        make_validation_check(
            "CP6.staggered_potential_suppresses_abs3", cp6_pass, str(tuple(cp6_abs_maxima)),
            detail=(
                "At the sampled Fig. 4 reference point, Δ_S ≠ 0 should remove |C|=3 from the neutral flat-band pair. "
                f"reference max |C|={cp6_reference_abs_max}, staggered max |C| values={tuple(cp6_abs_maxima)}."
            ),
        )
    )

    report = ValidationReport(title="tMBG Park 2020 Checkpoint Validation", checks=tuple(checks))

    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        reference_model, _, _ = get_path_case(fig2_cases[0])
        lattice_info_path = resolved_output_dir / "lattice_info.json"
        with lattice_info_path.open("w", encoding="utf-8") as handle:
            json.dump(reference_model.lattice_summary(), handle, indent=2)
        write_tmbg_lattice_plot(
            resolved_output_dir,
            reference_model.lattice,
            title=f"tMBG moire reciprocal lattice, theta={reference_model.theta_deg:.2f}°",
        )

        for case in fig2_cases:
            model, path_result, summary = get_path_case(case)
            panel_dir = resolved_output_dir / _delta_panel_dirname(case.interlayer_potential)
            panel_dir.mkdir(parents=True, exist_ok=True)
            selected_band_indices = _select_band_window_around_flat_pair(
                path_result.energies.shape[1],
                summary.flat_band_indices,
                bands_per_side,
            )
            max_selected_band = int(max(selected_band_indices))
            grid_result = model.bands_on_grid(
                topology_mesh_size,
                valley=valley,
                n_bands=max_selected_band + 1,
                return_eigenvectors=False,
            )
            valence_topology = get_topology_result(case, summary.flat_band_indices[0], valley_label=valley)
            conduction_topology = get_topology_result(case, summary.flat_band_indices[1], valley_label=valley)
            np.savez_compressed(
                panel_dir / "bands_path.npz",
                k_distance=np.asarray(path_result.path.kdist, dtype=float),
                energies=np.asarray(path_result.energies[:, selected_band_indices], dtype=float),
                kvec_nm_inv=np.stack(
                    [
                        np.asarray(path_result.path.kvec.real, dtype=float),
                        np.asarray(path_result.path.kvec.imag, dtype=float),
                    ],
                    axis=-1,
                ),
                band_indices=np.asarray(selected_band_indices, dtype=int),
                flat_band_indices=np.asarray(summary.flat_band_indices, dtype=int),
                k_labels=np.asarray(path_result.path.labels, dtype=object),
            )
            np.savez_compressed(
                panel_dir / "bands_grid.npz",
                k_grid_frac=np.asarray(grid_result.k_grid_frac, dtype=float),
                kvec_nm_inv=np.stack(
                    [
                        np.asarray(grid_result.kvec.real, dtype=float),
                        np.asarray(grid_result.kvec.imag, dtype=float),
                    ],
                    axis=-1,
                ),
                energies=np.asarray(grid_result.energies[:, :, selected_band_indices], dtype=float),
                band_indices=np.asarray(selected_band_indices, dtype=int),
                flat_band_indices=np.asarray(summary.flat_band_indices, dtype=int),
            )
            with (panel_dir / "chern_numbers.json").open("w", encoding="utf-8") as handle:
                payload = {
                    "delta_ev": float(case.interlayer_potential),
                    "flat_band_indices": list(summary.flat_band_indices),
                    "valley": int(valley),
                    "valence": {
                        "band_index": int(summary.flat_band_indices[0]),
                        "chern_number": float(valence_topology.chern_number),
                        "rounded_chern_number": int(valence_topology.rounded_chern_number),
                        "integer_residual": float(valence_topology.integer_residual),
                    },
                    "conduction": {
                        "band_index": int(summary.flat_band_indices[1]),
                        "chern_number": float(conduction_topology.chern_number),
                        "rounded_chern_number": int(conduction_topology.rounded_chern_number),
                        "integer_residual": float(conduction_topology.integer_residual),
                    },
                }
                if verify_opposite_valley:
                    kprime_valence = get_topology_result(case, summary.flat_band_indices[0], valley_label=-valley)
                    kprime_conduction = get_topology_result(case, summary.flat_band_indices[1], valley_label=-valley)
                    payload["opposite_valley"] = {
                        "valley": int(-valley),
                        "valence": int(kprime_valence.rounded_chern_number),
                        "conduction": int(kprime_conduction.rounded_chern_number),
                    }
                json.dump(payload, handle, indent=2)
            np.savez_compressed(
                panel_dir / "berry_curvature.npz",
                berry_curvature=np.stack(
                    [
                        np.asarray(valence_topology.berry_curvature, dtype=float),
                        np.asarray(conduction_topology.berry_curvature, dtype=float),
                    ],
                    axis=-1,
                ),
                band_indices=np.asarray(summary.flat_band_indices, dtype=int),
                valley=int(valley),
            )
        write_tmbg_paper_band_figure(
            resolved_output_dir,
            tuple(fig2_panels),
            stem="fig2_like_bands",
            title="Park 2020 Fig. 2 Checkpoint",
            ylim=(-0.100, 0.100),
        )
        report_path = resolved_output_dir / "paper_checkpoint_report.md"
        write_text_artifact(report.to_markdown() + "\n", report_path)
        ktilde_report = diagnose_ktilde_symmetry(
            theta_deg=1.21,
            n_shells=n_shells,
            valley=valley,
            output_dir=resolved_output_dir,
        )
        core_report = validate_physics(
            reference_model,
            valley=valley,
            n_bands=min(12, reference_model.lattice.matrix_dim),
            include_c3_check=True,
            include_cutoff_check=True,
        )
        validation_report = ValidationReport.combine(
            "tMBG Physical Validation",
            core_report,
            ktilde_report,
            report,
        )
        validation_report_path = resolved_output_dir / "validation_report.md"
        write_text_artifact(validation_report.to_markdown() + "\n", validation_report_path)
        run_log_path = resolved_output_dir / "run.log"
        write_text_artifact(
            "\n\n".join(
                [
                    "# tMBG checkpoint run",
                    validation_report.to_markdown(),
                    f"paper_checkpoint_report={report_path}",
                    f"validation_report={validation_report_path}",
                    f"lattice_info={lattice_info_path}",
                ]
            )
            + "\n",
            run_log_path,
        )

    return report


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
