from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403
from ._runners_summaries import *  # noqa: F401,F403

def write_bm_unstrained_benchmark_artifacts(
    output_dir: Path | str,
    result: BMUnstrainedBenchmarkRun,
    *,
    write_contract_sidecars: bool = True,
    overwrite_contract_sidecars: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if write_contract_sidecars:
        _ensure_tbg_zero_field_contract_sidecars_writable(
            output_dir,
            overwrite_contract_sidecars=overwrite_contract_sidecars,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    path_tsv_path = _write_bm_path_tsv(output_dir / "computed_bm_path.tsv", result.run)
    nodes_tsv_path = _write_bm_nodes_tsv(output_dir / "computed_nodes.tsv", result.run)
    summary_path = _write_bm_summary(output_dir / "computed_summary.txt", result.run)
    parity_path = _write_key_value_summary(
        output_dir / "parity_to_reference_summary.txt",
        [
            ("implementation", "python_b0"),
            ("reference_impl", "b0_reference"),
            ("reference_path_tsv", str(result.reference.path_tsv_path)),
            ("kdist_max_abs_diff", str(result.parity.kdist_max_abs_diff)),
            ("max_abs_band_diff_mev", str(result.parity.max_abs_band_diff_mev)),
            ("rms_band_diff_mev", str(result.parity.rms_band_diff_mev)),
            ("mean_abs_band_diff_mev", str(result.parity.mean_abs_band_diff_mev)),
            ("k_middle_gap_diff_meV", str(result.parity.k_middle_gap_diff_mev)),
            (
                "valence_bandwidth_diff_meV",
                "" if result.parity.valence_bandwidth_diff_mev is None else str(result.parity.valence_bandwidth_diff_mev),
            ),
            (
                "conduction_bandwidth_diff_meV",
                ""
                if result.parity.conduction_bandwidth_diff_mev is None
                else str(result.parity.conduction_bandwidth_diff_mev),
            ),
        ],
    )
    runtime_summary_path = _write_bm_runtime_summary(output_dir / "runtime_summary.txt", result.run)
    runtime_parity_path = _write_bm_runtime_parity_summary(output_dir / "runtime_to_reference_summary.txt", result)
    plot_paths = write_bm_band_plot(
        output_dir,
        theta_deg=result.reference.theta_deg,
        path=result.run.path,
        path_solution=result.run.path_solution,
        stem="band_plot",
    )

    artifacts = {
        "path_tsv": path_tsv_path,
        "nodes_tsv": nodes_tsv_path,
        "summary_txt": summary_path,
        "parity_summary_txt": parity_path,
        "runtime_summary_txt": runtime_summary_path,
        **plot_paths,
    }
    if runtime_parity_path is not None:
        artifacts["runtime_parity_summary_txt"] = runtime_parity_path
    if write_contract_sidecars:
        write_bm_unstrained_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifacts,
            overwrite=overwrite_contract_sidecars,
        )
    return artifacts


def write_b0_hf_benchmark_artifacts(
    output_dir: Path | str,
    result: B0HFBenchmarkRun,
    *,
    write_contract_sidecars: bool = True,
    overwrite_contract_sidecars: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    if write_contract_sidecars:
        _ensure_tbg_zero_field_contract_sidecars_writable(
            output_dir,
            overwrite_contract_sidecars=overwrite_contract_sidecars,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    path_tsv_path = output_dir / "computed_hf_path.tsv"
    scf_path_tsv_path = output_dir / "computed_hf_scf_path.tsv"
    nodes_tsv_path = output_dir / "computed_nodes.tsv"
    summary_path = output_dir / "computed_summary.txt"
    parity_path = output_dir / "parity_to_reference_summary.txt"
    runtime_summary_path = output_dir / "runtime_summary.txt"
    runtime_parity_path = output_dir / "runtime_to_reference_summary.txt"

    write_hf_path_tsv(path_tsv_path, result.path_result)
    write_hf_path_nodes_tsv(nodes_tsv_path, result.path_result)
    write_hf_path_summary(summary_path, result.path_result)
    scf_plot_result = build_restricted_hf_scf_path_plot_result(
        result.hf_run,
        result.grid_solution,
        path=result.path,
        init_mode=result.path_result.init_mode,
    )
    write_hf_scf_path_tsv(scf_path_tsv_path, scf_plot_result)

    _write_key_value_summary(
        parity_path,
        [
            ("benchmark_id", result.case.benchmark_id),
            ("implementation", "python_b0"),
            ("reference_impl", "b0_reference"),
            ("reference_path_tsv", str(result.case.reference_path_tsv_path)),
            ("kdist_max_abs_diff", str(result.parity.kdist_max_abs_diff)),
            ("max_abs_band_diff_mev", str(result.parity.max_abs_band_diff_mev)),
            ("rms_band_diff_mev", str(result.parity.rms_band_diff_mev)),
            ("mean_abs_band_diff_mev", str(result.parity.mean_abs_band_diff_mev)),
            ("energy_sorting", result.parity.energy_sorting),
        ],
    )
    _write_hf_runtime_summary(runtime_summary_path, result)
    runtime_parity_written = _write_hf_runtime_parity_summary(runtime_parity_path, result)
    plot_paths = write_hf_band_plot(output_dir, result.path_result, stem="band_plot")
    scf_plot_paths = write_hf_scf_band_plot(output_dir, scf_plot_result, stem="band_plot_scf_grid")

    advisor_path, advisor_compatibility = _build_advisor_hf_benchmark_kpath(
        result.case.benchmark_id,
        result.params,
        lk=result.path_result.lk,
        points_per_segment=result.path_result.points_per_segment,
    )
    advisor_path_result = evaluate_restricted_hf_path(
        result.hf_run,
        result.grid_solution,
        points_per_segment=result.path_result.points_per_segment,
        lg=result.path_result.lg,
        overlap_lg=result.path_result.overlap_lg,
        beta=result.path_result.beta,
        relative_permittivity=result.path_result.relative_permittivity,
        screening_lm=result.path_result.screening_lm,
        finite_zero_limit=result.path_result.finite_zero_limit,
        zero_cutoff=result.path_result.zero_cutoff,
        init_mode=result.path_result.init_mode,
        path=advisor_path,
    )
    advisor_scf_plot_result = build_restricted_hf_scf_path_plot_result(
        result.hf_run,
        result.grid_solution,
        path=advisor_path,
        init_mode=result.path_result.init_mode,
    )
    advisor_path_tsv_path = output_dir / "computed_hf_path_advisor.tsv"
    advisor_scf_path_tsv_path = output_dir / "computed_hf_scf_path_advisor.tsv"
    advisor_nodes_tsv_path = output_dir / "computed_nodes_advisor.tsv"
    advisor_summary_path = output_dir / "computed_summary_advisor.txt"
    advisor_selection_path = output_dir / "advisor_path_selection.txt"
    write_hf_path_tsv(advisor_path_tsv_path, advisor_path_result)
    write_hf_path_nodes_tsv(advisor_nodes_tsv_path, advisor_path_result)
    write_hf_path_summary(advisor_summary_path, advisor_path_result)
    write_hf_scf_path_tsv(advisor_scf_path_tsv_path, advisor_scf_plot_result)
    _write_advisor_path_selection(advisor_selection_path, compatibility=advisor_compatibility)
    advisor_plot_paths = write_hf_band_plot(output_dir, advisor_path_result, stem="band_plot_advisor")
    advisor_scf_plot_paths = write_hf_scf_band_plot(output_dir, advisor_scf_plot_result, stem="band_plot_scf_grid_advisor")

    artifacts = {
        "path_tsv": path_tsv_path,
        "scf_path_tsv": scf_path_tsv_path,
        "nodes_tsv": nodes_tsv_path,
        "summary_txt": summary_path,
        "parity_summary_txt": parity_path,
        "runtime_summary_txt": runtime_summary_path,
        **plot_paths,
        "scf_band_plot_png": scf_plot_paths["band_plot_png"],
        "scf_band_plot_pdf": scf_plot_paths["band_plot_pdf"],
        "advisor_path_tsv": advisor_path_tsv_path,
        "advisor_scf_path_tsv": advisor_scf_path_tsv_path,
        "advisor_nodes_tsv": advisor_nodes_tsv_path,
        "advisor_summary_txt": advisor_summary_path,
        "advisor_selection_txt": advisor_selection_path,
        "advisor_band_plot_png": advisor_plot_paths["band_plot_png"],
        "advisor_band_plot_pdf": advisor_plot_paths["band_plot_pdf"],
        "advisor_scf_band_plot_png": advisor_scf_plot_paths["band_plot_png"],
        "advisor_scf_band_plot_pdf": advisor_scf_plot_paths["band_plot_pdf"],
    }
    if runtime_parity_written is not None:
        artifacts["runtime_parity_summary_txt"] = runtime_parity_path
    if write_contract_sidecars:
        write_b0_hf_benchmark_contract_sidecars(
            output_dir,
            result,
            artifact_paths=artifacts,
            overwrite=overwrite_contract_sidecars,
        )
    return artifacts

__all__ = [name for name in globals() if not name.startswith('__')]
