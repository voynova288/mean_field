from __future__ import annotations

import numpy as np

from ._runners_shared import *  # noqa: F401,F403
from ._runners_helpers import *  # noqa: F401,F403
from ._runners_summaries import *  # noqa: F401,F403
from .artifacts import (
    TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY,
    TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME,
    _exact_saved_scf_path_coverage,
    write_tbg_zero_field_complete_hf_state_archive_npz,
)
from .hf_runners import _has_typed_hf_source


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

    if _has_typed_hf_source(result.hf_run):
        # Production typed artifacts are saved-grid-only. The exact-source
        # validator inside the builder and archive writer runs before any plot.
        scf_plot_result = build_restricted_hf_scf_path_plot_result(
            result.hf_run,
            result.grid_solution,
            path=result.path,
            init_mode=result.hf_run.init_mode,
        )
        write_hf_scf_path_tsv(scf_path_tsv_path, scf_plot_result)
        complete_archive_path = write_tbg_zero_field_complete_hf_state_archive_npz(
            output_dir / TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_FILENAME,
            result,
        )
        coverage = _exact_saved_scf_path_coverage(result)
        if (
            not np.array_equal(
                coverage.path_sample_indices,
                np.asarray(scf_plot_result.path_sample_indices, dtype=int),
            )
            or not np.array_equal(
                coverage.grid_indices,
                np.asarray(scf_plot_result.grid_indices, dtype=int),
            )
        ):
            raise ValueError(
                "High-level exact saved-SCF path selection differs from the shared coverage gate"
            )
        advisor_unavailable_path = _write_key_value_summary(
            output_dir / "advisor_path_selection.txt",
            [
                ("status", "unavailable"),
                ("reason", "typed_production_path_advisor_disabled"),
            ],
        )
        path_status = (
            "exact_saved_scf_plot_written"
            if coverage.meaningful
            else "limitation_report_only"
        )
        _write_key_value_summary(
            runtime_summary_path,
            [
                ("benchmark_id", result.case.benchmark_id),
                ("artifact_mode", "typed_exact_saved_scf_grid_only"),
                ("hf_elapsed_sec", str(result.runtime.hf_elapsed_sec)),
                ("path_status", path_status),
                ("exact_saved_path_point_count", str(coverage.exact_point_count)),
                ("exact_distinct_coordinate_count", str(coverage.distinct_coordinate_count)),
                (
                    "represented_segment_count",
                    str(len(coverage.represented_segment_indices)),
                ),
                (
                    "represented_segment_indices_zero_based",
                    ",".join(str(index) for index in coverage.represented_segment_indices),
                ),
                ("advisor_status", "unavailable"),
                ("off_grid_reconstruction", "forbidden"),
                ("paper_figure_claim", "none"),
            ],
        )
        artifacts = {
            "scf_path_tsv": scf_path_tsv_path,
            TBG_ZERO_FIELD_COMPLETE_HF_STATE_ARCHIVE_ARTIFACT_KEY: complete_archive_path,
            "advisor_selection_txt": advisor_unavailable_path,
            "runtime_summary_txt": runtime_summary_path,
        }
        if coverage.meaningful:
            scf_plot_paths = write_hf_scf_band_plot(
                output_dir,
                scf_plot_result,
                stem="band_plot_scf_grid",
            )
            artifacts.update(
                {
                    "scf_band_plot_png": scf_plot_paths["band_plot_png"],
                    "scf_band_plot_pdf": scf_plot_paths["band_plot_pdf"],
                }
            )
        else:
            limitation_path = _write_key_value_summary(
                output_dir / "computed_hf_exact_path_limitation.txt",
                [
                    ("status", "limited"),
                    (
                        "reason",
                        "insufficient_meaningful_exact_saved_scf_path_coverage",
                    ),
                    ("required_distinct_coordinates", "3"),
                    ("observed_distinct_coordinates", str(coverage.distinct_coordinate_count)),
                    ("required_spanned_segments", "2"),
                    (
                        "observed_spanned_segments",
                        str(len(coverage.represented_segment_indices)),
                    ),
                    (
                        "segment_distinct_coordinate_counts",
                        ",".join(
                            str(count)
                            for count in coverage.segment_distinct_coordinate_counts
                        ),
                    ),
                    (
                        "missing_interior_path_nodes",
                        ",".join(coverage.missing_interior_node_labels),
                    ),
                    ("band_plot_written", "false"),
                    ("off_grid_reconstruction", "forbidden"),
                    ("paper_figure_claim", "none"),
                ],
            )
            artifacts["path_limitation_txt"] = limitation_path
        if write_contract_sidecars:
            write_b0_hf_benchmark_contract_sidecars(
                output_dir,
                result,
                artifact_paths=artifacts,
                overwrite=overwrite_contract_sidecars,
            )
        return artifacts

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
    path_interaction_spec = result.hf_run.state.interaction_spec
    if path_interaction_spec is None:
        advisor_screening_kwargs: dict[str, object] = {
            "legacy_untyped": True,
            "relative_permittivity": result.path_result.relative_permittivity,
            "screening_lm": result.path_result.screening_lm,
            "finite_zero_limit": result.path_result.finite_zero_limit,
            "zero_cutoff": result.path_result.zero_cutoff,
        }
    else:
        advisor_screening_kwargs = {"interaction_spec": path_interaction_spec}
    advisor_path_result = evaluate_restricted_hf_path(
        result.hf_run,
        result.grid_solution,
        points_per_segment=result.path_result.points_per_segment,
        lg=result.path_result.lg,
        overlap_lg=result.path_result.overlap_lg,
        beta=result.path_result.beta,
        init_mode=result.path_result.init_mode,
        path=advisor_path,
        **advisor_screening_kwargs,
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
