from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

import mean_field.crpa.chunk_merge as chunk_merge
from mean_field.core.io import write_text_artifact
from mean_field.crpa.diagnostics import write_all_epsilon_diagnostics
from mean_field.crpa.workflow import load_crpa_result
from mean_field.runtime import ensure_not_running_compute_on_login_node
from mean_field.workflows import (
    WorkflowJobSpec,
    WorkflowJobState,
    WorkflowManifest,
    WorkflowRunState,
    collect_slurm_metadata,
    write_workflow_manifest,
    write_workflow_run_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEVTOOL_DISPATCHER = (_REPO_ROOT / "scripts" / "mean_field_tools.py").resolve()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge q-point TBG cRPA chunk artifact directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True, help="Chunk artifact directory.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a nonempty, in-bounds, duplicate-free subset instead of exact lk^2 q coverage.",
    )
    return parser


def _merge_command(output_dir: Path, chunks: tuple[Path, ...], *, allow_partial: bool) -> tuple[str, ...]:
    command: list[str] = [
        sys.executable,
        str(_DEVTOOL_DISPATCHER),
        "merge_tbg_crpa_chunks",
        "--output-dir",
        str(output_dir.resolve()),
    ]
    for chunk in chunks:
        command.extend(["--chunk", str(chunk.resolve())])
    if allow_partial:
        command.append("--allow-partial")
    return tuple(command)


def _merge_workflow_manifest(
    output_dir: Path,
    chunks: tuple[Path, ...],
    *,
    allow_partial: bool,
) -> WorkflowManifest:
    coverage_policy = chunk_merge.coverage_policy(allow_partial)
    launch_cwd = Path.cwd().resolve()
    resolved_output_dir = output_dir.resolve()
    resolved_chunks = tuple(chunk.resolve() for chunk in chunks)
    chunk_jobs = tuple(
        WorkflowJobSpec(
            name=f"input_chunk_{index}",
            command=("provenance-only", str(chunk)),
            output_dir=chunk,
            metadata={
                "kind": "input_chunk",
                "chunk_dir": str(chunk),
                "provenance_only": True,
                "executable": False,
                "working_directory": str(launch_cwd),
                "launch_cwd": str(launch_cwd),
                "replay_command_owner": str(_DEVTOOL_DISPATCHER),
            },
        )
        for index, chunk in enumerate(resolved_chunks)
    )
    merge_job = WorkflowJobSpec(
        name="merge",
        command=_merge_command(resolved_output_dir, resolved_chunks, allow_partial=allow_partial),
        output_dir=resolved_output_dir,
        dependencies=tuple(job.name for job in chunk_jobs),
        metadata={
            "kind": "crpa_merge",
            "chunk_count": len(chunks),
            "allow_partial": bool(allow_partial),
            "coverage_policy": coverage_policy,
            "working_directory": str(launch_cwd),
            "launch_cwd": str(launch_cwd),
            "replay_command_owner": str(_DEVTOOL_DISPATCHER),
        },
    )
    return WorkflowManifest(
        name="tbg_crpa_merge",
        root=resolved_output_dir,
        jobs=chunk_jobs + (merge_job,),
        metadata={
            "system": "TBG",
            "workflow": "cRPA merge",
            "chunk_count": len(chunks),
            "allow_partial": bool(allow_partial),
            "coverage_policy": coverage_policy,
            "working_directory": str(launch_cwd),
            "launch_cwd": str(launch_cwd),
            "replay_command_owner": str(_DEVTOOL_DISPATCHER),
            "input_jobs": "provenance-only; not executable workflow jobs",
            "slurm_hint": "Run production cRPA merge/diagnostics on compute nodes or through Slurm if inputs are large.",
        },
    )


def _merge_workflow_state(
    manifest: WorkflowManifest,
    merge_status: str,
    *,
    inputs_validated: bool = False,
    message: str | None = None,
    coverage: Mapping[str, object] | None = None,
) -> WorkflowRunState:
    slurm_metadata = collect_slurm_metadata()
    merge_metadata = {"slurm": slurm_metadata} if slurm_metadata and merge_status != "pending" else {}
    states: list[WorkflowJobState] = []
    for job in manifest.jobs:
        if job.name.startswith("input_chunk_"):
            states.append(
                WorkflowJobState(
                    name=job.name,
                    status="succeeded" if inputs_validated else "pending",
                    message="input chunk loaded and validated" if inputs_validated else "awaiting input validation",
                    metadata={"provenance_only": True, "executable": False},
                )
            )
        elif job.name == "merge":
            states.append(WorkflowJobState(name=job.name, status=merge_status, message=message, metadata=merge_metadata))
        else:
            states.append(WorkflowJobState(name=job.name, status="pending"))
    state_metadata: dict[str, object] = {"manifest": "workflow_manifest.json"}
    if coverage is not None:
        state_metadata["coverage"] = dict(coverage)
    if slurm_metadata:
        state_metadata["slurm"] = slurm_metadata
    return WorkflowRunState(
        name=manifest.name,
        jobs=tuple(states),
        metadata=state_metadata,
    )


def _write_merge_workflow_artifacts(
    output_dir: Path,
    manifest: WorkflowManifest,
    state: WorkflowRunState,
) -> None:
    write_workflow_manifest(manifest, output_dir / "workflow_manifest.json")
    write_workflow_run_state(state, output_dir / "workflow_run_state.json")
    write_text_artifact(state.to_markdown() + "\n", output_dir / "workflow_run_state.md")

def _run_merge(
    args: argparse.Namespace,
    *,
    on_validated: Callable[[Mapping[str, object]], None] | None = None,
) -> None:
    chunks = tuple(Path(item) for item in args.chunk)
    output_dir = Path(args.output_dir)
    params, arrays, coverage = chunk_merge.load_and_validate_chunks(
        chunks, allow_partial=bool(args.allow_partial)
    )
    if on_validated is not None:
        on_validated(coverage)

    chunk_merge.write_merged_crpa_base_artifacts(params, arrays, coverage, output_dir)

    effective = arrays["effective_epsilon"]
    q_abs = arrays["q_abs"]
    q_abs_nm_inv = arrays["q_abs_nm_inv"]
    coulomb_params = params["coulomb_params"]
    if not isinstance(coulomb_params, dict):
        raise ValueError("Validated coulomb_params unexpectedly ceased to be a mapping")
    epsilon_bn = float(coulomb_params["epsilon_bn"])

    np.savetxt(
        output_dir / "epsilon_vs_q.tsv",
        np.column_stack(
            [
                q_abs.reshape(-1),
                q_abs_nm_inv.reshape(-1),
                effective.reshape(-1),
                (effective * epsilon_bn).reshape(-1),
            ]
        ),
        delimiter="\t",
        header="q_abs_dimless\tq_abs_nm_inv\teffective_epsilon\teffective_epsilon_times_epsilon_bn",
        comments="",
    )

    import matplotlib.pyplot as plt

    q_flat = q_abs_nm_inv.reshape(-1)
    eps_flat = (effective * epsilon_bn).reshape(-1)
    plot_order = np.argsort(q_flat)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.scatter(q_flat[plot_order], eps_flat[plot_order], s=12, linewidths=0.0, alpha=0.8)
    ax.set_xlabel(r"$|\mathbf{q}|$ (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon(\mathbf{q})\,\epsilon_{\rm BN}$")
    ax.set_title("cRPA effective dielectric constant")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "epsilon_vs_q.pdf")
    plt.close(fig)

    report = [
        "# cRPA merged chunk validation report",
        "",
        "## Parameters",
        "",
        f"- theta_deg: {params['theta_deg']}",
        f"- lk: {params['lk']}",
        f"- lg: {params['lg']}",
        f"- q_lg: {params['q_lg']}",
        f"- q_point_count: {params['q_point_count']}",
        "",
        "## Coverage",
        "",
        f"- allow_partial: {str(bool(coverage['allow_partial'])).lower()}",
        f"- policy: {coverage['policy']}",
        f"- complete: {str(bool(coverage['complete'])).lower()}",
        f"- expected_q_point_count: {coverage['expected_q_point_count']}",
        f"- actual_q_point_count: {coverage['q_point_count']}",
        f"- missing_canonical_flat_indices: {coverage['missing_flat_indices']}",
        "",
        "## Convention Metadata",
        "",
    ]
    for key, value in sorted(dict(params.get("metadata", {})).items()):
        report.append(f"- {key}: {value}")
    report.extend(
        [
            "",
            "## Checks",
            "",
            f"- effective_epsilon_times_bn_min: {float(np.min(effective * epsilon_bn)):.12g}",
            f"- effective_epsilon_times_bn_max: {float(np.max(effective * epsilon_bn)):.12g}",
            "",
            "## Chunks",
            "",
        ]
    )
    report.extend(f"- `{chunk}`" for chunk in chunks)
    write_text_artifact("\n".join(report) + "\n", output_dir / "validation_report.md")
    diagnostic_summary = write_all_epsilon_diagnostics(load_crpa_result(output_dir), output_dir)
    print(f"[crpa-merge] wrote merged artifact to {output_dir}", flush=True)
    print(
        f"[crpa-merge] q_point_count={params['q_point_count']} chunks={len(chunks)} "
        f"coverage={coverage['policy']} complete={str(bool(coverage['complete'])).lower()}",
        flush=True,
    )
    print(
        "[crpa-merge] diagnostics "
        f"q_peak_nm_inv={diagnostic_summary.q_peak_nm_inv:.6g} "
        f"eps_total_peak={diagnostic_summary.eps_total_peak:.6g} "
        f"eps_total_q12={diagnostic_summary.eps_total_q12:.6g} "
        f"eps_diag_imag_max_abs={diagnostic_summary.eps_diag_imag_max_abs:.6g}",
        flush=True,
    )

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ensure_not_running_compute_on_login_node("TBG cRPA chunk merge and diagnostics")

    chunks = tuple(Path(item) for item in args.chunk)
    output_dir = Path(args.output_dir)
    manifest = _merge_workflow_manifest(output_dir, chunks, allow_partial=bool(args.allow_partial))
    _write_merge_workflow_artifacts(
        output_dir,
        manifest,
        _merge_workflow_state(manifest, "running", message="cRPA merge started"),
    )
    validated_coverage: dict[str, object] | None = None

    def record_validated_inputs(coverage: Mapping[str, object]) -> None:
        nonlocal validated_coverage
        validated_coverage = dict(coverage)
        _write_merge_workflow_artifacts(
            output_dir,
            manifest,
            _merge_workflow_state(
                manifest,
                "running",
                inputs_validated=True,
                message="all input chunks loaded and validated; cRPA merge running",
                coverage=coverage,
            ),
        )

    try:
        _run_merge(args, on_validated=record_validated_inputs)
    except Exception as exc:
        _write_merge_workflow_artifacts(
            output_dir,
            manifest,
            _merge_workflow_state(
                manifest,
                "failed",
                inputs_validated=validated_coverage is not None,
                message=str(exc),
                coverage=validated_coverage,
            ),
        )
        raise
    _write_merge_workflow_artifacts(
        output_dir,
        manifest,
        _merge_workflow_state(
            manifest,
            "succeeded",
            inputs_validated=True,
            message="cRPA merge outputs written",
            coverage=validated_coverage,
        ),
    )


if __name__ == "__main__":
    main()
