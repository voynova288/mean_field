from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mean_field.core.io import write_text_artifact
from mean_field.crpa.diagnostics import write_all_epsilon_diagnostics
from mean_field.crpa.workflow import load_crpa_result
from mean_field.devtools._runtime import write_json
from mean_field.workflows import (
    WorkflowJobSpec,
    WorkflowJobState,
    WorkflowManifest,
    WorkflowRunState,
    collect_slurm_metadata,
    write_workflow_manifest,
    write_workflow_run_state,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _metadata_for_compare(metadata: object) -> dict[str, object]:
    normalized = dict(metadata) if isinstance(metadata, dict) else {}
    normalized.setdefault("legacy_zero_fill_test", False)
    return normalized


def _values_match_for_key(key: str, left: object, right: object) -> bool:
    if key == "metadata":
        return _metadata_for_compare(left) == _metadata_for_compare(right)
    return left == right


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge q-point TBG cRPA chunk artifact directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=Path, action="append", required=True, help="Chunk artifact directory.")
    return parser


def _merge_command(output_dir: Path, chunks: tuple[Path, ...]) -> tuple[str, ...]:
    command: list[str] = [
        "python",
        "-m",
        "mean_field.devtools.merge_tbg_crpa_chunks",
        "--output-dir",
        str(output_dir),
    ]
    for chunk in chunks:
        command.extend(["--chunk", str(chunk)])
    return tuple(command)


def _merge_workflow_manifest(output_dir: Path, chunks: tuple[Path, ...]) -> WorkflowManifest:
    chunk_jobs = tuple(
        WorkflowJobSpec(
            name=f"input_chunk_{index}",
            command=("external", str(chunk)),
            output_dir=chunk,
            metadata={"kind": "input_chunk", "chunk_dir": str(chunk)},
        )
        for index, chunk in enumerate(chunks)
    )
    merge_job = WorkflowJobSpec(
        name="merge",
        command=_merge_command(output_dir, chunks),
        output_dir=output_dir,
        dependencies=tuple(job.name for job in chunk_jobs),
        metadata={"kind": "crpa_merge", "chunk_count": len(chunks)},
    )
    return WorkflowManifest(
        name="tbg_crpa_merge",
        root=output_dir,
        jobs=chunk_jobs + (merge_job,),
        metadata={
            "system": "TBG",
            "workflow": "cRPA merge",
            "chunk_count": len(chunks),
            "slurm_hint": "Run production cRPA merge/diagnostics on compute nodes or through Slurm if inputs are large.",
        },
    )


def _merge_workflow_state(
    manifest: WorkflowManifest,
    merge_status: str,
    *,
    message: str | None = None,
) -> WorkflowRunState:
    slurm_metadata = collect_slurm_metadata()
    merge_metadata = {"slurm": slurm_metadata} if slurm_metadata and merge_status != "pending" else {}
    states: list[WorkflowJobState] = []
    for job in manifest.jobs:
        if job.name.startswith("input_chunk_"):
            states.append(WorkflowJobState(name=job.name, status="succeeded", message="input chunk present"))
        elif job.name == "merge":
            states.append(WorkflowJobState(name=job.name, status=merge_status, message=message, metadata=merge_metadata))
        else:
            states.append(WorkflowJobState(name=job.name, status="pending"))
    state_metadata: dict[str, object] = {"manifest": "workflow_manifest.json"}
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


def _run_merge(args: argparse.Namespace) -> None:
    chunks = [Path(item) for item in args.chunk]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = _load_json(chunks[0] / "crpa_params.json")
    params["metadata"] = _metadata_for_compare(params.get("metadata", {}))
    q_indices_list = []
    q_tilde_real_list = []
    q_tilde_imag_list = []
    chi0_list = []
    epsilon_list = []
    epsilon_inv_list = []
    screened_v_list = []
    effective_list = []
    q_real_list = []
    q_imag_list = []
    q_abs_list = []
    q_abs_nm_inv_list = []
    q_shifts_ref = None

    for chunk in chunks:
        chunk_params = _load_json(chunk / "crpa_params.json")
        comparable_keys = ("theta_deg", "lk", "lg", "q_lg", "bands_per_valley", "eta_mev", "coulomb_params", "metadata")
        for key in comparable_keys:
            if not _values_match_for_key(key, chunk_params.get(key), params.get(key)):
                raise ValueError(f"Chunk {chunk} has incompatible {key}: {chunk_params.get(key)!r} != {params.get(key)!r}")

        with np.load(chunk / "chi0_q.npz") as chi0_npz:
            q_indices_list.append(np.asarray(chi0_npz["q_indices"], dtype=int))
            q_tilde_real_list.append(np.asarray(chi0_npz["q_tilde_real"], dtype=float))
            q_tilde_imag_list.append(np.asarray(chi0_npz["q_tilde_imag"], dtype=float))
            chi0_list.append(np.asarray(chi0_npz["chi0"], dtype=np.complex128))
            q_shifts = np.asarray(chi0_npz["q_shifts"], dtype=int)
            if q_shifts_ref is None:
                q_shifts_ref = q_shifts
            elif not np.array_equal(q_shifts_ref, q_shifts):
                raise ValueError(f"Chunk {chunk} has incompatible q_shifts")

        with np.load(chunk / "dielectric_matrix.npz") as eps_npz:
            epsilon_list.append(np.asarray(eps_npz["epsilon"], dtype=np.complex128))
            epsilon_inv_list.append(np.asarray(eps_npz["epsilon_inv"], dtype=np.complex128))

        with np.load(chunk / "screened_coulomb.npz") as screened_npz:
            screened_v_list.append(np.asarray(screened_npz["screened_v"], dtype=np.complex128))

        with np.load(chunk / "effective_epsilon.npz") as effective_npz:
            effective_list.append(np.asarray(effective_npz["effective_epsilon"], dtype=float))
            q_real_list.append(np.asarray(effective_npz["q_real"], dtype=float))
            q_imag_list.append(np.asarray(effective_npz["q_imag"], dtype=float))
            q_abs_list.append(np.asarray(effective_npz["q_abs"], dtype=float))
            q_abs_nm_inv_list.append(np.asarray(effective_npz["q_abs_nm_inv"], dtype=float))

    if q_shifts_ref is None:
        raise ValueError("No chunks supplied")

    q_indices = np.concatenate(q_indices_list, axis=0)
    order = np.lexsort((q_indices[:, 0], q_indices[:, 1]))
    q_indices = q_indices[order]
    q_tilde_real = np.concatenate(q_tilde_real_list, axis=0)[order]
    q_tilde_imag = np.concatenate(q_tilde_imag_list, axis=0)[order]
    chi0 = np.concatenate(chi0_list, axis=0)[order]
    epsilon = np.concatenate(epsilon_list, axis=0)[order]
    epsilon_inv = np.concatenate(epsilon_inv_list, axis=0)[order]
    screened_v = np.concatenate(screened_v_list, axis=0)[order]
    effective = np.concatenate(effective_list, axis=0)[order]
    q_real = np.concatenate(q_real_list, axis=0)[order]
    q_imag = np.concatenate(q_imag_list, axis=0)[order]
    q_abs = np.concatenate(q_abs_list, axis=0)[order]
    q_abs_nm_inv = np.concatenate(q_abs_nm_inv_list, axis=0)[order]

    params["q_point_count"] = int(q_indices.shape[0])
    params["q_shift_count"] = int(q_shifts_ref.shape[0])
    write_json(output_dir / "crpa_params.json", params)

    _save_npz(
        output_dir / "chi0_q.npz",
        chi0=chi0,
        q_indices=q_indices,
        q_tilde_real=q_tilde_real,
        q_tilde_imag=q_tilde_imag,
        q_shifts=q_shifts_ref,
    )
    _save_npz(
        output_dir / "dielectric_matrix.npz",
        epsilon=epsilon,
        epsilon_inv=epsilon_inv,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
    )
    epsilon_bn = float(params["coulomb_params"]["epsilon_bn"])
    _save_npz(
        output_dir / "effective_epsilon.npz",
        effective_epsilon=effective,
        epsilon_times_bn=effective * epsilon_bn,
        q_abs=q_abs,
        q_abs_nm_inv=q_abs_nm_inv,
        q_real=q_real,
        q_imag=q_imag,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
    )
    _save_npz(
        output_dir / "screened_coulomb.npz",
        screened_v=screened_v,
        effective_epsilon=effective,
        q_indices=q_indices,
        q_shifts=q_shifts_ref,
        q_abs_nm_inv=q_abs_nm_inv,
        q_vectors_real=q_real,
        q_vectors_imag=q_imag,
    )

    np.savetxt(
        output_dir / "epsilon_vs_q.tsv",
        np.column_stack([q_abs.reshape(-1), q_abs_nm_inv.reshape(-1), effective.reshape(-1), (effective * epsilon_bn).reshape(-1)]),
        delimiter="\t",
        header="q_abs_dimless\tq_abs_nm_inv\teffective_epsilon\teffective_epsilon_times_epsilon_bn",
        comments="",
    )

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
    print(f"[crpa-merge] q_point_count={params['q_point_count']} chunks={len(chunks)}", flush=True)
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
    chunks = tuple(Path(item) for item in args.chunk)
    output_dir = Path(args.output_dir)
    manifest = _merge_workflow_manifest(output_dir, chunks)
    _write_merge_workflow_artifacts(
        output_dir,
        manifest,
        _merge_workflow_state(manifest, "running", message="cRPA merge started"),
    )
    try:
        _run_merge(args)
    except Exception as exc:
        _write_merge_workflow_artifacts(
            output_dir,
            manifest,
            _merge_workflow_state(manifest, "failed", message=str(exc)),
        )
        raise
    _write_merge_workflow_artifacts(
        output_dir,
        manifest,
        _merge_workflow_state(manifest, "succeeded", message="cRPA merge outputs written"),
    )

if __name__ == "__main__":
    main()
