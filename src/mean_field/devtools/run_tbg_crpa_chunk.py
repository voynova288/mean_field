from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from mean_field.core.io import write_text_artifact
from mean_field.crpa import CRPACoulombParams, write_crpa_outputs
from mean_field.crpa.band_classifier import classify_flat_bands
from mean_field.crpa.bm import read_all_band_bm_solution
from mean_field.crpa.grid import build_uniform_crpa_grid
from mean_field.crpa.plotting import write_epsilon_vs_q_plot
from mean_field.crpa.validation import validation_summary, write_validation_report
from mean_field.crpa.workflow import compute_crpa_from_solution
from mean_field.workflows import (
    WorkflowJobSpec,
    WorkflowJobState,
    WorkflowManifest,
    WorkflowRunState,
    collect_slurm_metadata,
    write_workflow_manifest,
    write_workflow_run_state,
)


def _parse_range(value: str) -> tuple[int, int]:
    pieces = str(value).replace(",", ":").split(":")
    if len(pieces) != 2:
        raise ValueError(f"Expected q range as start:stop, got {value!r}")
    start, stop = int(pieces[0]), int(pieces[1])
    if stop <= start:
        raise ValueError(f"Expected q range stop > start, got {value!r}")
    return start, stop


def _chunk_range(n_items: int, chunk_index: int, chunk_count: int) -> tuple[int, int]:
    if chunk_count <= 0:
        raise ValueError(f"chunk_count must be positive, got {chunk_count}")
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise ValueError(f"chunk_index must be in [0, {chunk_count}), got {chunk_index}")
    start = (n_items * chunk_index) // chunk_count
    stop = (n_items * (chunk_index + 1)) // chunk_count
    return start, stop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute one q-point chunk from a cached all-band BM solution.")
    parser.add_argument("--bm-solution", type=Path, required=True)
    parser.add_argument("--q-lg", type=int, required=True)
    parser.add_argument("--epsilon-bn", type=float, default=4.0)
    parser.add_argument("--ds-angstrom", type=float, default=400.0)
    parser.add_argument("--eta-mev", type=float, default=1.0)
    parser.add_argument("--q-range", default=None, help="Flat q-index range start:stop, stop exclusive.")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--chunk-count", type=int, default=None)
    parser.add_argument(
        "--form-factor-mode",
        choices=("k_periodic_zero_fill", "hf_periodic"),
        default="k_periodic_zero_fill",
        help="Plane-wave form-factor convention for production chunks.",
    )
    parser.add_argument(
        "--hf-compatible",
        action="store_true",
        help="Retained compatibility alias; production chunks are HF-compatible by default.",
    )
    parser.add_argument(
        "--legacy-zero-fill-test",
        action="store_true",
        help="Diagnostic/test only: use the old zhang_zero_fill convention with a non-periodic-G BM cache.",
    )
    parser.add_argument(
        "--occupation-mode",
        choices=("cnp_index", "energy_step"),
        default="cnp_index",
        help="Reference occupation for cRPA. Production Zhang/HF chunks use cnp_index.",
    )
    parser.add_argument(
        "--chi0-energy-mode",
        choices=("bm", "hf_active_flat", "eq19_flat_remote"),
        default="bm",
        help="Band energies/eigenvectors used in chi0. hf_active_flat uses the HF C2T flat basis; eq19_flat_remote applies the Eq.19 flat-band correction.",
    )
    parser.add_argument(
        "--chi0-eq19-overlap-lg",
        type=int,
        default=None,
        help="Optional Q shell for the Eq.19 flat-band correction; defaults to the cached BM lg.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _crpa_chunk_command(args: argparse.Namespace) -> tuple[str, ...]:
    command: list[str] = [
        "python",
        "-m",
        "mean_field.devtools.run_tbg_crpa_chunk",
        "--bm-solution",
        str(args.bm_solution),
        "--q-lg",
        str(int(args.q_lg)),
        "--epsilon-bn",
        f"{float(args.epsilon_bn):.12g}",
        "--ds-angstrom",
        f"{float(args.ds_angstrom):.12g}",
        "--eta-mev",
        f"{float(args.eta_mev):.12g}",
        "--form-factor-mode",
        str(args.form_factor_mode),
        "--occupation-mode",
        str(args.occupation_mode),
        "--chi0-energy-mode",
        str(args.chi0_energy_mode),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.q_range is not None:
        command.extend(["--q-range", str(args.q_range)])
    if args.chunk_index is not None:
        command.extend(["--chunk-index", str(int(args.chunk_index))])
    if args.chunk_count is not None:
        command.extend(["--chunk-count", str(int(args.chunk_count))])
    if args.chi0_eq19_overlap_lg is not None:
        command.extend(["--chi0-eq19-overlap-lg", str(int(args.chi0_eq19_overlap_lg))])
    if bool(args.hf_compatible):
        command.append("--hf-compatible")
    if bool(args.legacy_zero_fill_test):
        command.append("--legacy-zero-fill-test")
    return tuple(command)


def _crpa_chunk_workflow_manifest(args: argparse.Namespace) -> WorkflowManifest:
    output_dir = Path(args.output_dir)
    return WorkflowManifest(
        name="tbg_crpa_chunk",
        root=output_dir,
        jobs=(
            WorkflowJobSpec(
                name="crpa_chunk",
                command=_crpa_chunk_command(args),
                output_dir=output_dir,
                metadata={
                    "kind": "crpa_chunk",
                    "bm_solution": str(args.bm_solution),
                    "q_range": "" if args.q_range is None else str(args.q_range),
                    "chunk_index": None if args.chunk_index is None else int(args.chunk_index),
                    "chunk_count": None if args.chunk_count is None else int(args.chunk_count),
                    "q_lg": int(args.q_lg),
                },
            ),
        ),
        metadata={
            "system": "TBG",
            "workflow": "cRPA chunk",
            "bm_solution": str(args.bm_solution),
            "slurm_hint": "Run production cRPA chunks on compute nodes or through Slurm; do not run heavy chunks on login nodes.",
        },
    )


def _crpa_chunk_workflow_state(
    manifest: WorkflowManifest,
    status: str,
    *,
    message: str | None = None,
) -> WorkflowRunState:
    slurm_metadata = collect_slurm_metadata()
    job_metadata = {"slurm": slurm_metadata} if slurm_metadata and status != "pending" else {}
    state_metadata: dict[str, object] = {"manifest": "workflow_manifest.json"}
    if slurm_metadata:
        state_metadata["slurm"] = slurm_metadata
    return WorkflowRunState(
        name=manifest.name,
        jobs=(
            WorkflowJobState(
                name="crpa_chunk",
                status=status,
                message=message,
                metadata=job_metadata,
            ),
        ),
        metadata=state_metadata,
    )


def _write_crpa_chunk_workflow_artifacts(
    output_dir: Path,
    manifest: WorkflowManifest,
    state: WorkflowRunState,
) -> None:
    write_workflow_manifest(manifest, output_dir / "workflow_manifest.json")
    write_workflow_run_state(state, output_dir / "workflow_run_state.json")
    write_text_artifact(state.to_markdown() + "\n", output_dir / "workflow_run_state.md")


def _run_crpa_chunk(args: argparse.Namespace) -> None:
    solution = read_all_band_bm_solution(args.bm_solution)
    lk_float = math.sqrt(float(solution.nk))
    lk = int(round(lk_float))
    if lk * lk != solution.nk:
        raise ValueError(f"Cached BM solution nk={solution.nk} is not a square k grid")
    grid = build_uniform_crpa_grid(solution.params, lk)
    if not np.allclose(grid.kvec, solution.lattice_kvec):
        raise ValueError("Cached BM k vectors do not match the reconstructed CRPA grid")

    if args.q_range is not None:
        start, stop = _parse_range(args.q_range)
    elif args.chunk_index is not None and args.chunk_count is not None:
        start, stop = _chunk_range(grid.nk, int(args.chunk_index), int(args.chunk_count))
    else:
        start, stop = 0, grid.nk
    start = max(0, int(start))
    stop = min(grid.nk, int(stop))
    if stop <= start:
        raise ValueError(f"Empty q chunk after clipping: {start}:{stop}")

    q_indices = [grid.unravel_index(iq) for iq in range(start, stop)]
    if args.legacy_zero_fill_test and args.hf_compatible:
        raise ValueError("--legacy-zero-fill-test cannot be combined with --hf-compatible.")
    if args.legacy_zero_fill_test:
        if bool(solution.periodic_g_grid):
            raise ValueError("Legacy zero-fill test chunks require a non-periodic-G BM cache.")
        form_factor_mode = "zhang_zero_fill"
    else:
        form_factor_mode = str(args.form_factor_mode)
        if not bool(solution.periodic_g_grid):
            raise ValueError("Production cRPA chunks require a BM cache prepared with periodic_g_grid=True.")
    classification = classify_flat_bands(solution.spectrum, method="center")
    coulomb = CRPACoulombParams(epsilon_bn=float(args.epsilon_bn), ds_angstrom=float(args.ds_angstrom))
    bands_per_valley = None if solution.nb == solution.basis_dimension else int(solution.nb)
    theta_deg = float(solution.params.dtheta_rad) * 180.0 / math.pi

    print(
        "[crpa-chunk] start "
        f"bm={args.bm_solution} lk={lk} lg={solution.lg} q_lg={args.q_lg} "
        f"q_range={start}:{stop} q_points={len(q_indices)} form_factor_mode={form_factor_mode} "
        f"occupation_mode={args.occupation_mode} chi0_energy_mode={args.chi0_energy_mode} "
        f"legacy_zero_fill_test={str(args.legacy_zero_fill_test).lower()}",
        flush=True,
    )
    result = compute_crpa_from_solution(
        solution,
        classification,
        grid,
        theta_deg=theta_deg,
        q_lg=int(args.q_lg),
        bands_per_valley=bands_per_valley,
        q_indices=q_indices,
        coulomb_params=coulomb,
        eta_mev=float(args.eta_mev),
        form_factor_mode=form_factor_mode,
        allow_legacy_zero_fill_test=bool(args.legacy_zero_fill_test),
        occupation_mode=str(args.occupation_mode),
        flat_method="center",
        chi0_energy_mode=str(args.chi0_energy_mode),
        chi0_eq19_overlap_lg=args.chi0_eq19_overlap_lg,
    )
    out = write_crpa_outputs(result, args.output_dir)
    write_epsilon_vs_q_plot(result, out / "epsilon_vs_q.pdf")
    report_path = write_validation_report(result, out / "validation_report.md")
    summary = validation_summary(result)
    print(f"[crpa-chunk] wrote outputs to {out}", flush=True)
    print(f"[crpa-chunk] report={report_path}", flush=True)
    print(
        "[crpa-chunk] summary "
        f"eps_times_bn_max={summary['effective_epsilon_times_bn_max']:.6g} "
        f"q_points={len(q_indices)}",
        flush=True,
    )



def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    manifest = _crpa_chunk_workflow_manifest(args)
    _write_crpa_chunk_workflow_artifacts(
        output_dir,
        manifest,
        _crpa_chunk_workflow_state(manifest, "running", message="cRPA chunk started"),
    )
    try:
        _run_crpa_chunk(args)
    except Exception as exc:
        _write_crpa_chunk_workflow_artifacts(
            output_dir,
            manifest,
            _crpa_chunk_workflow_state(manifest, "failed", message=str(exc)),
        )
        raise
    _write_crpa_chunk_workflow_artifacts(
        output_dir,
        manifest,
        _crpa_chunk_workflow_state(manifest, "succeeded", message="cRPA chunk outputs written"),
    )

if __name__ == "__main__":
    main()
