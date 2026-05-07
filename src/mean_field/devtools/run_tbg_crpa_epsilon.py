from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mean_field.crpa import CRPACoulombParams, compute_crpa, write_crpa_outputs
from mean_field.crpa.plotting import write_epsilon_vs_q_plot
from mean_field.crpa.validation import compute_c1_cross_check, validation_summary, write_validation_report
from mean_field.crpa.workflow import default_crpa_output_dir
from mean_field.systems.tbg.params import TBGParameters


def _q_indices_for_run(lk: int, stride: int, max_q_points: int | None) -> list[tuple[int, int]]:
    stride = int(stride)
    if stride <= 0:
        raise ValueError(f"q stride must be positive, got {stride}")
    coords = [(i, j) for j in range(int(lk)) for i in range(int(lk)) if i % stride == 0 and j % stride == 0]
    if max_q_points is not None:
        coords = coords[: int(max_q_points)]
    return coords


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Zhang-style cRPA epsilon(q) for zero-field TBG.")
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--vf", type=float, default=2135.4, help="Dirac velocity parameter in meV.")
    parser.add_argument("--w0", type=float, default=79.7, help="AA tunneling in meV.")
    parser.add_argument("--w1", type=float, default=97.4, help="AB tunneling in meV.")
    parser.add_argument("--lk", type=int, default=6, help="Periodic cRPA k mesh, lk x lk.")
    parser.add_argument("--lg", type=int, default=3, help="BM plane-wave cutoff grid, lg x lg.")
    parser.add_argument("--q-lg", type=int, default=3, help="Q-vector dielectric matrix cutoff grid.")
    parser.add_argument(
        "--bands-per-valley",
        type=int,
        default=None,
        help="Centered BM band window. Omit to keep all bands for the chosen lg.",
    )
    parser.add_argument("--epsilon-bn", type=float, default=4.0)
    parser.add_argument("--ds-angstrom", type=float, default=400.0)
    parser.add_argument("--eta-mev", type=float, default=1.0)
    parser.add_argument(
        "--periodic-g-grid",
        action="store_true",
        help="Use the legacy B0/Jacobian benchmark periodic G-grid tunneling convention instead of physical zero-fill.",
    )
    parser.add_argument("--q-stride", type=int, default=1)
    parser.add_argument("--max-q-points", type=int, default=None)
    parser.add_argument("--check-c1", action="store_true", help="Run direct constrained vs full-minus-flat-flat chi0 check.")
    parser.add_argument("--c1-q-index", default="1,0", help="q-grid coordinate for --check-c1, e.g. 1,0.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _parse_q_index(value: str) -> tuple[int, int]:
    pieces = str(value).replace(":", ",").split(",")
    if len(pieces) != 2:
        raise ValueError(f"Expected q index as i,j, got {value!r}")
    return int(pieces[0]), int(pieces[1])


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else default_crpa_output_dir(Path("outputs"))
    q_indices = _q_indices_for_run(args.lk, args.q_stride, args.max_q_points)

    params = TBGParameters.from_degrees(
        args.theta_deg,
        vf=float(args.vf),
        w0=float(args.w0),
        w1=float(args.w1),
        strain=0.0,
        alpha=0.5,
        deformation_potential=0.0,
    )
    coulomb = CRPACoulombParams(epsilon_bn=float(args.epsilon_bn), ds_angstrom=float(args.ds_angstrom))

    print(
        "[crpa] starting "
        f"theta={args.theta_deg} lk={args.lk} lg={args.lg} q_lg={args.q_lg} "
        f"bands_per_valley={args.bands_per_valley} q_points={len(q_indices)}",
        flush=True,
    )
    start = time.perf_counter()
    result = compute_crpa(
        params,
        theta_deg=float(args.theta_deg),
        lk=int(args.lk),
        lg=int(args.lg),
        q_lg=int(args.q_lg),
        bands_per_valley=args.bands_per_valley,
        q_indices=q_indices,
        coulomb_params=coulomb,
        eta_mev=float(args.eta_mev),
        sigma_rotation=True,
        periodic_g_grid=bool(args.periodic_g_grid),
    )
    elapsed = time.perf_counter() - start

    out = write_crpa_outputs(result, output_dir)
    plot_path = write_epsilon_vs_q_plot(result, out / "epsilon_vs_q.pdf")
    extra_checks = None
    if args.check_c1:
        print(f"[crpa] running C1 check at q_index={args.c1_q_index}", flush=True)
        extra_checks = compute_c1_cross_check(
            params,
            lk=int(args.lk),
            lg=int(args.lg),
            q_lg=int(args.q_lg),
            bands_per_valley=args.bands_per_valley,
            q_index=_parse_q_index(args.c1_q_index),
            eta_mev=float(args.eta_mev),
            sigma_rotation=True,
        )
        (out / "c1_cross_check.json").write_text(json.dumps(extra_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = write_validation_report(result, out / "validation_report.md", extra_checks=extra_checks)
    summary = validation_summary(result)
    np.savetxt(
        out / "epsilon_vs_q.tsv",
        np.column_stack(
            [
                np.abs(result.physical_q_vectors.reshape(-1)),
                np.abs(result.physical_q_vectors.reshape(-1)) / (float(result.coulomb_params.graphene_lattice_angstrom) / 10.0),
                result.effective_epsilon.reshape(-1),
                result.effective_epsilon.reshape(-1) * float(result.coulomb_params.epsilon_bn),
            ]
        ),
        delimiter="\t",
        header="q_abs_dimless\tq_abs_nm_inv\teffective_epsilon\teffective_epsilon_times_epsilon_bn",
        comments="",
    )
    print(f"[crpa] wrote outputs to {out}", flush=True)
    print(f"[crpa] plot={plot_path}", flush=True)
    print(f"[crpa] report={report_path}", flush=True)
    print(
        "[crpa] summary "
        f"eps_min={summary['effective_epsilon_min']:.6g} "
        f"eps_max={summary['effective_epsilon_max']:.6g} "
        f"eps_times_bn_max={summary['effective_epsilon_times_bn_max']:.6g} "
        f"elapsed_sec={elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
