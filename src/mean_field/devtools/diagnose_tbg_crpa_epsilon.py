from __future__ import annotations

import argparse
from pathlib import Path

from mean_field.crpa.diagnostics import write_all_epsilon_diagnostics
from mean_field.crpa.workflow import load_crpa_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write diagnostic CSVs and plots for a TBG cRPA epsilon artifact.")
    parser.add_argument("--crpa-dir", type=Path, required=True, help="cRPA artifact directory containing effective_epsilon.npz.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for diagnostics. Defaults to --crpa-dir.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.output_dir if args.output_dir is not None else args.crpa_dir
    result = load_crpa_result(args.crpa_dir)
    summary = write_all_epsilon_diagnostics(result, out)
    print(f"[crpa-diagnostics] wrote diagnostics to {out}", flush=True)
    print(
        "[crpa-diagnostics] checkpoints "
        f"q_peak_nm_inv={summary.q_peak_nm_inv:.6g} "
        f"eps_total_peak={summary.eps_total_peak:.6g} "
        f"eps_total_q0={summary.eps_total_q0:.6g} "
        f"eps_total_q04={summary.eps_total_q04:.6g} "
        f"eps_total_q08={summary.eps_total_q08:.6g} "
        f"eps_total_q12={summary.eps_total_q12:.6g} "
        f"eps_diag_imag_max_abs={summary.eps_diag_imag_max_abs:.6g} "
        f"radial_std_max_0_1p2={summary.radial_std_max_0_1p2:.6g}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
