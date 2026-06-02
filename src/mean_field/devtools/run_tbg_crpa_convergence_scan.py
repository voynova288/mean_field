from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from mean_field.crpa import CRPACoulombParams, compute_crpa, write_crpa_outputs
from mean_field.crpa.plotting import write_epsilon_vs_q_plot
from mean_field.crpa.validation import compute_c1_cross_check, validation_summary, write_validation_report
from mean_field.systems.tbg.params import TBGParameters


def _parse_case(value: str) -> dict[str, int | None | str]:
    pieces = str(value).split(":")
    if len(pieces) not in {4, 5}:
        raise ValueError(f"Expected case as label:lk:lg:q_lg[:bands], got {value!r}")
    label, lk, lg, q_lg = pieces[:4]
    bands = None if len(pieces) == 4 or pieces[4].lower() in {"none", "all", ""} else int(pieces[4])
    return {
        "label": label,
        "lk": int(lk),
        "lg": int(lg),
        "q_lg": int(q_lg),
        "bands_per_valley": bands,
    }


def _window_stats(result) -> dict[str, float]:
    q_nm = np.abs(result.physical_q_vectors.reshape(-1)) / (float(result.coulomb_params.graphene_lattice_angstrom) / 10.0)
    eps_bn = np.real(result.effective_epsilon.reshape(-1)) * float(result.coulomb_params.epsilon_bn)
    stats: dict[str, float] = {
        "eps_bn_min_all": float(np.min(eps_bn)),
        "eps_bn_median_all": float(np.median(eps_bn)),
        "eps_bn_max_all": float(np.max(eps_bn)),
    }
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.65), (0.65, 1.0), (1.0, 2.0)):
        mask = (q_nm >= lo) & (q_nm < hi)
        key = f"{lo:.2f}_{hi:.2f}_nm_inv".replace(".", "p")
        stats[f"count_{key}"] = int(np.count_nonzero(mask))
        if np.any(mask):
            stats[f"eps_bn_min_{key}"] = float(np.min(eps_bn[mask]))
            stats[f"eps_bn_median_{key}"] = float(np.median(eps_bn[mask]))
            stats[f"eps_bn_max_{key}"] = float(np.max(eps_bn[mask]))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run serial HF-compatible cRPA convergence cases for TBG epsilon(q).")
    parser.add_argument("--theta-deg", type=float, default=1.05)
    parser.add_argument("--vf", type=float, default=2135.4)
    parser.add_argument("--w0", type=float, default=79.7)
    parser.add_argument("--w1", type=float, default=97.4)
    parser.add_argument("--epsilon-bn", type=float, default=4.0)
    parser.add_argument("--ds-angstrom", type=float, default=400.0)
    parser.add_argument("--eta-mev", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--periodic-g-grid",
        action="store_true",
        help="Retained for CLI compatibility. Production cRPA uses periodic G-grid by default.",
    )
    parser.add_argument(
        "--legacy-zero-fill-test",
        action="store_true",
        help="Diagnostic/test only: run the old zhang_zero_fill, non-periodic-G convention.",
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
        help="Optional Q shell for the Eq.19 flat-band correction; defaults to each BM lg.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Case as label:lk:lg:q_lg[:bands]. Can be repeated.",
    )
    parser.add_argument("--check-c1-first", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.legacy_zero_fill_test and args.periodic_g_grid:
        raise ValueError("--legacy-zero-fill-test cannot be combined with --periodic-g-grid.")
    cases = args.case or [
        "lk6_lg5_b40:6:5:5:40",
        "lk8_lg5_b40:8:5:5:40",
        "lk10_lg5_b40:10:5:5:40",
    ]
    parsed_cases = [_parse_case(item) for item in cases]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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
    rows: list[dict[str, object]] = []

    for index, case in enumerate(parsed_cases):
        label = str(case["label"])
        print(f"[scan] start {label}: lk={case['lk']} lg={case['lg']} q_lg={case['q_lg']} bands={case['bands_per_valley']}", flush=True)
        start = time.perf_counter()
        periodic_g_grid = not bool(args.legacy_zero_fill_test)
        form_factor_mode = "zhang_zero_fill" if args.legacy_zero_fill_test else "k_periodic_zero_fill"
        result = compute_crpa(
            params,
            theta_deg=float(args.theta_deg),
            lk=int(case["lk"]),
            lg=int(case["lg"]),
            q_lg=int(case["q_lg"]),
            bands_per_valley=case["bands_per_valley"],
            coulomb_params=coulomb,
            eta_mev=float(args.eta_mev),
            sigma_rotation=True,
            periodic_g_grid=periodic_g_grid,
            form_factor_mode=form_factor_mode,
            allow_legacy_zero_fill_test=bool(args.legacy_zero_fill_test),
            chi0_energy_mode=str(args.chi0_energy_mode),
            chi0_eq19_overlap_lg=args.chi0_eq19_overlap_lg,
        )
        elapsed = time.perf_counter() - start
        out = output_root / label
        write_crpa_outputs(result, out)
        write_epsilon_vs_q_plot(result, out / "epsilon_vs_q.pdf")
        extra_checks = None
        if args.check_c1_first and index == 0:
            extra_checks = compute_c1_cross_check(
                params,
                lk=int(case["lk"]),
                lg=int(case["lg"]),
                q_lg=int(case["q_lg"]),
                bands_per_valley=case["bands_per_valley"],
                q_index=(1, 0),
                eta_mev=float(args.eta_mev),
                periodic_g_grid=periodic_g_grid,
                form_factor_mode=form_factor_mode,
                allow_legacy_zero_fill_test=bool(args.legacy_zero_fill_test),
            )
            (out / "c1_cross_check.json").write_text(json.dumps(extra_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_validation_report(result, out / "validation_report.md", extra_checks=extra_checks)
        summary = validation_summary(result)
        row: dict[str, object] = {
            "label": label,
            "lk": int(case["lk"]),
            "lg": int(case["lg"]),
            "q_lg": int(case["q_lg"]),
            "bands_per_valley": -1 if case["bands_per_valley"] is None else int(case["bands_per_valley"]),
            "elapsed_sec": float(elapsed),
            **summary,
            **_window_stats(result),
        }
        if extra_checks:
            row.update(extra_checks)
        rows.append(row)
        print(
            f"[scan] done {label}: eps_bn_max={row['effective_epsilon_times_bn_max']:.6g} "
            f"window_0p40_0p65_median={row.get('eps_bn_median_0p40_0p65_nm_inv', float('nan')):.6g} "
            f"elapsed_sec={elapsed:.3f}",
            flush=True,
        )

    summary_path = output_root / "scan_summary.tsv"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(keys) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(key, "")) for key in keys) + "\n")
    (output_root / "scan_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[scan] wrote summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
