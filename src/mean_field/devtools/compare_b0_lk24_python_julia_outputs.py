from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathBandData:
    labels: tuple[str, ...]
    kdist: tuple[float, ...]
    energies: tuple[tuple[float, ...], ...]


def _read_path_tsv(path: Path) -> PathBandData:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return PathBandData(
        labels=tuple(header[1:]),
        kdist=tuple(float(row[0]) for row in rows),
        energies=tuple(tuple(float(value) for value in row[1:]) for row in rows),
    )


def _find_one_path(pattern_root: Path, pattern: str) -> Path:
    matches = sorted(pattern_root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match under {pattern_root} for {pattern}, found {len(matches)}")
    return matches[0]


def _compare_paths(reference_path: Path, computed_path: Path) -> dict[str, str]:
    reference = _read_path_tsv(reference_path)
    computed = _read_path_tsv(computed_path)
    if len(reference.kdist) != len(computed.kdist):
        raise ValueError(f"k-point count mismatch: {reference_path} has {len(reference.kdist)}, {computed_path} has {len(computed.kdist)}")
    if len(reference.energies[0]) != len(computed.energies[0]):
        raise ValueError(
            f"band count mismatch: {reference_path} has {len(reference.energies[0])}, {computed_path} has {len(computed.energies[0])}"
        )
    kdist_max = max(abs(a - b) for a, b in zip(reference.kdist, computed.kdist, strict=True))
    diffs: list[float] = []
    for ref_row, got_row in zip(reference.energies, computed.energies, strict=True):
        diffs.extend(a - b for a, b in zip(sorted(ref_row), sorted(got_row), strict=True))
    abs_diffs = [abs(value) for value in diffs]
    rms = (sum(value * value for value in diffs) / len(diffs)) ** 0.5
    mean_abs = sum(abs_diffs) / len(abs_diffs)
    return {
        "kdist_max_abs_diff": f"{kdist_max:.16e}",
        "max_abs_band_diff_mev": f"{max(abs_diffs):.16e}",
        "rms_band_diff_mev": f"{rms:.16e}",
        "mean_abs_band_diff_mev": f"{mean_abs:.16e}",
        "energy_sorting": "ascending_per_k",
    }


def _read_case_ids(manifest_path: Path) -> tuple[str, ...]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return tuple(row["benchmark_id"] for row in reader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Python lk=24 B0 HF outputs against same-lk Julia B0 reference outputs.")
    parser.add_argument("--python-root", type=Path, required=True, help="Mean_Field result root containing <case>/computed_hf_path.tsv")
    parser.add_argument("--julia-root", type=Path, required=True, help="TBG_HartreeFock result root containing <case>/path_bands/*_hf_path.tsv")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/data/home/ziyuzhu/Mean_Field/benchmarks/b0/benchmark_manifest.tsv"),
        help="Mean_Field B0 benchmark manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for comparison summary files. Defaults to python-root.")
    args = parser.parse_args(argv)

    python_root = args.python_root.resolve()
    julia_root = args.julia_root.resolve()
    output_dir = python_root if args.output_dir is None else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "python_to_julia_lk24_summary.tsv"
    report_path = output_dir / "python_to_julia_lk24_report.md"
    case_ids = _read_case_ids(args.manifest)

    rows: list[dict[str, str]] = []
    for benchmark_id in case_ids:
        python_path = python_root / benchmark_id / "computed_hf_path.tsv"
        julia_path = _find_one_path(julia_root / benchmark_id / "path_bands", "*_hf_path.tsv")
        if not python_path.is_file():
            raise FileNotFoundError(f"Missing Python path output: {python_path}")
        metrics = _compare_paths(julia_path, python_path)
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "python_path_tsv": str(python_path),
                "julia_path_tsv": str(julia_path),
                **metrics,
            }
        )

    fieldnames = [
        "benchmark_id",
        "python_path_tsv",
        "julia_path_tsv",
        "kdist_max_abs_diff",
        "max_abs_band_diff_mev",
        "rms_band_diff_mev",
        "mean_abs_band_diff_mev",
        "energy_sorting",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    max_band = max(float(row["max_abs_band_diff_mev"]) for row in rows) if rows else 0.0
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# B0 lk=24 Python-vs-Julia HF comparison\n\n")
        handle.write(f"- python_root: `{python_root}`\n")
        handle.write(f"- julia_root: `{julia_root}`\n")
        handle.write(f"- summary_tsv: `{summary_path}`\n")
        handle.write(f"- max_abs_band_diff_mev: `{max_band:.16e}`\n\n")
        handle.write("This comparison uses same-angle, same-filling, same-lk Julia B0 outputs as the reference.\n")

    print(f"summary_tsv={summary_path}")
    print(f"report_md={report_path}")
    print(f"max_abs_band_diff_mev={max_band:.16e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
