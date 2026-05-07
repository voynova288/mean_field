from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SAMPLE_Q = (0.4, 0.5, 0.540481454, 0.6, 0.8, 1.0, 1.2)
DEFAULT_WINDOWS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.65), (0.65, 1.0), (1.0, 1.2), (1.0, 2.0))


def _parse_case(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    elif ":" in value:
        label, path = value.split(":", 1)
    else:
        path = value
        label = Path(value).name
    label = label.strip()
    if not label:
        raise ValueError(f"Empty case label in {value!r}")
    return label, Path(path)


def _parse_float_list(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    return tuple(float(piece) for piece in value.replace(":", ",").split(",") if piece.strip())


def _parse_windows(value: str) -> tuple[tuple[float, float], ...]:
    if not value:
        return ()
    windows: list[tuple[float, float]] = []
    for item in value.split(","):
        lo_hi = item.split(":")
        if len(lo_hi) != 2:
            raise ValueError(f"Expected window as lo:hi, got {item!r}")
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if hi <= lo:
            raise ValueError(f"Expected window hi > lo, got {item!r}")
        windows.append((lo, hi))
    return tuple(windows)


def _parse_anchor(value: str) -> tuple[float, float]:
    pieces = value.replace(",", ":").split(":")
    if len(pieces) != 2:
        raise ValueError(f"Expected anchor as q:epsilon, got {value!r}")
    return float(pieces[0]), float(pieces[1])


def _load_paper_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter="\t", comments="#", skiprows=1)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected two-column paper curve table, got {path}")
    q = np.asarray(arr[:, 0], dtype=float)
    eps = np.asarray(arr[:, 1], dtype=float)
    order = np.argsort(q)
    return q[order], eps[order]


def _case_npz_path(path: Path) -> Path:
    if path.is_dir():
        return path / "effective_epsilon.npz"
    return path


def _median_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path = _case_npz_path(path)
    with np.load(npz_path) as data:
        q = np.asarray(data["q_abs_nm_inv"], dtype=float).reshape(-1)
        eps = np.asarray(data["epsilon_times_bn"], dtype=float).reshape(-1)
    mask = np.isfinite(q) & np.isfinite(eps)
    q = q[mask]
    eps = eps[mask]
    rounded_q = np.round(q, 12)
    xs: list[float] = []
    ys: list[float] = []
    for value in np.unique(rounded_q):
        same_q = rounded_q == value
        xs.append(float(np.median(q[same_q])))
        ys.append(float(np.median(eps[same_q])))
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order], q, eps


def _write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(item) for item in row) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TBG cRPA epsilon(q) artifacts to a digitized Zhang Fig. 1(e) curve.")
    parser.add_argument("--paper-curve", type=Path, required=True, help="Two-column TSV: q_nm_inv and epsilon_times_bn.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Case as label=path, label:path, or just path. Path can be an artifact dir or effective_epsilon.npz.",
    )
    parser.add_argument("--sample-q", default=",".join(str(q) for q in DEFAULT_SAMPLE_Q))
    parser.add_argument(
        "--windows",
        default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_WINDOWS),
        help="Comma-separated q windows in nm^-1, formatted lo:hi.",
    )
    parser.add_argument("--x-max", type=float, default=1.2)
    parser.add_argument("--y-min", type=float, default=3.5)
    parser.add_argument("--y-max", type=float, default=21.5)
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        help="Optional visual/reference anchor as q_nm_inv:epsilon_times_bn. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_q, paper_eps = _load_paper_curve(Path(args.paper_curve))
    sample_q = _parse_float_list(args.sample_q)
    windows = _parse_windows(args.windows)
    anchors = [_parse_anchor(value) for value in args.anchor]
    cases = [_parse_case(item) for item in args.case]

    metrics_rows: list[tuple[object, ...]] = []
    point_rows: list[tuple[object, ...]] = []
    window_rows: list[tuple[object, ...]] = []
    anchor_rows: list[tuple[object, ...]] = []
    curves: list[tuple[str, np.ndarray, np.ndarray]] = []

    for label, case_path in cases:
        q_curve, eps_curve, q_raw, eps_raw = _median_curve(case_path)
        curves.append((label, q_curve, eps_curve))
        interp = np.interp(paper_q, q_curve, eps_curve)
        diff = interp - paper_eps
        metrics_rows.append(
            (
                label,
                str(case_path),
                float(np.sqrt(np.mean(diff * diff))),
                float(np.mean(diff)),
                float(np.max(np.abs(diff))),
                float(np.median(diff)),
            )
        )

        for q_value in sample_q:
            paper_value = float(np.interp(q_value, paper_q, paper_eps))
            computed_value = float(np.interp(q_value, q_curve, eps_curve))
            point_rows.append((label, q_value, paper_value, computed_value, computed_value - paper_value))

        for q_anchor, eps_anchor in anchors:
            computed_value = float(np.interp(q_anchor, q_curve, eps_curve))
            anchor_rows.append((label, q_anchor, eps_anchor, computed_value, computed_value - eps_anchor))

        for lo, hi in windows:
            in_window = (q_raw >= lo) & (q_raw < hi)
            if not np.any(in_window):
                window_rows.append((label, lo, hi, "", "", "", 0))
                continue
            values = eps_raw[in_window]
            window_rows.append(
                (
                    label,
                    lo,
                    hi,
                    float(np.min(values)),
                    float(np.median(values)),
                    float(np.max(values)),
                    int(values.size),
                )
            )

    _write_tsv(
        output_dir / "comparison_metrics.tsv",
        ("label", "path", "rmse_vs_digitized", "mean_diff", "max_abs_diff", "median_diff"),
        metrics_rows,
    )
    _write_tsv(
        output_dir / "comparison_at_q.tsv",
        ("label", "q_nm_inv", "paper_eps_times_bn", "computed_eps_times_bn", "diff"),
        point_rows,
    )
    _write_tsv(
        output_dir / "window_stats.tsv",
        ("label", "q_lo_nm_inv", "q_hi_nm_inv", "min_eps_times_bn", "median_eps_times_bn", "max_eps_times_bn", "n_points"),
        window_rows,
    )
    if anchors:
        _write_tsv(
            output_dir / "anchor_comparison.tsv",
            ("label", "q_nm_inv", "anchor_eps_times_bn", "computed_eps_times_bn", "diff"),
            anchor_rows,
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(paper_q, paper_eps, color="black", lw=2.0, label="digitized Zhang Fig. 1(e)")
    for idx, (q_anchor, eps_anchor) in enumerate(anchors):
        label = "reference anchor" if idx == 0 else None
        ax.axvline(q_anchor, color="tab:blue", lw=0.8, alpha=0.65)
        ax.axhline(eps_anchor, color="tab:blue", lw=0.8, alpha=0.65)
        ax.scatter([q_anchor], [eps_anchor], color="tab:blue", s=18, zorder=5, label=label)
    for label, q_curve, eps_curve in curves:
        ax.plot(q_curve, eps_curve, marker="o", ms=3.5, lw=1.2, label=label)
    ax.set_xlim(0.0, float(args.x_max))
    ax.set_ylim(float(args.y_min), float(args.y_max))
    ax.set_xlabel(r"$|q|$ (nm$^{-1}$)")
    ax.set_ylabel(r"$\epsilon(q)\,\epsilon_{BN}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / "epsilon_vs_q_digitized_overlay.pdf")
    fig.savefig(output_dir / "epsilon_vs_q_digitized_overlay.png", dpi=180)
    plt.close(fig)

    summary_lines = [
        "# Zhang Fig. 1(e) comparison",
        "",
        "## Metrics versus digitized paper curve",
        "",
        "| case | RMSE | mean diff | max abs diff | median diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, _path, rmse, mean_diff, max_abs_diff, median_diff in metrics_rows:
        summary_lines.append(f"| {label} | {rmse:.6g} | {mean_diff:.6g} | {max_abs_diff:.6g} | {median_diff:.6g} |")
    summary_lines.extend(
        [
            "",
            "## Files",
            "",
            "- `comparison_metrics.tsv`",
            "- `comparison_at_q.tsv`",
            "- `window_stats.tsv`",
        ]
    )
    if anchors:
        summary_lines.append("- `anchor_comparison.tsv`")
    summary_lines.extend(
        [
            "- `epsilon_vs_q_digitized_overlay.pdf`",
            "- `epsilon_vs_q_digitized_overlay.png`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
