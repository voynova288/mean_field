from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from mean_field.core.lattice import KPath
from mean_field.systems.tmbg import PathBandsResult, TMBGBandPlotPanel, write_tmbg_paper_band_figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redraw a saved tMBG Fig. 2-like path-band dataset inside a fixed energy window without rerunning the solver."
    )
    parser.add_argument("--source-root", type=Path, required=True, help="Directory that contains delta_*/bands_path.npz.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If omitted, write the redraw files back into --source-root.",
    )
    parser.add_argument("--window-mev", type=float, default=100.0, help="Half-window in meV for the y-axis.")
    parser.add_argument("--stem", type=str, default="fig2_like_bands_pm100mev")
    return parser.parse_args()


def _delta_folder_order(folder_name: str) -> tuple[int, float]:
    token = folder_name.removeprefix("delta_").removesuffix("mev")
    sign = 1.0
    if token.startswith("+"):
        token = token[1:]
    elif token.startswith("-"):
        token = token[1:]
        sign = -1.0
    delta_mev = sign * float(token)
    if abs(delta_mev) < 1.0e-12:
        return (0, 0.0)
    if delta_mev > 0.0:
        return (1, delta_mev)
    return (2, abs(delta_mev))


def _delta_label(folder_name: str) -> str:
    token = folder_name.removeprefix("delta_").removesuffix("mev")
    if token == "+000":
        return "Δ = 0 meV"
    if token.startswith("+"):
        return f"Δ = {token} meV"
    return f"Δ = {token} meV"


def _load_run_metadata(source_root: Path) -> dict[str, object]:
    metadata_path = source_root / "run_metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _resolve_node_indices(n_points: int, n_labels: int, run_metadata: dict[str, object]) -> tuple[int, ...]:
    params = run_metadata.get("parameters")
    if isinstance(params, dict):
        points_per_segment = params.get("points_per_segment")
        if isinstance(points_per_segment, int) and points_per_segment > 0:
            node_indices = tuple(1 + i * points_per_segment for i in range(n_labels))
            if node_indices and node_indices[-1] == n_points:
                return node_indices

    n_segments = n_labels - 1
    if n_segments <= 0 or (n_points - 1) % n_segments != 0:
        raise ValueError(
            "Cannot infer K-path node indices from the saved bands_path.npz. "
            f"n_points={n_points}, n_labels={n_labels}."
        )
    points_per_segment = (n_points - 1) // n_segments
    return tuple(1 + i * points_per_segment for i in range(n_labels))


def _load_panel(source_root: Path, folder: Path, run_metadata: dict[str, object]) -> tuple[TMBGBandPlotPanel, dict[str, object]]:
    data = np.load(folder / "bands_path.npz", allow_pickle=True)
    k_distance = np.asarray(data["k_distance"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    kvec_xy = np.asarray(data["kvec_nm_inv"], dtype=float)
    band_indices = np.asarray(data["band_indices"], dtype=int)
    flat_band_indices = np.asarray(data["flat_band_indices"], dtype=int)
    labels = tuple(str(value) for value in np.asarray(data["k_labels"], dtype=object).tolist())

    node_indices = _resolve_node_indices(k_distance.size, len(labels), run_metadata)
    kvec = np.asarray(kvec_xy[:, 0] + 1j * kvec_xy[:, 1], dtype=np.complex128)
    path = KPath(
        kvec=kvec,
        kdist=k_distance,
        labels=labels,
        node_indices=node_indices,
    )

    band_lookup = {int(index): ilocal for ilocal, index in enumerate(band_indices.tolist())}
    local_flat = tuple(int(band_lookup[int(index)]) for index in flat_band_indices.tolist())
    conduction = energies[:, local_flat[1]]
    valence = energies[:, local_flat[0]]
    gap_values = conduction - valence
    gap_index = int(np.argmin(gap_values))

    path_result = PathBandsResult(path=path, energies=energies)
    panel = TMBGBandPlotPanel(
        label=_delta_label(folder.name),
        path_result=path_result,
        flat_band_indices=local_flat,
        annotation=f"flat_gap: {gap_values[gap_index] * 1.0e3:.2f} meV",
    )
    panel_metadata = {
        "folder": folder.name,
        "label": panel.label,
        "energy_min_mev": float(np.min(energies) * 1.0e3),
        "energy_max_mev": float(np.max(energies) * 1.0e3),
        "band_indices": [int(index) for index in band_indices.tolist()],
        "flat_band_indices_global": [int(index) for index in flat_band_indices.tolist()],
        "flat_band_indices_local": [int(index) for index in local_flat],
        "flat_gap_mev": float(gap_values[gap_index] * 1.0e3),
    }
    return panel, panel_metadata


def _write_markdown(
    output_dir: Path,
    *,
    stem: str,
    source_root: Path,
    window_mev: float,
    panels: list[dict[str, object]],
) -> Path:
    note_path = output_dir / f"{stem}_note.md"
    lines = [
        f"# {stem}",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "这组文件：",
        "",
        f"- `{stem}.png`",
        f"- `{stem}.pdf`",
        f"- `{stem}_note.md`",
        f"- `plot_metadata.json`",
        "",
        "是基于当前目录中已保存的 `bands_path.npz` 轻量重画得到的 `±100 meV` 窗口能带图。",
        "",
        "## 说明",
        "",
        f"- 源目录：`{source_root}`",
        "- 这次只读取已有路径能带数据并重画，没有重新运行 tMBG 求解或拓扑计算。",
        f"- y 轴窗口固定为 `[-{window_mev:.0f}, +{window_mev:.0f}] meV`。",
        "- 面板顺序保持为 `Δ = 0, +60, -40 meV`。",
        "",
        "## 面板摘要",
        "",
    ]
    for panel in panels:
        lines.extend(
            [
                f"### {panel['label']}",
                "",
                f"- 来源子目录：`{panel['folder']}`",
                f"- 保存的 band index：`{panel['band_indices']}`",
                f"- 平带全局 index：`{panel['flat_band_indices_global']}`",
                f"- 平带局部 index：`{panel['flat_band_indices_local']}`",
                f"- 平带最小 gap：`{panel['flat_gap_mev']:.3f} meV`",
                f"- 当前保存带窗口的能量范围：`[{panel['energy_min_mev']:.3f}, {panel['energy_max_mev']:.3f}] meV`",
                "",
            ]
        )
    note_path.write_text("\n".join(lines), encoding="utf-8")
    return note_path


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    output_dir = source_root if args.output_dir is None else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = _load_run_metadata(source_root)
    panel_dirs = sorted(
        [path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("delta_") and (path / "bands_path.npz").exists()],
        key=lambda path: _delta_folder_order(path.name),
    )
    if not panel_dirs:
        raise FileNotFoundError(f"No saved delta_*/bands_path.npz datasets found under {source_root}.")

    panels: list[TMBGBandPlotPanel] = []
    panel_metadata: list[dict[str, object]] = []
    for folder in panel_dirs:
        panel, metadata = _load_panel(source_root, folder, run_metadata)
        panels.append(panel)
        panel_metadata.append(metadata)

    plot_paths = write_tmbg_paper_band_figure(
        output_dir,
        tuple(panels),
        stem=args.stem,
        title="Park 2020 Fig. 2 Checkpoint (±100 meV window)",
        ylim=(-args.window_mev / 1000.0, args.window_mev / 1000.0),
    )
    note_path = _write_markdown(
        output_dir,
        stem=args.stem,
        source_root=source_root,
        window_mev=args.window_mev,
        panels=panel_metadata,
    )

    metadata_path = output_dir / "plot_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_dir": str(output_dir),
                "window_mev": [-args.window_mev, args.window_mev],
                "stem": args.stem,
                "panels": panel_metadata,
                "artifacts": {
                    "png": str(plot_paths["paper_band_plot_png"]),
                    "pdf": str(plot_paths["paper_band_plot_pdf"]),
                    "note_md": str(note_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[done] source_root={source_root}")
    print(f"[done] output_dir={output_dir}")
    print(f"paper_band_plot_png={plot_paths['paper_band_plot_png']}")
    print(f"paper_band_plot_pdf={plot_paths['paper_band_plot_pdf']}")
    print(f"note_md={note_path}")
    print(f"plot_metadata_json={metadata_path}")


if __name__ == "__main__":
    main()
