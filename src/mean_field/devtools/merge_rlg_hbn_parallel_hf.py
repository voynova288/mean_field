from __future__ import annotations

from pathlib import Path

from mean_field.api.artifacts import ModelRecord, write_contract_artifacts

# Historical RnG/hBN parallel paper-HF merge workflow was retired from tracked
# command surface and archived under
# local_archive/retired_surface/09b4946_rlg_hbn_parallel_merge_devtool/.
# Keep only the metadata sidecar compatibility helper used by schema tests.


def _write_contract_sidecars(
    source_root: Path,
    *,
    paper_target: str,
    merged_config: dict[str, object],
    selected_rows: list[dict[str, object]],
    ignored_panel_dirs: list[dict[str, object]],
    tasks_root: Path,
) -> dict[str, Path]:
    return write_contract_artifacts(
        source_root,
        workflow="rlg_hbn.parallel_hf_merge",
        system_name="rlg_hbn",
        model=ModelRecord(
            system_name="rlg_hbn",
            params={
                "paper_target": str(paper_target),
                "layer_count": int(merged_config["layer_count"]),
                "xi_values": [int(value) for value in merged_config["xi_values"]],
                "v_values_mev": [float(value) for value in merged_config["v_values_mev"]],
                "hbn_moire_scale": float(merged_config.get("hbn_moire_scale", 1.0)),
            },
            lattice={"theta_deg": float(merged_config["theta_deg"]), "shell_count": int(merged_config["shell_count"])},
        ),
        config=merged_config,
        conventions={
            "energy_unit": "meV",
            "density_convention": "stored_delta",
            "density_axis_order": "abk",
            "system": "RLG/hBN",
            "paper_target": str(paper_target),
        },
        validation={
            "status": "pass",
            "selected_panel_count": int(len(selected_rows)),
            "ignored_candidate_count": int(len(ignored_panel_dirs)),
            "tasks_root": str(tasks_root),
        },
        observables={
            "paper_target": str(paper_target),
            "selected": selected_rows,
            "ignored_candidates": ignored_panel_dirs,
        },
        files={
            "paper_hf_config": "paper_hf_config.json",
            "cache_manifest": "cache_manifest.json",
            "parallel_selection_summary": "parallel_selection_summary.json",
            "selected_panels": [str(row["panel"]) for row in selected_rows],
        },
        metadata={"tasks_root": str(tasks_root), "paper_target": str(paper_target)},
    )


def main() -> None:
    raise SystemExit(
        "merge_rlg_hbn_parallel_hf was retired from the tracked command surface. "
        "Consult local_archive/retired_surface/09b4946_rlg_hbn_parallel_merge_devtool/ or git history if needed."
    )


__all__ = ["_write_contract_sidecars", "main"]
