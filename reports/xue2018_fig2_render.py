from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
BRANCH_DATA = DATA / "xue2018_fig2_branch_data.json"
STALE_POSTFLIGHT = DATA / "xue2018_fig2_stale_postflight.json"
PAPER_DIGITIZATION = DATA / "xue2018_fig2_digitized_markers.json"
OUTPUT = FIGURES / "xue2018_fig2_branch_lineage_audit.png"
MANIFEST = DATA / "xue2018_fig2_artifact_manifest.json"
REPORT = ROOT / "xue2018_fig2_blue_branch_report.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def paper_series(payload: dict, name: str) -> np.ndarray:
    rows = sorted(payload["series"][name], key=lambda row: row["sequence"])
    if [row["sequence"] for row in rows] != list(range(1, 63)):
        raise ValueError(f"paper {name} series does not contain points 1..62")
    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
    if name == "black":
        values[np.abs(values) < 2.0e-3] = 0.0
    return values


def main() -> None:
    branch_data = load_json(BRANCH_DATA)
    stale = load_json(STALE_POSTFLIGHT)
    paper = load_json(PAPER_DIGITIZATION)
    dataset_id = str(branch_data["dataset_id"])

    expected_stale = branch_data["full_path_dataset"]["source_json_sha256"]
    expected_paper = branch_data["paper_digitization"]["source_json_sha256"]
    if sha256(STALE_POSTFLIGHT) != expected_stale:
        raise ValueError("stale postflight hash does not match branch-data manifest")
    if sha256(PAPER_DIGITIZATION) != expected_paper:
        raise ValueError("paper digitization hash does not match branch-data manifest")

    rows = sorted(stale["rows"], key=lambda row: row["point"])
    if [row["point"] for row in rows] != list(range(1, 63)):
        raise ValueError("stale postflight does not contain points 1..62")
    x = np.arange(1, 63)
    calc_phi = np.asarray([row["selected_phi1_ry"] for row in rows])
    calc_gap = np.asarray([row["selected_gap_ry"] for row in rows])
    stale_trs_gap = np.asarray([row["trs_gap_ry"] for row in rows])
    paper_phi = paper_series(paper, "black")
    paper_gap = paper_series(paper, "red")
    paper_trs_gap = paper_series(paper, "blue")
    anchors = branch_data["strong_trs_anchor_dataset"]["anchors"]
    anchor_x = np.asarray([row["point"] for row in anchors])
    anchor_gap = np.asarray([row["gap_grid_ry"] for row in anchors])
    stationary = branch_data["stationary_trs_branch_dataset"]["zero_temperature_points"]
    stationary_x = np.asarray([row["point"] for row in stationary])
    stationary_gap = np.asarray([row["gap_grid_ry"] for row in stationary])

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.8), sharex=True)
    for axis, target, calculated, ylabel in (
        (axes[0], paper_phi, calc_phi, r"$|\Phi_1|/Ry^*$"),
        (axes[1], paper_gap, calc_gap, r"ground gap $/Ry^*$"),
    ):
        axis.plot(x, target, color="0.2", lw=1.8, label="paper digitization")
        axis.plot(x, calculated, color="tab:green", ls="--", lw=1.5, label="historical calculation")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18)
    axes[0].legend(frameon=False)

    axes[2].plot(x, paper_trs_gap, color="0.2", lw=1.8, label="paper blue digitization")
    axes[2].plot(
        x,
        stale_trs_gap,
        color="0.55",
        ls="--",
        lw=1.3,
        label="stale weak-seed/normal-attractor artifact",
    )
    axes[2].scatter(
        anchor_x,
        anchor_gap,
        color="tab:orange",
        edgecolor="black",
        linewidth=0.4,
        s=38,
        zorder=4,
        label="strong-TRS ODA attractor anchors",
    )
    axes[2].plot(
        stationary_x,
        stationary_gap,
        color="tab:purple",
        marker="D",
        ms=4.5,
        lw=1.5,
        zorder=5,
        label="full stationary TRS branch (p21--p26)",
    )
    axes[2].set_ylabel(r"TR-preserving gap $/Ry^*$")
    axes[2].set_xlabel("Fig. 2 point index")
    axes[2].grid(alpha=0.18)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Xue--MacDonald Fig. 2 branch-lineage audit (no fitted scale)")
    fig.tight_layout()
    fig.savefig(
        OUTPUT,
        dpi=180,
        metadata={
            "Dataset-ID": dataset_id,
            "Full-Path-Branch-ID": branch_data["full_path_dataset"]["branch_id"],
            "Strong-Anchor-Branch-ID": branch_data["strong_trs_anchor_dataset"]["branch_id"],
            "Stationary-Branch-ID": branch_data["stationary_trs_branch_dataset"]["branch_id"],
        },
    )
    plt.close(fig)

    manifest = {
        "schema": "xue2018-fig2-artifact-manifest-v1",
        "dataset_id": dataset_id,
        "report": REPORT.name,
        "data": [
            {"path": str(BRANCH_DATA.relative_to(ROOT)), "sha256": sha256(BRANCH_DATA)},
            {"path": str(STALE_POSTFLIGHT.relative_to(ROOT)), "sha256": sha256(STALE_POSTFLIGHT)},
            {"path": str(PAPER_DIGITIZATION.relative_to(ROOT)), "sha256": sha256(PAPER_DIGITIZATION)},
        ],
        "figures": [
            {
                "path": str(OUTPUT.relative_to(ROOT)),
                "sha256": sha256(OUTPUT),
                "dataset_id": dataset_id,
            }
        ],
        "full_path_branch_id": branch_data["full_path_dataset"]["branch_id"],
        "strong_anchor_branch_id": branch_data["strong_trs_anchor_dataset"]["branch_id"],
        "stationary_branch_id": branch_data["stationary_trs_branch_dataset"]["branch_id"],
        "mixed_branch_curve_forbidden": True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
