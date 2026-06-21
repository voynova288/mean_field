from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan_utils import *  # noqa: F401,F403

def _manifest_candidate_roots(root: Path, max_candidates: int) -> list[Path]:
    candidates: list[Path] = []
    if root.is_file():
        if root.name == "manifest.json":
            return [root.parent]
        return []
    if not root.exists():
        return []
    if (root / "manifest.json").is_file():
        candidates.append(root)
    for manifest_path in sorted(root.rglob("manifest.json")):
        parent = manifest_path.parent
        if parent not in candidates:
            candidates.append(parent)
        if len(candidates) >= max_candidates:
            break
    return candidates


def _rlg_hbn_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if root.is_file():
        return [root] if root.name in {"hf_ground_state.npz", "hf_run_state.npz"} else []
    if not root.exists():
        return []
    candidates = list(sorted(root.rglob("hf_ground_state.npz")))
    if len(candidates) >= max_candidates:
        return candidates[:max_candidates]
    # Individual run archives are useful inventory evidence, but ground-state
    # archives remain the preferred historical backfill target.
    for archive_path in sorted(root.rglob("hf_run_state.npz")):
        if archive_path not in candidates:
            candidates.append(archive_path)
        if len(candidates) >= max_candidates:
            break
    return candidates[:max_candidates]


def _tdbg_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if not _path_text_mentions(root, "tdbg"):
        return []
    if root.is_file():
        return [root] if root.name == "hf_state.npz" else []
    if not root.exists():
        return []
    return list(sorted(root.rglob("hf_state.npz")))[:max_candidates]


def _htg_archive_candidates(root: Path, max_candidates: int) -> list[Path]:
    if not _path_text_mentions(root, "htg"):
        return []
    archive_names = {
        "hf_ground_state.npz",
        "hf_supercell_ground_state.npz",
        "hf_supercell_ground_state_best.npz",
        "hf_supercell_ground_state_best_copy_of_candidate.npz",
    }
    if root.is_file():
        return [root] if root.name in archive_names else []
    if not root.exists():
        return []
    candidates: list[Path] = []
    for name in sorted(archive_names):
        for archive_path in sorted(root.rglob(name)):
            if archive_path not in candidates:
                candidates.append(archive_path)
            if len(candidates) >= max_candidates:
                return candidates[:max_candidates]
    return candidates[:max_candidates]

__all__ = [name for name in globals() if not name.startswith('__')]
