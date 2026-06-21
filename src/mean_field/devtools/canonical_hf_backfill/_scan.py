from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan_utils import *  # noqa: F401,F403
from ._scan_contracts import *  # noqa: F401,F403
from ._scan_discovery import *  # noqa: F401,F403
from ._scan_classify import *  # noqa: F401,F403

def scan_backfill_candidates(
    roots: Sequence[str | Path],
    *,
    include_archives: bool = True,
    max_candidates: int = 10000,
) -> list[BackfillCandidate]:
    """Return dry-run canonical HF sidecar backfill candidates.

    The scan is metadata-only except for recognized TDBG/HTG/RLG-hBN NPZ
    header/scalar inspection.  It never writes into result directories.
    """

    records: list[BackfillCandidate] = []
    seen_manifest_roots: set[Path] = set()
    seen_archives: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        for candidate_root in _manifest_candidate_roots(root, max_candidates=max_candidates):
            resolved = candidate_root.resolve()
            if resolved in seen_manifest_roots:
                continue
            seen_manifest_roots.add(resolved)
            records.append(_classify_manifest_root(candidate_root))
        if include_archives:
            for archive_path in _tdbg_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_tdbg_archive(archive_path))
            for archive_path in _htg_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_htg_archive(archive_path))
            for archive_path in _rlg_hbn_archive_candidates(root, max_candidates=max_candidates):
                resolved = archive_path.resolve()
                if resolved in seen_archives or archive_path.parent.resolve() in seen_manifest_roots:
                    continue
                seen_archives.add(resolved)
                records.append(_classify_rlg_hbn_archive(archive_path))
    return records


__all__ = [name for name in globals() if not name.startswith("__")]
