#!/usr/bin/env python3
from __future__ import annotations

"""Report tracked source line counts for cleanup budgeting.

This tool is intentionally report-only by default.  Pass ``--fail-on-over`` to
turn a budget overrun into a non-zero exit status.
"""

import argparse
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileCount:
    path: Path
    lines: int


def _git_tracked_files(repo: Path) -> list[Path]:
    try:
        raw = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"failed to list tracked files with git ls-files: {exc}") from exc
    return [repo / line for line in raw.splitlines() if line]


def _count_lines(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _top_dir_key(path: Path, depth: int) -> str:
    parts = path.parts[: max(1, int(depth))]
    return "/".join(parts)


def _optional_feature_hint(path: Path) -> str | None:
    text = str(path)
    hints = [
        ("systems/tbg/finite_field", "tbg finite-field lane"),
        ("core/hf/finite_field", "generic finite-field HF lane"),
        ("core/hf/_finite_field", "generic finite-field HF lane"),
        ("core/magnetic_field.py", "generic magnetic-field helpers"),
        ("systems/htg/supercell", "HTG supercell lane"),
        ("systems/htg/_supercell", "HTG supercell lane"),
        ("systems/atmg", "ATMG optional system"),
        ("systems/htqg", "HTQG optional system"),
        ("analysis/topology/quantum_geometry.py", "topology quantum-geometry helpers"),
        ("api/tdhf", "TDHF API lane"),
        ("core/hf/tdhf", "core TDHF lane"),
        ("systems/RnG_hBN/tdhf", "RLG-hBN TDHF lane"),
        ("systems/RnG_hBN/_tdhf", "RLG-hBN TDHF lane"),
    ]
    for needle, label in hints:
        if needle in text:
            return label
    return None


def build_report(repo: Path, root: Path, suffix: str, dir_depth: int) -> tuple[list[FileCount], Counter[str], Counter[str]]:
    root = root.as_posix().rstrip("/")
    counts: list[FileCount] = []
    dirs: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    for absolute in _git_tracked_files(repo):
        if not absolute.exists() or not absolute.is_file():
            continue
        try:
            rel = absolute.relative_to(repo)
        except ValueError:
            continue
        rel_text = rel.as_posix()
        if root and rel_text != root and not rel_text.startswith(root + "/"):
            continue
        if suffix and not rel_text.endswith(suffix):
            continue
        lines = _count_lines(absolute)
        counts.append(FileCount(rel, lines))
        dirs[_top_dir_key(rel, dir_depth)] += lines
        hint = _optional_feature_hint(rel)
        if hint is not None:
            candidates[hint] += lines
    counts.sort(key=lambda item: item.lines, reverse=True)
    return counts, dirs, candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="src", help="tracked path prefix to count, e.g. src")
    parser.add_argument("--suffix", default=".py", help="file suffix to count, e.g. .py")
    parser.add_argument("--max-lines", type=int, default=35_000, help="line budget for the selected files")
    parser.add_argument("--top-files", type=int, default=30, help="number of largest files to print")
    parser.add_argument("--dir-depth", type=int, default=3, help="directory depth for top-directory aggregation")
    parser.add_argument("--fail-on-over", action="store_true", help="return non-zero if total exceeds --max-lines")
    args = parser.parse_args(argv)

    repo = Path.cwd().resolve()
    counts, dirs, candidates = build_report(repo, Path(args.root), str(args.suffix), int(args.dir_depth))
    total = sum(item.lines for item in counts)
    over = total - int(args.max_lines)

    print(f"root: {args.root}")
    print(f"suffix: {args.suffix}")
    print(f"tracked_files: {len(counts)}")
    print(f"total_lines: {total}")
    print(f"max_lines: {int(args.max_lines)}")
    print(f"over_budget: {max(0, over)}")
    print()

    print("top_directories:")
    for name, lines in dirs.most_common(30):
        print(f"{lines:7d}  {name}")
    print()

    print(f"top_files ({int(args.top_files)}):")
    for item in counts[: int(args.top_files)]:
        print(f"{item.lines:7d}  {item.path.as_posix()}")
    print()

    print("candidate_optional_features:")
    if candidates:
        for name, lines in candidates.most_common():
            print(f"{lines:7d}  {name}")
    else:
        print("      0  <none matched>")

    if args.fail_on_over and over > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
