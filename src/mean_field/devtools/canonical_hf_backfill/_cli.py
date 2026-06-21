from __future__ import annotations

from ._shared import *  # noqa: F401,F403
from ._scan import scan_backfill_candidates
from ._write import execute_backfill_writes, plan_backfill_writes
from ._report import inventory_payload, render_markdown_inventory

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first inventory and allowlisted staging helper for historical canonical HF sidecar backfill eligibility."
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[DEFAULT_RESULT_ROOT],
        help="Result roots to scan. Defaults to repository results/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Default mode: scan/report without writing staged sidecars or modifying historical results.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Explicitly materialize eligible sidecars into --target-root staging directories; scanned historical roots are never mutated.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=None,
        help="Caller-specified staging root for --write outputs or dry-run write plans.",
    )
    parser.add_argument(
        "--allow-target-root",
        type=Path,
        action="append",
        default=[],
        help="Allowlist parent for --target-root. Required with --target-root/--write; repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing files in the staging target root. Default refuses overwrites.",
    )
    parser.add_argument("--no-archives", action="store_true", help="Skip TDBG/HTG/RLG-hBN NPZ archive inventory.")
    parser.add_argument("--max-candidates", type=int, default=10000, help="Safety cap per root and candidate kind.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional explicit JSON report path.")
    parser.add_argument("--report-md", type=Path, default=None, help="Optional explicit Markdown report path.")
    parser.add_argument(
        "--fail-on-ineligible",
        action="store_true",
        help="Return nonzero if any scanned candidate is not already canonical or eligible via an existing loader.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write and args.target_root is None:
        parser.error("--write requires --target-root")
    if args.target_root is not None and not args.allow_target_root:
        parser.error("--target-root requires at least one --allow-target-root")

    roots = list(args.roots)
    records = scan_backfill_candidates(
        roots,
        include_archives=not bool(args.no_archives),
        max_candidates=int(args.max_candidates),
    )
    write_plan: dict[str, object] | None = None
    if args.target_root is not None:
        if bool(args.write):
            write_plan = execute_backfill_writes(
                records,
                roots=roots,
                target_root=args.target_root,
                allow_target_roots=args.allow_target_root,
                overwrite=bool(args.overwrite),
            )
        else:
            write_plan = plan_backfill_writes(
                records,
                roots=roots,
                target_root=args.target_root,
                allow_target_roots=args.allow_target_root,
                overwrite=bool(args.overwrite),
            )
    payload = inventory_payload(records, roots=roots, dry_run=not bool(args.write), write_plan=write_plan)
    markdown = render_markdown_inventory(payload)
    if args.report_json is not None:
        write_json_artifact(payload, args.report_json)
    if args.report_md is not None:
        write_text_artifact(markdown, args.report_md)
    print(markdown, end="")
    if write_plan is not None and not bool(write_plan.get("dry_run", True)):
        write_summary = _mapping(write_plan.get("summary"))
        if int(write_summary.get("write_error_count", 0)):
            return 3
    if bool(args.fail_on_ineligible):
        allowed = {"already_canonical", "eligible_with_existing_archive_loader"}
        if any(record.decision not in allowed for record in records):
            return 2
    return 0
