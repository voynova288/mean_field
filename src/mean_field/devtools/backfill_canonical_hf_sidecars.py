from __future__ import annotations

"""Compatibility shim for historical canonical HF sidecar backfills."""

from .canonical_hf_backfill import (
    BackfillCandidate,
    backfill_strategy,
    build_parser,
    execute_backfill_writes,
    inventory_payload,
    main,
    plan_backfill_writes,
    render_markdown_inventory,
    scan_backfill_candidates,
)

__all__ = [
    "BackfillCandidate",
    "backfill_strategy",
    "build_parser",
    "execute_backfill_writes",
    "inventory_payload",
    "main",
    "plan_backfill_writes",
    "render_markdown_inventory",
    "scan_backfill_candidates",
]

if __name__ == "__main__":
    raise SystemExit(main())
