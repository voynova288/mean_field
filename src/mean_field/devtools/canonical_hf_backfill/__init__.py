"""Compatibility package for canonical HF sidecar backfill internals."""

from ._shared import BackfillCandidate
from ._scan import scan_backfill_candidates
from ._write import execute_backfill_writes, plan_backfill_writes
from ._report import backfill_strategy, inventory_payload, render_markdown_inventory
from ._cli import build_parser, main

__all__ = [
    'BackfillCandidate',
    'backfill_strategy',
    'build_parser',
    'execute_backfill_writes',
    'inventory_payload',
    'main',
    'plan_backfill_writes',
    'render_markdown_inventory',
    'scan_backfill_candidates',
]
