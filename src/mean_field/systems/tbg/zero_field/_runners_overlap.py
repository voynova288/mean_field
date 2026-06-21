from __future__ import annotations

from ._runners_shared import *  # noqa: F401,F403

def export_overlap_diagnostics(theta_deg: float, *, lattice_kind: str, m: int, n: int, points_per_segment: int = 120, lg: int = 9, grid_lk: int = 33) -> OverlapDiagnostics:
    run = run_bm_unstrained(theta_deg, points_per_segment=points_per_segment, lg=lg, grid_lk=grid_lk)
    if lattice_kind == "path":
        solution = run.path_solution
    else:
        if run.grid_solution is None:
            raise ValueError("Grid overlap requested but grid solution was not computed.")
        solution = run.grid_solution
    overlap = calculate_overlap_compact(solution, m, n, valley_index=0)
    return summarize_overlap(theta_deg, lattice_kind, overlap, m, n, valley_label="K")

__all__ = [name for name in globals() if not name.startswith('__')]
