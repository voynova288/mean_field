# Exact saved-grid curve workflow

## Purpose

`mean_field.core.curve_workflow` is the system-independent path from a validated
solver result to auditable one-dimensional curves and optional plan-bound raster
comparison. A raster is called held out only when its evaluation plan records
target isolation, post-freeze evaluation, preregistration before the final run,
and blind contract selection. Post-freeze but nonblind comparisons remain
explicitly labeled evidence-only.  A physical system supplies a thin adapter; the common layer owns:

- finite supplied-tree structural closure;
- exactly-once evaluation of every computed terminal;
- immutable exact saved-grid curves;
- raw-to-output value transforms with units and semantics;
- exact-node extrema, explicit piecewise-linear crossings, and all-curve spread;
- hash-bound raster extraction and preregistered numerical criteria;
- deterministic JSON/NPZ/CSV artifacts and generic plotting.

It does **not** run an SCF loop, define a physical branch universe, choose an
observable, infer a chemical potential, select a best branch, or grant paper,
production, UV, unrestricted-state, Hessian, or TDHF authority.

## Layering

```text
system solver/result
  -> system ExactGridCurveAdapter
  -> build_exact_grid_curve_bundle
  -> optional RasterEvaluationPlan
  -> extract_raster_curve
  -> compare_raster_to_all_branches
  -> write_curve_workflow_artifacts / plot_exact_grid_curve_bundle
```

The compute bundle carries a system-supplied opaque `SourceAuthorityReceipt`.
The generic layer hashes and preserves that payload but never interprets or
promotes it. `payload()` first revalidates the live canonical JSON/hash and then
returns a fresh decoded copy; mutating that copy cannot mutate the receipt.

## Adapter contract

A system adapter implements `ExactGridCurveAdapter`:

```python
from mean_field.api import (
    CurveDomainReceipt,
    DiscreteBranchNode,
    EnumerationReceipt,
    ExactGridObservableEvaluation,
    ObservableReceipt,
    SavedGridReceipt,
    ValueTransformReceipt,
    build_exact_grid_curve_bundle,
    make_source_authority_receipt,
)

class MyAdapter:
    branch_tree: tuple[DiscreteBranchNode, ...]
    enumeration_receipt: EnumerationReceipt
    source_authority = make_source_authority_receipt(
        "my-system-candidate-source",
        {
            "candidate_only": True,
            "paper_reproduction_verified": False,
            "production_authority": False,
        },
    )

    def evaluate_terminal(self, terminal_id: str) -> ExactGridObservableEvaluation:
        # Return one exact saved-grid evaluation.  The callback is called once
        # for every computed leaf in canonical ID order.
        ...

bundle = build_exact_grid_curve_bundle(MyAdapter())
```

Required system-owned evidence includes:

1. a finite tree whose parent/child records are structurally complete;
2. an `EnumerationReceipt` binding the enumeration implementation, source
   inputs, choice inventory, frontier count, and terminal payload hashes;
3. an exact saved-grid index/coordinate receipt;
4. an observable identity, basis, units, and validity statement;
5. an affine `ValueTransformReceipt` from raw values to output values;
6. an opaque source-authority payload containing the system's actual positive
   and negative authority fields.

`system_claims_exhaustive_enumeration=True` remains a system claim about its
explicit finite branch universe.  The generic certificate says only
`supplied_finite_tree_structurally_resolved`.

## Value and domain conventions

The raw observable and displayed value are separate:

```text
output_y = scale * raw_y + offset
```

This covers identity transforms, unit conversions, and declared additive
references without making chemical-potential subtraction a generic assumption.
Every transform records input/output units and semantics.

Curve topology is explicit.  Current crossing/interpolation/raster comparison
supports `CurveDomainReceipt(topology="open_interval")`.  Periodic domains must
supply period and seam, but comparison fails closed until periodic-seam logic is
implemented.

Source point indices need only be unique, nonnegative, and int64-safe; they need
not be ordered like the curve coordinate.  The coordinate `x` must be strictly
increasing.

## Raster evaluation

A comparison is a separate phase:

1. hash the target with `raster_source_receipt`;
2. declare or detect the plot-frame calibration;
3. create a `RasterEvaluationPlan` binding the compute bundle, target hash,
   calibration, extraction policy, value convention, optional criterion,
   lineage, and preregistration evidence;
4. extract the unique qualifying component;
5. compare the same raster samples against **every** computed curve.

The only decisions are:

- `evidence_only`;
- `criterion_satisfied`;
- `criterion_not_satisfied`.

These are numerical evaluation outcomes, not reproduction authority.  A
criterion is ineligible unless its plan carries preregistration evidence and
satisfies the declared lineage conditions.  `RasterBundleComparison` is
factory-only and cannot be constructed independently of its plan and source.

## Vituri adapter

`mean_field.systems.abc_trilayer.vituri2024_curves` connects the existing
fixed-sector BFS result:

```python
from mean_field.systems.abc_trilayer.vituri2024_curves import (
    build_vituri2024_fixed_sector_curve_bundle,
)

bundle = build_vituri2024_fixed_sector_curve_bundle(
    prepared,
    search_result,
    flavor=3,
)
```

The adapter, not the generic layer, owns:

- the homogeneous fixed-half-metal sector and exact-shell coordinate choices;
- canonical sibling completeness for every expanded BFS frontier;
- exact `k_y=0` selection and `k_x a_0` coordinates;
- `Re H_ff` in the fixed flavor basis;
- the common intersection of endpoint chemical-potential intervals;
- the `eV -> meV` transform `1000 * (Re H_ff - mu_common)`;
- the finite-square/no-wrap source meaning and all candidate-only authority
  limits.

The adapter accepts only a rejection-free, all-stationary, exhausted typed
search result.  It never converts the in-process candidate receipt into sealed
independent, paper, UV, unrestricted, Hessian, TDHF, or production authority.

## Artifacts

`write_curve_workflow_artifacts` writes fixed generic names:

```text
metadata.json
arrays.npz
curves.csv
```

`load_curve_workflow_artifacts` verifies hashes, reconstructs the bundle,
raster, plan, and comparison, parses every CSV row against NPZ values, and
re-derives comparison metrics and decisions.  The loader rejects metadata,
array, CSV, source-authority, plan, or comparison drift.

## Validation boundary

Unit tests cover structural closure, exact-grid mathematics, transforms,
source authority, raster extraction, plan binding, criterion decisions,
tampering, and full artifact roundtrip.  Physical solver validation remains a
system responsibility and heavy HF recomputation remains Slurm-only.
