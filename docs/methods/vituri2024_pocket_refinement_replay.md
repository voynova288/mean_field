# Vituri-2024 frozen-source pocket-refinement replay v1

## Scope

The executable contract is system-local:

- implementation: `src/mean_field/systems/abc_trilayer/vituri2024_hf_pocket_replay.py`;
- synthetic contract tests: `tests/test_abc_trilayer_vituri2024_hf_pocket_replay.py`;
- prerequisites: factory-created `Vituri2024HalfMetalHFReplayReceipt`, exact
  `Vituri2024SCFReplayApproval`, and its factory-created
  `Vituri2024SCFReplayReceipt`;
- prerequisite-source baseline: commit
  `ae6fadf3b7e4a70e5390d73f724b9484bcbc7abd`.

A PASS replays a frozen selected-HF source on a registered refined mesh. It is
not a refined SCF calculation, a new fixed-density solve, or a real Vituri
artifact replay. The tracked evidence is synthetic contract/formula evidence;
no real refinement archive path exists in this repository.

## Trust model

The evidence model is

```text
trusted_live_selected_source_evaluator_distinct_refinement_archive_object
```

Detached approval binds complete canonical prerequisite fingerprints,
including the exact SCF approval and receipt. The SCF receipt's approval
fingerprint must equal the supplied SCF approval fingerprint; SCF provider,
source/spec/state, contract, archive, core provenance, and selected-source
identities are rebound to the current binding. At replay ingress the base
provider binding is reconstructed and every derivable pocket-approval field is
recomputed field-by-field: selected spin and source identities, base hashes,
SCF contract/archive/core hashes, ordered pocket-receipt fingerprints,
preflight mesh/count/evidence/margins/tolerances, provider and archive snapshots,
and callable manifests. Only the expected refinement-archive manifest and a
non-scientific provenance note remain detached external values. The expected
manifest closes immediately against the loaded archive before the live call.

Evaluator and archive metadata are semantic derivations, not merely snapshots:
the live evaluator and archive-loader implementation fingerprints equal their
verifier-derived callable-manifest fingerprints; request/evaluation/archive
schemas equal locked constants; `pocket_refinement_provider_fingerprint(...)`
and `pocket_refinement_archive_authority_fingerprint(...)` are recomputed; and
replay loader/schema plus archive `source/spec/state` equal the current attested
source. The same binding, semantic, snapshot, and manifest checks run again
after delegated calls. Authority and live provider must be different objects
with different fingerprints. The live request contains source identity, base
hashes, chemical potential, and the refinement mesh; it contains no archived
expected fields, topology, or margin. All arrays are immutable copies and
archive/request/result storage must be disjoint.

These controls prevent accidental same-object coupling. They do not exclude a
trusted same-code evaluator that reads a hidden copy of archive data. Therefore
all successful receipts retain:

```text
archive_live_computational_independence_verified = false
hostile_provider_resistance_verified = false
hidden_live_dependency_state_excluded = false
```

## Nested finite-domain mesh

For base shape `(n1,n2)` and strict integer subdivision factors `(r1,r2)`, with
`r1,r2 >= 1` and not both one, the only v1 refined shape is

```text
(r1*(n1-1)+1, r2*(n2-1)+1).
```

Both meshes are row-major affine Cartesian grids on the same closed domain.
The exact index embedding is `(i,j) -> (r1*i,r2*j)`. V1 rejects halos,
periodic wrapping, reciprocal carry, non-affine coordinates, collinear axes,
non-unique embeddings, and hash/count drift. The synthetic fixture is
`4x5 -> 7x9`: 20 base points and 63 refined points. Both preflight pocket
receipts bind the actual 63-point mesh hash, verifier evidence hash, and raw
margin rather than placeholder hashes/counts.

## Frozen-source fields and occupations

The trusted live evaluator returns only complex128 arrays
`h0`, `interaction_h`, and `fock`, each shaped `(4,4,Nref)`. The verifier checks
finiteness, Hermiticity, the locked diagonal flavor basis,
`fock = h0 + interaction_h`, embedded-base array/energy/occupation/projector
hash parity, and locked archive/live fine-grid parity. Energies are derived
from the real Fock diagonal. No refined eigensolve or SCF is performed.

With the selected-source chemical potential `mu` and pre-registered threshold
uncertainty `eta`, occupations are derived strictly:

```text
n=1 for E<mu; n=0 for E>mu.
```

Any `abs(E-mu) <= eta` fails closed. Both selected-spin valleys must contain
holes; opposite-spin holes are forbidden.

## Digital topology

For each selected-spin valley in order `(-1,+1)`, holes use four-neighbor
finite-domain connectivity with no wrap. Acceptance at a threshold requires:

1. a nonempty hole mask;
2. exactly one four-connected hole component;
3. no hole vertex on the domain boundary;
4. zero enclosed components of the eight-connected occupied complement.

The 4/8 dual convention is a digital non-annularity test. It does not claim
radiality, convexity, or continuum topology.

## Discrete threshold-topology margin

Degenerate refined energy vertices are grouped at one exact level. The
verifier evaluates the topology predicate on every open interval between
consecutive unique levels and finds the maximal consecutive accepted interval
block containing `mu`. If its critical boundary levels are `lambda_lower` and
`lambda_upper`, the raw margin is

```text
min(mu-lambda_lower, lambda_upper-mu).
```

The receipt records both critical levels, their degeneracies, and rejected
outside signatures. It separately requires `min(abs(E-mu)) > eta` and raw
margin `> eta`. A near-`mu` level that changes only pocket cardinality does not
limit the topology margin.

For archive/live lanes, the certified lower bound is

```text
min(raw_archive, raw_live) - eta - max_abs(E_archive-E_live),
```

and must be positive. This is a finite-grid threshold-topology margin. It is
not `min(abs(E-mu))`, a continuum saddle energy, a continuum-stability result,
or a refinement-convergence certificate.

## Evidence hash and claims

Each preflight `refinement_evidence_sha256` equals the verifier-defined
canonical manifest of:

- conventions and source identity;
- base/refined mesh fingerprint and hashes/counts;
- archive/live H0, interaction, Fock, energy, occupation, and projector hashes;
- hole-mask hashes and digital-topology signatures;
- critical levels, multiplicities, rejected signatures, raw margins, certified
  margin, uncertainty, and parity residual/bounds.

The pocket/spec fingerprint is excluded to prevent a fingerprint cycle.
Factory-only positive status fields stop at prerequisite binding, archive load,
mesh registration, one live frozen-source evaluation, recomputed occupations,
bilateral topology, discrete margins, and `pocket_refinement_replayed`.
Real-artifact, independence/hostile-provider, refined-SCF/fixed-density,
continuum stability, refinement convergence, ground-state, scientific/paper,
and TDHF-readiness claims remain false.

## Current evidence and uncertainty

- Synthetic executable evidence: the dedicated test path above, including
  mesh/order/embedding, 4-vs-8/no-wrap, disconnected/boundary/annular,
  degenerate-level, near-threshold, strict uncertainty, opposite-spin,
  embedded-base, archive/live corruption, cross-source same-branch SCF identity,
  complete approval-field mutation, arbitrary-valid metadata/schema,
  metadata/identity/AST, and hidden same-code archive-copy canaries.
- Existing Vituri array and SCF evidence remains in
  `tests/test_abc_trilayer_vituri2024_hf_preflight.py`.
- No real Vituri refinement archive has been supplied or replayed. Physical
  pocket stability and paper reproduction remain unverified.
