# v5 held-chain submission incident

- UTC observation date: 2026-09-19
- Capsule: `physical_three_subcell_shell6_v5`
- Oracle job 511566 and projection job 511567 were created held; neither was released.
- Oracle live dependency: `Dependency=(null)`.
- Projection live dependency immediately after held submission: `Dependency=afterok:511566(unfulfilled)`.
- Workflow expected `afterok:511566` exactly and failed before completing the chain receipt.
- Both jobs otherwise matched the pinned names, account hmt03, user, multi-partition request, node exclusion, 64 tasks/CPUs, full-node memory, OverSubscribe=NO, wrapper commands, and role comments.
- No OpenMX/MPI/scientific calculation ran.
- Disposition: cancel exact held jobs 511566 and 511567, then create immutable v6 with explicit Slurm dependency lifecycle states: submitted-unfulfilled, satisfied/consumed before projection release, and null inside the running projection, while retaining original afterok lineage in durable receipts.
