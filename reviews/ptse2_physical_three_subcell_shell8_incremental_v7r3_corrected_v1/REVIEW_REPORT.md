# Corrected independent external static review — V7R3

## Verdict

**PASS — static/parser scope only.**  **FAIL / not Authorized for execution.**

The V7R3 parser matches the exact reviewed grammar: one nonempty ASCII LF-terminated record, terminal ASCII spaces `job/node/partition = 1/1/0`, and exact unique nonempty required target fields. Internal ASCII space-run length is **not** constrained. The authoritative job and node records contain double-space runs (`[2]` and `[2,2,2]`) and correctly pass as positive raw fixtures.

This corrects the prior broad statement that V7R3 rejects “multiple spaces.” That statement is valid only for multiple **terminal** spaces. It is not valid for internal runs. Frozen wording that implies global internal single-space canonicality is a nonblocking scope/documentation defect; it grants no extra parser requirement.

## Bounded static evidence

- Source manifest: 71 rows, PASS.
- Capsule manifest: 73 rows, PASS.
- Scientific-input manifest: 22 rows, PASS.
- Failed-job evidence manifest: 15 rows, PASS.
- Python AST-only syntax: 26 files, PASS.
- `bash -n run_shell8.sbatch`: PASS.
- Environment, runtime-protocol, scheduler-mock, and static-protocol suites: PASS with `scheduler_contact=false` and `numerical_imports=false`.
- `bootstrap.py --verify-prep`: PASS; approval absent; scheduler/science false.
- Independent parser oracle: all three authoritative raw records pass; 6 internal-double/triple-space positive mutations pass; 33 terminal/framing/required-field negatives fail closed.
- Immutable mode, empty-runtime, no-symlink, no-cache, control-archive, and zero-science closure: PASS.

No scheduler command, numerical package/project import, scientific numerical action, MPI/OpenMX execution, submission, release, checkpoint, or science publication was performed.

## Genuine blockers and authority boundary

1. Terminal raw `sacct` accounting for job `519263` is absent. `ACCOUNTING_GAP.json` remains fail-closed.
2. This sibling review is not `REVIEW_APPROVAL.json` at V7R3’s configured required path and therefore cannot authorize submission, release, or execution.

Accordingly, the static review is Complete and PASS, but execution authority remains FAIL / false. No approval file was created.

## Pinned subject identities

- `SOURCE_FROZEN.json`: `109c1aabacc944f4035ef6b51895f3a8054a5b3609c5334d101ba5d6ecbbf3fd`
- `SOURCE_SHA256SUMS.txt`: `401432d1f7c3b06eefdc6d4fe86fae7b02d5d5d280fd4f825413b790c0bb5118`
- `CAPSULE_SHA256SUMS.txt`: `3dd49738419019bca5fd6a8a7f45479d6a1a0e1fadb0e167660314873f19d615`
- `CONFIG.json`: `bbef10742b8f84f0bf5189e6f38deb34b92580848d6a0b1abe40019c9026d20b`
- `STATIC_CHECKS.json`: `cee107154cddbc98acf643d0658c33e04e3590e3f96bb7b5de641d148f65904b`
- Prior V7R3 `STATIC_REVIEW.json`: `f1a31af57c08b5a3e9cf866036dd756440441fa8adb39b4cbfe9a22cb15a8854`
- Independent receipt: `401ee53760b060aa90c9dfddecf55c6a6aa049b8708eecc15bee57aaa4c5233d`

V7R2, V7R3, V7R4, and all pre-existing review directories were treated as read-only.
