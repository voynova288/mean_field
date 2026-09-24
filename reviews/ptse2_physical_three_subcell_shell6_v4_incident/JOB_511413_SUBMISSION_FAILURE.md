# v4 held-submission incident

- UTC observation date: 2026-09-19
- Capsule: `physical_three_subcell_shell6_v4`
- Workflow return: 2, after `sbatch` created oracle job 511413 held.
- Failure: `parse_scontrol_line` rejected the live `scontrol show job -o` record with `ValueError: duplicate, empty, or multiword target field: Partition`.
- Root cause: parser key regex did not recognize the non-target Slurm key `AllocNode:Sid`, so it was absorbed into `Partition`; additionally this cluster reports pending multi-partition requests as `Partition=regular256,regular6430` and reports `OverSubscribe=NO`, not an `Exclusive` field.
- Live record before cancellation showed: JobName `pt7p_n2B1_missing258_v4`, UserId `ziyuzhu(1091)`, Account `hmt03`, JobState `PENDING`, Reason `JobHeldUser`, Priority `0`, Dependency `(null)`, Partition `regular256,regular6430`, ExcNodeList `node037`, NumCPUs/NumTasks `64/64`, MinMemoryNode `0`, OverSubscribe `NO`, exact v4 wrapper command, and authorization comment `ptse2-shell6-v4-a8d94c36f102e75b4ce68139:oracle`.
- No projection job was submitted. No job was released. No OpenMX/MPI/scientific computation ran.
- Disposition: cancel exact held orphan job 511413 and create a new immutable v5 with cluster-specific parser/field fixes and live-record mocks.
