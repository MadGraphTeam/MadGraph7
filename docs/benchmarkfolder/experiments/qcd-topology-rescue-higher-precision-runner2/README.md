# Corrected private512 capture controller — source only

This is a separately prepared replay controller. It requires a frozen runtime-correction inspection, independent source review and explicit execution clearance. It does not rebuild or relink anything, and does not modify the original failed attempt, its controller, the retained256/NinjaRS controls or production/campaign artifacts.

The first attempt's numerical process exited0, but its capture predicate mistakenly required the loader reported by the host `ldd` tool instead of the interpreter embedded in the unchanged executable. The corrected protocol keeps those two closures separate. A bounded dependency listing by the exact pinned `PT_INTERP` establishes the expected native closure; all other runtime paths and hashes must match the retained listing. Actual `/proc/PID/maps` capture must then show exactly that closure plus the unchanged executable, including the private512 DSO. The native process is still launched normally through `taskset`; the inspection does not replace its entrypoint or loader policy.

The maps observer now retains first-read, first/last nonempty, first/last incomplete, final-read, complete and mismatch snapshots in distinct files when those events occur. An empty zombie read cannot erase the last nonempty map evidence. Its progress records the latest reason and read count. As before, a successful map and leader-affinity sample proves only that snapshot, not continuous monitoring.

All previously reviewed ownership protections remain: deferred handled signals across spawn ownership and group kill/reap/state update; default parent `SIGCHLD`; `waitid(WNOWAIT)` before reaping; no signaling after ownership release; and a monotonic deadline rechecked after mapping/exit observation. The `ldd` tool uses the same bounded supervisor. No native signal/trap policy changes are introduced.

Admission is nonblocking on the common numerical lock. Busy admission creates no native attempt and never retries automatically. Once this new namespace's `attempt1` exists, any success or failure consumes its only attempt. The one proposed numerical process uses CPU11, one thread and a180-second cap. It reuses the exact topology-c executable, ordinary/stress/ordinary point stream, cards, cache settings and private512 DSO. Only private resource, trace and temporary-directory paths move to the new attempt. The original113-bit input and512-bit arithmetic/guard configuration remain unchanged.

Successful capture additionally requires every selected numerical/status/IEEE stdout row and the **entire raw trace file** to match the prior incomplete attempt byte for byte. Ordinary rows/raw records must also match the retained256 control. Both tool/native runtime closures, all protected sources/cards and private resource bytes are checked before/after. Failure retains all raw outputs and snapshots and does not trigger a retry.

Completion is protocol qualification only. The prior attempt stays incomplete. Offline same-input/group/normalization checks, full finite/single-pole comparisons and their limitations must be reported separately. Neither runtime capture nor a checked host status establishes general physical or performance acceptance; working precision and the guard changed together.

Read-only source verification:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/verify_source.py
```

Proposed command, only after exact-freeze execution clearance:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/controller.py
```
