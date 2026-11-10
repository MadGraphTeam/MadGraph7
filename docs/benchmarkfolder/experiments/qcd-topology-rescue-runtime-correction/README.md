# Exact-interpreter runtime closure correction

One bounded dependency inspection used the unchanged executable's exact `PT_INTERP`, whose path and SHA256 were already frozen in the failed-attempt audit. It ran `taskset -c 11 <interpreter> --list <executable>` under the reviewed30-second ownership/deadline supervisor. This invoked the loader's dependency listing, **not the amplitude entrypoint**. No compiler or reducer evaluation ran.

The inspection passed. Its15 listed files equal the prior `ldd` closure except for one explicit correction: glibc2.40's embedded interpreter replaces the glibc2.42 loader used by the host `ldd` tool. The private512 bridge, every other library, all source/card/resource bytes and both loader hashes remained unchanged. `inspection1/inspection.json` records the command/environment, original and corrected closures, confirmed spawn/reap/exit, and all protected before/after hashes. `interpreter-list.txt` retains the actual listing.

The first numerical attempt remains incomplete. This listing cannot retroactively prove its actual library mappings or CPU affinity. A separately reviewed and authorized replay must capture its own mappings against the corrected native closure; its `ldd` check must continue to use the distinct tool closure. The numerical executable should continue to launch normally, without replacing its loader or applying a preload.

Read-only verification:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-runtime-correction/verify.py --live
```

The source and listing are append-only evidence. Neither this package nor its verifier authorizes another amplitude process. The inspection is one-shot and refuses an existing output directory.
