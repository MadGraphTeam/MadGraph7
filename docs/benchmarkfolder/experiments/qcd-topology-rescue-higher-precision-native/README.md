# Private512 attempt: runtime capture failed

The single authorized process exited0 after5.36seconds, but the controller correctly rejected the attempt because it did not capture a complete actual process mapping. The final saved `/proc/PID/maps` snapshot is empty. This attempt remains incomplete; there was no automatic retry and no production change.

All protected sources, executable, cards, resources and `ldd`-resolved file hashes remained unchanged. The pinned executable requests the glibc2.40 interpreter through `PT_INTERP`. The retained `ldd` listing substitutes its own glibc2.42 loader, and the controller incorrectly required that replacement in the actual executable's mappings. This establishes a static incompatibility in the capture predicate. It does not reconstruct the actual loaded library set or affinity from the empty snapshot.

The frozen decoder accepts the raw trace: all232 stress calls report native success/stability, the host returns323, all three health acknowledgements are clean, and ordinary printed/raw coefficient/status/IEEE records match the retained256-bit control exactly. These are **provisional raw results**, not qualified runtime or full physical acceptance. The unqualified stress coefficients are retained in the log and failure audit. No output or original execution record has been rewritten to turn this attempt into a success.

`failure-audit.json` binds the exact source revision, interpreter/executable bytes, output records and frozen ordinary comparator. Reproduce its read-only analysis with:

```sh
python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-native/audit_failure.py
```

Any corrected runtime capture or separately authorized repeat belongs in a new sibling namespace. Working precision and the admission guard changed together in this experiment; even a later successful protocol cannot isolate their respective effects or establish finite/single-pole truth by itself.
