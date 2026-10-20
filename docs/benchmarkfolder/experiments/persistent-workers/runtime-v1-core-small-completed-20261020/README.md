# Completed-case subset of a failed benchmark run

Source: `docs/benchmarkfolder/experiments/persistent-workers/runtime-v1-core-small-20261020`. Its status remains **failed**.

This directory contains every case that completed before the failure, in the original order: `qcd-1`, `ew-1`, `qcd-2`, `ew-2`. No completed case, backend or repetition was selected out.

`parent-run/` retains the entire original failed run byte-for-byte, including its rejected measurements, incomplete later-case data, logs and status. `derivation.json` binds the copied evidence and the source snapshot. The complete status here applies only to the explicit completed-case subset. The remaining requested cases and the original failure remain outstanding.

The unchanged persistent-worker validator requires all seven targets and at least seven contiguous repetitions for every extracted case. Extraction does not promote data or imply numerical/performance passage.
