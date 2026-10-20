# Natural QCD4 CutTools initialization timeout

The frozen full-eight corrected-original collector exited with ProtocolError after QCD4 CutTools failed to reach READY within its 7200-second phase timeout. The worker was not interrupted or modified; the controller terminated it through its ordinary timeout cleanup. The entire original report remains `failed`, with only the first six cases completed. QCD4 has no accepted calibration or repetition, and EW4 was not evaluated. No complete reference binding or benchmark acceptance is claimed.

All 54 completed records (18 calibration plus 36 two-command measurements) retain complete 100-point/order streams, source/build/settings identities, statuses and timing bookkeeping. The verifier independently checks those records and their three-backend comparisons while requiring the frozen whole-series validator to reject the failed run. The completed prefix is retained as evidence, not promoted or omitted from the fresh full-eight retry.

The rejected QCD4 initialization snapshot includes values/timing/dimensions copied from the generated parent before initialization. `stale-output-provenance.json` identifies exact parent-byte matches; retained parent copies permit offline verification. These snapshots are explicitly not current QCD4 measurements. Startup stdout contains no READY, and no command was issued.

`collector.log` is the closed outer controller log, including the timeout traceback. The manifest binds every file in the original failed report, the original prepared launch, this retention package, and the read-only warmup-policy/workload-scaling diagnostics. The earlier ten-second perf sample remains in its previously committed diagnostic retention package.

Run `python -B verify.py` from this directory or pass its repository-relative path from the repository root. The verifier reads retained evidence only and starts no builds or amplitudes. The separately versioned `reference-collection-plan-24h` retries all eight cases with unchanged numerics and a 86400-second phase timeout.
