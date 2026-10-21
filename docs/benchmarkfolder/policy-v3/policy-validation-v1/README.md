# NinjaRS-first policy software checkpoint

The final owned boundary suite passed **28 tests**, with no failures, errors or skips, on CPU4 in 1.781 seconds. `result.json` binds the exact tested policy/test sources, final log, command, scope and limitations. `sources/` contains byte-identical final source snapshots. The earlier 23- and 27-test passing logs are retained as historical attempts; they do not bind the final source revision.

The tests exercise evidence and numerical-comparison boundaries using small synthetic records, without importing a physics model or running a compiler, numerical engine or amplitude. They retain the normal statuses, full point/split-order data, frozen numerical settings and tolerance. A changed finite coefficient in an otherwise valid literal-original timing denominator rejects performance without erasing successful independent NinjaRS correctness.

`schema-rehearsal-summary.json` is an exact copy of the result of a separate read-only rehearsal against the actual retained v2 policy, representative proof and fourteen accepted CT/Ninja calibrations for seven processes. Both full legacy runtime groups were retained. Its original incomplete/failed parent statuses remain unchanged. EW4, NinjaRS correctness data and paired timing were absent, so every core/publication gate remained pending or false. The private rehearsal reference assets are intentionally outside this software checkpoint; the summary is not a replacement numerical-reference dataset.

The final log was produced by:

```sh
source .ninjars-build/dev-env.sh
taskset -c 4 python -B -m unittest -v tests.unit_tests.loop.test_ninjars_benchmark_policy_v3
```

The manifest binds every file in this checkpoint except itself. This checkpoint establishes software behavior only; the numerical campaign and performance acceptance remain separate.
