# Full-process unconditional T3 compensation experiment

This retained experiment compares the production NinjaRS archive with the
isolated guarded compensated T3 archive from `../qcd3-point67-t3`. It is
diagnostic evidence, not a final correctness or performance gate.

Both variants reuse the same `fd72f8914` CLI exports with the corrected model
helpers. They differ only in their private linked NinjaRS archive. They predate
the subsequent `-fno-automatic` classifier repair. All 100 original points,
all physical orders, complete warmed evaluation, stability thresholds, and
genuine QP rescue remain enabled. The benchmark driver, cards, generated-source
inventories, archive identities, build logs, raw measurements, and parsed
results are retained under `first-three-cases-20261020`.

Three alternating pairs ran on CPU 4 with one evaluation thread. This was a
contended host, not a reserved timing window. No confidence interval or
acceptance conclusion is inferred from these timings.

| Case | Timed loops per point | Guarded/native median ratio | Native DP/QP points | Guarded DP/QP points | Largest fraction of coefficient tolerance |
|---|---:|---:|---:|---:|---:|
| QCD 2→1 | 100 | 2.105173 | 100/0 | 100/0 | 2.35e-8 |
| EW 2→2 | 20 | 1.161956 | 100/0 | 100/0 | 1.05e-6 |
| QCD 2→3 | 1 | 0.779724 | 97/3 | 98/2 | 0.717465 |

Every coefficient passes the prescribed `1e-7 max(abs(a),abs(b)) + 1e-10 S`
tolerance, using the common point's largest Born/Laurent magnitude. Within
each variant, every repetition has exactly identical parsed coefficients,
accuracy estimates, and return codes. Status 218 denotes stable NinjaRS DP;
322 denotes its higher-precision rescue. Timed status totals match every
point and loop.

The added arithmetic avoids one QP rescue in Zgg but is expensive on the two
all-DP cases in this exploratory series. The next experiment applies this
arithmetic only after an ordinary DP stability failure, repeating the full
unchanged stability checks before deciding whether QP rescue is required.
Unconditional compensation has not been promoted to production.

To reproduce inside the repository after preparing the recorded CLI exports
and the separately documented guarded bridge:

```bash
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/experiments/t3-compensation-performance/run.py \
  prepare --run-id new-run
python docs/benchmarkfolder/experiments/t3-compensation-performance/run.py \
  run --run-id new-run --case qcd-1 --loops 100 --repetitions 3 --cpu 4
python docs/benchmarkfolder/experiments/t3-compensation-performance/run.py \
  run --run-id new-run --case ew-2 --loops 20 --repetitions 3 --cpu 4
python docs/benchmarkfolder/experiments/t3-compensation-performance/run.py \
  run --run-id new-run --case qcd-3 --loops 1 --repetitions 3 --cpu 4
```

Use a new run identifier: the runner refuses to overwrite existing evidence.
The sealed `SHA256SUMS.json` includes raw `.log` files despite the repository's
general log ignore rule. The production archive remained
`72a820a1fb74904e9b3d3a43562132b365a947304ac41abced46f5d407a0c816`.
