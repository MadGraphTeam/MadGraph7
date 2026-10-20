# Allocation-free scalar-cache lookup prototype

Production is unchanged. This isolated ABI3 prototype follows the exact scoped
cache in `../master-cache/`, whose first comparison showed a DP regression and
only modest MP savings.

DP keys now contain a fixed eleven-element array of complex binary64 bits. MP
queries borrow the provider input slices and hash/compare exact values, tracked
precision and zero signs. They allocate owned Float keys only for new misses.
Full equality is checked after hashing; a forced-collision test covers this.
The FENV snapshot is taken once before each provider batch. All previous epoch,
bounded eviction, finite-success, flag observation and panic recovery rules
remain in force.

Diagnostic modes are 0 (disabled), 1 (DP and MP), and 2 (MP only). Mode 2 delegates
DP directly to its native provider without cache lookup or FENV observation.
Each complete MadLoop evaluation begins and ends a cache scope; rotations also
clear it. No integral value survives a full phase-space-point evaluation.

The nineteen bridge tests passed in 4.62s. This includes the new collision and
MP-only controls plus actual master integrals, precision and signed zero,
changed masses/scales/points, failures, FENV transitions, panic recovery, short
buffers and complex-mass convergence at 256 and 384 bits. `test-results.json` identifies the
tested executable and core sources; `all-tests.txt` retains its output.

The first generated qcd3 comparison used points 1/44/67/80, with mode order
0/2/1/1/2/0 after initialization and warm calls. All48 split-order rows have
identical printed coefficients, accuracy and status. Point 1 is 218 and the three
difficult points are 322. Mode 2 has zero DP hits and successful MP reuse. All
full-call boundaries have zero entries.

No timing benefit is established. CPU 4 wall-time medians relative to disabled
mode were:

| Point | DP+MP cache | MP-only cache |
|---|---:|---:|
|1|1.385|1.106|
|44|1.154|1.047|
|67|1.377|1.433|
|80|1.428|1.391|

The disabled DP call itself drifted from 32.45ms to 59.31ms, and MP-only DP calls
were 38.65ms and 62.81ms despite bypassing the DP cache. Rescued calls varied from
about 6.3s to 9s and back. Other builds were active; these two samples per mode
provide neither an acceptance interval nor a reliable isolation of cache cost.
The complete negative/inconclusive run is retained in `qcd3/`.

Reconstruct with `python .../master-cache-v2/reproduce.py`. Its patch applies
only after production source hashes match the recorded base and its output is
restricted to a new ignored experiment directory. Build with the pinned dev
environment, `cargo test --release --locked --lib`, and the bridge Makefile.
The process reproduction commands are:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/master-cache-v2/process-probe.py prepare
SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks" \
  python docs/benchmarkfolder/experiments/master-cache-v2/process-probe.py run --cpu 4
```

Preparation requires the isolated archive/module under
`.benchmark-work/experiments/master-cache-v2/install`; it refuses to overwrite
an existing generated output. The separate `process-probe-cpu-time.py` is
prepared for a subsequent CPU 207 run with both wall and process CPU time. That
follow-up was deliberately skipped after the warmed Zgg profile found scalar masters account for only 7.7% of total samples; no cache change was promoted.
