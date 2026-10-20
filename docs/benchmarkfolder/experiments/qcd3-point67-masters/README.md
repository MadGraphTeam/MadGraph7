# Point 67: scalar, integration, and numerator error isolation

These diagnostics use the six frozen reductions from
`../qcd3-point67-dp/oracle-inputs.json`: loops 3, 52 and 71 in both independent
reduction directions. The reference is the previously retained full Float256
reduction of those exact binary64 tensors. It is not a replacement for evaluating
the generated numerator in QP. Production code, installed libraries, directions,
and stability thresholds are unchanged.

The unmodified native fast replay matches all six Laurent components exactly for
every input. Its 66 scalar requests (114 integral outputs) are retained with raw
binary64 bits, the native `1e-9` on-shell threshold and the thresholded arguments.
`main.rs` copies the threshold preprocessing from pinned NinjaRS
`0d0d7e7a8033431b60a8580731084742e0655cda`,
`crates/ninjars-symbolica/src/oneloop_native.rs` (GPL-3.0-or-later). Thresholding is
performed in binary64 before exact widening to Float256; the generic provider's
own threshold is disabled, preserving native branch decisions.

Replacing every requested scalar integral by its Float256 result rounded back
to binary64 leaves the finite real coefficient bitwise unchanged in five of the
six reductions. Loop 71 reverse changes by only `3.55e-15`, against its original
`3.63e-12` error. Other Laurent components move by at most `2.85e-14`. The largest
individual complex-relative master discrepancy is `1.81e-13` in C0; this does not
explain the full reduction error. `results.json` retains all Laurent outputs for
each master, high-precision decimals, statuses and full native/substituted results.

`weights.rs` isolates each final integration coefficient with one unit finite
master and every other master zero. It asserts identical request arguments and
unchanged rational terms. Exact decimal products/sums of these binary64 weights
and masters expose strong cancellation: loop 3 sums absolute real contributions
of about `1.43e5` to produce `0.0167`; loop 52 sums about `1.22e5` to produce
`-0.0181`. Removing all final product/sum rounding changes forward finite-real
errors by `2.59e-12` and `1.93e-12`, respectively. Earlier arithmetic still leaves
reverse errors of about `-2.31e-11` and `-9.69e-12`. `weights-summary.json` retains
all six components and absolute-term sums. This calculation removes product
rounding too, so it bounds more than summation compensation alone.

`expansion.rs` next keeps the native DP cut vectors, reduction algebra and masters,
but computes numerator callbacks in Float256 before rounding their outputs back
to binary64. It compares all callbacks together and each family separately.
T3 alone reproduces the all-callback result to about `1.6e-17`; direct evaluation
and T2 substitution have negligible effect here.

| Finite real error versus full Float256 | Native forward | Native reverse | Accurate T3 forward | Accurate T3 reverse |
|---|---:|---:|---:|---:|
|loop 3|−2.75e-12|−2.46e-11|+4.60e-12|+8.23e-12|
|loop 52|+4.10e-12|−1.05e-11|+1.14e-11|+1.13e-11|
|loop 71|−4.48e-13|−3.63e-12|+6.82e-13|+6.85e-13|

Accurate T3 reduces directional spread but does not remove the common error in
DP cut geometry/reduction. The loop 52 common bias resembles original Ninja's
bias. This is not evidence that a smaller direction difference alone certifies
accuracy, and no point is forced to pass. All callback inputs and both native
and high-precision rounded outputs are retained in `expansion-results.json`.
The attributed `tensor.rs` is the existing production generic tensor adapter.

From the repository root and pinned development environment:

```sh
source .ninjars-build/dev-env.sh
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python docs/benchmarkfolder/experiments/qcd3-point67-masters/reproduce.py build
python docs/benchmarkfolder/experiments/qcd3-point67-masters/reproduce.py run --cpu 207
python docs/benchmarkfolder/experiments/qcd3-point67-masters/weights.py build
python docs/benchmarkfolder/experiments/qcd3-point67-masters/weights.py run
python docs/benchmarkfolder/experiments/qcd3-point67-masters/expansion.py build
python docs/benchmarkfolder/experiments/qcd3-point67-masters/expansion.py run --cpu 207
```

Run the scalar and expansion diagnostics with no other Symbolica process active.
The scalar and expansion runs took about two seconds and less than one second,
respectively; these are correctness diagnostics, not performance measurements.
Each driver records executable/input/source hashes and the executed command.
The first build commands were retained from the shell and their hashes observed
after build, before execution, as explicitly marked in metadata.
