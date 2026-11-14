# Scalar pole kernels for MadLoop

This crate provides the scalar master part of TIRust's accelerated pole-only
execution. `ScalarPoleEvaluator<T>` returns exactly two coefficients,
`[epsilon^-1, epsilon^-2]`. It never computes a finite coefficient. UV and IR
origin separation belongs to TIRust's tensor executor; these scalar outputs are
the total poles, including the cancellation in a scaleless bubble.

The committed Rust kernels are generated from oneloopmaster revision
`bd703d09a78f7849a6b86a07096e6f222fc01193`. The crate builds against
`2e7afabb5903814196d7ab39c695ed9cac777aa4`, the revision TIRust `a5d4afb` uses; it
changes only oneloopmaster's Symbolica dependency, Python bindings and one test, so
`src/` and the generated kernels are unchanged. The generator requests only the Laurent
tags `[-1, -2]` through the public `OneLoopExpressions::evaluator` and
`RustEmitter` APIs. The kernels use oneloopmaster's public native numeric runtime.
No master formula, numerical primitive, or Symbolica source is copied or patched.
The generated manifest records graph sizes and confirms that there are no
finite roots. Generation rejects a graph containing a dilogarithm call.

`T` can be `f64`, Symbolica `DoubleFloat`, or Symbolica `Float`. The input order
is oneloopmaster's ordinary scalar order, with squared renormalization scale
last. Constructors reject unsupported families and invalid precision. Evaluation
validates shape, finite components, real invariants, nonpositive mass imaginary
parts, and a positive real scale before touching output. D0 uses the public
`regular_d0_labelling` operation to preserve soft poles under propagator
relabelling. Arbitrary-precision inputs are evaluated at the selected working
precision. Nonfinite residues are explicit errors for the caller's rescue path.
This includes a completely collapsed massless D0, for which the upstream exact
expressions divide by zero; this crate does not supply a new limiting formula.

From the MadGraph checkout, regenerate and test through the standalone
maintenance workspace:

```sh
source tools/tirust_env.sh
cargo run --locked --release --manifest-path vendor/tirust/pole-kernels/maintenance/Cargo.toml --bin generate-poles -- vendor/tirust/pole-kernels/src
cargo test --locked --manifest-path vendor/tirust/pole-kernels/maintenance/Cargo.toml --test scalar_poles
```

The maintenance workspace pins the same Symbolica revision as TIRust and
compiles the production source directly. Installed builds instead use the
embedding workspace's dependency patches. Generated source is committed so
installation does not run symbolic code generation. The
`benchmark_poles` example emits warm scalar-only timings in CSV; they do not
represent an end-to-end MadLoop benchmark.

Tests compare pole-only output with full oneloopmaster evaluation in binary64,
double-double, and 192-bit MPFR arithmetic, including scaleless integrals,
threshold and small-momentum bubbles, massless triangle permutations, soft and
collinear box permutations, and complex masses. Invalid inputs must leave the
caller's output unchanged.
