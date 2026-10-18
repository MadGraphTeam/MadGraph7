# NinjaRS consumer boundary

This standalone Rust crate exercises the actual reducer and scalar master
provider before they are integrated into the application's shared core. It
does not generate physical amplitudes or implement a second reduction method.

`PolynomialReducer<T>` binds lane-major polynomial parameters and carries
`Complex<T>` through NinjaRS, Symbolica and the master provider. `T` can be
`f64`, `DoubleFloat`, or arbitrary-precision `Float`. Its fifth polynomial
variable is explicit μ², with weighted rank two. `FastReducer<N>` accepts
NinjaRS's tensor numerator or its generic polynomial adapter in binary64.
All numerical stages use the upstream batch interfaces; this does not claim
hardware SIMD for every stage or arithmetic mode.

Both adapters replace results on each call, retain caller lane IDs, and return
no numerical value for failed lanes. A structural error returns `Err` for the
whole call. Parameter rows must follow the supplied lane order, including
compacted retry batches. The adapters do not perform precision rescue or
convert inputs through binary64.

`Laurent` names the finite, single-pole and double-pole entries. Its epsilon
convolution combines the separately reduced numerator sectors N0 + εN1 + ε²N2
through finite order. All values retain Ninja/OneLOop normalization:
μ_R^(2ε)/(i π^(2−ε) rΓ). Model prefactors and the public interference
normalization have not been applied. In particular, the reducer's rational
diagnostic is already included in the finite coefficient.

## Dependency setup and checks

The ignored `upstream` symlink points to a NinjaRS checkout with its own
dependency references populated. This temporary standalone layout keeps the
application's ordinary checkout buildable while upstream packaging is pending.
Its Symbolica version is aligned with the application's shared core.
The implementation workspace uses:

- NinjaRS `89156ddfa7d4f08f27386d53c0fd985d56ec321d`.
- Symbolica `e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`, the owner's integration
  revision published on `https://github.com/ValentinHirschi/symbolica`.
- FeynKit `474478a7a80bc9f9f21bead0e1cde6acbcffbe93`, the owner's patched
  dependency revision recorded by NinjaRS (not asserted merged upstream).
- oneloopmaster `d10eaa073aab3acc68feab9b04866532241c1443`.

From this crate's directory, using the shared workspace environment:

```sh
ln -s "$PYLOOP_WORKSPACE/references/ninjars-consumer" upstream
export CARGO_TARGET_DIR="$PYLOOP_WORKSPACE/builds/loop-reduction"
export RUST_MIN_STACK=134217728
cargo test --locked --test anchors -- --test-threads=1
```

Supply a valid `SYMBOLICA_LICENSE` in the process environment when needed;
never persist it in this crate. Cargo's lockfile covers registry dependencies.
The upstream sources are read-only to this consumer.

The analytic tests exercise A0 with real and complex squared masses, massless
B0, the divergent massless C0, μ² C0, and a four-dimensional q² numerator
whose nonzero rational term requires auxiliary μ² expansions,
all three arithmetic carriers, parameter rebinding, compact and empty batches,
retained input bits beyond binary64, replacement semantics, failed lanes,
and epsilon-sector ordering. They establish only these anchors, not the full
scalar-integral domain or process-level correctness.

`cargo test --locked --test diphoton_fixture` connects the independent Community
dimensional-algebra fixture in `tools/loop_reference/diphoton_r2.json` to the
reducer. It preserves rational inputs and verifies the evanescent remainder in
all three arithmetic modes, including changing batch sizes and epsilon-sector
assembly. This internal fixture does not enable loop-induced production.

## Experimental native Python consumer

The `python` member builds `_loop_reduction`, a developer-only PyO3 extension.
It links the real shared core and reuses its production `Evaluator.save()`
decoder. `CoefficientEvaluator` maps exact coefficients to the chosen carrier
and calls Symbolica's instruction-batched evaluator. `PolynomialPlan` reduces
three explicit numerator epsilon sectors and combines their Laurent values
inside Rust. Both release the Python GIL during batch evaluation.

```sh
cargo build --locked -p rusticol-loop-python
mkdir -p "$PYLOOP_WORKSPACE/builds/loop-python-module"
cp "$CARGO_TARGET_DIR/debug/lib_loop_reduction.so" \
  "$PYLOOP_WORKSPACE/builds/loop-python-module/_loop_reduction.so"
PYTHONPATH="$PYLOOP_WORKSPACE/builds/loop-python-module" \
  python -m pytest python/test_consumer.py -q
```

Precision 16 uses binary64, precision 32 uses true 106-bit DoubleFloat, and
higher requested decimal precision uses an MPFR-backed Float. Decimal input
pairs avoid binary64 conversion; explicitly sized decimal output retains enough
digits to round-trip each carrier. Buffers are flattened lane-major complex
pairs, with four pairs per momentum and one original ID per lane. The three
returned entries are **finite, single pole, double pole** in raw integral
normalization. Failed lanes carry `None` instead of numerical values.

This is a native batch conformance boundary, not a public virtual-amplitude
API or a completed fast evaluation pipeline. Production coefficient-DAG
scheduling, buffer reuse across Python boundaries, physical normalization,
precision rescue, and the binary64 fast reduction path remain integration work.
The main workspace excludes this experimental workspace and has no NinjaRS path
dependency, so its normal builds do not require the `upstream` symlink.

## Remaining gates

- The generic upstream coordinator accepts at most six propagators; selected
  seven-propagator high-multiplicity cases need owner support and validation.
- Rank is limited to n+1 by NinjaRS. No numerator term may be truncated to fit.
- The binary64 native master provider fixes its on-shell snapping threshold
  at 1e-9. The generic constructor requires an explicit threshold policy;
  `None` disables snapping. These policies can differ for small invariants.
- Broad B0/C0/D0 analytic continuation, threshold and complex-mass validation
  remain upstream readiness gates. Passing A0 and μ² C0 is insufficient.
- Native batching is established by the called interfaces, but end-to-end
  throughput, SIMD coverage and the MadLoop/Recola/OpenLoops performance
  targets require physical process benchmarks after integration.
- Explicit μ² reduction is only a numerical prerequisite for direct R2.
  Correct regularized numerator generation, epsilon sectors, UV terms and
  scheme alignment are separate requirements.
- Dependency licensing and packaging, including oneloopmaster's unpublished
  development status, must be resolved before distributing this stack.
