# Pinned source review, no execution

The practical first candidate is denominator-only native dispatch, at the same
Float256 working precision. Full native is feasible as a separately validated
arm with different complex reciprocal ordering. No native integration or full
cut-geometry replacement exists in the pinned factory. This is a source
feasibility conclusion, not a speedup or correctness acceptance result.

Relevant local primary-source locations:

- `ninjars/crates/ninjars-symbolica/Cargo.toml`: native-kernels implies
  rust-codegen; native-masters is independent.
- `src/amplitude.rs:128` and `:610`: defaults and native policy constructors;
  `:678` records actual family coverage and rank n+2 fallback.
- `src/native_kernels/mod.rs:115`: semantic denominator plans, explicit working
  precision constant context, reusable workspace and whole/bounded dispatch.
- `src/native_kernels/bubbles.rs:40`, `tadpoles.rs:34`: named catalog coverage.
- `src/kernel.rs:248`: installation validates dimensions and removes the
  interpreter only for that installed kernel; batch dimensions checked first.
- `src/rust_codegen.rs:117`: exportable exact plans are broader than installed
  native coverage; full cut/integration factories have no corresponding selector.
- `oneloopmaster/src/native/primitives.rs:239`: Float NativeFloat carrier;
  `:309` uses separately rounded complex products, `:337` scaled inverse.
- `oneloopmaster/src/native/constants.rs:28` and `batch.rs:231`: exact constants
  at explicit precision and value copies without input precision widening.
- `oneloopmaster/src/rust_codegen.rs:1`: instruction-graph emission preserves
  branches/shared bodies, with scalar execution for one row and batched storage
  for larger row counts.
- Pinned numerica `domains/float/complex.rs:466` and `:865`: interpreted complex
  multiply and inverse; reciprocal path differs from the full-native primitive.
- Upstream `tests/native_denominators.rs:147,:271,:312`,
  `tests/native_bubbles.rs:148,:260,:303`, and
  `tests/native_tadpole_dispatch.rs:135,:273`: Float256/tolerance tests, emitted
  source regeneration, exact tracked-precision equality, and shape validation.
- `tests/native_amplitude.rs:186,:301`: whole-amplitude status/reset/failure
  controls use192-bit Float examples; they do not supply a complete256-bit
  MadLoop replay. `tests/native_domain.rs` exercises native reciprocal domain
  behavior, including Float256 and visible nonfinite fixed-precision results.

All NinjaRS/oneloop paths above are relative to the read-only `../references`
clones whose revisions are fixed in PLAN.md. No native-kernel build or evaluator
was run for this review. The bridge's tensor source, working precision and
scalar on-shell threshold stay outside the proposed dispatch delta.
