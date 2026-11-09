# Pinned TIRust pole-only extension

`0001-direct-poles-only.patch` applies to the exact revision recorded in
`manifest.json`. The installer checks the original source hashes and patch
hash, copies `../pole-kernels` into the staged TIRust tree as
`madloop-pole-kernels`, and applies the patch with zero fuzz. Reference
checkouts are never patched.

The existing full `DirectWorkspace` remains the default. The new
`DirectPoleWorkspace` instantiates the same reduction with a compile-time
pole-only parameter. Scalar master constructors select generated exact
oneloopmaster roots `[-1,-2]`; no finite evaluator is constructed. The
parameter also follows double-double and small-Gram auxiliary executors,
linear recurrence, dimension factors, lightlike-bubble residues, gathers,
normalization and convergence tests. In particular, epsilon times an IR
double pole still contributes to the IR simple pole. Finite rational
terms are retained by full evaluation and omitted by pole-only execution.

`PoleBatchResult` exposes only `[UV1, IR1, IR2]` per Cartesian monomial.
Its private scratch representation cannot be converted into a public full
result. `PoleTensorStore` has a separate type and namespace, preventing
partial entries from being reused by a full workspace. Scalar masters and
tensor subsets retain the existing exact kinematic/precision keys inside
that namespace. Pole selection happens after reduction.

The public pole entry point admits ordinary, undotted `D=4-2epsilon`
integrals with at most six propagators and rank six. It preserves the
requested native numerical carrier. Unsupported requests return an error;
failed lanes remain failed and never silently use full evaluation.

The patch includes `tests/pole_only.rs` and focused internal tests for
epsilon mixing, untouched finite scratch, and absence of full master
evaluators. Create a lean test workspace against an already patched source
tree (the optional local oneloopmaster path must match its pinned revision):

```sh
python3 vendor/tirust/patches/prepare_test_workspace.py \
  /path/to/patched/TIRust .tirust-build/pole-tests \
  --oneloop /path/to/pinned/oneloopmaster
cargo test --manifest-path .tirust-build/pole-tests/Cargo.toml --test pole_only
cargo test --manifest-path .tirust-build/pole-tests/Cargo.toml --lib pole_
```

This avoids resolving TIRust's unrelated CLI, Python and symbolic host tools.
The tests cover all supported topology sizes, native precision carriers,
nonzero infrared poles, scaleless residues, normalization epsilon mixing,
separate cache identity, rotation and the actual auxiliary rescue paths.

The scalar crate has its own native-precision, branch and permutation
tests, plus generated-root provenance. This file describes the mechanism;
the benchmark report records the validation and timings actually run.
