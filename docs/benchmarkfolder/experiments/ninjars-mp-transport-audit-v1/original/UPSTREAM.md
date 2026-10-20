The bridge uses the pinned NinjaRS and Symbolica revisions in `Cargo.toml` and
`Cargo.lock`. NinjaRS is GPL-3.0-or-later; this bridge uses the same license.

`src/guarded.rs` adapts the tensor indexing and transposed recurrence traversal
from NinjaRS commit `0d0d7e7a8033431b60a8580731084742e0655cda`,
`crates/ninjars/src/fast/{tensor.rs,tensor_layout.rs,tensor_jets.rs}`. Its added
FMA product residuals and compensated sums apply only to the explicitly
requested DP retry. This is not a universal correctly-rounded polynomial
guarantee. Unsupported layouts, exponent ranges and floating-point environments
use the original callback, and MadLoop retains independent stability checks and
genuine higher-precision rescue.

`tests/data/compensated` contains six generated MadLoop QCD3 tensor requests,
their actual Float256 callback references, and compensated full-reduction
results. They were captured during the retained point-67 diagnosis. Complete
commands, input provenance, independent Ninja comparisons and sealed hashes
are recorded under `docs/benchmarkfolder/experiments/qcd3-point67-{dp,masters,t3}`
and `t3-dp-retry`. These fixtures do not require sibling reference clones.

The `fenv-test-support` Cargo feature compiles test-only floating-point flag and
rounding mutators. Normal installation leaves this feature disabled and builds
only the environment observer. `make check` enables it for the serial test
suite.
