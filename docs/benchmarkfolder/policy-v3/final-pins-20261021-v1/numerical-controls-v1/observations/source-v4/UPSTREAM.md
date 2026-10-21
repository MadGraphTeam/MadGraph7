The bridge uses the pinned NinjaRS and Symbolica revisions in `Cargo.toml` and
`Cargo.lock`. NinjaRS is GPL-3.0-or-later; this bridge uses the same license.

The final dependency cohort pins NinjaRS
`aa6a6307dd081dc3714f9accebc0df9e5561ae73`, oneloopmaster
`8ec256fc26cb361b20e10f59c19beb35717ef32c`, and the coupled
Symbolica/numerica/graphica revision
`9cdc2949cdb153478c1eca0d40690dc0abfc616e`. The consuming bridge owns the
Symbolica patches because Cargo does not inherit dependency workspace patches.
NinjaRS's transitive gammaloop kinematics and Symbolica helpers are locked at
`d1904bf2bd5bf4f26969a73b415095e941a55610`; this also advances their earlier
`1e8063a763d3aa515749c33616ee59aa827705be` source revision.
Both native binary64 masters and generic higher-precision masters come from
this oneloopmaster revision. NinjaRS includes the matching generic soft-box
relabelling and precision-provider changes; a pin-only oneloopmaster update
against the earlier NinjaRS revision is not equivalent.

Every new installation records its exact manifest and resolved lock hashes and
all transitive Git sources in `lib/ninjars-config.json`, alongside copies of
`Cargo.toml` and `Cargo.lock`. Explicit local overrides additionally record the
source revision and a digest of the copied source files, so a modified checkout
is distinguished from the same clean revision. Reference checkouts are copied
before building. Existing compatible ABI-3 installations remain readable;
after updating the MadGraph checkout with `git pull`, run
`install ninjars --update` to rebuild an older installation with that checkout's
current pins. This uses the committed dependency revisions and lockfile;
it does not select unpinned upstream `main` revisions. The installer validates
the replacement before publishing it and preserves the previous installation
if the build or numerical smoke test fails. `--force` remains an equivalent
backward-compatible alias.

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
