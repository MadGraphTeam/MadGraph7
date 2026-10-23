The bridge uses the pinned NinjaRS and Symbolica revisions in `Cargo.toml` and
`Cargo.lock`. NinjaRS is GPL-3.0-or-later; this bridge uses the same license.

The final dependency cohort pins NinjaRS
`41a4c00e7635e3ab188fc26bacf0aae6941e8082`, oneloopmaster
`5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5`, and the coupled
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
The benchmark report retains its original measured dependency cohort; this
upstream pole-selection update is qualified separately in `tests/poles`.

Every new installation records its exact manifest and resolved lock hashes and
all transitive Git sources in `lib/ninjars-config.json`, alongside copies of
`Cargo.toml` and `Cargo.lock`. Explicit local overrides additionally record the
source revision and a digest of the copied source files, so a modified checkout
is distinguished from the same clean revision. Reference checkouts are copied
before building. The pole-selector interface requires bridge ABI 4; older
ABI-3 installations are rejected before export or linking. After updating the
MadGraph checkout with `git pull`, run
`install ninjars --update` to rebuild an older installation with that checkout's
current pins. This uses the committed dependency revisions and lockfile;
it does not select unpinned upstream `main` revisions. The installer validates
the replacement before publishing it and preserves the previous installation
if the build or numerical smoke test fails. `--force` remains an equivalent
backward-compatible alias.

When a usable NinjaRS installation is configured, ordinary optimized one-loop
standalone and NLO exports start with `MLReductionLib=8|6|7|1`. Without it,
the default remains `6|7|1`. Ordinary `check timing` and `check profile` select
NinjaRS by default, or Ninja when NinjaRS is unavailable. Explicit reduction
options such as `--reduction=6` and active parameter-card choices take
precedence. Loop-induced exports retain their existing `7|6|1` selection and
scalar-integral pole settings.

`LoopPoleMode` selects the poles of the bare loop contribution through a common
card option: `0` returns both origins, `1` UV poles, `2` IR poles,
and `3` no poles. Its default, `-1`, preserves the existing backend behavior,
including `COLLIERComputeUVpoles` and `COLLIERComputeIRpoles` for COLLIER.
Modes `1`, `2`, and `3` require a reduction/rescue list containing only NinjaRS
and/or COLLIER; other backends cannot provide the requested pole selection.
Finite coefficients and rational terms are unchanged; model counterterms are
outside this selection. MadLoop forwards the selection to the upstream
NinjaRS master providers, which use oneloopmaster's public pole-selection API
before contraction with the reduction coefficients. The bridge retains only
ABI mode validation, with no local numerical pole-projection formulas.
UV-only and IR-only modes use the generic NinjaRS
engine in both double precision and higher-precision rescue, retaining the
scaleless-bubble coefficients whose UV and IR residues cancel in the combined
result. Their compensated-DP entry uses the same generic DP path, without the
fast-engine compensated numerator callback; higher-precision rescue remains
available. Combined-pole and no-pole DP calculations retain the fast engine.
All explicit modes (`0`–`3`) bypass existing helicity/loop-filter decisions so
that cancelled combined-pole contributions remain available for separation;
default mode `-1` keeps the usual filtering policy. Explicit modes can
therefore incur additional numerator and reduction costs. The committed
dependency revisions and lockfile include the upstream pole-selection APIs.
See [Selecting UV and IR poles in MadLoop](../../doc/loop-pole-selection.md)
for card examples, the Fortran setter, and the scope of the returned poles.

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
are preserved in the [immutable measurement archive](https://github.com/MadGraphTeam/MadGraph7/tree/19d67c8605337c176680878a521b3a014176a7fb/docs/benchmarkfolder/experiments)
under `qcd3-point67-{dp,masters,t3}` and `t3-dp-retry`. These fixtures do not
require sibling reference clones. The final measured report and its offline
reproduction instructions are in [`doc/benchmarkfolder`](../../doc/benchmarkfolder/README.md).

The `fenv-test-support` Cargo feature compiles test-only floating-point flag and
rounding mutators. Normal installation leaves this feature disabled and builds
only the environment observer. `make check` enables it for the serial test
suite.
