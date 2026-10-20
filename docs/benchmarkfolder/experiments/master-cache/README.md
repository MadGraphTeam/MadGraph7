# Isolated exact scalar-master cache prototype

This experiment changes no production library or installed dependency.
The working bridge is copied under
`.benchmark-work/experiments/master-cache/bridge`. The retained patch and
`base.json` identify its inputs. The release library builds and all nine focused
tests pass; numerical performance measurements are pending. A first test
compilation was stopped before evaluation to leave the acceptance benchmark
uncontended.

`python docs/benchmarkfolder/experiments/master-cache/reproduce.py` reconstructs
the draft in a new ignored directory, verifies the production source hashes
before applying `prototype.patch`, then verifies every resulting prototype
source hash. It performs no build or evaluation.

The motivation is the
[normal qcd4 rescue profile](../mp-qcd4-point81/README.md): scalar masters account
for about 15.5% of NinjaRS's sampled two-call work, including a material cost on
the second call. Both existing NinjaRS providers retain generated evaluators but
do not retain numerical master values. Original Ninja's configured integral
cache retains exact-argument values within MadLoop's normal cache scope.

The prototype uses a separate ABI 3 and library name. Each context owns
independent DP and MP caches, with at most 4,096 entries in each and FIFO
eviction. Only successful finite Laurent results enter the cache. Entries key
the master family and every argument, including squared scale. DP keys retain
all binary64 bits, including signed zero. MP keys retain exact Float values,
tracked arithmetic precision, and signs, including zero signs. A provider owns
an immutable working precision and on-shell threshold, so these also define
its namespace. The pinned Float's own Hash/Eq omit precision and zero sign;
the extra key fields are required.

Scope behavior:

- `ninjars_point_begin` advances the epoch, clears existing values and opens
  one full amplitude scope. Calling it again replaces that scope.
- `ninjars_point_end` clears values and closes the scope. Repeated end is safe.
- `ninjars_clear_cache` advances the epoch while preserving whether the scope
  is open. It is intended for existing MadLoop rotation/cache invalidation.
- Without an open scope, each C reduction receives a fresh epoch and clears its
  values before returning. Repeated standalone C calls cannot retain values
  across points accidentally.
- Begin/clear/end never create MP state or start the Symbolica worker. A later
  first MP evaluation inherits the current epoch and enabled setting.
- The intended production integration is an unchanged public
  `SLOOPMATRIX_THRES` wrapper around a private implementation, with begin before
  and end after the private call. This guarantees closure despite its internal
  early returns. That integration has not been installed.

The cache also observes the floating-point environment through a small GNU C
shim. Rounding mode and the enabled exception-trap mask enter the exact key,
along with non-status MXCSR controls and the x87 control word on x86, or FPCR
on AArch64. This covers FTZ/DAZ and independently adjusted hardware rounding.
Other architectures conservatively disable keying until their control
registers are supported.
If sticky exception flags differ from those observed after the previous
provider attempt, all values are invalidated before lookup. Clearing flags
therefore forces recomputation. Failed environment observations disable reuse.
No exception flags are synthesized or replayed. Observation is local to the DP
caller thread or the MP worker, matching where the provider executes.

Requests containing repeated exact rows evaluate a unique miss only once.
Cache hits and newly computed finite successes are staged before output
commit. A failed row retains its caller's output buffer. Strict MP evaluation
does not commit any output when one row fails, although statistics and
independently successful finite cache entries may advance. Cache state is not
part of output transactionality.

A caught DP engine panic discards DP state inside the held mutex guards,
preserving the shared scope and independent MP domain. A caught MP engine
panic removes that context's MP state. Both paths leave C output buffers
untouched and permit later calls to rebuild fresh state.

Prototype-only controls expose cache enable/disable and eight counters:
requested rows, hits (including duplicate rows in a batch), unique misses,
insertions, evictions, clears, failed rows, and current entries.
Focused regressions cover exact enabled/disabled values, mass/scale/invariant
changes, scope boundaries and late MP creation, failures and malformed buffers,
signed zero/precision, bounded eviction, rounding/flag transitions, mixed
hit/miss/failure batches, and panic recovery. All nine passed in 1.06 seconds
after compilation; see `focused-test-build.json` and `focused-tests.txt`.
The first run's eight passes and one failed hit expectation are retained in
`focused-tests-first.txt`. Initial MP formatting raised `FE_INEXACT` after the
first provider call, correctly invalidating that first entry. The corrected
test allows this observed environment transition and requires a hit on the
third identical reduction, with identical output throughout.

The focused test command is:

```sh
source .ninjars-build/dev-env.sh
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
export CARGO_TARGET_DIR="$PWD/.benchmark-work/experiments/master-cache/target"
cargo test --release --locked --offline \
  --manifest-path .benchmark-work/experiments/master-cache/bridge/Cargo.toml \
  master_cache -- --test-threads=1
```

`CARGO_BUILD_BUILD_DIR` may point at the existing task dependency cache to reuse
compilation artifacts. This does not change source or numerical state. Only
one MP process may run at a time under the available Symbolica license.
Full generated-process timing will follow correctness and independent review.
No speedup or acceptance claim follows from the successful tests.

The subsequent full bridge suite also passed all 18 tests in 4.21 seconds,
including the retained complex-mass pentagon at 256 and 384 bits
(`all-tests.txt`). The isolated qcd3 process probe then evaluated points
1, 44, 67 and 80 in cache modes off/on/on/off, after warming all four points.
Every printed coefficient, accuracy and status matched exactly. The three
rescued points retained status 322, and both caches were empty before and
after every full amplitude.

| Point | Status | Median cache-on / cache-off time | DP row hits | MP row hits |
| --- | --- | ---: | ---: | ---: |
| 1 | 218 | 1.0795 | 62.0% | — |
| 44 | 322 | 0.9607 | 59.0% | 87.3% |
| 67 | 322 | 0.9741 | 67.5% | 86.7% |
| 80 | 322 | 0.9598 | 58.9% | 87.3% |

These are two samples per mode on four selected points, with no confidence
interval or acceptance claim. The DP sample regressed despite many hits;
the whole-rescue improvement was only 2.6–4.0%. This allocating-key prototype
is therefore **not promoted**. All counters and times are retained in
`qcd3/results.json`; `process-probe.py` rebuilds the isolated wrapper and
driver. The next experiment removes per-hit allocations and adds an MP-only
mode whose DP provider bypasses lookup.

Pinned implementations: NinjaRS
`0d0d7e7a8033431b60a8580731084742e0655cda`, oneloopmaster
`09b2c7497743db875ed446da95123175dd00b1c8`, Symbolica/numerica
`e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`. The last source's
`lib/numerica/src/domains/float/multiprecision.rs` defines the exact Float
identity used here. The read-only original Ninja source's
`src/integral_cache.hh`, `src/avholo_interface.cc` and `include/quadninja/avholo.hh`
document the baseline's exact integral keys and cache clearing behavior.
