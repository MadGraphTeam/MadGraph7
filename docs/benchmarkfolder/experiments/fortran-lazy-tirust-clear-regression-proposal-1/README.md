# Canonical generated-clear regression proposal

`regression.patch` extends `tests/unit_tests/loop/test_tirust_transport_cache.py`.
Production and canonical tests are not edited here. The existing actual-MTIRUST
module plus independent caching C-provider harness is reused; no Rust library or
real reducer is linked or executed.

The new test extracts the clear procedure from the live Fortran export template,
renders two subprocess prefixes and compiles it with the actual `mtirust.f90` at
both `-O0` and `-O3`. It checks fresh/post-finalize absence without native creation,
preserved inactive diagnostics/IEEE flags/rounding, explicit public clearing
remaining creating, real C-provider cached entries cleared by the other prefix,
and registration/recreation epochs. This tests behavior instead of matching the
guard's implementation text. The original unguarded template fails the absence
and diagnostic assertions; the production guard must be applied first.

The provider adds independent native create/destroy/register counters while
keeping its existing transport/error cases unchanged. The new fixture uses
successful QP transport to create a cache entry, proving that a prefix which
never registered a plan still clears the shared initialized module context.

Preparation performed only Python AST parsing, live-procedure extraction/render
checks and `git apply --check`. No compilation, C-stub execution or real-reducer
execution has occurred for this proposal. Root will apply and run it with the
reviewed production guard:

```sh
git apply docs/benchmarkfolder/experiments/fortran-lazy-tirust-clear-regression-proposal-1/regression.patch
source tools/tirust_env.sh
python3 -B -m unittest tests.unit_tests.loop.test_tirust_transport_cache -v
```
