# Proposed real-provider transport boundary

`real-provider.f90` is an unexecuted diagnostic proposal. The regression proof
for the cache fix is the compiled mocked-provider test
`tests/unit_tests/loop/test_tirust_transport_cache.py`, which links the actual
Fortran module without Rust or a reduction library.

The proposed native driver first seeds a valid QP tadpole cache entry, then
repeats an N=1, rank=0 full evaluation with binary128 mass squared near
`1e4932` and scale squared 1. Its finite tadpole term is expected to exceed
binary128's range while remaining finite in MPFR. The real provider may refuse
the extreme kinematics earlier; that outcome must be retained and does not
demonstrate the public transport failure. The driver prints the exact decimal
inputs, public error, named IEEE flags and native QP cache statistics for every
call. It disables traps only for this diagnostic and restores their original
settings afterward.

After installation qualification is authorized, compile and link this same
source independently against each frozen old/new module and archive, with the
corresponding DSO and runtime-library identity recorded. Execute serially under
the shared numerical lock. Preserve both logs and report provider refusals.
Only the explicit public transport error permits a cache-invalidation
comparison; native cache clearing resets its counters, so zero counters alone
do not measure re-evaluation. No native result or timing claim is recorded here.
