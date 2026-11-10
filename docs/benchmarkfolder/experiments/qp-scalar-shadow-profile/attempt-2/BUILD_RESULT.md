# Corrected private build result

The single authorized attempt-2 build completed successfully in 709.0908 s
within its 900 s cap, with actual affinity CPUs 2 and 3 and two Cargo jobs.
The unsafe-code rejection from attempt 1 is resolved by the safe callback;
the core retains `forbid(unsafe_code)`. Existing core warnings and one
private overwritten-initial-value warning are retained in `build.log`.

Both Fortran replay executables compiled against the same frozen historical
`mtirust` module/archive. Their exact compiler commands and all artifact
hashes are in `build-result.json`. The concurrent production Fortran
unpack-failure fix is not included in these historical diagnostic drivers.

Reproduce the guarded build from the repository root:

```sh
source tools/tirust_env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
taskset -c 2,3 python docs/benchmarkfolder/experiments/qp-scalar-shadow-profile/attempt-2/build.py --root-reviewed-source
```

The exact Cargo invocation is retained in `build-result.json`: locked,
offline, release, poles-only, two jobs, and a unique private cdylib. This
record reports the completed invocation; it does not authorize a retry.

The source inventory, isolated source copies, historical Fortran runtime,
artifact hashes, and all 29 hash-enumerated attempt-1 evidence files were reverified after
the build. `readelf` and `nm` inspect the produced DSO without executing a
reducer; only uniquely prefixed `ml_tirust_*` symbols are exported.

**No native replay has run.** Original/counters-off/counters-on output,
status, cache, input-bit, host-flag, and private MPFR-flag equivalence remain
unqualified. Scalar-repeat or timing conclusions are unavailable. The
separate guarded replay launcher remains deferred pending artifact review
and explicit root authorization.
