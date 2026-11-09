# Generated Fortran adapter cache qualification

All 59 records passed: 16 uncached references and 43 subsequent cache actions.
Every evaluated tensor coefficient agrees with its same-precision, same-pole-mode,
same-loop-order uncached reference at the retained decimal precision (maximum
scaled residual zero). The acceptance thresholds were `1e-10` for DP and `1e-24`
for QP. Seven focused protocol/parser tests also passed.

The driver links the compiled candidate `qcd_n1_uux_z` generated Fortran adapter.
Its deliberately supplied diagnostic tensor is a massless triangle of rank 2,
with shifts `(0,0,0,0)`, `(5,0,0,5)`, `(10,0,0,0)`, and `mu²=7`.
These are tensor diagnostics, not an incomplete physical process amplitude.
The fixture has nonzero UV and IR coefficients, so changing the pole selection
does exercise different returned values. Independent reversal calls use their
own CT2/CT5 uncached references because reversing the propagator order changes
the tensor origin.

The retained sequence verifies:

- DP full and poles-only stores populate independently, then reuse entries.
  Together they retain eight entries and 1,440 reported bytes for this fixture.
- Six QP keys, comprising full/poles-only evaluation and both/UV/IR pole modes,
  each miss once and then hit, retaining six entries and 58,464 reported bytes.
- DP and QP operations preserve the other precision's entries and counters.
  Returning from poles-only to full evaluation restores finite values; every
  poles-only finite slot is explicitly `NaN + 0i`.
- `TIRustGlobalCache=0`, a topology limit below the fixture's three propagators,
  and independent CT2/CT5 checks preserve the retained numerical cache. QP
  bypass counts increase as expected.
- A 64-byte total budget refuses QP insertion twice without retaining any bytes.
  A zero-byte budget refuses both DP full and poles-only insertions. Restoring
  the default budget repopulates the stores; clearing resets entries and
  statistics; finalizing lets the generated adapter create a new context and
  reproduce the same coefficients.

The budget assertion concerns the public numerical cache accounting, including
its entry allowance. It is not a bound on process RSS, reduction plans or working
arrays. Public statistics expose cache activity, not counters of finite
evaluations: the finite NaN mask and pole agreement alone do not prove skipped
arithmetic. The separate Rust profiling/fast-path evidence covers that claim.
No performance claim or timing acceptance is made by this diagnostic.

Reproduce from the repository root after sourcing `tools/tirust_env.sh`:

```sh
python3 tools/tirust_benchmark_cache.py .benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z --label NEW_LABEL
python3 -m unittest tests.unit_tests.loop.test_tirust_benchmark_cache
```

The runner clones the built output into ignored working space, requires original
source/build provenance, checks compiler equality and dependency identity,
and serializes execution with `.tirust-build/numerical-tests.lock`.
`compilation.json` binds the driver, process sources, actual compiler flags,
linked archives, resolved shared libraries and parameter card. The TIRust DSO
SHA-256 is `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
`cache-values.dat` retains every raw coefficient and before/after counter array;
`results.json` retains every assertion result; `specification.json` and the
copied protocol sources define the actions. `manifest.json` hashes all portable
evidence files other than itself.
