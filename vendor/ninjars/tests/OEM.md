# Scoped Symbolica license validation

Run the retained Fortran lifecycle test against the installed NinjaRS library:

```sh
python3 vendor/ninjars/tests/oem.py \
  --lib HEPTools/ninjars/lib \
  --output .ninjars-build/oem-lifecycle
```

For a direct bridge build, `make -C vendor/ninjars check-oem` builds the library
and runs the same test. Override `PYTHON` and `OEM_TEST_OUTPUT` when needed.

The harness removes `SYMBOLICA_LICENSE` from each child's environment without
changing the caller's environment or home directory. Two processes start
together and each retains its numerical worker at a barrier. Every process
must reach that barrier before either can continue. This distinguishes a
working library license from two serial evaluations that happen to fit the
restricted one-thread allowance.

Each process performs three complete create/evaluate/finalize cycles. The
checks include:

- Combined and compensated double-precision tadpoles, with changing inputs.
- Selected UV/IR double-precision reductions that initialize Symbolica-backed
  masters on the retained worker.
- Binary128-input rescue with a coefficient containing `2^-100`, verified in
  the result beyond binary64 precision.
- Eight interleaved combined, UV, IR and no-pole evaluations of a lightlike
  scaleless bubble in both precisions, checked against its analytic residues.
- Finalization, repeated finalization and subsequent worker recreation.

`summary.json` records the library and test-source hashes, child exit codes,
barrier timestamps and overall result. Build and per-process logs remain in
the selected output directory. An installed `ninjars-config.json` is also
hashed when available. No license key or environment dump is recorded.

This test verifies that the actual Fortran interface and its workers can
execute concurrently without a personal license. Upstream NinjaRS's license
scope tests separately check that the library key does not license unrelated
Symbolica operations outside a NinjaRS operation.

For broader numerical coverage against retained independent tensor references:

```sh
env -u SYMBOLICA_LICENSE python3 vendor/ninjars/tests/poles/tensor_validation.py \
  --lib HEPTools/ninjars/lib \
  --output .ninjars-build/oem-tensors --focused --precision both
```

That test covers 61 tensor components, eight pole-mode transitions and both
precisions. Its ordinary combined-DP diagnostics are retained as documented
in [poles/README.md](poles/README.md); it does not silently relax their numerical
thresholds. The independent reference data are already committed, so running
this check does not require building or executing COLLIER.
