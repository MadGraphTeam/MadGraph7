# Compressed calls lose or retain a wrong imaginary return component

Affected SymJIT: 2.25.0, revision
[`f1c193d301897149de6609f706297b0c97a4f018`](https://github.com/siravan/symjit-crate/commit/f1c193d301897149de6609f706297b0c97a4f018).
This is a separate call-boundary bug, exposed more often after correcting
[real-input locations](../real-input-locations/README.md).

## Two minimal failures

The caller's previous return-register value must not determine the type of a
new compressed-helper result. Tiny native programs reproduce both failures:

| Previous temporary | Helper returns | Incorrect result |
| --- | --- | --- |
| Real | `3+4i` | `3+0i` |
| Complex, imaginary part 4 | `7+0i` | `7+4i` |

The identical inline programs are correct. Both failures occur in scalar and
SIMD evaluation, including after serialization and reload.

`mre.py` contains four tiny portable MIR programs, not native machine code.
It loads them through the C API of an existing SymJIT library, without Cargo,
Symbolica or pyAmpliCol imports. The real value is parameter zero, whose marker
is recognized with either location convention; this reproducer does not require
the separate real-input-location repair:

```sh
python mre.py /path/to/libsymjit.dylib --expect-bug  # before the fix
python mre.py /path/to/libsymjit.dylib             # after the fix
```

Several library paths may be supplied. The test checks inline/called versions
at one and two input rows. The two defects are also
covered by numerical Rust tests in `fix.patch`.

## Cause and fix

`Complexifier::call_funclet` retains the old real/complex classification of
`Reg::Ret`. Saving a new complex result then inserts an imaginary zero if the
previous temporary was real. Conversely, a helper producing a real result
returns without zeroing its unused imaginary register.

The two-method fix marks `Reg::Ret` complex after the call, without writing
any register, and materializes a complete complex value before helper return.
Zeroing before the call would be wrong: argument setup has already populated
registers which overlap the return registers. Ordinary calls are unchanged.
Previously saved compressed MIR can be repaired on reload; no regeneration
or format migration is required for this particular bug.

Apply `fix.patch` after the real-input-location patch, then run:

```sh
cargo test --no-default-features --jobs 2 --lib compressed_return_tests
```

The focused tests pass on Apple ARM64, covering inline controls, both return
types, scalar/SIMD calls and save/load. Before the fix, both called cases fail
with the exact values in the table. The installed pre-fix C-API reproducer
independently shows the same failures. A compressed pyAmpliCol top-pair process
exposed the issue with a relative numerical error of `5.116e-4`; its
uncompressed and higher-precision evaluations agree.

After rebuilding and installing both native libraries, the C-API reproducer
passes through pyAmpliCol's RustiCol library and Symbolica's embedded SymJIT.
The same top-pair process now agrees with its uncompressed reference within
`3.4e-16` relative error in double precision, for batches of 1, 2 and 128
points; the 32-digit evaluation agrees within `1.7e-15`. All results are finite.
