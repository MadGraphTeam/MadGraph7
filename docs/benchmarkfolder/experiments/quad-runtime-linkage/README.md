# Preserve the GNU Fortran runtime when linking NinjaRS

The initial NinjaRS static-library link resolved the generated process's
`__addtf3`, `__multf3`, and other binary128 helpers to hidden symbols from Rust's
`compiler_builtins`. The untouched original Ninja executable instead used
versioned symbols from `libgcc_s.so.1`.

This changed observable compiler-runtime behavior. With upward IEEE rounding,
`REAL(16)` addition of `1 + 2^-114` returned exactly `1` under the Rust-first
link, while GNU Fortran returned the next representable binary128 number.
The ordinary nearest-rounding result agreed. This is a correctness issue
independent of the relative speed of the two implementations.

The validated GNU/Linux fix places `-lgcc_s` before `libninjars.a`, preserving
the Fortran compiler's runtime resolution. Cargo's remaining native-library
flags still follow the archive. The installer records this distinction in
`link_before`, `link_order_version: 1`, and the two installed linker flag files.
Both standalone and NLO exporters consume that ordering; internal dependency
exports copy the metadata. Existing metadata without the ordering requirement
is rejected with a reinstall instruction. The C ABI remains version 2.

Reproduce the helper checks after installing the bridge:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/quad-runtime-linkage/verify.py
```

`verification.json` records the compiler, target, commands, archive hash and
resolved helper symbols. The probe compares baseline GNU Fortran, Rust-first,
and GCC-first executables across four rounding modes. It covers real arithmetic,
comparisons and conversions, complex multiplication/division, very small and
large inputs, and IEEE exception flags. The GCC-first output matches the
baseline bit for bit; the Rust-first output differs. These probes do not invoke
Symbolica.

`qcd4-relink.json` and `qcd4-resolved-helpers.txt` independently record the same
resolution in a complete generated qcd4 executable. The installed Fortran smoke
test also checks upward binary128 rounding before its actual DP and MP scalar
tests. The installer lifecycle tests verify prefix placement and rejection of
missing metadata. This evidence supports GNU Fortran on Linux; it does not
establish a runtime policy for other platforms.

The small `quad-runtime.f90` arithmetic loop was exploratory only. One noisy
sample did not establish that Rust's arithmetic was slower, and no performance
claim or acceptance result is based on it.
