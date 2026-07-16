# Legal Release Status

The original pyAmpliCol and Rusticol sources are licensed under 0BSD. The
shipped `pyamplicol._rusticol` extension and `librusticol_capi.a` use a
f64-only Rust feature closure containing SymJIT but not Symbolica, Rug/GMP, or
Malachite. Symbolica remains a separately installed generation and
higher-precision Python dependency under its own terms.

The authoritative gate is
`licenses/STATIC_LINK_COMPLIANCE.toml`. The strict release check is:

```console
python tools/release/check_rust_licenses.py --mode release
```

It derives the native closure independently for every release target. The
wheel audit also scans the extension and static archive for GMP, MPFR, and
Malachite markers. A future feature change that makes an LGPL family reachable
will fail this gate unless complete target-specific compliance evidence is
provided. Development can run the non-publishable candidate check:

```console
python tools/release/check_rust_licenses.py --mode candidate
```

Candidate success never implies publication readiness: JSON output reports
`candidate_ready` and `release_ready` separately, and release mode is the
default. Other release blockers, including exact published dependency builds,
artifact hashes, target fixtures, and deployment tests, remain independent of
this native-license gate.
