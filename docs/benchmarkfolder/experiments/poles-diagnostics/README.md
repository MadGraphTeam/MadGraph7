# Generated tensor pole diagnostics

The final installed release bridge was exercised through the actual generated
MadLoop Fortran tensor interfaces. These are artificial tensor fixtures in
compiled `u u~ > z` and `u u~ > z z g g` process exports, not partial physical
amplitudes or performance acceptance measurements. Each compilation record
binds generated sources, compiler recipes, linked objects, parameter card,
executable and resolved shared libraries. The TIRust DSO SHA256 is
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.

The two runs contain 108 full-versus-partial pairs: 96 pass, and 12 DP pairs
return `STABLE=false` in both full and partial evaluation. Every one of the
54 QP pairs passes, as do all 42 served DP pairs. The largest served scaled
pole residual is 3.4694469519536141888e-18. Unevaluated finite coefficients
have a NaN real part and zero imaginary part; evaluated pole components are
finite. DP and QP comparison bounds are respectively 1e-10 and 1e-24, using
`abs(a-b)/max(1,abs(a),abs(b))` on each real and imaginary component.

The rank-two run covers ten fixtures with N=1 through 6, including complex
masses, scaleless UV/IR cancellation and a massless infrared triangle. The
second run uses the eight original rank-three/four pole fixtures, N=1 through
4. No fixture was omitted for rank capacity. The DP unavailable cases are the
massive triangle at ranks two and four and the massive five/six-point rank-two
fixtures, in all three pole selections. Their QP versions succeed. These
statuses remain failures of complete qualification; no particular underlying
conditioning gate is inferred from the public adapter's boolean status.

Independent analysis checks that UV-only plus IR-only results reconstruct
the combined poles, UV-only double poles vanish, and full finite terms do
not change with pole selection. All 64 served split checks pass; eight
checks remain unavailable because of the same DP statuses. Both full and
partial variants are checked. `analysis-protocol/` freezes this later parser,
and `analyze-retained.py` repeats it using only retained raw output.

The runner disables and clears numerical caches for these comparisons.
`final-qcd-n1-cache/` separately passes 59 records: 16 uncached references
and 43 cache actions on an infrared rank-two triangle, through the same
generated adapter. All cached, bypassed and reinitialized tensors match
their corresponding uncached references exactly at printed precision.
DP full and partial stores retain eight entries using 1,440 bytes, while
the six QP full/partial and UV/IR mode keys retain six entries using 58,464
bytes. The QP 64-byte and DP zero-byte budgets refuse insertion and retain
zero entries and bytes. Independent checks, topology exclusions and disabled
caching preserve retained entries; clearing and finalization/reinitialization
also pass. Reversed-order checks use separate uncached reversed-order
references. The cache runner's seven focused tests pass.

Public cache statistics do not expose finite-evaluation counters;
these adapter checks establish NaN masking and pole agreement, not finite
work omission by themselves. The latter is established by the native
constructor, arithmetic and generated scalar-kernel tests in
`../../qualification/poles-only/`.

The initial `final-qcd-n1-poles/` and `final-qcd-n4-poles/` directories retain
failed compile attempts: the diagnostic driver's local `NF` collided with
the real model's named constant. Renaming it to `NDIAG` fixes the driver;
the compiled unit fixture now includes the model constant. No production
adapter or numerical policy changed. `pole-runner-tests.log` records seven
passing focused tests. Fixed-order per-call times are retained only as
diagnostics and are excluded from the campaign performance gate.
