# Static localization of the new trapped failures

Both closed trapped workers terminated with SIGFPE during warmup at saved
point 2. The first backtrace frame after the signal handler maps into the
new guarded library, SHA256
`69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b`.
Neither instruction belongs to the guarded `cayley_cancellation::<f64>`
function. These are separate division sites.

| Worker | Reported frame 3 PC | ELF address | Symbol and offset | Instruction |
|---|---|---|---|---|
| checked-trapped | `0x703f7a10aecb` | `0x11fcecb` | `gram_indicator::<f64> + 0x99b` | `divsd %xmm1,%xmm0` |
| forced-qp-trapped | `0x753298f8ccc3` | `0x2523cc3` | `oneloop::native::dilog::series::<MPFR Float> + 0x603` | `divsd %xmm0,%xmm1` |

The checked-worker source computes `largest / sqrt(cnorm2(delta).to_f64())`
at line 724 of the retained `gram.rs`. This normalization remains distinct
from the guarded modified-Cayley ratio below it.

The forced-QP worker's symbol is a multiprecision series, but the failing
instruction operates on binary64 values. The source's inlined `series_terms`
helper computes `4*pi*pi/modulus` at line 74 of `oneloop-dilog.rs`. The
literal constant loaded immediately before the instruction is
39.47841760435743. The source guards an exactly zero modulus; this evidence
does not reveal the actual divisor. A tiny positive divisor causing overflow
is a possible explanation, not an established exception classification.
Read-only Git object comparison confirms this source equals
`src/native/dilog.rs` at the pinned oneloopmaster revision
`bd703d09a78f7849a6b86a07096e6f222fc01193`.

The retained maps and ELF PT_LOAD headers determine each load bias. Static
`nm`, `addr2line` and `objdump` output identify function bounds and the exact
DIVSD bytes. The guarded Cayley function occupies
`[0x12031f0, 0x1203d07)`, outside both reported instruction addresses.
The literal ELF hash was checked before and after static tool use. No DSO,
MadLoop executable or numerical worker was loaded or executed for this
localization; no debugger, new compilation or amplitude rerun was used.
The two original failed runs, controls and stack traces remain unchanged.

No registers, MXCSR, signal subtype or core dump were captured. Thus the
specific exception type and numerical operands remain unknown. The optimized
ELF lacks a source line table; source-expression association is an inference
from symbol bounds, instruction context, literal constants and the retained
source. This evidence does not establish accuracy, precision usage, stability,
trap completion or performance qualification.

The local verifier reads only retained UTF-8 text and gzip files, checks both
address calculations and source associations, and needs no historical native
paths:

```sh
python3 -I -B verify.py
```

`localize.py` records the initial static extraction procedure and requires the
historical ELF and toolchain. The additional read-only Git/source comparison
and post-tool ELF hash are recorded in `analysis.json`. They are not executed
by the portable verifier.
