The exact denominator-order reversal completed successfully as a diagnostic.
Historical TIRust still refused threshold point 2/group 5. All 35 NinjaRS unit
numerators returned native status zero in both orders, providing 105 complete
forward/reverse coefficient comparisons. Failed TIRust outputs are excluded;
there is no TIRust tensor comparison for this group.

The original order gave TIRust Gram conditioning `5.224078113663414e-30`; the
reversed order gave `1.261590123326023e-22`. Both returned status `1` with an
explicit ill-conditioned-Gram error. The denominator permutation changes the
reported condition measure without changing the exact denominator product or
polynomial numerator. This observation does not establish a correct tensor,
remove the failure, or justify disabling the guard.

The NinjaRS comparisons use the complex norm
`abs(a-b)/max(abs(a),abs(b))`, with zero–zero set to zero. They are before
amplitude normalization and contraction.

| Laurent slot | Successful unit pairs | Maximum absolute difference | Maximum complex-relative difference |
|---|---:|---:|---:|
| Finite | 35 | `1.136580820453906e-6` | `8.187343950458839e-9` |
| Single pole | 35 | `0` | `0` |
| Double pole | 35 | `0` | `0` |

The largest absolute finite difference occurs for degree-three unit 32, whose
forward value is approximately `-17.2514330378 + 198.128250909 i`; the change
is approximately `1.13658e-6` in its real part. The largest relative difference
occurs for degree-three unit 24, whose forward value is approximately
`-4.37197315601 + 50.2110395526 i`, with absolute difference `4.12650e-7`.
The complex relative metric includes both real and imaginary parts; it is not
a bound on their separate relative errors or on the eventual real amplitude.
The scalar unit zero is identical at recorded precision in all three slots.
The zero pole differences in the table mean equality of the recorded
coefficients, not an independently established error bound.

The source/input/output validator passed. It checked the exact decimal echoes
of the permuted shifts, masses and both invariant-matrix axes; all 35 degrees
and tensor mappings; output coverage; cache-disabled counters; actual private
historical DSO mapping; unchanged objects, libraries and protected files; and
exclusion of failed native outputs. The executable and object were reused
byte-for-byte, without compilation. Both bridges use their historical 256-bit
working precision and share oneloopmaster ancestry, so the comparison is not
an independent scalar-integral accuracy certificate. No amplitude numerator
coefficients are reconstructed or contracted here. The original forward
evidence, inputs, implementation and failed amplitude fixtures remain unchanged.

`queue.py` was added separately from the frozen preparation to wait for the
normal numerical lock. Its one-second idle observation and non-atomic handoff
are recorded in `queue-execution.json`; the unchanged `run.py` acquired the
actual exclusive lock. The queue wait was `909.360` seconds. The native run
exited zero after `3.2365` seconds on CPU 9, within its 180-second limit.
These times describe admission and execution only and are not performance
measurements. The queue helper and frozen runner hashes were unchanged.

Validation can be repeated without native evaluation from the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/analyze.py
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/reversal/compare.py
```

The completed evidence identities are:

| File | SHA-256 |
|---|---|
| `summary.json` | `1e70d5ae70f58e73439357e1b6b3c12b67bbbdfee632b874980bb9fe4908c2bc` |
| `comparison.json` | `46a5de9e332787af5a7034e333925acbfe5374dfaec29dae9a1a36879ade2030` |
| `execution.json` | `bd8af718eb23f0369c0188cc31499de1eb128796f6de9a768f4600b3ae319935` |
| `queue-execution.json` | `b8fb0fddcfed2f21fe4f4b530a2d11671c244502069034985704f5d906f73fdc` |
| `output.log` | `fba510d224b1f9f9fc0bc19b8c64d24db1844d2d3de43529ecaa889f0f696e49` |
