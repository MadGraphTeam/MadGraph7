# Default stability policy: retained correlated DP error

This package only reparses existing evidence and compares source bytes. It
adds no compilation, native evaluation, test execution, timing, or production
change. The historical default-policy failure remains visible. The narrow
strict-policy rescue is established; general default-policy robustness and
full physical-amplitude accuracy are not established by this experiment.

The fixture is `li-rows.json LI-32 N4-massive R4 row 57 R=5 lane 44`:
four propagators, tensor rank five, masses squared `29929`, and scale squared
`15625`. Its supplied truth is documented as NinjaRS generic at 40 digits on
inputs consistent with the binary64 shifts. That reference is not a
measurement of the current MadLoop NinjaRS DP/default-host check.

## What was executed historically

The release experiment used bridge SHA256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
It ran the extracted production rescue control block, actual Fortran tensor
adapter, accuracy calculation, and return-code finalization. The injected
real or imaginary tensor contraction replaces the physics callback; mature
filter/helicity state is forced. Rotations select retained input arrays.
The complete physical helicity, color, Born, UV/R2, initialization and filter
pipeline is outside this experiment.

For the adapter input, the nearly zero final shift is placed first and removed
by an exact polynomial translation before binary64 rounding. QP promotes the
same binary64 inputs and reconstructs its invariants in QP. `inputs.json`
retains both the original fixture lane and the transformed adapter input.
`observations.json` retains all eight original host rows without changing
their values, alongside independently recomputed metrics.

The original-origin raw C-ABI call and the transformed adapter contraction are
different evaluations: the former has finite symmetric error `3.479475e-6`;
the latter's primary DP/QP finite symmetric error is `2.665155e-4`.
Neither number is the accepted midpoint error below.

## Error metrics and policy

Zero-rotation results against the separately evaluated same-input QP value:

| Projection | Reported host accuracy | Accepted finite midpoint | Symmetric true error | Ordinary relative true error |
| --- | ---: | ---: | ---: | ---: |
| Real | `3.798332e-5` | `524.1950567875288` | `1.525129e-4` | `3.050724e-4` |
| Imaginary | `2.170827e-4` | `166.31134525596826` | `7.028800e-4` | `1.404773e-3` |

Symmetric error is `abs(value-truth)/(abs(value)+abs(truth))`; ordinary
relative error is `abs(value-truth)/abs(truth)`. Both default-accepted
projections fail the independent `3e-6` symmetric gate. Both symmetric errors
are below `1e-3`, but the imaginary ordinary relative error **exceeds** that
default target. It would be incorrect to say all actual errors are simply
within the default tolerance.

The production host accuracy measures consistency across numerical
evaluations. It takes each Laurent coefficient's range and midpoint, then
weights the relative ranges by the absolute midpoints and divides by one
third of the absolute sum of the three midpoints. The exact routine is
retained in `current-compute-accuracy.inc`; the API explicitly describes
`PREC_FOUND` as an estimated relative accuracy. This is not a certified error
bound against truth. Correlated numerical errors can give a small spread.

- At default `MLStabThres=1e-3`, all four projection/rotation cases return
  `219`, retaining inaccurate DP midpoints. Adding two rotations changes
  the estimates only at rounding level.
- At shared strict policy `1e-6`, all four cases execute modes `[1,2,4,5]`,
  or `[1,2,1,1,4,5,4,4]` with two rotations, and return `323`.
  Every callback succeeds. The final values equal the separately recorded
  same-input QP results. Relative errors against the supplied fixture truth
  are at most `1.194582e-15`.
- QP host accuracy zero means the reported QP evaluations round to identical
  values; it does not mean the mathematical error is zero.
- Raw reversed check-pass, cache-disabled, and ordinary-cache calls agree;
  bypass calls leave cache counters unchanged. Cache correlation does not
  explain this retained discrepancy.

## Scope and present source identity

The frozen campaign `split-dp-terminal-1` records current bridge SHA256
`eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`.
**This exact fixture was not replayed against that bridge by this audit.**
The full host template, host-policy excerpt, and return-code excerpt match
their historical execution bytes at capture. Some bridge/interface files
have changed. `source-bindings.json` records each historical/current hash and
the precise equality result; `current-source-excerpts.json` preserves the
audited source excerpts even if the live checkout later changes.

There are no same-fixture default-host measurements for CutTools, Ninja,
COLLIER, or current MadLoop NinjaRS in this package. The general host
consistency mechanism is shared with other reducers; COLLIER has an internal
stability special case. Shared code is not empirical evidence that those
reducers exhibit the same error. Other physical stress failures are separate
observations and must not be used to fill this comparison gap.

The evidence demonstrates a correlated-error blind spot and sensitivity to
the tolerance policy, not a dispatcher or cache defect. The narrow historical
strict-rescue check is closed. Default robustness, the current-bridge replay,
and comparative default-policy behavior remain limitations. This one fixture
does not justify an arbitrary backend-9 threshold cap or native condition
cutoff. No production fix is established as necessary merely to use the
separately registered shared `1e-6` campaign policy; that campaign must still
pass its independent numerical gates. This audit establishes no guarantee
that `1e-6` catches every unstable physical point.

## Read-only reproduction

From the MadGraph repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/default-stability-policy-audit/audit.py
```

This checks the local inventory, historical source/data hashes, all eight raw
rows, host formula, statuses, mode sequences, metrics, and recorded cache
bypass equalities. It does not import or execute the historical native runners.
The external read-only fixture is checked when present; its selected lane and
source metadata are retained locally. Add `--check-current` to additionally
require the live source hashes to equal those captured in this audit.

`--capture` was used once to create this package and refuses to overwrite its
generated artifacts. Existing historical records under
`qualification/tirust-bridge/stability-audit` remain unchanged. The index is
local to this package; referenced historical evidence stays in its original
repository paths. Numerical or performance acceptance must not be inferred
from this read-only verification.
