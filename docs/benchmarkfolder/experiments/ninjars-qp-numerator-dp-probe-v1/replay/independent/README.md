# Independent stored-input replay check

`check.py` independently parses the replay's tagged raw stream; it does not
import the owner's matrix constructor, parser or contrast analyzer. It reuses
the previously sealed independent capture parser and exact binary64/binary128
decoder/rounder, then verifies all eight actual input constructions including
zero sign, all 1,392 reproduction controls, complete loop/key coverage and all
exact per-record Laurent differences and reference norms in contrasts.json.

The offline check completed successfully, with no engine calls. All 3,712
records are finite with status 0 and all 464 keys have all eight cells. Results and
the raw/checker hashes are retained in result.json; check.stderr is empty.
This independent check does not reconstruct a physical amplitude, repeat a
library evaluation, or claim universal validity of the mixed input bundles.

Source review found no blocking construction or comparison issue. The driver
checks binary layout and nearest rounding, records actual pre-call arguments,
checks all input words after each call, and retains the same context until
finalization. The runner binds the old archive/module/driver/inputs and resolved
libraries before and after execution. Controls 1/2/8 match captured native,
compensated and full-QP records exactly. The off-diagonal cells preserve the
declared geometry and perform exactly the specified coefficient conversion.

The maximum native-versus-generic discrepancy at fixed original DP geometry
is approximately 1.12600265e-9 at point 80/direction 2/loop 91. Substituting rounded
captured QP coefficients leaves approximately 1.12596445e-9 at the same key.
The generic original/rounded-QP coefficient contrast is at most 7.04162e-16.
These independent per-loop results support the reported negative diagnosis
for a rounded-QP-coefficient DP retry on this sample.

They do not isolate arithmetic precision as the cause. Native/generic master
thresholds and primitive algorithms differ; the historical generic transport
canonicalizes negative zero; and captured QP coefficients include the improved
external-point/parameter/Born bundle. Very large relative discrepancies near
zero are not physical accuracy failures. The rational split is deliberately
excluded from cross-domain physical metrics. Formal unweighted loop sums are
not MadLoop amplitude assembly. The corrected transport library must not be
substituted into this historical replay's runtime attribution.

Recheck without compiling or evaluating an engine:

```sh
python -B docs/benchmarkfolder/experiments/ninjars-qp-numerator-dp-probe-v1/replay/independent/check.py
```

This four-file independent scope is frozen for the enclosing replay seal.
