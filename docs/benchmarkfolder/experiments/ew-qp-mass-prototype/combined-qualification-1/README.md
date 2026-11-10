# Combined mass qualification: continuation and incomplete-QP evidence

This is a portable record of the completed mass-transport continuation and the
additive original-QP controls. The retained backend9 fixed-QP sums are
**incomplete and unqualified**. Passing input comparisons does not qualify their
coefficients, physical accuracy, host stability, Born/graph changes or runtime.
No native work was performed while packaging this evidence.

| Retained stage | What actually happened |
|---|---|
| Continuation | Closed report with 234 events: 218 previously completed caller-fixture events reused, seven new loader checks, six new amplitude workers and three validators |
| Additive original-QP | Two loader checks and two native workers exited zero; the offline validator failed on a stale parent-report path |
| Offline repair | A single analyzer path override selects the completed continuation; unchanged input-comparison criteria pass |

The 218 fixture events include the intentional trapping control. They were not
rerun by the continuation. The original loader/validator failures remain in the
captured history. The initial preflight package
[`qp-mass-general-preflight-private`](../../qp-mass-general-preflight-private/README.md),
index `a079a55f…`, remains unchanged. The offline correction is the exact frozen
`4b2e0cd4…` record; removing its one added path-override line reproduces the
original analyzer byte-for-byte. The old additive run remains failed, with its
two successful native events intact.

The raw `stdout.log` warning counts are:

| Current arm | NinjaRS backend8 | TIRust backend9 |
|---|---:|---:|
| Original QP | 0 | 74 |
| Guarded QP | 0 | 74 |

These counts cover each complete run, including warmup. Independently, the
saved native journals record **37 refused groups at observed point2 in each
backend9 arm**, and zero refusals at point1. The raw warning text, exact input
words, failure statuses/messages, available tensor outputs, contracted results
and host statuses are retained. Agreement between omitted sums cannot establish
accuracy. Backend8 also remains an unchecked fixed-CT4 diagnostic despite its
absence of W03 warnings.

The successful offline result establishes the originally specified nonmass
input invariance across the mass change and the separately labeled historical
runtime controls. It permits mass-dependent kinematic changes and records output
changes; it does not require or assert coefficient equality. The historical
`49af12f0…` library is not a same-runtime control for the current `eb3d67fd…`
library. These distinctions remain in the complete analyzer result.

## Verification and retained bytes

Run this from any directory with Python3; no ignored build directory, compiler,
reducer, network connection or native binary is needed:

```sh
python3 -B /path/to/combined-qualification-1/verify.py
```

The verifier checks the complete outer index and exact UTF-8 payload hashes,
the 234-event continuation, the original failed additive envelope, the
single-line offline correction, and the raw W03 counts. It then loads only the
retained pure-Python parser/comparator definitions and replays all six journal
comparisons, including both native-refusal proofs. It never invokes those
modules' main functions or any native library/executable. A relocated copy has
the same verification requirements.

`payload.json.gz` contains 905 original text paths with 532 unique contents:
controllers, source/manifests, exact parameter/point inputs, complete execution
reports/logs, caller-fixture output, native observer journals, historical
comparison journals, frozen Python analysis and all offline-repair history.
`payload-inventory.json` identifies every path. The 119 native dependencies and
artifacts are recorded by their original pinned hashes in
`native-identities.json`; their bytes and original symlink objects are not
distributed. These are recorded execution/preparation identities, not a new
native evaluation performed by the portable verifier.

The small top-level JSON copies expose the completed continuation, failed
additive run, warning audit, offline execution and source lineage directly.
Their exact equality to the original payload paths is verified. The full
9.5MB analyzer JSON is preserved losslessly inside the compressed payload.
`evidence-index.json` binds all physical package files. No prior package,
scientific input, generated source or run record was rewritten.
