# Source-only bulk QP output-decoding proposal

This private proposal changes only the Fortran output decoder in MTIRUST. It is
uncompiled and unexecuted. Production, all closed profile capsules and installed
libraries remain unchanged. No speedup or numerical equivalence is claimed.

The closed ordinary QCD n3 point16 profile measured 0.074575912 s in output
decoding **plus subsequent mapping**, or 5.317% of its 1.402470612 s sequential
served-QP interval sum. Native FFI occupied 92.331%; input allocation/encoding
1.640%. Those are single instrumented occurrences, not full-amplitude timing or
a repeatable saving estimate. Mapping is outside this proposal, so even the
entire measured 5.317% cannot be attributed to work this patch could remove.
All seven exact within-event repeat keys already hit the native cache; no new
tensor cache or cache-budget change is proposed.

The candidate copies output records into a CHARACTER(64) array, then issues one
list-directed internal-file READ for the whole real array only when every record
is an eligible singleton numeric token. The byte values and record order are
unchanged. It keeps the existing scalar unpack subroutine byte-for-byte and uses
it in the original order for every ineligible batch. No explicit ES/G format,
ROUND= mode, arithmetic conversion, C ABI, Rust, tensor mapping, plan, cache,
numerator or reduction policy changes are made.

The narrow eligibility gate is:

- Full evaluation with both pole selections; every partial/pole selection stays
  scalar, even though this bridge currently fills unevaluated native slots with
  zeros. Partial mode is never inferred from the output spelling.
- Nearest rounding and no enabled supported IEEE halting flag. The helper only
  queries these settings. Other settings retain scalar conversion.
- Exactly `0` or `-0`, or an optional minus, one nonzero decimal digit, a point,
  exactly 38 decimal digits, lower-case `e`, and an unpadded signed-negative or
  unsigned decimal exponent with magnitude at most 4900. Only trailing ASCII
  spaces are permitted. The current pinned float-mpfr/Rug1.30 formatter's .39e
  precision means 39 significant digits, not 39 digits after the decimal point.
- The exponent margin deliberately excludes binary128 overflow, underflow and
  normal/subnormal boundaries. Those records retain scalar behavior.

The integer-only grammar rejects null fields, repetition syntax, slash,
commas, leading or embedded whitespace, NUL, line breaks, NaN/Inf, extra tokens,
non-ASCII bytes and unsupported formatter spellings before any bulk numeric
READ. Thus the array read cannot legitimately consume a second token from one
record or skip a blank record into another. No delimiter normalization occurs.

The existing public invalid-output error, zero coefficients and native-cache
clear remain in place. If an eligible bulk read unexpectedly fails or produces
nonfinite values, the candidate takes that public error/cache-clear path without
reparsing already converted records. An unexpected difference from the original
parser is a qualification failure, not permission to promote the candidate.
The restricted exponent/trap gate is a reason to test that branch, not a proof
of runtime equivalence. The additional
record copy, eligibility scan, IEEE queries and temporary storage may erase any
I/O setup saving. The automatic CHARACTER array is at most 107,520 bytes for the
currently admitted rank6 (210 coefficients); its allocation/storage cost must be
measured. No benefit is inferred from source shape.

`proposal.diff` is the exact private preview against the retained production
module. `prepare.py` only creates/verifies source files and bindings;
`test_source.py` checks source confinement and a Python model of the lexical
gate. These pure tests do not compile Fortran or establish numeric/IEEE parity.
`EQUIVALENCE.md` defines the required later public-parser, actual-physical and
descriptive paired timing gates. Each later build/run needs a separate reviewed
bounded controller and admission; none exists or has been invoked here.

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/fortran-qp-output-bulk-proposal-1/prepare.py
python3 -B -m pytest -q -p no:cacheprovider \
  .tirust-build/fortran-qp-output-bulk-proposal-1/test_source.py
```

The profile provenance is pinned in `bindings.json`; `formatter-source.json`
retains exact source excerpts and their whole-file hashes from the pinned Rust
formatter chain, together with the installed dependency receipt. Those source
facts do not substitute for an eventual actual wire-corpus eligibility count.
Precise <=1.05 timing acceptance remains Hua-Sheng's local TODO.
