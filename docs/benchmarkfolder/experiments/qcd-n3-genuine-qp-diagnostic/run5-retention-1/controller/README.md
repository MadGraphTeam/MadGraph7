# Closed QCD n3 genuine-QP diagnostic — source proposal 5

This source-only amendment accepts a production formatter's exact zero spelling
in successful ASCII output, and proposes continuing the six workers that run4
never launched. It changes no numerical code, existing executable, runtime
provider, input, card policy or observer. No compiler, loader, native amplitude
or real run5 runtime directory is created during preparation.

Run4 completed ten native workers; nine passed the frozen source4 parser. Its
tenth worker, NinjaRS/mode4/observed, returned successful native output containing
literal `0` followed by 63 spaces. The parser incorrectly required an exponent for
that valid zero. All 630 rejected records are this spelling, all 33,099 input
records retain scientific notation, all 348 native output statuses are 0 and all
buffers have the correct eight 64-byte records. No ABI, pointer, count or stride
mismatch was found. The entire failed attempt is immutable and retained at
failed-run4-retention-1, seal fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361.

The pinned static transport proof traces both production wrappers through their
Float LowerExp implementations to locked Rug 1.30.0, which emits `0` or `-0` before
exponent formatting. NinjaRS and TIRust both use that formatter and Fortran
list-directed output reads. Actual run4 evidence is NinjaRS only; TIRust's
contract is source-proven, not a claim that its unexecuted native arms passed.
Rug's actual nonzero output has 39 significant digits; the older header’s 40-digit
wording is not used as proof. Neither formatter nor library is modified here.

ascii113 keeps the exact scientific-notation input/nonzero grammar. Only the
successful backend 8/9 output path permits canonical `0` and `-0`. Bare `+0`, `0.0`, `1`, NaN,
Inf, malformed fields and truncated records still reject. Failed native buffers
are never decoded. Decimal-to-Fraction reconstruction remains exact before
nearest-even 113-bit rounding. The parser's nonzero qualification remains the
normal binary128 exponent range only: subnormal/underflow requests reject
explicitly, and a rounded carry beyond maximal range now rejects explicitly.
The observed nonzero output exponents are -41 through -1; this fixture does not
qualify extreme exponent behavior or general binary128 subnormal transport.

Fractions do not retain a zero sign. New lexical metadata records positive and
negative zero indices separately for ASCII input/output fields, while complete
raw records stay in the journal. Two negative-zero input records were observed.
Fraction equality is not claimed to establish signed-zero bit parity. Existing
Born/original/DP/QP/MP transition bytes, full common-input digest, status/group/
nonfinite/IEEE/W03 checks and coefficient-adjudication logic remain unchanged.

closed_run4_replay.py verifies every retained prefix byte and fully replays all
ten arms with the proposed parser. closed-run4-preflight.json retains their full
parsed records, exact fixture/parser/raw-trace identities, original failure and
scope. All ten parse through every remaining gate; five plain/observed public
byte pairs and 18 applicable shared-input comparisons match. That is an offline
reinterpretation of a failed prefix, not an original-run4 success, a full
sixteen-arm adjudication, a numerical tolerance result or a performance result.
All returned diagnostic status 120/accuracy -1 values remain unchecked. Earlier
source4 offline-replay files are explicitly named prior-source4-* provenance.

The continuation reader verifies the sealed prefix, exact started-worker set,
all fifteen original runtime providers/maps/alias guards, fixture/hash/protocol
identities and the pinned full-prefix preflight. It reuses those ten parsed
results without launching them again. run5 can append only the six never-started
IDs, in their original order:

- backend 8, mode 5, plain then observed;
- backend 9, mode 4, plain then observed;
- backend 9, mode 5, plain then observed.

No duplicate worker or retry is permitted. If all six new arms qualify, unchanged
adjudicate receives exactly sixteen unique IDs: ten inherited plus six new.
Counts and provenance stay separate, and the original failed run4 is untouched.
The new alias directory has the same fifteen original providers; the build2
executable remains ed6a51fff0312b06a45cd97a409aafa7d2447eef5ae7800579b44c0c904b8474,
with its original glibc2.40 interpreter. There is no relink or new runtime byte.

The original 3000 s stage spent 152.56272240309045 s in run4. The new inclusive phase
therefore has exactly 2847.43727759690955 s, with no budget expansion. Prefix
validation, integrity checks, aliases, traces, workers and adjudication count
against it. CPU 12 remains nonexclusive, one thread/4 GiB, each trace at most 30 s,
each new worker at most 180 s. The common nonblocking lock precedes fresh run5
creation; busy 75 is a scheduling hold. Execution stops on the first failure.
Actual /proc maps remain sampled observations, not a pre-entry barrier or
continuous race-proof monitor; separate strict traces precede new workers.

All 99 pure formatter, precision, failure and continuation tests pass.
Independent decimal and continuation reviews pass before the source seal. The source does not authorize a
native continuation; root reviews the sealed proposal and retained evidence
before separately admitting it.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal5/run.py verify
# Offline ten-arm full prefix replay:
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal5/closed_run4_replay.py
# Only after root admission, launch exactly the six never-started suffix workers:
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal5/run.py run
```

prepare.py is retained provenance and must not be rerun over this proposal.
The controller depends on named closed workspace donors; a portable combined
scientific result can only be retained after actual continuation termination.
