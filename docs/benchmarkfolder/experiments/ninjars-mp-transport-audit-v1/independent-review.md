# Independent source review

Reviewed `production-ready.patch` SHA256
`3863c862562f37d352c6c9463be4d1e77f24193653377ff7cbb765871d265b97`
against the production parser and pinned Symbolica/numerica revision
`e679e8c129ce15ec50c46ffb4ff4b59aeb7813b8`. No build or engine call was
performed for this review.

The runtime change is narrowly correct for the actual working precisions, 256
and 384 bits. Parsing still first rounds the decimal record to 113 significant
binary bits and rejects nonfinite inputs. Increasing that Float's precision
preserves every represented finite dyadic exactly, while retaining its zero
sign. At this dependency configuration `Float::set_prec` delegates to Rug/MPFR
precision rounding; the previous exact-rational conversion necessarily erased
negative zero. The input width, ASCII parsing/error paths, statuses, ABI,
provider thresholds, reducer construction and output formatting are unchanged.
The statement concerns widening from 113 bits, not arbitrary calls requesting
less than 113 bits or a different numerical backend.

The two contract tests use independent integer-and-power-of-two expected
values rather than deriving expectations from the parser. Their extrema,
subnormal/normal transition, last significand bit, beyond-binary64 value and
both signs are appropriate. Rational equality is supplemented by explicit
zero-sign, tracked precision and formatted-output reparse checks. The complex
log/sqrt assertions exercise an actual observable branch distinction.

The ignored replay keeps immutable expected strings, requires all 1,392
requests, checks successful statuses and all eight fields, and records failures
before asserting. It calls the real `MpState::evaluate` rather than a mock.
It is a state/evaluation transport regression, not a new public-ABI lifecycle
test or a universal physical branch proof. The retained completed result
reports zero changed fields among 11,136 fields; this supports sampled
compatibility without relabeling the corrected runtime as the old artifact.

No blocking source issue found. The reported scalar/REAL16 red-green checks
and complete bridge/replay results are consistent with this source change;
they were inspected, not rerun by this independent reviewer.
