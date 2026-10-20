# Experimental binary64 derivative-row factorization

This separate experiment adapts the proved FFV derivative-row identities to the
full compact DP polynomial updater. It changes no production DP code and makes
no process-performance claim. Initial scope is ranks 2–4, both `FFV1L1_2` and
`FFV1L2_1` orientations, GNU binary64 arithmetic, and the actual standalone/NLO
compiler flags. Rank 1 and DP terminal process support are excluded.

`factor.py` keeps the original B0 contraction and its K order, computes the
shared U contraction with the original B-zero predicate, then applies the
proved derivative-row signed permutations. It deliberately adds no A-zero
pruning, so that separate optimization is not included in this experiment.
Accepted outputs may change rounding and zero signs. Unsupported inputs use
the unchanged public DP updater.

The actual generated DP vertex sources are retained under `oracle/proof-inputs`.
The source binding proves 160 coefficient-expression identities and 128
derivative-row identities in total. The mathematical oracle decodes stored
binary64 components to exact dyadic rationals and contracts the original A/B
monomials directly. It never uses the prototype's U values as its reference.
`oracle/safe-bound.md` derives the admitted component exponent interval
[-200,200], its 2^-504 product lattice, and the conservative intermediate bound.

The completed compiled campaigns are:

- `oracle-nlo-rank2-v1`, `oracle-nlo-rank34-v1`, and
  `oracle-sa-rank234-v1`: all twelve rank/orientation/compiler configurations,
  ten fixtures each. These cover actual rounded DP ALOHA rows, surviving
  binary64 perturbations, cancellation, scales, sparse inputs, signed-zero
  components, input immutability, flags, and false-entry fallback.
- `oracle-endpoints-all-v1`: twelve further streams. Cases 4 and 5 use exact
  upper/lower admitted exponents with maximum mantissas in A and B; case 6
  mixes both endpoints with zero mantissas in A. All raw bits and exact
  contraction results are retained.
- `safety-all-ranks-flags-v1`: 240 true-entry guard violations, 384 initial
  sticky-flag combinations, and 120 original/candidate trap pairs. Every
  nonnearest, out-of-range, subnormal, nonfinite, shape, and row-mismatch case
  reaches the original fallback. Separate processes compare actual invalid
  exceptions from signaling-NaN A/B inputs, overflow, underflow, and inexact.
  Division-by-zero is an enabled-mask preservation control because this updater
  does not divide.
- `safety-combined-sticky-traps-v2`: 960 original/candidate pairs combining an
  enabled trap with every compatible sticky-flag subset. The actual masks and
  flags immediately before each call are recorded and matched. Both the
  original updater and the guarded helper preserve that measured state on the
  exact arithmetic fixture.
- `safety-zero-work-snan-v1`: 24 true-entry cases with signaling NaNs in A and
  all B coefficients zero. The public routine skips every product, so this
  specifically checks that the new bounded-input scanner adds no invalid flag
  before falling back. Real and imaginary sNaN components are covered, with
  input bits, complete output bits, and flags preserved.

The 24 oracle streams contain 240 fixtures and 295,680 complex output
coefficients: 216 accepted paths and 24 false-entry fallbacks. The exact oracle
reports 29,300 strongly cancelled coefficients. The largest candidate forward
error is 0.719030 times epsilon times the absolute product scale; the largest
fraction of the fixed forward bound is 0.048841. These are observed corpus
bounds, not universal correctly-rounded guarantees. `oracle/verification-v2.json`
records a complete independent exact replay; `verification-v3.json` additionally
binds the zero-work sNaN controls and the earlier exact replay.

The DP-specific IEEE procedure boundary is tested directly: the original
compact DP routine has no IEEE imports, while the experimental helper does.
Both are called from the driver without placing an extra IEEE wrapper around
the original routine. The trap tests pass explicit true entry permission to
reach the halting guard; the separate 32-mask campaign derives entry permission
from the actual inexact flag. The latter therefore covers both accepted and
false-entry paths with consistent captured state.

Three harness issues are retained, without changing arithmetic or discarding
their outputs. `safety-initial-v1` contains a compiler rejection of constructing
a constant signaling NaN through `CMPLX`; later drivers install those raw bits
with `TRANSFER`. `safety-combined-sticky-traps-v1` assumed flags survived GNU's
halting-mode setter. `combined-sticky-diagnosis-v1` demonstrates that the flags
were cleared during setup before either updater call; v2 installs flags after
enabling traps and measures the actual pre-call state. Finally,
`oracle/verification-v1.stderr.log` records a verifier comparison of integer
dictionary keys with their JSON string representation; the verifier now
normalizes that representation before comparing the unchanged exact proof.

Run the read-only aggregate verifier with:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/oracle/verify.py --recompute
```

It checks source/driver/proof bindings, generated Fortran and raw dump hashes,
complete case counts, exact endpoint bits, all stored-input contractions,
guard/flag/trap outcomes, and retained failures. `--recompute` reruns exact
rational algebra only; it does not build or run a floating-point helper.
All builds and temporary artifacts remain inside the repository. No Symbolica
process is used. Physical amplitude accuracy, stability/rescue behavior, and
quiet performance comparisons remain separate pending gates.
