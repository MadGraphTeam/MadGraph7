# Native pole-only qualification

The dedicated pole path passed nine tensor integration tests, seven filtered
core unit tests, three standalone numeric-kernel tests, and four scalar-kernel
regressions (288 point/precision comparisons). Counts overlap: the pole arithmetic
unit test also appears in the standalone numeric-kernel run. These checks establish
correctness and finite-work omission; they are not full-amplitude timing gates.

Tensor coverage includes N=1 through 6 at rank 6 in binary64, double-double and
113-bit MPFR; nonzero massless IR poles for boxes, pentagons and hexagons;
IR-double-to-IR-single normalization mixing in binary64 and 113-bit MPFR;
scaleless UV/IR cancellation; cache isolation, changed inputs/scales and rotations;
successful double-double partial rescue with actual master-cache counters;
and a small-Gram expansion compared with 192-bit full evaluation. Unsupported
requests remain explicit errors. Internal tests verify that the pole constructor
creates neither native full-master nor symbolic full-master evaluators, and that
finite scratch containing NaNs does not enter pole arithmetic.

The initial integration run passed seven tests and explicitly refused two lanes
at auxiliary Gram conditioning gates. Those two tests had simultaneously enabled
Gram expansion and double-double partial rescue; upstream dispatch prioritizes
the Gram route. Removing the competing Gram option made the intended partial
rescue tests exercise successful double-double evaluations. The separate
small-Gram test passed in both runs. The initial log is retained. No production
arithmetic or gate was weakened in response.

`test-only-amendment.json` and `.diff` identify the exact difference between the
installer's compiled candidate patch 53b559ade233201659a1f9678188b4c761b9cc587267067bd354e7769ac62d9a
and the final tested patch 80ef17b367511419e734f760edca4593f15a324996c4830b815c03cf884cbab5.
Only three test-configuration lines changed; all eight production files match
byte-for-byte. This evidence does not claim that an older compiled artifact was
built from the later test patch. `patch-validation.json` separately verifies the
final patch, original-source hashes and zero-fuzz application against the tested
source tree.

Scalar generation is byte-reproducible and emits only Laurent roots [-1,-2].
A completely collapsed massless scalar box remains an explicit nonfinite error
because its upstream pole expression divides by zero.
No alternate formula or silent full-evaluation fallback is introduced.

The scalar CSV records one illustrative warm optimized-development invocation,
with medians over seven internal batches of 20,000 actual evaluations. It has no
confidence interval and does not establish an end-to-end performance gate.
Simple massless C0 costs 72.901 ns versus 48.563 ns for full evaluation, and massless
D0 costs 128.775 ns versus 54.910 ns. The massive/soft examples are faster with poles
only. Neither result justifies changing the default MadLoop reduction chain.
See `generation-provenance.txt` for exact methodology and limitations.
