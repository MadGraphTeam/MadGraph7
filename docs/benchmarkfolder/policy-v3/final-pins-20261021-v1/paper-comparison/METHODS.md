# Four-tool timing and stability comparison protocol

This note translates the requested presentation into a reproducible measurement
design. It contains no benchmark results and does not qualify an unexecuted run.
The existing 100-point correctness test and the complete, warmed, full-helicity
runtime gate remain unchanged. CutTools, Ninja, COLLIER, and NinjaRS must all be
visible in the additional comparison, including unavailable or failed results.

## Reference and inspected assets

Hirschi and Peraro, [arXiv:1604.01363](https://arxiv.org/pdf/1604.01363),
Tables 2–3, report generation time, diagram/group counts, maximum numerator rank,
numerator computation, and reduction times. Their runtime boundary is one
helicity with colors summed. Figures 4–7 show cumulative DP stability tails and
ratios to Ninja; loop-induced panels distinguish helicities. Section 7.2 uses
direction reversal and Lorentz rotations. Its point counts are 100,000 for
2→2, 10,000 for 2→3, and 1,000 above that, with stated isolation cuts; Fig. 7
omits the transverse-momentum cut. These describe the reference experiment,
not the inputs or sample size of this campaign.

The user-provided [Drive archive](https://drive.google.com/file/d/1Ep8wSY0OwZp1EtCnDNMHXbeCe7aIvAJ0/view?usp=sharing)
is `madninja.tar.gz`, 414,015,349 bytes, SHA-256
`f75dbf951a5ce41f6c13a2bfb3ed61edbc7284f4b8867faa0d34065e2e66a2e5`.
Its 1,867 members were inventoried without executing code or restoring links.
Selected text members were read privately. No license or README was found.
The Python 2 `StabilityExtractor.py` selects existing log columns; the gnuplot
files arrange those columns into a main cumulative panel and a Ninja-ratio
inset. They are useful as documentation of presentation, rather than a required
runtime dependency. No external implementation is copied here. The large
archive and downloaded paper stay in the ignored private work directory.

## Three distinct observations

1. **Production runtime and stability:** use the retained benchmark driver,
   complete helicity/color sums, all Laurent poles, the agreed cards and point
   stream, normal MadLoop acceptance and genuine rescue enabled. Record actual
   backend, DP/MP usage, return codes, numerical failures, and elapsed time.
   A slower or failing observation remains in its row. The NinjaRS acceptance
   gate compares against the literal original MadLoop revision with Ninja,
   with the already specified repetitions and confidence interval.
2. **DP self-consistency:** a separate driver and execution authority evaluate
   every point in forced DP with rescue disabled. Keep independent forward,
   reversed-propagator, and spatially rotated evaluations. This stream estimates
   numerical self-consistency before rescue. It cannot be reconstructed from a
   successful rescued value or from the normal controller's last accuracy.
3. **Stage decomposition:** a separate one-helicity experiment measures
   numerator work and total evaluation under a clearly identified boundary.
   It complements the complete runtime gate. It must never be substituted for
   that gate or described as a direct reduction stopwatch if obtained by
   subtracting independent executions.

Use the established fixed parameter cards, zero widths, real masses, seeds,
energies, core points, loop-induced points, and separately specified stress
points. Do not silently replace them with the paper's physical inputs or cuts.
One-particle final states use the existing on-shell boosted configurations.
Near-singular and other targeted samples are separate distributions with their
own denominators, never pooled into a claim about random physical sampling.

## DP estimator and edge cases

The current repository's `check_stability` implementation
(`madgraph/various/process_checks.py`, approximately lines 1487–1544 and
1790–1835 at source revision `856e76e669618ec8d0f799b9c8e6f86819dcd986`)
uses mode 1 at the original point, mode 2 at that point, and mode 1 after
`(E, px, py, pz) -> (E, -py, px, pz)`. The latter exact component permutation
avoids adding trigonometric rounding to the transform. For real observable
values `v = (v_forward, v_reverse, v_rotated)`, its spread is

```text
delta = (max(v) - min(v)) / abs(mean(v))
F_tool(Delta) = count(delta > Delta) / N
```

The finite contribution is the closest analogue to the reference plots.
Retain raw unnormalized Born and Laurent coefficients as well, so pole and
split-order diagnostics can be shown independently. A common, unchanged
normalization can cancel in the relative spread, but raw values avoid a zero
Born normalization and make the chosen observable auditable. The current
MG7 source, rather than an unprinted equation in the paper, is the source for
this precise estimator.

The new comparison must define cases that the old helper does not handle:

- Complete, finite, successful nonzero observations use the formula above.
- Three exact zeros have zero measured spread but receive a separate
  `all_zero` classification. Agreement on zero is not evidence of relative
  accuracy or of successful parsing. Keep the point in the denominator.
- A zero mean with at least one nonzero value is `zero_mean_nonzero_spread`:
  its relative spread is unbounded.
- Missing output, process failure, backend rejection, and nonfinite values
  are distinct failure classifications. They count as failures at every
  finite threshold in the primary failure-inclusive curve. They are never
  replaced by zero or omitted from the denominator.
- Serialize unbounded spreads as a classification plus `null`, not a
  nonstandard JSON Infinity/NaN. Preserve the raw text for forensic analysis.
- A conditional curve over valid finite observations can be supplementary,
  provided its denominator and failure count are printed. It cannot replace
  the primary curve.

The primary reported tail is therefore
`(failed_or_unbounded + valid_finite_spreads_exceeding_Delta) / N`.
This explicit failure policy extends the legacy helper and should be stated in
the caption. Show zero counts and failure counts in the companion table.

Self-consistency can miss a common error. Keep the independent four-tool
Born/Laurent/split-order comparison, analytic pole controls, and high-precision
convergence tests as separate numerical evidence. Their accepted tolerances
must not be inferred from, or relaxed to match, these DP curves.

## Counts, uncertainty, and ratio panels

First produce the separate DP observations for the agreed 100 core points.
Choose any larger independent random sample only after measuring cost and
freezing its seed, point count, cuts, and budget. Use the same point identities
for every tool. The paper's large samples are context, not a promise that this
campaign has already evaluated them. Extending a sample changes its manifest;
retain both the original and extended authorities.

Show the exact `k/N` at selected thresholds and binomial intervals. The smallest
nonzero empirical fraction is `1/N`: it is 1% for 100 points. With zero observed
failures, the exact one-sided 95% binomial upper bound is
`1 - 0.05**(1/N)`, approximately 2.95% for 100 points, 0.299% for 1,000,
and 0.0300% for 10,000. Zero events do not establish a zero population rate.
These uncertainty statements apply to an independent random sample; targeted
or correlated points need explicit descriptive labeling.

Use step curves on logarithmic threshold axes. Preserve the discrete counts;
do not smooth sparse tails into unsupported precision. For a zero empirical
tail, show an upper-limit marker or leave the logarithmic curve empty with
the numerical bound in the table. Do not plot an invented epsilon.

For matched samples, the inset ratio is `F_tool(Delta)/F_Ninja(Delta)` only
where the Ninja denominator is nonzero. Otherwise show a gap labeled
`Ninja: 0/N`; `0/0` is undefined and `positive/0` is unbounded. A supplementary
paired difference of failure fractions remains defined and can make sparse
regions easier to read. Do not calculate uncertainty by treating two tools
evaluated at the same points as independent samples; use matched-point
resampling if an interval on the ratio or difference is needed.

## Tables and figures to place in benchmark.pdf

- A process-complexity table: process/model/order, multiplicity, Born or
  loop-induced, actual loop diagram and group counts, rank, helicity selection,
  generation, compilation, and initialization times. Print the respective
  timing boundaries and machine/toolchain in the caption.
- A four-tool timing table: complete warmed evaluation median, dispersion,
  confidence interval, point count, repetition count, actual backend/rescue
  counts, and failures. Add a distinct one-helicity numerator/total/inferred
  reduction table if that stage experiment has actually been measured.
- QCD core and EW core multipanel stability figures, using consistent colors
  for the four tools, a cumulative tail above, and a Ninja-ratio inset below.
- A dedicated loop-induced figure, and a dedicated difficult 2→4 figure.
  Print the actual helicity scope and cuts. A longitudinal-vector comparison
  is meaningful only for a process containing such external vectors; do not
  invent a longitudinal analogue for scalar `gg -> H/HH`.
- Companion stability tables at predeclared thresholds, including the actual
  acceptance threshold, with DP self-consistency counts, normal-controller
  rescue counts, unresolved/rejected outputs, and independent disagreement
  counts. Show a historical `1e-3` reference line only when clearly labeled
  as a reference; it must not imply that this campaign changed its threshold.

The requested Typst styling remains authoritative. The paper supplies the
comparison layout and definitions, not a replacement document style.

## Minimum retained row schema

The exact serialization is an implementation choice, but the following facts
must be recoverable without running a reduction engine:

```text
manifest:
  process/model/order; source/GEN/full-runtime/archive identities;
  driver/executable/library/card hashes; compiler/flags/CPU/thread settings;
  sample kind, seed, cuts, points hash and N; split-order/helicity selections;
  timing boundary; policy version; thresholds; raw artifact checksums
row:
  process_id; point_id; input_point_hash; requested_tool_id;
  stream = normal_controller | forced_dp_self_consistency | stage_timing;
  transformation = original_forward | original_reverse | z90_forward;
  requested_precision; CTModeRun/Init; rescue_enabled; actual_tool_and_precision;
  result_complete; process_exit; backend_status; MadLoop_return_code;
  raw Born/finite/single_pole/double_pole and split-order values;
  elapsed_seconds and actual DP/MP/backend counters if timed;
  raw_stdout/stderr/input artifact hashes
derived point row:
  matched three raw row IDs; classification; mean; spread numerator;
  finite delta or null/unbounded reason; exact sample denominator
```

Finite values should retain sufficient round-trip decimal digits. Capture
controller status independently of numerical zero. Bind each figure/table to
the specific raw and derived artifacts used to generate it. Unavailable tool
rows, incomplete campaigns, and failed gates must remain visible.

## Existing helper limitations and reusable hooks

Static inspection of the frozen MG7 source found that unmodified
`check_stability` uses a falsey guard that treats valid zero as failure, a
rescue selector restricted to tools 1 and 6, and a parser whose default finite
value can mask a missing `FIN` line. It also disables COLLIER poles and global
cache for its standalone mode-1/mode-2 protocol. These observations justify
using a separate explicit driver and tested parser for the new stream. They
do not establish a scalar-library defect.

Explicit forced-DP modes intentionally return an untested `H=1` and
`accuracy=-1` after initialization. Those fields alone are not numerical
failure. The normal accepted-output parser requires a tested accuracy and is
therefore unsuitable for this stream. Conversely, complete finite output is
not sufficient to certify that every reduction succeeded: retain reduction
rejection information independently. A forced-DP control must not implicitly
invoke the normal forced-stability/rescue wrapper while collecting what it
labels as one unrescued DP result.

`LoopMatrixElementTimer.skip_loop_evaluation_setup` toggles generated
`SKIPLOOPEVAL` for the DP coefficient-only experiment; `boot_time_setup` toggles
`BOOTANDSTOP`. The legacy stage results subtract separate executions and
boot costs. Any adaptation should keep coefficient-only builds and their
source hashes distinct from the unmodified production evaluation executable,
use matched point streams and identical pole settings, and report the
subtraction uncertainty. A negative or unresolved inferred stage cost is a
measurement limitation, not zero cost. No stage experiment may suppress
production rescue merely to improve the headline runtime.
