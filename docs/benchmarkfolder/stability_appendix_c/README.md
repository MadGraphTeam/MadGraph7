# Standard Model stability comparisons

The [31-page benchmark report](../benchmark.pdf) includes four raw stability
plots on pages 27–30 and eight post-rescue comparisons on pages 18–25. Both
studies cover top-pair QCD, Zgg QCD, dimuon EW and ZZgg QCD processes, with
CutTools, Ninja, COLLIER, NinjaRS and TIRust on identical saved points.

The [closed raw pilot](raw-dp-pilot-5-1/README.md) has 100 newly generated,
cut-qualified points per process. Its [four plots](raw-dp-pilot-5-figures-1/README.md)
show three-probe consistency tails and ratios to Ninja. Eighteen backend/process
samples have 100 available diagnostics. TIRust has 24/100 available for Zgg and
0/100 for ZZgg; all 176 unavailable outcomes remain explicit unknown bounds.
Nineteen completed requests do not mean nineteen accuracy passes. Full raw
data, failures and [independent relocation checks](raw-dp-pilot-5-verification-1/README.md)
are retained.

The [first-failure diagnosis](raw-dp-pilot-5-diagnosis-1/README.md) identifies
rank-4 pentagon and rank-5 hexagon failures. MadLoop's persistent flag is a
software reduction-failure latch, not an observed hardware IEEE exception.
The ZZgg latch was already set during bootstrap. Bridge status/error details
were not recorded, so the underlying cause remains open. The diagnosis does
not replace unavailable outcomes with successful points or change the plots.

The separate [eight retrospective plots](retrospective-main-closed11-1/README.md)
use the closed ordinary campaign's accepted post-rescue estimates and finite
disagreement with current CutTools. Those original ordinary samples are not
asserted to pass the paper's cuts. Neither study supplies independently known
true errors or performance acceptance. Production sources are unchanged.
The protocol below describes the executed raw pilot; sealed earlier source
proposals and failed attempts remain preserved.

The separate [three-point genuine-QP diagnostic](../experiments/qcd-n3-genuine-qp-diagnostic/run5-coefficient-analysis-1/README.md)
investigates ordinary Zgg target misses, using four reducers with observed
common upstream QP inputs. All finite and pole symmetric discrepancies are
below `1e-10`. These unchecked returned values establish consistency only;
CutTools retains its binary64 denominator ABI. This is neither a stability
distribution nor a replacement for the raw pilot or 100-point campaign.
The [complete journal verifier](../experiments/qcd-n3-genuine-qp-diagnostic/run5-retention-1/verify.py)
replays ten inherited and six new workers without running numerical code.

## Reference and interpretation

Hirschi and Peraro, [arXiv:1604.01363v1](https://arxiv.org/pdf/1604.01363),
section 7.2 and appendix C, study the tail fraction whose estimated relative
inaccuracy exceeds a target. Their double-precision diagnostic combines reversed
propagators and a Lorentz transformation. Ordinary samples use 1 TeV,
final-state transverse momenta above 50 GeV and pair separation above 0.5.
Samples contain 100,000, 10,000 or 1,000 points for two, three or more outgoing
particles. Most results sum helicities; loop-induced profiles select helicities.
Figure 4 covers top-pair production with additional gluons; figure 6 covers
high-multiplicity Standard Model processes; figure 7 covers loop-induced
multi-Z production, without the transverse-momentum cut and with specified
helicities. Figure 5 uses effective Higgs/spin-2 interactions. The appendix
uses logarithmic tail panels and ratios to Ninja. These are internal
consistency diagnostics, not independently known true errors.

## The first current-source pilot

The pilot uses four already CLI-generated and fully compiled Standard Model
processes, with current explicit backends 1, 6, 7, 8 and 9:

| Case | Model | Process |
|---|---|---|
| QCD 2→2 | `loop_sm` | `g g > t t~ [virt=QCD]` |
| QCD 2→3 | `loop_sm` | `u u~ > z g g [virt=QCD]` |
| EW 2→2 | pinned `loop_qcd_qed_sm_Gmu` | `e+ e- > mu+ mu- [virt=QED]` |
| QCD 2→4 | `loop_sm` | `u u~ > z z g g [virt=QCD]` |

Each process receives a newly named, deterministic stream of 100 physical
points with the declared cuts. The existing ordinary fixtures cannot simply be
relabelled: 100/100 top-pair, 80/100 Z+two-gluon and 98/100 dimuon points pass
both strict cuts. The new stream is generated before any result is evaluated;
it does not select points according to reducer performance. All five backends
use exactly the same stream and physical card. `points.py` reuses the pinned
massive RAMBO map, checks momentum conservation and external mass shells, then
rejects input configurations outside the declared cuts. It preserves local RNG
state and reproducible prefixes. It makes no integration-weight or uniform
massive-density claim. The actual external masses must match the card-bound
compiled input evidence before the point files are frozen.

The private driver records original ordering (mode 1), reversed ordering
(mode 2) and a signed 90-degree rotation around the beam (mode 1). It sums
helicities and retains every answer order: Born, finite, single and double
poles, raw host accuracy and return code. The main curve uses the total
answer, order zero, once per physical point. Duplicate split-order answers
cannot increase the sample size. Initialization calls are retained separately
and never enter the measured distribution.

The card must request `CTModeInit=1`, `CTModeRun=1` and a singleton backend
before the first matrix-element call: MadLoop loads its card inside that call.
The driver never requests automatic precision rescue. Forced DP normally
returns unchecked code 110. Code 120 indicates unchecked QP work and is
excluded from a raw-DP profile; initialization code 130 is separate. These
codes cannot identify the selected reducer through their unit digit, so a
card-selected backend is labelled as such unless an independent call observer
records an actual backend. No accepted backend ID is invented from code 110.
An unchecked finite amplitude can also conceal a failed reduction whose tensor
coefficients were cleared. Every diagnostic therefore requires explicit
native stability and finite-output coverage for its active nonzero numerator
loop groups. Failed or unknown coverage makes the point unavailable, even
when its unchecked return code is 110 and a finite remainder was printed.

COLLIER global caching and TIRust native caching are disabled for independent
probes. COLLIER uses the external direction check, requested accuracy 1e−13,
and explicit UV/IR pole computation; the known COLLIER cache bug is not fixed
by this work. Finite computation and both pole types stay enabled for all
libraries. The production TIRust library is used, without private adaptive,
mass, geometry or rescue patches. Full stdout and stderr, including `##W03`,
are retained. Warning-invalidated and unavailable results cannot become
successful numerical samples.

The protocol requests 20 workers and 6,000 primary DP matrix-element calls,
plus a bounded DP-only bootstrap. The delivered pilot retains 5,954 returned
primary calls and 46 unexecuted calls; bootstrap counts stay separate.
Workers have a 1,200-second execution cap and 4 GiB memory
cap, on nonreserved CPU 12. Backends 8 and 9 respect the existing common
numerical lock, with separately accounted bounded lock waiting. No failed or
timed-out point is silently replaced. Existing source exports are read-only;
private copies hold cards, generated driver, objects, filters and output files.
Only the new private driver is compiled/relinked: no numerical library or
generated amplitude source is rebuilt. Preparation contains four private
driver compile/link jobs and four exact-PT_INTERP loader-list queries, with
zero amplitude evaluations. Compiler and helper utilities use a clean outer
environment; target-specific native library paths are applied only at the
final target execve. Actual target maps must match the fresh executable and
its loader closure; a historical host ldd interpreter is not a required
native mapping.

## Quantities plotted

For the three finite observables at point i, the raw-DP consistency metric is

```
a_i = (max(x_original, x_reversed, x_rotated)
       - min(x_original, x_reversed, x_rotated)) / abs(mean(x_*))
```

The Born-process observable follows the finite/Born convention; the common
coupling normalization cancels from this relative spread. All identical zeros
give zero; nonzero spread with zero mean gives positive infinity. Neither
result is clamped. Raw coefficients remain available so this convention can
be audited. Finite, single-pole and double-pole coefficients are retained
separately for further comparisons; the raw plots here use finite/Born only.

For target Δ, the curve is `count(a_i > Δ)/100`, with a strict greater-than
comparison. Smaller tails mean more points pass this particular diagnostic.
The top panel has logarithmic axes and the lower panel shows the ratio to
Ninja on a linear axis, retaining valid zero ratios. Ratios are masked if
either intended population is incomplete or Ninja has no exceedance; no
pseudocount or denominator floor is inserted. Exact-zero metrics are counted
in an annotation and raw data, and omitted from logarithmic y drawing.

Failures, timeouts, warning-invalidated samples and unexecuted points remain
in the intended denominator. With k known exceedances and m unavailable
points, the unknown tail lies in `[k/100, (k+m)/100]`; the plot shows these
bounds rather than an apparently complete curve. Missingness reasons and
point IDs remain in the machine-readable evidence. Ordinary and stress
populations are shown separately.

At 100 points, the smallest nonzero empirical fraction is 0.01. The pilot
cannot establish 10−3 or 10−4 tail rates. It reports finite fixture frequencies
without population confidence claims. A larger frozen random campaign may
add explicitly pointwise binomial intervals; those are not simultaneous
confidence bands across all thresholds. Even zero observed failures would
not establish a population failure probability of zero.

## What existing accepted outputs can establish

The earlier closed QCD 2→3 portable analysis supplies 100 physical points per lane:
`experiments/invariant-symmetry-qcd-n3-target-analysis/portable-analysis2`,
index SHA256
`79f57a8d83ea74249af83d12908de299372714cd7587d59d4ae31cfa9f63e457`.
Its accepted coefficients can support reducer-disagreement tails; its host
accuracy values can support a separately titled, post-rescue composite
stability-estimate plot. Both require the closed evidence verifier before
conversion. Their coefficient disagreement uses the existing regression
metric `abs(a-b)/abs(a+b)`, not the three-probe metric above. Neither plot is
an independent true-error distribution.

The accepted outputs do not contain pre-rescue DP values. Their accuracy
combines finite and pole diagnostics, and COLLIER may use a different internal
estimator. Accepted DP can follow QP work; the H/T/U status must be preserved,
not collapsed into a “native DP” label. Exact diagnostic equality, including
the 13 zero TIRust estimates in that closed QCD 2→3 analysis, does not certify
infinitely many correct digits. The current four-case retrospective capsule
uses the later closed eleven-process ordinary campaign. Stress and CMS results
have separate scopes and do not enter these ordinary or raw-pilot curves.

## Reproduction and larger local study

Verify the delivered raw data and recompute all four statistical sidecars
without compiling or running any amplitude:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1/verify.py
python3 -B docs/benchmarkfolder/stability_appendix_c/figure-source-1/figures.py verify \
  docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-figures-1 \
  --data-root docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1 \
  --plotter-root docs/benchmarkfolder/stability_appendix_c/plotter-source-6
```

The expected data and figure index hashes are respectively
`ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e` and
`b131e6294fb3c6900e6776339948928ae71518092cc99de59a8bb0d39fdabcfd`.
Render the four plots into a fresh ignored directory:

```sh
python3 -B docs/benchmarkfolder/stability_appendix_c/figure-source-1/figures.py render \
  .tirust-build/stability-figures-reproduction-1 \
  --data-root docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1 \
  --plotter-root docs/benchmarkfolder/stability_appendix_c/plotter-source-6 \
  --expected-data-index-sha256 ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e
```

Rebuild the combined PDF with the [report commands](../README.md#build-the-report).

The public CLI already offers the basic diagnostic:

```
import model loop_sm
check stability 100 g g > t t~ [virt=QCD] --reduction=1|6|7|8|9 --energy=1000 --seed=20261121
```

This is a reference command, not the paired pilot execution. The current CLI
draws fresh random points inside each backend loop, only tries QP when its
raw DP spread exceeds 0.1, and uses a driver without per-call return-code or
warning attribution. The paired pilot therefore uses common frozen points
and a private instrumented driver. It preserves the raw DP diagnostics even
when an individual check fails.

Source tests, which launch no compiler or amplitude:

```
source tools/tirust_env.sh
python3 -B -m unittest discover -s docs/benchmarkfolder/stability_appendix_c -p 'test_*.py'
```

The controller exposes separate `plan`, `prepare --execute` and `run --execute`
stages. Root review and stage admission precede execution. `plot_metrics.py`
accepts only an explicitly named normalized closed-data JSON file; it never
discovers “latest” attempts. Rendering creates vector PDF, PNG and JSON
sidecars under a fresh output prefix and refuses overwrites.

Hua-Sheng’s local follow-up should enlarge the cut-qualified samples to
100,000/10,000/1,000 points for 2→2/3/4, and compare raw DP, genuine QP and
normal rescue-chain profiles in separate panels. Additional useful SM
channels are top-pair plus one/two gluons and loop-induced `g g > z z`
(optionally three/four Z bosons). The latter needs separately declared
helicity choices and an independently labelled small-pT population, rather
than inheriting this pilot’s pT cut. Finite-term truth needs an independent
qualified reference; agreement and self-consistency alone cannot supply it.
The precise runtime acceptance remains the separate local timing handoff.
