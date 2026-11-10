# Closed QCD n3 genuine-QP diagnostic — source proposal 4

This source-only amendment corrects the diagnostic parser's model of existing
MadLoop call order. Generated loop_matrix.f computes Born on original P_USER,
then repairs separate DP and QP copies for loop evaluation. Original Born and
repaired loop momenta can differ within a call. The diagnostic must observe
that behavior without changing it or pretending the two inputs are identical.

Closed run3 entered two CTmode4 workers and both native processes completed.
The frozen source3 parser rejected observed input equality before adjudication:
it incorrectly required original Born momenta to equal later DP-repaired loop
momenta. failed-run3-retention-1 preserves every actual file, loader/alias/maps
proof and trace. Its portable verifier reproduces that exact failure. It remains
a failed attempt; this amendment does not relabel the original result.

Independent static/generated-source and byte analysis shows the same order on
all three points: public/Born/DP-before are byte-identical; QP-before is exact
binary128 widening of original public input; DP-after is rounded QP-after;
MP construction uses QP-after. The documented repair changes8/9/8 DPcomponents
at points1/93/39. Observer ABI extents and offsets are consistent with the
pinned Fortran and captured records; this is an analyzer assumption error.

The new parser requires Born==public-original==DP-before and byte-exact
binary64→binary128 widening(original)==QP-before, including signed zero. It
records full original public, Born entry/return, DP repair before/after, QP
repair before/after and MP construction before/after bytes separately. Changed
DP component indices and within-call equality are explicit witnesses. No repair
is suppressed or substituted. All subsequent observed-input assertions,
native failure/group checks, warning rules, common-input digest and cross-arm
adjudication remain unchanged. A cross-worker difference in repaired inputs
still blocks coefficient comparison. parser-delta-guard.json records those
unchanged functions and the exact adjacent-gate source block.

closed_replay.py first verifies the immutable source3 failure-retention packet,
then dry-parses BOTH complete retained CTmode4 arms through all subsequent new
parser gates. closed-run3-preflight.json retains their full parsed records,
original failure, input transitions and native observations. Both parse; public
input/output/status/IEEE bytes match. Each of the three observed points has
116 native evaluations,48 helicities and7625 actual QP numerator callback pairs.
This is a two-arm pure replay, not a sixteen-worker adjudication or scientific
acceptance. The other fourteen arms are not inferred, and status120/accuracy-1
remains unchecked. No new amplitude is evaluated by this preflight.

All68 pure tests pass, including changed Born/DP-before, low-bit or signed-zero
QP-before errors, cross-worker DP-after mismatches, exact finite widening and
the existing native/group/IEEE/W03 failures. A passing parser is not evidence of
truth by reducer voting, either numerical tolerance, or campaign acceptance.

The driver, observers, process supervisor, exec wrapper, fixtures, generated
numerical objects, all fifteen original runtime libraries and existing build2
executable remain byte-identical to source3. There is no compilation or relink.
The source2 build proof and its original glibc2.40 PT_INTERP remain mandatory.
All previous failures and source seals stay bound. The only controller change
is the new one-shot run4 namespace and its private closed-runtime alias path.
The exact donor SONAME/hash alias mechanism, separate strict trace before each
worker and actual sampled /proc map validation remain unchanged. Maps remain
observational, not a pre-entry barrier or continuous race-proof monitoring.

The unchanged future matrix has sixteen serial workers: backends1/6/8/9,
modes4/5, plain/observed, physical points1/93/39 after bounded original point1
initialization. Maximum192 public calls include48 diagnostic calls and96
returned order columns. QP flags/MLStabThres1e-6/ImprovePSPoint2/full finite and
poles, journal200000records/128MiB, CPU12nonexclusive/one thread/4GiB, 180s worker
caps,30s trace caps and shared3000s budget are unchanged. All traces and checks
count toward that budget. The nonblocking common lock precedes run4 creation;
busy75 is a scheduling hold. Execution stops on the first failure with no retry.

No native run4, loader trace or real runtime alias directory was created while
preparing this proposal. Root review of the sealed source, retained evidence
and independent review remains required before any actual run4 admission.

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal4/run.py verify
# Pure full replay of the two closed CTmode4 arms:
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal4/closed_replay.py
# Only after root admission:
python3 -B docs/benchmarkfolder/experiments/qcd-n3-genuine-qp-diagnostic/source-proposal4/run.py run
```

prepare.py remains provenance and must not be rerun over this proposal. The
future controller depends on closed workspace donors; portable retention of a
new scientific attempt can only be produced after its actual terminal exit.
