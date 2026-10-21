# Pristine-checkout software fixture portability

The final selected run executed **308 tests across 16 modules: 307 passed, one
unchanged conditional skip, zero failures or errors**, on CPU 180. Every module
loaded from a private `git archive` of revision 690202028535afb2cb699cc95d5e9a4b365a3c28
with declared exact test/helper overlays. Both ignored fixture parents were
absent before each module, so execution order cannot hide missing setup.

This is a narrow, additive software/fixture qualification. The unchanged
`../regressions-update-v1/manifest.json` (e4752a6a…) remains the separate production
interface regression qualification from source 856e76e669618ec8d0f799b9c8e6f86819dcd986.
No existing report qualification reader was changed. Earlier exact-source
paper/report/timing proof capsules are historical and correctly reject changed
current test files with their optional live checks.

The fixes create temporary-directory parents in test fixtures, replace one
untracked historical finalizer dependency with a bounded synthetic fixture, and
make two old report tests check the sealed historical 90/89/1 observation and its
rejection against the changed current generator. No production validator,
helper, bridge, precision threshold, compiler or physics source changed here.
The separately owned DP semantic checkpoint b0fa3802/4688413f is explicitly bound
as an independent overlay; its 30 software tests are included without claiming a
new numerical run.

The retained initial 14-module run attempted 276 tests and exposed 212 errors and
two failures. The added loop-preparation module separately exposed two missing
parent errors. The first fixed 15-module replay still exposed five historical
untracked-source errors and the two stale report expectations; those observations
remain retained. The final 15-module replay passed 278 tests with one existing
skip, followed by the cumulative 16-module, 308-test replay. Raw logs and source
images are retained in deterministic compressed payloads; the large private
source archive and generated files are excluded.

The sole skip is `test_actual_generated_qcd2_count_definitions_agree` in the
paper-report module: **Actual final generated source is not retained in this
checkout**. Its original conditional skip was preserved; no new skip was added.
The independent actual generation/numerical observations remain in their own
qualification capsules.

`python3 verify.py` checks the payload seal, actual raw-log unittest footers,
module/source/command custody, exact skipped test, all before/after observations,
and snapshot hashes. `python3 verify.py --live ROOT` also requires every selected
source and the separate production regression manifest/source binding to match
that checkout. The offline check remains useful after later source changes; a
live mismatch is an error and must not be waived.
The verifier explicitly rejects optimized Python (`-O`/`-OO`) so its assertion
checks cannot be disabled.

`result.json` records the exact runtime commands, environment, source revisions,
counts and separate production qualification. Replay using the recorded base
archive plus `software-sources.tar.gz`; the retained controller creates only
private ignored parents and launches software unit modules, fake protocol
children and Typst. There were **zero native reduction, actual installation or
amplitude calls** in this campaign. Timing above is software test elapsed time,
not a MadLoop performance result.
