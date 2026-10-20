# EW3 stability and generated-model precision investigation

These are diagnostic experiments, not performance acceptance data. All use
retained `e+ e- > t t~ a [virt=QED]` physical inputs. The original archive,
original generated outputs, failed core run, model download and parameter card
remain unchanged.

## Findings

The original Collier run returned status 410 at points 68 and 74 with integral
accuracy `1e-10`. Mode 1 at `1e-12` removed these flags, but finite coefficients
at points 49 and 78 still differed from original Ninja's normal QP rescues.
Modes 2 and 3 were also explored; failed status flags and mode-3 crashes are
retained rather than counted as successful observations.

Original Ninja and CutTools reproduced each other with either numerator
precision. Their QP-integrand results differed from their DP-integrand results
even when both reductions used QP. The compiled coupling probes isolated the
difference to model counterterms before reduction. Copying rounded QP
counterterms into a DP diagnostic reproduced the old QP amplitude, but **this
was not a valid correction**: independent helper tests subsequently established
that the generated QP model helpers were inaccurate.

The actual generator defects were:

- Nineteen two-argument `CMPLX` calls without kind 16, rounding trajectory and
  pure-axis arguments through default complex precision.
- Six mistyped pi constants in continuation formulas.
- Default-kind `REAL` casts in MP expression generation, rounding real
  binary128 inputs through default real precision.

The fixes are in `madgraph/iolibs/export_v4.py` and
`madgraph/iolibs/ufo_expression_parsers.py`. The UFO itself was not modified.
The independently compiled analytic controls and source audit are under
`mp-helper-audit/`; the production regression tests also compile actual
generated helpers and parsed expressions.

The fresh corrected CLI export passes all six backend pairs at all 100 points
and both total/split-order slots. The maximum tolerance fraction is
`0.000912253`, well below one. Counts are CutTools `211×99,329×1`, Ninja
`216×98,328×2`, Collier `217×100`, and NinjaRS `218×98,322×2`. These results use
normal stability/rescue, common `MLStabThres=1e-9` and common Collier accuracy
`1e-12`. They are single-traversal numerical diagnostics; no performance
acceptance or completed final-policy claim follows.

## Retained variants and limitations

| Evidence | Purpose and interpretation |
| --- | --- |
| `mode*-accuracy*` | Original Collier accuracy/mode diagnostics, including failures |
| `backend*-qp-integrand*` | Separate numerator precision from reduction precision |
| `couplings-v2`, `internal-parameters` | Original DP/QP model arithmetic comparison; old QP is not an oracle |
| `rounded-qp-counterterms` | Establish counterterm origin only; deliberately not a production fix |
| `mp-helper-audit` | Independent axis, continuation, threshold and cast proofs |
| `corrected-helpers-*` | First helper-only experiment; all 100 points agree across legacy reducers, but it accidentally omitted the unused user-function tail |
| `corrected-helpers-v2-*` | Scoped helper correction that explicitly preserves the entire user-function tail |
| `fresh-export` | New actual CLI export with helper and parser corrections; identical parameter card |

The first `corrected-helpers-*` script regenerated built-in helpers with an empty
UFO function list. This omitted `CRECMS`/`MP_CRECMS` after `MP_ARG`; although the
linked process ran, that source delta exceeded the claimed helper-only scope.
Those outputs and sources remain retained. Version 2 preserves that tail
byte-for-byte and is the relevant scoped diagnostic.

Some early forced-QP initialization variants stalled because the benchmark
driver requested stability estimates in a mode that does not provide them.
The failed runs are retained. The `forced-after-warm` diagnostics initialize
normally, then enter forced QP without requesting a stability estimate; their
status 120 means a forced diagnostic, not a normally accepted point. The final
fresh-export numerical checks use normal stability and rescue.

## Reproduction

Run from the repository root in the pinned development shell. Each script
refuses to overwrite its existing work directory. Use a fresh checkout/output
location when reproducing retained evidence.

```sh
python docs/benchmarkfolder/experiments/ew3-collier-stability/probe.py
python docs/benchmarkfolder/experiments/ew3-collier-stability/numerator_probe.py
python docs/benchmarkfolder/experiments/ew3-collier-stability/coupling_probe.py
python docs/benchmarkfolder/experiments/ew3-collier-stability/coupling_probe.py --internal
python docs/benchmarkfolder/experiments/ew3-collier-stability/helper_fix_probe.py
python docs/benchmarkfolder/experiments/ew3-collier-stability/fresh_export.py
python docs/benchmarkfolder/experiments/ew3-collier-stability/fresh_validation.py
```

The scripts preserve raw coefficients, return codes, cards, build/run logs, and
hashes. The fresh validation's single traversal is numerical evidence only.
The independent benchmark policy review explains why original timings must
remain unchanged while corrected original exports serve as separately named
correctness references. Final core acceptance requires a versioned policy,
refreshed protocol proof and complete measurements.
