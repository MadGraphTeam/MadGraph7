# Final loop-induced supplement: preparation only

The preregistered policy is `../final-supplement-plan-v1`. The eight core cases,
their original failures, thresholds and frozen worker implementation are unchanged.
No loop amplitude or Symbolica worker has been evaluated in this namespace.
Final candidate generation and all numerical collection require the root's
source/runtime freeze and serialized engine handoff.

Actual CLI exports from untouched MadLoop revision
`27147b3300f556a75661a886e92bdbe00558ee25` generated and built
`g g > h [sqrvirt=QCD]` and `g g > h h [sqrvirt=QCD]` on CPU affinity 6,7.
The frozen driver ran only `--masses`, which exits before amplitude evaluation.
All 6,009 members of `git archive` at that revision were compared byte for byte
against the archived source; no original source was changed.

The first seeded point streams are retained without replacement:

| Stream | Points | SHA256 |
|---|---:|---|
| ggH | 100 | `146ab1c08480c2687352d1800ba12c00633097d109c9f3b084bb69c9a41dfa5b` |
| ggHH | 100 | `3f05e279caa8057906bce524597066c479454b89d455ca57a42691f5060004f6` |
| ggHH top-threshold grid | 12 | `14fa79666ef3448a3d90d7d0b001b45188e62acb2783fac64b17e8da8f6ec299` |

The ggH stream contains 100 distinct boosted configurations and one constrained
physical invariant, s = mH². Its serialized invariant spans only floating-point
roundoff around 15,625 GeV²; those variations are not independent invariant
configurations. ggHH uses the unchanged RAMBO generator at 1,000 GeV.
The threshold grid uses the final physical card's mt = 173 and mH = 125 GeV,
the twelve preregistered delta/cos(theta) pairs, zero widths and azimuth zero.
The largest normalized shell residual is 1.22e-16; conservation is exact for
the serialized grid. Both original and portable metadata records are retained;
the latter points to the same physical card in retained report inputs.

Private helper-corrected references are separate generated-output clones.
Only the exact two MODEL postimages from the core policy's control-gg-h and
control-gg-hh correction records were copied. Full generated-source inventories
verify that every other file is unchanged. All MODEL objects/archive and
subprocess objects/executable were removed before clean rebuilding; original
DHELAS and reduction libraries remain unchanged. These references never replace
the literal-original timing denominator.

Three failed preparation/checker attempts remain visible: raw `git ls-tree`
includes export-ignored `.bzrignore`, so archive integrity uses actual archive
members; initial physical-card extraction used list indexing instead of the
ParamCard tuple key; and an initial checker expected a nonexistent `passed`
field rather than recomputing the rebuild conditions. No reduction ran in
these attempts. Their corrected checks pass. Early source-anchor drafts are
retained; `source-proofs/with-cli-v2` additionally binds the actual CLI card/log.

## Reproduction and launch order

Run from the repository root in the pinned development environment:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python tools/ninjars_supplement_v2.py verify-preparation --live
python -m unittest tests.unit_tests.various.test_ninjars_supplement \
  tests.unit_tests.various.test_ninjars_precision
```

The current namespace is immutable once sealed. A fresh checkout reproduces
generation using `prepare`, `generate-baseline`, `threshold`, `helpers` and
`anchors`, each with a fresh explicit `--report`/`--work` namespace. `helpers`
consumes the retained core correction rather than changing the original archive.

After the final committed candidate and installed archive are declared, root may
run `generate-candidate` and `specs` here. That writes four separate collection
specifications: seven-target original/candidate timing (CPU207, seven pairs),
three helper references (two commands), and the separate four/three-target
threshold correctness collections (two commands each). Every collection uses
unchanged `workers.run_case` and retains failures. Execution requires
`collect --spec SPEC --execute-numerical`; it must be serialized with all other
Symbolica work. The numerical source/runtime signature must match the final
core cohort before report assembly.

`assemble --expected IDENTITIES --output DATASET` creates an immutable raw-bound
supplement manifest, with precision controls pending unless `--precision-proof`
supplies the independently validated final installed-archive result. Report
validation recomputes all pairwise comparisons, estimates, status/rescue counts,
and timing statistics. Estimates are never reported as actual precision, and
fixed streams/grids never imply a probabilistic instability-rate bound.
