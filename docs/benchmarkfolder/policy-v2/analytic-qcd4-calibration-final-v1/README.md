# Reproduce the QCD4 leading-pole calibration

This capsule checks the retained original-helper corrected CutTools
calibration for `u u~ > z z g g [virt=QCD]`. It contains 100 fixed phase-space
points and 200 checked double-pole rows: the total and the sole physical
split order at each point. Those two rows coincide; they are not 200
independent events.

Run with Python's standard library:

```sh
python -B docs/benchmarkfolder/policy-v2/analytic-qcd4-calibration-final-v1/reproduce.py
```

The directory can also be copied elsewhere and `reproduce.py` run there.
Every source, card, point, raw calibration row and normalization input it
reads is retained locally. Historical worker/reference paths are resolved
only through the explicit byte-exact `input-index.json` map. No live worker,
reference clone, model import, network access, compiler or amplitude call is
needed. The historical executable hash is preserved; no executable is
included.

The reconstruction uses the actual card's `alpha_s=0.118`, generated
`G=2*sqrt(alpha_s*pi)` and DP pi literal, common Born/loop averaging and
CutTools interference normalization. With two massless fundamental colored
legs and two gluons, the same retained binary64 expression is
`double/Born = -(2*(4/3)+2*3)*G**2/(8*pi**2)`.
Its coefficient is `-0.16276245513531162`. The worker does not evolve
alpha_s in its `UPDATE_AS_PARAM` path. Archived FKS and Binoth sources
establish the convention; standalone execution of those routines is not
claimed.

The reproducer verifies all retained hashes and the actual source/card
bindings, then recomputes every prediction, residual and row check using the
historical common-point Laurent/Born scale and unchanged tolerance
`1e-7*max(abs(observed),abs(predicted)) + 1e-10*point_scale`.
All 200 checks agree with the preserved historical records. The maximum
relative double-pole error is `4.646150815304775e-13`; the maximum fraction
of tolerance is `4.617779533184807e-6`.

`historical/calibration/` preserves the four original payloads and manifest
unchanged. The independent derivation and root reproduction receipts also
remain unchanged under `historical/`. Their old paths document the original
state; the new local input index maps each bound path to deduplicated,
content-addressed bytes. No historical record is silently rewritten.

This is only a leading-double-pole proportionality diagnostic using the
same retained Born amplitude. It is not an independent Born calculation,
finite/single-pole oracle, stability/rescue test, timing measurement,
all-backend comparison or complete-campaign acceptance gate.
