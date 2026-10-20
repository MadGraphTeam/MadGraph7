# Generated-process numerical controls

The retained controls pass the required numerical comparisons. The separate
`1e-20` random-point diagnostic also exposes a stability rejection in original
Ninja; that rejection remains a failure in the evidence and is not used as a
successful timing measurement.

The machine-readable overview is
[`data/generated-controls-summary.json`](data/generated-controls-summary.json).
These runs check correctness. They ran alongside other task work, use one
recorded repetition, and do not establish the 1.05 runtime gate.

## Results

Every stream contains ten points. Each comparison includes the Born term,
finite part, both poles, the summed result, and every generated split order.
The tolerance is unchanged from the core comparison:
`abs(a-b) <= 1e-7 max(abs(a),abs(b)) + 1e-10 S`, where `S` is the largest
absolute Born/Laurent coefficient at that point across the compared streams.

| Control | Independent comparisons | Result |
| --- | --- | --- |
| Loop-induced `g g > h [sqrvirt=QCD]` | CutTools, Ninja, Collier, NinjaRS; legacy reducers against untouched baseline | Pass |
| Loop-induced `g g > h h [sqrvirt=QCD]` | Same four reducers and baseline regression | Pass |
| Mixed QCD/QED `u u~ > d d~ QCD=2 QED=2 [virt=QCD QED]` | Same four reducers, all split orders, baseline regression | Pass |
| Complex-mass `e+ e- > mu+ mu- [virt=QED]` | Same four reducers, nonzero model widths, baseline regression | Pass |
| Spatially rotated `g g > h h` | Four reducers and original-versus-rotated result for each | Pass |
| Changed top mass and renormalization scale in `g g > h h` | Four reducers with the same parameter sequence | Pass |
| Retained versus freshly destroyed/recreated NinjaRS context | Same changed-parameter sequence and points | Bitwise-equal printed coefficients |
| Near-singular `g g > t t~`, stability threshold `1e-20` | CutTools QP, original Ninja QP, own NinjaRS QP | Pass; ten genuine QP rescues per reducer |
| Random `g g > t t~`, stability threshold `1e-20` | CutTools QP versus own NinjaRS QP | Pass; ten genuine own QP rescues |
| Same random-point diagnostic including original Ninja | All returned coefficients agree, but original Ninja rejects its stability estimate | **Fail: original Ninja status `420` at all ten points** |

The near-singular comparison has a maximum point-scaled absolute difference
of `2.06e-13`; the random-point CutTools/own-NinjaRS QP comparison reaches
`2.18e-15`. All comparison decisions and their numerical margins are retained
in the individual `data/agreement-*.json` files.

Successful NinjaRS rescue is identified by the hundreds/units digits `(3,2)`.
Status `322` means that the QP integrand was evaluated as well as the MP
reduction; `312` means MP reduction of the DP integrand. The default full-QP
path used here returns `322`. The matching near-singular statuses are `329`
for CutTools and `328` for original Ninja. Collier has no QP path and is not
included in the forced-QP comparison.

For the random-point diagnostic, original Ninja's QP stability estimates range
from approximately `4.9e-16` to `1.5e-13`, exceeding `1e-20`, although its returned
coefficients agree with the successful QP references. MadLoop computes this
spread from final results stored as `REAL*8`; the threshold is a deliberate
rescue trigger, not a claim that the reported observable has twenty accurate
decimal places. The original Ninja failure, output rows, and diagnostic log
are retained under `diagnostics/control-forced-qp/6/`. Neither the threshold
nor the coefficient-comparison tolerance was relaxed. The independent
near-singular stream passes the same strict threshold in all three reducers.

## Inputs and transformations

`controls.json` records the models, processes, seed, and stability settings.
The ordinary phase-space points come from the common benchmark generator.
Complex masses use the model's complex-mass scheme and retained nonzero
widths. Parameter cards and backend-specific MadLoop cards are in `inputs/`.

`tools/ninjars_controls.py prepare` creates isolated variants of the generated
candidate outputs; it does not modify the core eight-process outputs:

* **Rotation:** apply `R_x(cos=5/13,sin=12/13)` after
  `R_z(cos=3/5,sin=4/5)` to every spatial momentum in the original `ggHH` stream.
  Energies remain unchanged.
* **Changed parameters:** repeat the first `ggHH` point, assign top-mass factors
  `[1,1.125,.875,1,1.125,.875,1,1.125,.875,1]` and renormalization-scale factors
  `[1,2,.5,1,.5,2,1,2,.5,1]`, and call the generated full `COUP()` routine after
  each assignment. The same compiled custom driver is used for all four
  reducers. The changed mass produces different physical results, and each
  return to the original parameter state reproduces its original result.
  The fresh-context variant additionally calls `NINJARS_FINALIZE()` before
  each evaluation; its coefficients match the retained-context variant.
  These process-level parameter-change checks use the stable DP path.
  Actual MP mass/scale reset coverage is also present in the bridge tests.
* **Near-singular:** use physical `gg -> ttbar` momenta at `sqrt(s)=1000 GeV`,
  the generated top mass, scattering angles
  `[1e-2,7e-3,5e-3,3e-3,2e-3,1e-3,7e-4,5e-4,3e-4,1e-4]`,
  and azimuthal cosine/sine `(0.6,0.8)`. Both incoming momenta lie on the
  beam axis and outgoing momenta are opposite. Both MadLoop rotations are
  enabled in DP and QP.

The exact transformed momentum streams, parent checksums, transformation
parameters, and conservation/mass-shell residuals are retained in `points/`.
`control-variants.json` records each variant's parent and settings.

## Reproduction

Use the pinned environment and installed libraries described in `README.md`.
Run commands from the repository root. Choose an available CPU in place of
`1`. Serialize Symbolica MP processes under the shared
`.ninjars-build/symbolica-locks` directory.

```sh
python tools/ninjars_benchmark.py generate --source both \
  --config docs/benchmarkfolder/controls.json
python tools/ninjars_benchmark.py build --source both \
  --config docs/benchmarkfolder/controls.json
python tools/ninjars_controls.py measure --cpu 1 \
  --cases control-gg-h,control-gg-hh,control-mixed,control-complex
python tools/ninjars_benchmark.py measure --source baseline --cpu 1 \
  --config docs/benchmarkfolder/controls.json --backends 1,6,7 --repetitions 1 \
  --cases control-gg-h,control-gg-hh,control-mixed,control-complex
python tools/ninjars_benchmark.py compare --source both \
  --config docs/benchmarkfolder/controls.json \
  --cases control-gg-h,control-gg-hh,control-mixed,control-complex
python tools/ninjars_controls.py measure --cpu 1 --cases control-forced-qp
```

The last command retains every reducer's results and then exits nonzero if
the strict diagnostic fails, as it did for original Ninja in this run.
Continue with the separate controls after inspecting that diagnostic:

```sh
python tools/ninjars_controls.py prepare
python tools/ninjars_controls.py variants --cpu 1
python tools/ninjars_controls.py summarize
```

The baseline revision is `27147b3300f556a75661a886e92bdbe00558ee25`.
The model/library provenance and pinned toolchain are recorded in
`provenance.json`; compilation inputs, executable checksums, and generated
driver/source manifests are retained with each case. These candidate controls
were generated before per-generation source snapshots were introduced.
Their provenance explicitly marks that identity as unrecorded and the
generated-source snapshot as observed after the build; it does not assign a
later checkout revision retroactively. Numerical records remain bound to the
recorded executable checksums. The bridge's independent
256/384-bit convergence and complex-mass tensor inputs are retained in
`vendor/ninjars/tests/data/convergence.json`, with upstream attribution in
`vendor/ninjars/tests/data/provenance.json` and the result in
`data/bridge-convergence.json`.
