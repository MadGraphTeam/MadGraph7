# Corrected-model validation policy v2

The collection protocol remains the frozen `persistent-worker-v1`. This directory
separately records `madloop-helper-correction-policy-v2`: original Ninja remains
the literal timing denominator; corrected candidate backends and three original
backends with only the verified model corrections form the correctness cohort.
Original numerical disagreements remain ordinary failed diagnostics.

`cases.json` preserves the eight processes, physical settings and seeds from the
committed core specification. It applies `MLStabThres=1e-9` and
`COLLIERRequiredAccuracy=1e-12` to every final role. `controls.json` uses the same
Collier setting and retains the strict forced-rescue control.

The tools do not edit v1 runners, measurements, inputs or candidate outputs.
All paths below are relative to the MadGraph repository; run commands there
after sourcing `.ninjars-build/dev-env.sh`.

## Exact correction evidence

`correction-spec.json` points to actual original and final CLI exports and their
command cards/logs. `model-correction.json` retains content-addressed copies of
the exporter/parser snapshots, model preimages/postimages, generation/build
identities, physical cards, UFO provenance and analytical helper tests.

The original generated manifests, generation records and cards are anchored to
commit `80fca669eb29ca507239e8a655c82c19d55818ee`, before the helper repair. This
anchor retains the original revision
`27147b3300f556a75661a886e92bdbe00558ee25`; it does not relabel a modified source
as original. Corrected generation must have a recorded clean source identity
whose exporter/parser hashes match its Git revision, and a matching manifest
recorded before and verified after compilation.

Fresh original/candidate CLI exports may legitimately differ outside MODEL.
Their complete manifests and those additional changes are recorded. Their
entire `Source/MODEL` trees must differ only by the two independently bound
postimages. For QCD, `mp_intparam_definition.inc` is unchanged but still checked;
for EW, both model files change. Constructing a corrected-original reference
has the stronger check: its *entire generated source tree* may differ from the
anchored original only in the specified model postimages and frozen driver.

```sh
python tools/ninjars_benchmark_policy_v2.py correction \
  --spec docs/benchmarkfolder/policy-v2/correction-spec.json \
  --output docs/benchmarkfolder/policy-v2/model-correction.json
```

This command refuses an existing manifest. The retained correction is independent
of later removal of disposable fresh CLI outputs. Creating a new correction or
binding a compiled target additionally verifies the live original tree.

## Corrected original references

The collector clones original outputs and never modifies their source/cards.
It removes all MODEL objects/modules, model archive, subprocess objects and
executable. MODEL compilation runs serially because the original makefile does
not declare all Fortran module dependencies; subprocess compilation uses the
unchanged flags and original reduction/DHELAS libraries. Rebuild commands,
logs, object/archive/executable hashes and complete manifests are retained.

```sh
python tools/ninjars_benchmark_references_v2.py \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --config docs/benchmarkfolder/policy-v2/cases.json \
  --cases all --run-id corrected-original-final --cpu 207
```

Each target runs initialization/warmup, calibration and two retained full-point
commands through the unchanged worker. Its timing is diagnostic only.
Representative runs add `--fresh-parity --representative`, use the retained
control configuration and may select `--backends 1,6` for a QP control.
`--points-root` explicitly selects an existing immutable point-stream directory.
Core reference runs always require all three original backend types.

## Explicit input roots for timing and representative collection

The same tool accepts `--timing-spec path.json`. An example final timing spec is:

```json
{
  "run_id": "v2-core-final",
  "configuration": "docs/benchmarkfolder/policy-v2/cases.json",
  "report": "docs/benchmarkfolder/policy-v2/timing-runs/v2-core-final",
  "work": ".benchmark-work/policy-v2/timing-runs/v2-core-final",
  "output_roots": {
    "baseline": ".benchmark-work/outputs/baseline",
    "candidate": ".benchmark-work/policy-v2/generation/outputs/candidate"
  },
  "points_root": "docs/benchmarkfolder/points",
  "cases": ["qcd-1", "ew-1", "qcd-2", "ew-2", "qcd-3", "ew-3", "qcd-4", "ew-4"],
  "cpu": 207,
  "minimum_seconds": 1.0,
  "timeout": 7200.0
}
```

The adapter changes only the outer source lookup. Frozen `prepare` creates
private output copies; its nested configuration/build lookup resolves those
copies normally. The adapter/spec are hash-bound in the run. The frozen
`run_case` retains the seven targets, adjacent original-Ninja/NinjaRS pair,
alternating order and all initialization/timer/status behavior.

Representative specs set `"representative": true`, `"fresh_parity": true`
and may explicitly choose `sources` and `backends`. They run two retained
commands. Representative coverage must include all three source roles, DP
backend types, actual legacy and own QP outcomes, loop-induced amplitudes and
multiple split orders. Loop-induced classification additionally requires zero
Born columns. The proof binds the final policy/collector implementation and
final candidate generator identity; historical v1 proof is insufficient.

## Bind, validate and publish

`policy.json` is the superseded pre-proof policy retained with the initial
implementation commit. `policy-offline.json` retains the subsequent in-memory
proof-validation revision. Neither has a completed dataset. `policy-runtime.json`
adds strict reduction-runtime binding to the refreshed proof. Validation still
needs no disposable work directory. The collection adapter and numerical data
are unchanged.

After implementation review/freeze, seal the policy:

```sh
python tools/ninjars_benchmark_policy_v2.py policy \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --config docs/benchmarkfolder/policy-v2/cases.json \
  --output docs/benchmarkfolder/policy-v2/policy-runtime.json
```

The timing adapter writes `binding-spec.json` with the report path, role map and
actual private output paths. Seal its live source/runtime bindings:

```sh
python tools/ninjars_benchmark_policy_v2.py bind \
  --spec path/to/binding-spec.json \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --output path/to/role-bindings.json
```

Corrected-reference collection does this automatically. Bindings retain complete
source manifests, exact frozen worker/fresh-driver identities, linked library
hashes and MODEL rebuild evidence. Offline validation uses retained evidence;
it does not require disposable executables or object files. Live binding is
strict and compares those artifacts before recording them.

Before using a binding in the current proof or dataset, derive its runtime
binding outside the immutable run directory:

```sh
python tools/ninjars_benchmark_policy_v2.py runtime-bind \
  --bindings path/to/role-bindings.json \
  --correction docs/benchmarkfolder/policy-v2/model-correction.json \
  --output path/outside/run/runtime-role-bindings.json
```

This checks the existing executable, source and library hashes against the live
worker, preserves its original binding, and retains external Fortran modules and
NinjaRS native-link metadata. It does not reevaluate physics or rewrite measured
data. Every role/backend must use one identical runtime signature throughout
the proof and final timing/reference series. The signature includes all linked
reduction/scalar libraries, modules, native-link metadata and numerical linker
definitions; only process-specific MODEL and DHELAS archives are excluded.
Changing a Rust archive or scalar library requires refreshed representative
evidence even if the generator tree is unchanged.

```sh
python tools/ninjars_benchmark_policy_v2.py proof \
  --policy docs/benchmarkfolder/policy-v2/policy-runtime.json \
  --bindings representative-runtime-bindings-1.json representative-runtime-bindings-2.json \
  --output docs/benchmarkfolder/policy-v2/representative-proof.json

python tools/ninjars_benchmark_policy_v2.py publish \
  --policy docs/benchmarkfolder/policy-v2/policy-runtime.json \
  --proof docs/benchmarkfolder/policy-v2/representative-proof.json \
  --bindings timing-runtime-bindings.json corrected-reference-runtime-bindings.json \
  --output docs/benchmarkfolder/policy-v2/datasets
```

Publication requires complete contiguous seven-target timing and three-target
reference streams. It recomputes all 21 corrected-cohort pairs at every point,
order and coefficient, plus all original diagnostic comparisons. It preserves
failed numerical/performance gates in an immutable dataset rather than dropping
points. A `410`, missing point/order, incomplete command or changed input remains
a structural error. Performance always uses original backend6 and candidate8;
corrected-original times cannot enter the denominator. A complete case subset
is explicitly distinguished from completion of all eight cases.

Python API for reporting: `build_dataset(policy, proof, timing_bindings,
reference_bindings)` validates/recomputes without publishing;
`validate_dataset(dataset_json)` independently reproduces the recorded dataset
from retained raw evidence and rejects any changed summary.

Render a published dataset into a new directory:

```sh
python tools/ninjars_report_v2.py path/to/dataset.json --compile
```

The renderer defaults to a `report` directory beside the dataset. It refuses an
existing destination, preserving every historical snapshot and PDF. It validates
the complete retained evidence, then separately recomputes all numerical
comparisons, mismatch records and runtime intervals before writing
`report-data.json`, `benchmark.typ`, `style.typ` and `benchmark.pdf`.
`--timestamp` fixes the displayed snapshot date for reproducible rendering.
The final top-level benchmark report is replaced only after the complete
required measurements are qualified. No v2 acceptance data is implied by the
layout fixture used during development.

## Checks and development attempts

Run focused tests with:

```sh
python -m unittest tests.unit_tests.various.test_ninjars_benchmark_policy_v2
```

The suite exercises postimage/non-model corruption, forged original provenance,
wrong generator binding, source-role exchange, stale rebuilds, escaped writes,
missing points/commands, unsuccessful statuses, unpaired timing, all corrected
pairs, visible original discrepancies, and offline retained-data validation.

`reference-runs/reference-smoke-qcd1-20261020` retains an initial failed clean
MODEL build: the original makefile attempted module consumers before the module
producer under `make -j4`. The collector now uses `make -j1` for MODEL. This
failure is historical development evidence and cannot satisfy any v2 gate.
