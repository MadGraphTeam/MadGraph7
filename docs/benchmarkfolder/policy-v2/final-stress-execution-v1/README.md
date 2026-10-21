# Parameterized final stress execution

This source-only template does not launch anything. The operational scripts
have been checked with Python AST parsing without importing project modules.
They have not run CLI generation, builds, reductions or tests.
The tested stress reader/API remains frozen separately in
`../stress-reader-validation-v1`. All actual work requires the final committed
source/archive and the root's build or serialized engine handoff.

`write_specs.py prepare` takes explicit final revision, installed archive SHA256,
all four final core cohort identities, the complete measured NinjaRS runtime
descriptor, actual-CLI installation proof and fixed physical cards. It checks
these against the current clean generator/main installation before writing a
preparation spec. It never calls `bin/madgraph`, make, a worker or an engine.
The numerical setup is the approved twenty-point/two-process plan, unchanged.

## Early CLI/build phase without a future core identity

The final numerical revision is
`d05b59f3277ce0d727cb2816d8b7ad45107e58c4`, while the actual generation HEAD
also contains later documentation/evidence commits. The controller proves exact
tracked generator/vendor source equality, records allowed documentation/test
differences separately and rejects untracked numerical input. The pre-task
`vendor/StdHEP/fail` installer diagnostic is explicitly inventoried separately.
It checks the main publication, all ten installed artifacts, exact archived
bridge source hashes and the full reducer descriptor again on each built output.
Every CLI/make command owns a new process group with a six-hour timeout and
TERM/KILL/reaping on failure. Only the existing CLI/build/masses/point functions
run; no loop amplitude or engine is initialized.

After root commits these scripts and holds HEAD fixed, use the reserved CPU6,7
for the existing loop-induced candidate exports and then the two stress builds:

```sh
taskset -c 6,7 python tools/ninjars_supplement_v2.py generate-candidate
python docs/benchmarkfolder/policy-v2/final-stress-execution-v1/write_specs.py early \
  --generation-revision "$FROZEN_HEAD" \
  --numerical-revision d05b59f3277ce0d727cb2816d8b7ad45107e58c4 \
  --archive-sha256 800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42 \
  --runtime-signature "$RUNTIME_SIGNATURE_JSON" \
  --publication docs/benchmarkfolder/policy-v2/main-install-publication-v1/transition.json \
  --loop-sm-card "$FIXED_LOOP_SM_PARAM_CARD" --ew-card "$FIXED_EW_PARAM_CARD" \
  --report "$STRESS_REPORT_ROOT" --work "$STRESS_WORK_ROOT"
taskset -c 6,7 python docs/benchmarkfolder/policy-v2/final-stress-execution-v1/early_build.py \
  docs/benchmarkfolder/policy-v2/final-stress-execution-v1/early-spec.json
```

This writes immutable source/runtime registration and exact first point bytes
without asserting any core dataset identity. It retains real generation/build
records and derived controls. Once the actual core dataset is available, use
the `prepare` writer command below with `--commit "$FROZEN_HEAD"`,
`--numerical-revision d05b59f3277ce0d727cb2816d8b7ad45107e58c4` and
`--early-result "$STRESS_REPORT_ROOT/early-result.json"`. That operation creates
a new real-core-linked preparation manifest and validates it with the existing
stress reader; it never changes or regenerates the early assets. Skip the
separate `tools/ninjars_stress_v2.py prepare` command when reusing early assets.
No early measurement or alternate numerical validation layer is introduced.

Use full paths within this repository. The expected-identities JSON has exactly:

```json
{
  "core_dataset_identity_sha256": "FINAL_CORE_DATASET_IDENTITY",
  "literal_baseline_revision": "27147b3300f556a75661a886e92bdbe00558ee25",
  "candidate_generator_identity_sha256": "FINAL_GENERATOR_TREE_IDENTITY",
  "candidate_runtime_identity_sha256": "FINAL_REDUCER_DESCRIPTOR_IDENTITY"
}
```

The separate runtime-signature JSON is the complete self-sealed
`policy.reducer_signature(candidate8_target)` descriptor from the same final
core timing bindings, not a manually supplied archive label.

After the root releases builds, materialize the spec and run actual generation:

```sh
python docs/benchmarkfolder/policy-v2/final-stress-execution-v1/write_specs.py prepare \
  --commit "$FINAL_COMMIT" --archive-sha256 "$FINAL_ARCHIVE_SHA256" \
  --expected "$EXPECTED_IDENTITIES_JSON" --runtime-signature "$RUNTIME_SIGNATURE_JSON" \
  --installation "$FINAL_INSTALLATION_JSON" \
  --loop-sm-card "$FIXED_LOOP_SM_PARAM_CARD" --ew-card "$FIXED_EW_PARAM_CARD" \
  --report "$STRESS_REPORT_ROOT" --work "$STRESS_WORK_ROOT"
taskset -c "$BUILD_CPUS" python tools/ninjars_stress_v2.py prepare \
  docs/benchmarkfolder/policy-v2/final-stress-execution-v1/prepare-spec.json
python docs/benchmarkfolder/policy-v2/final-stress-execution-v1/write_specs.py bind \
  --cpu "$STRESS_EVALUATION_CPU"
```

The preparation command invokes the real CLI, clean builds and `--masses` only,
then generates and binds first ordinary/derived point streams before reductions.
It checks all prescribed source/runtime/card/kinematic bindings. Its six-hour
stage limits and complete failure records remain active. A generation failure
is not replaced with a different process or seed.

After the root releases the single numerical-engine slot, run ordinary,
rotation and scale collection separately. Each command prepares all four
candidate backends from private generated-output clones and retains full worker
initialization, calibration and the two required command traversals:

```sh
python tools/ninjars_stress_v2.py collect \
  docs/benchmarkfolder/policy-v2/final-stress-execution-v1/collect-ordinary.json --execute-numerical
python tools/ninjars_stress_v2.py collect \
  docs/benchmarkfolder/policy-v2/final-stress-execution-v1/collect-rotation.json --execute-numerical
python tools/ninjars_stress_v2.py collect \
  docs/benchmarkfolder/policy-v2/final-stress-execution-v1/collect-scale.json --execute-numerical
```

The scale collection is the actual compiled adapter smoke and full control.
It first builds the unchanged frozen worker, then rebuilds only its private
check driver with the three reviewed MU_R assignments and fixed schedule
reader. Both Fortran sources, their diff, schedule, compiler log and final
executable/source manifests are retained. The first actual initialization and
calibration exercise all eight scheduled inputs before the two complete command
traversals. The reader checks that every other generated numerical source is
unchanged. A Python transformation test alone is not counted as this compiled
smoke. No separate input, weakened threshold or omitted contribution is used.

The optional-helper template remains visibly unbound until actual untouched
same-process CLI exports, exact two-file MODEL correction and private rebuilds
have been reviewed. Complete that template in a new immutable spec, run its
ordinary-only collection separately, and add its report entry to a new assembly
spec if practical. Retain any failure or unavailable proof explicitly; helpers
cannot replace missing mandatory candidate targets.

Finally, `assemble assemble-spec.json` writes a new immutable raw-bound dataset;
`validate validate-spec.json` recomputes all correctness/control conclusions.
Pass that dataset to the final renderer's `--stress` argument. Both processes,
their real runtime/initialization costs, raw estimates, observed discrepancies
and actual final DP/QP usage remain separately attributed. These new cases add
no performance acceptance gate and leave every existing core/loop-induced gate
unchanged.
