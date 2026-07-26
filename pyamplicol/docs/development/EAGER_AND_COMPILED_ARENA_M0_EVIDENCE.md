# Eager/Compiled Arena Milestone-0 Evidence

`tools/developer/eager_compiled_arena_m0.py` is the fail-closed evidence
orchestrator for the eager/compiled Direct-Arena migration. It runs no
generation or timing itself. It accepts only already-captured, content-addressed
raw evidence and always writes a content-addressed `accepted` or `rejected`
decision.

The orchestrator cannot promote one layout capture, a generation-fixed
diagnostic, a report cache row, or a hand-entered timing number. In particular,
the existing adaptive `docs/result_tables.py measure-cell` AmpliCol result has
only one measurement subprocess and is not sufficient for this gate.

## Required evidence

One request combines exactly six distinct JSON files:

1. built-in SM, topology replay, schema-6 Z+6g capture;
2. built-in SM, all-flow union, schema-6 Z+6g capture;
3. equivalent UFO-SM, topology replay, schema-6 Z+6g capture;
4. equivalent UFO-SM, all-flow union, schema-6 Z+6g capture;
5. fresh AmpliCol selected-flow/helicity-sum raw evidence;
6. fresh AmpliCol all-flow/single-helicity raw evidence.

Every pyAmpliCol capture is independently revalidated from its raw profiles,
profile schedule, worker records, validation values, and configuration. The
recomputed validation summary, layout acceptance, and deliberately incomplete
per-layout M0 record must exactly equal the stored values. The orchestrator
requires:

- the exact `u u~ > Z g g g g g g` standard candle;
- compiled, eager, and recurrence lanes;
- batches 1, 128, and 1024;
- an O3 process-generation request and at least seven interleaved worker
  subprocesses per cell;
- at least seven native wall blocks per worker;
- warmed, unprofiled `runtime_core_repeated_wall_time` evidence;
- complete physical flow and helicity axes;
- no generation-specialized axis;
- runtime-selected stable flow and source-helicity IDs;
- selected-total versus resolved-sum closure;
- pairwise component-level parity across compiled, eager, and recurrence,
  including identical resolved flow/helicity axes;
- compiled runtime no greater than 1.15 times recurrence for both LC layouts
  at batches 128 and 1024, independently for built-in SM and UFO-SM;
- matching source, runtime, host, sample, momenta, normalization, physical-axis,
  selector, logical reduction, and applicable execution-schedule identities.

Generation-model identities follow an exact role matrix. Built-in compiled
captures use `built-in-sm-source` and execute the requested O3 process-local
stages. Built-in eager and recurrence captures use the packaged
`built-in-sm-jit-o2` prepared model and must agree exactly. Every UFO-SM lane
uses the same explicit portable prepared-model file identity. Compiled mode
still materializes its process-local stages at the requested O3 level when
that explicit bundle supplies the model; eager and recurrence execute the
bundle's immutable portable O2 applications. The process-generation request
and semantic generation signature remain O3 for every lane. A different
supported identity kind remains invalid even if configuration, generation
signatures, both layout captures, and the request pin are all rewritten
consistently.

The topology-replay workload selects one physical flow at runtime and sums
helicities. The all-flow-union workload selects one source-ordered helicity at
runtime and sums all physical flows. Generation-fixed artifacts are only
diagnostic lower bounds and are rejected here.

Built-in and UFO captures must agree pointwise at `rtol=1e-12`,
`atol=1e-15` for each layout. The two layouts are different workloads, so their
values are not cross-compared with each other.

## Compiled-versus-recurrence runtime ceiling

The 15% ceiling is an acceptance condition, not a report-only comparison. For
each built-in/UFO model, each `topology-replay`/`all-flow-union` layout, and
each required batch 128/1024, the combiner requires:

```text
compiled subprocess median seconds/point
    <= 1.15 * recurrence subprocess median seconds/point
```

The boundary is inclusive. Batch 1 remains required raw evidence but is not a
cell in this throughput ceiling.

The combiner does not trust stored aggregate timing fields. It first
recomputes each lane's median and raw MAD from the retained positive
per-subprocess seconds-per-point samples, checks the retained sample count,
median, MAD, and native timing boundary, and only then computes the ratio.
The accepted decision retains all eight model/layout/batch results under
`validation.compiled_recurrence_runtime_ceiling`, including both recomputed
medians, raw MADs, sample counts, the ratio, and the policy limit. The complete
raw samples remain under `timings`.

If any one ratio exceeds 1.15, the whole six-input decision is rejected with
exit code 2. As with every other rejection, `validation`, `timings`, and
`comparisons` are all null, so no partial headline comparison can be mistaken
for accepted evidence.

## Request schema

The request kind is
`pyamplicol-eager-compiled-arena-m0-request`, schema 1. Unknown keys are
rejected. Every evidence reference has exactly `path`, `size_bytes`, and
`sha256`; relative paths resolve against the request file. The request itself
is pinned separately on the command line.

```json
{
  "kind": "pyamplicol-eager-compiled-arena-m0-request",
  "schema_version": 1,
  "captures": {
    "built-in-sm": {
      "topology-replay": {
        "path": "captures/builtin-topology.json",
        "size_bytes": 0,
        "sha256": "<64 lowercase hex>"
      },
      "all-flow-union": {
        "path": "captures/builtin-union.json",
        "size_bytes": 0,
        "sha256": "<64 lowercase hex>"
      }
    },
    "ufo-sm": {
      "topology-replay": {
        "path": "captures/ufo-topology.json",
        "size_bytes": 0,
        "sha256": "<64 lowercase hex>"
      },
      "all-flow-union": {
        "path": "captures/ufo-union.json",
        "size_bytes": 0,
        "sha256": "<64 lowercase hex>"
      }
    }
  },
  "amplicol_evidence": {
    "selected-flow-helicity-sum": {
      "path": "amplicol/selected.json",
      "size_bytes": 0,
      "sha256": "<64 lowercase hex>"
    },
    "all-flow-single-helicity": {
      "path": "amplicol/all-flow.json",
      "size_bytes": 0,
      "sha256": "<64 lowercase hex>"
    }
  },
  "expected": {
    "pyamplicol_source_revision": "<40 lowercase hex>",
    "amplicol_source_revision": "<40 lowercase hex>",
    "process": "u u~ > z g g g g g g",
    "runtime_provenance_sha256": "<64 lowercase hex>",
    "host_sha256": "<64 lowercase hex>",
    "momenta_points_sha256": "<64 lowercase hex>",
    "normalization_sha256": "<64 lowercase hex>",
    "model_common_physics_identity_sha256": {
      "built-in-sm": "<64 lowercase hex>",
      "ufo-sm": "<64 lowercase hex>"
    },
    "generation_model_identities_sha256": {
      "built-in-sm": "<64 lowercase hex>",
      "ufo-sm": "<64 lowercase hex>"
    },
    "color_flow": {
      "id": "flow:2,4,5,6,7,8,9,1",
      "word": [2, 4, 5, 6, 7, 8, 9, 1]
    },
    "helicity": {
      "id": "h:-1,+1,-1,+1,-1,+1,-1,+1,-1",
      "values": [-1, 1, -1, 1, -1, 1, -1, 1, -1]
    },
    "external_leg_permutation": [0, 1, 3, 4, 5, 6, 7, 8, 2]
  }
}
```

`external_leg_permutation` is the original-AmpliCol source-to-generated row
mapping, not the pyAmpliCol artifact's identity ordering. The pinned generator
moves the source-order `Z` leg from index 2 to generated-row index 8, giving
exactly `[0, 1, 3, 4, 5, 6, 7, 8, 2]`. Request-template builders must retain
that mapping.

Runtime provenance is hashed after removing location-only `path`,
`resolved_path`, `checkout`, and `working_directory` fields. All versions,
sizes, build inputs, dependency identities, and content hashes remain in that
semantic identity. The generation-model identity uses the same path stripping.
Host identity is the exact schema-6 host object.

## AmpliCol raw-evidence schema

Each AmpliCol file has kind
`pyamplicol-amplicol-m0-raw-evidence`, schema 1, and a canonical
`content_sha256` over the object without that field. Its exact root keys are:

```text
kind, schema_version, complete, evidence_scope, workload, source, host,
process, physical_axes, selector, momenta, normalization_sha256, timing,
validation, binary_evidence, content_sha256
```

`evidence_scope` must be `authoritative-host-capture-v1`. The source section
pins a clean revision; exact compiler `id`, `version`, `target`, and flags
digest; and a source-tree digest recomputed from the content-addressed source
file set.
`binary_evidence` content-addresses the executable, linked libraries, and
source files. `momenta.raw_file` content-addresses the common point file.

The selector records stable IDs, the physical flow word or source helicity
vector, complete physical axes, an empty `generation_specialized_axes`, the
summed axis, and the exact request-pinned nine-leg source-to-generated
permutation.

`validation.selected_totals` and `validation.resolved_sums` contain raw
`[real, imag]` pairs for every common point. Their point comparisons and
maxima are recomputed.

`timing.samples` contains at least seven independent subprocess records. The
two workload manifests share one `interleave_group_sha256`, which is
recomputed over their combined ordered schedule. Each record has a unique
combined `interleave_position`, paired `interleave_round`, command digest, UTC
start/finish, evaluated-point count, elapsed seconds, recomputable seconds per
point, uninterrupted status, and a content-addressed raw-output file. The
combined positions must be chronological and alternate selected then all-flow
for every round. The selected workload boundary is
`amplitude-evaluation`; the all-flow workload boundary is
`direct-library-total`. Both state
`batch_semantics=scalar-normalized-per-point`. The median and raw MAD are
recomputed rather than trusted.

Each sample command starts with the exact content-addressed executable and
contains the workload, round, raw momenta path, clean source revision, and
stable selector ID. Its `raw_output_file` is strict
`pyamplicol-amplicol-m0-raw-sample` schema 1 with exactly:

```text
kind, schema_version, role, sample_index, command_sha256,
evaluated_point_count, elapsed_seconds, seconds_per_point, stdout,
stdout_sha256, content_sha256
```

The canonical content digest excludes only `content_sha256`. `stdout` is
itself strict JSON of kind `amplicol-m0-probe-result`, schema 1, containing the
same role/index/count/timing scalars plus raw selected totals and resolved
sums. The orchestrator parses that stdout and binds it to the manifest values;
an arbitrary log file or wrapper-provided timing scalar is rejected. Measured
elapsed time must also fit inside the recorded subprocess start/finish
envelope.

## Capture command templates

Run every substantial capture under the 30 GiB watchdog. Use distinct output
roots so no artifact or result is silently reused across model/layout roles.
The selected flow and helicity below are placeholders for the stable IDs pinned
in the request.

```sh
PYTHON=.venv/bin/python
WATCHDOG="tools/ci/memory_watchdog.py"
HARNESS="tools/developer/recurrence_z6g_benchmark.py"

$PYTHON "$WATCHDOG" --limit-gib 30 -- \
  $PYTHON "$HARNESS" \
  --jit-optimization-level 3 \
  --lc-flow-layout topology-replay \
  --color-flow '<stable-flow-id>' \
  --helicity '<stable-helicity-id>' \
  --output-root /private/tmp/arena-m0/builtin-topology \
  --result-json /private/tmp/arena-m0/builtin-topology.json

$PYTHON "$WATCHDOG" --limit-gib 30 -- \
  $PYTHON "$HARNESS" \
  --jit-optimization-level 3 \
  --lc-flow-layout all-flow-union \
  --color-flow '<stable-flow-id>' \
  --helicity '<stable-helicity-id>' \
  --output-root /private/tmp/arena-m0/builtin-union \
  --result-json /private/tmp/arena-m0/builtin-union.json

# Repeat both commands for UFO-SM, adding its exact prepared-model file.
$PYTHON "$WATCHDOG" --limit-gib 30 -- \
  $PYTHON "$HARNESS" \
  --prepared-model /absolute/pinned/ufo-sm-prepared-model.pacbin \
  --jit-optimization-level 3 \
  --lc-flow-layout topology-replay \
  --color-flow '<stable-flow-id>' \
  --helicity '<stable-helicity-id>' \
  --output-root /private/tmp/arena-m0/ufo-topology \
  --result-json /private/tmp/arena-m0/ufo-topology.json
```

The fourth command changes the UFO layout/output paths to
`all-flow-union`/`ufo-union`. Do not add
`--specialize-flow-at-generation`.

Create `request-template.json` with the final four capture references and
`expected` object. Its `amplicol_evidence` object must reserve exactly the
`selected-flow-helicity-sum` and `all-flow-single-helicity` keys; their values
may remain empty objects until the capture below finishes. Then run the tracked
strict producer against the clean detached original-AmpliCol checkout:

```sh
REQUEST_TEMPLATE=/private/tmp/arena-m0/request-template.json
AMPLICOL_SOURCE=/private/tmp/pyamplicol-eager-compiled-arena-amplicol-m0-src
AMPLICOL_OUTPUT=/private/tmp/arena-m0/amplicol
AMPLICOL_CAPTURE=tools/developer/amplicol_z6g_m0_capture.py

$PYTHON "$WATCHDOG" --limit-gib 30 -- \
  $PYTHON "$AMPLICOL_CAPTURE" capture \
  --request-template "$REQUEST_TEMPLATE" \
  --repository "$AMPLICOL_SOURCE" \
  --output-directory "$AMPLICOL_OUTPUT" \
  --jobs 4 \
  --target-seconds 5 \
  --warmup-points 100 \
  --minimum-points 100 \
  --maximum-points 100000
```

The producer refuses a nonempty output directory. Before building anything it
revalidates all four schema-6 captures with the M0 validator and requires their
host, momenta, axes, normalization, selectors, source, and runtime contracts to
agree. It also requires the current host, the exact clean pyAmpliCol producer
revision, and the exact clean contributor-lock AmpliCol revision to match those
pins. Cleanliness is checked before the original-AmpliCol build with
`git status --porcelain=v1 --untracked-files=all`; tracked and untracked
preexisting state are both rejected.

The producer builds and snapshots `amplicol_library_benchmark`,
`amplicol_color_probe`, the generated `libamp` library, and every regular file
in the generated `Library/` tree while preserving its runtime-relative path.
After the build, one real selected-flow probe authenticates the executable's
emitted group, integral, PDG multiset, and serialized color order. The producer
adopts that emitted row, recomputes its source permutation and colored physical
flow, and requires both to match the request rather than trusting the
pre-generation process-file serialization.

A content-addressed launcher invokes the exact Python interpreter with
`-I -S -B` and an immutable runtime copy of the producer, its complete imported
`legacy_amplicol` / `legacy_oracle` helper-module set, and the contributor lock.
It never imports those helpers from the live checkout or creates untracked
bytecode beside them. The retained command for every sample additionally pins
the raw momenta file and digest, source revision, stable flow or helicity ID,
explicit flow word and helicity values, authenticated executable row,
external-leg permutation, workload, round, and evaluated-point count.

`capture-index.json` contains the two final evidence references. The output
also retains both authoritative manifests, exactly fourteen raw-sample JSON
records, the launcher, both probe executables, the generated library, and the
generated process file. The two workloads always run as seven chronological
selected-then-all-flow subprocess pairs. Each worker's only stdout is the
strict `amplicol-m0-probe-result`; the coordinator stores it verbatim and
revalidates its identity and numerical values. Wrapping an old report row or
copying its scalar timing into seven rows is therefore impossible.

Merge the content-addressed references into the final request without changing
the template's four capture references or `expected` pins:

```sh
REQUEST=/private/tmp/arena-m0/request.json
jq --slurpfile amplicol "$AMPLICOL_OUTPUT/capture-index.json" \
  '.amplicol_evidence = $amplicol[0].amplicol_evidence' \
  "$REQUEST_TEMPLATE" > "$REQUEST"
```

After recording exact file sizes and SHA-256 values in the request, run:

```sh
DECISION=/private/tmp/arena-m0/decision.json
REQUEST_SHA256="$(shasum -a 256 "$REQUEST" | awk '{print $1}')"

.venv/bin/python tools/developer/eager_compiled_arena_m0.py \
  --request "$REQUEST" \
  --request-sha256 "$REQUEST_SHA256" \
  --output "$DECISION"
```

Exit code 0 means the emitted decision is accepted. Exit code 2 means rejected;
the decision still has a canonical `content_sha256` and a non-empty `errors`
list. An accepted decision retains every raw median/MAD and reports explicit
pyAmpliCol/AmpliCol ratios for each model, lane, and batch, with both timing
boundaries attached. It also records the eight mandatory compiled/recurrence
ceiling cells described above. A rejected decision never contains partial
headline comparisons.

## Frozen accepted-base inputs

The immutable local baseline root is
`/private/tmp/pyamplicol-eager-compiled-arena-baseline`. Its runtime wheel was
built offline from the clean detached source checkout
`/private/tmp/pyamplicol-eager-compiled-arena-base-src` after
`443f354a467cdda187996bef1a41fbd5a00ae28d` became both the recurrence
source-freeze and `origin/main`. The accepted production-source parent is
`585456ed1726c43eef3ce35c7a126c17730e8a0d`; the child adds only acceptance
and audit ledgers. The wheel build ran under the 30 GiB watchdog (peak RSS
1.937 GiB) using only a copy-on-write clone of the existing offline dependency
and Cargo caches.

- wheel:
  `pyamplicol-0.1.0.dev0+candidate.c0fd7ce438fb-cp311-abi3-macosx_11_0_arm64.whl`;
- wheel SHA-256:
  `e0fb076738201d11fc0aa4c73c5b55e5d9440efa18a22a2f4c0c16f0de9655f6`;
- native build-input SHA-256:
  `f91ebcc3eb431e3e1e72ac8a4e02dea194c17c2118f57a2742d0c8c5a73b3088`;
- package version: `0.1.0.dev0+candidate.c0fd7ce438fb`;
- native ABI version: `1`;
- prepared SymJIT version/revision: `2.21.1` /
  `48197f32536c894b51ef25b2cf05ddd05c22675f`.

The root also freezes the accepted recurrence source-freeze artifact directory.
Its principal raw evidence hashes are:

| model/layout | result SHA-256 | artifact manifest SHA-256 |
|---|---|---|
| built-in/topology replay | `8d191a87f51ae7b78911aca6c866e3697f81fa5ad3426d34112ddc33015d8c0d` | `93c8a3154d3110e161e10951bd17d41e92a8bea6afbe3f692a6a6ce07ca61497` |
| built-in/all-flow union, explicit nonzero helicity | `38b6dba38501e6e5494d78ee0ec8f2910a49fbf9219229fcf3fd4cf84f55cd98` | `0440ad3b345f716a5a08d40be6725d9cc09468f99b6b5b4ec3d2d0b975aa30ed` |
| UFO/topology replay | `2dd0f4902044d5d89a0f92bdad8dfd5c8a47658d8467d9dedf7ab8fc9b1ce6e1` | `be6167ab6b6287b988f80531a5271e1962469f73e3ccc93cdc379a4d2bfd5a30` |
| UFO/all-flow union | `788810015306f01755eed59aff1812e3b15811a680c84dddea44bcd9f5077662` | `cf7547a59fceaba01622d4a226bae761c7180e0f4b3c164af4c6205c07646a87` |

These accepted recurrence files remain reference evidence only: their stored
benchmark source revision is the earlier runtime checkpoint and their
generation used JIT O2. They cannot satisfy the stricter M0 O3 six-input gate.
Fresh complete-artifact O3 captures from the frozen wheel, plus fresh
content-addressed AmpliCol captures at pinned revision
`79c96cecf2a722e50c3d2030b6894d755f96518a`, remain mandatory.
