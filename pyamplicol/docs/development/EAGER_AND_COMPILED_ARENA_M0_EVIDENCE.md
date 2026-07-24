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

1. built-in SM, topology replay, schema-5 Z+6g capture;
2. built-in SM, all-flow union, schema-5 Z+6g capture;
3. equivalent UFO-SM, topology replay, schema-5 Z+6g capture;
4. equivalent UFO-SM, all-flow union, schema-5 Z+6g capture;
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
- JIT O3 and at least seven interleaved worker subprocesses per cell;
- at least seven native wall blocks per worker;
- warmed, unprofiled `runtime_core_repeated_wall_time` evidence;
- complete physical flow and helicity axes;
- no generation-specialized axis;
- runtime-selected stable flow and source-helicity IDs;
- selected-total versus resolved-sum closure;
- pairwise component-level parity across compiled, eager, and recurrence,
  including identical resolved flow/helicity axes;
- matching source, runtime, host, sample, momenta, normalization, physical-axis,
  selector, logical reduction, and applicable execution-schedule identities.

Generation-model identities follow an exact role matrix. Built-in compiled
captures use `built-in-sm-source`; built-in eager and recurrence captures use
the packaged `built-in-sm-jit-o2` prepared model and must agree exactly.
Every UFO-SM lane uses the same explicit prepared-model file identity. A
different supported identity kind remains invalid even if configuration,
generation signatures, both layout captures, and the request pin are all
rewritten consistently.

The topology-replay workload selects one physical flow at runtime and sums
helicities. The all-flow-union workload selects one source-ordered helicity at
runtime and sums all physical flows. Generation-fixed artifacts are only
diagnostic lower bounds and are rejected here.

Built-in and UFO captures must agree pointwise at `rtol=1e-12`,
`atol=1e-15` for each layout. The two layouts are different workloads, so their
values are not cross-compared with each other.

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
    "external_leg_permutation": [0, 1, 2, 3, 4, 5, 6, 7, 8]
  }
}
```

Runtime provenance is hashed after removing location-only `path`,
`resolved_path`, `checkout`, and `working_directory` fields. All versions,
sizes, build inputs, dependency identities, and content hashes remain in that
semantic identity. The generation-model identity uses the same path stripping.
Host identity is the exact schema-5 host object.

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

The AmpliCol capture producer must run the matching raw probe seven times per
workload in an interleaved subprocess schedule, on the same momenta file and
host. It must write the schema above directly from raw outputs. Wrapping an old
report row or copying its scalar timing into seven rows is invalid.

After recording exact file sizes and SHA-256 values in the request, run:

```sh
REQUEST=/private/tmp/arena-m0/request.json
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
boundaries attached. A rejected decision never contains partial headline
comparisons.
