# Early raw stress collection, with the core link pending

This operational wrapper permits the approved stress raw streams to be
collected after their actual CLI/build cohort is complete and before the final
core dataset is available. It invents no core identity and does not label an
assembled stress dataset or its final gates as passed.

Implementation does not authorize execution. CPU190 currently belongs to the
root's separately scheduled work. Each numerical launch requires a later
explicit serialized MP/CPU handoff bound to that variant's exact spec.

The source cohort is the completed result at
`docs/benchmarkfolder/policy-v2/final-stress-generation-v1/early-result.json`.
That result binds actual generation HEAD, d05 numerical-source equivalence,
the published archive and full reducer runtime, fixed physical cards, source
proofs and point streams. The wrapper checks those actual bytes again before
and after collection. The private installation/publication proof supplies the
exact archive and source hashes; they are never inferred from short labels.

The ordinary, rotation and scale variants retain the approved 20, 4 and 8
points, all four backends, two command traversals and CPU190. They call the
unchanged `workers.run_case` with its existing calibration, initialization and
stability behavior. The fixed rotation and scale functions, `input_roots`,
aliases, `scale_adapter`, `checked_series` and `bind_collected` are reused.
The wrapper retains itself as the actual input adapter, together with all
helper sources. Its input spec explicitly records actual HEAD and whether this
docs-only wrapper is tracked, and binds its actual source bytes even before
the next documentation commit.

The actual CLI generation revision stays bound permanently to its existing
export records. Before collection, root may commit the tested documentation
milestone and record a later orchestration HEAD whose generator/vendor bytes
still match the numerical freeze and export generator identity. The raw spec
captures that actual orchestration HEAD separately and requires it to remain
unchanged throughout the declared run; it never relabels the CLI export as the
later commit. No extra exception is added for a changed `.gitignore`.

Operational subprocess ownership adds bounded six-hour build/copy waits and
process-group cleanup. Worker initialization/commands retain the unchanged
24-hour protocol bound and existing worker cleanup; PID/start-tick/PGID/affinity
events are recorded immediately when each worker process is assigned. These
hooks change no worker inputs or numerical math. Every failed attempt remains
in its own namespace, with raw failure records and diagnostics preserved.

Once the early build is complete, the following writes an ordinary spec only;
it does not launch compilation or an amplitude engine:

```sh
python -B docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/collect.py write-spec \
  --early-result docs/benchmarkfolder/policy-v2/final-stress-generation-v1/early-result.json \
  --variant ordinary \
  --report docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/runs/ordinary-v1 \
  --work .benchmark-work/policy-v2/final-stress-raw-collection-v1/ordinary-v1 \
  --output docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/ordinary-spec.json
```

Use corresponding distinct `rotation-v1` and `scale-v1` namespaces and variant
arguments for the remaining two streams. No output or spec may be overwritten.

Root's later handoff file must contain these fields, with the actual spec hash
and requested variant:

```json
{
  "root_authorized": true,
  "phase": "early-stress-raw-collection",
  "cpu": 190,
  "variant": "ordinary",
  "specification_sha256": "ACTUAL_SPEC_SHA256"
}
```

Only after that explicit handoff:

```sh
python -B docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/collect.py collect \
  docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/ordinary-spec.json \
  --execute-numerical \
  --handoff docs/benchmarkfolder/policy-v2/final-stress-raw-collection-v1/ordinary-root-handoff.json
```

The handoff example is an argument contract, not a preauthorized launch or a
substitute for the missing core identity. No root handoff or completed early
result is fabricated by this package.

When the actual core dataset exists, reuse `early_build.attach_core` to create
a **new** real-core preparation manifest. Point the normal stress assembly spec
at that preparation and the already collected raw report directories. The
unchanged `stress.assemble` and `report_snapshot` then recompute every required
raw/source/runtime/card/point/control check through `validate_preparation` and
`validate_group`. Keep the original raw specs, protocols, data and bindings
unchanged; no expensive numerical replay is required merely to attach the real
core link. This path adds no performance acceptance gate and changes none of
the core or loop-induced gates.
