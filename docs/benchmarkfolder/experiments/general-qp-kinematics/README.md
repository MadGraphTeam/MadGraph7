# General QP kinematics: private sources and standalone checks

This archive retains byte-for-byte copies of the private graph-identity exporter,
QP matrix-policy consumer, and completed standalone validation. It is evidence
for these fixtures; it does not qualify amplitudes, Born/loop coordinate
consistency, general CMS support, or runtime performance. Production sources
and installed reducers were not changed by these experiments. No binary,
object, module, graph pickle, or reducer library is included.

## Sources and scope

- `classification/` contains the complete 12-file frozen source package:
  exporter patch, generator, signed-incidence helper, generated QCD lookup and
  routing data, source tests and review. Source index:
  `97f5100a4f7a679e53a6508f669b54d4f71aa2a40868f90e435f4d932cced533`.
- `consumer/` contains the complete 40-file frozen source package: patch,
  helpers, exporter, exact composition templates, generated source, tests and
  review. Source index:
  `7d002155a9d43f0e172570ab06e7e2e1a2ec29019fe20dd72c5cf0928ba27048`.
  Explicit group/direction metadata reaches private paired-mass QP entries
  for backends 8 and 9. Policy 0 delegates to the original matrix builder;
  policy 1 uses supported external **real** MP model masses for classified
  single-leg or conservation-complement channels subject to residual/closure
  checks. Composite, unknown, off-shell and inadmissible cases retain computed
  invariants. Internal CMS pole masses remain separate. Graph shell intent is
  not a numerical shell or effective-coordinate certificate. Policy 1 remains
  separate from `UseQPIntegrand` and Born-coordinate ownership.
- `integer-validation/` retains all eight indexed source payloads and completed
  execution/logs. `matrix-validation/` retains all 16 indexed source payloads,
  the failed first compile, and corrected completed execution/logs. Original
  indices and protected-input bindings are unchanged.
- `width-correction/run.py` is the exact separately authorized controller that
  added `-ffixed-line-length-132` and distinct output names to the matrix job.

## Recorded outcomes

The integer driver passed **8,325** cases with `-fcheck=all`: actual QCD pairs
and signed witnesses in CT4/5, CT1/2 aliases, reversed pairs, actual N=2 groups,
invalid inputs, and an N=1 serialization boundary. It used integer/logical
operations and integer/character I/O. It did not test physical tadpoles or
floating-point matrix policy.

The first matrix compile failed because the retained `MadLoopParams.inc`
declarations exceeded default fixed-form width 72. No matrix executable ran.
Its log and `execution.json` are retained unchanged. The authorized command-only
width correction passed **72 real-model plus 72 CMS-model cases** using the
same sources and `-O0 -g -fcheck=all -fbacktrace -ffixed-line-length-132`.
All 16 source payloads and 187 protected inputs were unchanged. The CPU-13 job
took 0.945 seconds within its combined 120-second bound.

Each matrix case checks policy-0 whole-matrix bits against a separate call to
the original builder. The suite includes all 19 retained composite pairs in
each direction (38 per model), nonzero exact observer-word oracles, independent
rational invariants, genuine QP external mass variation, distinct CMS internal
poles, invalid metadata/masses, off-shell and closure guards, and soft/extreme
common-shift cases. The driver verifies unchanged IEEE halting modes. Fixed
model values test matrix policy and transport; they are not physical model-card
or amplitude results. No amplitude, reducer DSO, license service or model
initialization library ran.

`matrix-validation/execution-2.json` SHA256:
`8af01513ca71609d87e72de3763fa387b192bb1e9ea7a4a18a10339535a817a6`.
`integer-validation/execution.json` SHA256:
`91f3ab8084c679e99f8fe14913d4d3edf538aa4b3ca11a10ac01f67bed992e1c`.
Execution records retain exact compiler identities, argument lists,
environments, log hashes and binary hashes; binary bytes are excluded.

## Integrity and portability

`inventory.json` binds every file except itself, records each original path,
and retains the union of protected-input hashes. Every copied file matched its
original before packaging. Original frozen source indices remain valid inside
their archive directories. Earlier source-only README, review and protocol
records are historical snapshots; completed execution records and the outcomes
above describe the later standalone checks.

The generated Fortran, declarations, numeric/integer cases and expected
outcomes needed for standalone checks are included. Historical scripts retain
original workspace paths, overwrite guards and authorization bounds; packaging
did not rerun or rewrite them. Reproducing generator tests additionally needs
the pinned repository/model inputs and routing evidence identified by the
bindings. This is not a standalone MadGraph distribution. Standalone drivers
can be compiled from archived sources with equivalent compiler options without
importing a graph pickle or loading a reducer.

Validate archive bytes without preparation, compiler or native execution,
from the repository root:

```sh
python -B - <<'PY'
from pathlib import Path
import hashlib, json
p = Path('docs/benchmarkfolder/experiments/general-qp-kinematics')
i = json.loads((p / 'inventory.json').read_text())
actual = {f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file()}
assert actual == set(i['files']) | {'inventory.json'}
for name, entry in i['files'].items():
    b = (p / name).read_bytes()
    assert len(b) == entry['bytes']
    assert hashlib.sha256(b).hexdigest() == entry['sha256'], name
print('archive bytes verified:', len(i['files']))
PY
```
