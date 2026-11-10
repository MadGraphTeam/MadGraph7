# Production higher-rank DP terminal qualification

This folder qualifies the narrow extension from rank pair `(1,1)` to `(2,1)`
and `(3,1)`. It is separate from the frozen private kernel experiment in
`../dp-higher-rank/` and contains no performance acceptance measurements.

The final deterministic pair passes: all 800 coefficients are bit-identical,
and all 100 per-point statuses are `211` (CutTools, double precision). The new
control also matches the retained committed control. The main configuration,
generator sources, physical inputs, compiler policy and external native
libraries remain bound to the recorded unchanged identities.

The higher-rank routines preserve the public compact updater's traversal,
zero-B skip, complete output initialization and surviving arithmetic order.
Only the matrix entries used by the final trace are computed after the IEEE
guard accepts. Every other case calls the unchanged dense updater. The
existing private `(1,1)` routine remains byte-for-byte unchanged.

Selection remains limited to ordinary optimized standalone Born-interference
exports with four-component input, output and cut wavefunctions. Higher-rank
selection additionally requires vertex rank one and process rank at most six.
The completed-source proof checks the canonical full coefficient map, bounds
of all reachable entries, read-only table use, the complete consumer/alias
closure, and every reference to the private routines. Unknown formats and
unsupported exports retain dense calls. Storage must equal the canonical
coefficient count for the process rank; shared mixed-process storage can
conservatively decline the optimization.

`compiled/` records seven passing compiled tests, 4,608,000 IEEE/output
comparisons and 104 paired trap probes. These include the actual exporter
capture followed by CREATE/MERGE, four rounding modes, incoming IEEE flags,
active and inactive coefficient boundaries, and unchanged dense fallback.
The compiler corpus predates the final source-proof census hardening; the
generated arithmetic and compiled-test source hashes are identical. The
source-proof tests and adversarial mutations are recorded separately in
`proof/`.

The final fresh QCD `u u~ > z g g` pair uses `PYTHONHASHSEED=0` for both actual
CLI exports. A private control wrapper filters only terminal selection to
rank `(1,1)`; every other export setting and generator source is identical.
The control selects 26 updates, and the candidate selects 147: the original
26 plus 85 rank `(2,1)` and 36 rank `(3,1)` updates. None is declined.
The strict audit binds 193 identical native source/build files and exactly
two changed files: the new caller substitutions and two added helper bodies.
The existing private `(1,1)` and public compact bodies are identical.
Both builds use the same ordinary
100-point input, physical parameter card, explicit CutTools backend 1 and
`MLStabThres=1e-6`. Numerical qualification requires identical coefficient
bits and per-point statuses. The paired control is also compared with the
retained replay of the immutable committed `(1,1)`-only export. Exact source,
compiler, library, card and input
bindings are retained with the observations. The validation summary records
the final result; elapsed times in these records are not benchmark samples.

`initial-reference-census/` retains the first successful fresh export before
independent review found an uncertified private-call reference that the proof
accepted. It was neither compiled nor evaluated. Its work directory was moved
to the location in `status.json`; recorded original paths remain unchanged.
The final proof rejects this case, including standalone `(1,1)` exports.

`source-order-attempt/` preserves a second pair that passed all 800 coefficient
bit and status comparisons, but whose source audit refused two independent
temporary-assignment reorderings and two makefile target reorderings from
nondeterministic export ordering. Its exact differences remain recorded. The
final fixed-seed pair resolves this attribution limitation without normalizing
or editing any generated arithmetic.

`capture_cli.py` observes the actual CLI finalizer without changing its
decisions. All `set` commands use `--no_save`, and the main configuration is
hashed before and after generation. The original retained control source files
are never rebuilt or edited; its earlier replay used a separate copied build.

The focused source and existing polynomial/QP regression suite passed 114
tests and 220 subtests on unchanged production/test bytes. The post-build
source audit and completed consumer proofs are repeated after installation of
the benchmark driver. Native executables and objects are represented by hashes;
only text sources, logs and metadata are copied here.

Generation and later build/replay phases retain their actual helper snapshots.
An independently authorized benchmark-harness metadata change landed between
them: subsequent helpers record and validate the fixed export seed. The
generator source identity excludes benchmark tools and stayed unchanged.
Generation already enforced seed zero explicitly in the retained runner.
Its protocol snapshot contains the benchmark helper but does not claim a
contemporaneous copy of the campaign helper; later snapshots include both.
This metadata update did not require another physics replay.

From the repository root after sourcing `tools/tirust_env.sh`, the source and
numerical qualification phases are:

```sh
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank-production/run_qualification.py generate
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/dp-higher-rank-production/source-audit.py --control .tirust-build/dp-higher-rank-production/outputs/control/qcd_n3_uux_zgg --candidate .tirust-build/dp-higher-rank-production/outputs/candidate/qcd_n3_uux_zgg --output docs/benchmarkfolder/experiments/dp-higher-rank-production/final-source-audit.json
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 360s taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank-production/run_qualification.py build
PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 180s taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank-production/run_qualification.py observe
```

Reproduction needs a fresh working output while retaining earlier attempts.
`verify.py --live-sources` checks the portable evidence and source bindings
without numerical execution. After staging, add `--tracked` to check the Git
index bytes as well.
