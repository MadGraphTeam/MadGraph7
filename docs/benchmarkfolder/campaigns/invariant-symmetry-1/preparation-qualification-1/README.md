# Completed current-source preparation

All five preparation stages passed on 10 November 2026, from
08:24:57.479524 to 08:44:28.113233 UTC: **1170.633709 seconds (19m30.6s)**.
There were **zero amplitude evaluations**. These are preparation wall times;
they do not qualify numerical agreement, initialization performance, or the
per-process runtime acceptance criterion.

| Stage | Wall seconds | Outcome |
|---|---:|---|
| Prove and reuse baseline objects | 136.489779 | passed, exit 0 |
| Relink baseline workers | 150.833180 | passed, exit 0 |
| Generate all 11 candidates through the MG CLI | 502.344030 | passed, exit 0 |
| Assert all 22 declared physical parameter cards | 2.783992 | passed, internal assertion |
| Build all 11 candidates from fresh source objects | 378.139980 | passed, exit 0 |

The four child stages reported no surviving owned processes. Every stage used a
3600-second inclusive bound. Compiler affinity was CPUs 12–15, nonexclusive,
with four jobs. The total includes verification between stages. Reported
`compilation.json` make durations sum to 37.8457603934221 seconds for baseline
worker relinking and 285.93329836800694 seconds for candidate builds. Inherited
baseline `generation.json` durations describe the earlier CLI exports; those
exports were not repeated in this preparation.

## Retained evidence

`evidence.json.gz` retains 316 exact UTF-8 path records deduplicated into 241
text contents. `inventory.json` lists each original repository-relative path,
byte count and SHA-256; `results.json` is the derived preparation summary.
The bundle includes:

- All 41 closed `preparation-1` files: the unchanged source packet, exact stage
  specification and commands, status/events, console logs, and every stage's
  source/configuration snapshots before and after execution.
- All 22 immutable report physical parameter cards and their CLI-generation and
  compilation records; all 11 actual candidate MG command files; all compiler
  logs and baseline byte-identical object-relink proofs, including predecessor
  compilation records and the proof helper source.
- Frozen point streams, model/source inventories, numerical-source snapshots,
  shared policy, and the campaign manifest and registration-time pending summary.
  That summary is an archival input to this preparation, not a statement of the
  concurrently running numerical campaign's current status.

Native executables, libraries, objects and generated-source trees are represented
only by their recorded hashes where applicable. The immutable compiler evidence
binds their original source/link inventories and commands; this packet does not
rebuild them or revalidate their current runtime. No active work directory,
worker output, or mutable `MadLoopParams.dat` was read for this retention.
Absolute paths embedded in records remain historical text and are never opened
by the portable verifier. Model pins are retained as exact inventories; no copied
model binaries or pickles are included.

The original preparation source index remains
`0757f9f829fbc4d472823cee226bb54a0a62efcbf70fbab59c137beeb030aa87`.
The canonical manifest is
`ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`.
The registered numerical source was committed at `2c13246ffa`; actual generation
observed revision `584dae7875d20e881a4f9dc27bdb67f5e0fa5512`, with the same registered
generator-file identity
`a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19`.
These distinct revision observations are preserved. Baseline source remains
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`.

## Offline verification

Copy this complete directory anywhere and run its verifier with Python 3.9+:

```sh
python3 -B /path/to/preparation-qualification-1/verify.py
```

From this repository's pinned environment:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-qualification-1/verify.py
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-qualification-1/test_verify.py
```

The verifier checks the outer inventory, every retained byte digest, original
source seal and explicit campaign ancestry, five complete successful stages,
unchanged source/configuration snapshots, all 22 card bindings before and after
compilation, fresh candidate build policy, baseline object proof identities,
compiler-policy equality, reconstructed worker hashes and derived summary.
It uses the standard library only and does not import benchmark tools, invoke
commands, or load a native library. Tampering checks exercise semantic rejection
with updated container hashes, separately from ordinary checksum rejection.

The completed preparation was invoked from the repository root with:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py --execute
```

This command is historical: the original controller is append-only and rejects a
second attempt in the completed directory. The exact effective child commands
are in the retained specification/status. Full-amplitude numerical qualification
and subsequent timing remain separate work; no prior-source measurements are
relabeled here.
