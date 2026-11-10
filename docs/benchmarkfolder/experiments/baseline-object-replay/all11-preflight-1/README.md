# Eleven baseline private-copy relink proofs

All eleven frozen baseline exports reproduced their recorded executables byte
for byte from untouched numerical objects and the original benchmark driver.
The proofs ran serially on allowed CPU12 with one job. They executed eleven
link commands, no Fortran source compilation, no amplitudes and no reducer
evaluations. They supply readiness evidence only and are excluded from every
performance comparison.

The original baseline revision remains
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. Each case first checks the retained
generation/compilation records, exact generated source inventory, numerical
object/archive hashes, executable hash, effective compiler policies and
link-time dependencies. It then copies the export into its own ignored tree
under `.benchmark-work/tirust/preflights/baseline-all11-1/`, preserving symlinks.
Copied source and object bytes must match; copied library paths are mapped back
to their original paths solely for hash equality checks.

The retained link helper invokes make only with `--dry-run`, suppressing all
existing object/library targets. It accepts exactly one link recipe, rejects
source compilation or compound shell commands, and changes only the executable
output path to a new private proof directory. It never executes the result.
The linked executable must match the historical recorded SHA256. Original
sources, objects, executable, physical/numerical cards, dependencies, runtime
library metadata and effective compiler policies are checked again afterward.
All eleven original before/after bindings are unchanged.

The relink environment uses `PYTHONHASHSEED=0`, but it does not repeat the old
source generation. The historical export seed remains explicitly unrecorded.
No candidate is registered, exported or relinked, and neither TIRust package is
modified. The earlier single-QCD-n1 proof remains preserved; this campaign used
fresh private-copy proofs for every case, including n1.

The generated binaries and copied build trees remain ignored. The committed
text evidence includes full before/after inventories, compiler/driver hashes
and policies, static/shared dependencies and metadata, mapped private inputs,
the exact link commands/logs, all per-case results and an aggregate hash index.
The verifier checks this frozen ledger without invoking make, a compiler or a
native binary:

```sh
python3 docs/benchmarkfolder/experiments/baseline-object-replay/all11-preflight-1/verify.py
```

The existing final-campaign `reuse_baseline` path will repeat its normal
untouched-driver proof and all current binding checks. There is intentionally
no cached-proof import shortcut. These preflight results catch changed inputs
early; they cannot justify skipping later verification if any byte or compiler
setting changes. Reproduction requires a fresh output namespace, because the
runner refuses to overwrite this one.
