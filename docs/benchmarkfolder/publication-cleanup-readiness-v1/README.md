# Bounded publication cleanup readiness

No cleanup, production edit, old-seal change, compiler or numerical engine was
performed. The snapshot was taken at
`2bf47cd477c6a8eaeef07a4f649434dfe38a4260`; root owns any later cleanup decision.
The exact nine candidate paths are in `cleanup-candidate-paths.txt`.

## Small explicit cleanup recommendation

The six uncompressed native-study materializations total **107,437,190 bytes**.
Each plaintext hash/size matches its sealed `compressed-artifacts.json` original,
and each corresponding tracked `.gz` hash/size matches the same index. The native
seal explicitly excludes only these plaintext duplicates while retaining the
compressed observations, analyses, index, materializer and all failed attempts.
The study owner confirms no native study job remains live. Root may remove these
six exact files after a fresh process-use check; reproduction restores them with
`archive_large.py materialize` before running the frozen analyzers.

The remaining three candidates are the completed broad-regression private
comparison scratch directories, totaling **64,736,017 bytes**:

- `.benchmark-work/final-regression-review-v1`
- `.benchmark-work/final-regression-fks-review-v1`
- `.benchmark-work/final-regression-fks-review-v2-absent`

They contain 2,445, 2,450 and 2,451 ordinary files respectively, no symlinks and no
tracked files. Their durable commands, source/fixture review and logs survive in
sealed `policy-v2/final-regressions-v1`. Its read-only verifier does not consume
these scratch trees. Root previously approved only these three exact regression
scratch roots for disposal after that milestone was accepted. No other ignored
build/source/cache subtree is covered by this recommendation.

The point-in-time process observation checked same-account process working
directories, open file descriptors and mapped paths for these nine candidates.
It observed **zero uses** among 78 processes. One process vanished and ten
permission errors prevented complete observation; therefore this is not an
unqualified claim that every process was inspected. Recheck relevant task-owned
jobs before deleting, and preserve a candidate if there is any uncertainty about
active use. No process arguments, signals or unrelated open files were retained.

## Publication versus disposal

At the snapshot there were 333 untracked paths outside existing ignore rules:

| Scope | Paths | Disposition |
|---|---:|---|
| Existing interrupted/fresh reference campaigns | 322 | Preserve as append-only evidence; owner selects final seal. |
| Reference pause/resume ledger | 2 | Preserve historical process/protocol provenance. |
| Native plaintext duplicates | 6 | Exact cleanup/ignore candidates described above. |
| `vendor/StdHEP/fail` | 1 | Preexisting workspace file; leave untouched. |
| New stress `early_build.py` | 1 | Active owner implementation; retain for review/publication. |
| This audit's collector | 1 | New durable audit source; retain with this package. |

No untracked ELF, archive, object, Fortran module, bytecode or cache path was
found in that bounded untracked list. The now-committed final private installation
proof and main-publication preparation are required evidence, not disposal
candidates. Their ignored source trees, new prepared archive, old installation
backups and alias journals must remain available through root's final provenance
checks. All original reference archives, live workers, caches required by pending
builds, and old sealed evidence are outside this cleanup list.

## Ignore correction

The eight portable experiment executable rules are already published. The six
native materializations are still visible as untracked files. Add only the six
anchored rules in `proposed-native-ignore-rules.txt` if retaining or regenerating
these working copies. These rules hide neither the gzip evidence nor any source,
manifest, failed attempt, controller log or report. Root decides whether to add
these rules and/or remove the exact materializations; this audit changes no
ignore file.

## Retained evidence

`result.json` preserves each candidate's byte counts, native original/archive
hashes, explicit file classification, related manifest identities and process-use
observation. `native-owner-confirmation.json` records the owner's completed-study
and reconstruction handoff. `verify.py` checks this package's seal and retained
compressed/durable inputs; removed disposable plaintext is allowed to stay
absent. It neither reruns a process scan nor requires scratch trees to survive:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/publication-cleanup-readiness-v1/verify.py
```

The snapshot is historical readiness evidence. Verification does not authorize
cleanup, prove later process idleness, or turn incomplete reference results into
accepted measurements.
