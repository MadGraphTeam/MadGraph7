# Authorized nine-path cleanup

Root authorized only the nine exact paths from the sealed readiness audit.
They were removed after fresh checksum, tracked-file absence, durable-proof and
same-account process-use checks. No other file or directory was removed.

The cleanup reclaimed **172,173,207 bytes**: six native-study plaintext duplicates
(107,437,190 bytes) and three completed regression comparison scratch trees
(64,736,017 bytes). The six committed gzip records and their materializer remain.
All 195 native evidence files and all 27 broad-regression evidence files were
verified before and after cleanup; both manifests retained their exact hashes.
Earlier readiness records and seals were not changed.

Before deletion, every file checksum matched its retained original/archive record.
Each scratch tree matched the earlier file count/size, contained no symlink or
mount crossing, had no tracked member, and produced the same full file-map digest
in two consecutive passes. A fresh process observation found no candidate use
among 77 same-account processes' cwd/descriptors/mappings. Ten permission errors
limited that observation; this is explicitly not a claim of visibility into all
processes. An independent command-name check found no completed-regression worker
still live and no unreadable same-account command line. No process was signaled.

`cleanup-started.json` is the before-deletion journal; `cleanup-result.json` adds
all exact removed paths, bytes, hashes, process observations and durable-proof
checks. The helper is an exclusive one-shot operation, not a general cleanup tool.
It refuses a repeat when either record exists. It invokes no compiler or numerical
engine. `vendor/StdHEP/fail`, reference campaigns, original sources, private
validated installations, backups and sealed evidence were left untouched.

## Deferred exact ignore rules

The cleanup initially appended the approved six exact native materialization
rules. Root then requested deferral because the active stress generation guard
requires no non-documentation source changes. The exact diff survives as
`deferred-native-ignore.patch`. Only this helper's unique header and six lines
were removed again; `.gitignore` was restored byte-for-byte to its recorded
pre-cleanup hash. No other ignore lines or source-equivalence whitelist changed.

`ignore-deferral.json` records both intermediate and restored identities without
rewriting `cleanup-result.json`. Root may apply the small retained patch after
the generation round ends. The six materializations remain absent, so none is
currently available to be accidentally staged. Future frozen-native analysis
requires `archive_large.py materialize` first.

Verify this historical record without deleting or scanning processes:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/publication-cleanup-execution-v1/verify.py
```
