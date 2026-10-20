# Installer publication transaction

The previous installer protected the main destination replacement, then
published `prefix/lib` aliases and deleted the stage outside that rollback
guard. A failed alias replacement or common-directory creation could report
installation failure while leaving the new destination active and prior
aliases partly changed. A stage-cleanup error could report failure after the
validated installation had already been published.

The bounded fix publishes the validated destination and installer-owned aliases
inside one recoverable transaction. It restores only aliases actually changed,
including custom and dangling symlink targets verbatim. It preserves regular
user files, directories and hidden temporary sentinels. Alias candidates and
backups live in a unique private directory on the aliases' filesystem, so both
publication and rollback work when the common directory is a symlink or mount.
New relative links use that directory's actual location. A failed rollback
attempts the remaining restorations and identifies retained recovery locations.
Publication cancellation also restores prior state before re-raising.

After the publication commits, cleanup failure reports a warning with the
retained path and leaves the validated installation successful. Cleanup of the
alias stage and main build stage is attempted independently. The installer
never recursively removes an existing common library directory.

Seven new MG7 regression tests cover publication failures and successful retries,
first installation, existing custom/broken aliases, user files, a symlinked
common directory, cleanup failure, failed rollback and interrupted publication.
Compiler and smoke execution are mocked; the staging, copying, renaming,
preservation and restoration use real temporary files.

The original installer is retained verbatim. The collector injects that
original module only into the test module and runs three new regressions to
record the old failures without reverting any live production source. It then
runs the final focused installer/statistics/boundary/upstream suite. It invokes
no compiler or reduction engine. Run only after root releases the quiet window:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/experiments/ninjars-installer-transaction-v1/collect.py \
  --cpu 3 --quiet-released 'ROOT_HANDOFF_IDENTIFIER'
```

The collector uses exclusive-create evidence files, records exact source hashes,
and retains the red and green logs plus source snapshots. Its outputs and the
independent review must pass before this milestone is sealed or committed.
Actual CLI installation and relocated numerical exports remain a separate
final-source gate; this engine-free proof does not replace them.
