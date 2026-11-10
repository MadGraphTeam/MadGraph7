# Optional closed shared-host appendix

This support consumes an explicitly selected, sealed evidence package. It does
not discover running attempts, execute benchmarks, edit a campaign selection,
or change a performance gate. The canonical report is rendered only after the
closed package and its index hash have been reviewed and selected.

Add an optional member to a new immutable closed-report selection:

```json
"closed_provisional": {
  "index": "campaigns/split-dp-terminal-1/experiments/provisional-timing/CLOSED-PACKAGE/index.json",
  "sha256": "EXACT-REVIEWED-INDEX-SHA256"
}
```

The index schema is `madloop-tirust-closed-provisional-report-v1`, with:

- `closed: true`, `performance_acceptance_eligible: false`, and the exact
  `campaign_binding` from the registered report.
- `files`: every package-relative stored file, with its `sha256` and `bytes`.
  The index itself is pinned by the selection and is not self-indexed.
- `paths`: `specification`, `closure`, and `manifest` logical paths. The selected
  closure may be the administrative closure after a partially published summary;
  both original records remain retained by the package.
- `comparisons`: exactly the 66 schedule IDs. Each entry has `completion`
  (a logical path, or null for unstarted/uncompleted budget rows). Complete entries
  also have `result` and `ready: {baseline: path, candidate: path}`.
- `replay: {path: "FROZEN-WRAPPER.py", function: "replay_completed"}`. This path
  must be a physically stored, indexed file, even in compact packages.

For compact storage, `retained_files` maps every original logical file to its
exact raw `sha256` and `bytes`. `files` then binds the actual compressed payload
and wrapper. The frozen wrapper exports:

```python
read_retained_json(path)  # verifies raw bytes against retained_files, then parses
replay_completed()       # {comparison_id: exact verify_comparison-reparsed result}
```

Both default to the wrapper's own package location and are portable after
relocation. Importing the wrapper must have no execution side effects. Its
accessors and replay use only retained text and frozen sources, with no current
workspace, compiler, numerical library, or native process dependency. The wrapper
must preserve every failed/partial raw record, validate the complete schedule and
closed statuses, including exact preregistered ordering and the original default
chains from retained generation/preparation, and invoke the unchanged frozen
numerical parser. The collector
binds all physical package bytes before importing the wrapper. Plain packages may
omit `retained_files`; their logical paths must be physically indexed files.
For the reviewed compact wrapper, the collector can call its frozen
`_load(expected_index_sha256=...)` once and parse retained bytes from that verified
inventory. This avoids decompressing the entire payload for every JSON access.
The selected wrapper remains unchanged, and its `replay_completed()` still does
an independent load and invokes the unchanged portable parser.

Only rows whose selected closure, completion, and result all establish a complete
numerical pass receive displayed measurements. Result files left by a later
timeout or failed verification never become timing samples. Every other schedule
row remains visible with its specific unavailable, numerical-failure, or budget
status and empty timing fields.

The collector additionally checks three paired repetitions, all 100 points,
initialization against READY records, durations against traversal counts,
accepted-usage totals, and the frozen campaign source/numerical identities.
Completed comparison elapsed time must fit its original 120/180-second cap;
administrative closure elapsed is retained separately and may disclose an overrun.
The appendix shows descriptive ratios, initialization, all batch durations,
ratio ranges, and accepted backend/precision counts. Internal attempts and IEEE
flags remain unobserved. Standalone backend 9 uses baseline CutTools as comparator.
These observations belong to the registered older candidate, not later symmetry
or other experimental changes. No CI or runtime acceptance verdict is produced;
all final performance gates stay deferred to Hua-Sheng's local machine.

Synthetic support checks, without reading the active campaign:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  docs/benchmarkfolder/test_campaign_report_provisional.py \
  docs/benchmarkfolder/test_campaign_report_refresh.py
```

The Typst test compiles synthetic data into a temporary directory. It does not
write the canonical PDF, selection, report context, or ledger.
