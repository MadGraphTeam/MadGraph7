This source-only helper prepares retention of the single provisional timing attempt supervised by session `82678`. No active attempt is read by importing it or running its synthetic tests. Sealing is a separate, append-once action requiring root confirmation that session 82678 has ended and authorization to retain its selected terminal closure. It performs no native, loader, compiler, timing or current-library checks.

After root authorization only, create a terminal-confirmation JSON with `terminal_session_id: 82678`, `terminal: true`, `seal_authorized: true`, `baseline_generation_metadata_authorized: true`, the exact absolute `source_directory`, and the exact selected `closure_sha256`. An existing `administrative-closure.json` takes precedence over `summary.json`; partial original summary bytes remain retained. Then run from the repository root:

```bash
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/provisional-retention-2/retain.py seal --source EXACT_CLOSED_REPORT_DIRECTORY --terminal-confirmation EXACT_CONFIRMATION_JSON
```

The fixed output is a new `sealed-1` directory beside this helper. An existing destination is never overwritten or retried automatically. The helper verifies all 66 schedule rows and portable completed-result replay before publication, then checks the source snapshot again. Any publication failure remains visible in the destination.

The explicit input allowlist includes root specification, campaign manifest, cases, preparation, numerical-prerequisite record, run start, progress, terminal closure and controller failure; frozen protocol source names; point `.dat`/`.json` files; and comparison `.json`, `.dat`, `.txt`, `.log`, and partial `.tmp` files. Symlinks, nonregular files, unknown paths, ELF signatures and non-UTF-8/binary bytes are rejected. No ignored build tree or native library is traversed. The sole external metadata allowlist is the eleven exact `preparation.clones[case].baseline.private_output/generation.json` paths approved by root. Each must match its frozen `generation_sha256` before it is retained under `prepared-generation/baseline/<case>.json`; missing or changed metadata fails sealing. This proves every scheduled baseline default chain even when initialization timed out before a comparison could copy generation metadata. No other live output paths are read.

Every original file is bound by logical path, SHA-256 and byte count in `retained_files`. UTF-8 contents are deduplicated by SHA-256 in one deterministic gzip JSON payload. The payload filename is its compressed SHA-256. `files` binds the outer payload, frozen helper and terminal confirmation. Original JSON/raw bytes are preserved, including partial or failed files. `paths` names the specification, manifest and selected closure; `comparisons` binds all 66 completion/result/READY paths. `completion_retained` preserves an incomplete completion file even when it cannot substantiate the closure row.

Portable API, called only after the report pins `index.json` by exact SHA-256 and verifies `files`:

- `read_retained_json(path)` returns an exact retained JSON object after checking the payload and complete inventory.
- `replay_completed()` returns `{comparison_id: exact_reparsed_result}` for rows whose closure, completion and result all say complete and whose numerical check passes.

Replay extracts into a temporary repository and imports only the frozen tools. It calls the unchanged controller's `verify_comparison(run)` and independently checks READY/initialization binding, host-snapshot ordering, numerical prerequisite/runtime identities and source/build identity. It never activates a live campaign or rehashes installed native dependencies. Unavailable/deadline/numerical-failure/not-run/not-completed rows remain retained, with no returned timing ratio even if a result file exists. Original partial output is never retrospectively promoted.

Controller SHA-256 remains `7cccff659bdb180c070a62f9637fc4865892eaa739f681927c538bfae58e41c3`. No `GENERATOR_PATHS` or production code is changed. Three alternating full100 pairs are descriptive shared-host observations, never seven-pair confidence-bound performance acceptance. READY cold time includes initialization and warmup; accepted usage describes final amplitudes, not internal attempts or FPE flags.

Synthetic tests only:

```bash
python3 -B -m unittest discover -s docs/benchmarkfolder/campaigns/split-dp-terminal-1/provisional-retention-2 -p test_retain.py -v
```

V2 preserves the complete frozen v1 proposal and corrects only its runtime-to-manifest join. The full recorded library/archive/metadata closures remain joined to preparation, numerical prerequisites and after-observation records. TIRust DSO, archive, config and both link metadata files are required by exact recorded installation-prefix-relative paths. Other loaders with the same basename are independent entries. Cargo and header/source pins remain in the exact canonical manifest and its original guard-attested provenance; they are not claimed as runtime-observed files. closed-runtime-shape.json binds the already-closed EW n4 input used for the focused regression fixture. No actual provisional files were inspected while preparing this correction.
