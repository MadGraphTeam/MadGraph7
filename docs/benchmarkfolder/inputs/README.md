# Retained benchmark inputs and source identity

Each `baseline/<case>` or `candidate/<case>` directory retains the parameter card,
per-backend MadLoop cards, generation and compilation metadata, and available source
manifests. `compiled-source-manifests` and `compilation-history` preserve previous
content-addressed records when an output is rebuilt.

New generation records capture the source revision before export, the tracked
generator patch, hashes of tracked and untracked generator inputs, and the bytes
of untracked inputs. The scope is `madgraph`, `aloha`, `Template`, `bin`, `models`
and `input`; report files and benchmark results are excluded. Source-identity
records and patches live in `source-identities`; untracked input blobs are stored
as `source-identities/untracked/<sha256>`. External dependency and UFO model
identities are recorded separately in `provenance.json`.

Generation provenance predating this instrumentation is not presented as if it
were captured at the time:

- Original-source outputs refer to the independently verified, unchanged archive
  at `27147b3300f556a75661a886e92bdbe00558ee25`.
- The eight regenerated core candidate cases have explicitly reconstructed
  identities: `d47f0bc4f3fd47f8bf6e62c50447cbd5300d9561` plus the retained compact
  `UPDATE_WL` loop-order patch. This reconstruction uses implementation chronology
  and checks the compact generated kernels where present.
- Older candidate controls without a reliable generation-time snapshot retain
  status `not-recorded` and a null revision.

`generated-sources-observed.json` is a later observation of existing source files,
not proof of their contents at the earlier compilation. For new builds,
`compiled-sources.json` hashes generated source, include and make files before
compilation and checks that they remain unchanged afterward. Individual file
symlink targets are hashed; external library directory trees are covered by
dependency provenance. Measurements record the generation identity, compiled
source digest when available, compilation metadata hash and actual executable
hash. An executable that differs from its build metadata is rejected.

The original EW UFO inventory and content hash can be checked without repinning:

```sh
python tools/ninjars_setup_references.py --verify-model-only
```

This verifies the imported model's top-level Python files using the retained
sorted-filename, NUL-separator and raw-byte digest.

Rejected and superseded measurement records are retained under `../rejected`,
outside accepted-data globs. Repetition validation occurs before acceptance and
checks inputs, executable, settings, dimensions, CPU and all numerical coefficients.
