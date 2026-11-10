This append-only package binds the eight offline analysis files without
changing the previously frozen source or execution indices. Its verifier
checks their hashes, retained input/source bindings and parent inventories,
then recomputes the saved exact determinants, differences, decomposition
sums and complex residuals. It invokes no compiler, reducer or executable.

From the repository root:

```sh
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/execution-1/geometry-audit/verify.py
```

This verification works from retained repository evidence. Full regeneration
with `audit.py` or `rank-one.py` also requires the hash-bound ignored source
snapshots used for the source-formula audit. Those source identities are
listed separately as `working_source_bindings_hash_only` in the index;
their current availability is not silently substituted for historical
provenance. The current vendor bridge is likewise a recorded source identity,
not a requirement that future production source stay unchanged. `rounding.py`
uses the retained arithmetic source and captures. In-workspace reproduction
was checked before freezing this package; a clean checkout can always run
the portable integrity/algebra verifier above.

The scalar-key-only subset and exact common-translation subset remain
distinct in the rank-one appendix. All 37 threshold refusals are excluded
from tensor residuals. None of these checks establishes checked-QP acceptance,
complete threshold amplitudes, physical accuracy or performance.
