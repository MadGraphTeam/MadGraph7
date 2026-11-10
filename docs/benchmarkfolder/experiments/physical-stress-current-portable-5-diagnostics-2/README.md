# Closed stress console diagnostics

This metadata successor preserves companion1 unchanged at index `a9d9320ade61da7710ac0810739299984147f3d0e0fda2035a20e4d9f7817c94`. The original-four ancestor summary did not serialize a `complete_runtime_map` field, so those four fields are now unknown (`null`) with an explicit scope annotation. Absence does not imply a map failure: the ancestor binder is still independently replayed. The20 continuation fields retain their explicit true/false values, including19 complete maps and the unqualified EW COLLIER request. No native evidence or numerical policy changed. The predecessor is metadata lineage, not an additional raw-input dependency.

This pure companion authenticates the closed continuation capsule `a520199c…`, its original-four ancestor `6fc20356…`, and preparation `49cb1ec2…` before deriving diagnostics from all24 retained consoles. It copies no console logs and launches no compiler, loader, worker, or amplitude. Original evidence is unchanged.

`qualification-summary.json` separates runtime-reported `IEEE_*` flag notes from hardware register capture, which was not performed. Every selected W03, IEEE, signal, reduction-exception or library-error/warning line retains its exact one-based line number and text. Reported flags can be sticky across calls; this evidence does not attribute them to individual amplitudes. An absent signal diagnostic does not establish that hardware exceptions never occurred.

The QCD four-particle TIRust request timed out in startup before READY, with no completed warmup or final rows and no RUN command journal. Its complete runtime map establishes process identity, while internal in-progress amplitude calls were not instrumented. The frozen worker/source quotes identify the protocol timeout and180-second phase limit without inventing a stack trace. The EW three-particle COLLIER request has observed warmup calls despite lacking a complete runtime map; it remains unqualified. Numerical failures and comparison outcomes are inherited unchanged.

Run from the repository root after `source tools/tirust_env.sh`:

```bash
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/physical-stress-current-portable-5-diagnostics-2 -p test_diagnostics.py
python3 -B docs/benchmarkfolder/experiments/physical-stress-current-portable-5-diagnostics-2/verify_evidence.py --seal
python3 -B docs/benchmarkfolder/experiments/physical-stress-current-portable-5-diagnostics-2/verify_evidence.py
```

Sealing is append-only. The reader API is `verify(root=None, *, read_bytes=None, derive_summary=False)`. A supplied callback must expose `.paths`, the exact physical payload inventory plus each index for the companion and prefixes `base/`, `ancestor/`, and `preparation/`. Dependency directories must be the fixed siblings named in the index. No workspace fallback is permitted. The return value includes `verified`, `index_sha256`, `base_index_sha256`, and the derived `summary`.
