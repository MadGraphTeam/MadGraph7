# Closed preparation and failed raw-DP run3

Preparation completed successfully in root session 48253 (exit 0): four private
driver builds, four exact-interpreter dependency queries, no surviving owned
processes and zero amplitude evaluations. The four new deterministic streams
contain 100 physical points each at 1 TeV with strict pT > 50 GeV and
pairwise delta R > 0.5. Saved metadata binds the compiled masses and physical
parameter-card identities. Group layouts and complete preparation receipts
are retained without executable binaries.

The separate raw-DP run then closed in root session 87730 (exit 1). All 20
requests failed during card readback before numerical launch: the parser
rejected the canonical terminal footer `/* End of param file */`. Every one
of the 6,000 scheduled primary probes remains unexecuted. The secondary
read-only guard errors reflect a missing declaration of intended card writes;
they do not establish a numerical failure. The 20 exact written cards and
complete failure records are retained for the source4 correction.

This capsule contains an explicit allowlist of closed receipts, command
stdout/maps and frozen points. Capture did not walk the work root or copy
binaries or amplitude output. `retention.json` lists every original path and
its before/after hash. Recorded host paths are provenance strings: the pure
verifier never follows them.

```
python3 -B verify.py
```

The verifier uses only this directory. It checks the closed preparation and
run identities, all command receipts and cleanup, physical point constraints,
cuts, mass/card bindings and all 20 unexecuted request populations. These are
setup and sampling results; no raw stability curve or precise timing is
qualified by this capsule.
