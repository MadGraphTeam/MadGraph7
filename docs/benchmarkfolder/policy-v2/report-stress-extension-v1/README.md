# Report extension for the additional two-to-four stress cases

This is a new source/test checkpoint. It preserves the previously pushed
`final-report-pipeline-v1` software evidence and its original 36-test snapshot;
that old source verifier must not be relabelled as checking this extension.
At preparation time no new test, Typst compilation or numerical evaluation has
run. Root must release the benchmark quiet window before validation.

The report's optional CLI argument `--stress FILE` reads the separate raw-bound
`madloop-final-stress-v1` dataset. Although the argument is optional for rendering
an intermediate report, the final campaign remains pending until the mandatory
additional stress cases pass:

- `g g > t t~ g g [virt=QCD]` with `loop_sm`;
- `u u~ > e+ e- mu+ mu- [virt=QED]` with the checksum-bound Gmu EW model.

Each case has twenty ordinary points, four fixed rotations and eight ordered
scale-sequence calls, using all four candidate backends and two complete commands
per stream. The independent reader compares every physical split order and
Born/Laurent component, checks rotation invariance and scale restoration, and
retains failure and state histories. Changed-scale amplitudes are allowed to
change. Optional corrected-original references remain separately identified,
including unavailable or failed attempts.

`stress.report_snapshot(path, expected, inputs, runtime_signature=descriptor)`
must match the exact core dataset identity, literal original baseline, candidate
generator and complete measured NinjaRS reducer-runtime descriptor. The report
rejects different identities and contradictory completion flags. The stress and
transitive supplement reader source hashes join the rendering manifest.

The PDF section displays the ordinary/rotation/invariance/scale/restoration
comparisons, optional reference status, diagnostic warmed time, initialization,
actual final DP/QP counts, observed discrepancies and raw MadLoop estimates.
All estimates and status/control histograms remain in the stress snapshot.
These two new cases add **no performance acceptance gate**. Their diagnostic
times cannot relax or replace any existing eight-core or ggH/ggHH 1.05 gate.

The source changes add four report tests (forty report cases in total):
mandatory missing/failed/unresolved stress gating; exact descriptor forwarding
without imposing a new timing threshold; identity/flag rejection; and actual
stress/transitive-reader hashing. The existing explicit synthetic PDF fixture
also exercises both new process tables, optional unavailable references and
diagnostic/stability columns. These are software/layout fixtures, not physical
benchmark observations.

After root's compute handoff, run the focused report and stress suites, retain
their logs in this new namespace, obtain an independent review/rerun, and seal
the evidence. Keep the historical top-level `benchmark.typ`/`benchmark.pdf`
unchanged until the complete final dataset is validated and visually inspected.
