# Upstream watch: report-only NinjaRS update

Official `git ls-remote` observations on 2026-10-21 found NinjaRS `main` at
`ca1009e89ecb03d3ff477f57b170241f7a9f8669`, while oneloopmaster remained at
`8ec256fc26cb361b20e10f59c19beb35717ef32c`. Both symbolic HEADs point to `main`.
Exact advertisements, commands, timestamps and hashes are in `evidence.json`.

The six commits after frozen NinjaRS `aa6a6307dd081dc3714f9accebc0df9e5561ae73`
change 65 report/benchmark files: eight under `docs`, 56 under `validation`, and
`TODO_NEXT_PLAN.md`. All 582 other tracked entries are identical, including the
entire five-crate workspace, dependency patches, Cargo manifests/lockfile and
toolchain files. The generic/native master providers and fast tensor engine
also match byte for byte. oneloopmaster, Symbolica and gammaloop pins are
unchanged. Independent numerical review reached the same conclusion.

This is an evidence/report refresh using the already frozen aa6/8ec/9cdc
implementation. It supplies no new numerical, acceptance, sign, threshold or
performance implementation. The revised report still discloses unflagged
binary64 cancellation lanes and oneloopmaster's adversarial/high-tracked-bit
tails. Its optional cancellation checks and rank n+2/component timings are
separate from the MadLoop bridge's own full-process gates.

The root therefore retained the actual aa6/b0d cohort, with no rebuild, restart
or relabelling of measured results for this report-only update. The observed
upstream commit and the tested commit are recorded separately. A future
production change requires its own pinned candidate and qualification.

From the repository root, run the small offline capsule verifier:

```sh
python3 -B docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/dependencies/upstream-watch-ca1009e-v1/verify.py
```

To repeat the source comparisons using an existing Git clone containing both
commits, append `--repo PATH`. This reads Git objects only; it never fetches,
checks out, builds, evaluates an engine or changes pins. No new binary identity,
scientific validation or performance measurement is claimed by this capsule.
The earlier dependency capsule remains unchanged.
