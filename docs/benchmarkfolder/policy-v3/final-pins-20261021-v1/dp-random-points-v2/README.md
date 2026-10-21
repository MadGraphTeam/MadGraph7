# Independent random DP measurement points

This namespace freezes a new Python-only phase-space sample for the eight core
processes, two loop-induced processes and two challenging 2→4 processes. It does
not replace the original fixed correctness streams or certify numerical results.
The configuration is frozen before generation and keeps the five release
identifiers, actual CLI generation authorities, physical cards, retained external
masses, source snapshots and a distinct seed for every process.

Each case has 100 accepted points, with seeds 20261021–20261032 in the recorded
case order. For multiplicity at least two, MadGraph's existing RAMBO generates
proposals at 1 TeV. Ordinary SM final particles must all satisfy pT > 50 GeV and
every pair must satisfy ΔR > 0.5, using pseudorapidity and wrapped azimuth. The
loop-induced reference sample retains ΔR > 0.5 and omits the pT cut. Rejected
proposals are counted once: pT failure first, then ΔR failure.

For 2→1, the invariant energy is the final mass (Z: 91.188 GeV, W: 80.419 GeV,
H: 125 GeV). A common boost has three independent components uniform on
[-0.25, 0.25]. These configurations are descriptive views of the constrained
on-shell kinematics, with no cuts. The configuration's 1 TeV energy field applies
only to RAMBO multiplicities; the physical metadata records the actual invariant
s for every serialized point.

The generator adapts the existing `ninjars_benchmark.make_points` construction
and reuses `RAMBO`, `FortranList` and `ninjars_supplement_v2.point_metadata`.
It never invokes a generated program, compiler, reduction backend or Rust
context. Its random generator is private; the ambient Python RNG is preserved.

After review of the frozen configuration, the commands are:

```sh
source .ninjars-build/dev-env.sh
python3 -B tools/ninjars_dp_random_points.py generate docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/dp-random-points-v2/configuration.json
python3 -B tools/ninjars_dp_random_points.py verify docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/dp-random-points-v2/manifest.json
```

The resulting point-authority manifest binds each point file and its metadata.
Validation replays the frozen seeds and cuts to require identical serialized
momenta and exact proposal/rejection accounting, then independently recomputes
on-shell and conservation residuals. The two reduction tools must consume the
same bound point files. Their numerical evaluation requires its own qualification
and is outside point generation.

This successor also rechecks each definition, alias, mass and card against its
retained core/control export or sealed stress authority chain, and binds the full
local import closure used by generation and validation. The initial v1 held
configuration was reviewed before any point generation; it is preserved as
historical preparation evidence and is not the measurement authority.

Generation and deterministic physical replay completed successfully on CPU 180:
**1,200 accepted points from 1,524 proposals, with 324 rejected proposals**. Each
case has 100 distinct serialized configurations. The largest normalized
conservation residual is 1.643e-13; the largest mass-shell residual is 4.349e-13.
These are checks of momenta, not numerical amplitude stability measurements.

The active point authority is `manifest.json`, SHA-256
`b0164afb38d02ff1135155e676c09ffa3528fa27c45df1a48632018ae136768c`.
`provenance.json`, `execution-record.json` and `logs/` retain the exact generation
and replay commands, Python environment, proposal accounting, and the 19 passing
software tests. `software-test-source.py` captures that tested checkpoint.
`inventory.json` lists every other retained file with its byte size and SHA-256.
The inventory includes only text, momenta, metadata, logs and source snapshots;
there are no native binaries or compiled outputs.

The separate `../dp-random-points-v1/` directory holds seven preparation files
(229,184 bytes): its initial configuration, five source snapshots and README.
It contains no generated point files and makes no numerical claim. Its held
configuration and source snapshots were preserved unchanged after review exposed
two validator closure gaps, which the active v2 implementation fixes.
