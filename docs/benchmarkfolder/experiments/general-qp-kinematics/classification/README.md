# Private graph-derived kinematic classification

This prototype generates static identity metadata from actual HELAS objects. It
does not alter a matrix entry, threshold, reducer call, Born calculation,
coordinate preparation, mass value, normalization or acceptance status. No
compiler or numerical amplitude is run. Production files and the frozen
campaign remain unchanged. The append-only private exporter patch is for source
review, not production adoption.

`src/madgraph/loop/kinematic_identity.py` derives external wavefunction momentum
signs using the actual external call dictionary and spin-state convention:
IXXXXX writes `-NSF*P`; OXXXXX/SXXXXX/VXXXXX write `+NS*P`. It checks these against
physical crossing. For intermediate currents it asks the existing ALOHA writer
for momentum-conservation signs and supports its current all-positive mother
sum, including the standard massless `P0` propagator. Other custom/projected
propagator paths and spins outside scalar-export SM scope produce unknown
rows. The helper recursively retains signed integer incidence, rejecting cycles,
repeated legs and disagreement with structural external-leg sets.

Actual amplitude mothers and PAIRING define corners through the existing
`PL=-sum(W%P)` convention. The helper checks the disjoint external partition and
the ordered ancestry returned by `get_denominators`. Every grouped member must
share the reference's corner and selected mass-symbol routing. Grouped IDs are
the finalized group IDs; ungrouped and loop-induced IDs match the actual call
writer's amplitude-number path. The identity table never substitutes one ID
scheme for another.

The primary shifts preserve the literal first shift, the sum of all corners.
CT2/CT5 reverses and negates corners and permutes masses for N>2, matching the
existing switch helper; N<=2 is unchanged. Each unordered pair retains its
literal integer vector. Class codes are 0 unknown, 1 direct single leg,
2 conservation complement, 3 composite and 4 closure-only. Single/complement
witnesses obey `v = sign*e_leg + closure*(1,...,1)` in the all-incoming basis.
This formal conservation identity does not assert exact floating-point closure.

The generated fixed-form subroutine is:

```fortran
CALL PREFIX_KINEMATIC_IDENTITY(G,CT,N,I,J,CLASS,LEG,SGN,
     $ CLOSURE,DECLARED_ONSHELL,KNOWN)
```

It uses static integer DATA arrays and explicit arguments, without COMMON state
or floating-point arithmetic. Reversed pair arguments negate the signed
witness. Invalid IDs, unsupported modes, size mismatches, diagonal requests or
unknown graphs return `CLASS=0`, zero witness and false logical outputs. The
one-denominator case needs no pair entry. `DECLARED_ONSHELL` reflects only the
external process flag and identity; it is not a repair, shell-residual or physics
certificate. Off-shell leg descriptors disable it without altering identity.

`prepare.py` copies the exact production exporter into `base/`, creates a
private copy in `src/`, and appends this standalone lookup to the existing
optimized `write_TIR_interface` output. All original emitted bytes remain a
prefix of the new output. No caller consumes the table yet, so this prototype
does not implement the later shell policy or unknown-channel fallback dispatch;
it makes unknown metadata explicit for that future integration. The separate
QP-mass prototype edits different helper/call paths and remains independent.

Run from the MadGraph repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/general-kinematics-classification/prepare.py
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/general-kinematics-classification/test_generation.py
```

The tests freshly convert a private UFO copy and construct LoopAmplitude and
optimized LoopHelasMatrixElement objects; no output/launch command is used.
The retained exploratory `probe_graph.py`/pickle is not needed by the tests.
Nine checks cover:

* Actual `u u~ > z g g [virt=QCD]`: 116 groups and 477 pairs per direction,
  matching all literal corners/shifts, mass permutations, pair classes and leg
  witnesses in the independent frozen source-routing proof. All 19 targeted
  small composite channels per direction remain composite.
* Actual grouped/ungrouped reduction IDs, crossed `u g > z u g`, and loop-induced
  `g g > h h` graph generation and emitted call ordering.
* Actual N=2 groups and a constructed N=1 graph made from real HELAS classes.
  The latter is a boundary test, not a physical SM tadpole-process qualification.
* Inconsistent grouped ancestry returning unknown; off-shell descriptor
  rejection; integer DATA/witness serialization and an append-only exporter
  source comparison.

`source-tests.json` records results and before/after protected source hashes.
Generated lookup/table files are under `generated/`; `implementation.patch` and
`preparation.json` bind the private derivation. These are source-generation
results only. Fortran compilation/runtime bounds behavior, active reducer
integration, CMS numerical behavior, Born/loop effective-point consistency and
performance remain untested here. Unsupported broader off-shell/vector/model
paths are outside this prototype, without changing their existing legacy code.
