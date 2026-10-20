# Final generator and native-runtime regeneration

The candidate was regenerated after the independently tested expanded/compact
Fortran polynomial generator and GNU/Linux native link ordering were frozen.
Every generation used an actual `bin/madgraph` command card. The old candidate
tree is retained under `.benchmark-work/outputs/pre-final-candidate`, with its
internal symlinks relocated and recorded in `archive.json`. The earlier
`pre-optimization-candidate` archive was preserved.

`status.json` records the eleven generation/build operations and their frozen
inputs. `core-snapshot.json` contains exact generated-source and executable
hashes for all eight core cases. Every core parameter card is byte-identical to
the untouched baseline and pre-final archive; all retained point files are
unchanged. Every generated subprocess build starts without object files.
The snapshot also verifies that the actual executables import versioned GNU
`__addtf3`, `__multf3`, and `__gttf2` helpers.

Generation times include the CLI's model import, diagram generation, export,
and any source-library work performed during export. Subprocess build times
cover `make -j4 check`; they do not represent rebuilding external dependencies.
The original command cards and retained inputs are in `archived-commands/` and
`archived-inputs/`.

The refreshed worker-protocol runs use the frozen controller/driver/main-harness
identity and the regenerated outputs. All eighteen source/backend/case
fresh/persistent comparisons are bit-for-bit exact, including actual
CutTools/Ninja/NinjaRS higher-precision rescue. The final ggH, mixed-order, and
near-singular rows also pass the required cross-backend numerical tolerance.
`parity-refresh.json` binds those checks to their retained measurements and the
completed protocol proof.

## Focused-control checklist

The following must be evaluated with the final chosen generated code and
numerical runtime before full task correctness can be claimed. Existing results
from before this regeneration remain historical evidence.

- [ ] Loop-induced `gg → HH`: regenerate and compare all four backends.
- [ ] Complex masses: regenerate and compare all four backends.
- [ ] Forced higher precision: regenerate and compare CutTools/Ninja/NinjaRS.
- [ ] Rotations: regenerate and compare backends and unrotated inputs.
- [ ] Changed masses/scales: regenerate rebinding and fresh-context controls;
  compare backends and reused/fresh NinjaRS contexts.
- [x] Regenerated representative `gg → H`, mixed QCD/QED, and near-singular
  controls: NinjaRS parity and final-runtime comparisons across applicable
  backends pass.
- [ ] After any further promoted generator or numerical-library optimization,
  rerun affected controls and core measurements with exact new provenance.

Protocol parity validates persistent execution and repeated state. It does not
by itself certify cross-backend physics agreement, independent analytic poles,
or the five-percent performance gate. Those outcomes are recorded separately.
