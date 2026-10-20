# Terminal QP production-port validation

This namespace validates the production generator implementation separately
from the sealed experiment evidence in its parent directory. These checks do
not establish a NinjaRS performance gate. Numerical evaluation uses original
Ninja6; the CLI explicitly disables NinjaRS while the independently developed
ABI3 installer is being validated.

`../validate_port.py` performs actual CLI exports, captures relevant generator
sources before each export, compiles both variants and compares the complete
printed coefficients, accuracy estimates and statuses. The reference variant
is the retained candidate export with the previously accepted sparse QP
updater and the common ASSOCIATE classifier storage fix. It is not the untouched
original performance denominator. Each case retains its exact script version,
source snapshots, generated/compiled manifests, cards/point hashes, executable
hashes, generation/build logs and raw numerical records.

| Case | Unchanged points | Private terminal calls | Exact comparison |
| --- | ---: | ---: | --- |
| QCD3 | 100 | 224 | 200 physical/total rows; 98 DP216, 2 QP328 |
| EW2 | 100 | 76 | 200 physical/total rows; 100 DP216 |
| Mixed orders | 10 | 151 | 50 physical/total rows; forced unchecked QP120 |
| Loop-induced ggH | 10 | 4 | 20 physical/total rows; forced unchecked QP120 |

The two control cases deliberately use `CTModeRun=4` with
`FORCE_STABILITY_CHECK(.FALSE.)` and bounded initialization. Status120 is
unchecked QP: these are full-stream structural/arithmetic regressions, not
stability acceptance evidence. The QCD3/EW2 cases use normal stability checks.
All numerical rows are finite and match the full reference updaters exactly.

`verify.py` independently checks complete point/order coverage, finite values,
allowed status sets and exact retained rows. It also reruns the current
post-render consumer proof on each generated process and compares 96 public
DP/QP updater source strings with the recorded committed reference revision.
Its results and current source hashes are in `verification.json`. For QCD3,
this verification adds the explicit finite/completeness check that was added
to the runner only after that case began; the original run script is preserved.

The focused validation consists of:

- 12 graph/rendered-consumer tests, including unknown whole-array consumers,
  aliased COMMON blocks, modified CREATE/MERGE calls, labels, cache isolation
  and per-batch IEEE boundaries.
- 108,000 compiled comparisons of consumed diagonal bits, full fallback bits,
  input immutability and cumulative IEEE flags across six rank/shape fixtures
  and three compiler flag sets. The sets include `-fno-automatic`; operands
  include rounding modes, signed zero, finite-bound boundaries, exceptional
  values and signaling NaNs. Instrumented fixtures exercise fast and fallback
  branches. The fixture uses the independent full updater oracle.
- 39 existing polynomial unit tests.
- Eight exporter IO tests spanning optimized standalone, FKS QCD/EW, MatchBox
  and TIR outputs. Only polynomial/private-call snapshots were updated.

Test logs are copied here. Documentation-graphics warnings about missing
Ghostscript occurred in the FKS IO run; the selected source comparisons passed.
The public full updaters remain available; only proven private terminal calls
omit unconsumed off-diagonal products. Full clearing and diagonal accumulation
order are preserved. Unsupported layouts, arithmetic modes, exponent ranges,
trap modes, ranks or graphs use the full routine.

Reproduce in the repository root after sourcing `.ninjars-build/dev-env.sh`:

```sh
python docs/benchmarkfolder/experiments/fortran-qp-followup/production-port/verify.py
python -m unittest tests.unit_tests.loop.test_terminal_wavefunctions
python -m unittest tests.unit_tests.various.test_q_polynomial_compiled.TerminalQPWavefunctionUpdateTest
```

Fresh generation must use a new output namespace or remove only an explicitly
selected disposable local copy; the runner intentionally refuses to overwrite
retained case evidence.
