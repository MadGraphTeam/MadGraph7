# Fresh QCD3 production-port diagnostic

The retained `run.py` uses actual MG5 CLI generation and a clean Fortran build
from the recorded development patch. It selects 136 private QP tensor-factor
calls. With the original parameter card and all100 retained phase-space points,
Ninja6 reproduces the sealed original coefficients, accuracy values and statuses
exactly:98 DP evaluations and two QP rescues. NinjaRS8 agrees within the frozen
pairwise coefficient tolerance, using98 DP evaluations and two QP rescues; the
largest tolerance fraction is0.8194450141.

This validates the fresh standalone generator integration. It is a diagnostic
from an explicitly retained dirty patch, not final clean-source generation or a
performance acceptance run. The timing fields have no acceptance role. A fresh
NLO export and all final source/runtime bindings remain separate requirements.

`capture.py` copies the completed build/evaluation logs, selected generated
Fortran, exporter/test snapshots and the last backend8 raw files. Both parsed
measurement streams, source identities, cards and compiled manifests are retained
by the unchanged benchmark harness. `runtime.json` binds the actual reduction
libraries, module/link metadata and executable. `verify.py` only rehashes the
capture and recomputes its comparisons; it never launches an evaluation.
