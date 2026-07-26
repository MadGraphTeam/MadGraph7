# Three-Open-Line Recurrence Oracle

This note records the independent original-AmpliCol check used to validate
three-open-quark-line recurrence construction.  It is a correctness oracle,
not a performance result.

## Contract

- Process: `d d~ > u u~ s s~`
- Phase-space point: massless RAMBO at 1 TeV, seed 731
- Legacy revision: `79c96cecf2a722e50c3d2030b6894d755f96518a`
- Legacy path: direct LC `amplicol_color_probe`, followed by all 64 physical
  helicity assignments
- pyAmpliCol layout: recurrence `topology-replay`

AmpliCol stores three adjacent open-string blocks in a different block order.
Canonicalizing blocks by their fundamental source label gives the public
pyAmpliCol flow identifiers below.

| Physical flow | AmpliCol helicity sum |
|---|---:|
| `flow:2,1,3,4,5,6` | `1.7260373034739047e-11` |
| `flow:2,1,3,6,5,4` | `1.6570372188601389e-11` |
| `flow:2,4,3,1,5,6` | `1.0167883928661888e-10` |
| `flow:2,4,3,6,5,1` | `1.6445040193096446e-10` |
| `flow:2,6,3,1,5,4` | `6.5052811045824496e-10` |
| `flow:2,6,3,4,5,1` | `1.2411150983558527e-10` |

The direct summed-helicity result is `1.0745996067347547e-09`; independently
summing the 64 fixed-helicity probes gives `1.0745996067347539e-09`.

## Finding

The recurrence result agreed with every mapped AmpliCol flow and with the
aggregate.  This independently validates the dynamic multi-line colour state,
closure partner terms, exchange signs, and normalization for this case.

During the audit, older built-in and UFO-SM compiled artifacts disagreed with
AmpliCol and with one another.  Fresh artifacts generated from the corrected
feature worktree now agree component by component with the same AmpliCol
oracle.  The regression in
`tests/integration/test_recurrence_three_line_execution.py` requires all three
comparisons: recurrence versus AmpliCol, fresh compiled versus recurrence, and
built-in versus UFO-SM.

The diagnostic build, process generation, and 64-probe pass peaked at
0.426 GiB, 0.247 GiB, and 0.054 GiB respectively under the 30 GiB guard.
