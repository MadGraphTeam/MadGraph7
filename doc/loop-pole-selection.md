# Selecting UV and IR poles in MadLoop

`LoopPoleMode` selects the ultraviolet (UV) and infrared (IR) poles of the
bare loop contribution computed by NinjaRS or COLLIER. It leaves finite
coefficients, rational terms, and model counterterms unchanged.

After pulling this MadGraph version, update an existing NinjaRS installation
from the MG7 prompt:

```text
install ninjars --update
```

The selector requires bridge ABI 4. Older ABI-3 installations are rejected
before export/linking, with instructions to update. Generate a new process
output to obtain the corresponding Fortran interface and card option.

## Parameter card

For UV poles with NinjaRS followed by COLLIER as a rescue backend, set these
entries in the generated process's `Cards/MadLoopParams.dat`:

```text
#MLReductionLib
8|7

#LoopPoleMode
1
```

| `LoopPoleMode` | Loop poles returned |
| --- | --- |
| `-1` (default) | Existing backend behavior: combined poles for NinjaRS; the two legacy COLLIER switches for COLLIER. |
| `0` | Both UV and IR poles. |
| `1` | UV poles only. |
| `2` | IR poles only, including the double pole. |
| `3` | No poles; finite coefficients are still returned. |

An explicit mode (`0`–`3`) overrides `COLLIERComputeUVpoles` and
`COLLIERComputeIRpoles`. Modes `1`–`3` require every entry in `MLReductionLib`,
including rescue backends, to be NinjaRS (`8`) or COLLIER (`7`). For example,
`8|6|7|1` is rejected in these modes because Ninja and CutTools cannot preserve
the selected pole origin during rescue. Use `8`, `7`, or an ordered combination
of the two available backends.

## Calling from Fortran

Generated optimized processes also expose the process-prefixed
`SET_LOOP_POLE_MODE(MODE)` routine. If the matrix routine is named
`ML5_0_SLOOPMATRIX`, the setter is:

```fortran
      CALL ML5_0_SET_LOOP_POLE_MODE(1)
```

Call it before evaluating the desired sample. It accepts the same five card
values and preserves the requested choice through MadLoop initialization.
Use the prefix of an actual generated subprocess. The choice is global within
a shared MadLoop library: the most recent setter call applies to every
subprocess, including when another subprocess performs the first evaluation.
Call it with `0` to restore combined poles or `-1` to restore the legacy backend
policy. The reduction and rescue list must still satisfy the selected mode's
requirements.

## Interpretation and cost

The selector acts on loop-integral poles before the reduced coefficients are
contracted. In particular, a scaleless bubble has cancelling UV and IR
single-pole residues that must be retained separately. The NinjaRS bridge
selects these residues at its oneloopmaster-backed master-provider boundary.
UV-only and IR-only evaluations use the generic NinjaRS engine in double
precision and in higher-precision rescue so that scaleless-bubble coefficients
are retained. These modes share the Symbolica worker and licensing requirements
of higher-precision rescue. A UV-only or IR-only retry through the compensated-DP entry also
uses this generic DP engine; the fast-engine compensated numerator callback
does not apply. Higher-precision rescue remains available. Combined and
no-pole double-precision evaluations retain the fast reduction engine.

All explicit modes (`0`–`3`) bypass MadLoop's helicity and loop-filter decisions,
including previously learned filters. This prevents filters trained on
combined poles from discarding contributions whose separate UV and IR poles
are nonzero. The default `-1` retains the usual filtering policy. These
diagnostic choices can therefore increase numerator and reduction costs,
including for explicit mode `0` compared with the default combined-pole run.

For the bare loop amplitude's pole residues, UV plus IR reconstructs the
combined single pole; the double pole is IR. This statement concerns the pole
coefficients, not a sum of the finite coefficients, which each mode retains.
Standard MadLoop outputs can additionally contain model counterterms; this
option does not split their UV and IR origins or remove them.

Loop-induced outputs involve squaring the loop amplitude. Squaring introduces
interference between its components, so UV-only and IR-only **squared outputs
are not generally additive**. Check pole reconstruction at amplitude level,
and compare NinjaRS and COLLIER using identical modes, inputs, and stability
settings.

## Validation

The independent [COLLIER comparison](../vendor/ninjars/tests/poles/README.md)
covers 455 tensor components through rank four, including scaleless bubbles,
complex masses, and UV-divergent tensor triangles and boxes. All selected DP
results and all higher-precision modes agree with COLLIER at the documented
tolerance. Repeated mode changes reproduce fresh results. Ordinary fast-DP
diagnostics are recorded separately and reproduce the pre-feature library.

The following processes were also generated with the actual MadGraph CLI and
compared at three fixed points each, including all returned split-order rows:

| Process | Model | Precision checks |
| --- | --- | --- |
| `u u~ > z [virt=QCD]` | `loop_sm` | Forced DP/QP and automatic stability checks |
| `g g > t t~ [virt=QCD]` | `loop_sm` | Forced DP/QP and automatic stability checks |
| `e+ e- > mu+ mu- [virt=QED]` | `loop_qcd_qed_sm_Gmu` | Forced DP/QP |
| `g g > h [sqrvirt=QCD]` | `loop_sm` | Forced DP/QP |

All four selectors agree with COLLIER. The automatic QCD runs accept these
points in DP; forced QP runs exercise the higher-precision path. These comparisons isolate the bare loop
by clearing UFO UV/R2 counterterm slots only in disposable generated validation
copies. They leave the production numerator and reduction code intact. Tests
also cycle the selector after loading learned filters. The loop-induced check
compares each mode between backends; it does not assume additivity of squared
outputs. The [retained receipt](../vendor/ninjars/tests/poles/madloop-validation.json)
records the precise run matrix, inputs and tolerances.

Local regression qualification passed 107 Python/compiled-Fortran tests,
30 ordinary Rust tests, and nine export IO tests. Seven pre-existing opt-in Rust
diagnostics remain excluded. The real CLI `install ninjars --update` passed its
compiled DP/QP smoke check, including a binary128 perturbation below DP
resolution. A compiled two-subprocess regression checks that the latest global
selector survives the first card read and subsequent runtime changes.
A generated shared library containing both `u u~ > z` and `d d~ > z` also
passed these checks with NinjaRS and COLLIER, including 40 amplitude evaluations.
The Python/Fortran regression modules can be run with:

```sh
python3 -B -m unittest discover -s tests/unit_tests -p 'test_ninjars*.py'
python3 -B -m unittest tests.unit_tests.loop.test_loop_poles
```
