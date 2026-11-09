# CutTools QP input transport: isolated causal diagnostic

Preserving QP propagator momenta through the CutTools input boundary removes
the remaining point-20 discrepancy after lowering its scalar-integral
threshold. This is a private diagnostic, not a production patch or an
acceptance result. The original-policy observations remain failed or
unqualified. All four runs return `120` with accuracy `-1`; no runtime
comparison is inferred from their diagnostic elapsed times.

The physical process, model, masses, scale, point, QP numerator and
coefficient construction are unchanged. The exact input is the frozen CMS
forward point 20, with physical transfer
`t = -2.499999999999791305007018193257151140e-7` GeV².

| Momentum input | CutTools scalar threshold | Normalized finite | Normalized double pole |
|---|---:|---:|---:|
| Existing DP transport | `1e-6` | `12.88097128305223` | `-0.1279754851102243` |
| Existing DP transport | `1e-10` | `-54.18406587886088` | `-0.2559475017848395` |
| Private QP transport | `1e-6` | `12.88097171071167` | `-0.1279754820732312` |
| Private QP transport | `1e-10` | `-54.18341448215070` | `-0.2559461514807408` |

The first two rows come from the parent threshold experiment; the last two
are recorded here. Coefficients use the generated standalone driver's
normalization by `Born * alphaS/(2*pi)`.

With threshold `1e-10`, the private QP result agrees with the separately
recorded QP Ninja, NinjaRS and TIRust diagnostics to at most `2.11e-11`
relative across Born, finite and both poles. The maximum forward/reverse
relative difference is `3.17e-11`; mode 6 selects the reverse result. The
double pole agrees with the independent four-massless-lepton charge-sum
prediction `-4*abs(aEW)/alphaS` to `5.54e-16` relative using the printed model
parameters. This formula follows the QED double-pole sum in
`Template/NLO/SubProcesses/fks_singular.f:7789` and the generated coupling
and standalone-driver normalization. The model's imaginary `aEW` component
is `3.45e-20`.

The actual OneLOop `printall` inputs show why both changes matter. The
physical `t` and its occurrence count are identical before and after the
transport change. The nonzero external on-shell residual falls from
`2.64385014548454258e-12` to `2.52435489670723778e-29` GeV². Both are below
the lowered scalar threshold; however, the original OPP denominator
construction still used the DP-rounded momenta. Preserving QP momenta
addresses that second effect. Keeping the original `1e-6` scalar threshold
still misclassifies the physical small transfer even with QP inputs.

## Exact scope and provenance

`run.py` clones the completed baseline threshold diagnostic into
`.tirust-build/complex-stress-cuttools-qp-transport`. It changes only the
generated `CTLOOP` momentum arguments and prefix-sum temporaries to
`REAL*16`, passes the already-constructed `MP_PL`, and calls a new private
external symbol, `CTSXCUT_QPINPUT`.

The private routine is copied from the existing CutTools source. Its only
algorithmic changes are the `REAL*16` momentum argument and lossless
assignments to direct/reverse QP denominator momenta. Masses and scales
retain the original DP interface. The OPP coefficient routines, scalar
integral code, numerator callbacks, and installed libraries remain
unchanged. Only modes 4, 5 and 6 are accepted. The private object precedes
the original archive at link time; the linked executable contains the new
`ctsxcut_qpinput_` symbol.

`build/build.json` binds the parent build, exact source patches, compiler
commands, headers/modules, executable, physical card, and all object/archive
hashes. Exactly `CT_interface.o` changes and
`cts_cutroutines_qpinput.o` is added. Each observation binds its executable
through that build manifest, input cards, physical point, threshold input,
runner snapshot, raw log and result. `analyze.py` verifies those bindings and
unchanged numerical basis before writing `summary.json`.

The initial private compilation used a generated fixed-form include in a
free-form source. It failed before any native evaluation; the source,
compiler log and explanation remain in `build-include-failure/`. The
successful build compiles the private routine in its own directory using
copies of the original free-form CutTools headers.

## Reproduction

After reproducing the parent threshold build in a fresh ignored workspace,
run these commands from the MadGraph repository root. Existing build/run
directories are deliberately never overwritten.

```sh
source tools/tirust_env.sh
probe=docs/benchmarkfolder/experiments/complex-mass/direct-qp/cuttools-adjudication/qp-transport
python3 "$probe/run.py" build
python3 "$probe/run.py" run --label threshold-1e-10 --threshold 1e-10 --mode 4 --timeout 30
python3 "$probe/run.py" run --label original-threshold --threshold 1e-6 --mode 4 --timeout 30
python3 "$probe/run.py" run --label threshold-1e-10-reverse --threshold 1e-10 --mode 5 --timeout 30
python3 "$probe/run.py" run --label threshold-1e-10-both --threshold 1e-10 --mode 6 --timeout 30
python3 "$probe/analyze.py"
```

Reanalyzing the preserved evidence requires only the last command. All
original-policy failures and the distinction between fixed-QP diagnostics
and stability-certified amplitudes are retained.
