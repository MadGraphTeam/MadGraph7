# Independent EW infrared-pole check

The frozen `e+ e- > t t~ a [virt=QED]` threshold fixture has an independent
pole constraint. Specializing MadFKS's UV-renormalized QED pole formula to the
actual generated model gives the single-to-double-pole ratio without using
any reducer output, Born normalization or electromagnetic coupling. The
NinjaRS automatic and fixed-QP values satisfy that ratio to `1.6762e-11`.
The accepted Ninja automatic result does not: its ratio differs by `111.13`
relative to the predicted ratio. This is an accuracy failure despite its
historical host status `328` and reported accuracy `5.37e-7`.

This does not establish a correct finite term. Historical TIRust threshold
amplitudes omit 37 refused tensor groups and are incomplete; their pole
values are shown only to illustrate that invalidity. Automatic TIRust status
`420` correctly rejects this point. An unchecked positive-mode status `120`
does not change that conclusion.

`audit.py` uses the actual captured binary128 momenta, top mass, scale and
coupling; it decodes the printed decimal round trips exactly before evaluating
the formula with 90 decimal digits. The source identities include the generated
`NF=5`, `NL=3`, exact UFO charges, physical card and process command. `virt`
mode forbids tagged photons. The independently reviewed formula is

```
P = B * Re(GAL(1))^2 / (8*pi^2)
c[-2] = -2*P
c[-1] = P*(5/9 - 2*sum(i<j, Q[i]*Q[j]*S[i,j]))
Q = (-1,+1,+2/3,-2/3) in all-outgoing charge convention.
```

The soft factors `S` use the massless, mixed-mass and massive-pair expressions
in `fks_singular.f`, including its small-velocity series. The diagonal `5/9`
includes the untagged-photon anomalous dimension `gamma_ph=-40/9`. Choosing
`QES2=mu2` eliminates the explicit scale-prefactor logarithm. No separately
fitted counterterm or Gmu conversion is added. The source audit and its limits
are recorded in `source-review.md`.

The ordinary control predicts single pole `7.704415681834645e-10` and double
pole `-4.399335851312526e-10`; the complete historical TIRust amplitude agrees
to `1.91e-15` and `1.93e-15`, respectively. The threshold prediction is
single pole `14590.7746363863` and double pole `-20202.6081248159`.

| Threshold observation | Historical host outcome | Relative residual of single/double ratio |
|---|---|---:|
| CutTools baseline/candidate automatic | rejected `420` | `1.3251e-6` |
| Ninja automatic | accepted `328` | `1.1113e2` |
| COLLIER automatic | rejected `410` | `1.3783e-2` |
| NinjaRS automatic | accepted `322` | `1.6762e-11` |
| CutTools fixed QP | unchecked `120` | `3.6948e-6` |
| Ninja fixed QP | unchecked `120` | `1.1060e2` |
| NinjaRS fixed QP | unchecked `120` | `1.6762e-11` |
| Private CutTools QP denominator transport | unchecked `120` | `9.7989e3` |

The metric here is `abs(observed-predicted)/abs(predicted)`, not the campaign's
symmetric regression metric. `comparisons.json` retains every printed finite,
single and double coefficient, status, accuracy, point/card identity and raw
file hash. It also retains incomplete TIRust residuals, explicitly marked
invalid. Original failures and source files are unchanged.

The returned Born is a DP evaluation at the original point, promoted for
printing, whereas the eikonal momenta have undergone MadLoop's QP repair.
Consequently the absolute pole comparison has an unbounded mixed-input
normalization uncertainty; 90-digit arithmetic does not imply 90-digit
physical precision. The threshold NinjaRS absolute single-pole residual is
`7.8578e-10`. The ratio cancels that Born/coupling prefactor exactly, but still
uses the recorded repaired kinematics and scale. Pole agreement cannot certify
an independent QP Born, finite coefficient or complete stable amplitude.

Reproduce from the repository root, without a compiler or reducer invocation:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/analytic-poles/audit.py
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/analytic-poles/compare.py
```
