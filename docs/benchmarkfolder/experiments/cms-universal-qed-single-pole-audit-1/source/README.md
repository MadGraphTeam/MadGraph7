# Universal QED single-IR-pole consistency at the CMS forward point

The actual FKS pole and charge-link source gives an unambiguous prediction for
the renormalized virtual interference of e+ e- -> mu+ mu-, with four massless
charged external fermions. Candidate9 QP at point20 agrees with this prediction
to 5.23e-16 relative; the retained CutTools QP value differs by about52.985%.
This is an IR single-pole consistency check, not finite-amplitude truth or
checked stability. The prior double-pole packet5748eeb6 and its publication
remain unchanged. A byte-identical copy is retained under `double-pole/`.

## Derivation from the primary implementation

`Template/NLO/SubProcesses/fks_singular.f:getpoles` assigns -3/2 times each
massless charged fermion's squared charge to the collinear single-pole
coefficient. Four unit charges therefore give -6. For each unordered soft
pair it calls `sborn_sf`, multiplies the returned charge-linked Born by -2,
and multiplies by log(2*p_i.p_j/QES^2)/(8*pi^2).

`madgraph/iolibs/template_files/sborn_sf_fks.inc` returns
Born*charge_i*charge_j*GAL(1)^2, with a sign flip when exactly one leg is
incoming. Thus the physical charges [+1,-1,+1,-1] give all-outgoing charges
Q=[-1,+1,+1,-1]. Their sum is zero and sum(i<j,Q_i*Q_j)=-2. GAL(1)^2/(8*pi^2)
is alpha/(2*pi). At the end `getpoles` expands the scale prefactor by adding
double*log(mu^2/QES^2). Using double=-4*Born*alpha/(2*pi), the auxiliary QES
dependence cancels, giving

```
A[-1] = Born * alpha/(2*pi) *
        [-6 - 2*sum(i<j, Q_i*Q_j*log(|2*k_i.k_j|/mu^2))].
```

Here k_i are all-outgoing momenta. Equivalently, use future-directed physical
momenta and the explicit incoming/outgoing charge signs. The FKS source uses
real positive physical dot products. Imaginary analytic-continuation phases
of the soft operator do not contribute to this real interference when alpha
is real, even though the complex-mass tree amplitude can be complex.

An independent algebraic regrouping checked by the script is

```
factor = -6 + 2*log(s12*s13*s24*s34 / (s14*s23*(mu^2)^2)),
sij = |2*k_i.k_j|.
```

The script also evaluates the original FKS QES form at QES^2=(13/7)*mu^2 and
checks cancellation to 60 decimal places. This is pure algebra/arithmetic, not
a new scale-varied amplitude evaluation.

## Saved inputs and numerical result

The four actual complex binary128 external W/MP_W leaf records and native
mu^2 come from the closed CMS point1/20, group3, mode4 traces in public packet
730f8f22. Only `KT_EXTERNAL_QP` and `KT_TIRUST_QP` input records are decoded.
Their context, leg order, real/imaginary words, source lines, trace hashes,
specifications and closed-worker receipt joins are retained. No failed tensor
coefficient or old incomplete amplitude is used. The old stress stage remains
failed; an input capture from it is not reclassified as a successful reduction.

The HELAS sources prove incoming W1/W2=-p1/-p2 and outgoing W3/W4=+p3/+p4.
The leaf vectors thus already have all-outgoing signs. Pair dot products are
computed as exact Fractions of the input bits, without cancellation in a
binary64/128 squared-difference operation. Seventy-digit Decimal arithmetic
evaluates the logarithms. Exact shell and conservation residuals are retained;
the external model masses are zero. No infinite-precision physical point is
inferred from those rounded inputs.

The private trial and old capture have identical physical/numerical cards and
point-file hashes. The current trial did not make a new external-word capture:
this calculation uses the earlier closed actual QP leaves, with their explicit
input ancestry and unchanged momentum-source path. It does not disguise that
ancestry as a fresh measurement.

At point20 native mu^2=1000000 and the two small positive pair invariants are
2.499999999999791352612888445781735378001939330938987494923342325137128e-7.
The full single-pole factor is
-122.0692619081930891440208252556538670852994310136419483315453927590847.
Using the existing source/card Gmu-CMS alpha from the double-pole audit gives

| Quantity | Value |
|---|---:|
| Expected single pole | -0.00496751292099208065560699419052798 |
| Candidate9 QP | -0.00496751292099207805708083185436408 |
| Candidate relative residual | 5.2310405703e-16 |
| Original CutTools QP | -0.00233546698305832399825132661419502 |
| CutTools relative residual | 0.5298518554 |

Both order columns are retained. At ordinary point1 the same formula agrees
with CutTools at5.48e-17 and candidate9 at2.43e-16 relative. These ordinary
checks support the normalization/crossing interpretation; they are not used
to fit alpha, charges or any coefficient.

## UV and qualification limits

The formula applies to the intended UV-renormalized full ANS single pole.
It must not be compared with a bare tensor's IR1 slot alone. The actual card
has LoopPoleMode=0 and finite evaluation enabled. The generated MP HELAS
source supplies explicit UV 1/epsilon counterterm amplitudes; MP coefficient
construction sums their Born interference, and loop_matrix adds the loop
groups to that counterterm buffer. Those source paths are pinned/excerpted.
Counterterms are therefore part of the compared ANS rows. This audit does not
separately measure UV cancellation or independently validate every model
counterterm. The Gmu-CMS model selects its existing real alpha definition;
runtime coupling bits and full Fortran complex-rounding emulation are absent,
as already disclosed in the double-pole packet.

Neither pole check establishes the finite coefficient, full amplitude truth,
checked stability, general process coverage or performance. Fixed120 remains
unchecked. There is no finite leading-log estimate or finite accuracy claim.
The failed DP contribution and retained stress/regression failures remain open.
No native process, reducer, compiler, model/card change or threshold change is
part of this work. An initial pure development attempt read `passed` at the
wrong closure level; its recorded failure is kept in `development-note.json`.
Correcting that metadata access changed no inputs or pole formula.

## Reproduction

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B .tirust-build/cms-universal-qed-single-pole-audit-1/verify.py
```

The entire packet can be copied elsewhere and verified with standard Python3.
Default verification reads only retained text, traces and pure Python. It
authenticates the original packet seals, the selected closed trace receipt
joins and both calculations. `calculate.py --verify-live-sources` additionally
rehashes the named current source/card/specification files; it does not load
their recorded native binary paths or execute a scientific calculation.
