# CMS universal QED double-pole consistency audit

For the retained massless `e+ e- > mu+ mu- [virt=QED]` fixture, the private
candidate TIRust QP double pole at point20 agrees with the universal four-leg
QED formula to `2.679e-16` relative residual. The retained original CutTools QP
double pole is approximately half the formula's value. This finding supports
only the double-pole coefficient: finite and single-pole accuracy, complete
amplitude accuracy, normal MadLoop stability and candidate promotion remain
unqualified. The parent candidate comparison gate remains false.

The calculation uses

```
A[-2] = -4 * Born * alpha / (2*pi)
alpha = sqrt(2)*Gf*abs(muW^2*(1-muW^2/muZ^2))/pi
muV^2 = MV^2 - i*MV*GammaV
```

The retained model excerpts select the existing complex-mass Gmu definition
with `GFRedefinitionChoice=1`. The reconstruction parses the actual card values
through binary64 before 70-digit Decimal arithmetic evaluates the mathematical
expression. This packet has no card-reader or MP-widening source excerpt, so
that parsing boundary is an explicit reconstruction assumption. Runtime input
and coupling bits and complete Fortran rounding are not observed or emulated.
The original calculation uses
the retained model's decimal pi literal. Its bootstrap point1 observations
check the same pole normalization for both selected QP workers.

`source/` preserves all eight original files and their index unchanged, sealed
by `5748eeb64f53f4a201ae0e5a355221c03e5dac39790d55531e52b0d290e057e3`.
The original README's private paths and proposed future experiments are
historical text. Use the public verifier below for portable reproduction.

The added wrapper also checks the algebra independently using
`abs(w*(z-w)/z)^2 = abs(w)^2*abs(z-w)^2/abs(z)^2` at 90 working digits.
It joins all eight retained rows word for word, including
worker, eligibility, phase, call, point, order and return code, to the closed
parent's point20 comparison. It also matches the physical card byte for byte
to the retained source input and selected worker/build records. The rows cover
two eligible fixed-QP workers, their point1 bootstrap and point20 measurement,
and both order columns. Failed TIRust workers are excluded.

Ten original source-file hashes, one card hash and the comparison hash are
historical provenance. Seven generated-source hashes additionally match the
parent's closed build inventory. Retained excerpts are authenticated text;
this packet does not freshly rehash or reconstruct all ten complete source
files. It does not reopen original binaries or verify their current contents.

CutTools' recorded `1e-6` scalar on-shell classification cutoff is a hypothesis
for the discrepancy, not a proven causal scalar-call diagnosis. The retained
source distinguishes classification magnitudes from invariant arguments and
the separate MadLoop chord-based projection rule. Ninja and NinjaRS use the
legacy helper, so future comparisons must establish actual input geometry
and scalar policies. This audit contains no native backend6/8 comparison or
change to a cutoff, model, card, production template or installed library.

## Portable reproduction

Keep the sibling `../kinematic-invariant-amplitude-qualification-1` beside this
directory, with seal
`c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4`.
The wrapper authenticates that packet's bytes and joins its recorded rows;
the parent's own verifier performs its broader source/build/protocol replay.
To run that broader replay, retain its three declared sibling dependencies too.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/cms-universal-qed-pole-audit-1/verify.py
```

Only Python's standard library is used. The wrapper invokes the retained pure
calculation, never its optional live-source branch. Verification runs no
compiler, loader, native amplitude, timing campaign or network operation.
