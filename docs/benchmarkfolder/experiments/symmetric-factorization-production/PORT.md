# Private QP tensor-factor integration

The study motivated by [arXiv:1604.01363](https://arxiv.org/pdf/1604.01363)
first verified that MadLoop already stores independent symmetric monomials.
The retained optimization instead shares derivative contractions for two FFV
orientations: compute the first derivative row times the incoming tensor once,
then distribute signed row permutations over the four momentum components.
The momentum-independent contribution keeps its original accumulation order.

The generated Fortran checks the actual stored row relations and its finite
binary128 range/IEEE contract on every call. Unsupported shapes, identities,
rounding modes, traps or inputs use the unchanged public updater. The exporter
selects only adjacent recognized vertex/update calls at ranks2–4 with disjoint
wavefunction slots; terminal calls retain their complete consumer proof.
Each private call receives a freshly captured entry INEXACT flag. DP routines,
stability thresholds, rescue order and numerical inputs are unchanged.

The isolated fixed21-pair QCD3 study resolves a3.19% full-process reduction in
runtime, with95% interval1.79–4.15%, against its retained pre-factor candidate.
All16,800 compared printed coefficients, accuracy values and rescue counts match.
The production helper emits the same audited arithmetic and reproduces all
250,880 retained FFV oracle outputs exactly. Forty-seven polynomial tests and
twenty exporter/terminal tests pass, including408 compiled guard/IEEE cases.
Independent helper/exporter reviews found no blocking defect.

Fresh CLI QCD3 generation selects136 private calls. All100 Ninja rows equal the
sealed original, and NinjaRS agrees within the frozen coefficient tolerance;
each uses98 DP evaluations and two QP rescues. Fresh NLO generation/build selects
nine empty-prefix rank2 calls and passes soft/collinear and20 normal Ninja pole
checks. Separate bounded driver clones pass20 forced-QP poles each with Ninja
and NinjaRS. Their status120 means genuine QP execution with an unchecked H1
result; this diagnostic does not claim stable-point acceptance. All initial
harness failures remain retained and explained.

Evidence is split between the191-file helper seal,40-file fresh-QCD3 seal and
81-file NLO seal. Their verifiers run no amplitudes. `dev-nlo-v3/verify.py --live`
additionally rehashes the disposable generated trees and numerical runtimes.
The clean13-export/runtime-proof refresh and final eight-process gates remain
outstanding. The fresh NinjaRS QCD3 diagnostic is about2× the candidate Ninja
time; the tensor gain does not resolve that separate performance requirement.
