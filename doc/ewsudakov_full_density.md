# Full EW Sudakov spin interference

The FxFx decay-density path uses the full complex Born and linear Sudakov
interference matrices. Spectator helicities are summed after forming each
interference. They are never averaged into a helicity-dependent amplitude ratio.

For resonance helicities `h,h'`, spectator tuple `s` and colour metric `G`,

```
B[h,h']  = sum_s M[h,s] G M[h',s]^*
K[h,h']  = sum_s dM[h,s] G M[h',s]^*
dB       = K + K^dagger
w        = 1 + sum(dB * C) / sum(B * C)
```

Each shifted amplitude retains its own colour basis; rectangular colour
metrics connect it to the physical Born. Complex parameter derivatives are
taken before interference. Colour components orthogonal to the same-helicity
Born, imaginary terms, and interference involving a zero Born helicity are
therefore retained. The weight is linear in the correction, with no squared
one-loop term. Only a contracted Born that vanishes within rounding,
`|<B,C>| <= 1e-10 sum|B*C|` (counted as `events_density_vanishing_contraction`),
gives the unit fallback; there is no absolute cutoff on a small, nonzero Born
amplitude.

## Longitudinal vectors

The default is `change sudakov_density_prescription amplitude`. The physical
Born appears in `B` and in the conjugate side of `K`; the longitudinal Sudakov
correction is evaluated with the corresponding Goldstone amplitude. For the
supported Sudakov SM UFOs, each outgoing longitudinal W+, W- or Z contributes
the phase `-i`. This is a phase factor per replaced leg. Single Wh and Zh
checks fix the phases that a WW check alone cannot distinguish.

An optional comparison is available:

```
change include_sudakov True
change sudakov_density_prescription goldstone_relative
launch
```

It forms `(G_Born^dagger dG)/(G_Born^dagger G_Born)` separately for each
spectator tuple and resonance helicity, then applies that relative correction
to the physical Born. It divides by the Goldstone norm, not the physical
longitudinal Born. This is a different finite-mass prescription and projects
away colour components orthogonal to that Born. It is not the full amplitude
interference. A nonzero physical Born with zero Goldstone norm makes this
prescription undefined and raises an error; it does not quietly select the
scalar path. The default amplitude prescription needs no such division.

With the amplitude prescription, a missing Goldstone tree gives zero
leading-power longitudinal correction. The comparison prescription needs a
Goldstone tree for every longitudinal resonance: `q q~ > Z Z` has none, so a
decayed ZZ event stops a run with that prescription. An auxiliary shifted
process that fails momentum reshuffling contributes zero, matching the scalar
kernel's threshold prescription.

The scalar kernel treats such a state differently: it takes the first
Goldstone amplitude whose legs are all longitudinal, or else the Born with
its Born-based operators. Events that use the scalar path (no decayed
resonance, no decay matrix, too large a helicity space, or a kernel error)
keep that treatment, so one FxFx sample can mix both for longitudinal states
without an exact Goldstone tree. In `u u~ > Z Z` at 2.5 TeV this is 0.17% of
the correction.

## Parameters and generated libraries

The full Born and correction use the same NLO model, parameters, momenta and
coupling-order selection, with the NLO convention of zero external widths.
The production matrices come from this kernel; only the decay matrices use a
dedicated standalone output, which reads the event banner's parameter card.
A comparison with a standalone production amplitude must also match its
width convention: a finite W width affects the complex off-diagonal entries
of photon-induced WW.

The kernel is written only into the reweighting library (`output
ewsudakovsa`), whose generated dispatcher declares `DENSITY_VERSION=4` and
returns `density_born`, `density_correction`, `density_relative`, and
`zero_gold`. The latter two fields are for the comparison prescription; with
`goldstone_relative=False` they are not evaluated (`None` and 0). Matrices are
in Fortran helicity order (first resonance fastest); the event path permutes
both matrix axes to its decay-density ordering. An FxFx (`ickkw=3`) run
rejects a library, or a result, without version 4: regenerate `rw_me`.

The five native variants are, in order: QED, weak, s/r off, both s/r and
invariant floor off, and invariant floor off. `density_sudakov(...,
nvariants=n)` evaluates the first `n` of them (all five by default). Each
variant builds its own invariants with its own floor setting, as the scalar
kernel does. The three FxFx output variants (QED, central, s/r off) are the
first three, and all use the full spin contraction; the event path evaluates
only these. The LHE weight group of an `ickkw=3` run records
`density_kernel="4"` and the selected `density_prescription`.

The retained size limit is 36 resonance spin states and 10 resonance legs.
Unsupported large spaces and unavailable decay matrices use the scalar
fallback and its accounting. A polarised leg cannot be retained: its helicity
table is restricted, and the kernel refuses it. The Sudakov charge and
reshuffling implementation assumes massless beams: generating any EW Sudakov
output for a massive beam (W, Z, top) stops with an error, so neither kernel
runs on them. Both kernels also need `YMT = MT` in the param card, which EW
Sudakov reweighting checks before it starts; `ntadpole` is set to 1 when the
Sudakov library is compiled.

## Scope and validation

### Keeping incoming spins open

For the FxFx event path (`ickkw=3`), add this to the reweight card:

```
change include_sudakov True
change sudakov_density_keep_initial True
launch
```

This retains a process tensor in each event for every evaluated xi value.
It requires the default reweighting output, with all weights in one event
file. For `q q~ -> W+ W-`, it is **36 by 36**: a 9-by-9 array of 4-by-4
incoming spin blocks, including entries with different incoming helicities.
It stores the physical Born tensor and the linear Sudakov correction tensors
of the QED, weak and s/r-off variants; with the `goldstone_relative`
prescription, it also stores the Goldstone-relative tensors (`relative`)
behind its weights. Every event that reaches the kernel passed the
small-invariant cut, which makes the invariant floor inert, so the record's
`variant_aliases` maps the floor-off variants to the s/r-off and weak tensors.
It does not store a squared-loop term or normalise these arrays to unit trace.

The combined index is `i = a + 4*h`, with zero-based indices. Incoming leg 1
varies fastest, then incoming leg 2, then the supplied resonance legs (also
first leg fastest). Incoming states are `(--), (+-), (-+), (++)`. Fermion
labels `-1,+1` are HELAS signs, corresponding to physical helicities `-1/2,+1/2`.
For WW the array can be reshaped to `(9,4,9,4)` without permuting its elements.
Stored leg indices, helicity tables, canonical PDGs and momenta fix the basis.
The PDG lookup key is stored separately because its final-state IDs are sorted
and need not be in the native momentum order.

Writing `R[h,a;h',a'] = M[h,a] G M[h',a']*`, the tensor removes the native
incoming **spin** divisor only. Colour averages, identical-particle factors
and sums over any other unretained outgoing spins remain. For an incoming
spin density `sigma`, the outgoing matrix is

```
B[h,h'] = sum_(a,a') R[h,a;h',a'] sigma[a,a'].
```

The ordinary unpolarised result uses `sigma = I_4/4`, including its factor
of one quarter. The event weights use this reduction, so keeping the larger
tensor leaves the original unpolarised prediction unchanged apart from
floating-point summation order. The incoming labels describe a scattering
response; they are not additional outgoing particles in the WW spin state.

The generated Python interface accepts

```python
result = dispatcher.density_sudakov(pdgs, p, g, positions, dims,
                                    keep_initial=True)
tensor = result['process_tensor']
# result['density_born'], result['density_correction']: reduced matrices.
from madgraph.various.process_spin_tensor import (
    contract_process_tensor, save_process_tensor, read_event_process_tensors)
reduced = contract_process_tensor(tensor)  # I_4/4 by default
save_process_tensor('process_tensor.npz', tensor)
records = read_event_process_tensors(event)
```

LHE records use a separate `<sudakov_process_tensor>` extension containing
compressed NPZ data encoded as base64. The arrays are Hermitian in their two
matrix indices, and only their upper triangles are stored, which is lossless
up to rounding. Each record includes its xi, normalisation, index order,
momenta, coupling, prescription and variant order.
No pickle is required. A record marked `not_evaluated` identifies the
scalar or unit-weight path; it is not a zero tensor. Only events whose weights
use the density contraction carry an `available` tensor. Records already in
an input event are removed when its Sudakov weights are recomputed, with or
without this option. Saved tensors are the
amplitude calculation before the runaway-weight protection. Reading
and writing the LHE event preserves the extension. Storage is opt-in because
each tensor contains substantially more data than the reduced matrix.

The open-input mode supports two incoming quarks, charged leptons,
photons or gluons, with **at most 36 combined spin states** and ten retained
legs including the two incoming ones. WW saturates that state capacity.
An unsupported space or an export failure raises an error rather than silently
dropping the requested indices. The Python dispatcher must declare
`PROCESS_TENSOR_VERSION=1`; any version-4 kernel supports the open positions.
Different partonic channels and helicity conventions must be kept identified
when combining stored tensors.

### Approximation and checks

“Exact” describes the spin/spectator/colour contraction of the implemented
linear Sudakov amplitude. The electroweak correction remains a logarithmic,
high-energy approximation. Mass-suppressed helicity channels and finite-mass
differences between physical and Goldstone amplitudes are not made exact by
the contraction. The operator content and coupling-order limitations of the
scalar kernel apply (including absent QCD corrections to LO2).

The two prescriptions differ most for mixed transverse-longitudinal vector
pairs. For left-handed quarks, the Goldstone amplitude of such a pair is
typically a few per cent of the physical one at high energy, because it
couples only through hypercharge. The amplitude prescription therefore leaves
these states almost uncorrected, whereas the comparison prescription applies
the full Goldstone correction. The leading-power approximation does not decide
between them.

Unit tests in `tests/unit_tests/ewsudakov/test_full_density.py` check the
Python contraction with constructed amplitudes (coherent amplitude closure,
spectator incoherence, orthogonal colour contributions, zero Born entries,
scale invariance), and, with a mocked kernel, the dispatcher, the variant and
prescription selection and the rejection of older interfaces. The FKS
export fixtures compare the generated kernel source for ZZ and top pairs. The
kernel's numerical results were validated outside these tests, against
independent colour/helicity contractions using five-point amplitude
derivatives, and on real decayed-event weights in both MG5 3.8.1 and MG7.
