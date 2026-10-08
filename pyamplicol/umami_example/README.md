# MadSpace integration through the generated UMAMI API

This example generates `g g > g g g` in leading colour (LC) and full colour,
builds each generated UMAMI shared library, runs its linked C driver, compares
batched MadSpace evaluations with native pyAmpliCol, and performs a real partonic
integration with MadSpace and VEGAS. LC and full-colour integrals need not agree:
they are different colour approximations.

`test_umami.py` and `test_integrate.py` are adapted from the examples by
R. Frederix and T. Vitos in the
[AmpliCol `madspace_interface` branch](https://github.com/rikkert-frederix/AmpliCol/tree/madspace_interface):
[loading example](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_umami.py)
and [integration example](https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_integrate.py).
The loading example retains MadSpace's batched matrix-element calls; the
integration example retains the original seven-section layout, phase-space
mappings, differential cross section, per-channel VEGAS and optional MadNIS/PDF
workflow.

The upstream `madspace_interface` branch contains no explicit license file or
license notice in these two scripts. Their attribution is retained, and this
repository does not assign them a new 0BSD license. The generated UMAMI header
has its own separate `UMAMI_LICENSE` notice.

## Run the complete example

Use an environment containing this version of pyAmpliCol, a C compiler and Make.
The following are optional example dependencies, not pyAmpliCol requirements:

```sh
python -m pip install -r umami_example/requirements.txt
python umami_example/run.py
```

Generation uses recurrence JIT O2, one worker and one optimization core, with
numerical current-relation discovery explicitly disabled. Full colour uses the
configured default exact colour contraction (adjoint FFT). Both accuracies use
automatic symmetry grouping.

The output is `.artifacts/umami-example/`. Artifact payloads are in `lc/` and
`full/`; compiled libraries and drivers are outside them in `build_lc/` and
`build_full/`. Each integration writes `integration_lc.json` or
`integration_full.json`, with a cross section in pb and statistical uncertainty.
Defaults are intentionally small: two VEGAS training iterations with 256 and 512
points, then 2048 integration points **per independent channel**. These are
smoke demonstrations, not precision cross-section predictions.

Existing artifacts are never overwritten. Continue using the saved artifacts
with `--reuse`, or choose another `--output`. For example:

```sh
python umami_example/run.py --reuse --accuracy full --n 8192 --training 1024 2048
```

If `rusticol-config` is not on `PATH`, pass
`--rusticol-config /absolute/path/to/rusticol-config`. It must belong to the same
pyAmpliCol installation as the generation/runtime environment. Its `--cflags`
and `--libs` supply the headers and native library to the generated Makefile.
The C driver links to the resulting UMAMI `.so`, not to a second copy of
Rusticol. The Makefile sets a relative runtime-library search path so the driver
and `.so` can be moved together.

## Run the individual steps

For an already generated artifact, use absolute paths (substitute your paths):

```sh
make -C /path/to/artifact/API/umami BUILD_DIR=/path/to/build
/path/to/build/umami_driver /path/to/artifact
python umami_example/test_umami.py /path/to/artifact \
  --library /path/to/build/libumami.so
python umami_example/test_integrate.py /path/to/artifact/API/umami/metadata.json \
  --library /path/to/build/libumami.so --artifact /path/to/artifact \
  --no-pdf --vegas-only --n 8192 --training 1024 2048
```

UMAMI's `param_card_path` argument is the **absolute artifact directory**, not a
parameter card. The artifact supplies the saved model parameters. The loading
test varies the declared αs parameter event by event when available and checks
the resulting matrix elements against native pyAmpliCol, including refreshed
dependent parameters. A sampled helicity label is informational: the returned
matrix element remains summed over the artifact's available helicities.
For complete LC coverage, the pointwise check expands the physical members and
their momentum mappings and checks that their sum reconstructs the native LC
total. Integration multiplicities are not pointwise factors.

Mixed process sets can produce multiple fixed-metadata providers. Select one
explicitly using `--provider p0` (or another ID from the metadata index) and its
matching `libumami_p0.so`. Its standalone driver is `umami_driver_p0`. The
orchestrator generates a single-process artifact and therefore needs no provider
selection.

## Integration options and assumptions

The default integration is at fixed partonic centre-of-mass energy 1 TeV, with
no PDFs and no downloads. All outgoing legs have `pT > 30 GeV`, `|eta| < 6` and
pairwise `delta R > 0.4`. Masses come from provider metadata; there are no
built-in particle-category or mass tables. This scattering example requires two
massless incoming particles, matching MadSpace's `1/(2 s_hat)` flux; the UMAMI
provider itself has no such restriction. The integration script's `--sqrts`,
`--pt-min`, `--eta-max` and `--dr-min` options change these settings. All compact
UMAMI integrations assume permutation-invariant, flavour-blind cuts and
observables. The script therefore has no species-selective cut option, and its
programmatic cut builder rejects species selections.

UMAMI automatically groups proven equivalent contributions with different
physical final-state flavours, and always groups integration orbits of identical
final-state PDGs, using oriented physical colour topology and permutations that
preserve the available helicity coverage. There is no additional reflection
quotient, equal PDF assumption or beam exchange. For complete LC coverage of `g g > g g g`,
24 physical colour contributions reduce to four representatives with
integration multiplicity six each. Full colour remains one contracted
contribution without an extra factorial multiplier. Equivalence comes from the
model expressions and generated computation, not equal parameter defaults or
Standard Model-specific particle rules. Unproved relations remain separate;
there is no grouping option.

The integration script applies metadata multiplicities once, assuming cuts
invariant under the recorded permutations and blind to final-state flavour,
and does not sum alternative mappings of the same contribution. Arbitrary
labelled-leg or species-sensitive cuts, or pointwise sums, require expanding the
retained physical members and their momentum mappings; multiplying one
representative by its integration multiplicity is not a substitute. The compact
integration script does not implement such cuts. Physical incoming PDGs keep
their own PDF weights.

For the optional neural-flow training, pass `--madnis --epochs 200` to
`test_integrate.py`. `--learn-discrete` also lets MadNIS learn the phase-space
mapping's discrete choices. The default is VEGAS only and one CPU thread.

For hadronic integration, use `--pdf --sqrts 13000 --pdf-dir /path/to/PDF/set`
and `--pdfset NAME`. The directory must already contain `NAME.info` and
`NAME_0000.dat`; this example does not download PDF data. If `--pdf-dir` is
omitted, the optional `lhapdf` Python binding locates the installed set. A
provider must expose a model-declared αs parameter for this running-coupling
workflow. A fixed scale can be given with `--scale`; otherwise the original
example's dynamical scale is used.

## Necessary adaptations from the AmpliCol examples

- Load the generated library and artifact path, including explicit provider
  selection, rather than a process-specific AmpliCol library and parameter card.
- Use current MadSpace call signatures, and derive masses, physical flavours,
  colour orderings and contribution maps from the generated metadata.
- Bind `FunctionRuntime` explicitly to the one-thread context holding the
  loaded provider; MadSpace 0.2.1's convenience calls use its global context.
- Construct MadSpace's matrix-element function with its explicit particle
  count, avoiding the convenience overload's unconditional diagram-count query.
  No diagrams are invented for the recursion-based provider.
- Integrate one representative per identical-final-particle LC orbit, with its
  metadata multiplicity. Full colour is an already contracted result and is not
  evaluated repeatedly as if it were separate colour flows.
- Retain Rusticol's incoming averages and identical-final-state factors exactly
  once. MadSpace's differential cross section adds flux, phase-space weights,
  optional PDFs and the conversion to pb.
- In the no-PDF workflow, use the artifact's default αs rather than requesting
  running αs without a running-coupling object.
- Sum only independent metadata contributions, not alternative phase-space
  maps without a partition of unity.
- Use bounded deterministic defaults, finite-result assertions, native numerical
  comparisons, and machine-readable integration results.
  Both Torch and VEGAS's independent NumPy generator are explicitly seeded.

The standalone SDK and complete UMAMI conventions are documented in the
project's UMAMI API page and in the generated `API/umami/README.md`.

## Optional grouping acceptance check

```sh
python umami_example/test_grouping.py
```

This generates one automatically compacted provider for the much smaller LC
process `g g > g g` and reuses the same real MadSpace/VEGAS integration code. Its
cuts are symmetric in the outgoing legs (`pT > 100 GeV`, `|eta| < 6`,
`delta R > 0.4` at 1 TeV). The fixture must compact six physical colour
contributions into three representatives with integration multiplicity two.
The reference expands each retained member into a separate integration channel,
composes its momentum mapping with the representative's map, and uses only its
physical weight, not the integration multiplicity. The provider's representative
computations remain certified and unchanged. It compares the compact and
explicit-member integrals using independent, fixed-seed
estimates within six combined statistical standard errors, with each estimate's
relative error below 5%. This is
stochastic validation, not the authority establishing amplitude equivalences.
The default is 4096 points per channel after 256/512-point training. Estimates,
errors, counts and comparisons are retained in
`.artifacts/umami-grouping/grouping_comparison.json`; `--reuse` skips regeneration.
