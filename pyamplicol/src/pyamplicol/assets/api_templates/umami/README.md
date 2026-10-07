# UMAMI provider

Build with the pyAmpliCol environment active:

```sh
make
make run
```

`rusticol-config --cflags` and `--libs` locate the installed native SDK. Set
`RUSTICOL_CONFIG=/absolute/path/to/rusticol-config` if needed. Build products go
to the sibling `.pyamplicol-api-build/<artifact>/umami` directory, not into the
artifact payload; override `BUILD_DIR` to choose another location.

A single provider produces `libumami.so` and `umami_driver`. Multiple providers
produce `libumami_p0.so`, `umami_driver_p0`, etc.; select `PROVIDER=p0` with
`make run`. `metadata.json` identifies the available providers and their
compatible external states. The driver takes `ABSOLUTE_ARTIFACT [CHANNEL [FLAVOUR]]`
and evaluates a saved validation point.

The shared library statically includes Rusticol. The driver links only that
shared library, with a relative loader search path; keep the driver and library
together when moving them. Runtime does not require Python. Retain all artifact
files and their native evaluator dependencies. The matching SDK is needed only
when compiling the provider.

Unlike the standard parameter-card convention, `umami_initialize` requires the
absolute artifact directory in `param_card_path`. Saved defaults initialize each
independent handle; `umami_set_parameter` updates mutable model parameters through
Rusticol. Optional per-event alpha_s is temporary and restored before returning.
The interface reports unsupported alpha_s when the model has no explicit mapping.
`UMAMI_META_MASSES` describes the saved default masses, not later handle-local
parameter changes.

Momenta use `input[event + stride*(particle + particle_count*component)]`, with
components `(E,px,py,pz)`. Channel and flavour indices are zero-based. The returned
matrix element includes the native incoming averaging and identical-particle
normalization. It sums the artifact's available helicities. A random helicity
input produces only an informational label; it does not sample the matrix element.
Explicit helicity selection, random colour selection, diagrams and GPU streams
are not supported by this adapter. LC rows select physical additive colour flows;
NLC/full-colour rows use the complete contracted colour result, without a flow.
These contracted providers do not support colour-count metadata or colour-index
output; the corresponding capability query reports this explicitly.
Integration multiplicities in metadata are not silently included in the returned
matrix element. Consult metadata for restricted coverage and grouping assumptions.

All grouping modes, including `none`, group integration orbits of identical
final-state PDGs using oriented physical colour topology and permutations that
preserve available helicity coverage. They do not add a reflection quotient or
exchange incoming beams. With complete LC coverage, `g g > g g g` has four
representatives for 24 physical colour contributions, each with integration
multiplicity six, in every mode. Full colour remains one contracted contribution
with no extra factorial multiplier; native identical-particle normalization is
already included.

`exact` is the default: it retains distinct physical flavour labels and allows
certified computation reuse. `flavour_blind_observables` additionally groups
proven equivalent contributions of different flavours under flavour-blind cuts
and observables. `none` disables optional UMAMI runtime/helicity reuse and
cross-flavour grouping, not the identical-particle integration quotient.
Equivalence uses model expressions and generated computation, not equal default
parameters or Standard Model-specific particle rules.

Integration multiplicities require cuts invariant under the recorded particle
permutations; they are not factors for pointwise sums. Metadata retains physical
members and their momentum mappings. Arbitrary labelled-leg cuts require
expanding those members even in `exact` or `none`; species-sensitive cuts are
supported in those two modes. No grouping mode assumes equal PDFs.

The public header is copied from MadGraph7's UMAMI interface. Its copyright and
license terms are retained in `UMAMI_LICENSE`; distribute that notice alongside
compiled providers. The pyAmpliCol adapter and driver are distributed under 0BSD.
