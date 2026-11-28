# NLO matrix elements through UMAMI

Nov 17, 2026, last updated Nov 28, 2026 · @Ramon

## What this is

One proposal for how the NLO integrands get their matrix elements through UMAMI, in place of three documents:

- Theo's design `umami-nlo-design.md` (1 November), with the madspace review R1 to R12 (4 November)
- the survey of the providers `umami-nlo-providers.md` (4 November)
- the key proposal "UMAMI keys for NLO, by correlation order" (14 November)

It keeps what the integrands need for QCD, EW and mixed QCD-EW corrections, the keys and conventions, what the providers have and the open questions. [What changed from Theo's design](#what-changed-from-theos-design) is listed at the end. A madspace prototype on the branch madspace-fks implements the NLO keys as provisional ([The prototype in madspace](#the-prototype-in-madspace)).

## Implemented real-provider scope

The native NLO real-emission path implements the real-library part of these
conventions through the additive UMAMI 1.5 interface. One library evaluates one
real matrix element, without FKS-row inputs. It returns every signed local
squared-order component in batch-contiguous layout. For these NLO libraries,
`OUT_MATRIX_ELEMENT` is the sum of the complete returned block; the native FKS
caller applies its own order projection and subtraction factors. Ordinary LO
libraries retain their process-selected total. The process fingerprint binds
the NLO full-block convention, so an older selected-total library cannot be
silently reused.

Unit `IN_G_STRONG` (or `IN_ALPHA_S = 1/(4 pi)`) and `IN_G_EW` give coupling-stripped
blocks at fixed card masses, widths and mixing. Physical strong/EW powers are
restored by the caller. CPU and SIMD multi-amplitude-order component requests
retain all helicities;
GPU real libraries currently support one amplitude order. Unsupported requests
and native diagram-weight requirements keep their declared Fortran path. All
generated providers use the same header and report unsupported capabilities
explicitly; the Fortran-backed LO provider does not advertise the new blocks.

The dated provider survey and proposed numbering below are historical. The
released UMAMI 1.3 flavour-switch keys retain their numeric positions; ABI 1.5
appends the NLO inputs and outputs after them. The earlier draft NLO 1.4 used
colliding positions, so its generated outputs and libraries must be regenerated.
The native bridge checks the new ABI and fingerprint, and MadSpace rejects the
incompatible draft version before initializing it. Born providers and their colour/spin correlations, the
MadLoop UMAMI shim, colour-flow outputs and further GPU capabilities require
additional development in a separate branch and PR. The flattened coloured-leg
pair convention applies to that future Born work, not to the real block here.

## What the integrands need

FKS, Catani-Seymour and MC@NLO take seven blocks, all at the Born of the event except the real, and all per coupling order except amp2 and the colour flows. The integrands multiply each order by its powers of g\_s and e, g\_s at the scale of its own kinematics.

| Block | Momenta | FKS | CS | MC@NLO |
| --- | --- | --- | --- | --- |
| Born per order | Born | n-body part, every counterterm | n-body stream, every dipole | S events, MC counterterms |
| Real per order | real | real emission | real emission | H events |
| amp2 per Born diagram | Born | multichannel weight | multichannel weight | multichannel weight |
| Colour-correlated Born, every pair | Born | soft counterterms, n-body part | dipoles, I, K and P | not needed |
| Spin-correlated Born, gluon or photon leg | Born | collinear counterterms of a gluon or photon mother | dipoles with a gluon emitter | azimuthal part of the MC counterterms |
| Virtual per order, finite part and poles | Born | n-body part | n-body stream | S events |
| Born per colour flow | Born | no | no | colour partners of the shower |

The charge correlations of QED follow from the charges and the Born, so they need no block of their own. Mixed QCD-QED corrections add extra counterterms with the Born of another process (the other mother of a quark pair), at the same Born momenta.

## Libraries and coupling orders

One library per matrix element, as at LO, and no FKS rows. The Born library answers the Born, amp2 and all correlations, a real library its real, a loop library the virtual. The caller knows which matrix element each kinematics needs, so the subtraction scheme stays on its side. Extra counterterms and Catani-Seymour dipoles call the Born library of their own Born process.

The libraries return building blocks, not counterterms. The caller assembles the soft, collinear and soft-collinear counterterms and the integrated subtraction. The one exception is the spin correlation, which the library contracts with the vector of the caller. The virtual comes from MadLoop, wrapped by [a shim](#the-madloop-shim) and not reimplemented.

Every new output carries an axis over the squared coupling orders that the library lists in its metadata. `OUT_MATRIX_ELEMENT` stays their sum. The blocks come without couplings, and the caller multiplies each order by its powers of g\_s and e, at its own scale and in its own scheme. Two inputs carry the couplings, the strong one as `IN_ALPHA_S` (or `IN_G_STRONG` of Daniele's branch) and the electroweak one as `IN_G_EW` of Theo's design, double `()` each. Their unit values, alpha\_s = 1/(4 pi) and g\_ew = 1, give the blocks without couplings (open question 2). Only the power of e comes out of the blocks. The masses, widths and mixing angles stay those of the parameter card.

New metadata:

- `META_SQUARED_ORDER_COUNT`, int, the number of squared coupling orders
- `META_SQUARED_ORDER_POWERS`, int `(orders, couplings)`, the power of each coupling in each order
- `META_COUPLING_NAMES`, comma-separated, such as `QCD,QED`
- `META_COLOR_REPRESENTATIONS` (1, 3, -3, 8) and `META_CHARGES` per leg, so that the caller builds Casimirs and charge correlations without process data of its own
- `META_ALPHA_SCHEME`, the scheme of the electroweak coupling (alpha(0), G\_mu or alpha(m\_Z)), which fixes the renormalization of the EW virtual and the alpha the caller multiplies with (new)
- `META_LOOP_SCHEME`, the regularization scheme of the virtual, such as CDR (kept from Theo's design)
- `META_TAGGED_PHOTONS`, int, the number of tagged photons, whose couplings the library takes in the alpha(0) scheme (new)
- `META_EW_SUDAKOV_INTERFERENCE_COUNT`, int, the number of interference amplitudes of the EW Sudakov blocks, with their helicity combinations those of `META_HELICITY_COUNT` (new, see [EW Sudakov](#ew-sudakov))

## Keys by correlation order

NLO needs four new outputs, besides the existing `OUT_DIAGRAM_AMP2`. A correlation order (c, s) counts c colour insertions T\_i . T\_j and s legs whose polarization a vector of the caller replaces. The reserved rows name what the stability of the virtual and the colour partners of MC@NLO will need, so that a library that has them can report them without a new design.

| Level, order | Output key | Shape after the batch | Extra inputs | Needed for | When |
| --- | --- | --- | --- | --- | --- |
| tree (0, 0) | `OUT_SQUARED_ORDERS` | `(orders)` | none | everything | NLO |
| tree (1, 0) | `OUT_COLOR_CORRELATED` | `(orders, n_c (n_c - 1) / 2)` | none | FKS soft, CS dipoles, I, K, P | NLO |
| tree (0, 1) | `OUT_SPIN_CORRELATED` | `(orders)` | one spin leg and vector | collinear limits of gluons and photons, MC@NLO | NLO |
| one loop (0, 0) | `OUT_LOOP` | `(orders, 3)` | `IN_SCALE`, the renormalization scale | the virtual | NLO |
| one loop | `OUT_LOOP_STATUS`, `OUT_LOOP_ACCURACY` | `()` | none | stability of the virtual | reserved |
| tree, colour flows | `OUT_COLOR_FLOW_WEIGHTS` | `(flows)` | none | colour partners of MC@NLO | reserved |
| tree, EW Sudakov | `OUT_EW_SUDAKOV_DIAGONAL`, `OUT_EW_SUDAKOV_LO2` | `(h)` | none | the Sudakov approximation of the EW virtual | NLO EW |
| tree, EW Sudakov | `OUT_EW_SUDAKOV_DERIVATIVES` | `(h, 6)` | none | its parameter renormalization | NLO EW |
| tree, EW Sudakov | `OUT_EW_SUDAKOV_INTERFERENCE` | `(h, t, 2)` | none | its non-diagonal terms | NLO EW |

n\_c is the number of coloured external legs, those with a representation other than 1 in `META_COLOR_REPRESENTATIONS`. The colour-correlated output holds one value per pair of them, see Conventions.

The spin leg and its vector are two inputs. `IN_SPIN_LEGS` is int `(2)`, the 0-based legs whose polarization is replaced, -1 for an unused slot. `IN_SPIN_VECTORS` is double `(2, 4)`. Both are required exactly when a spin-correlated output is requested. At NLO the first slot holds the leg and the second is -1.

**Towards NNLO.** NNLO is left out, but nothing here blocks it. Keys are named by correlation order, so NNLO adds its orders as new keys at the end of the header, such as the colour-spin correlation (1, 1), the double colour correlation (2, 0) and the correlations of the one loop for the real-virtual. Two spin legs (0, 2) fill the second slot of the spin inputs for the same output, which is why the inputs have two slots now and keep their shape. One library per matrix element also serves the double real, as a real library with two emissions, and the real-virtual, as a loop library at real kinematics. A library reports what it supports through `umami_supported_outputs`, so the caller needs no metadata on the highest order.

## Conventions

- **Normalization.** Every output is summed over helicities and colours and averaged as `OUT_MATRIX_ELEMENT`, with no factor of a subtraction scheme (no iden\_comp, no FKS symmetry factor). Helicity sampling stays for blocks evaluated on their own, the virtual and later the EW Sudakov part.
- **Colour.** T\_i . T\_j B for the pairs i < j of the coloured legs, all legs taken as outgoing, without g\_s^2 and without a factor 1/2. The pairs are flattened in the order (1, 2), (1, 3), ..., (1, n\_c), (2, 3), ... of the coloured legs as they appear among the external legs, so the pair (a, b) of coloured legs counted from 0 sits at a (2 n\_c - a - 1) / 2 + b - a - 1. The rest of the full matrix carries no information, since it is symmetric, the colourless legs give zero, and its diagonal is C\_i B with the Casimir of the leg and the Born of the same order, which the caller builds from `META_COLOR_REPRESENTATIONS` and `OUT_SQUARED_ORDERS`. Then sborn\_sf(m, n) of MadFKS is g\_s^2 T\_m . T\_n B for m different from n and g\_s^2 C\_m B / 2 on the diagonal. Checked with pyAmpliCol to 1e-12.
- **Spin.** B(v) = v\_mu v\_nu B^{mu nu}, the Born with the polarization of each spin leg replaced by its real vector and the other helicities summed. The vector is in the frame of the momenta, and a library that evaluates in another frame boosts it with them. A part along the leg drops out by gauge invariance, so B(v) scales with the square of the transverse part of v. FKS passes k\_perp with v^2 = -1, and its azimuthal term is B / 2 - B(v). The B / 2 of quark mothers stays with the caller.
- **Virtual.** Absolute, not divided by the Born: finite part, 1/eps and 1/eps^2 per order, as BinothLHA gives them (CDR, normalization (4 pi)^eps / Gamma(1 - eps)), at the scale `IN_SCALE`. It can be requested alone, for the virtual on a fraction of the points.
- **Charges.** The caller builds the charge correlations from `META_CHARGES` and the Born, T\_m replaced by eta\_m Q\_m with eta = -1 for incoming legs.
- **Layout.** Column-major as for the momenta: the batch contiguous, the other dimensions strided by the batch, the first one fastest. The order index thus sits next to the batch. Theo's design listed shapes with the fastest axis last, so its `(3, orders)` of the virtual is the same memory as `(orders, 3)` here.
- **Complex values.** An extra axis of size 2, real part first, the slowest of all. With the first axis fastest as here it comes last, so a complex output per order has the shape `(orders, 2)` and its real part is a valid real tensor of shape `(orders)`. No new scalar type, the ABI stays `double`. No NLO key needs it, the EW Sudakov amplitudes later will.
- **Capabilities.** `umami_supported_inputs` and `umami_supported_outputs` say per key what a library can give at all. An input needed only with some output is required exactly when that output is requested, `IN_SCALE` with `OUT_LOOP` and the spin inputs with `OUT_SPIN_CORRELATED`, otherwise the call returns `UMAMI_ERROR_MISSING_INPUT`. A request the library cannot answer for the given values, such as a spin leg that is no gluon or photon, returns `UMAMI_ERROR_UNSUPPORTED_OUTPUT`.
- **Helicities.** The loop library takes `IN_RANDOM_HELICITY` or `IN_HELICITY_INDEX` and reports the helicity it took in `OUT_HELICITY_INDEX`, since the virtual samples helicities. Without either it sums.

## EW and mixed QCD-EW corrections

MadFKS calls the EW corrections QED. Their virtual has the full EW loops, their reals and counterterms photons. The keys above carry them and the mixed QCD-EW corrections with no further output, given the metadata of [Libraries and coupling orders](#libraries-and-coupling-orders).

- **Orders.** A QCD correction raises the power of g\_s by two, a QED correction that of e. The Born of order (a, b) in powers of (g\_s, e) feeds the QCD counterterms of the NLO order (a + 2, b) and the QED counterterms of (a, b + 2), as counterterm\_position of MadFKS. `META_SQUARED_ORDER_POWERS` gives the caller this map, so the order axes of the Born, real and loop libraries need not match.
- **Colour.** The colour-correlated Born of every Born order, the interference orders included, such as alpha\_s alpha in dijets. The library needs split orders for it.
- **Charges.** The QED soft counterterms take eta\_m eta\_n Q\_m Q\_n times the Born of the same order, from `META_CHARGES`. No block.
- **Spin.** The spin-correlated Born of a photon leg, for its splitting into a fermion pair, with the keys of a gluon leg. The coupling of the leg, QCD for a gluon and QED for a photon, picks the NLO order it feeds.
- **Extra counterterms.** A quark pair can come from a gluon or a photon, whose Borns can be different processes. The caller calls the Born library of the other process at the same Born momenta (R11).
- **Virtual.** The EW virtual depends on the scheme of alpha through its renormalization, not only through the power of e. The loop library renormalizes in its `META_ALPHA_SCHEME`, and the caller multiplies with the alpha of that scheme. Another scheme takes another library.
- **Tagged photons.** MadFKS takes the couplings of tagged photons in the alpha(0) scheme and the others in the G\_mu scheme. The conversion of the virtual (get\_virtual\_a0Gmu\_conv) changes its renormalization, so the loop library applies it, in the MadLoop shim. The factor of each contribution, a power of alpha(0) / alpha(G\_mu), stays with the caller.
- **EW Sudakov.** The Sudakov approximation of the EW virtual takes blocks of its own, see below.
- **Beams.** Photon and lepton densities and the luminosities of mixed corrections stay with the caller and need nothing from UMAMI.

## EW Sudakov

The Sudakov approximation of the EW corrections (Denner and Pozzorini, as compute\_ewsudakov of MadFKS and the `EWSudakov` of madspace) takes the Born per helicity combination h, of the `META_HELICITY_COUNT` of the library. The base amplitude of h is the Born, or the Goldstone amplitude where the longitudinal W and Z bosons of h are replaced by their Goldstone bosons. Four outputs give its blocks, all averaged and with the identical-particle factor of the Born, in the centre-of-mass frame of the incoming particles (open question 8, the first of its ways).

- `OUT_EW_SUDAKOV_DIAGONAL`, double `(h)`, |M\_h|^2 of the base amplitude at the LO1, the leading coupling order of the Born.
- `OUT_EW_SUDAKOV_LO2`, double `(h)`, the same at the LO2, with two powers of g\_s more.
- `OUT_EW_SUDAKOV_DERIVATIVES`, double `(h, 6)`, the derivatives of the first with respect to the six parameters of its parameter renormalization: 1/alpha or G\_F by the scheme of the EW input parameters, M\_Z, M\_W, m\_t with the Yukawa coupling, M\_H and the tadpole switch (numder\_\*.f).
- `OUT_EW_SUDAKOV_INTERFERENCE`, double `(h, t, 2)`, the interference M'\_t M\_h^\* of the base amplitude with each interference amplitude t of its base (a process with legs replaced by their SU(2) partners, at momenta reshuffled to their masses as momentum\_reshuffling.f does), at the LO1, zero for the interference amplitudes of another base, complex as the last axis. t is `META_EW_SUDAKOV_INTERFERENCE_COUNT`.

The caller takes the logarithms, the coefficients of the helicity combinations and interferences (ewsudakov\_functions.f) and the parameter renormalization from the process and the parameters, and the strong coupling and the scale of the LO2 part from its own scales. The blocks carry the couplings of `IN_ALPHA_S` and `IN_G_EW` as the others.

These keys follow the algorithm of MadGraph (Goldstone amplitudes, reshuffling, the parameters of numder) and fix it in the interface. Complex helicity amplitudes of every partner process would be more general but need many more calls. The prototype of madspace has the four outputs as provisional keys, served by the Fortran provider, and runs EW Sudakov through its C++ integrand with them.

## The MadLoop shim

The virtual stays in MadLoop. A C-callable shim presents `sloopmatrix*_thres_flavor` of a generated MadLoop output as a loop library, in the style of the real bridge of MadMatrix.

- **Initialization.** MadLoop reads its resource directory (`MadLoop5_resources` with the helicity filter `ML5_*_HelFilter.dat` and the colour factors) and `MadLoopParams.dat`. `umami_initialize` takes only the parameter card, so the shim finds the resource directory from the location of the card, with an environment override.
- **Threads.** UMAMI promises an independent instance per thread, while MadLoop keeps common blocks and saved state. The shim serializes its calls with a lock per library, so it never blocks the Born and real libraries. The OpenMP-safe vector entry (`BinothLHA_vec`, `sloopmatrix_thres_flavor`) can replace the lock later. This is the main risk for the speed of the virtual.
- **Several processes.** MadLoop symbols are not namespaced per process. The shim exports only `umami_*` and is loaded with `RTLD_LOCAL`, so the shims of two processes do not clash.
- **Flavours.** `BinothLHA` maps the FKS process to its virtual flavour through `VIRTUAL_FLAVOR_INDEX_D`. Without FKS rows the shim maps the flavour index of the caller to it.
- **Helicities.** With a random number the shim samples a helicity from the helicity filter of MadLoop (`goodhel`, `hel`, `mc_hel_flavor`) and reports its index. A helicity index selects one. With neither it sums.
- **Stability.** `OUT_LOOP_STATUS` and `OUT_LOOP_ACCURACY` give `ret_code` of MadLoop and its estimated accuracy. The pole check stays with the caller or a debug build, off the hot path. It compares with the Born of the Born library, which must be the Born MadLoop uses.
- **Tagged photons.** The shim applies `get_virtual_a0Gmu_conv`, since it multiplies the virtual and needs only parameters.

## MadMatrix and the GPU

- **madspace.** `MatrixElement` takes the new inputs and outputs, with shapes from the order and particle counts. The prototype has this for the CPU runtime, the GPU runtime needs the same.
- **Spin correlations.** The Born library takes a vector for one external gluon or photon. The generated `CPPProcess` builds the wavefunction of that leg from the vector instead of a helicity state, and the helicity loop skips the two states of the leg. This is a generator change in `madmatrix/model_handling.py` and the helicity tables, not a runtime one.
- **Colour correlations.** The colour-correlated Born needs the jamps and the colour matrix with insertions, not only the squared amplitude.
- **Other implementations.** The Fortran-backed `mg7/api.cpp` reports the new keys unsupported. The copies of `umami.h` stay in sync, checked by a test. The real bridge of Daniele's branch keeps serving the Fortran path until the new blocks agree with the Fortran.
- **GPU.** The Born and the real stay on the GPU for a single order, as now. All orders on the GPU need a kernel that copies the `nsqampso` components. The colour and spin correlations need the jamps and the polarization kernel on the GPU and come later. The MadLoop shim runs on the CPU only, so a GPU run takes the Born and the real on the GPU and the virtual on the host. Without an aggregating library the caller handles that mix.

## Headers, versions and providers today

Four versions of the header are in play, and two of them give the same values to different keys. The proposal is one list for 1.3: the keys of main, then those of #426, then Daniele's, then the NLO keys.

| Header | Minor version | Keys beyond main | Clash |
| --- | --- | --- | --- |
| main (madspace, MadMatrix) | 2 | none |  |
| #426, Theo, 11 November | 3 | `IN_RANDOM_FLAVOR`, `IN_PARTON_DENSITIES`, `OUT_FLAVOR_INDEX` | none, appended |
| feat-nlo-acceleration, Daniele, 27 October | 1 | `META_ABI_*`, `META_PROCESS_FINGERPRINT`, `META_SQUARED_ORDER_COUNT`, `IN_G_STRONG`, `OUT_SQUARED_ORDERS` | `IN_G_STRONG` sits where main has `IN_PARTICLE_COUNT`, `OUT_SQUARED_ORDERS` where main has `OUT_CUT` |
| pyAmpliCol pip 1.0.1 (4 November) | 2 | another list of outputs (`OUT_SCALE`) | `OUT_PDF`, `OUT_BIAS`, `OUT_MATRIX_ELEMENT_RATIO` at other positions |

What the providers have, as read on 4 November:

| Block | pyAmpliCol | MadMatrix | Fortran provider of aMC@NLO |
| --- | --- | --- | --- |
| Born per order | missing, no split orders | Daniele's branch only, CPU | yes |
| Colour correlations | Python and native SDK, not in UMAMI | missing, jamps and colour matrix exist | the pairs MadFKS registers |
| Spin correlation | Python and native SDK, not in UMAMI, leg fixed at generation | missing, the density code comes closest | mothers of FKS configurations |
| Real per order | missing | Daniele's branch, through the NLO real bridge | yes |
| Virtual | none | none, MadLoop stays Fortran | MadLoop |
| GPU | no | Born only | no |

A possible split: pyAmpliCol first for the Born blocks (its UMAMI export and adapter are smaller work than the MadMatrix generator changes, but QCD on Borns of one coupling order only), MadMatrix for split orders and the GPU, and [the MadLoop shim for the virtual](#the-madloop-shim).

Gaps of the MadMatrix UMAMI implementation on Daniele's branch, as read on 1 November:

- `umami.cc` rejects `IN_HELICITY_INDEX` and `IN_CHANNEL_INDEX`.
- `META_COLOR_COUNT` is unsupported, `set_parameter` and `get_parameter` return `NOT_IMPLEMENTED`.
- `g_externalMasses` is a global shared by all instances.
- `OUT_SQUARED_ORDERS` works on the GPU for one component only.
- `mg7/api.cpp`, the Fortran-backed implementation, is still at ABI 1.0.

## The prototype in madspace

The NLO keys work end to end in a prototype on the branch madspace-fks, first in three commits:

1. **Keys.** `IN_SPIN_LEGS`, `IN_SPIN_VECTORS`, `OUT_SQUARED_ORDERS`, `OUT_COLOR_CORRELATED`, `OUT_SPIN_CORRELATED`, `OUT_LOOP` and `META_SQUARED_ORDER_COUNT`, appended to the header of main and marked provisional. `MatrixElement` takes them, and `PythonFunction` gets outputs with shapes and reports its counts as metadata.
2. **Adapter.** The Fortran provider of each subprocess as UMAMI libraries through `PythonFunction`: one Born, one real per real matrix element, one loop.
3. **Integrand.** The C++ FKS integrand of one channel on these libraries. It gives the results of the Python driver on 13 processes, to 8e-16 of the summed absolute weights.

What it showed:

- One library per matrix element works. The caller splits a batch by real matrix element in the graph.
- Without FKS rows the provider evaluates the Born in the frame of another configuration than the Python driver does, so the two agree to the last bits instead of bit for bit.
- The runtime calls one library from several threads at once. The adapter needed a lock, as one provider process serves them all.
- `me_frame` boosts the momenta but not the spin vectors, so `MatrixElement` refuses the two together.
- FKS takes the Born, amp2 and both correlations in one call per Born kinematics, so one Born library call per event suffices.
- The scale and PDF variations need no key beyond these.

Since then the prototype takes `IN_G_EW` (the libraries at the e of the model), the Borns of the extra counterterms (one Born library per extra counterterm, which the integrand calls at the Born momenta of the configurations with it, R11) and the scale of the user (a scale library with the scale outputs UMAMI has, through `CustomFunction` at the kinematics of the real process). Not covered yet: the status and accuracy of the loop. The EW Sudakov blocks come through the four provisional outputs of [EW Sudakov](#ew-sudakov) (open question 8). QED corrections and tagged photons work. Among the 13 processes are p p > e+ e- \[QED\] and p p > a a with tagged photons, whose factors per order the integrand applies and whose conversion of the virtual comes with the loop library, as BinothLHA adds it.

## Open questions for the meeting

1. **Numbering.** One key list for 1.3: main, #426, Daniele's keys, the NLO keys. Who changes which header, and when does pyAmpliCol release with it?
2. **Couplings.** One form for both couplings, `IN_G_STRONG` and `IN_G_EW` or `IN_ALPHA_S` and an alpha of QED, with unit values for the blocks without couplings? Or outputs that carry no couplings at all? And the metadata of the alpha scheme and the tagged photons?
3. **Fixed keys or a request list.** Fixed keys per correlation order cover NLO and extend to NNLO by new keys. pyAmpliCol takes general colour connections by name, which would need a request format.
4. **Who provides what.** The Born blocks in pyAmpliCol or MadMatrix first, and where the split orders come from for mixed QCD-EW Borns.
5. **The loop.** The MadLoop shim first, or the one loop of pyAmpliCol?
6. **Flavours.** One flavour index across the Born, real and loop libraries of a subprocess, or a translation in the caller?
7. **Colour flows.** `OUT_COLOR_FLOW_WEIGHTS` summed or per order, and whether the colour choice of LO takes the same weights.
8. **EW Sudakov.** The proposal is the first of the three ways, keys for the blocks of the Born that the `EWSudakov` of madspace (compute\_ewsudakov of MadFKS) takes, written out in [EW Sudakov](#ew-sudakov) and in the prototype as provisional keys. The other ways were complex helicity amplitudes of every partner process with the derivatives from calls at shifted parameters, more general but with many more calls, and the correction itself per mode computed in the library, which would put the Sudakov logic into every provider. Open: whether the keys are acceptable as they are, and whether pyAmpliCol or MadMatrix can give the Goldstone and interference amplitudes. The interference output is `(h, t, 2)` rather than the `(t, 2)` of the first draft, since the interferences are those of the base amplitude of each helicity combination. The status and accuracy of the loop are not designed yet either.
9. **MadLoop setup.** How the shim finds the resources of moved or gridpack trees, and which settings are cheap per call (the scale) and which only per handle (the scheme).
10. **Colour accuracy.** A library at leading colour, as pyAmpliCol by default, fills the same pairs, with zeros for the pairs that no colour line joins, which ones depending on the process. The caller needs to know it, for the leading-colour Casimirs of the diagonal (C\_F becomes N\_c / 2). A sparse output of only the non-zero pairs, linear rather than quadratic in n\_c, can come later as a key of its own for high multiplicities.

## What changed from Theo's design

Theo's design of 1 November routed every block through an aggregating library per process with FKS rows. The review of 4 November (R1 to R12) and the proposal of 14 November moved the scheme to the caller and named the keys by correlation order.

| Topic | Theo's design, 1 November | Now | From |
| --- | --- | --- | --- |
| Libraries | an aggregating library per process with FKS rows (`IN_FKS_ROW`), routing to the Born, real and MadLoop libraries | one library per matrix element, the scheme on the caller side | R1, 14 November |
| Born blocks | per FKS row | per Born process, the caller passes the leg | R1 |
| Born kinematics | slots with an `active` mask | one Born per event | R2 |
| Colour | a list of colour links (`META_COLOR_LINKS`) | every pair i < j of the coloured legs, flattened, the diagonal from the Casimirs and the Born | R3, 20 November |
| Spin | one leg per row (`META_SPIN_CORR_LEG`) and one vector | the leg and vector as inputs, contracted in the library, B / 2 - B(v) the azimuthal term, a second slot kept for NNLO | R4, 14 November |
| Normalization | open | summed over helicities and colours, no iden\_comp | R5, R6 |
| Frame of the vector | open | the library boosts it with the momenta | R7 |
| Virtual | `OUT_LOOP` `(3, orders)` at `IN_SCALE_RENORM`, with `OUT_LOOP_BORN` | `OUT_LOOP` `(orders, 3)` at `IN_SCALE`, requestable alone, the same memory | R8, 14 November |
| Charge correlations | an output | built by the caller from the charges | R9 |
| Extra counterterms | not covered | the Born library of the other Born process | R11 |
| Key names | per FKS need (`OUT_BORN_SQUARED_ORDERS`, `OUT_COLOR_CORRELATED_BORN`, `OUT_SPIN_CORR_BORN`) | by correlation order, the same keys for the Born, the real and the loop | 14 November |
| Metadata | a row table, the blocks per row, split order names, Born charges, loop scheme | order count and powers, coupling names, colour representations and charges per leg, loop scheme, alpha scheme and tagged photons | 14 and 17 November |
| EW coupling | `IN_G_EW`, for the EW virtuals | kept, in the same form as the strong coupling, the unit value for the blocks without e | 17 November |
| Tagged photons | the conversion of the virtual (get\_virtual\_a0Gmu\_conv) in the MadLoop shim | kept in the loop library, the factor of each order with the caller | 17 November |
| EW Sudakov | not covered | four outputs for the blocks of the Born per helicity combination and the interference count as metadata, a proposal (open question 8) | 28 November |
| Reserved | MC@NLO colour flows as a separate design | colour flows and the status of the loop named now, NNLO left to later keys | 14 and 17 November |
| Version | 1.2 | 1.3, since 1.2 now carries other keys | 14 November |

Kept from Theo's design and written out above: building blocks rather than counterterms, MadLoop wrapped by a shim with its lock, `RTLD_LOCAL` loading and helicity sampling, the complex axis, capability reporting, `IN_G_EW`, `META_LOOP_SCHEME`, `META_SQUARED_ORDER_POWERS`, the GPU split and the colour flows of MC@NLO as a later output. Polarized processes need nothing further, since the library boosts the vector with the momenta, and reweighting takes the blocks per order. R10 and R12 only updated the status of madspace and noted EW Sudakov for later.
