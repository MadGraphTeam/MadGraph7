# QP transport and pole source audit

This is a source audit, not a numerical adjudication. Generated source paths
below are relative to
`.tirust-build/physical-stress/unchecked/ew_n3_epem_ttxa/candidate-9-qp-init-dp/`.
The corresponding backend-1/6/8 diagnostic exports use the same transport code.

## Common input preparation

- `SubProcesses/P0_epem_ttxa/loop_matrix.f:2130–2156` promotes the DP input and
  calls the actual QP phase-space repair, then copies `MP_PS` to `MP_P`.
- `SubProcesses/P0_epem_ttxa/improve_ps.f:104–123` selects the ORIG repair for
  `ImprovePSPoint=2` **and for values ≤0**. Setting 0 does not disable QP repair.
- The ORIG repair uses `MP__MDL_MT` for legs 3 and 4 at lines 620–621, solves
  outgoing longitudinal components at 657–665, and repairs incoming momenta at
  695–723. The geometry probe measures this exact routine.
- `Source/MODEL/param_read.inc:25–27` calls both mass readers. The MP reader's
  `REAL*16` output passes through a `REAL*8 buff` at
  `Source/MODEL/lha_read.f:374–375,403–404`; it retains DP parsing precision.

## Numerator and denominator transport

`SubProcesses/P0_epem_ttxa/mp_compute_loop_coefs.f:174–179` honors all four QP
integrand flags. With the frozen cards' `.TRUE.` values, lines 308–309 and
396–399 compute the numerator in QP. The repaired QP wavefunction momenta are
then copied into DP wavefunctions at 421–433 for the legacy CutTools path.

The generated `CT_interface.f:1230–1237` constructs both DP `PL` and QP `MP_PL`.
Dispatch at 1260 passes DP `PL` to CutTools, while 1268, 1277 and 1282 pass QP
`MP_PL` to Ninja, NinjaRS and TIRust respectively. This distinction remains
present in forced-QP mode.

| Backend | Denominator shifts at its API | Squared masses | Scale |
|---|---|---|---|
| CutTools 1 | Repaired momenta downcast to DP; DP prefix sums | DP | DP `MU_R` |
| Ninja 6 | QP prefix sums | Promoted DP square | Promoted DP `MU_R**2` |
| NinjaRS 8 | QP prefix sums, lossless binary128 transport | Promoted DP square | Promoted DP `MU_R**2` |
| TIRust 9 | QP prefix sums, lossless binary128 transport | Promoted DP square | QP square of promoted `MU_R` |

Specific source evidence:

- CutTools: `CT_interface.f:28–32,112–121` declares DP inputs and sums the shifts
  in DP. `vendor/CutTools/src/cts/cts_cutroutines.f90:69–74` is itself a DP API;
  lines 298–308 first load DP denominators and promote those already rounded
  values for the QP reduction. `loop_num.f:90–96` evaluates the genuine QP
  numerator. Thus enabling the QP integrand does not upgrade denominator
  transport.
- Ninja: `CT_interface.f:649–651,712–724,754–789` promotes the DP squared masses,
  retains QP shifts, supplies MadLoop's thresholded invariants and passes QP
  numerator coefficients. `HEPTools/ninja/Ninja/src/mninja.F90:67–76,349–365`
  binds that overload with kind 16. The C++ wrapper consumes both shifts and
  the supplied matrix at `src/quadsources/ninja_wraps.cc:48–63`.
- NinjaRS: `CT_interface.f:1062,1087–1120` uses the same ordered QP sums and
  thresholded matrix as Ninja. Installed `HEPTools/ninjars/include/mninjars.f90`
  selects the QP kind at line 10, verifies radix 2 / 113 digits at 184, and packs
  with `ES64.39E4` at 144. `vendor/ninjars/src/lib.rs:577` parses at 113 bits and
  widens to the 256-bit working precision declared at line 40. This preserves
  the input binary128 values; no DP rounding path was found here.
- TIRust: `TIRUST_interface.f:128–136,157–197` retains QP shifts and invariants,
  promotes the DP mass square, and computes the QP scale square. The separate
  bridge transport qualification is outside this geometry-only experiment.

## Two different kinds of threshold

MadLoop's `OSThres` is applied by `CT_interface.f:296–318` to relative
mass-shell differences and to a massless invariant divided by a local energy
normalization. It can change the supplied scalar matrix without changing the
shift vectors used in tensor reduction. The matrix probes demonstrate this
distinction explicitly. They do not identify which physical scalar topology
causes the observed amplitude discrepancy.

The scalar-library cutoff is a separate control. CutTools initializes its
OneLOop implementation with absolute `1e-6` at
`vendor/CutTools/src/cts/cts_cuttools.f90:82–84`. The Ninja QP default is
`1e15 * FLT128_EPSILON`, approximately `1.925929944e-19`, at
`HEPTools/ninja/Ninja/src/quadninja/types.hh:55` and is sent to OneLOop at
`src/quadsources/avholo_interface.cc:38–42`. NinjaRS deliberately uses the same
nominal QP cutoff at `vendor/ninjars/src/lib.rs:620–631`; it evaluates its QP
scalars through oneloopmaster rather than Ninja's OneLOop implementation.
Matching cutoff values alone does not establish matching classifications or
stable finite terms near a threshold.

The pinned NinjaRS dependency is
`41a4c00e7635e3ab188fc26bacf0aae6941e8082`; its oneloopmaster is
`5b710437ada83cbc6e8c6d51e56f7d6ee6a0edb5`. In
`.tirust-build/cargo-home/git/checkouts/ninjars-3842d69907c1a3a3/41a4c00/crates/ninjars-symbolica/src/masters.rs:346–366`,
`snap_onshell` mutates all small scalar invariants, including both D0 channels,
to zero, and snaps real invariant values close to real internal squared masses.
Lines 607–621 apply that snapping before D0 relabelling. The corresponding
`src/integration.rs:168–180` replaces the scalar invariants by entries of the
supplied MadLoop matrix. The relabelling is defined in
`.tirust-build/cargo-home/git/checkouts/oneloopmaster-1e046f4c0b16c2f9/5b71043/src/labelling.rs:39–91`.

By comparison, native OneLOop
`HEPTools/oneloop/OneLOop-3.6/avh_olo.f90:22492–22573` thresholds classification
magnitudes for four external invariants and four masses, attempts a permutation
when a channel is small, and zeros selected on-shell-distance classifiers. It
does not perform NinjaRS's blanket mutation of all six invariant arguments.
This is a source-backed semantic difference; no captured EW scalar row yet
establishes that it causes the observed discrepancy.

## Independent double-pole expectation

`Template/NLO/SubProcesses/fks_singular.f:7752–7753,7791–7804` gives a QED double
pole of `-2 * Born * alphaEW/(2*pi)` for this process. Only the two massless
incoming charged leptons contribute; massive top legs and the photon do not
add a double pole. Single poles also contain charge-linked soft terms and
cannot be inferred from charge counting alone.

Using the reported DP Born and the geometry probe's real MP alphaEW gives
approximately `-20202.60812481593`. CutTools, Ninja and NinjaRS agree with this
check within `7.7e-10` relative, while the frozen TIRust diagnostic differs by
51.296%. An exact repaired-QP Born would sharpen this check. Agreement among
reducers is not used as a proof of the finite or single-pole coefficients.

## Focused diagnostic implications

1. Preserve the frozen point and actual QP repair, and capture physical per-loop
   masses, shifts, invariant matrix and native status before attributing a
   failure to any scalar topology.
2. Vary `OSThres` independently of the scalar-library cutoff. `1e-24` removes
   the top-shell snapping seen at `1e-8` and `1e-14`; this deliberately changes
   the consistency relation between the supplied mass square and repaired
   external shell and is not automatically a better setting.
3. A private CutTools QP denominator-transport diagnostic can isolate its known
   DP transport loss, as in the earlier complex-mass experiment. It must remain
   separate from the original-policy acceptance records.
4. Ninja/NinjaRS already share the same QP shift precision and MadLoop matrix
   construction. Their remaining discrepancy requires reduction/scalar
   diagnostics, including native status and per-loop coefficients; another
   promotion-only transport patch would not address an identified loss there.
   A subsequent private scalar diagnostic should vary snapping and working
   precision independently. No public NinjaRS card control sets either its
   absolute scalar cutoff or MPFR working precision.
