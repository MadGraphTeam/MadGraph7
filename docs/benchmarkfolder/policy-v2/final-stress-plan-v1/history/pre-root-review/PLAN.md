# Additional generic two-to-four correctness stress campaign

This is a new, bounded supplement to the unchanged eight core cases and the
separately frozen ggH/ggHH/stability campaign. It launches nothing. Root must
approve this plan, release the current quiet window and supply the final clean
candidate source, installed archive and measured runtime descriptor before CLI
generation, compilation, point generation or numerical execution. No process
specific implementation change is authorized by this plan.

## Fixed process and ordinary-point inventory

| Case | Model | Exact MG5aMC process | Ordinary points | Effective seed |
|---|---|---|---:|---:|
| stress-qcd-4-gg-ttgg | loop_sm | `g g > t t~ g g [virt=QCD]` | 20 | 1940083 |
| stress-ew-4-uu-4l | loop_qcd_qed_sm_Gmu | `u u~ > e+ e- mu+ mu- [virt=QED]` | 20 | 1940084 |

Use actual `bin/madgraph --plain` command files, optimized loop output and
external dependencies. Record clean source identities, generator inputs, model
checksums, compiler flags, generated source inventories and complete CLI/build
logs. Retain the first actual output, including failures. Use the existing final
physical parameter cards, real masses, zero widths and 1,000 GeV center-of-mass
energy. Imported EW-model provenance must be the same checksum-bound model as
the core campaign. Do not retune masses, widths, cuts, helicities or orders to
improve stability or reduce work.

Generate exactly twenty ordinary RAMBO points per case with the unchanged
`ninjars_benchmark.make_points` utility. The configuration's master seed is
1940083, with the fixed process order above, so the utility's existing seed-plus-
case-index rule yields the two effective seeds shown. Record actual external
masses from the generated `--masses` path. Validate all stored momenta for
finiteness, conservation and mass shells, retain their exact bytes and metadata,
and bind their hashes before the first reduction. No rejection sampling, point
replacement or seed search is allowed. The finite-mass/positive-energy
assumptions of the existing point reader are satisfied by these generated
physical inputs; this plan does not amend its separately documented parser gap.

## Mandatory candidate correctness comparisons

For each process, use all candidate backends 1, 6, 7 and 8 with the same final
candidate generator and numerical runtime as the core campaign. Run two complete
commands per target. Reuse the unchanged persistent worker and its full
helicity/color sums, pole settings and initialization policy. The two ordinary
commands each traverse the complete twenty-point stream once; the separate
calibration and initialization/warm pass are retained but cannot replace either
command. No timing acceptance interval or new five-percent gate is introduced.

Common settings are `MLStabThres=1e-9`,
`COLLIERRequiredAccuracy=1e-12`, UV/IR poles enabled and the same common settings
already enforced by the core policy. Preserve every generated physical split
order and total-order slot. Compare Born, finite, single-pole and double-pole
components in all six candidate-backend pairings using

`abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*S`,

where S is the largest absolute Born/Laurent component at that same point across
all orders and the compared cohort. Investigate every failed component without
dropping its input or output. Invalid/nonfinite outputs and unaccepted final
statuses remain failed observations. Repeatability of each command is checked
against its own bound calibration; it is not substituted for cross-backend
agreement.

## Fixed rotation and scale-change controls

Keep these controls in their own report/worker namespaces and point streams.
Every control uses all four candidate backends and two complete commands.

1. Rotate ordinary points 1 through 4 using the same rational rotation as the
   existing physical-point utility: first Rz with cos=3/5 and sin=4/5, then Rx
   with cos=5/13 and sin=12/13. Transform every external spatial momentum, retain
   energies, and round only when writing the same 17-digit momentum records.
   Check shells and conservation on the serialized points. Compare backends at
   matched rotated points, then compare the summed physical Born/Laurent/order
   values against the corresponding ordinary configurations using the same
   component tolerance. Record estimated accuracies/statuses independently;
   those need not be invariant under rotation.
2. Use ordinary points 1 and 2 with the fixed scale multiplier sequence
   `{1, 2, 0.5, 1}` for each point, giving eight ordered evaluations. Set
   `MU_R = multiplier*sqrt(s)` independently before every call, then invoke the
   existing generated coupling-update path. Retain one process/context through
   the sequence to exercise state invalidation. All physical masses, momenta
   and input parameters remain fixed. The control is scale-only: the older
   mass-and-scale rebinding utility must not be reused unchanged. Compare
   backends only at matching momenta and scales. The final restored multiplier
   1 result must agree with that point's first multiplier 1 result. Do not
   require changed-scale results to equal ordinary results.

The scale test needs a narrowly scoped driver adapter outside the frozen core
harness, derived from its existing physical state-update utility. Retain the
complete generated adapter, diff and input schedule; confirm that only MU_R and
the corresponding existing coupling update are added. It may not modify the
reduction implementation, thresholds, retry policy, helicity/color sums or
physical card. Numerical execution stays held until root reviews this adapter.

## Corrected-original references, where practical

Attempt actual untouched-original CLI exports for both processes from revision
`27147b3300f556a75661a886e92bdbe00558ee25`, in separate generated-output trees.
Construct private helper-corrected references by the existing core method:
compare original/final-candidate MODEL inventories for this exact process,
validate the known two numerical MODEL postimages against the correction for
the same UFO and mass scheme, copy only those files, and rebuild every MODEL
object/archive and subprocess object/executable. Verify all other generated
sources and original reduction/DHELAS libraries unchanged. Never alter the
literal original archive or silently correct an original-labelled output.

If generation, exact model-postimage proof or a legacy reference fails within
the bounds below, retain its complete failure. Do not replace it with a different
process/backend or claim the optional reference cohort passed. The mandatory
four-candidate cohort remains a distinct gate. When all three corrected-original
backends 1/6/7 are available, add the full seven-target ordinary comparison
(all 21 pairs). Keep optional-reference availability, all individual failures
and the four-target result visible separately. Optional references are not
required for the rotation/scale controls and never supply a performance gate.

## Runtime, numerical stability and bounded execution

Report complete warmed evaluation time for every observed command, separate
generation, compilation and initialization/warm durations, actual point counts,
split-order dimensions, final H/T/U status histograms and accepted DP/QP rescue
counts. Retain calibration data, cold-start logs, raw internal accuracy estimates
for every point/order, and observed cross-backend component discrepancies as
separate quantities. Final-status counts are not counts of internal reduction
attempts; no such claim is made. Report deterministic tested counts only,
without inferring a probabilistic instability rate.

Use one evaluation thread and one pinned CPU per comparison cohort, recorded
before launch. Root assigns that CPU when the final campaign schedule is known;
no quantitative performance acceptance is attached to these diagnostic times.
All Symbolica/MP work remains globally serialized, and these tests may not
overlap another campaign's timing quiet window. Generation/Fortran builds may
use only the root-assigned nonbenchmark CPUs.

Predeclared upper bounds are six hours per CLI export and clean build, and
24 hours per worker initialization or numerical command. There is one attempt
per stage/target, with no automatic retry, adaptive scan, seed replacement,
threshold relaxation or rescue disabling. The parent orchestration records
timeouts and unfinished targets explicitly. A failed or incomplete mandatory
candidate/control comparison remains outstanding. Root may stop an expensive
attempt while retaining the partial failure; it cannot be represented as a
completed twenty-point result.

## Generic implementation audit and report integration

Review the generated rank, propagator multiplicity, coefficient dimensions,
split-order dimensions and resource use against the public generic interface
contract. Exercise existing buffer/domain failures and record any unsupported
shape through the normal error/rescue path. No process name, PDG combination,
case identifier, diagram number or benchmark momentum may enter production
dispatch, thresholds or arithmetic. Existing generic Fortran tensor/HELAS
optimizations remain subject to their same guards and fallback.

Keep the current final-supplement-v1 schema and 87-file checkpoint unchanged.
After this plan is approved, implement a separate raw-bound stress dataset
(`madloop-final-stress-v1`) and validator. It must match the final core's clean
generator and full measured NinjaRS runtime descriptor; an identity string
alone cannot relabel another archive. The final report may accept an optional
`--stress` manifest and add one correctness/stability section with the two
processes, ordinary/rotation/scale counts, comparison maxima, all failures,
rescue/accuracy distributions and diagnostic durations. Display an explicit
“no performance acceptance gate” label. These results do not replace or relax
any of the eight core or loop-induced performance gates.

Freeze this plan/configuration first. Freeze actual cards, point bytes, derived
control schedules, source/runtime bindings and the reviewed scale adapter before
the first numerical call. No result is claimed by this source-only plan.
