# Independent fixed 21-pair confirmation

This is a new, fixed-length confirmation of the same QP FFV derivative
factorization. The completed seven-pair series remains sealed and inconclusive.
Its samples are not reused in the primary result of this series. Production
source and the final benchmark dataset remain unchanged.

Before execution, the runner writes the exact protocol to the new run directory.
It uses all 100 retained QCD3 points, identical parameter cards, original Ninja
backend 6, `MLStabThres=1e-9`, and `COLLIERRequiredAccuracy=1e-10`. Each repetition
runs both variants on one preflight-selected physical core, reversing their
order on alternate repetitions. All generated Fortran and build recipe files
retained from the first seven-pair run must match byte-for-byte before compilation.
The helpers are uninstrumented. No new arithmetic optimization is included.

The coefficient gate is
`abs(a-b) <= 1e-7*max(abs(a),abs(b)) + 1e-10*point_scale`, where the scale is the
largest absolute Born/Laurent coefficient among both variants and every split
order at that point. Every output and accuracy estimate must be finite, every
physical split order must be present, and every status must identify a stable
original Ninja DP or QP result. Accuracy estimates and rescue counts are retained
and compared; added rescues require investigation. Bit equality is reported when
observed but is not required by the changed-association numerical contract.

The primary statistic is the ratio of the two medians from exactly 21 paired
repetitions, with 10,000 paired bootstrap resamples and seed 410083. A demonstrated
improvement requires the upper endpoint of the 95% interval to be at most 1.0.
All samples remain in the result. There is no timing-dependent stopping,
selection, or further adaptive rerun without a source change. Any pooled
28-pair analysis is secondary and must retain the campaign/core distinctions.

The quiet controller records topology, CPU affinity, current frequency and
five-second activity samples for the selected core and the allowed reference
CPU207. Other task-owned numerical and compilation work is paused during the
series; the corrected-original reference retry may continue on CPU207. This is
a coordinated local window on a shared host, not exclusive host ownership.

After root review and a recorded idle-core preflight, run from the repository:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/symmetric-factorization/confirmatory21/quiet_factor.py \
  --run-id qcd3-factor-confirmatory21-v1 --cpu SELECTED_IDLE_CPU
```

The new namespace is excluded from the completed campaign seal. The result
remains `acceptance_eligible=false`; it cannot directly populate the final
cross-backend benchmark or justify a production change without the remaining
independent reviews and process controls.
