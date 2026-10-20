# Final isolated complete-process allocation comparison

This implements the retained `../ninjars-mp-tensor-allocation-transport-v1/process-plan.md`
without changing its sample count, numerical settings, lifecycle, exact matching,
statistics or decision rules. Root authorized private preparation and jobs1
compilation on CPU234 at resumption. Numerical execution and timing remain HELD
until a separate explicit root handoff. Neither arm replaces an installation.

The control is the freshly CLI-installed corrected signed-zero archive with
SHA256 `832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b`.
The experimental source starts from the sealed corrected-production source in
the combined-correctness package; its sole production source change is the
minimal `production-tensor.rs` (SHA256
`240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70`).
The source has no added test dispatch or experimental runtime switch. Release
compilation uses the retained Cargo.lock and the exact installer
`cargo rustc --release --locked --lib -- --print native-static-libs` command,
with identical effective Cargo fingerprint/profile and no compiler overrides.
The unchanged manifest requests thin LTO and one codegen unit; effective parity
is established by the command and fingerprint rather than inferred from that
request. An independent cache, one build job and pinned CPU234 apply. Archive identities and the
complete source inventory are retained before process preparation.

Both process arms clone the same fresh f4b Fortran output from
`policy-v2/generation-abi3-tensor-factor-v1`, use the frozen persistent-worker
driver and exact100-point streams, common MLStab1e-9/Collier1e-12 and normal
rescue. The numerical library path is private; equal non-NinjaRS libraries,
module/header/link metadata, compiler flags, cards, sources and point bytes
are mandatory. No native Float256 change is combined with this experiment.

Exactly21 QCD3 pairs and7 EW2 pairs alternate original/allocation ordering.
Each arm has a fresh worker, completes initialization and full-stream warmup,
performs one100-point timed RUN, writes every coefficient/status/accuracy,
then STOP/BYE and confirmed exit before the next arm starts. Initialization is
retained separately. Every timed call includes the normal numerator/reduction/
masters/stability/rescue work. Every output must exactly match the corresponding
arm and other arm across all repetitions, in addition to the frozen tolerance.

The measurement CPU is selected before samples using only host activity:
three consecutive3s intervals each at most5%busy, excluding reference CPU207.
All other owned numerical/build work is held except the declared reference207.
Retain host/topology samples and every attempted preflight. One shared MP owner.
No sample is discarded/replaced. Fixed100,000 paired bootstrap draws with
`random.Random(20261020)` and linear2.5/97.5percentiles estimate the ratio of
arm medians. Primary gain requires upper95%endpoint below1; EW2 regression
gate requires upper endpoint at most1.05. No optional extension after inspection.

These are optimization decision studies, not the final literal-original
eight-process acceptance benchmark. Root alone decides production promotion,
adds durable tests, rebuilds installations and commits/pushes the result.
