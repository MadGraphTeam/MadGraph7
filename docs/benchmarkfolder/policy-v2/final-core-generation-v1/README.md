# Held final core generation and build procedure

Preparation only: no CLI export, compiler, amplitude, numerical initialization or
new worker has been started from this directory. Root must first finish the last
optimization decision, commit and push the chosen implementation on
`madloop_ninjars`, and validate the final installed archive in
`HEPTools/ninjars`. The existing literal baseline and helper collector are not
modified by this procedure.

The new generated output root is
`.benchmark-work/policy-v2/final-core-generation-v1/outputs/candidate`. All eight
exports are fresh actual `bin/madgraph --plain` executions, using the unchanged
`policy-v2/cases.json`, original physical parameter cards, retained point streams
and model provenance. Generated command cards under `commands/` show the exact
intended CLI invocation. The frozen benchmark helper writes these same cards
again when it actually runs. The EW UFO comes from the independently verified
model inside the original archive; the built-in QCD model is `loop_sm`.

After source freeze and release of the current quiet measurement window:

1. Root provides the exact final commit and archive SHA256, confirms the matching
   installed Fortran module/header, Cargo manifest/lockfile and link metadata,
   and selects coordinated generation CPUs excluding reserved CPU 207. Keep HEAD,
   numerical sources and installation fixed until this generation round ends.
2. Copy `freeze.template.json` to `freeze.json`, fill its exact commit, bridge
   source-inventory digest, ten installed-asset hashes and selected CPU list, and
   set both release flags true. The template contains no anticipated revision or
   archive identity. `make_freeze.py` can create this file from root's explicit
   commit/archive inputs after checking them against the actual installation.
3. Set `GENERATION_CPUS` to that same list and run `commands.sh`. Two independent
   cases may export/build concurrently; each uses the unchanged frozen
   `bench.generate` and `bench.build` functions and their original make `-j4`
   command. All descendants inherit the explicitly selected CPU set. Source
   library compilation can occur inside CLI export, so export is included in
   the compiler embargo until release. No automatic CPU selection occurs.
4. The independent summarizer and runtime audit then verify all eight successful
   builds, exact cards, source/runtime inventories and original preservation.
   Multiple runtime groups require representative proof coverage for every group;
   a convenient summary is not permission to omit one. Controls and loop-induced
   exports remain separately required in the supplemental campaign.

The runner refuses existing outputs or an existing status file. An interrupted
or failed attempt must be retained; use a newly reviewed namespace for a retry.
It verifies the exact committed HEAD, clean generator and bridge source scopes,
bridge inventory, all installed hashes and coordinated affinity before each
case and after compilation. The frozen generator's own before/after tree checks
remain active. Provenance captures actual generator tree identity, full tracked
bridge bytes, complete dependency hashes, actual installed ABI/precision/link
settings, compiler, command cards, logs, generated and compiled source manifests,
physical cards/points, exact MODEL correction deltas and GNU quad-runtime imports.

Each fresh MODEL postimage must match `model-correction.json` exactly. Every
physical card must match its retained original before it is copied verbatim.
Masses and dimensions come from the existing Fortran driver's `--masses` path,
which exits before `FORCE_STABILITY_CHECK` or any loop-amplitude call. This checks
build and parameter loading only; it initializes no NinjaRS/Symbolica engine.

The literal original source at revision
`27147b3300f556a75661a886e92bdbe00558ee25` is compared byte for byte against a
streamed `git archive` before and after the round, without extracting or writing
to the archive. Existing original/candidate exports and retained global
inputs/data/points are hashed before and after. Scratch, executables and objects
stay under `.benchmark-work`; only explicit source, commands, manifests, logs and
reproducible evidence are candidates for retention. Never stage scratch trees.

Generation and compilation durations are separate preparation measurements,
potentially concurrent, and cannot certify warmed performance. Cold QCD4 legacy
initialization remains a substantial independent cost: the earlier interrupted
CutTools helper used 2h19m CPU before READY, while the frozen worker requires a
full accepted 100-point warm pass. Its logs reveal no iteration progress, so
there is no justified finite remaining-time estimate. The fresh full-eight
helper campaign continues unchanged on CPU 207. Neither this generation round
nor the historical partial report passes any numerical or performance gate.
Literal-original QCD4 cold time has not been measured for the final campaign;
the helper observation must not substitute for that distinct measurement.
