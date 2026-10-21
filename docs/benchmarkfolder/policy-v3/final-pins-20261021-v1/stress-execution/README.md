The two additional challenging processes were freshly generated through the
actual MadGraph CLI and built from committed source
`856e76e669618ec8d0f799b9c8e6f86819dcd986`:

- `g g > t t~ g g [virt=QCD]` with `loop_sm`;
- `u u~ > e+ e- mu+ mu- [virt=QED]` with `loop_qcd_qed_sm_Gmu`.

The complete generation record is `../stress-generation/final-result.json`.
Both exports use generator tree
`77b16df0251af4104a2c541bab913c11f4e67496a290495ce1a1c5fc526dfe3c`,
installed archive
`b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c`,
and the independently observed full runtime
`a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008`.
The actual ten-file installed image and every archived bridge source are
checked against the final CLI-update publication. NinjaRS is pinned to
`aa6a6307dd081dc3714f9accebc0df9e5561ae73` and oneloopmaster to
`8ec256fc26cb361b20e10f59c19beb35717ef32c`.

`prepare_final.py` reuses the unchanged retained `early_build.prepare()`;
only installed-publication validation and subprocess ownership are adapted.
CLI and make concurrency were reduced to one job, with controller affinity
CPU6/7. Each `check --masses` call only obtains external masses. There were
zero amplitude evaluations and zero initialized Symbolica workers. All six
CLI/build/masses process groups were reaped. `terminal.json` records the tool
exit and subsequent process-identity absence; CPU6/7 and the source-HEAD hold
have been released.

The first-retained ordinary20, rotation4 and scale8 momentum streams, their
seed/mass/energy metadata, fixed variant configurations, physical parameter
cards, and scale schedule match byte for byte. These are input-generation
checks, with no scientific or runtime-acceptance claim.

`collect_final.py` retains the original four-backend stress collector as a held
extension, using registered subprocess ownership and the final installed
publication. `collect_nrs_first.py` prepares a separate raw-only specification
with explicit targets: NinjaRS8 for ordinary points; Ninja6 and NinjaRS8 for
rotation and scale controls. Each selected specification binds an immutable
base specification and the exact wrapper sources. A resealed change in the
requested backends is rejected. It does not satisfy or weaken the original
four-backend dataset reader. Ordinary legacy reference rows require a separate
strict provenance and numerical audit, and keep their actual original source,
runtime, parent status, and calibration-only classification. Missing independent
ordinary references remain outstanding controls.

Fresh NinjaRS numerical collection remains held until the root controller supplies the
explicit resource handoff after the final core run. No numerical specifications
were written at generation completion: their orchestration HEAD must be bound
after the next reviewed milestone commit. Every evaluation requires CPU190,
one thread, complete poles and the unchanged stability thresholds. The retained
workers, amplitude path, scale-input adapter, and physical configurations remain
unchanged.

`collection-adapter-checks.json` binds nine passing software checks and the
read-only fresh-export reader result. These validate source freezing, rejection
of unexpected numerical inputs, unchanged numerical function objects, dispatch
restoration, and target/base-specification integrity. They execute no compiler
or numerical engine. The earlier six-check logs are retained separately; they
are not an additional scientific sample.

The root subsequently authorized separate legacy-only correctness controls on
CPU12. `collect_legacy_cpu12.py` permits only EWordinary20 tools1/6/7 and both
rotation4/scale8 Ninja6, with no NinjaRS numerical evaluation or Symbolica context.
Normal `EXIT_MADLOOP` teardown may call `destroy_context(NULL)`; this allowed
no-op creates no context and evaluates no integral. These controls completed
with the preserved EWordinary Collier7 point19 stability rejection. Accepted
CutTools/Ninja calibrations, all rotation/scale commands, independent audits,
and exact scopes are curated in `../legacy-controls-capsule-v1`. All three
controllers and their children were reaped; CPU12 was released. Their traversal
times are excluded from final performance acceptance.

Both new scoped collectors allow unrelated orchestration commits only while
exact numerical source856e, generator77b, archiveb0d and every bound imported
helper remain unchanged. They append the actual observed HEAD; retained/default
collectors retain their original guards. `nrs-guard-v1-checks.json` records
thirteen fresh software checks and independent source review of the held
NinjaRS wrapper. Its exact pre-guard source is retained under
`source-pre-nrs-guard` with the original552aee authority, preserving the earlier
nine-check proof rather than rewriting it.

All paths below are relative to the MadGraph repository. To repeat read-only
validation in the pinned development environment:

```sh
export TMPDIR="$PWD/.ninjars-build/tmp"
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/stress-execution/check_collection_adapter.py
python docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/stress-execution/collect_final.py check \
  --early-result docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/stress-generation/early-result.json
```

The generation procedure refuses existing output/work directories. Preserve the
current attempt when reproducing and select a new namespace; do not overwrite
its source proofs, first-retained streams, or logs. Compiled programs and the
large reduction archives stay in ignored build directories inside MadGraph.
