# Additive current-trace build admission

This packet leaves source index `7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed` and derived index `25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184` unchanged. It is a narrow wrapper around the sealed controller's **build** stage. No generation, ordinary/stress stage, Fortran main or amplitude call follows it.

Before importing the controller it verifies both complete packets and the approved generation receipt `73d32bc26ee0641faaf90c7b1060471c2339a64d9dbc33d6f064356b106a093a`, then requires its raw source inventory to equal the sealed CMS bindings. The retained EW compilation receipt is hash-checked against CONTEXT before parsing its compiler/archive fields. Original CMS/EW source inventories, module inputs, both shared configurations, all generation dependency pins, retained EW archives, and pinned source/model/card context are checked before and after. Guard receipts are append-only under `results/build-admission-1`; failure never triggers a retry.

`bindings.json` identifies the actual retained compiler, its `nix-support/dynamic-linker` declaration and the declared loader bytes. It also proposes two explicit supplemental runtime provider pins: that loader and the exact `chqq8…` GNU 15.2.0 libquadmath seen in retained CMS evidence. The historical prelink omitted that particular DSO while admitting differently hashed quadmath providers. The evidence is pinned; basename equivalence is never accepted. These supplemental pins are part of the packet requiring root review, not discoveries automatically accepted during a build.

The wrapper replaces only the in-memory `L.runtime` callback after the sealed controller initializes its inherited owner. For exactly four expected build executables (CMS/EW × unmodified/instrumented), it checks actual PT_INTERP against the declared loader **before** the original owned `--list` query. It then requires every returned provider's full resolved path and hash to match the per-link prebinding or the explicitly reviewed supplement. Rejected results are retained. Queries retain the existing shared lock, CPU12–13 nonexclusive affinity, stage3000s limit/60s cleanup reserve, process ownership and output bounds. This wrapper adds no subprocess launcher. A build admits loader `--list`, not amplitude execution.

The write/recipe audit found all commands operate in fresh private clones. MODEL/DHELAS archive targets are private `../../lib/` paths. The CMS MODEL force-build can touch the regular private empty ldme.inc, onia_read.inc, run.inc and maxparticles.inc files and rewrite private cuts.inc to its existing generated text. The process `check` target only links. Vendor builds in the unused top-level Source makefile, OLP/f2py targets, cleaning, reference repositories, installation and production edits are outside the command path. Existing dependency aliases are read-only inputs verified by the sealed clone admission. The guard's pre/post checks further protect original source inventories and dependencies.

After independent review and sealing, read-only preflight is:

```
source tools/tirust_env.sh
python3 -B .tirust-build/kinematic-provenance-current-executor-1/build-admission-1/guard.py --reviewed-guard-index <INDEX_SHA256>
```

Only a separate root admission authorizes adding `--execute` to that command. The in-process controller argv is fixed to `build --execute --reviewed-source-index 7ea709… --reviewed-derived-index 25c211…`. Ordinary/stress remain separately gated. Any later worker admission must bind its actual providers before launching the worker; this packet only qualifies build-loader admission.

Eight pure tests cover hash-before-parse, generation identity/inventory mismatch, exact supplemental provider paths/hashes, pre-query PT_INTERP rejection, rejected-provider retention/no retry and passing bounded provider admission. All test loader callbacks are Python fakes; no native executable is loaded. Default preflight performs file reads and imports only. No numerical cards, thresholds, mass policy, arithmetic, mode1/4/pole0/force-false contract or one-bootstrap/one-measurement protocol is altered.
