# Genuine original compilation metadata compatibility

This version retains the root-authorized compatibility adjustment to the NEW
explicit timing-pair collector only. Existing workers, policyv2, numerical
sources, archived originals, cards and origin metadata remain unchanged.

The actual literal-original exports predate modern source_identity and
compiled-source digest fields in compilation.json. They retain genuine driver,
executable, compiler and make-settings records plus original generation and
observed-source manifests. The collector accepts those old fields only for
literal-original roles after checking actual generation, compilation, source
payload, driver and executable bytes. It never fabricates missing fields.
Final candidates still require modern source identity and compiled-source hash.

Observed source manifests also carry historical recording/time/scope annotations.
These are retained unchanged; comparison uses their exact files payload and
recomputed files_sha256, rather than mistaking annotations for numerical source
differences. Any real source-file drift remains rejected. Existing source-role
and original-archive anchors, full numerical runtime and physical cards still
apply independently.

Attempt1 and sources/ retain the initial22-test old-schema adjustment. Attempt2
and sources-attempt-2/ retain the final22-test adjustment including actual old
manifest annotations. Both passed onCPU4; prior owner iterations and failure
logs remain unchanged in their earlier namespaces. The three focused additions
cover genuine old-record acceptance with no mutation, strict rejection of a
candidate missing modern fields, and rejection of observed source drift.

The final code also passed actual read-only preflight against all16 origins
for the eight fixed processes, original and candidate runtime groups, physical
cards/100-point streams and recomputed representative qualification. That held
spec and compiler-comparability audit are retained separately in
../timing-pair-plan-v1. No configure, compiler, generation, model-import command,
amplitude, MP evaluation, handoff or launcher was executed/created.

Independent review receipts bind the source versions they actually reviewed.
The final payload-comparison follow-up is separate from the earlier whole-dict
compatibility review. No source/test change follows the final actual preflight.
