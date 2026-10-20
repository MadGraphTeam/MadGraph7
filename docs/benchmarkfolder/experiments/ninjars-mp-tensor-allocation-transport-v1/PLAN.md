# Corrected transport plus exact allocation patch

Prepare an isolated combination of the production binary128 transport correction
from `9f13d10cc` with the exact allocation-only tensor source already validated
and measured in the two preceding experiments. This is a correctness prerequisite
for any later process experiment, not a production promotion or timing result.

1. Snapshot every tracked `vendor/ninjars` source, verify its bytes against the
   committed transport revision, and retain full hashes. Require the original
   tensor source to match the prior dense oracle. Verify the prior frozen
   allocation manifest and the exact three reused tensor/test files before copy.
2. Create a private crate under the repository's disposable work tree. Change
   only `src/tensor.rs` to the frozen allocation source and add its dense oracle
   and two test modules. Keep the corrected production parser, ABI, masters,
   precision, algorithms, Cargo manifest and lockfile unchanged.
3. Compile one serial release test executable with the existing fenv test
   feature, thin LTO and one codegen unit. Use CPU234, one build job, a private
   target/build directory and a copied/reflinked immutable dependency cache.
   Compilation requires a separate root handoff; installation is forbidden.
4. After the exclusive MP-slot handoff, run all inherited bridge tests, including
   the negative-zero/branch-lip and exact finite-dyadic transport tests, plus the
   retained tensor exactness tests. Run all21 golden lanes, the complete1,392
   request tensor-oracle replay and the production MpState replay of all1,392
   requests. Require exact11,136 final ASCII fields in both full replays; no
   skipped records or rewritten expected outputs. Compare every callback value,
   tracked precision and sign to the unchanged dense tensor oracle.
5. Retain command/environment/source/executable identities, raw logs and failed
   attempts. Recheck source and installed archive before/after every execution.
   New negative-zero inputs are covered by the inherited exact transport tests
   and synthetic tensor signed-zero controls; the existing capture's contents
   remain unchanged. No new performance series is authorized by this plan.

Estimated build:6–9 minutes of one CPU, dominated by test-crate thin LTO, with
private cached dependencies. Expected tests/replays:under a few minutes once the
sole Symbolica MP slot is available. These estimates do not cap mandatory work.
The corrected-original reference on207 remains unchanged. Main installation,
production code and all previous seals remain untouched.
