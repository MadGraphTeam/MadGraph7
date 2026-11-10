# Standalone matrix-policy validation preparation

This prospective native test extracts four exact subroutines from frozen
consumer7d002155: the legacy QP matrix builder, actual QCD identity lookup,
physical external-mass getter and new policy builder. No reducer, DSO, amplitude,
license service or model initialization library is linked. Standard Fortran
runtime support is required. A second table is generated from the actual QCD
process with external leg4 marked off shell; its original input model and
classifier remain unchanged.

The same bounded test is compiled once against the retained real-model
mp_coupl.inc and once against the retained CMS model declaration file. The test
initializes the existing real MZ symbol explicitly. CMS mode also initializes
the existing complex pole symbol and uses its square for internal denominator
masses. Thus the external physical target and internal complex pole are distinct
and their imaginary subtraction is checked. These deliberately fixed values
test transport/policy; they are not a model-card or physical-amplitude result.

The 38 retained composite cases use exact binary128 shifts from topology-c's
backend9 trace: all19 targeted pairs in each direction. The oracle retains both
an independently computed rational Minkowski invariant and the original
observer's computed DIFFSQ words. The latter must remain nonzero and bit-exact
through policy1 when internal masses are zero. Constructed fixtures distinguish
near-shell massless/massive targets, actual QP model mass beyond binary64,
off-shell descriptors, invalid group/direction, an unrelated NaN shift,
negative/nonfinite external masses, failed complement closure, and a soft leg
beside a hard common shift. The extreme-common-shift case exposes historical
normalization overflow while leaving the selected pair invariant finite.

Every case compares policy0 bit-for-bit to a separate call to the original
legacy builder over the complete matrix. Policy1's selected result is checked
against a fixed intended target or the computed value, and that computed value
is checked against the independent rational oracle. No expected admission
decision is calculated by calling the consumer's predicate. Halting modes are
recorded and required unchanged; the driver does not alter traps or clear
arithmetic flags. Invalid-input computations may retain existing exceptions.

No source is compiled or executed by prepare.py. After source/command review,
the one proposed job runs on CPU13, one thread, with a combined hard120-second
process-group deadline. It compiles at -O0 -g -fcheck=all -fbacktrace and executes
the two standalone binaries once each. A failure preserves logs and prevents
automatic retry. Source inputs, compiler and frozen consumer are hashed before
and after. No shared numerical/reducer lock is required because no reducer runs.

From the repository root, after `source tools/tirust_env.sh`, preparation is
`PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/general-kinematics-consumer-validation/prepare.py`.
The separately authorized execution command is the same with `run.py`.
Success would qualify only these matrix-policy fixtures, not general CMS,
Born-coordinate consistency, reduction status, amplitudes or performance.
