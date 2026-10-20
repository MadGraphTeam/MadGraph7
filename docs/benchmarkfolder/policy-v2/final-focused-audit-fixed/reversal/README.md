# Prepared installed-ABI3 reversal invariant

This control is prepared but not executed. It is independent of the recording
MadLoop wrapper test and of the historical ABI2 point-67 oracle.

Pinned NinjaRS `0d0d7e7a8033431b60a8580731084742e0655cda`,
`crates/ninjars/src/fast/mod.rs:66`, defines
`D_i(q) = (q + V_i)^2 - m_i^2 - mu^2`. Its tensor-layout documentation defines
the numerator as ordinary monomials in q with no symmetry factors. Therefore
the change of integration variable q → −q requires:

1. Negate every shift, then reverse propagator order and associated complex
   squared masses.
2. Permute both axes of the supplied invariant matrix; its values do not change
   under simultaneous shift negation.
3. Negate every odd-degree numerator coefficient.
4. Keep the squared scale unchanged.

No affine translation is required. In particular, the reversed first shift is
not forced to zero: doing that would require an additional numerator translation
and would test a different transformation. The helper supports arbitrary real
shifts under the documented denominator convention.

The explicit driver tests a rank-three triangle and rank-four box with distinct
complex masses, finite spacelike invariants, and dense dyadic tensors containing
nonzero odd-degree coefficients. MP coefficients also contain `1+2^-100`.
`inputs.json` records all original/transformed buffers as exact rational
strings. The driver calls native DP, explicit compensated DP and genuine MP,
then repeats original inputs on the retained context and after finalization.
Every status and all finite/single-pole/double-pole components are retained.
R1 is recorded separately as a basis-dependent diagnostic.

A fifth phase deliberately reverses the shifts and masses while retaining the
original numerator. All three engines must resolve a difference larger than
`10 × (1e-7 + 1e-10) × S`, where S is the common largest Laurent magnitude of
the correct and negative cases. This bound is fixed before execution and is
ten times an upper bound on the approved mixed comparison tolerance. It ensures
the nonzero odd coefficients contribute observably to the integral. Negative
outputs are also compared with the genuine MP result and retained in full.

The DP/compensated comparisons use the approved mixed tolerance
`1e-7 max(|a|,|b|) + 1e-10 S`; MP reversal uses the existing convergence
control's `1e-31 S` bound. The repeated/fresh checks are exact. The program also
compares DP and compensated full Laurent outputs against its genuine MP result.
Any failure is retained and rejected, without changing tolerances.

Preparation only, from the repository root:

```sh
source .ninjars-build/dev-env.sh
python -B docs/benchmarkfolder/policy-v2/final-focused-audit-fixed/reversal/control.py prepare
```

After root allocates the numerical slot and permits compilation:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
export SYMBOLICA_LOCK_DIR="$PWD/.ninjars-build/symbolica-locks"
python -B docs/benchmarkfolder/policy-v2/final-focused-audit-fixed/reversal/control.py run --cpu 5
```

The runner refuses to overwrite a prior numerical result, uses repository-local
build products, and links the installed Fortran module/archive with its recorded
GNU-runtime prefix metadata. No reference clone or production source is changed.
