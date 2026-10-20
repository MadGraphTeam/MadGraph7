# Independent quadruple-precision helper audit

These probes copy the generated EW3 helpers into isolated executables. They
do not edit production code or change model parameters, continuation
regulators, reduction thresholds, or accepted benchmark records.

`probe.py` isolates twelve missing `KIND=16` constructors in the logarithm
and square-root normalization helpers, and the inaccurate pi constant.
For a nontrivial real or imaginary axis input, the original logarithm error
is about 7.69e-9 and the square-root error about 4.27e-9 against direct
binary128 intrinsics. Corrected helpers agree exactly. Three continuation
controls expose an original 2*pi*i shift error of 1.99862e-4; the corrected
results agree with a reference using `ACOS(-1.0_16)`.

`trajectory_probe.py` changes **only** the four variable-argument constructors
`CMPLX(M**2,-GAI*M)` in the two trajectory helpers. Equal squared masses
`m12=m22=1` and external squared momentum
`p2=(4−2^-30)−i` provide an analytic branch test. The discriminant
`1−4*(1−i*1e-24)/p2` starts just below threshold and crosses its negative
real axis from above to below. Its continued square root is therefore the
negative of the final principal root. Rounding the intermediate external
mass squared to default complex moves it to exactly 4, and the old routine
selects the wrong sign.

At 100, 200, 300 and 400 trajectory segments, the original error is
0.9849581211188239 while the corrected error is zero. On the above-threshold
control `p2=(4+2^-30)−i`, both versions match the principal-root reference to
6.81e-35. The accompanying log-trajectory old/new difference is retained as a
diagnostic only; that probe does not claim an independent oracle for the
complete log continuation.

The trajectory probe also demonstrates that Fortran `REAL(real16)` without
an explicit KIND returns default REAL and loses a `2^-100` perturbation.
The MP UFO expression parser used this form in `re`, `.real`, `tan` and
`atan` expressions. This is distinct from `REAL(complex32)`, whose result
inherits the component kind.

Sources, commands, compiler version, hashes and raw values are retained in
the local result files. The root agent owns the corresponding production
fixes and compiled exporter/parser regressions.

One separate legacy limitation remains outside these corrections:
`MP_B0F`'s tree cache treats relative input differences up to 1e-14 as equal.
That approximate lookup is unsuitable as an unrestricted exact-QP rebinding
contract. The studied EW model uses expanded counterterm expressions and
does not call this cached generic B0 helper. No cache behavior was changed.
