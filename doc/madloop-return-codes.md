# MadLoop return-code format 2

Current generated MadLoop code returns `RET_CODE = 100*H + 10*T + U`.
`ML_GET_RETURN_CODE_VERSION()` returns `2` so standalone and OLP callers can
identify this convention at runtime.

The units digit `U` identifies the reduction backend:

| U | Backend |
|---|---|
| 0 | Unidentified |
| 1 | CutTools, double precision |
| 2 | NinjaRS |
| 3 | IREGI |
| 4 | Golem95 |
| 5 | Samurai |
| 6 | Ninja, double precision |
| 7 | COLLIER |
| 8 | Ninja, quadruple precision |
| 9 | CutTools, quadruple precision |

The tens digit `T` keeps its existing meaning: `1` means only double precision
was used, `2` means quadruple precision was used, and `3`/`4` have those same
precision meanings during initialization. It describes the evaluation of the
point, including rescue attempts. The hundreds digit `H` records stability:
`1` unknown, `2` stable, `3` rescued, and `4` exceptional/unrescued. The counters
interpret precision through `T`, independently of `H`. For example, `212` is
a stable NinjaRS result using only double precision; `322` is a rescued
NinjaRS point for which quadruple precision was used.

Backend selection and return-code digits are separate conventions.
`MLReductionLib=8` still selects NinjaRS. PJFry's selection `MLReductionLib=2`
is retired and produces an actionable error; it is not an alias for NinjaRS.

## Compatibility and saved statistics

This is an intentional reassignment of PJFry's return-code slot. In format 1,
`212` denoted PJFry; in format 2 it denotes NinjaRS. The earlier experimental
NinjaRS mapping, DP `(H,U)=(2,8)` and QP `(3,2)`, is also superseded. External
consumers should query the format version and interpret `U=2` as NinjaRS in
format 2, using `T` for precision.

New run-statistics XML contains `<return_code_version>2</return_code_version>`.
NLO logs contain `#ReturnCodeVersion = 2`. Their unit-2 histogram is entirely
NinjaRS. Dedicated DP/QP counters split that bucket directly, without backing
NinjaRS out of another backend's counts.

The MG7 statistics readers continue to read unmarked historical files as
format 1, including genuine PJFry counts and the earlier optional NinjaRS
alias counters. Each input is decoded before aggregation, so old and new
runs can be combined without relabeling historical data. Unknown format
versions and inconsistent format-2 precision counters are rejected.
