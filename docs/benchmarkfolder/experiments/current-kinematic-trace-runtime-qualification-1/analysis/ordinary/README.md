# Four admitted ordinary trace decodes

Root admitted pure decoding of four closed enabled traces after independently
checking run closure `56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf`
and unmodified/disabled/enabled parity. Root also authenticated GNU 15.2.0,
ELF64 little-endian x86_64 (`e_machine=62`) and the 113-bit/128-bit format
assertions. The explicit QP word-order choice is recorded in `receipt.json`.
The sealed decoder `df0c1113…` was checked before and after use and unchanged.

All four pure Python decoder invocations exited zero. No compiler, library or
native numerical program ran. No other current run artifact was consumed;
the previously pinned derived sources were rechecked by the sealed decoder.
The exact admitted trace hashes, commands, byte counts and output/log names are
in the receipt; the output JSONs retain every decoded word and exact ratio.

| Ordinary trace | Selected calls | Recorded native status | Exact DP-squared mass transport matches | Exact routed-vector matches |
|---|---:|---|---:|---:|
| CMS point1 mode1 | 5 | 0 for all 5 | 5 | 5 |
| CMS point1 mode4 | 5 | 0 for all 5 | 5 | 5 |
| EW point1 mode1 | 1 | 0 | 1 | 0 |
| EW point1 mode4 | 1 | 0 | 1 | 1 |

All 12 calls expose residual, matrix, native-input and native-return boundaries;
none has an unavailable rational analysis. Status0 is an observation, not an
independent integral/amplitude accuracy certificate. Modes2/5 and exceptional
points remain outside this analysis.

For CMS, the captured conservation residual is exactly zero in both modes.
The actual pair(2,4) vector exactly equals the captured-word identity
`p1-p3-r`, and no threshold branch is taken for that pair. Native invariant
minus captured RAW real is zero. The difference from an exact rational square
of captured shifts is approximately −3.45982e−11 in DP and −1.35016e−29 in QP;
these are arithmetic observations, not relative accuracy failures.

For EW, pair(1,2) is `p3+r`. The DP captured residual has spatial entries
approximately +1.06581e−14 and −5.32907e−15. The actual DP shift difference
minus the exact captured-external-word identity is
`(−2.8421709430404007434844970703125e−14, 0,
1.0658141036401502788066864013671875e−14, 0)`.
QP instead matches that routed-vector identity exactly while retaining a
nonzero residual component about −1.54074e−32. No residual was set to zero by
the decoder.

The original EW `MASSI` branch is taken at pair(1,2) in both modes. The final
native invariant equals the real part of captured DP-squared mass1, widened
without changing its value in QP. Its QP
native invariant differs from the captured RAW real by approximately
−8.84484e−13; the matrix masses equal the captured already-squared DP inputs
exactly. This documents the original on-shell projection onto those mass
inputs; it does not propose a mass or threshold change.

Across all 144 ordered off-diagonal pair observations, the 92 taken branches
are 80 CMS ZERO plus EW MASSI 4 / MASSJ 4 / ZERO 4. All 432 observed branch decisions
agree with the decoder's ideal rational predicates on these captured ordinary
inputs. Those predicates explicitly omit Fortran intermediate rounding and
cannot override the original trace or establish a general threshold theorem.

Recompute the summary from these decoded outputs only:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B .tirust-build/kinematic-provenance-analysis-ordinary-1/summarize.py
```

An independent read-only review re-decoded all 3,620 scalars from the derivative
record words and checked the relations, mass values and branch tags. It did not
read other raw traces or perform native work.

The summary and full outputs preserve exact values. No physics acceptance,
timing result, stress qualification or completed-goal claim follows here.
