# QCD stress observer qualification

The private observer reproduces the unobserved prototype exactly for both backends. The stress amplitude remains rejected with code **420**. The new trace identifies actual native refusals on finite numerators and records the common matrix construction that precedes them. No production source, threshold, physical input, working precision, or acceptance policy changed.

The frozen source design is the sibling `qcd-loop-kinematics-observer` package, index SHA256 `03dcf0b638a32edbd8362fa8cf6d12148250f276f095a53b7579e392aae4ad90`. One private build compiled exactly its six sources and linked the historical remaining objects and libraries. The observed executable is `ec4bbd29c7461dc4c8aa1ade4906d5e0ae5b364bd98e4bde83915a360740a7a4`; its loaded TIRust library remains historical `49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`. Exact compiler, command, object order, library order and hashes are in `attempt1/build.json`. The full compile log is retained. No source correction or build retry was needed.

Each backend ran once, on CPU 11 under the shared numerical lock with a 180-second cap. TIRust finished in 4.30 seconds and NinjaRS in 10.86 seconds. These are bounded diagnostic execution times, not performance comparisons. Every one of the 18 selected original output records, including six amplitude rows, agrees exactly with its frozen unobserved prototype. All before/after input and runtime hashes agree. The ordinary/stress/ordinary statuses are `219/420/219` for TIRust and `218/420/218` for NinjaRS. The named invalid/divide-by-zero/overflow flags remain `FFF/TTF/FFF` and `TFF/TFF/TFF`, respectively, exactly as in the prior unobserved runs. All six independent post-flush health acknowledgements and all trace fault counters are zero.

An independent read-only audit verified the six source inputs, 24-object ordered link, historical compiler and flags, 134 build pins/artifacts, 160 run pins, 15 runtime files, full numerical-row parity, IEEE flags and trace footers. `decode_trace.py` then validated complete payloads, scopes, call stages, error references, output chronology, and stdout/trace IEEE agreement. Its checks include 18 exact IEEE patterns and 20 malformed-input negative controls. No failed native result buffer was read.

Both traces contain every one of the 116 groups in CT4 and CT5 at the stress point. The ordinary controls retain the automatic DP path and have no QP group visits. All 232 matched QP calls have bit-identical original numerators, corners, native shifts, masses, supplied S/distances and scale between the two backends. Every captured numerator and successful native output is finite.

| Backend | CT mode | Native successes | Native unstable refusals | TIRust detailed reason |
|---|---:|---:|---:|---|
| TIRust | 4 | 85 | 31 | 12 singular, 19 ill-conditioned Gram systems |
| TIRust | 5 | 85 | 31 | 10 singular, 21 ill-conditioned Gram systems |
| NinjaRS | 4 | 106 | 10 | Public unstable-kinematics status; empty supplementary error text |
| NinjaRS | 5 | 106 | 10 | Public unstable-kinematics status; empty supplementary error text |

NinjaRS refuses groups `42,46,56,57,60,61,67,68,69,72` in both directions. TIRust refuses those ten plus `10,38,41,44,54,55,58,59,62,63,70,71,81,82,84,91,92,94,105,106,108`, also in both directions. Full call identities, statuses, error strings and finiteness records are retained in `analysis1/calls.json`; all raw data remain in each `trace.dat`.

The observed matrix builder takes only its existing massless energy-threshold branch. Per direction it clamps 178 known massless single-leg/complement pairs and **19 distinct composite pairs** carrying the soft `p1-p4` channel. Every targeted composite has both a nonzero computed QP `DIFFSQ` and a nonzero exact Cartesian invariant from the captured shifts, but the supplied native distance is zero. No mass-i or mass-j clamp occurs in this QCD case. The source-derived single/complement/composite classification is preserved, including the actual nonzero closure shift; it is not inferred from adjacency or the invariant's magnitude.

For example, group 20, CT4, pair `(1,2)` has computed `DIFFSQ ≈ -4.9999999997510091e-14` and an energy reference near `62499.975`. Their absolute ratio is about `8.0000032e-19`, below the unchanged `OSThres=1e-8`, so the existing branch supplies zero. The fixed Cartesian invariant is independently nonzero, approximately `-4.9999999997510080e-14`. Exact fractions for this and all 1,908 unordered-pair observations are in `analysis1/pairs.jsonl`.

The exact Gram audit goes beyond the small invariant. For all **22** native reports of singular Gram systems, the determinant of the supplied matrix is exactly zero, whereas the determinant derived from the fixed captured Cartesian shifts is nonzero. Every one of those reported subsets contains a clamped target composite. For all **40** nonzero ill-conditioned reports, the reconstructed source-defined condition indicator agrees with the native reported value to relative error below `1.914e-63`. Among the 20 common backend refusal visits, 18 reported TIRust subsets contain the target. The other two are CT5 groups 69 and 72: their first reported TIRust failure is a different ill-conditioned subset. These distinctions are retained in `analysis1/gram.json`.

This establishes an observed change in the integral input and exact Gram degeneracy, plus actual native refusal status. It does **not** establish that changing the classification restores an accurate amplitude. NinjaRS provides no detailed internal failure reason, the Cartesian comparison changes several supplied invariants, and nonzero Gram systems can still fail a conservative stability gate. The earlier Born B/C consistency replay also showed that momentum repair changes this tiny physical channel and its Born normalization. A future policy must resolve that representation issue separately from deciding which topological channels may be treated as on shell. No new scalar/tensor evaluation, matrix replacement, threshold reduction, or working-precision change was performed here.

From the repository root, verify the portable inventory and complete traces with:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native/verify.py
```

`--live` additionally checks the retained native/source paths against execution hashes. `build.py` and `run.py` refuse to replace their one-shot attempt. Exact offline analyses can be reproduced into new output destinations with `analyze.py --output <new-directory>` and `gram.py --output <new-json-file>`; neither runs a native reducer. The original raw failures and the source-only preparation attempts remain unchanged.
