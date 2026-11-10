# QCD private 512-bit replay with complete runtime capture

The single corrected replay completed its protocol in 5.25684 seconds on CPU 11.
The observed ELF mappings contain the intended private 512-bit library, SHA256
`e8b65f86b78f77d8090e7c406274da32920d89279ee4d5d3ef40e518fb05afdc`,
and the executable's actual PT_INTERP loader. All protected inputs, initialized
resources and runtime bytes match before and after execution. The child was
reaped and the shared numerical lock released before the next test lane.
There was no build, automatic retry, installed-library replacement or production
change. The inclusive 180-second cap is administrative, not a performance test.

The full raw trace and selected amplitude/status/IEEE rows are byte-identical
to the retained first attempt, whose runtime capture failed. That earlier
package remains incomplete and unchanged. This new process independently
provides one successful loaded-library and leader-affinity snapshot; it is not
continuous runtime monitoring. The corrected source/controller index is
`69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4`.
The inspection and loader correction remain in their separate frozen package.

The ordinary/stress/ordinary host statuses are 219/323/219. Both ordinary
results, repeated raw words and baseline flags are exact. All 232 stress tensor
calls—116 groups in each independent direction—report success; the historical
256-bit library served 181 and refused 51 on these same topology-C inputs.
The trace acknowledgements contain zero faults and zero I/O failures. Stress
sticky IEEE invalid/division-by-zero/overflow flags remain T/T/T, exactly as
before; this replay did not clear or reinterpret them as an IEEE-clean result.

The offline comparison retains all 696 complex contracted Laurent slots
(1392 real components) against NinjaRS, at identical effective C coordinates,
numerators, propagator shifts, supplied invariants, historical promoted-DP
mass squares and native scale. Both recorded scales are already mu squared
and equal 1,000,000; the earlier provisional script's extra squaring was wrong
and is not used here. The raw finite/single/double host relative differences
are respectively 7.243e-16, 4.129e-16 and 2.920e-16. The analytic double-pole
prediction using the same C-point Born has relative error 1.03749e-15.

All 46 slots exceeding the 3e-6 component/complex relative criterion remain
reported: 2 finite, 30 single and 14 double. No small component was discarded.
Their absolute discrepancies and cancellation context matter: across the six
direction/Laurent cohorts, the sum of absolute differences in actual DP-projected
real contractions divided by the host amplitude is 2.86e-14 to 1.62e-13, and
divided by the corresponding group L1 sum is 3.26e-16 to 1.13e-15. These
unaveraged group ratios describe magnitude and cancellation; they are not an
independent bound on the final assembled amplitude. Exact squared complex
norms are serialized for the 46 flagged rows; the remaining 650 are exactly
reconstructible from the retained rational real and imaginary components.

This is a private topology/C-Born plus 512-bit arithmetic/admission experiment.
The original 256-bit guard remains unchanged; 512 uses 2^-(512-120), preserving
the same arithmetic-headroom heuristic with a changed raw threshold. Precision
and admission changed jointly. No conclusion isolates primitive precision as
the sole cause. C is the existing repaired effective point, not the unchanged
original Cartesian input. The private graph classification and C-Born handling
are not production features. The historical promoted-DP mass convention and
its known residual discrepancy remain; no general electroweak or complex-mass
claim follows. No adaptive mixed 256/512 dispatch was used by this replay.

NinjaRS comparison covers these contracted numerators, not all tensor basis
components. Only the double pole has the stated independent analytic check;
finite and single-pole agreement is a cross-backend diagnostic, not independent
truth. The host's status 323 and this complete protocol capture do not promote
the result to general physics, production or benchmark acceptance.

The authoritative arithmetic uses the unchanged frozen `compare2.py` and
`summarize.py` from `qcd-topology-rescue-higher-precision-comparison`, whose
earlier provisional results and independent review remain preserved. Here
`comparison.json` records complete capture and `comparison-summary.json`
retains all slot errors and magnitudes. `audit-result.json` binds the new
execution and actual map snapshot. The package contains only text evidence;
compiled files are identified by hashes.

From the MadGraph repository root, `python -B <this-folder>/verify.py` checks
the exact inventory, authenticated dependencies, captured mapping/record
consistency, old full-trace parity and all offline comparisons. Add `--live`
to rehash the 381 protected source/resource/runtime paths. It never executes
a native process or loads a shared library.
