# Separate four-tool single-helicity timing producer

This is software/source readiness, with no new numerical measurements. The producer reuses the existing normal Worker and identity-owned cleanup. Its private driver changes three public calls to SLOOPMATRIXHEL_THRES and adds one fixed helicity parameter. The existing timer, full split-order/color computation, normal stability/rescue and status accounting remain.

The held spec covers the eight core processes and ggH/HH with all four backends, fixed common physical helicities,100 unchanged points, full poles and eight balanced repetition rounds. Calibration is separately bound and never counted as a timing sample. A failed Collier status remains failed physics even when its raw elapsed cost is available. The reader replays raw parsing/count checks and rejects missing, extra or calibration-derived samples.

Twenty-two distinct software tests pass. Synthetic fixtures and mocked workers invoke no compiler or numerical engine. Actual metadata-only ten-case source/runtime preflight also passes. The original source/audit/spec snapshots remain, including an initial32-method suite with inherited duplicates; subsequent suites use22 distinct tests.

The final production/full-helicity Ninja6/NinjaRS8 gate remains separate and pending. Numerator-only DP construction and total-minus-construction remain explicitly pending; no pure reduction timing or final profiling claim is made. The independent DP-stability producer has its own diagnostic authority. Root must authorize all later builds and the quiet numerical resource handoff.
