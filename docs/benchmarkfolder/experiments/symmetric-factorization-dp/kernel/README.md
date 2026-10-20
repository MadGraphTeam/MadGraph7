# Isolated binary64 kernel timing

This experiment measures four pure-Fortran updater variants on source-derived, exact-oracle-validated stored binary64 inputs. It does not use a captured MadLoop call stream and makes no generated-process or integration performance claim. Production DP code is unchanged.

All 36 fixture configurations completed all 21 fixed blocks: 3,024 timed arm samples and 15,120,000 updater calls, plus validation and warm-up. Inputs, cumulative IEEE flags and expected outputs passed throughout. Original, guarded-original and factor-only outputs matched their respective retained exact-oracle output bits. A-zero-only matched the original numerical output; zero-sign differences were permitted and recorded by the initial validation.

Factor-only/original CPU-time median ratios range from 0.8267 to 1.1310. Thirty-four fixture confidence intervals lie below one and two lie above one. The losses are retained: NLO FFV1L1_2 rank3 independently rounded fixture (1.1310), and NLO FFV1L2_1 rank2 independently rounded fixture (1.0036). Admission-only overhead is 13.3–26.3%. With that admission held constant, factor-only/guarded-original ratios are 0.7141–0.9562. A-zero-only is slower than original in 35 of 36 fixtures; its separate comparison prevents attributing zero-skipping effects to factorization. CPU and wall measurements reach the same gain/loss classification. These are pointwise intervals, without a multiple-comparison correction or a process-weighted aggregate.

## Fixed protocol

The 12 builds cover standalone/NLO flags, FFV1L1_2/FFV1L2_1 and full ranks2–4. Each build runs cases1,2,7 from the sealed correctness bit dumps: dense phase-related rows, sparse phase-related rows, and independently rounded actual ALOHA expressions. They are transparent source-derived fixtures, not measured frequencies of physical wavefunctions.

The four arms are original; guarded-original (factor admission followed by unchanged original accumulation); factor-only (unchanged guarded derivative-row factorization, no A-zero pruning); and A-zero-only (the same admission followed by original accumulation plus an A==0 skip). Public original fallbacks remain in all guarded arms. Trace counters are removed for timing. B0 accumulation remains unchanged in factor-only.

Each fixture receives 100 warm calls per arm, then exactly21 blocks of 5,000 calls per arm. The predefined eight-order schedule alternates forward/reverse and cyclic orders; its exact rows are in plan.json. Every sample is retained. No iteration calibration, adaptive stopping, failed-sample removal or rerun selection occurs. Statistics are medians of21 within-block ratios with percentile95% paired-bootstrap intervals (10,000 resamples, seed20261020). CPU_TIME and SYSTEM_CLOCK wall time are reported separately.

Compiler: gfortran from the retained dev environment, exact version in compiler.log. Standalone flags are -O3 -w -fPIC -ffixed-line-length-132; NLO additionally uses -fno-automatic. No fast-math or LTO is enabled. Each input run uses nearest rounding, disabled traps and an already-set sticky INEXACT flag; the independent frozen correctness campaign covers inadmissible conditions and fallback behavior.

**Array-shape limitation:** these kernels retain LOOPMAXCOEFS=210 from the broad rank2–4 oracle. Generated QCD3 uses70. Inactive coefficients are zero, but output clearing and memory layout still use210. Consequently these ratios cannot be treated as actual QCD3 kernel or full-process ratios. A subsequent process experiment must use its actual generated shape and separately assess stability/rescue effects.

## Host admission and retained failures

A five-second preflight requires target idle fraction at least90% and each exposed SMT sibling at least80%. CPU4 (0% idle), CPU6 (0%) and CPU209 (75.6%) failed before any sample was taken. Their plans, copied generated sources, preflight snapshots and failure logs remain intact. CPU212 passed at100% idle with no additional exposed sibling; all measurements ran pinned there. Other task-owned work remained on separate logical CPUs, so this was not an exclusive whole-host run. No process was terminated to obtain a CPU. Rejected preflights did not change arithmetic, fixtures, schedule or loop counts.

`kernel-source-fixtures-v1` contains the original compilation; the three CPU-specific namespaces copied those sources/binaries without rebuilding. Each plan records its preparation origin and executable hashes. Executables/object files are reproducible build products excluded from the checksum package; their recorded hashes remain in plans. The completed run is `kernel-source-fixtures-cpu212-v1`.

## Results

Each cell below is the CPU-time median ratio with its pointwise95% interval. All wall-time samples and intervals, per-block ratios, absolute seconds, fixture zero counts and hashes are in the completed run's analysis.json.

| Build / vertex / rank | Fixture | Guard/original | Factor/original | A-zero/original | Factor/guard | A-zero/guard |
|---|---:|---:|---:|---:|---:|---:|
| standalone-FFV1L1_2-rank2 | 1 | 1.2343 [1.2307, 1.2352] | 0.9140 [0.9130, 0.9150] | 1.3722 [1.3707, 1.3734] | 0.7408 [0.7398, 0.7421] | 1.1132 [1.1108, 1.1141] |
| standalone-FFV1L1_2-rank2 | 2 | 1.2325 [1.2318, 1.2354] | 0.9814 [0.9810, 0.9826] | 1.2763 [1.2756, 1.2767] | 0.7962 [0.7952, 0.7965] | 1.0354 [1.0332, 1.0359] |
| standalone-FFV1L1_2-rank2 | 7 | 1.2522 [1.2502, 1.2544] | 0.9927 [0.9913, 0.9939] | 1.4312 [1.4290, 1.4324] | 0.7925 [0.7917, 0.7943] | 1.1419 [1.1406, 1.1441] |
| standalone-FFV1L1_2-rank3 | 1 | 1.1797 [1.1783, 1.1818] | 0.8643 [0.8634, 0.8656] | 1.3933 [1.3911, 1.3944] | 0.7326 [0.7314, 0.7338] | 1.1809 [1.1784, 1.1836] |
| standalone-FFV1L1_2-rank3 | 2 | 1.1753 [1.1736, 1.1761] | 0.8521 [0.8508, 0.8524] | 1.1879 [1.1859, 1.1895] | 0.7252 [0.7239, 0.7267] | 1.0113 [1.0091, 1.0120] |
| standalone-FFV1L1_2-rank3 | 7 | 1.1846 [1.1832, 1.1865] | 0.8717 [0.8711, 0.8733] | 1.4077 [1.4062, 1.4091] | 0.7362 [0.7346, 0.7370] | 1.1885 [1.1872, 1.1893] |
| standalone-FFV1L1_2-rank4 | 1 | 1.1426 [1.1405, 1.1439] | 0.8359 [0.8350, 0.8369] | 1.2940 [1.2919, 1.2957] | 0.7317 [0.7308, 0.7324] | 1.1322 [1.1300, 1.1338] |
| standalone-FFV1L1_2-rank4 | 2 | 1.1380 [1.1348, 1.1429] | 0.8273 [0.8263, 0.8292] | 1.0157 [1.0129, 1.0177] | 0.7279 [0.7234, 0.7300] | 0.8941 [0.8908, 0.8953] |
| standalone-FFV1L1_2-rank4 | 7 | 1.1457 [1.1399, 1.1478] | 0.8488 [0.8443, 0.8512] | 1.2966 [1.2927, 1.2977] | 0.7416 [0.7390, 0.7428] | 1.1318 [1.1286, 1.1343] |
| standalone-FFV1L2_1-rank2 | 1 | 1.2562 [1.2535, 1.2587] | 0.9701 [0.9678, 0.9713] | 1.4275 [1.4234, 1.4308] | 0.7721 [0.7712, 0.7734] | 1.1354 [1.1328, 1.1391] |
| standalone-FFV1L2_1-rank2 | 2 | 1.2408 [1.2399, 1.2441] | 0.9784 [0.9773, 0.9786] | 1.2814 [1.2767, 1.2835] | 0.7875 [0.7864, 0.7889] | 1.0308 [1.0289, 1.0327] |
| standalone-FFV1L2_1-rank2 | 7 | 1.2394 [1.2381, 1.2409] | 0.9848 [0.9832, 0.9860] | 1.4363 [1.4338, 1.4373] | 0.7949 [0.7943, 0.7968] | 1.1585 [1.1574, 1.1609] |
| standalone-FFV1L2_1-rank3 | 1 | 1.1854 [1.1833, 1.1859] | 0.8740 [0.8720, 0.8754] | 1.3976 [1.3943, 1.4003] | 0.7372 [0.7357, 0.7378] | 1.1802 [1.1748, 1.1809] |
| standalone-FFV1L2_1-rank3 | 2 | 1.1774 [1.1723, 1.1800] | 0.8538 [0.8531, 0.8547] | 1.1796 [1.1735, 1.1820] | 0.7249 [0.7233, 0.7277] | 1.0010 [0.9986, 1.0044] |
| standalone-FFV1L2_1-rank3 | 7 | 1.1845 [1.1831, 1.1861] | 0.8725 [0.8717, 0.8732] | 1.3982 [1.3954, 1.3987] | 0.7362 [0.7356, 0.7374] | 1.1802 [1.1791, 1.1805] |
| standalone-FFV1L2_1-rank4 | 1 | 1.1435 [1.1359, 1.1478] | 0.8887 [0.8866, 0.8916] | 1.2837 [1.2782, 1.2865] | 0.7761 [0.7756, 0.7792] | 1.1234 [1.1207, 1.1274] |
| standalone-FFV1L2_1-rank4 | 2 | 1.1395 [1.1357, 1.1416] | 0.8357 [0.8339, 0.8371] | 1.0326 [1.0304, 1.0353] | 0.7336 [0.7318, 0.7340] | 0.9071 [0.9025, 0.9079] |
| standalone-FFV1L2_1-rank4 | 7 | 1.1471 [1.1452, 1.1480] | 0.8858 [0.8846, 0.8871] | 1.2897 [1.2870, 1.2920] | 0.7722 [0.7719, 0.7736] | 1.1248 [1.1237, 1.1257] |
| nlo-FFV1L1_2-rank2 | 1 | 1.2524 [1.2509, 1.2541] | 0.9613 [0.9596, 0.9627] | 1.4690 [1.4669, 1.4709] | 0.7667 [0.7661, 0.7681] | 1.1719 [1.1703, 1.1742] |
| nlo-FFV1L1_2-rank2 | 2 | 1.2490 [1.2475, 1.2514] | 0.9761 [0.9753, 0.9774] | 1.2862 [1.2859, 1.2875] | 0.7813 [0.7802, 0.7834] | 1.0303 [1.0282, 1.0331] |
| nlo-FFV1L1_2-rank2 | 7 | 1.2608 [1.2584, 1.2623] | 0.9700 [0.9682, 0.9710] | 1.4761 [1.4747, 1.4793] | 0.7694 [0.7686, 0.7708] | 1.1711 [1.1698, 1.1741] |
| nlo-FFV1L1_2-rank3 | 1 | 1.1607 [1.1595, 1.1618] | 0.8850 [0.8845, 0.8862] | 1.3802 [1.3798, 1.3824] | 0.7631 [0.7619, 0.7638] | 1.1889 [1.1885, 1.1912] |
| nlo-FFV1L1_2-rank3 | 2 | 1.1649 [1.1624, 1.1661] | 0.8394 [0.8369, 0.8403] | 1.1893 [1.1879, 1.1916] | 0.7212 [0.7185, 0.7228] | 1.0213 [1.0205, 1.0235] |
| nlo-FFV1L1_2-rank3 | 7 | 1.1811 [1.1784, 1.1834] | 1.1310 [1.1295, 1.1327] | 1.4350 [1.4335, 1.4374] | 0.9562 [0.9536, 0.9606] | 1.2173 [1.2134, 1.2196] |
| nlo-FFV1L1_2-rank4 | 1 | 1.1395 [1.1341, 1.1472] | 0.8373 [0.8316, 0.8393] | 1.3328 [1.3226, 1.3345] | 0.7328 [0.7289, 0.7372] | 1.1639 [1.1616, 1.1666] |
| nlo-FFV1L1_2-rank4 | 2 | 1.1331 [1.1302, 1.1363] | 0.8338 [0.8329, 0.8344] | 0.9646 [0.9623, 0.9675] | 0.7358 [0.7339, 0.7392] | 0.8522 [0.8492, 0.8538] |
| nlo-FFV1L1_2-rank4 | 7 | 1.1645 [1.1555, 1.1692] | 0.8527 [0.8513, 0.8535] | 1.3499 [1.3485, 1.3512] | 0.7325 [0.7298, 0.7369] | 1.1607 [1.1557, 1.1673] |
| nlo-FFV1L2_1-rank2 | 1 | 1.2491 [1.2446, 1.2557] | 0.9650 [0.9640, 0.9681] | 1.4760 [1.4731, 1.4797] | 0.7731 [0.7718, 0.7749] | 1.1804 [1.1728, 1.1834] |
| nlo-FFV1L2_1-rank2 | 2 | 1.2483 [1.2462, 1.2512] | 0.9588 [0.9574, 0.9615] | 1.2884 [1.2876, 1.2907] | 0.7677 [0.7660, 0.7692] | 1.0330 [1.0301, 1.0340] |
| nlo-FFV1L2_1-rank2 | 7 | 1.2633 [1.2611, 1.2659] | 1.0036 [1.0014, 1.0059] | 1.4866 [1.4814, 1.4907] | 0.7963 [0.7929, 0.7970] | 1.1774 [1.1746, 1.1825] |
| nlo-FFV1L2_1-rank3 | 1 | 1.1502 [1.1493, 1.1515] | 0.8646 [0.8638, 0.8652] | 1.3540 [1.3515, 1.3555] | 0.7514 [0.7508, 0.7526] | 1.1755 [1.1742, 1.1780] |
| nlo-FFV1L2_1-rank3 | 2 | 1.1567 [1.1557, 1.1597] | 0.8273 [0.8259, 0.8284] | 1.1822 [1.1799, 1.1841] | 0.7141 [0.7133, 0.7156] | 1.0222 [1.0202, 1.0227] |
| nlo-FFV1L2_1-rank3 | 7 | 1.1804 [1.1786, 1.1810] | 0.9235 [0.9226, 0.9252] | 1.4348 [1.4313, 1.4356] | 0.7830 [0.7814, 0.7834] | 1.2165 [1.2133, 1.2173] |
| nlo-FFV1L2_1-rank4 | 1 | 1.1517 [1.1497, 1.1552] | 0.8267 [0.8222, 0.8273] | 1.2667 [1.2607, 1.2685] | 0.7160 [0.7134, 0.7187] | 1.0977 [1.0935, 1.1012] |
| nlo-FFV1L2_1-rank4 | 2 | 1.1659 [1.1653, 1.1675] | 0.8405 [0.8394, 0.8418] | 1.0602 [1.0575, 1.0617] | 0.7202 [0.7192, 0.7217] | 0.9081 [0.9047, 0.9117] |
| nlo-FFV1L2_1-rank4 | 7 | 1.1834 [1.1770, 1.1864] | 0.8450 [0.8439, 0.8508] | 1.3428 [1.3417, 1.3460] | 0.7179 [0.7161, 0.7190] | 1.1405 [1.1369, 1.1411] |

## Verification and reproduction

From the repository root, the read-only verifier checks the scoped checksum manifest, all36 raw logs and3,024 parsed samples. Optional statistics recomputation runs Python only, without compilers or amplitudes:

```sh
source .ninjars-build/dev-env.sh
OPENBLAS_NUM_THREADS=1 python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/kernel/verify.py --recompute
```

To reproduce in a new namespace, choose an idle CPU and use a new run-id. The preparation reads retained correctness sources/dumps and compiles entirely under that new directory; it refuses an existing directory. A failed preflight must remain a rejected attempt, with any subsequent attempt assigned a fresh namespace.

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/kernel/run.py prepare --run-id kernel-new-attempt --cpu 212
python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/kernel/run.py measure --run-id kernel-new-attempt
OPENBLAS_NUM_THREADS=1 python -B docs/benchmarkfolder/experiments/symmetric-factorization-dp/kernel/analyze.py --run-id kernel-new-attempt > docs/benchmarkfolder/experiments/symmetric-factorization-dp/kernel-new-attempt/analysis.json
```

The correctness-seal.json is a separate immutable prerequisite. This kernel seal binds only the listed new kernel files, rejected attempts and completed timing evidence; it excludes ELF/object/module/bytecode build products. Future process studies belong to a new namespace and cannot alter this evidence.
