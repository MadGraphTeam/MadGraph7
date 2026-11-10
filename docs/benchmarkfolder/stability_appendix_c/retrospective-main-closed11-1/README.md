# Four Standard Model stability comparisons from the closed campaign

These eight figures compare CutTools, Ninja, COLLIER, NinjaRS and TIRust on
four actual Standard Model processes. Each process has the same 100 saved
physical points for all five backends. The figures use only the fully closed
eleven-case numerical capsule pinned at
`8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be`.
Its independent numerical replay must pass before the adapter constructs
either metric. No new amplitudes or timings are evaluated here.

The upper panels follow the empirical-tail presentation of Appendix C in
[arXiv:1604.01363](https://arxiv.org/pdf/1604.01363): fraction of points whose
relative metric exceeds the horizontal target. The lower panels divide
that tail by the Ninja tail. These are **retrospective accepted-output
diagnostics**, not a reproduction of the paper's raw-DP three-probe test.
The original ordinary campaign fixtures are not asserted to satisfy the
paper's transverse-momentum or angular cuts. A separately executed raw-DP
pilot uses explicitly cut-qualified fixtures.

| Process | Accepted composite estimate | Finite disagreement vs current CutTools | TIRust accepted DP/QP |
|---|---|---|---|
| `g g > t t~ [virt=QCD]` | [PDF](figures/qcd_n2_gg_ttx-accepted-internal-estimate.pdf) | [PDF](figures/qcd_n2_gg_ttx-finite-disagreement-current-cuttools.pdf) | 100 / 0 |
| `u u~ > z g g [virt=QCD]` | [PDF](figures/qcd_n3_uux_zgg-accepted-internal-estimate.pdf) | [PDF](figures/qcd_n3_uux_zgg-finite-disagreement-current-cuttools.pdf) | 87 / 13 |
| `e+ e- > mu+ mu- [virt=QED]` | [PDF](figures/ew_n2_epem_mupmum-accepted-internal-estimate.pdf) | [PDF](figures/ew_n2_epem_mupmum-finite-disagreement-current-cuttools.pdf) | 100 / 0 |
| `u u~ > z z g g [virt=QCD]` | [PDF](figures/qcd_n4_uux_zzgg-accepted-internal-estimate.pdf) | [PDF](figures/qcd_n4_uux_zzgg-finite-disagreement-current-cuttools.pdf) | 14 / 86 |

The accepted composite estimate is MadLoop's recorded post-rescue diagnostic,
including finite and pole information. COLLIER's estimate may differ from
the other host estimators. Exact-zero estimates represent equal diagnostics,
not certified digits or infinite precision. The finite comparison is exactly
`abs(a-b)/abs(a+b)` for the finite coefficient, using the **current candidate
CutTools** result as reference. This measures reducer agreement; the reference
is not independently known truth. CutTools self-comparisons contain 100
exact zeros.

Each individual figure is legible at page width and includes accepted DP/QP
counts. The abbreviations in the precision caption are CT=CutTools, N=Ninja,
C=COLLIER, NR=NinjaRS and T=TIRust. These counts describe final accepted
precision, not the number or precision of all internal attempts. In the
six-leg process CutTools accepts 88 DP / 12 QP points, Ninja 97 / 3,
COLLIER 100 / 0, NinjaRS 96 / 4, and TIRust 14 / 86. The stored H/U pairs
disambiguate the legacy precision-dependent aliases: for example H=2,U=9 is
TIRust DP while H=3,U=9 is CutTools QP; TIRust QP is H=3,U=3.

There are 100 points per case, never 200: total and split-order answer rows
are checked bit-exact duplicates and collapsed once. All five intended
backends remain present. This closed dataset has no unavailable accepted
points and no `##W03` messages in its five selected runtime consoles per
case. Unobserved internal retry counts and floating-point exception flags
remain unobserved. The complete selected per-point rows, point hashes,
physical-card hashes, recorded numerical settings, equal pre/post numerical
card hashes, executable identity and console hashes are retained in JSON.
**Effective `MadLoopParams.dat` bytes are absent from this capsule**; the
settings and card hashes are recorded snapshots, and no independently
byte-bound host-policy vertical marker is drawn.

The tail is the exact empirical staircase, including every observed
positive metric breakpoint within the displayed domain and the exact
labelled target values. A value equal to the target does not exceed it.
With 100 fixtures the empirical resolution is 0.01; no claim about rarer
population tails follows. Zero tails are omitted on the logarithmic
vertical axis and their zero counts are explicit. Ratios are masked when
Ninja has zero exceedances; no floor or pseudocount is introduced. Valid
zero ratios remain zero.

All selected finite disagreements pass the existing `3e-6` regression
criterion. The stricter ordinary-point `1e-8` target is **not universally
met**: the Zgg four non-CutTools comparisons each exceed it on 28 points,
and the ZZgg target misses also remain visible and retained. These
reference-dependent misses are not discarded because the host accepted
the amplitudes. The report's overall accuracy, stress and runtime
acceptance criteria remain separate from this figure delivery.

Reproduce into a fresh directory, from the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/retrospective-main-closed11-1/adapter.py --execute --destination /tmp/madloop-stability-main-reproduction-1
```

Purely verify the delivered figures' hashes, all derived point metrics,
accepted outcomes and exact tail/ratio summaries without rendering or
evaluating amplitudes:

```sh
python3 -B docs/benchmarkfolder/stability_appendix_c/retrospective-main-closed11-1/adapter.py --verify-only
```

The exact admitted plotter is source 5, index
`fe16afaae081694af3eb3a7e7c8ec9b9d037933b1938e6afa6ba56522c507e41`.
`--capsule` and `--plotter` accept explicitly relocated copies of these
pinned packages. No original absolute path recorded in their provenance
is followed. `portable_check.py --output /tmp/new-proof.json` copies all
dependencies and figures into a fresh temporary directory, changes its
working directory there, and verifies with original-workspace reads and
native loading/process creation denied. Its proof retains five deliberate
negative controls and requires zero unexpected denied accesses and
byte-identical relocated files.

The sealed source index, unit-test result and relocated verification proof
are adjacent to this README. Existing plotter and retrospective namespaces
remain unchanged.
