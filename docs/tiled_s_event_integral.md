# Tiled S-event integral (`FKSSectorLearning`)

An opt-in MC@NLO sampler that reduces negative S-event weights at folding
`(1,1,1)`. It is off by default; with `FKSSectorLearning=.false.` the event
generation is unchanged.

## What it does

With explicit FKS-sector summation, every S event sums the Born/virtual
part and the real-emission part of each outer FKS sector. With learning on,
each outer sector gets its own radiation sample: its flat adaptive-grid ξ/y
coordinates are translated by a static per-sector offset, and each outer
sector of each Born slot gets a bijection of 16 tiles (a 4×4 split of these
coordinates). For a sampled radiation point, the sector's tile is moved to
its mapped tile; the position inside the tile, φ and the Born point are
unchanged. MINT's adaptive-grid Jacobian ratio is applied, so each sector
integral is preserved; only the correlations between sector samples within
one S event change.

The maps are learned inside ordinary MINT0:

- At regular points (after 64 ordinary calls, every 16th), a probe evaluates
  one Born point at all 16 nodes of a (4,4,1) stencil. Probes are admitted
  while their CPU, plus fitting and archiving, stays below
  `FKSSectorLearningBudget` times the CPU of the ordinary MINT0 points:
  their sampling, integrand and grid accumulation, timed per block of 16
  points. The work between passes and iterations (point-count checks,
  channel scans, combining results) is not counted. A probe
  stores the routed S contributions before the absolute value, split into
  the Born part `B` and per-sector parts `R`, per canonical flavour. The
  separately generated virtual-residual lane (type 14) is excluded, so
  probes evaluate neither the exact one-loop virtual nor any H-event
  contribution. A probe is an ordinary MINT0 call with its own fold loop
  (`sector_learning_collect`); neither the grid, the MC-integer channel
  weights nor the virtual surrogate see it.
- At each iteration boundary, the maps are refitted to minimise the negative
  part of the summed S weight,
  `Σ max(-B(node) - Σ_s R_s(P_s(node)), 0)`, over the current bank of probes
  plus the immediately preceding bank at half weight (scaled by
  `N_current / (2 N_previous)`). Each sector's deployed offset is a static
  quarter-period gauge (the sector's Sobol offset when all sectors of the
  slot have quarter-aligned offsets, zero otherwise) composed with its tile
  map. The
  fit tries three starts (the current maps, the identity, and the maps that
  cancel the gauge), refines each with tile translations and per-sector
  assignment passes, and keeps the lowest loss. Map capacity grows from 4
  to 8 to 16 tiles once enough probes support it.
- Probes only train the maps: ordinary points alone adapt the grid and
  determine the integral and its error.
- Each sector of an ordinary point is evaluated in its own ξ/y grid cells,
  so in those two dimensions the point's ABS weight is split before it
  enters the grid: the Born part goes to the sampled cells, each sector's
  part (its H events and its share of the S event) to the cells where that
  sector was evaluated. An S event `|B + Σ_s R_s|` is shared in proportion
  to `|B|` and the `|R_s|`. Without the split the grid adapts to the sum at
  the sampled cells, which matches none of the shifted sectors, and MINT1
  and event generation need many more points.

The final maps are frozen. MINT1 recomputes the normalisation and event
generation uses the frozen maps; no extra matrix elements are evaluated
after MINT0.

## Use

In `Cards/FKS_params.dat`:

```text
#FKSSectorLearning
.true.
#FKSSectorLearningBudget
0.20d0
```

It requires `FKSExplicitSum` and `MCSubtractionAtFixedFlow` (both `.true.`
by default). In `Cards/run_card.dat`: `folding = 1, 1, 1` and
`born_spreading = False` (checked before MINT0 starts).

Learning applies to complete MC@NLO runs. At LO (`aMC@LO`, `noshowerLO`,
`[LOonly]`) and in fixed-order runs, `FKSSectorLearning` is ignored.

Run fresh MINT0 and MINT1 before generation whenever the method or credit
changes. `mint_grids` stores the frozen maps (`SCL10` table); MINT1 and
generation verify the setting, the fold counts and the adaptive grid
against it, and stop with "rerun MINT0" on a table written by another code
version. With `sector_learning_archive` set in `mint_module` (the tests
do this), MINT0 also writes `sector_learning_online.bin`, a binary
per-iteration record of the probes, fits and grid accumulators for auditing;
production runs neither write nor read it, and do not keep its grid
accounting. Probe admission depends on measured CPU time, so a seed alone does not reproduce
the training schedule.

## Measured effect

Five runs per process on `main` (2c7fcbd705) and on this branch (f37d302698)
with `FKSSectorLearning` at the default budget of .20, seeds 271828, 314159,
161803, 141421 and 173205. Each run generates 50,000 unshowered events at
13 TeV with folding 111, `born_spreading = False`, no `mcatnlo_delta` and
otherwise identical cards: NNPDF4.0 NLO (its nf=4 set for W+bb),
`dynamical_scale_choice = 3`, ptj > 10 GeV for W+jet and single top, and
mll > 30 GeV for Drell–Yan. Means over the five runs; ± is the standard
error of the mean.

| Process | Negative S events, main | Negative S events, learning | All negative events, main → learning | Events for equal precision | CPU for 50,000 events | CPU for equal precision |
|---|---:|---:|---:|---:|---:|---:|
| Drell–Yan | 2,162 ± 7 | 127 ± 9 | 3,620 → 1,719 | 0.84 | 1.12 | 0.94 |
| Top pair | 1,894 ± 23 | 180 ± 31 | 10,617 → 9,640 | 0.88 | 1.04 | 0.91 |
| W+jet | 9,340 ± 40 | 4,830 ± 69 | 13,282 → 9,876 | 0.60 | 0.90 | 0.54 |
| t-channel single top, 5FS | 2,197 ± 24 | 796 ± 18 | 13,932 → 13,218 | 0.88 | 1.03 | 0.91 |
| W+bb, 4FS | 2,727 ± 31 | 1,332 ± 17 | 7,946 → 6,858 | 0.88 | 1.69 | 1.49 |

The last three columns are ratios, learning over `main`. "Events for equal
precision" is the ratio of (1 − 2f)², with f the fraction of negative events:
the number of unweighted events needed for the same statistical precision,
which is what counts when showering and detector simulation dominate the
cost. CPU is the total of MINT0, MINT1 and event generation, with both arms
running side by side on one machine; "CPU for equal precision" is the
product of the two ratios. Cross sections agree within their errors. The
fraction of negative H events is unchanged; there are slightly more H events
because the absolute S integral is smaller.

W+bb remains expensive: MINT1 needs 2.3 times as many points as on `main`
and event generation 1.5 times as many points per event. The budget is a
soft target on probe and fit CPU, not a bound on total run time. Showered
samples were not studied.

In the research implementation this branch was taken from, a budget of .10
was also tested and not adopted: Drell–Yan worsened significantly and W+jet
MINT0 CPU increased. That study also found that part of the remaining
negative S weight comes from the Born-based virtual-residual lane, which the
radiation maps cannot move: about 48% in top pair and 58% in W+bb.

## Tests

`test_epoch_replay_reference`, `test_epoch_replay_fortran`,
`test_epoch_replay_iteration` and `test_sector_learning_capture` check the
replay bank construction, the fitting routines against an independent
pure-Python reference (`tests/input_files/tiled_s_event_reference.py`, which
certifies assignment optimality with dual potentials), a complete
MINT0/MINT1/generation fixture with restarts and wrong-setting guards, the
audited `sector_learning_online.bin` records, and the driver hooks that
capture probe contributions. `test_sector_learning_grid_split` checks how an
ordinary point's ABS weight is split over the ξ/y grid cells. They need
neither NumPy nor SciPy.
