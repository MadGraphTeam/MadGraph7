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
  `FKSSectorLearningBudget` times the ordinary MINT0 integrand CPU. A probe
  stores the routed S contributions before the absolute value, split into
  the Born part `B` and per-sector parts `R`, per canonical flavour. The
  separately generated virtual-residual lane (type 14) is excluded, so
  probes evaluate neither the exact one-loop virtual nor any H-event
  contribution.
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
- The probes also feed an S-only proxy, recomputed with the newly fitted
  maps, into the grid adaptation; ordinary points alone determine the
  integral and its error.

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
against it. MINT0 also writes
`sector_learning_online.bin`, a binary per-iteration record of the probes,
fits and grid feeds for auditing; production does not read it. Probe
admission depends on measured CPU time, so a seed alone does not reproduce
the training schedule.

## Measured effect

These numbers come from the research implementation this branch was taken
from. There, the per-sector radiation and fixed Sobol offsets were separate
options, and the samples used an experimental weighted correction for
envelope violations; neither is part of this branch. Mean negative S events
per 50,000 unshowered events, with credit .20, folding 111, no
`mcatnlo_delta` and the same settings in all processes, against fixed Sobol
sector offsets without learning:

| Process | Sector shifts (5 workflows) | Tiled maps, credit .20 (3 workflows) |
|---|---:|---:|
| Drell–Yan | 552 | 107 |
| Top pair | 467 | 238 |
| W+jet | 7,808 | 5,650 |
| t-channel single top, 5FS | 1,853 | 880 |
| W+bb, 4FS | 1,592 | 1,283 |

These are point estimates. A separate fresh W+bb comparison with five
workflows per arm confirmed the reduction of the weighted negative S
fraction, from 5.33% to 4.66%. A credit of .10 was also tested and not
adopted: Drell–Yan worsened significantly and W+jet MINT0 CPU increased.

MINT0 CPU was about 1–1.7 times that of plain folding 111 at credit .20; the
W+bb elapsed time reached 2.4 times. These measurements predate the change
that lets probes skip H events and the exact virtual. That change leaves the
probe records unchanged (bit-identical in a Drell–Yan MINT0 with a
deterministic clock) and cut the CPU per probe there by about 30%, so about
40% more probes fit in the same credit. The credit is a soft target on probe
and fit CPU, not a bound on total MINT0 time. A lower negative fraction does
not by itself establish a lower cost for a given precision; no universal
precision-per-CPU gain has been shown. Showered samples were not studied.

Part of the remaining negative S weight comes from the Born-based
virtual-residual lane, which the radiation maps cannot move: about 48% in
top pair and 58% in W+bb for the fitted grids of that study.

## Tests

`test_epoch_replay_reference`, `test_epoch_replay_fortran`,
`test_epoch_replay_iteration` and `test_sector_learning_capture` check the
replay bank construction, the fitting routines against an independent
pure-Python reference (`tests/input_files/tiled_s_event_reference.py`, which
certifies assignment optimality with dual potentials), a complete
MINT0/MINT1/generation fixture with restarts and wrong-setting guards, the
audited `sector_learning_online.bin` records, and the driver hooks that
capture probe contributions. They need neither NumPy nor SciPy.
