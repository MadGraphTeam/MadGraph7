<!-- SPDX-License-Identifier: 0BSD -->
# Manual table-filling campaign for `macbook_M3_manual`

The executable controller is the authoritative campaign interface:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py --help
```

For an installed release wheel, first copy a fresh reset campaign and invoke
the controller from that destination:

```console
pyamplicol profiling-campaign copy ./pyamplicol-profiling-campaign --force
./pyamplicol-profiling-campaign/steer_performance_campaign.py \
  dashboard-snapshot
```

PyAmpliCol-only dry runs and measurements do not need a legacy checkout.
Selections whose planned dependency closure includes original AmpliCol require
`run --original-amplicol PATH`; `PATH` must be a clean, complete checkout
exposing the PR #12 probe sources and Make targets. The
`amplicol_with_patches` branch works now, and a compatible upstream revision
will work unchanged after the PR is merged.

Inside a contributor checkout it re-executes with the repository `.venv`; an
installed copy uses the wheel's Python runtime. Both modes reuse compatible
same-source currents by default, supervise each process tree, and keep attempts
and large artifacts outside this publication directory.

Preview selections before launching workers:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py run \
  --dry-run --table z_table --multiplicity 3 4 \
  --model built_in sm_ufo
```

The preview lists direct cells and dependency work separately and prints a
wrapped public-CLI recipe for every selected runnable cell, even for broad
selections. Each recipe uses the repository `.venv/bin/pyamplicol` and states
whether it is exact, a template, or a diagnostic protocol exception. UFO
recurrence/eager recipes include their public `model compile` prerequisite.

Run a bounded multicore selection:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py run \
  --workers 4 --cores-per-worker 2 --table z_table --multiplicity 3
```

Defaults are one worker, one core, a one-hour generation/preparation limit, and
a decimal 30 GB process-tree RAM limit. Press `Ctrl-C` or `Esc` to stop
dispatch, terminate supervised process trees, preserve completed currents, and
restore the terminal.

Capture a running dashboard from another terminal without attaching to or
changing the campaign:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py \
  dashboard-snapshot --live --width 160 --height 48
```

This reads only the compact non-stale coordination lease. The newest active
invocation is the default; pass `--instance ID_OR_PREFIX` to choose one
concurrent run. Omit `--live` for the deterministic synthetic layout fixture.

Inspect coverage and AmpliCol-relative statistics:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py inspect \
  --color-approximation lc --generation-engine recurrence compiled
```

Rebuild all current JSON/TeX tables and atomically replace the PDF:

```console
./docs/performance_reports/macbook_M3_manual/steer_performance_campaign.py refresh-pdf
```

Human output is coloured by default. Use `--no-color` or `NO_COLOR`; JSON is
always uncoloured. Consult each subcommand’s `--help` for all selectors,
aliases, resource controls, profiling parameters, reuse rules, and examples.
