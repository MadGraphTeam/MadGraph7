#!/usr/bin/env python3
"""Complex-mass-scheme stress case for a registered campaign.

`prepare` builds a baseline/candidate export pair of the CMS process with the
campaign's own generator, model and flags (exactly like the ordinary cases) and
freezes the earlier CMS fixtures: 12 ordinary warm-up points and 21 stress
points (ordinary anchors, Z resonance, internal WW threshold, forward small
Gram). The lanes then run with tools/tirust_stress.py like any stress case:

  tirust_cms_stress.py prepare --campaign C [--jobs 4]
  tirust_stress.py run --campaign C --case cms_ew_n2_epem_mupmum --lane L --mode M --cpu N
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench  # noqa: E402
import tirust_stress as stress  # noqa: E402

FIXTURES = ROOT / "docs/benchmarkfolder/experiments/complex-mass/points"
CASE = stress.EXTRA_CASES["cms_ew_n2_epem_mupmum"]


def prepare(campaign, jobs=4):
    from tirust_benchmark_campaign import activate_campaign
    activate_campaign(bench, campaign)
    target = bench.REPORT / "stress-points"
    target.mkdir(parents=True, exist_ok=True)
    for source, name in ((f"{CASE['id']}-ordinary", CASE["id"]), (CASE["id"], f"{CASE['id']}-stress")):
        for suffix in (".dat", ".json"):
            destination = target / (name + suffix)
            if destination.exists() and bench.digest(destination) != bench.digest(FIXTURES / (source + suffix)):
                raise ValueError(f"Frozen CMS fixture differs: {destination}")
            shutil.copy2(FIXTURES / (source + suffix), destination)
    for source in ("baseline", "candidate"):
        bench.prepare_baseline()
        bench.generate(source, CASE)
        bench.build(source, CASE, jobs)
        print(f"built {source} {CASE['id']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    prepare(args.campaign, args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
