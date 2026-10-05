# SPDX-License-Identifier: 0BSD
"""Optional MadSpace integration check of the three UMAMI grouping modes.

This is stochastic validation of structurally established equivalences, not a
numerical proof of those equivalences. Run explicitly; no optional integration
packages are added to pyAmpliCol's ordinary test dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import madspace as ms
import torch
from run import execute, generate
from test_integrate import Config, Cuts, integrate_per_channel, load_channels

MODES = ("none", "exact", "flavour_blind_observables")


def check_agreement(results, *, sigma=6.0, maximum_relative_error=0.05):
    """Compare independent estimates, retaining their complete uncertainties."""
    for result in results.values():
        value, error = result["integral"], result["uncertainty"]
        if not (
            math.isfinite(value) and value > 0 and math.isfinite(error) and error > 0
        ):
            raise AssertionError("expected a positive finite integral and uncertainty")
        if error > maximum_relative_error * value:
            raise AssertionError(
                "grouping comparison is inconclusive: increase --n to reduce "
                "the integration uncertainty"
            )
    comparisons = []
    reference = results["exact"]
    for mode in ("none", "flavour_blind_observables"):
        result = results[mode]
        difference = abs(result["integral"] - reference["integral"])
        uncertainty = math.hypot(result["uncertainty"], reference["uncertainty"])
        rounding = 1e-10 * max(result["integral"], reference["integral"])
        comparisons.append(
            {
                "mode": mode,
                "difference": difference,
                "combined_error": uncertainty,
                "standard_errors": difference / uncertainty,
            }
        )
        if difference > sigma * uncertainty + rounding:
            raise AssertionError(
                f"{mode} differs from exact by {difference / uncertainty:.2f} "
                f"combined standard errors (limit {sigma:g})"
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/umami-grouping")
    )
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--rusticol-config", default="rusticol-config")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--training", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args()
    if min(args.n, *args.training) < 2:
        parser.error("integration and training counts must be at least two")
    config_binary = shutil.which(args.rusticol_config)
    if config_binary is None:
        parser.error("activate the pyAmpliCol environment or pass --rusticol-config")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = Config(sqrts=1000.0, cuts=Cuts(pt_min=100.0, eta_max=6.0, dr_min=0.4))
    results = {}
    for index, mode in enumerate(MODES):
        artifact = output / mode
        if artifact.exists():
            if not args.reuse:
                parser.error(
                    f"{artifact} already exists; use --reuse or another output"
                )
        else:
            generate(artifact, "lc", mode, "g g > g g")
        metadata = artifact / "API/umami/metadata.json"
        channels, n_out, data = load_channels(metadata)
        if (
            data["grouping"]["mode"] != mode
            or data["provider"]["color_accuracy"] != "lc"
        ):
            raise AssertionError("existing artifact does not match the requested mode")
        build = output / f"build_{mode}"
        execute(
            "make",
            "-C",
            artifact / "API/umami",
            f"BUILD_DIR={build}",
            f"RUSTICOL_CONFIG={config_binary}",
        )
        library = build / data["provider"]["library"]
        execute(
            sys.executable,
            Path(__file__).with_name("test_umami.py"),
            artifact,
            "--library",
            library,
            "--n",
            4,
            "--seed",
            args.seed,
        )
        # Distinct fixed seeds make the quadrature combination of errors valid.
        seed = args.seed + index * 100_003
        value, error, per_channel = integrate_per_channel(
            channels,
            n_out,
            cfg,
            ms.Context(1),
            library,
            artifact,
            n_points=args.n,
            seed=seed,
            training_points=args.training,
        )
        results[mode] = {
            "integral": value,
            "uncertainty": error,
            "unit": "pb",
            "seed": seed,
            "points_per_channel": args.n,
            "per_channel": per_channel,
            "exported_contributions": data["grouping"]["exported_contributions"],
            "physical_contributions": data["grouping"]["physical_contributions"],
        }
        print(f"{mode}: {value:.8g} +/- {error:.3g} pb", flush=True)
    summary = {"process": "g g > g g", "cuts": vars(cfg.cuts), "results": results}
    report = output / "grouping_comparison.json"
    # Retain estimates even if the statistical acceptance fails.
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not (
        results["flavour_blind_observables"]["exported_contributions"]
        < results["exact"]["exported_contributions"]
    ):
        raise AssertionError("the positive fixture must demonstrate useful compression")
    summary["comparisons"] = check_agreement(results)
    summary["accepted"] = True
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"All grouping integrals agree within six combined standard errors: {report}")


if __name__ == "__main__":
    main()
