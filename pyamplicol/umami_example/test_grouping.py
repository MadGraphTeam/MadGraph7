# SPDX-License-Identifier: 0BSD
"""Compare automatically grouped and explicit-member MadSpace integrals.

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
from test_integrate import Channel, Config, Cuts, integrate_per_channel, load_channels

FIXTURES = {
    # Process: (integration channels, physical colour contributions, orbit size).
    "g g > g g": (3, 6, 2),
    "g g > g g g": (4, 24, 6),
}


def check_identical_grouping(data, process):
    """Require the automatic identical-particle quotient for a known fixture."""
    channel_count, physical_count, orbit_size = FIXTURES[process]
    entries = [entry for ch in data["channels"] for entry in ch["processes"]]
    if (
        len(data["channels"]) != channel_count
        or len(entries) != channel_count
        or data["grouping"]["exported_contributions"] != channel_count
        or data["grouping"]["physical_contributions"] != physical_count
    ):
        raise AssertionError(
            f"{process} must group {physical_count} physical contributions into "
            f"{channel_count} automatic integration channels"
        )
    factors = []
    for entry in entries:
        terms = entry["matrix_elements"]
        if (
            len(terms) != 1
            or float(terms[0]["factor"]) != orbit_size
            or len(entry["members"]) != orbit_size
        ):
            raise AssertionError(
                f"each {process} representative must have {orbit_size} physical "
                f"members and explicit integration factor {orbit_size}"
            )
        factors.append(float(terms[0]["factor"]))
    return factors


def physical_member_channels(data):
    """Expand physical weights and momentum maps, never integration factors.

    This is a reference integrand for the acceptance check, not an export option.
    The provider still uses its certified representative computations.
    """
    channels = []
    for ci, channel in enumerate(data["channels"]):
        for fi, entry in enumerate(channel["processes"]):
            for member in entry["members"]:
                # p_rep[representative map] = p_physical[member map].
                permutation = [0] * data["provider"]["particle_count"]
                for rep_slot, physical_slot in zip(
                    entry["runtime"]["permutation"],
                    member["runtime"]["permutation"],
                    strict=True,
                ):
                    permutation[rep_slot] = physical_slot
                pdgs = member["pdgs"]
                channels.append(
                    Channel(
                        index=len(channels),
                        color_order=[
                            permutation[slot] for slot in channel["phasespace_order"]
                        ],
                        init_states=[tuple(pdgs[:2])],
                        pdg_final=list(pdgs[2:]),
                        masses=list(data["provider"]["masses"]),
                        processes=[
                            [
                                (
                                    float(member["factor"]),
                                    pdgs[0],
                                    pdgs[1],
                                    tuple(pdgs[2:]),
                                )
                            ]
                        ],
                        selector=(ci, fi),
                        momentum_permutation=permutation,
                    )
                )
    return channels


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
    reference = results["physical_members"]
    result = results["grouped"]
    difference = abs(result["integral"] - reference["integral"])
    uncertainty = math.hypot(result["uncertainty"], reference["uncertainty"])
    rounding = 1e-10 * max(result["integral"], reference["integral"])
    if difference > sigma * uncertainty + rounding:
        raise AssertionError(
            "grouped and physical-member integrals differ by "
            f"{difference / uncertainty:.2f} combined standard errors (limit {sigma:g})"
        )
    return {
        "difference": difference,
        "combined_error": uncertainty,
        "standard_errors": difference / uncertainty,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/umami-grouping")
    )
    parser.add_argument(
        "--process",
        choices=tuple(FIXTURES),
        default="g g > g g",
        help="identical-gluon acceptance fixture (default: %(default)s)",
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
    artifact = output / "lc"
    if artifact.exists():
        if not args.reuse:
            parser.error(f"{artifact} already exists; use --reuse or another output")
    else:
        generate(artifact, "lc", args.process)
    metadata = artifact / "API/umami/metadata.json"
    grouped_channels, n_out, data = load_channels(metadata)
    if data["provider"]["color_accuracy"] != "lc":
        raise AssertionError("existing artifact is not a leading-colour provider")
    integration_factors = check_identical_grouping(data, args.process)
    member_channels = physical_member_channels(data)
    if len(member_channels) != data["grouping"]["physical_contributions"]:
        raise AssertionError("reference must contain every physical contribution once")
    build = output / "build_lc"
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
    results = {}
    for index, (label, channels) in enumerate(
        (("grouped", grouped_channels), ("physical_members", member_channels))
    ):
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
        results[label] = {
            "integral": value,
            "uncertainty": error,
            "unit": "pb",
            "seed": seed,
            "points_per_channel": args.n,
            "per_channel": per_channel,
            "integration_channels": len(channels),
        }
        print(f"{label}: {value:.8g} +/- {error:.3g} pb", flush=True)
    summary = {
        "process": args.process,
        "cuts": vars(cfg.cuts),
        "grouping": data["grouping"],
        "integration_factors": integration_factors,
        "results": results,
    }
    report = output / "grouping_comparison.json"
    # Retain estimates even if the statistical acceptance fails.
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    summary["comparison"] = check_agreement(results)
    summary["accepted"] = True
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        f"Grouped/member integrals agree within six combined standard errors: {report}"
    )


if __name__ == "__main__":
    main()
