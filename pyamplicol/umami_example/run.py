# SPDX-License-Identifier: 0BSD
"""Generate gg -> ggg artifacts and run the standalone and MadSpace examples."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _metadata import load_metadata


def generate(artifact: Path, accuracy: str, grouping: str) -> None:
    from pyamplicol import Generator, ModelSource
    from pyamplicol.config import resolve_config

    config = resolve_config(
        {
            "schema_version": 1,
            "action": "generate",
            "color": {"accuracy": accuracy},
            "generation": {
                "workers": 1,
                "emit_api_bundle": True,
                "umami_grouping": grouping,
                "relation_discovery": {"mode": "off"},
            },
            "evaluator": {
                "backend": "jit",
                "execution_mode": "recurrence",
                "optimization": {"cores": 1},
                "jit": {"optimization_level": 2},
            },
        }
    )
    Generator(config).generate("g g > g g g", artifact, model=ModelSource.built_in_sm())


def execute(*command: object) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".artifacts/umami-example"))
    parser.add_argument(
        "--accuracy", choices=("lc", "full"), nargs="+", default=["lc", "full"]
    )
    parser.add_argument(
        "--grouping",
        choices=("exact", "flavour_blind_observables", "none"),
        default="exact",
    )
    parser.add_argument(
        "--reuse", action="store_true", help="reuse previously generated artifacts"
    )
    parser.add_argument("--rusticol-config", default="rusticol-config")
    parser.add_argument(
        "--n", type=int, default=2048, help="integration points per channel"
    )
    parser.add_argument("--training", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--check-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if min(args.n, *args.training) < 2 or args.check_points < 1:
        parser.error(
            "integration/training counts must be at least two; "
            "check-points must be positive"
        )
    rusticol_config = shutil.which(args.rusticol_config)
    if rusticol_config is None:
        parser.error(
            "activate pyAmpliCol's environment or provide "
            "--rusticol-config /absolute/path"
        )
    # Check optional integration dependencies before starting generation.
    for module in ("madspace", "madnis", "torch"):
        __import__(module)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    for accuracy in dict.fromkeys(args.accuracy):
        artifact = directory / accuracy
        if artifact.exists():
            if not args.reuse:
                parser.error(
                    f"{artifact} exists; choose another output or pass --reuse"
                )
        else:
            generate(artifact, accuracy, args.grouping)
        metadata = artifact / "API/umami/metadata.json"
        data = load_metadata(metadata)
        if (
            data["provider"]["color_accuracy"] != accuracy
            or data["grouping"]["mode"] != args.grouping
        ):
            raise ValueError(
                "existing artifact's colour accuracy or grouping differs "
                "from the request"
            )
        build_dir = directory / f"build_{accuracy}"
        execute(
            "make",
            "-C",
            artifact / "API/umami",
            f"BUILD_DIR={build_dir}",
            f"RUSTICOL_CONFIG={rusticol_config}",
        )
        library = build_dir / data["provider"]["library"]
        execute(build_dir / "umami_driver", artifact)
        execute(
            sys.executable,
            scripts / "test_umami.py",
            artifact,
            "--library",
            library,
            "--n",
            args.check_points,
            "--seed",
            args.seed,
        )
        execute(
            sys.executable,
            scripts / "test_integrate.py",
            metadata,
            "--library",
            library,
            "--artifact",
            artifact,
            "--no-pdf",
            "--vegas-only",
            "--n",
            args.n,
            "--training",
            *args.training,
            "--seed",
            args.seed,
            "--result",
            directory / f"integration_{accuracy}.json",
        )
    print(
        f"Completed standalone, batched-parity and integration examples in {directory}"
    )


if __name__ == "__main__":
    main()
