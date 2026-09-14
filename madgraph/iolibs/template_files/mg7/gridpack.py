#! /usr/bin/env python3

import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

# search order for madspace package:
#   1. a precompiled install (madspace/install/madspace)
#   2. bundled source that still needs to be built (madspace/install.py)
#   3. otherwise fall back to madspace is available in the environment
_GRIDPACK_DIR = Path(os.path.realpath(__file__)).parent.parent
_LOCAL_MADSPACE_DIR = _GRIDPACK_DIR / "madspace"
_LOCAL_INSTALL_DIR = _LOCAL_MADSPACE_DIR / "install"
if (_LOCAL_INSTALL_DIR / "madspace").is_dir():
    sys.path.insert(0, str(_LOCAL_INSTALL_DIR))
elif (_LOCAL_MADSPACE_DIR / "install.py").is_file():
    print()
    print("You don't have madspace installed for this gridpack")
    print("Running interactive madspace installation script")
    print()

    _result = subprocess.run([sys.executable, str(_LOCAL_MADSPACE_DIR / "install.py")])
    if _result.returncode != 0:
        raise RuntimeError("madspace installation failed — see output above")
    sys.path.insert(0, str(_LOCAL_INSTALL_DIR))

import madspace as ms
import glob
import json
import tomllib
import argparse

def resolve_verbosity(verbosity: str) -> str:
    """Resolve the run_card "auto" verbosity to "pretty"/"log" depending on
    whether stdout is attached to a terminal; other values pass through
    unchanged."""
    if verbosity == "auto":
        return "pretty" if sys.stdout.isatty() else "log"
    return verbosity


def resolve_seed(seed: int) -> int:
    """Resolve the run_card "seed": -1 draws a fresh 64-bit seed via
    os.urandom, any other value is used as-is."""
    if seed == -1:
        return int.from_bytes(os.urandom(8), "big")
    return seed


def main() -> None:
    # load run card and metadata. Use the RunCardMG7 representation when the
    # madgraph package is importable; gridpacks are meant to be portable, so
    # fall back to a plain tomllib parse otherwise (the card is the same TOML).
    run_card_path = os.path.join("Cards", "grid_run_card.toml")
    try:
        from madgraph.various.banner import RunCardMG7
        run_card = RunCardMG7(run_card_path)
    except ImportError:
        with open(run_card_path, "rb") as f:
            run_card = tomllib.load(f)
    run_args = run_card["run"]
    gen_args = run_card["generation"]
    param_card_path = os.path.join("Cards", "param_card.dat")
    with open(os.path.join("data", "data.json")) as f:
        madspace_data = json.load(f)
    if madspace_data["source_hash"] != ms.SOURCE_HASH:
        print()
        print(
            "\033[1m\033[31mWARNING\033[39m: The madspace version is not identical "
            "to the one used to generate the gridpack. This can lead to errors or "
            "incorrect results\033[0m"
        )
        print()

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=run_args["run_name"])
    parser.add_argument(
        "--seed", type=int, default=run_args.get("seed", -1),
        help="every run is reproducible from its seed; -1 draws a fresh random "
             "seed each run instead of fixing one here (still recorded in the "
             "run's info.json)"
    )
    parser.add_argument("--device", type=str, nargs="*")
    parser.add_argument(
        "--cpu_thread_pool_size", type=int, default=run_args["cpu_thread_pool_size"]
    )
    parser.add_argument(
        "--gpu_thread_pool_size", type=int, default=run_args["gpu_thread_pool_size"]
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default=run_args["verbosity"],
        choices=["none", "pretty", "log", "auto"]
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default=run_args["output_format"],
        choices=["lhe", "lhe_npy", "compact_npy"]
    )
    parser.add_argument("--events", type=int, default=gen_args["events"])
    parser.add_argument("--max_overweight_truncation", type=float, default=gen_args["max_overweight_truncation"])
    parser.add_argument("--freeze_max_weight_after", type=int, default=gen_args["freeze_max_weight_after"])
    parser.add_argument("--cpu_batch_size", type=int, default=gen_args["cpu_batch_size"])
    parser.add_argument("--gpu_batch_size", type=int, default=gen_args["gpu_batch_size"])
    args = parser.parse_args()
    seed = resolve_seed(args.seed)

    # initialize event directory
    run_name = args.run_name
    os.makedirs("Events", exist_ok=True)
    run_dir_prefix = os.path.join("Events", f"{run_name}_")
    existing_run_dirs = glob.glob(f"{run_dir_prefix}*")
    run_index = 1
    for run_dir in existing_run_dirs:
        run_index_str = run_dir[len(run_dir_prefix):]
        if run_index_str.isnumeric():
            run_index = max(run_index, int(run_index_str) + 1)
    while True:
        try:
            run_path = f"{run_dir_prefix}{run_index:02d}"
            os.mkdir(run_path)
            break
        except FileExistsError:
            run_index += 1

    # initialize context
    device_names = args.device if args.device else run_args["device"]
    cpu_mode = run_args["cpu_mode"]
    contexts = []
    backends = []
    for device_name in device_names:
        if ":" in device_name:
            device_type, device_index_str = device_name.split(":")
            device_index = int(device_index_str)
        else:
            device_type = device_name
            device_index = 0
        # cpu_mode names the SIMD width of the CPU code, so it applies to the
        # 'cpu' devices only: cuda/hip build the backend named after the device.
        backends.append(cpu_mode if device_type == "cpu" else device_type)
        if device_type == "cuda":
            device = ms.cuda_device(device_index)
            pool_size = args.gpu_thread_pool_size
        elif device_type == "hip":
            device = ms.hip_device(device_index)
            pool_size = args.gpu_thread_pool_size
        else:
            device = ms.cpu_device()
            pool_size = args.cpu_thread_pool_size
        contexts.append(ms.Context(device=device, thread_count=pool_size))

    # set up generator configuration
    config = ms.GeneratorConfig()
    config.target_count = args.events
    config.max_overweight_truncation = args.max_overweight_truncation
    config.freeze_max_weight_after = args.freeze_max_weight_after
    config.cpu_batch_size = args.cpu_batch_size
    config.gpu_batch_size = args.gpu_batch_size
    config.verbosity = resolve_verbosity(args.verbosity)
    config.combine_thread_count = run_args["combine_thread_pool_size"]
    config.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
    config.max_cut_repetitions = gen_args["max_cut_repetitions"]

    # set up contexts
    global_dir = os.path.join("data", "globals")
    for context, backend in zip(contexts, backends):
        context.load_globals(global_dir)
        for me_path in madspace_data["matrix_elements"]:
            context.load_matrix_element(
                me_path.format(device=backend), param_card_path
            )

    # set up generators
    channel_generators = [
        ms.ChannelEventGenerator.load(
            os.path.join("data", "channels", file),
            contexts,
            event_file=os.path.join(run_path, f"events.{name}.npy"),
            weight_file=os.path.join(run_path, f"weights.{name}.npy"),
            config=config,
        )
        for name, file in madspace_data["channels"].items()
    ]
    event_generator = ms.EventGenerator(
        contexts=contexts,
        channels=channel_generators,
        status_file=ms.StatusFile(os.path.join(run_path, "info.json")),
        config=config,
        seed=seed,
    )

    # run generation
    event_generator.generate()
    output_format = args.output_format
    if output_format == "compact_npy":
        event_generator.combine_to_compact_npy(
            os.path.join(run_path, "events.npy")
        )
    elif output_format == "lhe_npy":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        event_generator.combine_to_lhe_npy(
            os.path.join(run_path, "events.npy"), lhe_completer
        )
    elif output_format == "lhe":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        lhe_path = os.path.join(run_path, "events.lhe")
        event_generator.combine_to_lhe(lhe_path, lhe_completer)
        # Ship the LHE compressed, as the launcher that produced this gridpack
        # does: the file is large and very compressible, and the consumers of
        # an mg7 event file accept either form. The stdlib is used rather than
        # madgraph.various.misc.gzip because a gridpack is meant to run without
        # a madgraph installation, and copyfileobj streams the file instead of
        # holding it in memory.
        with open(lhe_path, "rb") as fin, \
                gzip.open(lhe_path + ".gz", "wb") as fout:
            shutil.copyfileobj(fin, fout)
        os.remove(lhe_path)
    else:
        raise ValueError("Unknown output format")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    try:
        main()
    except KeyboardInterrupt:
        pass
