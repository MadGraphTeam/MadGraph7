import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
import glob
import shutil
import json
import subprocess
import re
import logging
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
import resource

# Locate the madspace installation bundled alongside MadGraph.
# madgraph/__init__.py lives one level below the MadGraph root, so .parents[1]
# reaches the root and then "madspace/install" is the local install prefix.
import madgraph as _mg_pkg
_MG_ROOT = Path(_mg_pkg.__file__).parents[1]
_MADSPACE_DIR = _MG_ROOT / "madspace"
_INSTALL_DIR = _MADSPACE_DIR / "install"
if not (_INSTALL_DIR / "madspace").is_dir():
    print()
    print("You don't have madspace installed for this madgraph instance")
    print("Running the madspace installation script")
    print()

    _install_cmd = [sys.executable, str(_MADSPACE_DIR / "install.py")]
    # Expose madgraph on PYTHONPATH so the installer subprocess can import
    # cmd.ask for its prompts.
    _noninteractive = "-f" in sys.argv or not sys.stdin.isatty()
    # When the run is non-interactive (scripted / piped), install
    # non-interactively with a source build and default options (--source --yes),
    # and keep the installer away from our stdin (which may carry the run's
    # scripted card-editing commands); when interactive, let it share the
    # terminal so the user can answer.
    _install_stdin = subprocess.DEVNULL if _noninteractive else None
    if _noninteractive:
        _install_cmd += ["--source", "--yes"]
    _install_env = os.environ.copy()
    _install_env["PYTHONPATH"] = os.pathsep.join(
        [str(_MG_ROOT)] + ([_install_env["PYTHONPATH"]] if _install_env.get("PYTHONPATH") else [])
    )
    _result = subprocess.run(_install_cmd, env=_install_env, stdin=_install_stdin)
    if _result.returncode != 0:
        raise RuntimeError("madspace installation failed — see output above")
if str(_INSTALL_DIR) not in sys.path:
    sys.path.insert(0, str(_INSTALL_DIR))

if "LHAPDF_DATA_PATH" in os.environ:
    PDF_PATH = os.environ["LHAPDF_DATA_PATH"]
else:
    try:
        import lhapdf
        lhapdf.setVerbosity(0)
        PDF_PATH = lhapdf.paths()[0]
    except ImportError:
        # Do not abort at import time: lhapdf is only needed when a PDF grid is
        # actually loaded (see PdfGrid/AlphaSGrid below). Leave PDF_PATH unset
        # so that code paths which do not require an external PDF still work;
        # the missing-lhapdf error is raised lazily at the point of use.
        PDF_PATH = None

import madspace as ms
from models.check_param_card import ParamCard
from madgraph.various.banner import RunCardMG7
from madgraph.various import misc

_source_hash = subprocess.run(
    [sys.executable, str(_MADSPACE_DIR / "source_hash.py")],
    capture_output=True, text=True, check=True,
).stdout.strip()
if _source_hash != ms.SOURCE_HASH:
    print()
    print(
        "\033[1m\033[31mWARNING\033[39m: madspace source and installed binaries "
        "are not compatible (source hash mismatch) — consider recompiling "
        "madspace (e.g. `install madspace -y`)\033[0m"
    )
    print()

logger = logging.getLogger("madgraph7")
LOG_LEVEL_MAP = {
    ms.Logger.LogLevel.level_debug: logging.DEBUG,
    ms.Logger.LogLevel.level_info: logging.INFO,
    ms.Logger.LogLevel.level_warning: logging.WARNING,
    ms.Logger.LogLevel.level_error: logging.ERROR,
}
def ms_log_handler(level: ms.Logger.LogLevel, message: str):
    logger.log(LOG_LEVEL_MAP[level], message)
ms.Logger.set_log_handler(ms_log_handler)


def get_start_time():
    return time.time(), time.process_time()


def format_time(t: int, centi: bool = False):
    hours, t = divmod(t, 3600)
    minutes, seconds = divmod(t, 60)
    if centi:
        return f"{int(hours):02}:{int(minutes):02}:{seconds:02.2f}"
    else:
        return f"{int(hours):02}:{int(minutes):02}:{seconds:02.0f}"


def resolve_verbosity(verbosity: str) -> str:
    """Resolve the run_card "auto" verbosity to "pretty"/"log" depending on
    whether stdout is attached to a terminal; other values pass through
    unchanged."""
    if verbosity == "auto":
        return "pretty" if sys.stdout.isatty() else "log"
    return verbosity


def resolve_cpu_backend(build_path: str) -> str:
    """Ask the matrix-element Makefile to resolve ``cpu``.

    Given the produced shared library will have its name taken from the resolved
    backend name, we need to make sure the detection is taking place correctly
    and catch any possible error.
    """
    command = ["make", "-n", "BACKEND=cpu", "detect-backend"]
    try:
        result = subprocess.run(
            command,
            cwd=build_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not run make to resolve cpu in '{build_path}': {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu in '{build_path}': "
            f"Exit status {exc.returncode}.{detail}"
        ) from exc

    match = re.search(
        r"^BACKEND=(\S+) \(was cpu\)$", result.stdout, re.MULTILINE
    )
    if match is None or match.group(1) == "cpu":
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu in '{build_path}': "
            f"make failed to report a backend.{detail}"
        )
    return match.group(1)


@dataclass
class Channel:
    phasespace_mapping: ms.PhaseSpaceMapping
    adaptive_mapping: ms.Flow | ms.VegasMapping
    discrete_sym: ms.DiscreteSampler | ms.DiscreteFlow | None
    discrete_flavor: ms.DiscreteSampler | ms.DiscreteFlow | None
    channel_weight_indices: list[int] | None
    name: str
    active_flavors: list[int]
    event_generator: ms.ChannelEventGenerator | None = None


@dataclass
class PhaseSpace:
    mode: Literal["multichannel", "flat", "both"]
    channels: list[Channel]
    symfact: list[int | None]
    first_chan_weight_remap: list[list[int]] = field(default_factory=list)
    first_remapped_chan_count: int = 0
    second_chan_weight_remap: list[int] = field(default_factory=list)
    second_remapped_chan_count: int = 0
    prop_chan_weights: ms.PropagatorChannelWeights | None = None
    subchan_weights: ms.SubchannelWeights | None = None
    cwnet: ms.ChannelWeightNetwork | None = None


class MultiChannelData(NamedTuple):
    amp2_remaps: list[list[int]]
    symfact: list[int | None]
    topologies: list[list[ms.Topology]]
    permutations: list[list[list[int]]]
    channel_indices: list[list[int]]
    channel_weight_indices: list[list[list[int]]]
    diagram_indices: list[list[int]]
    diagram_color_indices: list[list[list[int]]]
    diagram_propagator_pdgs: list[list[list[int]]]
    active_flavors: list[list[list[int]]]
    qcd_s_channel_count: list[int]


@dataclass
class CutItem:
    observable_kwargs: dict
    min: float
    max: float
    mode: str


@dataclass
class HistItem:
    observable_kwargs: dict
    min: float
    max: float
    bin_count: int


class MadgraphProcess:
    def __init__(self):
        self.load_cards()
        self.init_backend()
        self.init_event_dir()
        self.init_context()
        self.init_cuts()
        self.init_histograms()
        self.init_generator_config()
        self.init_beam()
        self.init_subprocesses()

    def load_cards(self) -> None:
        self.run_card = RunCardMG7(os.path.join("Cards", "run_card.toml"))
        self.param_card_path = os.path.join("Cards", "param_card.dat")
        self.param_card = ParamCard(self.param_card_path)
        with open(os.path.join("SubProcesses", "subprocesses.json")) as f:
            self.subprocess_data = json.load(f)
        if self.run_card["phasespace"]["merge_subprocesses"]:
            with open(os.path.join("SubProcesses", "merged_subprocesses.json")) as f:
                self.merged_subprocess_data = json.load(f)
        else:
            self.merged_subprocess_data = None


    def init_backend(self) -> None:
        ms.set_simd_vector_size(self.run_card["run"]["simd_vector_size"])

    def init_event_dir(self) -> None:
        run_name = self.run_card["run"]["run_name"]
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
                self.run_path = f"{run_dir_prefix}{run_index:02d}"
                os.mkdir(self.run_path)
                break
            except FileExistsError:
                run_index += 1
        self.status_file = ms.StatusFile(os.path.join(self.run_path, "info.json"))

    def init_context(self) -> None:
        device_names = self.run_card["run"]["devices"]
        self.contexts = []
        self.device_types = []
        self.devices = []
        self.pool_sizes = []
        for i, device_name in enumerate(device_names):
            if ":" in device_name:
                device_type, device_index_str = device_name.split(":")
                device_index = int(device_index_str)
            else:
                device_type = device_name
                device_index = 0
            self.device_types.append(device_type)
            if device_type == "cuda":
                device = ms.cuda_device(device_index)
                pool_size = self.run_card["run"]["gpu_thread_pool_size"]
            elif device_type == "hip":
                device = ms.hip_device(device_index)
                pool_size = self.run_card["run"]["gpu_thread_pool_size"]
            else:
                device = ms.cpu_device()
                pool_size = self.run_card["run"]["cpu_thread_pool_size"]
            self.devices.append(device)
            self.pool_sizes.append(pool_size)
            self.contexts.append(ms.Context(device=device, thread_count=pool_size))

    def parse_observable(self, name: str, order_observable: str) -> dict:
        parts = name.split("-")
        sum_momenta = False
        sum_observable = False
        ordered = False
        multiparticles = self.run_card["multiparticles"]

        if len(parts) == 0:
            raise ValueError("Invalid observable name")
        elif len(parts) == 1:
            # event-level observables
            obs_name = parts[0]
            select_pids = []
        else:
            if parts[-1] == "sum":
                sum_observable = True
                obs_name = parts[-2]
                selection = parts[:-2]
            elif parts[-2] == "sum":
                sum_momenta = True
                obs_name = parts[-1]
                selection = parts[:-2]
            else:
                obs_name = parts[-1]
                selection = parts[:-1]
            select_pids = []
            order_indices = []
            for mp_name in selection:
                mp_parts = mp_name.split("_")
                if mp_parts[-1].isnumeric():
                    order_indices.append(int(mp_parts[-1]))
                    select_pids.append(multiparticles["_".join(mp_parts[:-1])])
                    ordered = True
                else:
                    order_indices.append(0)
                    select_pids.append(multiparticles[mp_name])

        return dict(
            observable=obs_name,
            select_pids=select_pids,
            sum_momenta=sum_momenta,
            sum_observable=sum_observable,
            order_observable=order_observable if ordered else None,
            order_indices=order_indices if ordered else [],
            ignore_incoming=True,
            name=name,
        )

    def init_cuts(self) -> None:
        inf = float("inf")
        order_observable = self.run_card["cuts"].get("order_by", "pt")
        self.cut_data = [
            CutItem(
                observable_kwargs=self.parse_observable(key, order_observable),
                min=values.get("min", -inf),
                max=values.get("max", inf),
                mode=values.get("mode", "all"),
            )
            for key, values in self.run_card["cuts"].items()
            if key != "order_by"
        ]

    def init_histograms(self) -> None:
        inf = float("inf")
        order_observable = self.run_card["histograms"].get("order_by", "pt")
        #TODO: add reasonable defaults for min, max, bin_count
        self.hist_data = [
            HistItem(
                observable_kwargs=self.parse_observable(key, order_observable),
                min=values["min"],
                max=values["max"],
                bin_count=values["bin_count"],
            )
            for key, values in self.run_card["histograms"].items()
            if key != "order_by"
        ]

    def ensure_pdf_set(self, pdf_set: str) -> None:
        """Make sure the requested LHAPDF set is available, downloading it if
        needed. The destination follows LHAPDF_DATA_PATH, otherwise the data
        dir of the configured lhapdf (e.g. lhapdf6 in HEPTools), otherwise a
        local directory -- and PDF_PATH is pointed at it so madspace uses it.
        Both LHAPDF_DATA_PATH and MADGRAPH_LHAPDF_CONFIG are provided by
        do_launch; nothing is downloaded when the set is already present."""
        global PDF_PATH
        data_path = os.environ.get("LHAPDF_DATA_PATH") or PDF_PATH
        if data_path and os.path.isdir(os.path.join(data_path, pdf_set)):
            PDF_PATH = data_path
            return
        lhapdf_config = os.environ.get("MADGRAPH_LHAPDF_CONFIG")
        if not lhapdf_config:
            return  # can't download; the missing-PDF error is raised below
        if not data_path:
            data_path = os.path.join(os.getcwd(), "lhapdf_pdfsets")
        try:
            from madgraph.interface.common_run_interface import CommonRunCmd
            os.makedirs(data_path, exist_ok=True)
            logger.info("PDF set %s not found; downloading into %s", pdf_set, data_path)
            CommonRunCmd.install_lhapdf_pdfset_static(lhapdf_config, data_path, pdf_set)
            PDF_PATH = data_path
        except Exception as err:
            logger.warning("Could not download PDF set %s: %s", pdf_set, err)

    def init_beam(self) -> None:
        beam_args = self.run_card["beam"]

        self.e_cm = beam_args["e_cm"]
        self.leptonic = beam_args["leptonic"]

        dynamical_scales = {
            "transverse_energy": ms.EnergyScale.transverse_energy,
            "transverse_mass": ms.EnergyScale.transverse_mass,
            "half_transverse_mass": ms.EnergyScale.half_transverse_mass,
            "partonic_energy": ms.EnergyScale.partonic_energy,
        }
        if beam_args["dynamical_scale_choice"] in dynamical_scales:
            dynamical_scale_type = dynamical_scales[beam_args["dynamical_scale_choice"]]
        else:
            raise ValueError("Unknown dynamical scale choice")
        self.scale_kwargs = dict(
            dynamical_scale_type=dynamical_scale_type,
            ren_scale_fixed=beam_args["fixed_ren_scale"],
            fact_scale_fixed=beam_args["fixed_fact_scale"],
            ren_scale=beam_args["ren_scale"],
            fact_scale1=beam_args["fact_scale1"],
            fact_scale2=beam_args["fact_scale2"],
        )

        pdf_set = beam_args["pdf"]
        self.ensure_pdf_set(pdf_set)
        if PDF_PATH is None:
            raise RuntimeError("Can't load lhapdf module. Please set LHAPDF_DATA_PATH manually")
        self.pdf_grid = ms.PdfGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}_0000.dat"))
        self.alphas_grid = ms.AlphaSGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}.info"))
        for context in self.contexts:
            self.pdf_grid.initialize_globals(context)
            self.alphas_grid.initialize_globals(context)
        self.running_coupling = ms.RunningCoupling(self.alphas_grid)

    def init_generator_config(self) -> None:
        run_args = self.run_card["run"]
        gen_args = self.run_card["generation"]
        vegas_args = self.run_card["vegas"]
        cfg = ms.GeneratorConfig()
        cfg.target_count = gen_args["events"]
        cfg.vegas_damping = vegas_args["damping"]
        cfg.max_overweight_truncation = gen_args["max_overweight_truncation"]
        cfg.freeze_max_weight_after = gen_args["freeze_max_weight_after"]
        cfg.start_batch_size = vegas_args["start_batch_size"]
        cfg.max_batch_size = vegas_args["max_batch_size"]
        cfg.survey_min_iters = gen_args["survey_min_iters"]
        cfg.survey_max_iters = gen_args["survey_max_iters"]
        cfg.survey_target_precision = gen_args["survey_target_precision"]
        cfg.optimization_patience = vegas_args["optimization_patience"]
        cfg.optimization_threshold = vegas_args["optimization_threshold"]
        cfg.cpu_batch_size = gen_args["cpu_batch_size"]
        cfg.gpu_batch_size = gen_args["gpu_batch_size"]
        cfg.verbosity = resolve_verbosity(run_args["verbosity"])
        cfg.combine_thread_count = run_args["combine_thread_pool_size"]
        cfg.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
        cfg.max_cut_repetitions = gen_args["max_cut_repetitions"]
        self.event_generator_config = cfg
        self.event_generator = None

    def init_subprocesses(self) -> None:
        self.backends = self.compile_matrix_elements()
        self.subprocesses = []
        if self.merged_subprocess_data is None:
            for subproc_id, meta in enumerate(self.subprocess_data):
                self.subprocesses.append(MadgraphSubprocess(self, meta, subproc_id))
        else:
            for subproc_id, meta in enumerate(self.merged_subprocess_data):
                self.subprocesses.append(
                    MadgraphSubprocess(self, meta, subproc_id, self.subprocess_data)
                )

    def compile_matrix_elements(self) -> list[str]:
        """Build the matrix-element library of every subprocess, and return the
        list of requested devices with 'cpu' replaced by the backend it
        resolves to on this machine.

        SubProcesses/makefile is a dispatcher over the P* directories, so a
        single 'make -j N' there builds all the subprocesses at once with one
        shared pool of N jobs: no subprocess is built with N jobs while the
        others wait, and none is limited to N/#subprocesses jobs either.
        """
        backends = self.run_card["run"]["devices"]
        if not isinstance(backends, list):
            backends = [backends]
        if not self.subprocess_data:
            return backends

        first_proc_path = self.subprocess_data[0]["path"]
        subproc_path = os.path.dirname(first_proc_path)

        # Resolve 'cpu' once (the build rules pick the best SIMD backend
        # available here), so that all subprocesses agree on the library names.
        cpu_backend = None
        if "cpu" in backends:
            cpu_backend = resolve_cpu_backend(first_proc_path)
            logger.info("Device 'cpu' resolved as '%s'", cpu_backend)
        resolved = [
            cpu_backend if backend == "cpu" else backend
            for backend in backends
        ]

        nb_core = self.run_card["run"]["cpu_thread_pool_size"]
        if not nb_core or nb_core < 0:
            nb_core = os.cpu_count() or 1

        log_path = os.path.join(self.run_path, "compile_subprocesses.log")
        for backend in resolved:
            missing = [
                meta for meta in self.subprocess_data
                if not os.path.isfile(meta["me_path"].format(device=backend))
            ]
            if not missing:
                continue
            logger.info(
                f"Start compilation of SubProcesses for device '{backend}' "
                f"({len(missing)} subprocess(es), {nb_core} parallel job(s)), "
                f"see log detail in {log_path}"
            )
            start_time = time.time()
            self.make_subprocesses(
                subproc_path, [f"BACKEND={backend}", "USEBUILDDIR=1"],
                nb_core, log_path,
            )
            logger.info(
                f"Compilation of SubProcesses done in {time.time() - start_time:.1f} s"
            )
        return resolved

    @staticmethod
    def make_subprocesses(
        subproc_path: str, args: list[str], nb_core: int, log_path: str
    ) -> None:
        """Run 'make -j nb_core' in SubProcesses/, appending the full build
        output to log_path (one run per device, so the file is appended to)."""
        command = ["make", f"-j{nb_core}"] + args
        with open(log_path, "a") as log:
            log.write(f"\n$ cd {subproc_path} && {' '.join(command)}\n")
            log.flush()
            returncode = subprocess.call(
                command, cwd=subproc_path, stdout=log, stderr=subprocess.STDOUT
            )
        if returncode != 0:
            raise RuntimeError(
                f"Compilation of the SubProcesses failed with exit status "
                f"{returncode}; see {log_path} for details"
            )

    def build_event_generator(self, phasespaces: list[PhaseSpace]) -> ms.EventGenerator:
        channel_generators = []
        for i, (subproc, phasespace) in enumerate(zip(self.subprocesses, phasespaces)):
            for integrand, channel in zip(
                subproc.build_integrands(phasespace), phasespace.channels
            ):
                if channel.event_generator is None:
                    channel.event_generator = ms.ChannelEventGenerator(
                        contexts=self.contexts,
                        integrand=integrand,
                        event_file=os.path.join(self.run_path, f"events.{i}.{channel.name}.npy"),
                        weight_file=os.path.join(self.run_path, f"weights.{i}.{channel.name}.npy"),
                        config=self.event_generator_config,
                        subprocess_index=i,
                        name=f"{i}.{channel.name}",
                        histograms=subproc.histograms
                    )
                channel_generators.append(channel.event_generator)

        event_generator = ms.EventGenerator(
            contexts=self.contexts,
            channels=channel_generators,
            status_file=self.status_file,
            config=self.event_generator_config,
        )
        unused_globals = (
            set(self.contexts[0].global_names()) - event_generator.used_globals()
        )
        for context in self.contexts:
            for global_name in unused_globals:
                context.delete_global(global_name)
        return event_generator

    def survey_phasespaces(
        self, phasespaces: list[PhaseSpace | None]
    ) -> ms.EventGenerator | None:
        ps_filtered = [ps for ps in phasespaces if ps is not None]
        if len(ps_filtered) == 0:
            return None
        event_generator = self.build_event_generator(ps_filtered)
        event_generator.survey()
        return event_generator

    def survey(self) -> None:
        phasespace_mode = self.run_card["phasespace"]["mode"]
        if phasespace_mode in ["multichannel", "both", "auto"]:
            self.phasespaces = [
                subproc.build_multichannel_phasespace()
                for subproc in self.subprocesses
            ]
            self.event_generator = self.survey_phasespaces(self.phasespaces)
        elif phasespace_mode == "flat":
            self.phasespaces = [
                subproc.build_flat_phasespace()
                for subproc in self.subprocesses
            ]
            self.event_generator = self.survey_phasespaces(self.phasespaces)
        else:
            raise ValueError("Unknown phasespace mode")

        channel_status = self.event_generator.channel_status()
        chan_offset = 0
        madnis_enabled = False
        for subproc, ps in zip(self.subprocesses, self.phasespaces):
            mean = 0.
            variance = 0.
            count_opt = 0
            for status in channel_status[chan_offset:chan_offset + len(ps.channels)]:
                mean += status.mean
                variance += status.error**2
                count_opt += status.count_opt
            rsd = (variance * count_opt)**0.5 / mean
            subproc.set_madnis_auto_settings(rsd)
            chan_offset += len(ps.channels)
            if subproc.madnis_settings["enable"]:
                madnis_enabled = True
        if not (
            phasespace_mode == "both" or (phasespace_mode == "auto" and madnis_enabled)
        ):
            return

        phasespaces_multi = self.phasespaces
        cross_sections = []
        index = 0
        for phasespace in phasespaces_multi:
            channel_count = len(phasespace.channels)
            cross_sections.append([
                abs(status.mean)
                for status in channel_status[index:index + channel_count]
            ])
            index += channel_count

        self.phasespaces = [
            subproc.simplify_phasespace(ps_multi, cross_secs)
            for subproc, ps_multi, cross_secs in zip(
                self.subprocesses, phasespaces_multi, cross_sections
            )
        ]
        if any(
            ps_multi is not ps_both
            for ps_multi, ps_both in zip(phasespaces_multi, self.phasespaces)
        ):
            self.event_generator = self.survey_phasespaces(self.phasespaces)

    def train_madnis(self) -> None:
        madnis_args = self.run_card["madnis"]
        if not any(subproc.madnis_settings["enable"] for subproc in self.subprocesses):
            return

        gen_args = self.run_card["generation"]
        run_args = self.run_card["run"]

        verbosity = resolve_verbosity(run_args["verbosity"])
        madnis_phasespaces = []
        training_args = []
        self.event_generator = None
        for subproc, phasespace in zip(self.subprocesses, self.phasespaces):
            for channel in phasespace.channels:
                channel.event_generator = None

            config = ms.MadnisConfig()
            config.learning_rate = subproc.madnis_settings["lr"]
            config.batches = subproc.madnis_settings["train_batches"]
            config.log_interval = madnis_args["log_interval"]
            config.integration_history_length = madnis_args["integration_history_length"]
            config.channel_dropping_interval = madnis_args["channel_dropping_interval"]
            config.channel_dropping_threshold = madnis_args["channel_dropping_threshold"]
            config.cpu_generator_batch_size = gen_args["cpu_batch_size"]
            config.gpu_generator_batch_size = gen_args["gpu_batch_size"]
            config.gpu_generator_batch_granularity = madnis_args["gpu_generator_batch_granularity"]
            config.generator_target_size_factor = madnis_args["generator_target_size_factor"]
            config.batch_size_offset = madnis_args["batch_size_offset"]
            config.batch_size_per_channel = subproc.madnis_settings["batch_size_per_channel"]
            config.uniform_channel_ratio = madnis_args["uniform_channel_ratio"]
            config.lr_schedule = madnis_args["lr_scheduler"]
            config.adam_beta1 = madnis_args["adam_beta1"]
            config.adam_beta2 = madnis_args["adam_beta2"]
            config.adam_eps = madnis_args["adam_eps"]
            config.adam_weight_decay = madnis_args["adam_weight_decay"]
            config.grad_clip_threshold = madnis_args["grad_clip_threshold"]
            config.buffer_capacity = madnis_args["buffer_capacity"]
            config.minimum_buffer_size = madnis_args["minimum_buffer_size"]
            config.buffered_steps = madnis_args["buffered_steps"]
            config.buffer_unweighting_quantile = madnis_args["buffer_unweighting_quantile"]
            config.fixed_cwnet_fraction = subproc.madnis_settings["fixed_cwnet_fraction"]
            config.softclip_threshold = madnis_args["softclip_threshold"]
            config.compressed_channel_weight_count = madnis_args["compressed_channel_weight_count"]
            phasespace = subproc.build_madnis(phasespace)
            madnis_phasespaces.append(phasespace)
            training_args.append(
                ms.TrainingArgs(
                    config=config,
                    integrands=subproc.build_integrands(
                        phasespace,
                        madnis_training=True,
                        drop_cuts_and_rescale=madnis_args["drop_zero_integrands"]
                    ),
                    cwnet=phasespace.cwnet,
                )
            )

        gen_context = self.contexts[0]
        opt_context = ms.Context(
            device=self.devices[0], thread_count=self.pool_sizes[0]
        )
        opt_context.copy_globals_from(gen_context)

        madnis_training = ms.MultiMadnisTraining(
            generator_context=gen_context,
            optimizer_context=opt_context,
            training_args=training_args,
            verbosity=verbosity,
            status_file=self.status_file,
        )
        madnis_training.train()
        for phasespace, active_channels in zip(
            madnis_phasespaces, madnis_training.active_channels()
        ):
            phasespace.channels = [
                phasespace.channels[index] for index in active_channels
            ]
        del madnis_training
        del opt_context
        self.phasespaces = madnis_phasespaces
        for context in self.contexts[1:]:
            context.copy_globals_from(self.contexts[0])
        self.event_generator = self.build_event_generator(madnis_phasespaces)

    def update_madnis_status_single(
        self, batch: int, batch_target: int, loss: float, lr: float, channel_count: int
    ) -> None:
        now = time.time()
        if batch + 1 < batch_target:
            if now - self.last_update_time < 0.1:
                return
            self.last_update_time = now
            progress_bar = ms.format_progress((batch + 1) / batch_target, 52)
            time_diff = now - self.madnis_wall_time
            time_str = f"{format_time(time_diff)}"
        else:
            progress_bar = ""
            wall_diff = now - self.madnis_wall_time
            cpu_diff = time.process_time() - self.madnis_cpu_time
            time_str = (
                f"{format_time(wall_diff, centi=True)} wall, "
                f"{format_time(cpu_diff, centi=True)} cpu"
            )
        batch_str = f"{batch + 1} / {batch_target}"
        self.madnis_box.set_column(1, [
            f"{batch_str:<15} {progress_bar}",
            f"{loss:>.4f}",
            f"{channel_count}",
            time_str
        ])
        self.madnis_box.print_update()

    def update_madnis_status_multi(
        self,
        subproc_id: int,
        batch: int,
        batch_target: int,
        loss: float,
        lr: float,
        channel_count: int
    ) -> None:
        now = time.time()
        subproc_count = len(self.subprocesses)
        if batch + 1 < batch_target:
            if now - self.last_update_time < 0.1:
                return
            self.last_update_time = now
            progress_bar = ms.format_progress((batch + 1) / batch_target, 34)
            progress_bar_all = ms.format_progress(
                (subproc_id * batch_target + batch + 1) / (subproc_count * batch_target),
                52
            )
            time_str = f"{format_time(now - self.madnis_wall_time)}"
            subproc_str = f"{subproc_id} / {subproc_count}"
        elif subproc_id < subproc_count - 1:
            progress_bar = ""
            progress_bar_all = ms.format_progress(
                ((subproc_id + 1) * batch_target + 1) / (subproc_count * batch_target),
                52
            )
            time_str = f"{format_time(now - self.madnis_wall_time)}"
            subproc_str = f"{subproc_id} / {subproc_count}"
        else:
            progress_bar = ""
            progress_bar_all = ""
            wall_diff = now - self.madnis_wall_time
            cpu_diff = time.process_time() - self.madnis_cpu_time
            time_str = (
                f"{format_time(wall_diff, centi=True)} wall, "
                f"{format_time(cpu_diff, centi=True)} cpu"
            )
            subproc_str = f"{subproc_count} / {subproc_count}"
        batch_str = f"{batch + 1} / {batch_target}"
        self.madnis_upper_box.set_column(1, [
            f"{subproc_str:<15} {progress_bar_all}",
            time_str,
        ])
        self.madnis_lower_box.set_row(subproc_id + 1, [
            f"{subproc_id}",
            f"{loss:>.4f}",
            f"{channel_count}",
            f"{batch_str:<15} {progress_bar}",
        ])
        self.madnis_upper_box.print_update()
        self.madnis_lower_box.print_update()

    def generate_events(self) -> None:
        start_time = get_start_time()
        self.event_generator.generate()
        output_format = self.run_card["run"]["output_format"]
        if output_format == "compact_npy":
            self.lhe_completer = None
            self.event_generator.combine_to_compact_npy(
                os.path.join(self.run_path, "events.npy")
            )
        elif output_format == "lhe_npy":
            self.lhe_completer = self.build_lhe_completer()
            self.event_generator.combine_to_lhe_npy(
                os.path.join(self.run_path, "events.npy"), self.lhe_completer
            )
        elif output_format == "lhe":
            self.lhe_completer = self.build_lhe_completer()
            lhe_path = os.path.join(self.run_path, "events.lhe")
            self.event_generator.combine_to_lhe(
                lhe_path, self.lhe_completer,
                self.build_lhe_meta(),
            )
            # Ship the LHE compressed by default. These files are large and
            # very compressible, madevent has always stored its events
            # gzipped, and every consumer here already accepts either form
            # (see _find_event_file). misc.gzip replaces events.lhe with
            # events.lhe.gz, and switches to an external multithreaded tool
            # above 256 MB.
            misc.gzip(lhe_path)
        else:
            raise ValueError("Unknown output format")
        self.save_gridpack()

    @staticmethod
    def _histogram_mean(hist):
        """Cross-section-weighted mean of a histogrammed observable."""
        values = list(hist.bin_values)
        n = len(values)
        total = sum(values)
        if n == 0 or total == 0:
            return None
        width = (hist.max - hist.min) / n
        return sum(v * (hist.min + (i + 0.5) * width)
                   for i, v in enumerate(values)) / total

    def get_result(self) -> dict:
        """Return the run result: cross-section (pb) with MC error, the number
        of (unweighted) events, and the mean of every observable declared in the
        [histograms] section. Used to build the scan summary."""
        status = self.event_generator.status()
        result = {'cross(pb)': status.mean, 'error(pb)': status.error,
                  'nb_event': status.count_unweighted}
        try:
            for hist in self.event_generator.histograms():
                mean = self._histogram_mean(hist)
                if mean is not None:
                    result['<%s>' % hist.name] = mean
        except Exception as err:
            logger.warning("could not extract observable means: %s", err)
        return result

    def _beam_info(self):
        """Return (beam_pdg_ids, beam_energies) for the LHE <init> block.

        `incoming` holds the *partonic* initial state (e.g. gluons), not the
        beam particle, so hadronic beams are protons (2212); leptonic beams are
        the incoming leptons themselves."""
        half_e = float(self.e_cm) / 2.
        if not self.leptonic:
            # hadronic collider: proton beams (p-pbar is not distinguished)
            return [2212, 2212], [half_e, half_e]
        data = self.subprocess_data[0]
        incoming = data["incoming"]
        flavor0 = data["flavors"][0]["options"][0]
        init_pdgs = flavor0[:len(incoming)]
        beam_pdgs = [pdg if abs(code) in (81, 82) else code
                     for code, pdg in zip(incoming, init_pdgs)]
        return beam_pdgs, [half_e, half_e]

    def _lhapdf_id(self):
        """Central LHAPDF id of the beam PDF set (read from its .info SetIndex),
        or -1 for a leptonic beam (no PDF)."""
        if self.leptonic:
            return -1
        pdf_set = self.run_card["beam"]["pdf"]
        info = os.path.join(PDF_PATH or "", pdf_set, "%s.info" % pdf_set)
        try:
            for line in open(info):
                if line.strip().startswith("SetIndex:"):
                    return int(line.split(":", 1)[1].strip())
        except Exception as err:
            logger.warning("could not read LHAPDF id from %s: %s", info, err)
        return -1

    def build_lhe_meta(self):
        """Build the LHE header/<init> metadata: the param_card (<slha>) and the
        run_card.toml (<MG7RunCard>) headers plus the beam/PDF/cross-section info
        needed by downstream tools (systematics, MadSpin, ...)."""
        beam_pdgs, energies = self._beam_info()
        lhaid = self._lhapdf_id()
        pdf_group = -1 if self.leptonic else 0
        status = self.event_generator.status()
        xsec, err = status.mean, status.error
        with open(self.param_card_path) as f:
            param_text = f.read()
        with open(os.path.join("Cards", "run_card.toml")) as f:
            run_text = f.read()
        headers = []
        # generation commands (model + process): read by MadSpin/reweight/...
        proc_card = os.path.join("Cards", "proc_card_mg5.dat")
        if os.path.exists(proc_card):
            with open(proc_card) as f:
                headers.append(ms.LHEHeader(name="MG5ProcCard", content=f.read()))
        headers.append(ms.LHEHeader(name="slha", content=param_text))
        headers.append(ms.LHEHeader(name="MG7RunCard", content=run_text))
        return ms.LHEMeta(
            beam1_pdg_id=beam_pdgs[0], beam2_pdg_id=beam_pdgs[1],
            beam1_energy=energies[0], beam2_energy=energies[1],
            beam1_pdf_authors=pdf_group, beam2_pdf_authors=pdf_group,
            beam1_pdf_id=lhaid, beam2_pdf_id=lhaid,
            weight_mode=3,
            # positional: the pybind arg name for max_weight is non-kwarg-safe
            processes=[ms.LHEProcess(xsec, err, xsec, 1)],
            headers=headers,
        )

    def build_lhe_completer(self):
        all_mcdata = (
            [subproc.build_multi_channel_data() for subproc in self.subprocesses]
            if self.merged_subprocess_data is None else
            [build_multi_channel_data(meta, self) for meta in self.subprocess_data]
        )
        subproc_args = [
            ms.SubprocArgs(
                topologies = [topo[0] for topo in mcdata.topologies],
                permutations = mcdata.permutations,
                diagram_indices = mcdata.diagram_indices,
                diagram_color_indices = mcdata.diagram_color_indices,
                diagram_propagator_pdgs = mcdata.diagram_propagator_pdgs,
                color_flows = meta["color_flows"],
                pdg_color_types = {
                    int(key): value
                    for key, value in meta["pdg_color_types"].items()
                },
                helicities = meta["helicities"],
                pdg_ids = [flavor["options"] for flavor in meta["flavors"]],
            )
            for mcdata, meta in zip(all_mcdata, self.subprocess_data)
        ]
        return ms.LHECompleter(
            subproc_args=subproc_args,
            bw_cutoff=self.run_card["phasespace"]["bw_cutoff"]
        )

    def save_gridpack(self) -> None:
        if not self.run_card["gridpack"]["save_gridpack"]:
            return

        gridpack_path = os.path.join(self.run_path, "gridpack")
        data_path = os.path.join(gridpack_path, "data")
        events_path = os.path.join(gridpack_path, "Events")
        os.mkdir(gridpack_path)
        os.mkdir(data_path)
        os.mkdir(events_path)
        self.contexts[0].save_globals(os.path.join(data_path, "globals"))

        channel_path = os.path.join(data_path, "channels")
        os.mkdir(channel_path)
        channel_files = {}
        for channel in self.event_generator.channels():
            name = channel.status().name
            file = f"channel{name}.json"
            channel_files[name] = file
            channel.save(os.path.join(channel_path, file))

        lib_path = os.path.join(gridpack_path, "lib")
        if self.run_card["gridpack"]["include_source"]:
            os.mkdir(lib_path)
            shutil.copytree("src", os.path.join(gridpack_path, "src"))
            shutil.copytree("SubProcesses", os.path.join(gridpack_path, "SubProcesses"))
        else:
            shutil.copytree("lib", lib_path)

        if self.run_card["gridpack"]["include_madspace_source"]:
            shutil.copytree(
                _MADSPACE_DIR,
                os.path.join(gridpack_path, "madspace"),
                ignore=shutil.ignore_patterns("build", "install"),
            )

        if self.run_card["gridpack"]["include_madspace"]:
            shutil.copytree(
                _INSTALL_DIR / "madspace",
                os.path.join(gridpack_path, "madspace", "install", "madspace"),
            )

        matrix_elements = []
        for subproc in self.subprocess_data:
            me_path = subproc["me_path"]
            matrix_elements.append(me_path)

        cards_path = os.path.join(gridpack_path, "Cards")
        os.mkdir(cards_path)
        shutil.copy(os.path.join("Cards", "param_card.dat"), cards_path)
        # Full run card with a header noting it is read-only in gridpack context.
        import io as _io
        _buf = _io.StringIO()
        self.run_card.write(_buf)
        _header = (
            "# This is the run card used to generate this gridpack.\n"
            "# Modifying this file will have no effect on gridpack execution.\n"
            "# To change event-generation settings, edit grid_run_card.toml.\n\n"
        )
        with open(os.path.join(cards_path, "run_card.toml"), 'w') as _f:
            _f.write(_header + _buf.getvalue())
        # Minimal card containing only the settings used by generate_events.
        self.run_card.write_gridpack_card(
            os.path.join(cards_path, "grid_run_card.toml"))

        bin_path = os.path.join(gridpack_path, "bin")
        os.mkdir(bin_path)
        gen_events_file = os.path.join(bin_path, "generate_events")
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "gridpack.py"), gen_events_file
        )
        os.chmod(gen_events_file, 0o755)

        data = {
            "channels": channel_files,
            "matrix_elements": matrix_elements,
            "source_hash": ms.SOURCE_HASH,
        }
        with open(os.path.join(data_path, "data.json"), "w") as f:
            json.dump(data, f)

        if self.lhe_completer is None:
            self.lhe_completer = self.build_lhe_completer()
        self.lhe_completer.save(os.path.join(data_path, "lhe.json"))

    def get_mass(self, pid: int) -> float:
        return self.param_card.get_value("mass", pid)

    def get_width(self, pid: int) -> float:
        return self.param_card.get_value("width", pid)


def clean_pids(pids: list[int]) -> list[int]:
    pids_out = []
    for pid in pids:
        pid = abs(pid)
        if pid == 81:
            pid = 1
        elif pid == 82:
            pid = 11
        elif pid == 83:
            pid = 12
        pids_out.append(pid)
    return pids_out


def pid_is_qcd(pid: int):
    return abs(pid) in [21, 1, 2, 3, 4, 5, 6, 81]


def build_topologies(
    incoming_masses: list[float],
    outgoing_masses: list[float],
    channel: dict,
    process: MadgraphProcess
) -> list[ms.Topology]:
    propagators = []
    for i, (pid, signed_pid) in enumerate(zip(
        clean_pids(channel["propagators"]), channel["propagators"]
    )):
        mass = process.get_mass(pid)
        width = process.get_width(pid)
        if i in channel["on_shell_propagators"]:
            bw_cutoff = process.run_card["phasespace"]["bw_cutoff"]
            e_min = mass - bw_cutoff * width
            e_max = mass + bw_cutoff * width
        else:
            e_min = 0
            e_max = 0
        propagators.append(ms.Propagator(
            mass=mass,
            width=width,
            integration_order=0,
            e_min=e_min,
            e_max=e_max,
            pdg_id=signed_pid,
        ))
    vertices = channel["vertices"]
    diag = ms.Diagram(
        incoming_masses, outgoing_masses, propagators, vertices
    )
    return ms.Topology.topologies(diag)


def build_multi_channel_data(
    meta: dict, process: MadgraphProcess, unmerged_meta: dict | None = None
) -> MultiChannelData:
    incoming_masses = [
        process.get_mass(pid) for pid in clean_pids(meta["incoming"])
    ]
    outgoing_masses = [
        process.get_mass(pid) for pid in clean_pids(meta["outgoing"])
    ]

    if unmerged_meta is None:
        diagram_count = meta["diagram_count"]
        amp2_remaps = [[-1] * diagram_count]
    else:
        amp2_remaps = [
            [-1] * unmerged_meta[subproc]["diagram_count"]
            for subproc in meta["subprocesses"]
        ]
    symfact = []
    topologies = []
    permutations = []
    channel_indices = []
    channel_weight_indices = []
    diagram_indices = []
    diagram_color_indices = []
    diagram_propagator_pdgs = []
    active_flavors = []
    channel_index = 0
    qcd_s_channel_count = []

    for channel in meta["channels"]:
        if unmerged_meta is None:
            topo_channel = channel
        else:
            topo_subproc = channel["subprocess"]
            topo_channel_index = channel["channel"]
            topo_channel = unmerged_meta[topo_subproc]["channels"][topo_channel_index]
        chan_topologies = build_topologies(
            incoming_masses, outgoing_masses, topo_channel, process
        )
        topo_count = len(chan_topologies)
        if topo_count == 0:
            continue

        topo = chan_topologies[0]
        s_chan_count = [0] * len(topo.decays)
        non_qcd = [False] * len(topo.decays)
        pdg_ids = [decay.pdg_id for decay in topo.decays]
        for i, pid in zip(topo.outgoing_indices, meta["outgoing"]):
            pdg_ids[i] = pid
        for decay in reversed(topo.decays):
            if len(decay.child_indices) == 0:
                continue
            if decay.index == 0 and topo.t_propagator_count > 0:
                s_chan_count[0] = sum(s_chan_count[i] for i in decay.child_indices)
                break

            is_qcd = pid_is_qcd(pdg_ids[decay.index]) and all(
                pid_is_qcd(pdg_ids[i]) for i in decay.child_indices
            )
            is_non_qcd = decay.on_shell or not is_qcd or any(
                non_qcd[i] for i in decay.child_indices
            )
            non_qcd[decay.index] = is_non_qcd
            s_chan_count[decay.index] = (
                0 if is_non_qcd else sum(s_chan_count[i] for i in decay.child_indices) + 1
            )
        qcd_s_channel_count.append(s_chan_count[0])

        diagrams = channel["diagrams"]
        chan_permutations = [d["permutation"] for d in diagrams]
        if unmerged_meta is None:
            amp2_remaps[0][diagrams[0]["diagram"]] = channel_index
        else:
            for amp2_remap, diag in zip(amp2_remaps, diagrams[0]["diagram"]):
                if diag != -1:
                    amp2_remap[diag] = channel_index

        channel_index_first = channel_index
        symfact_index_first = len(symfact)
        channel_index += 1
        symfact.extend([None] * topo_count)
        for d in diagrams[1:]:
            if unmerged_meta is None:
                amp2_remaps[0][d["diagram"]] = channel_index
            else:
                for amp2_remap, diag in zip(amp2_remaps, d["diagram"]):
                    if diag != -1:
                        amp2_remap[diag] = channel_index
            channel_index += 1
            symfact.extend(range(symfact_index_first, symfact_index_first + topo_count))

        topologies.append(chan_topologies)
        permutations.append(chan_permutations)
        channel_indices.append(list(range(channel_index_first, channel_index)))
        channel_weight_indices.append([
            [
                symfact_index_first + topo_index + i * topo_count
                for i in range(len(chan_permutations))
            ]
            for topo_index in range(topo_count)
        ])
        diagram_indices.append([d["diagram"] for d in diagrams])
        if unmerged_meta is None:
            diagram_color_indices.append([d["active_colors"] for d in diagrams])
            diagram_propagator_pdgs.append(
                [d["propagator_pdgs"] for d in diagrams]
            )
        active_flavors.append([d["active_flavors"] for d in diagrams])

    return MultiChannelData(
        amp2_remaps,
        symfact,
        topologies,
        permutations,
        channel_indices,
        channel_weight_indices,
        diagram_indices,
        diagram_color_indices,
        diagram_propagator_pdgs,
        active_flavors,
        qcd_s_channel_count,
    )


class MadgraphSubprocess:
    def __init__(
        self,
        process: MadgraphProcess,
        meta: dict,
        subproc_id: int,
        unmerged_meta: dict | None = None
    ):
        self.process = process
        self.meta = meta
        self.subproc_id = subproc_id
        self.multi_channel_data = None

        self.unmerged_meta = None
        if unmerged_meta is None:
            api_path_formats = [self.meta["me_path"]]
        else:
            api_path_formats = []
            for subproc in self.meta["subprocesses"]:
                submeta = unmerged_meta[subproc]
                api_path_formats.append(submeta["me_path"])
            if len(api_path_formats) == 1:
                self.meta = submeta
            else:
                self.unmerged_meta = unmerged_meta

        # The libraries were all built up front by MadgraphProcess.compile_matrix_elements
        all_api_paths = [
            [api_path_format.format(device=backend) for backend in self.process.backends]
            for api_path_format in api_path_formats
        ]

        self.incoming_masses = [
            self.process.get_mass(pid) for pid in clean_pids(self.meta["incoming"])
        ]
        self.outgoing_masses = [
            self.process.get_mass(pid) for pid in clean_pids(self.meta["outgoing"])
        ]
        self.particle_count = len(self.incoming_masses) + len(self.outgoing_masses)
        all_pids = clean_pids(self.meta["incoming"]) + clean_pids(self.meta["outgoing"])
        self.cuts = (
            ms.Cuts([
                ms.CutItem(
                    observable=ms.Observable(all_pids, **cut_item.observable_kwargs),
                    min=cut_item.min,
                    max=cut_item.max,
                    mode=cut_item.mode,
                )
                for cut_item in self.process.cut_data
            ])
            if len(self.process.cut_data) > 0
            else None
        )
        self.histograms = (
            ms.ObservableHistograms([
                ms.HistItem(
                    observable=ms.Observable(all_pids, **hist_item.observable_kwargs),
                    min=hist_item.min,
                    max=hist_item.max,
                    bin_count=hist_item.bin_count,
                )
                for hist_item in self.process.hist_data
            ])
            if len(self.process.hist_data) > 0
            else None
        )

        self.scale = ms.EnergyScale(
            particle_count=self.particle_count, **self.process.scale_kwargs
        )

        if self.process.run_card["run"]["dummy_matrix_element"]:
            self.matrix_elements = [None] * len(all_api_paths)
        else:
            self.matrix_elements = []
            for api_paths in all_api_paths:
                for context, api_path in zip(self.process.contexts, api_paths):
                    mat = context.load_matrix_element(
                        api_path, self.process.param_card_path
                    )
                self.matrix_elements.append(mat)

    def build_multi_channel_data(self) -> MultiChannelData:
        if self.multi_channel_data is not None:
            return self.multi_channel_data
        self.multi_channel_data = build_multi_channel_data(
            self.meta, self.process, self.unmerged_meta
        )
        return self.multi_channel_data

    def build_multichannel_phasespace(self) -> PhaseSpace:
        mcdata = self.build_multi_channel_data()
        channel_count = sum(len(topos) for topos in mcdata.topologies)
        drop_threshold = self.process.run_card["phasespace"]["drop_qcd_s_channel"]
        if drop_threshold >= 0 and channel_count > drop_threshold:
            mcdata = self.drop_qcd_s_channels(mcdata)

        channels = []
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        for channel_id, (chan_topologies, chan_permutations, chan_indices, active_flavors) in enumerate(zip(
            mcdata.topologies, mcdata.permutations, mcdata.channel_weight_indices,
            mcdata.active_flavors
        )):
            topo_count = len(chan_topologies)
            for topo_index, (topo, indices) in enumerate(zip(chan_topologies, chan_indices)):
                mapping = ms.PhaseSpaceMapping(
                    chan_topologies[0],
                    self.process.e_cm,
                    t_channel_mode=t_channel_mode,
                    cuts=self.cuts,
                    invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                    permutations=chan_permutations,
                    leptonic=self.process.leptonic,
                )
                prefix = f"subproc{self.subproc_id}.channel{channel_id}"
                if topo_count > 1:
                    prefix += f".subchan{topo_index}"
                discrete_sym, discrete_flavor = self.build_discrete(
                    len(chan_permutations), len(self.meta["flavors"]), prefix
                )
                channels.append(Channel(
                    phasespace_mapping = mapping,
                    adaptive_mapping = self.build_vegas(mapping, prefix),
                    discrete_sym = discrete_sym,
                    discrete_flavor = discrete_flavor,
                    channel_weight_indices = indices,
                    name = f"{channel_id}",
                    active_flavors = active_flavors,
                ))

        remapped_chan_count = sum(
            len(indices) for indices in mcdata.channel_indices
        )
        if self.process.run_card["phasespace"]["sde_strategy"] == "denominators":
            prop_chan_weights = ms.PropagatorChannelWeights(
                [topo[0] for topo in mcdata.topologies], mcdata.permutations,
                mcdata.channel_indices
            )
            chan_weight_remap = []
        else:
            prop_chan_weights = None
            chan_weight_remap = [
                [
                    len(mcdata.symfact) if remap == -1 else remap
                    for remap in amp2_remap
                ]
                for amp2_remap in mcdata.amp2_remaps
            ]

        if any(len(topos) > 1 for topos in mcdata.topologies):
            subchan_weights = ms.SubchannelWeights(
                mcdata.topologies, mcdata.permutations, mcdata.channel_indices
            )
        else:
            subchan_weights = None

        return PhaseSpace(
            mode="multichannel",
            channels=channels,
            first_chan_weight_remap=chan_weight_remap,
            first_remapped_chan_count=remapped_chan_count,
            symfact=mcdata.symfact,
            prop_chan_weights=prop_chan_weights,
            subchan_weights=subchan_weights,
        )

    def build_flat_phasespace(self) -> PhaseSpace:
        mapping = ms.PhaseSpaceMapping(
            self.incoming_masses + self.outgoing_masses,
            self.process.e_cm,
            mode=self.t_channel_mode(self.process.run_card["phasespace"]["flat_mode"]),
            cuts=self.cuts,
            leptonic=self.process.leptonic,
        )
        prefix = f"subproc{self.subproc_id}.flat"
        discrete_sym, discrete_flavor = self.build_discrete(
            1, len(self.meta["flavors"]), prefix
        )
        channel = Channel(
            phasespace_mapping = mapping,
            adaptive_mapping = self.build_vegas(mapping, prefix),
            discrete_sym = discrete_sym,
            discrete_flavor = discrete_flavor,
            channel_weight_indices = [0],
            name = "F",
            active_flavors = [],
        )
        if self.unmerged_meta is None:
            remap = [list(range(self.meta["diagram_count"]))]
        else:
            remap = [
                list(range(self.unmerged_meta[subproc]["diagram_count"]))
                for subproc in self.meta["subprocesses"]
            ]
        return PhaseSpace(
            mode="flat",
            channels=[channel],
            first_chan_weight_remap=remap,
            first_remapped_chan_count=1,
            symfact=[None],
        )

    def simplify_phasespace(
        self,
        multi_phasespace: PhaseSpace,
        cross_sections: list[float]
    ) -> PhaseSpace | None:
        assert multi_phasespace.mode == "multichannel"

        threshold = 1 - self.process.run_card["phasespace"]["combine_channel_threshold"]
        kept_channels = []
        tot_cs = sum(cross_sections)
        cum_cs = 0.
        seen_active_flavors = set()
        #seen_resonances = set()
        for index, (cs, chan) in sorted(
            enumerate(zip(cross_sections, multi_phasespace.channels)),
            key=lambda pair: pair[1][0],
            reverse=True
        ):
            cum_cs += cs
            has_unseen_flavors = False
            has_unseen_resonances = False
            for flavs in chan.active_flavors:
                for flav in flavs:
                    if flav not in seen_active_flavors:
                        has_unseen_flavors = True
                        seen_active_flavors.add(flav)
            #for resonance in chan.resonances:
            #    if resonance not in seen_resonances:
            #        has_unseen_resonances = True
            #        seen_resonances.add(flav)
            if has_unseen_flavors or has_unseen_resonances or cum_cs / tot_cs < threshold:
                kept_channels.append(index)
        if len(kept_channels) >= len(cross_sections) - 1:
            return multi_phasespace

        channels = []
        channel_map = {}
        symfact = []
        for old_chan_index in kept_channels:
            channel = multi_phasespace.channels[old_chan_index]
            perm_count = max(1, channel.phasespace_mapping.channel_count())
            channel_index = len(symfact)
            symfact.append(None)
            symfact.extend([channel_index] * (perm_count - 1))
            channel_map.update({
                old_index: new_index
                for new_index, old_index in enumerate(
                    channel.channel_weight_indices, start=channel_index
                )
            })
            channels.append(Channel(
                phasespace_mapping = channel.phasespace_mapping,
                adaptive_mapping = channel.adaptive_mapping,
                discrete_sym = channel.discrete_sym,
                discrete_flavor = channel.discrete_flavor,
                channel_weight_indices = list(range(
                    channel_index, channel_index + perm_count
                )),
                name = channel.name,
                active_flavors = channel.active_flavors,
                event_generator = channel.event_generator,
            ))

        flat_phasespace = self.build_flat_phasespace()
        flat_channel = flat_phasespace.channels[0]
        channels.append(Channel(
            phasespace_mapping = flat_channel.phasespace_mapping,
            adaptive_mapping = flat_channel.adaptive_mapping,
            discrete_sym = flat_channel.discrete_sym,
            discrete_flavor = flat_channel.discrete_flavor,
            channel_weight_indices = [len(symfact)],
            name = flat_channel.name,
            active_flavors = flat_channel.active_flavors,
        ))
        flat_index = len(symfact)
        symfact.append(None)
        channel_map[len(multi_phasespace.symfact)] = len(symfact)
        if multi_phasespace.subchan_weights is None and len(multi_phasespace.first_chan_weight_remap) > 0:
            first_chan_weight_remap = [
                [
                    channel_map.get(remap, flat_index)
                    for remap in cw_remap
                ]
                for cw_remap in multi_phasespace.first_chan_weight_remap
            ]
            first_remapped_chan_count = len(symfact)
            second_chan_weight_remap = []
            second_remapped_chan_count = 0
        else:
            first_chan_weight_remap = multi_phasespace.first_chan_weight_remap
            first_remapped_chan_count = multi_phasespace.first_remapped_chan_count
            chan_count = multi_phasespace.first_remapped_chan_count if multi_phasespace.subchan_weights is None else multi_phasespace.subchan_weights.channel_count()
            second_chan_weight_remap = [
                channel_map.get(i, flat_index)
                for i in range(chan_count)
            ]
            second_remapped_chan_count = len(symfact)

        return PhaseSpace(
            mode="both",
            channels=channels,
            first_chan_weight_remap=first_chan_weight_remap,
            first_remapped_chan_count=first_remapped_chan_count,
            second_chan_weight_remap=second_chan_weight_remap,
            second_remapped_chan_count=second_remapped_chan_count,
            symfact=symfact,
            prop_chan_weights=multi_phasespace.prop_chan_weights,
            subchan_weights=multi_phasespace.subchan_weights,
        )

    def drop_qcd_s_channels(self, mcdata: MultiChannelData) -> MultiChannelData:
        """Drop channels with non-resonant QCD s-channel propagators, to reduce the
        channel count for processes with many diagrams. Channels are grouped by the
        number of QCD s-channels, allowing for more if necessary to map out all flavor
        indices. Channel weights belonging to a dropped channel are left unmapped."""
        groups_by_s_count = {}
        for index, count in enumerate(mcdata.qcd_s_channel_count):
            groups_by_s_count.setdefault(count, []).append(index)

        covered_flavors = set()
        kept_groups = set()
        for s_count in sorted(groups_by_s_count):
            s_count_groups = groups_by_s_count[s_count]
            if s_count == 0:
                selected = s_count_groups
            else:
                selected = [
                    index
                    for index in s_count_groups
                    if any(
                        flav not in covered_flavors
                        for flavs in mcdata.active_flavors[index]
                        for flav in flavs
                    )
                ]
            kept_groups.update(selected)
            for index in selected:
                for flavs in mcdata.active_flavors[index]:
                    covered_flavors.update(flavs)

        kept_groups = sorted(kept_groups)
        if len(kept_groups) == len(mcdata.topologies):
            return mcdata

        amp2_remaps = [[-1] * len(remap) for remap in mcdata.amp2_remaps]
        symfact = []
        topologies = []
        permutations = []
        channel_indices = []
        channel_weight_indices = []
        diagram_indices = []
        diagram_color_indices = []
        diagram_propagator_pdgs = []
        active_flavors = []
        qcd_s_channel_count = []
        channel_index = 0

        for group in kept_groups:
            chan_topologies = mcdata.topologies[group]
            chan_permutations = mcdata.permutations[group]
            chan_diagram_indices = mcdata.diagram_indices[group]
            topo_count = len(chan_topologies)

            channel_index_first = channel_index
            symfact_index_first = len(symfact)
            for i, diag in enumerate(chan_diagram_indices):
                if self.unmerged_meta is None:
                    amp2_remaps[0][diag] = channel_index
                else:
                    for amp2_remap, d in zip(amp2_remaps, diag):
                        if d != -1:
                            amp2_remap[d] = channel_index
                if i == 0:
                    symfact.extend([None] * topo_count)
                else:
                    symfact.extend(range(symfact_index_first, symfact_index_first + topo_count))
                channel_index += 1

            topologies.append(chan_topologies)
            permutations.append(chan_permutations)
            channel_indices.append(list(range(channel_index_first, channel_index)))
            channel_weight_indices.append([
                [
                    symfact_index_first + topo_index + i * topo_count
                    for i in range(len(chan_permutations))
                ]
                for topo_index in range(topo_count)
            ])
            diagram_indices.append(chan_diagram_indices)
            if mcdata.diagram_color_indices:
                diagram_color_indices.append(mcdata.diagram_color_indices[group])
            if mcdata.diagram_propagator_pdgs:
                diagram_propagator_pdgs.append(
                    mcdata.diagram_propagator_pdgs[group]
                )
            active_flavors.append(mcdata.active_flavors[group])
            qcd_s_channel_count.append(mcdata.qcd_s_channel_count[group])

        return MultiChannelData(
            amp2_remaps,
            symfact,
            topologies,
            permutations,
            channel_indices,
            channel_weight_indices,
            diagram_indices,
            diagram_color_indices,
            diagram_propagator_pdgs,
            active_flavors,
            qcd_s_channel_count,
        )

    def set_madnis_auto_settings(self, rsd: float):
        madnis_args = self.process.run_card["madnis"]
        n_out = len(self.meta["outgoing"])
        n_events = self.process.run_card["generation"]["events"]
        is_gridpack = self.process.run_card["gridpack"]["save_gridpack"]
        train_batches = madnis_args["train_batches"]
        hidden_dim = min(max(int((7 * rsd) / 32) * 32 + 64, 64), 256)
        flow_layers = 3 if rsd < 32 else 4
        lr = min(max((10000 - train_batches) / 8000 * 7e-4 + 3e-4, 3e-4), 1e-3)
        if n_out <= 2:
            enable = False
        elif n_out == 3:
            enable = rsd > 10. or n_events > 1000000 or is_gridpack
        else:
            enable = True
        fixed_cwnet_fraction = max(0.33, 1.0 - 10000. / train_batches)
        batch_size_per_channel = min(max(int((7 * rsd) / 32) * 32 + 64, 128), 512)

        self.madnis_settings = {
            "enable": enable,
            "flow_layers": flow_layers,
            "flow_hidden_dim": hidden_dim,
            "discrete_hidden_dim": hidden_dim,
            "cwnet_hidden_dim": hidden_dim,
            "train_batches": train_batches,
            "lr": lr,
            "fixed_cwnet_fraction": fixed_cwnet_fraction,
            "batch_size_per_channel": batch_size_per_channel,
        }
        for key, value in self.madnis_settings.items():
            run_card_value = madnis_args[key]
            if run_card_value != "auto":
                self.madnis_settings[key] = run_card_value

    def build_madnis(self, phasespace: PhaseSpace) -> PhaseSpace:
        madnis_args = self.process.run_card["madnis"]
        channels = []
        for channel_id, channel in enumerate(phasespace.channels):
            prefix = f"subproc{self.subproc_id}.channel{channel_id}"
            cond_dim = 0

            flow_dim = channel.phasespace_mapping.random_dim()
            flow = ms.Flow(
                input_dim=flow_dim,
                condition_dim=cond_dim,
                prefix=prefix,
                bin_count=madnis_args["flow_spline_bins"],
                subnet_hidden_dim=self.madnis_settings["flow_hidden_dim"],
                subnet_layers=self.madnis_settings["flow_layers"],
                subnet_activation=self.activation(madnis_args["flow_activation"]),
                invert_spline=madnis_args["flow_invert_spline"],
            )
            if channel.adaptive_mapping is None:
                flow.initialize_globals(self.process.contexts[0])
            else:
                flow.initialize_from_vegas(
                    self.process.contexts[0], channel.adaptive_mapping.grid_name()
                )
            cond_dim += flow_dim

            # discrete_sym runs after the adaptive map, so it can condition on its latent.
            discrete_sym = channel.discrete_sym
            if discrete_sym is not None:
                perm_count = channel.phasespace_mapping.channel_count()
                discrete_sym = ms.DiscreteFlow(
                    option_counts=[perm_count],
                    prefix=f"{prefix}.discrete_flow_sym",
                    dims_with_prior=[],
                    condition_dim=cond_dim,
                    subnet_hidden_dim=self.madnis_settings["discrete_hidden_dim"],
                    subnet_layers=madnis_args["discrete_layers"],
                    subnet_activation=self.activation(madnis_args["discrete_activation"]),
                )
                discrete_sym.initialize_globals(self.process.contexts[0])
                cond_dim += perm_count

            discrete_flavor = channel.discrete_flavor
            if discrete_flavor is not None:
                discrete_flavor = ms.DiscreteFlow(
                    option_counts=[len(self.meta["flavors"])],
                    prefix=f"{prefix}.discrete_flow_flavor",
                    dims_with_prior=[0],
                    condition_dim=cond_dim,
                    subnet_hidden_dim=self.madnis_settings["discrete_hidden_dim"],
                    subnet_layers=madnis_args["discrete_layers"],
                    subnet_activation=self.activation(madnis_args["discrete_activation"]),
                )
                discrete_flavor.initialize_globals(self.process.contexts[0])

            channels.append(Channel(
                phasespace_mapping = channel.phasespace_mapping,
                adaptive_mapping = flow,
                discrete_sym = discrete_sym,
                discrete_flavor = discrete_flavor,
                channel_weight_indices = channel.channel_weight_indices,
                name = channel.name,
                active_flavors = channel.active_flavors,
            ))

        return PhaseSpace(
            mode="both",
            channels=channels,
            first_chan_weight_remap=phasespace.first_chan_weight_remap,
            first_remapped_chan_count=phasespace.first_remapped_chan_count,
            second_chan_weight_remap=phasespace.second_chan_weight_remap,
            second_remapped_chan_count=phasespace.second_remapped_chan_count,
            symfact=phasespace.symfact,
            cwnet=self.build_cwnet(len(phasespace.symfact)),
            prop_chan_weights=phasespace.prop_chan_weights,
            subchan_weights=phasespace.subchan_weights,
        )

    def build_vegas(self, mapping: ms.PhaseSpaceMapping, prefix: str) -> ms.VegasMapping:
        if not self.process.run_card["vegas"]["enable"]:
            return None

        vegas = ms.VegasMapping(
            mapping.random_dim(),
            self.process.run_card["vegas"]["bins"],
            prefix,
        )
        for context in self.process.contexts:
            vegas.initialize_globals(context)
        return vegas

    def build_discrete(
        self, permutation_count: int, flavor_count: int, prefix: str
    ) -> tuple[ms.DiscreteSampler | None, ms.DiscreteSampler | None]:
        is_adaptive = self.process.run_card["phasespace"]["adaptive_symmetry_sampling"]
        if is_adaptive and permutation_count > 1:
            discrete_sym = ms.DiscreteSampler(
                [permutation_count], f"{prefix}.discrete_sym"
            )
            for context in self.process.contexts:
                discrete_sym.initialize_globals(context)
        else:
            discrete_sym = None

        if flavor_count > 1:
            discrete_flavor = ms.DiscreteSampler(
                [flavor_count], f"{prefix}.discrete_flavor", [0]
            )
            for context in self.process.contexts:
                discrete_flavor.initialize_globals(context)
        else:
            discrete_flavor = None

        return discrete_sym, discrete_flavor

    def build_cwnet(self, channel_count: int) -> ms.ChannelWeightNetwork:
        #if channel_count == 1:
        #    return None
        madnis_args = self.process.run_card["madnis"]
        cwnet = ms.ChannelWeightNetwork(
            channel_count=channel_count,
            particle_count=self.particle_count,
            hidden_dim=self.madnis_settings["cwnet_hidden_dim"],
            layers=madnis_args["cwnet_layers"],
            activation=self.activation(madnis_args["cwnet_activation"]),
            prefix=f"subproc{self.subproc_id}.cwnet",
        )
        cwnet.initialize_globals(self.process.contexts[0])
        return cwnet

    def t_channel_mode(self, name: str) -> ms.PhaseSpaceMapping.TChannelMode:
        modes = {
            "propagator": ms.PhaseSpaceMapping.propagator,
            "rambo": ms.PhaseSpaceMapping.rambo,
            "chili": ms.PhaseSpaceMapping.chili,
        }
        if name in modes:
            return modes[name]
        else:
            raise ValueError(f"Invalid t-channel mode '{name}'")

    def activation(self, name: str) -> ms.MLP.Activation:
        activations = {
            "relu": ms.MLP.relu,
            "leaky_relu": ms.MLP.leaky_relu,
            "elu": ms.MLP.elu,
            "gelu": ms.MLP.gelu,
            "sigmoid": ms.MLP.sigmoid,
            "softplus": ms.MLP.softplus,
        }
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Invalid activation function '{name}'")

    def build_integrands(
        self,
        phasespace: PhaseSpace,
        madnis_training: bool = False,
        drop_cuts_and_rescale: bool = False
    ) -> list[ms.Integrand]:
        flavors = []
        flavor_remap = []
        flavor_factors = []
        flavor_mirror = []
        flavor_diff_xs_indices = []
        flavor_subproc_indices = []
        flavor_per_subproc_remap = []

        for flav in self.meta["flavors"]:
            if self.unmerged_meta is not None:
                diff_xs_index = flav["subprocess"]
                subproc_index = self.meta["subprocesses"][diff_xs_index]
                ps_flavor = flav["flavor"]
                flavor_diff_xs_indices.append(diff_xs_index)
                flavor_subproc_indices.append(subproc_index)
                flavor_per_subproc_remap.append(ps_flavor)
                flav = self.unmerged_meta[subproc_index]["flavors"][ps_flavor]
            flavors.append(flav["options"][0])
            flavor_remap.append(flav["index"])
            flavor_factors.append(len(flav["options"]))
            flavor_mirror.append(flav["mirror"])

        cross_sections = []
        for matrix_element in self.matrix_elements:
            if matrix_element:
                mat = ms.MatrixElement(
                    matrix_element,
                    ms.Integrand.matrix_element_inputs,
                    ms.Integrand.matrix_element_outputs,
                    True,
                )
            else:
                #TODO: not working in merged mode
                mat = ms.MatrixElement(
                    0xBADCAFE,
                    self.particle_count,
                    ms.Integrand.matrix_element_inputs,
                    ms.Integrand.matrix_element_outputs,
                    self.meta["diagram_count"],
                    True,
                )
            pdf_grid = None if self.process.leptonic else self.process.pdf_grid
            pdf_arg = None if self.process.leptonic else ms.CachedPdf()
            cross_sections.append(
                ms.DifferentialCrossSection(
                    matrix_element=mat,
                    cm_energy=self.process.e_cm,
                    running_coupling=None,
                    energy_scale=ms.CachedScale(),
                    pid_options=[],
                    pdf1=pdf_arg,
                    pdf2=pdf_arg,
                    input_momentum_fraction=True,
                )
            )
        partial_weights = self.process.run_card["generation"]["systematics"]
        madnis_args = self.process.run_card["madnis"]
        integrands = []
        for channel in phasespace.channels:
            integrands.append(ms.Integrand(
                channel.phasespace_mapping,
                cross_sections,
                channel.adaptive_mapping,
                channel.discrete_sym,
                channel.discrete_flavor,
                flavors,
                pdf_grid,
                self.process.running_coupling,
                self.scale,
                phasespace.prop_chan_weights,
                phasespace.subchan_weights,
                phasespace.cwnet,
                phasespace.first_chan_weight_remap,
                phasespace.first_remapped_chan_count,
                phasespace.second_chan_weight_remap,
                phasespace.second_remapped_chan_count,
                madnis_training,
                drop_cuts_and_rescale,
                partial_weights,
                channel.channel_weight_indices,
                channel.active_flavors,
                flavor_remap,
                flavor_factors,
                flavor_mirror,
                flavor_diff_xs_indices,
                flavor_subproc_indices,
                flavor_per_subproc_remap,
                madnis_args["compressed_channel_weight_count"]
            ))
        #print(integrands[1].function())
        #for i in integrands: print(i.function())
        return integrands

    def train_madnis(self, phasespace: PhaseSpace, status_func) -> None:
        # do import here to make pytorch and MadNIS optional dependencies
        from .train_madnis import train_madnis
        train_madnis(
            self.build_integrands(phasespace, madnis_training=True),
            phasespace,
            self.process.run_card["madnis"],
            self.process.contexts[0],
            status_func
        )


def load_mg5_options() -> dict:
    """Read the tool paths from the MG5aMC configuration so the launcher knows
    which optional programs (Pythia8/Delphes/MadSpin/reweight/analysis) are
    available.  Relative *_path entries are resolved against the MG5aMC root."""

    import madgraph
    mg5dir = os.path.dirname(os.path.dirname(os.path.abspath(madgraph.__file__)))

    options = {
        'pythia-pgs_path': None, 'pythia8_path': None, 'madanalysis_path': None,
        'madanalysis5_path': None, 'exrootanalysis_path': None, 'delphes_path': None,
        'rivet_path': None, 'contur_path': None, 'f2py_compiler': None,
        'lhapdf': None, 'timeout': 0,
        'mg5amc_py8_interface_path': None, 'heptools_install_dir': None,
    }
    config_files = [os.path.join(mg5dir, 'input', 'mg5_configuration.txt')]
    home = os.environ.get('HOME')
    if home:
        config_files.append(os.path.join(home, '.mg5', 'mg5_configuration.txt'))
        config_files.append(os.path.join(
            os.environ.get('XDG_CONFIG_HOME', os.path.join(home, '.config')),
            'mg5_configuration.txt'))
    for cfg in config_files:
        if not os.path.exists(cfg):
            continue
        with open(cfg) as fsock:
            for line in fsock:
                line = line.split('#', 1)[0]
                if '=' not in line:
                    continue
                name, value = (x.strip() for x in line.split('=', 1))
                if name not in options or value in ('', 'None'):
                    continue
                if name.endswith('_path') and value.startswith('.'):
                    value = os.path.join(mg5dir, value)
                options[name] = value
    options['mg5_path'] = mg5dir  # enables MadSpin/reweight
    return options


def build_selector_cmd():
    """Build the (monkey-patched) mother command + merged switch/card selector
    used by the mg7 output.  Returns the selector *class* and a mother instance
    understood by AskRun/AskforEditCard."""

    from madgraph.interface.common_run_interface import CommonRunCmd
    from madgraph.interface.extended_cmd import Cmd
    from madgraph.interface.madevent_interface import AskRunEditCard

    class MG7Cmd(Cmd):

        def __init__(self):
            super().__init__(".", {})
            self.me_dir = "."
            self.options = load_mg5_options()
            self.plugin_path = []
            self.proc_characteristics = {'grouped_matrix': False, 'limitations': []}

        def keep_cards(self, need_card=[], ignore=[]):
            return CommonRunCmd.keep_cards(self, need_card, ignore)

        def do_open(self, line):
            CommonRunCmd.do_open(self, line)

        def check_open(self, args):
            CommonRunCmd.check_open(self, args)

        def do_compute_widths(self, line):
            # The interactive card editor delegates 'auto' width computation to
            # the mother interface. Reuse the runtime helper (madgraph subprocess
            # + the model stored at output time). ``line`` looks like
            # "<pdgs> --path=<param_card> [--nlo]"; we only need the card path.
            m = re.search(r'--path=(\S+)', line or "")
            path = m.group(1) if m else os.path.join("Cards", "param_card.dat")
            compute_auto_widths(path)
            # return an empty mapping: the caller iterates out.items() for the
            # small-width treatment, which mg7 does not apply.
            return {}

    from madgraph.various import banner as _banner_mod
    from madgraph.various import misc as _misc

    class MG7Selector(AskRunEditCard):
        """Merged switch/card question for the mg7 output.

        The mg7 run_card is a TOML file, so define_paths/init_run/do_set are
        overridden *on this class* (not globally on AskforEditCard) to treat it
        as such. Scoping them here is important: MadSpin/reweight spawn their own
        legacy madevent runs in the same process, and a global patch would break
        their (run_card.dat based) card editing."""

        # param_card + the TOML run_card are always offered
        always_cards = ['param_card.dat', 'run_card.toml']
        optional_cards = []

        def define_paths(self, **opt):
            super().define_paths(**opt)
            self.paths["run"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            self.paths["run_card.toml"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
            # the TOML run_card uses its own default file (concrete defaults
            # written at output time); this powers "set <param> default".
            self.paths["run_default"] = os.path.join(self.me_dir, "Cards", "run_card_default.toml")

        def init_run(self, cards):
            # Make sure the run_card is loaded as a RunCardMG7 (the generic
            # editor does not recognise run_card.toml), else "set <param>" fails.
            out = super().init_run(cards)
            if not isinstance(getattr(self, "run_card", None), RunCardMG7):
                toml_path = self.paths.get("run") or os.path.join(
                    self.me_dir, "Cards", "run_card.toml")
                if os.path.exists(toml_path):
                    try:
                        # allow_scan so a run_card holding scan:[...] values loads
                        with _misc.TMP_variable(_banner_mod.RunCard, "allow_scan", True):
                            self.run_card = RunCardMG7(toml_path, consistency="warning")
                        self.run_set = list(self.run_card.keys())
                    except Exception as err:
                        logger.warning("could not load %s: %s", toml_path, err)
            if isinstance(getattr(self, "run_card", None), RunCardMG7):
                self.run_card.allow_scan = True
            return getattr(self, "run_set", out)

        def do_set(self, line, *args, **kwargs):
            # madevent-style shortcuts (lhc/lep/fixed_scale/no_parton_cut), cut
            # editing, energy units and arithmetic/mass expressions on the TOML
            # run_card, intercepted before delegating to the generic editor.
            targs = self.split_arg(line)
            run_card = getattr(self, "run_card", None)
            if isinstance(run_card, RunCardMG7) and targs:
                start = 1 if targs[0] == "run_card" else 0
                if len(targs) > start:
                    name = targs[start]
                    nlow = name.lower()
                    rest = " ".join(targs[start + 1:]).split("#")[0].strip()
                    masses = run_card.get_mass_shortcuts(getattr(self, "param_card", None))

                    if nlow in ("no_parton_cut", "nocut", "no_cut"):
                        run_card.remove_all_cut()
                        logger.info("removing all cuts from the run_card.toml")
                        self.modified_card.add("run")
                        return
                    if nlow in ("lhc", "lep", "ilc", "lcc") and rest:
                        ecm = run_card.set_collider(nlow, rest, masses)
                        logger.info("set %s collider: e_cm = %s GeV", nlow, ecm)
                        self.modified_card.add("run")
                        return
                    if nlow == "fixed_scale" and rest:
                        val = run_card.set_fixed_scale(rest, masses)
                        logger.info("set fixed scales to %s GeV", val)
                        self.modified_card.add("run")
                        return

                    if rest and run_card.is_cut_name(name):
                        cut, bound, val = run_card.set_cut(name, run_card.evaluate(rest, masses))
                        logger.info("modify cut %s.%s of the run_card.toml to %s", cut, bound, val)
                        self.modified_card.add("run")
                        return

                    # legacy madevent run_card names -> mg7 "section.key", so a
                    # madevent-style launch script ("set nevents 500", "set
                    # use_syst F", "set bwcutoff 10", ...) edits the mg7
                    # run_card.toml verbatim. RunCardMG7._LO_SCALAR_MAP is the
                    # same rename table used by the LO->MG7 run_card conversion.
                    if rest and nlow in run_card._LO_SCALAR_MAP \
                            and nlow not in [k.lower() for k in run_card.keys()]:
                        target = run_card._LO_SCALAR_MAP[nlow]
                        run_card.set(target, rest, user=True)
                        logger.info("set %s (mg7 %s) of the run_card.toml to %s",
                                    nlow, target, run_card[target])
                        self.modified_card.add("run")
                        return

                    if rest and nlow in [k.lower() for k in run_card.keys()]:
                        current = run_card[nlow]
                        if isinstance(current, (int, float)) and not isinstance(current, bool):
                            resolved = run_card.evaluate(rest, masses)
                            if not isinstance(resolved, str):
                                prefix = "run_card " if start == 1 else ""
                                line = "%s%s %s" % (prefix, name, resolved)
            return super().do_set(line, *args, **kwargs)

    return MG7Selector, MG7Cmd()


def ask_edit_cards() -> dict:
    """Single (MadDM-style) question letting the user both pick which programs
    to run after generation and edit the associated cards.  Returns the switch
    dict describing the selected tools."""

    selector_class, mother = build_selector_cmd()
    # path_msg is what makes Cmd.check_answer_in_input_file accept a bare path as
    # an answer (its "elif path:" branch), so that a scripted launch can hand the
    # question a card/banner path -- as the question itself advertises -- and have
    # it replace the corresponding card. Without it the path is rejected ("This
    # answer is not valid for current question") and the default is used instead.
    try:
        switch, question = mother.ask('', '0', [], path_msg='enter path',
                                      ask_class=selector_class,
                                      mode='auto', line_args=[], force=False,
                                      return_instance=True)
    except BaseException:
        # interrupted (e.g. ctrl-C) before the prune below could run: prune
        # here too, using the instance ask() stashed pre-interrupt, or the
        # eagerly-materialised tool cards get misread as "left ON" next time.
        instance = getattr(mother, '_last_ask_instance', None)
        if instance is not None and hasattr(instance, 'active_cards'):
            try:
                mother.keep_cards(instance.active_cards())
            except Exception as error:
                logger.debug('could not prune tool cards after aborted launch question: %s', error)
        raise
    switch = dict(switch)
    prune_unselected_tool_cards(mother, question, switch)
    return switch


def prune_unselected_tool_cards(mother, question, switch) -> None:
    """Hide the cards of the tools that were NOT selected in the question.

    The question materialises every candidate tool card (copying the
    *_default.dat) so that it can offer them all for edition; madevent's
    ask_run_configuration prunes them again afterwards (keep_cards on the
    selected cards), but the mg7 launcher had no such cleanup. The materialised
    cards therefore persisted, and a later re-launch of the same output saw them
    all present and defaulted every switch to ON (set_default_<tool> keys off
    card presence). Prune here too, using the question's own switch->card map, so
    a re-launch reflects the previous selection instead of turning everything on.
    """
    try:
        keep = list(question.always_cards)
        for spec in question.switch_cards:
            if spec['on'](switch):
                keep.append(spec['card'])
        mother.keep_cards(keep)
    except Exception as error:
        logger.debug('could not prune unselected tool cards: %s', error)


def _find_event_file(run_path):
    """Locate the LHE event file produced by generate_events (if any)."""
    for name in ("events.lhe", "events.lhe.gz"):
        path = os.path.join(run_path, name)
        if os.path.exists(path):
            return path
    return None


def _report_failure(log, what, error, directory=None):
    """Log a post-processing failure and write the full traceback to a file
    (whose path is printed) so the problem can be investigated."""
    import traceback
    if not directory or not os.path.isdir(directory):
        directory = os.getcwd()
    path = os.path.join(directory, "%s_crash.log" % what.replace(" ", "_"))
    try:
        with open(path, "w") as fsock:
            fsock.write(traceback.format_exc())
    except Exception:
        path = None
    log.warning("%s failed: %s", what, error)
    if path:
        log.warning("full traceback written to: %s", path)


def _off(value) -> bool:
    """True when a switch value means the tool was not selected."""
    return value in (None, "OFF", "Not Avail.", "Not Avail. (numpy missing)")


_TOOL_LOGGING_READY = False


def _setup_logging():
    """Make the reused madevent tool drivers' progress visible on screen.

    The drivers already narrate what they do through logger.info /
    update_status ("Running Pythia8 [arXiv:...]", "Splitting .lhe event
    file...", the live Idle/Running/Completed job counters, "Running
    MadSpin", ...), but the standalone mg7 launcher never attaches a handler
    to the 'madgraph'/'madevent' loggers, so all of it is swallowed. Attach a
    colored INFO console handler to them (idempotent)."""
    global _TOOL_LOGGING_READY
    if _TOOL_LOGGING_READY:
        return
    try:
        import madgraph.interface.coloring_logging  # registers ColorFormatter
        formatter = logging.ColorFormatter("%(message)s")
    except Exception:
        formatter = logging.Formatter("%(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler._mg7_tool_handler = True
    for name in ("madgraph", "madevent", "cmdprint", "madgraph7"):
        lg = logging.getLogger(name)
        if not any(getattr(h, "_mg7_tool_handler", False) for h in lg.handlers):
            lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    _TOOL_LOGGING_READY = True


def run_selected_tools(switch, process) -> None:
    """Run the optional post-processing programs selected in the merged
    question on the generated events.

    All the tools are driven through :class:`MG7RunCmd`, a thin adapter around
    the standard madevent run interface (``MadEventCmd``). This means the mg7
    output reuses the *exact* madevent tool drivers -- in particular Pythia8 is
    showered in parallel over the cluster/multicore backend, identically to a
    madevent run -- instead of re-implementing each tool here. The command
    sequence mirrors ``MadEventCmd.do_launch`` (reweight, MadSpin, MA5 parton,
    shower, Delphes, MA5 hadron, Rivet); each tool checks its own card, so a
    tool is invoked only when its switch is on.
    """
    log = logging.getLogger("madevent")

    active = {k: v for k, v in switch.items() if not _off(v)}
    if not active:
        return

    lhe_path = _find_event_file(process.run_path)
    if lhe_path is None:
        log.warning("No LHE event file in %s; cannot run %s.",
                    process.run_path, ", ".join(sorted(active)))
        return

    run_name = os.path.basename(os.path.dirname(os.path.abspath(lhe_path)))
    run_dir = os.path.dirname(os.path.abspath(lhe_path))

    # MadAnalysis5 hadron level analyses the shower/detector output, so it only
    # makes sense when a shower ran (mirrors madevent's card gating:
    # analysis == 'MadAnalysis5' and shower != 'OFF').
    ma5 = switch.get("analysis") == "MadAnalysis5"
    showered = not _off(switch.get("shower"))
    tools = [t for t, on in (
        ("reweighting", not _off(switch.get("reweight"))),
        ("MadSpin", not _off(switch.get("madspin"))),
        ("MadAnalysis5 (parton level)", ma5),
        ("Pythia8 shower", switch.get("shower") == "Pythia8"),
        ("Delphes", switch.get("detector") == "Delphes"),
        ("MadAnalysis5 (hadron level)", ma5 and showered),
        ("Rivet", switch.get("analysis") == "Rivet"),
    ) if on]
    log.info("")
    log.info("Post-processing the generated events with: %s", ", ".join(tools))

    try:
        from madgraph.iolibs.template_files.mg7.run_interface import MG7RunCmd
        cmd = MG7RunCmd(os.getcwd(), load_mg5_options(), run_name, lhe_path)
        cmd.set_run_name(run_name, None, "parton")
    except Exception as error:
        _report_failure(log, "post-processing setup", error, run_dir)
        return

    def run(tool, command):
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: %s", tool)
        log.info(bar)
        start = time.time()
        try:
            cmd.exec_cmd(command, postcmd=False, printcmd=False)
        except Exception as error:
            _report_failure(log, tool, error, run_dir)
        else:
            log.info("  -> %s done (%.1fs)", tool, time.time() - start)

    # Same order as MadEventCmd.do_launch. The commands take no run name and so
    # act on cmd.run_name: reweight runs on the generated events, then
    # decay_events (MadSpin) repoints cmd.run_name to the "<run>_decayed_i" run,
    # so the shower and the analyses that follow act on the decayed events --
    # exactly as a madevent run does. Each driver still guards on its own card.
    if not _off(switch.get("reweight")):
        run("reweight", "reweight -from_cards")
    if not _off(switch.get("madspin")):
        run("MadSpin", "decay_events -from_cards")
    if ma5:
        run("MadAnalysis5 (parton)", "madanalysis5_parton --no_default")
    if switch.get("shower") == "Pythia8":
        run("Pythia8 shower", "shower --no_default")
    if switch.get("detector") == "Delphes":
        run("Delphes", "delphes --no_default")
    # hadron-level MA5 needs the shower/detector output: only run it if a shower
    # was requested (otherwise there is no hadron-level event file to analyse).
    if ma5 and showered:
        run("MadAnalysis5 (hadron)", "madanalysis5_hadron --no_default")
    if switch.get("analysis") == "Rivet":
        run("Rivet", "rivet --no_default")

    # Finalize the run exactly as MadEventCmd.do_launch does after the
    # shower/detector/analysis tools: store_result() processes the deferred
    # "to_store" actions -- in particular it gzips the Pythia8 HepMC to
    # <tag>_pythia8_events.hepmc.gz (the pythia8_card default is HEPMCoutput:file
    # = hepmc.gz). The deferred Rivet job reads exactly that .gz path, so this
    # must run before the Rivet/Contur postprocessor below.
    try:
        cmd.store_result()
    except Exception as error:
        _report_failure(log, "store_result", error, run_dir)

    if switch.get("analysis") == "Rivet":
        # do_rivet only *prepared* the run (wrote Events/<run>/run_rivet.sh) and
        # deferred execution to the postprocessor -- the rivet_card default has
        # run_rivet_later = True, so with --no_default it logged "Skipping Rivet
        # for now, passing it to postprocessor" and appended the run to
        # cmd.postprocessing_dirs. MadEventCmd.do_launch runs that postprocessor
        # at the end of a run; do the same here so the .yoda (and, when
        # run_contur = True, the Contur limits) are actually produced.
        bar = "=" * 60
        log.info("")
        log.info(bar)
        log.info("  post-processing step: Rivet/Contur postprocessing")
        log.info(bar)
        start = time.time()
        try:
            cmd.postprocessing()
        except Exception as error:
            _report_failure(log, "Rivet-Contur postprocessing", error, run_dir)
        else:
            log.info("  -> Rivet/Contur postprocessing done (%.1fs)",
                     time.time() - start)


def _add_time_of_flight(lhe_path, threshold, param_card_path, log):
    """Add invariant-lifetime (vtim) information to the LHE events, drawing each
    unstable particle's decay length from its width (in mm). Mirrors
    common_run_interface.CommonRunCmd.do_add_time_of_flight but reads the widths
    from the local param_card instead of the LHE banner."""
    import random
    try:
        import madgraph.various.lhe_parser as lhe_parser
    except ImportError:
        import internal.lhe_parser as lhe_parser
    from models import check_param_card as param_card_mod
    from madgraph.various import misc as _misc
    import madgraph.iolibs.files as files

    need_zip = lhe_path.endswith('.gz')
    if need_zip:
        _misc.gunzip(lhe_path)
        lhe_path = lhe_path[:-3]

    param_card = param_card_mod.ParamCard(param_card_path)
    cst = 6.58211915e-25   # hbar in GeV s
    c = 299792458000       # speed of light in mm/s
    log.info('Adding time of flight information on %s', lhe_path)
    lhe = lhe_parser.EventFile(lhe_path)
    out = open('%s_2vertex.lhe' % lhe_path, 'w')
    out.write(lhe.banner)
    for event in lhe:
        for particle in event:
            # default=0 -> particles without a decay entry are treated as stable
            width = param_card['decay'].get((abs(particle.pid),), 0.).value
            if width:
                vtim = c * random.expovariate(width / cst)
                if vtim > threshold:
                    particle.vtim = vtim
        out.write(str(event))
    out.write('</LesHouchesEvents>\n')
    out.close()
    lhe.close()
    files.mv('%s_2vertex.lhe' % lhe_path, lhe_path)
    if need_zip:
        _misc.gzip(lhe_path)


def _lhapdf_config_path():
    """Best-effort path to lhapdf-config so systematics can import the python
    lhapdf module (required to compute PDF/scale variations)."""
    cfg = os.environ.get("MADGRAPH_LHAPDF_CONFIG")
    if cfg and os.path.exists(cfg):
        return cfg
    if PDF_PATH:
        # PDF_PATH is <prefix>/share/LHAPDF -> <prefix>/bin/lhapdf-config
        cand = os.path.join(os.path.dirname(os.path.dirname(PDF_PATH)),
                            "bin", "lhapdf-config")
        if os.path.exists(cand):
            return cand
    import shutil
    return shutil.which("lhapdf-config")


def _run_systematics(lhe_path, cfg, log):
    """Run systematics.py (scale/PDF variations) on the LHE file, reusing the
    exact worker madevent uses (systematics.call_systematics)."""
    try:
        import madgraph.various.systematics as systematics
    except ImportError:
        import internal.systematics as systematics

    def fmt(vals):
        return ','.join(str(v) for v in vals)

    opts = []
    if cfg.get('systematics_mur'):
        opts.append('--mur=%s' % fmt(cfg['systematics_mur']))
    if cfg.get('systematics_muf'):
        opts.append('--muf=%s' % fmt(cfg['systematics_muf']))
    if cfg.get('systematics_pdf'):
        opts.append('--pdf=%s' % fmt(cfg['systematics_pdf']))
    extra = cfg.get('systematics_str_options', '')
    if extra:
        opts.extend(extra.split())
    # The mg7 LHE has no <mgrwt> block, so systematics cannot read the per-event
    # LO reweighting info. For a process with a single alpha_s power (stored in
    # SubProcesses/proc_characteristics at output time) it can be reconstructed
    # from the events: pass that power as --lo_nqcd. -1 means the QCD power is
    # not uniform, so the reconstruction is not applicable.
    if not any(o.startswith('--lo_nqcd') for o in opts):
        try:
            from madgraph.various import banner as _banner_mod
        except ImportError:
            import internal.banner as _banner_mod
        pc_path = os.path.join('SubProcesses', 'proc_characteristics')
        if os.path.exists(pc_path):
            try:
                nqcd = int(_banner_mod.ProcCharacteristic(pc_path)['single_qcd_order'])
            except Exception:
                nqcd = -1
            if nqcd >= 0:
                opts.append('--lo_nqcd=%d' % nqcd)
    # tell systematics where to find lhapdf (so it can link the python module)
    if not any(o.startswith('--lhapdf_config') for o in opts):
        lhapdf_config = _lhapdf_config_path()
        if lhapdf_config:
            opts.append('--lhapdf_config=%s' % lhapdf_config)

    log.info('Running systematics on %s %s', lhe_path, ' '.join(opts))
    systematics.call_systematics([lhe_path, lhe_path] + opts,
                                 log=lambda x: log.info(str(x)))


def run_lhe_postprocessing(process) -> None:
    """Run the LHE-level post-processings configured in the run_card
    [postprocessing] section on the generated event file (displaced-vertex
    time-of-flight and systematics). Only applies when an LHE file exists."""
    lhe_path = _find_event_file(process.run_path)
    if lhe_path is None:
        return
    # process.run_card is a RunCardMG7; the section view exposes .get(key, def)
    try:
        cfg = process.run_card["postprocessing"]
    except Exception:
        return
    log = logging.getLogger('madevent')

    if cfg.get('systematics'):
        try:
            _run_systematics(lhe_path, cfg, log)
        except Exception as error:
            _report_failure(log, "systematics computation", error,
                            os.path.dirname(lhe_path))

    tof = cfg.get('time_of_flight', -1.0)
    try:
        if tof is not None and float(tof) >= 0:
            _add_time_of_flight(lhe_path, float(tof), process.param_card_path, log)
    except Exception as error:
        _report_failure(log, "add_time_of_flight", error,
                        os.path.dirname(lhe_path))


def compute_auto_widths(param_card_path=os.path.join("Cards", "param_card.dat")) -> None:
    """Fill any width set to ``auto`` in the param_card, using madgraph and the
    model stored at output time (``SubProcesses/model.txt``), and write the
    result back into the card. A no-op when the card has no ``auto`` width.

    Called before every generation, so a single run and each scan point (whose
    param_card is rewritten just before ``run_single``) both get model-computed
    widths."""
    try:
        with open(param_card_path) as f:
            text = f.read()
    except OSError:
        return
    # matches "DECAY <pdg> auto" (optionally "auto@NLO"), as in
    # common_run_interface.static_check_param_card
    pdgs = re.findall(r"(?im)^\s*decay\s+([+-]?\d+)\s+auto", text)
    if not pdgs:
        return
    pdgs = list(dict.fromkeys(pdgs))  # de-duplicate, keep order

    model_file = os.path.join("SubProcesses", "model.txt")
    if not os.path.exists(model_file):
        logger.warning(
            "The param_card requests 'auto' width(s) for %s but the model was "
            "not stored with this process; leaving them as-is.", " ".join(pdgs))
        return
    with open(model_file) as f:
        lines = f.read().splitlines()
    model = lines[0].strip() if lines else ""
    stored_hash = lines[1].strip() if len(lines) > 1 else ""
    if not model:
        logger.warning("SubProcesses/model.txt is empty; 'auto' widths not computed.")
        return

    # verify the model on disk still matches the one used at output time
    if stored_hash and os.path.isdir(model):
        current_hash = misc.hash_model_files(model)
        if current_hash and current_hash != stored_hash:
            logger.warning(
                "The model at %s has changed since this process was generated "
                "(hash mismatch); the 'auto' width(s) will be computed with the "
                "current model, which may be inconsistent with the matrix "
                "element.", model)

    mg5 = str(_MG_ROOT / "bin" / "madgraph")
    if not os.path.exists(mg5):
        logger.warning("Cannot find madgraph at %s; 'auto' widths not computed.", mg5)
        return

    import tempfile
    cmds = "import model %s\ncompute_widths %s --path=%s\n" % (
        model, " ".join(pdgs), os.path.abspath(param_card_path))
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as fh:
        fh.write(cmds)
        cmdfile = fh.name
    try:
        logger.info("Computing 'auto' width(s) for %s ...", " ".join(pdgs))
        proc = subprocess.run([mg5, cmdfile])
        if proc.returncode != 0:
            logger.warning(
                "compute_widths returned a non-zero exit code; the param_card "
                "may still contain 'auto' entries.")
    finally:
        try:
            os.remove(cmdfile)
        except OSError:
            pass


def run_single(switch=None) -> "MadgraphProcess":
    """Run a single generation and return the process (for its result)."""
    compute_auto_widths()
    process = MadgraphProcess()
    process.survey()
    process.train_madnis()
    process.generate_events()
    # run_card-driven LHE post-processing (displaced vertex + systematics)
    run_lhe_postprocessing(process)
    # run the post-processing tools (Pythia8/Delphes/MadSpin/reweight/analysis)
    # selected in the merged question above on the generated events.
    if switch:
        run_selected_tools(switch, process)
    return process


def detect_run_scan(run_card_path):
    """Return a banner.RunCardIterator if the run_card contains scan:[...]
    values, else None."""
    from madgraph.various import banner as banner_mod
    from madgraph.various import misc as _misc
    with _misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
        rc = banner_mod.RunCard(run_card_path, consistency=False)
    if getattr(rc, 'scan_set', None):
        return banner_mod.RunCardIterator(run_card_path)
    return None


def detect_param_scan(param_card_path):
    """Return a ParamCardIterator if the param_card contains scan:[...] values,
    else None."""
    if not os.path.exists(param_card_path):
        return None
    from models import check_param_card as param_card_mod
    it = param_card_mod.ParamCardIterator(param_card_path)
    for block in it.order:
        for param in block:
            if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                return it
    return None


def run_scan(iterator, card_path, switch=None) -> None:
    """Iterate over all scan points, running a full generation for each and
    accumulating the results, then write the scan summary. Works for both the
    run_card (RunCardIterator) and the param_card (ParamCardIterator); their
    interface (__iter__/write/store_entry/get_next_name/write_summary) is the
    same. The scan card is restored afterwards."""
    import tomllib
    with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
        run_name = tomllib.load(f).get("run", {}).get("run_name", "run")

    from models import check_param_card as param_card_mod
    is_param_scan = isinstance(iterator, param_card_mod.ParamCardIterator)

    backup = card_path + ".scan_bak"
    shutil.copy(card_path, backup)
    try:
        for i, point in enumerate(iterator):
            point.write(card_path)
            logger.info("=== scan point %d ===", i + 1)
            process = run_single(switch)
            # use the run directory the process actually created, so the
            # per-point params.dat written by write_summary has a home
            name = os.path.basename(process.run_path)
            if is_param_scan:
                # pass the (possibly auto-width-updated) card so the summary
                # records the model-computed width for each 'auto' entry
                iterator.store_entry(name, process.get_result(),
                                     param_card_path=card_path)
            else:
                iterator.store_entry(name, process.get_result())
        os.makedirs("Events", exist_ok=True)
        summary = os.path.join("Events", "scan_%s.txt" % run_name)
        iterator.write_summary(summary)
        logger.info("scan results written to %s", summary)
    finally:
        shutil.move(backup, card_path)


def run_generation(switch=None) -> None:
    """Run the generation, expanding a scan over the run_card or the param_card
    when one is present (scanning both simultaneously is not allowed)."""
    run_card_path = os.path.join("Cards", "run_card.toml")
    param_card_path = os.path.join("Cards", "param_card.dat")
    run_iter = detect_run_scan(run_card_path)
    param_iter = detect_param_scan(param_card_path)
    if run_iter and param_iter:
        raise RuntimeError(
            "Scanning simultaneously over the run_card and the param_card is "
            "not allowed. Please keep the scan:[...] entries in only one card.")
    if run_iter:
        run_scan(run_iter, run_card_path, switch)
    elif param_iter:
        run_scan(param_iter, param_card_path, switch)
    else:
        run_single(switch)


def force_lhe_output_if_needed(switch) -> None:
    """Any post-processing tool (shower/detector/madspin/reweight/analysis)
    operates on an LHE file, so make sure the events are written in that format
    when one of them is enabled."""
    if not switch:
        return
    if not any(switch.get(k, "OFF") not in ("OFF", "Not Avail.")
               for k in ("shower", "detector", "madspin", "reweight", "analysis")):
        return
    from madgraph.various.banner import RunCardMG7
    path = os.path.join("Cards", "run_card.toml")
    run_card = RunCardMG7(path, consistency=False)
    if run_card["run"]["output_format"] != "lhe":
        run_card["run"]["output_format"] = "lhe"
        run_card.write(path)
        logging.getLogger("madevent").info(
            "output_format set to 'lhe' (required by the selected post-processing).")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_false", dest="ask_edit_cards")
    args = parser.parse_args()
    switch = {}

    if args.ask_edit_cards:
        switch = ask_edit_cards()
        force_lhe_output_if_needed(switch)

    # Remove soft limit on number of open files as it can be quite low on some systems
    soft_lim, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))

    run_generation(switch)
