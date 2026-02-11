import argparse
import os
import time
from datetime import timedelta
import glob
import json
import subprocess
import logging
from dataclasses import dataclass
from typing import Literal, NamedTuple
try:
    import tomllib
except ModuleNotFoundError:
    # for versions before 3.11
    import pip._vendor.tomli as tomllib

MG_ROOT = os.path.abspath(os.path.realpath(__file__)).replace(os.path.join("madgraph", "iolibs", "template_files", "mg7", "madevent.py"), "")
cwd = os.getcwd()
os.chdir(MG_ROOT)
if "LHAPDF_DATA_PATH" in os.environ:
    PDF_PATH = os.path.abspath(os.environ["LHAPDF_DATA_PATH"])
else:
    try:
        import lhapdf
        lhapdf.setVerbosity(0)
        PDF_PATH = os.path.abspath(lhapdf.paths()[0])
    except ImportError:
        raise RuntimeError("Can't load lhapdf module. Please set LHAPDF_DATA_PATH manually")
os.chdir(cwd)

import madspace as ms
from models.check_param_card import ParamCard

logger = logging.getLogger("madevent7")


def get_start_time():
    return time.time(), time.process_time()


def print_run_time(start):
    start_time, start_cpu_time = start
    train_time = time.time() - start_time
    train_cpu_time = time.process_time() - start_cpu_time
    print(
        f"--- Run time: {str(timedelta(seconds=round(train_time, 2) + 1e-5))[:-4]} wall time, "
        f"{str(timedelta(seconds=round(train_cpu_time, 2) + 1e-5))[:-4]} cpu time ---\n"
    )


@dataclass
class Channel:
    phasespace_mapping: ms.PhaseSpaceMapping
    adaptive_mapping: ms.Flow | ms.VegasMapping
    discrete_before: ms.DiscreteSampler | ms.DiscreteFlow | None
    discrete_after: ms.DiscreteSampler | ms.DiscreteFlow | None
    channel_weight_indices: list[int] | None
    name: str


@dataclass
class PhaseSpace:
    mode: Literal["multichannel", "flat", "both"]
    channels: list[Channel]
    symfact: list[int | None]
    chan_weight_remap: list[int]
    prop_chan_weights: ms.PropagatorChannelWeights | None = None
    subchan_weights: ms.SubchannelWeights | None = None
    cwnet: ms.ChannelWeightNetwork | None = None


class MultiChannelData(NamedTuple):
    amp2_remap: list[int]
    symfact: list[int | None]
    topologies: list[list[ms.Topology]]
    permutations: list[list[list[int]]]
    channel_indices: list[list[int]]
    channel_weight_indices: list[list[list[int]]]
    diagram_indices: list[list[int]]
    diagram_color_indices: list[list[list[int]]]


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
        with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
            self.run_card = tomllib.load(f)
        self.param_card_path = os.path.join("Cards", "param_card.dat")
        self.param_card = ParamCard(self.param_card_path)
        with open(os.path.join("SubProcesses", "subprocesses.json")) as f:
            self.subprocess_data = json.load(f)

    def init_backend(self) -> None:
        # ms.set_simd_vector_size(self.run_card["run"]["simd_vector_size"])
        ms.set_thread_count(self.run_card["run"]["thread_pool_size"])

    def init_event_dir(self) -> None:
        run_name = self.run_card["run"]["run_name"]
        os.makedirs("Events", exist_ok=True)
        existing_run_dirs = glob.glob(f"Events/{run_name}_*")
        run_index = 1
        while f"Events/{run_name}_{run_index:02d}" in existing_run_dirs:
            run_index += 1
        while True:
            try:
                self.run_path = f"Events/{run_name}_{run_index:02d}"
                os.mkdir(self.run_path)
                break
            except FileExistsError:
                run_index += 1

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
        self.pdf_grid = ms.PdfGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}_0000.dat"))
        self.pdf_grid.initialize_globals(self.context)
        self.alphas_grid = ms.AlphaSGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}.info"))
        self.alphas_grid.initialize_globals(self.context)
        self.running_coupling = ms.RunningCoupling(self.alphas_grid)

    def init_generator_config(self) -> None:
        run_args = self.run_card["run"]
        gen_args = self.run_card["generation"]
        vegas_args = self.run_card["vegas"]
        cfg = ms.EventGeneratorConfig()
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
        cfg.batch_size = gen_args["batch_size"]
        cfg.verbosity = run_args["verbosity"]
        self.event_generator_config = cfg
        self.event_generator = None

    def init_context(self) -> None:
        device_name = self.run_card["run"]["device"]
        if device_name == "cpu":
            device = ms.cpu_device()
        elif device_name == "cuda":
            device = ms.cuda_device()
        elif device_name == "hip":
            device = ms.hip_device()
        else:
            raise ValueError("Unknown device")
        self.context = ms.Context(device)

    def init_subprocesses(self) -> None:
        self.subprocesses = []
        for subproc_id, meta in enumerate(self.subprocess_data):
            self.subprocesses.append(MadgraphSubprocess(self, meta, subproc_id))

    def build_event_generator(
        self, phasespaces: list[PhaseSpace], file: str
    ) -> ms.EventGenerator:
        integrands = []
        subproc_ids = []
        channel_names = []
        channel_hists = []
        for i, (subproc, phasespace) in enumerate(zip(self.subprocesses, phasespaces)):
            subproc_integrands = subproc.build_integrands(phasespace)
            integrands.extend(subproc.build_integrands(phasespace))
            subproc_ids.extend([i] * len(phasespace.channels))
            channel_names.extend([f"{i}.{chan.name}" for chan in phasespace.channels])
            if subproc.histograms is not None:
                channel_hists.extend([subproc.histograms] * len(phasespace.channels))
        #print(integrands[0].function())
        #integrands[0].function().save("test.json")
        #integrands[0] = ms.Function.load("test.json")
        return ms.EventGenerator(
            context=self.context,
            channels=integrands,
            temp_file_prefix=os.path.join(self.run_path, file),
            status_file=os.path.join(self.run_path, "info.json"),
            config=self.event_generator_config,
            channel_subprocesses=subproc_ids,
            channel_names=channel_names,
            channel_histograms=channel_hists,
        )

    def survey_phasespaces(
        self, phasespaces: list[PhaseSpace], mode: str | None = None
    ) -> ms.EventGenerator:
        event_generator = self.build_event_generator(
            phasespaces, "events" if mode is None else f"events_{mode}"
        )

        print()
        event_generator.survey()
        return event_generator

    def survey(self) -> None:
        phasespace_mode = self.run_card["phasespace"]["mode"]
        if phasespace_mode == "multichannel":
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
        elif phasespace_mode == "both":
            phasespaces_multi = [
                subproc.build_multichannel_phasespace()
                for subproc in self.subprocesses
            ]
            evgen_multi = self.survey_phasespaces(phasespaces_multi, "multichannel")

            phasespaces_flat = [
                subproc.build_flat_phasespace()
                for subproc in self.subprocesses
            ]
            evgen_flat = self.survey_phasespaces(phasespaces_flat, "flat")

            channel_status = evgen_multi.channel_status()
            cross_sections = []
            index = 0
            for phasespace in phasespaces_multi:
                channel_count = len(phasespace.channels)
                cross_sections.append([
                    status.mean
                    for status in channel_status[index:index + channel_count]
                ])
                index += channel_count

            self.phasespaces = [
                subproc.simplify_phasespace(ps_multi, ps_flat, cross_secs)
                for subproc, ps_multi, ps_flat, cross_secs in zip(
                    self.subprocesses, phasespaces_multi, phasespaces_flat, cross_sections
                )
            ]

            if not self.run_card["madnis"]["enable"]:
                self.event_generator = self.build_event_generator(self.phasespaces, "events")
                #TODO: avoid to run survey again
                self.event_generator.survey()
        else:
            raise ValueError("Unknown phasespace mode")

    def train_madnis(self) -> None:
        madnis_args = self.run_card["madnis"]
        if not madnis_args["enable"]:
            return

        madnis_phasespaces = []
        for subproc, phasespace in zip(self.subprocesses, self.phasespaces):
            phasespace = subproc.build_madnis(phasespace)
            subproc.train_madnis(phasespace)
            madnis_phasespaces.append(phasespace)
        self.phasespaces = madnis_phasespaces
        self.event_generator = self.build_event_generator(madnis_phasespaces, "events")
        self.event_generator.survey() #TODO: avoid

    def generate_events(self) -> None:
        start_time = get_start_time()
        self.event_generator.generate()
        output_format = self.run_card["run"]["output_format"]
        if output_format == "compact_npy":
            self.event_generator.combine_to_compact_npy(
                os.path.join(self.run_path, "events.npy")
            )
        elif output_format == "lhe_npy":
            lhe_completer = self.build_lhe_completer()
            self.event_generator.combine_to_lhe_npy(
                os.path.join(self.run_path, "events.npy"), lhe_completer
            )
        elif output_format == "lhe":
            lhe_completer = self.build_lhe_completer()
            self.event_generator.combine_to_lhe(
                os.path.join(self.run_path, "events.lhe"), lhe_completer
            )
        else:
            raise ValueError("Unknown output format")
        self.save_gridpack()

    def build_lhe_completer(self):
        subproc_args = []
        for subproc, meta in zip(self.subprocesses, self.subprocess_data):
            (
                _,
                _,
                topologies,
                permutations,
                _,
                _,
                diagram_indices,
                diagram_color_indices,
            ) = subproc.build_multi_channel_data()
            subproc_args.append(
                ms.SubprocArgs(
                    topologies = [topo[0] for topo in topologies],
                    permutations = permutations,
                    diagram_indices = diagram_indices,
                    diagram_color_indices = diagram_color_indices,
                    color_flows = meta["color_flows"],
                    pdg_color_types = {
                        int(key): value
                        for key, value in meta["pdg_color_types"].items()
                    },
                    helicities = meta["helicities"],
                    pdg_ids = [flavor["options"] for flavor in meta["flavors"]],
                    matrix_flavor_indices = [
                        flavor["index"] for flavor in meta["flavors"]
                    ],
                )
            )
        return ms.LHECompleter(
            subproc_args=subproc_args,
            bw_cutoff=self.run_card["phasespace"]["bw_cutoff"]
        )

    def save_gridpack(self) -> None:
        gridpack_path = os.path.join(self.run_path, "gridpack")
        os.mkdir(gridpack_path)
        self.context.save(os.path.join(gridpack_path, "context.json"))

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
        if pid == 82:
            pid = 11
        pids_out.append(pid)
    return pids_out


class MadgraphSubprocess:
    def __init__(self, process: MadgraphProcess, meta: dict, subproc_id: int):
        self.process = process
        self.meta = meta
        self.subproc_id = subproc_id
        self.multi_channel_data = None

        api_path = self.meta["path"]
        if not os.path.isfile(api_path):
            cwd = os.getcwd()
            api_dir = os.path.dirname(api_path)
            logger.info(f"Compiling subprocess {api_dir}")
            os.chdir(api_dir)
            subprocess.run(["make"])
            os.chdir(cwd)

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
            self.matrix_element = None
        else: 
            self.matrix_element = self.process.context.load_matrix_element(
                api_path, self.process.param_card_path
            )

    def build_multi_channel_data(self) -> MultiChannelData:
        if self.multi_channel_data is not None:
            return self.multi_channel_data

        diagram_count = self.meta["diagram_count"]
        bw_cutoff = self.process.run_card["phasespace"]["bw_cutoff"]

        amp2_remap = [-1] * diagram_count
        symfact = []
        topologies = []
        permutations = []
        channel_indices = []
        channel_weight_indices = []
        diagram_indices = []
        diagram_color_indices = []
        channel_index = 0

        for channel_id, channel in enumerate(self.meta["channels"]):
            propagators = []
            for i, pid in enumerate(clean_pids(channel["propagators"])):
                mass = self.process.get_mass(pid)
                width = self.process.get_width(pid)
                if i in channel["on_shell_propagators"]:
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
                ))
            vertices = channel["vertices"]
            diagrams = channel["diagrams"]
            chan_permutations = [d["permutation"] for d in diagrams]
            diag = ms.Diagram(
                self.incoming_masses, self.outgoing_masses, propagators, vertices
            )
            chan_topologies = ms.Topology.topologies(diag)
            topo_count = len(chan_topologies)

            amp2_remap[diagrams[0]["diagram"]] = channel_index
            channel_index_first = channel_index
            symfact_index_first = len(symfact)
            channel_index += 1
            symfact.extend([None] * topo_count)
            for d in diagrams[1:]:
                amp2_remap[d["diagram"]] = channel_index
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
            diagram_color_indices.append([d["active_colors"] for d in diagrams])
        self.multi_channel_data = MultiChannelData(
            amp2_remap,
            symfact,
            topologies,
            permutations,
            channel_indices,
            channel_weight_indices,
            diagram_indices,
            diagram_color_indices,
        )
        return self.multi_channel_data

    def build_multichannel_phasespace(self) -> PhaseSpace:
        (
            amp2_remap,
            symfact,
            topologies,
            permutations,
            channel_indices,
            channel_weight_indices,
            diagram_indices,
            _,
        ) = self.build_multi_channel_data()

        channels = []
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        for channel_id, (chan_topologies, chan_permutations, chan_indices) in enumerate(zip(
            topologies, permutations, channel_weight_indices
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
                discrete_before, discrete_after = self.build_discrete(
                    len(chan_permutations), len(self.meta["flavors"]), prefix
                )
                channels.append(Channel(
                    phasespace_mapping = mapping,
                    adaptive_mapping = self.build_vegas(mapping, prefix),
                    discrete_before = discrete_before,
                    discrete_after = discrete_after,
                    channel_weight_indices = indices,
                    name = f"{channel_id}",
                ))

        chan_weight_remap = list(range(len(symfact))) #TODO: only construct if necessary
        if self.process.run_card["phasespace"]["sde_strategy"] == "denominators":
            prop_chan_weights = ms.PropagatorChannelWeights(
                [topo[0] for topo in topologies], permutations, channel_indices
            )
            indices_for_subchan = channel_indices
        else:
            prop_chan_weights = None
            indices_for_subchan = diagram_indices

        if any(len(topos) > 1 for topos in topologies):
            subchan_weights = ms.SubchannelWeights(
                topologies, permutations, indices_for_subchan
            )
        else:
            subchan_weights = None
            if prop_chan_weights is None:
                chan_weight_remap = [
                    len(symfact) if remap == -1 else remap for remap in amp2_remap
                ]

        return PhaseSpace(
            mode="multichannel",
            channels=channels,
            chan_weight_remap=chan_weight_remap,
            symfact=symfact,
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
        discrete_before, discrete_after = self.build_discrete(
            1, len(self.meta["flavors"]), prefix
        )
        channel = Channel(
            phasespace_mapping = mapping,
            adaptive_mapping = self.build_vegas(mapping, prefix),
            discrete_before = discrete_before,
            discrete_after = discrete_after,
            channel_weight_indices = [0],
            name = "F",
        )
        return PhaseSpace(
            mode="flat",
            channels=[channel],
            chan_weight_remap=[0] * self.meta["diagram_count"],
            symfact=[None],
        )

    def simplify_phasespace(
        self,
        multi_phasespace: PhaseSpace,
        flat_phasespace: PhaseSpace | None,
        cross_sections: list[float]
    ) -> PhaseSpace:
        assert multi_phasespace.mode == "multichannel"

        kept_count = self.process.run_card["phasespace"]["simplified_channel_count"]
        if len(multi_phasespace.channels) <= kept_count:
            return multi_phasespace

        assert flat_phasespace is not None and flat_phasespace.mode == "flat"
        #TODO: need to be careful here in the case of flavor sampling
        #TODO: come up with some smarter heuristic than just channel cross section
        #TODO: deal with resonances in a smart way
        kept_channels = [
            index
            for index, cs in sorted(
                enumerate(cross_sections), key=lambda pair: pair[1], reverse=True
            )
        ][:kept_count]

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
                discrete_before = channel.discrete_before,
                discrete_after = channel.discrete_after,
                channel_weight_indices = list(range(
                    channel_index, channel_index + perm_count
                )),
                name = channel.name,
            ))

        flat_channel = flat_phasespace.channels[0]
        channels.append(Channel(
            phasespace_mapping = flat_channel.phasespace_mapping,
            adaptive_mapping = flat_channel.adaptive_mapping,
            discrete_before = flat_channel.discrete_before,
            discrete_after = flat_channel.discrete_after,
            channel_weight_indices = [len(symfact)],
            name = flat_channel.name,
        ))
        flat_index = len(symfact)
        symfact.append(None)
        channel_map[len(multi_phasespace.symfact)] = len(symfact)
        chan_weight_remap = [
            channel_map.get(remap, flat_index)
            for remap in multi_phasespace.chan_weight_remap
        ]

        return PhaseSpace(
            mode="both",
            channels=channels,
            chan_weight_remap=chan_weight_remap,
            symfact=symfact,
            prop_chan_weights=multi_phasespace.prop_chan_weights,
            subchan_weights=multi_phasespace.subchan_weights,
        )

    def build_madnis(self, phasespace: PhaseSpace) -> PhaseSpace:
        madnis_args = self.process.run_card["madnis"]
        channels = []
        for channel_id, channel in enumerate(phasespace.channels):
            discrete_before = channel.discrete_before
            if discrete_before is not None:
                #TODO: build discrete flows
                pass

            perm_count = channel.phasespace_mapping.channel_count()
            #cond_dim = perm_count if perm_count > 1 else 0
            flow_dim = channel.phasespace_mapping.random_dim()
            prefix = f"subproc{self.subproc_id}.channel{channel_id}"
            flow = ms.Flow(
                input_dim=flow_dim,
                condition_dim=0,
                prefix=prefix,
                bin_count=madnis_args["flow_spline_bins"],
                subnet_hidden_dim=madnis_args["flow_hidden_dim"],
                subnet_layers=madnis_args["flow_layers"],
                subnet_activation=self.activation(madnis_args["flow_activation"]),
                invert_spline=madnis_args["flow_invert_spline"],
            )
            if channel.adaptive_mapping is None:
                flow.initialize_globals(self.process.context)
            else:
                flow.initialize_from_vegas(
                    self.process.context, channel.adaptive_mapping.grid_name()
                )
            #cond_dim += flow_dim

            discrete_after = channel.discrete_after
            if discrete_after is not None:
                discrete_after = ms.DiscreteFlow(
                    option_counts=[len(self.meta["flavors"])],
                    prefix=f"{prefix}.discrete_after",
                    dims_with_prior=[0],
                    condition_dim=flow_dim,
                    subnet_hidden_dim=madnis_args["discrete_hidden_dim"],
                    subnet_layers=madnis_args["discrete_layers"],
                    subnet_activation=self.activation(madnis_args["discrete_activation"]),
                )
                discrete_after.initialize_globals(self.process.context)

            channels.append(Channel(
                phasespace_mapping = channel.phasespace_mapping,
                adaptive_mapping = flow,
                discrete_before = discrete_before,
                discrete_after = discrete_after,
                channel_weight_indices = channel.channel_weight_indices,
                name = channel.name,
            ))

        return PhaseSpace(
            mode="both",
            channels=channels,
            chan_weight_remap=phasespace.chan_weight_remap,
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
        vegas.initialize_globals(self.process.context)
        return vegas

    def build_discrete(
        self, permutation_count: int, flavor_count: int, prefix: str
    ) -> tuple[ms.DiscreteSampler | None, ms.DiscreteSampler | None]:
        #return None, None
        discrete_before = None
        #if permutation_count > 1:
        #    discrete_before = ms.DiscreteSampler(
        #        [permutation_count], f"{prefix}.discrete_before"
        #    )
        #    discrete_before.initialize_globals(self.process.context)
        #else:
        #    discrete_before = None

        if flavor_count > 1:
            discrete_after = ms.DiscreteSampler(
                [flavor_count], f"{prefix}.discrete_after", [0]
            )
            discrete_after.initialize_globals(self.process.context)
        else:
            discrete_after = None

        return discrete_before, discrete_after

    def build_cwnet(self, channel_count: int) -> ms.ChannelWeightNetwork:
        madnis_args = self.process.run_card["madnis"]
        cwnet = ms.ChannelWeightNetwork(
            channel_count=channel_count,
            particle_count=self.particle_count,
            hidden_dim=madnis_args["cwnet_hidden_dim"],
            layers=madnis_args["cwnet_layers"],
            activation=self.activation(madnis_args["cwnet_activation"]),
            prefix=f"subproc{self.subproc_id}.cwnet",
        )
        cwnet.initialize_globals(self.process.context)
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
        flags: int = ms.EventGenerator.integrand_flags
    ) -> list[ms.Integrand]:
        flavors = [flav["options"][0] for flav in self.meta["flavors"]]
        if self.matrix_element:
            matrix_element = ms.MatrixElement(
                self.matrix_element,
                ms.Integrand.matrix_element_inputs,
                ms.Integrand.matrix_element_outputs,
                True,
            )
        else:
            matrix_element = ms.MatrixElement(
                0xBADCAFE,
                self.particle_count,
                ms.Integrand.matrix_element_inputs,
                ms.Integrand.matrix_element_outputs,
                self.meta["diagram_count"],
                True,
            )
        pdf_grid = (
            None
            if len(flavors) > 1 or self.process.leptonic
            else self.process.pdf_grid
        )
        cross_section = ms.DifferentialCrossSection(
            matrix_element=matrix_element,
            cm_energy=self.process.e_cm,
            running_coupling=self.process.running_coupling,
            energy_scale=self.scale,
            pid_options=flavors,
            has_pdf1=not self.process.leptonic,
            has_pdf2=not self.process.leptonic,
            pdf_grid1=pdf_grid,
            pdf_grid2=pdf_grid,
            has_mirror=self.meta["has_mirror_process"],
            input_momentum_fraction=True,
        )
        integrands = []
        for channel in phasespace.channels:
            integrands.append(ms.Integrand(
                channel.phasespace_mapping,
                cross_section,
                channel.adaptive_mapping,
                channel.discrete_before,
                channel.discrete_after,
                self.process.pdf_grid,
                self.scale,
                phasespace.prop_chan_weights,
                phasespace.subchan_weights,
                phasespace.cwnet,
                phasespace.chan_weight_remap,
                len(phasespace.symfact),
                flags,
                channel.channel_weight_indices,
            ))
        #print(integrands[0].function())
        #print(integrands[1].function())
        return integrands

    def train_madnis(self, phasespace: PhaseSpace) -> None:
        print("Training MadNIS")
        # do import here to make pytorch and MadNIS optional dependencies
        from .train_madnis import train_madnis, MADNIS_INTEGRAND_FLAGS
        start_time = get_start_time()
        train_madnis(
            self.build_integrands(phasespace, MADNIS_INTEGRAND_FLAGS),
            phasespace,
            self.process.run_card["madnis"],
            self.process.context
        )
        print_run_time(start_time)


def ask_edit_cards() -> None:
    #TODO: these imports break when trying to generate flame graphs, so do them locally for now
    from madgraph.interface.common_run_interface import CommonRunCmd, AskforEditCard
    from madgraph.interface.extended_cmd import Cmd

    #TODO: some rather disgusting monkey-patching to make editing cards work
    class MG7Cmd(Cmd):
        def __init__(self):
            super().__init__(".", {})
            self.proc_characteristics = None
        def do_open(self, line):
            CommonRunCmd.do_open(self, line)
        def check_open(self, args):
            CommonRunCmd.check_open(self, args)
    old_define_paths = AskforEditCard.define_paths
    def define_paths(self, **opt):
        old_define_paths(self, **opt)
        self.paths["run"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
        self.paths["run_card.toml"] = os.path.join(self.me_dir, "Cards", "run_card.toml")
    AskforEditCard.define_paths = define_paths
    AskforEditCard.reload_card = lambda self, path: None

    cmd = MG7Cmd()
    CommonRunCmd.ask_edit_card_static(
        ["param_card.dat", "run_card.toml"],
        pwd=".",
        ask=cmd.ask,
        plot=False
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", action="store_false", dest="ask_edit_cards")
    args = parser.parse_args()
    if args.ask_edit_cards:
        ask_edit_cards()

    process = MadgraphProcess()
    process.survey()
    process.train_madnis()
    process.generate_events()
