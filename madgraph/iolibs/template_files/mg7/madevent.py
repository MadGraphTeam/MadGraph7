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

if "LHAPDF_DATA_PATH" in os.environ:
    PDF_PATH = os.environ["LHAPDF_DATA_PATH"]
else:
    try:
        import lhapdf
        lhapdf.setVerbosity(0)
        PDF_PATH = lhapdf.paths()[0]
    except ImportError:
        raise RuntimeError("Can't load lhapdf module. Please set LHAPDF_DATA_PATH manually")

import madevent7 as me
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
    phasespace_mapping: me.PhaseSpaceMapping
    adaptive_mapping: me.Flow | me.VegasMapping
    discrete_before: me.DiscreteSampler | me.DiscreteFlow | None
    discrete_after: me.DiscreteSampler | me.DiscreteFlow | None
    channel_weight_indices: list[int] | None


@dataclass
class PhaseSpace:
    mode: Literal["multichannel", "flat", "both"]
    channels: list[Channel]
    symfact: list[int | None]
    chan_weight_remap: list[int]
    prop_chan_weights: me.PropagatorChannelWeights | None = None
    subchan_weights: me.SubchannelWeights | None = None
    cwnet: me.ChannelWeightNetwork | None = None


class MultiChannelData(NamedTuple):
    amp2_remap: list[int]
    symfact: list[int | None]
    topologies: list[list[me.Topology]]
    permutations: list[list[list[int]]]
    channel_indices: list[list[int]]
    channel_weight_indices: list[list[list[int]]]
    diagram_indices: list[list[int]]
    diagram_color_indices: list[list[list[int]]]


class MadgraphProcess:
    def __init__(self):
        self.load_cards()
        self.init_backend()
        self.init_event_dir()
        self.init_context()
        self.init_cuts()
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
        me.set_simd_vector_size(self.run_card["run"]["simd_vector_size"])
        me.set_thread_count(self.run_card["run"]["thread_pool_size"])

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

    def init_cuts(self) -> None:
        observables = {
            "pt": me.Cuts.obs_pt,
            "eta": me.Cuts.obs_eta,
            "dR": me.Cuts.obs_dr,
            "mass": me.Cuts.obs_mass,
        }
        groups = {
            "jet": me.Cuts.jet_pids,
            "bottom": me.Cuts.bottom_pids,
            "lepton": me.Cuts.lepton_pids,
            "missing": me.Cuts.missing_pids,
            "photon": me.Cuts.photon_pids,
        }
        cut_args = self.run_card["cuts"]
        self.cut_data = []
        for group_name, group_args in cut_args.items():
            if group_name == "sqrt_s":
                if "min" in group_args:
                    self.cut_data.append(
                        me.CutItem(me.Cuts.obs_sqrt_s, me.Cuts.min, group_args["min"], [])
                    )
                if "max" in group_args:
                    self.cut_data.append(
                        me.CutItem(me.Cuts.obs_sqrt_s, me.Cuts.max, group_args["max"], [])
                    )
                continue

            if "-" in group_name:
                group_name1, group_name2 = group_name.split("-")
                pids1 = (
                    [int(group_name1)]
                    if group_name1.isnumeric()
                    else groups[group_name1]
                )
                pids2 = (
                    [int(group_name2)]
                    if group_name2.isnumeric()
                    else groups[group_name2]
                )
            else:
                pids1 = (
                    [int(group_name)] if group_name.isnumeric() else groups[group_name]
                )
                pids2 = []
            for obs_name, obs_args in group_args.items():
                obs = observables[obs_name]
                if "min" in obs_args:
                    self.cut_data.append(
                        me.CutItem(obs, me.Cuts.min, obs_args["min"], pids1, pids2)
                    )
                if "max" in obs_args:
                    self.cut_data.append(
                        me.CutItem(obs, me.Cuts.max, obs_args["max"], pids1, pids2)
                    )

    def init_beam(self) -> None:
        beam_args = self.run_card["beam"]

        self.e_cm = beam_args["e_cm"]

        dynamical_scales = {
            "transverse_energy": me.EnergyScale.transverse_energy,
            "transverse_mass": me.EnergyScale.transverse_mass,
            "half_transverse_mass": me.EnergyScale.half_transverse_mass,
            "partonic_energy": me.EnergyScale.partonic_energy,
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
        self.pdf_grid = me.PdfGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}_0000.dat"))
        self.pdf_grid.initialize_globals(self.context)
        self.alphas_grid = me.AlphaSGrid(os.path.join(PDF_PATH, pdf_set, f"{pdf_set}.info"))
        self.alphas_grid.initialize_globals(self.context)
        self.running_coupling = me.RunningCoupling(self.alphas_grid)

    def init_generator_config(self) -> None:
        run_args = self.run_card["run"]
        gen_args = self.run_card["generation"]
        vegas_args = self.run_card["vegas"]
        cfg = me.EventGeneratorConfig()
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
            device = me.cpu_device()
        elif device_name == "cuda":
            device = me.cuda_device()
        elif device_name == "hip":
            device = me.hip_device()
        else:
            raise ValueError("Unknown device")
        self.context = me.Context(device)

    def init_subprocesses(self) -> None:
        self.subprocesses = []
        for subproc_id, meta in enumerate(self.subprocess_data):
            self.subprocesses.append(MadgraphSubprocess(self, meta, subproc_id))

    def build_event_generator(
        self, phasespaces: list[PhaseSpace], file: str
    ) -> me.EventGenerator:
        integrands = []
        for subproc, phasespace in zip(self.subprocesses, phasespaces):
            integrands.extend(subproc.build_integrands(phasespace))
        #print(integrands[0].function())
        #integrands[0].function().save("test.json")
        #integrands[0] = me.Function.load("test.json")
        return me.EventGenerator(
            self.context,
            integrands,
            os.path.join(self.run_path, file),
            self.event_generator_config,
        )

    def survey_phasespaces(
        self, phasespaces: list[PhaseSpace], mode: str | None = None
    ) -> me.EventGenerator:
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
                me.SubprocArgs(
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
        return me.LHECompleter(
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
        self.cuts = me.Cuts(clean_pids(self.meta["outgoing"]), self.process.cut_data)

        self.scale = me.EnergyScale(
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
                propagators.append(me.Propagator(
                    mass=mass,
                    width=width,
                    integration_order=0,
                    e_min=e_min,
                    e_max=e_max,
                ))
            vertices = channel["vertices"]
            diagrams = channel["diagrams"]
            chan_permutations = [d["permutation"] for d in diagrams]
            diag = me.Diagram(
                self.incoming_masses, self.outgoing_masses, propagators, vertices
            )
            chan_topologies = me.Topology.topologies(diag)
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
                mapping = me.PhaseSpaceMapping(
                    chan_topologies[0],
                    self.process.e_cm,
                    t_channel_mode=t_channel_mode,
                    cuts=self.cuts,
                    invariant_power=self.process.run_card["phasespace"]["invariant_power"],
                    permutations=chan_permutations,
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
                ))

        chan_weight_remap = list(range(len(symfact))) #TODO: only construct if necessary
        if self.process.run_card["phasespace"]["sde_strategy"] == "denominators":
            prop_chan_weights = me.PropagatorChannelWeights(
                [topo[0] for topo in topologies], permutations, channel_indices
            )
            indices_for_subchan = channel_indices
        else:
            prop_chan_weights = None
            indices_for_subchan = diagram_indices

        if any(len(topos) > 1 for topos in topologies):
            subchan_weights = me.SubchannelWeights(
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
        mapping = me.PhaseSpaceMapping(
            self.incoming_masses + self.outgoing_masses,
            self.process.e_cm,
            mode=self.t_channel_mode(self.process.run_card["phasespace"]["flat_mode"]),
            cuts=self.cuts,
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
            ))

        flat_channel = flat_phasespace.channels[0]
        channels.append(Channel(
            phasespace_mapping = flat_channel.phasespace_mapping,
            adaptive_mapping = flat_channel.adaptive_mapping,
            discrete_before = flat_channel.discrete_before,
            discrete_after = flat_channel.discrete_after,
            channel_weight_indices = [len(symfact)],
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
            flow = me.Flow(
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
                discrete_after = me.DiscreteFlow(
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

    def build_vegas(self, mapping: me.PhaseSpaceMapping, prefix: str) -> me.VegasMapping:
        if not self.process.run_card["vegas"]["enable"]:
            return None

        vegas = me.VegasMapping(
            mapping.random_dim(),
            self.process.run_card["vegas"]["bins"],
            prefix,
        )
        vegas.initialize_globals(self.process.context)
        return vegas

    def build_discrete(
        self, permutation_count: int, flavor_count: int, prefix: str
    ) -> tuple[me.DiscreteSampler | None, me.DiscreteSampler | None]:
        #return None, None
        discrete_before = None
        #if permutation_count > 1:
        #    discrete_before = me.DiscreteSampler(
        #        [permutation_count], f"{prefix}.discrete_before"
        #    )
        #    discrete_before.initialize_globals(self.process.context)
        #else:
        #    discrete_before = None

        if flavor_count > 1:
            discrete_after = me.DiscreteSampler(
                [flavor_count], f"{prefix}.discrete_after", [0]
            )
            discrete_after.initialize_globals(self.process.context)
        else:
            discrete_after = None

        return discrete_before, discrete_after

    def build_cwnet(self, channel_count: int) -> me.ChannelWeightNetwork:
        madnis_args = self.process.run_card["madnis"]
        cwnet = me.ChannelWeightNetwork(
            channel_count=channel_count,
            particle_count=self.particle_count,
            hidden_dim=madnis_args["cwnet_hidden_dim"],
            layers=madnis_args["cwnet_layers"],
            activation=self.activation(madnis_args["cwnet_activation"]),
            prefix=f"subproc{self.subproc_id}.cwnet",
        )
        cwnet.initialize_globals(self.process.context)
        return cwnet

    def t_channel_mode(self, name: str) -> me.PhaseSpaceMapping.TChannelMode:
        modes = {
            "propagator": me.PhaseSpaceMapping.propagator,
            "rambo": me.PhaseSpaceMapping.rambo,
            "chili": me.PhaseSpaceMapping.chili,
        }
        if name in modes:
            return modes[name]
        else:
            raise ValueError(f"Invalid t-channel mode '{name}'")

    def activation(self, name: str) -> me.MLP.Activation:
        activations = {
            "relu": me.MLP.relu,
            "leaky_relu": me.MLP.leaky_relu,
            "elu": me.MLP.elu,
            "gelu": me.MLP.gelu,
            "sigmoid": me.MLP.sigmoid,
            "softplus": me.MLP.softplus,
        }
        if name in activations:
            return activations[name]
        else:
            raise ValueError(f"Invalid activation function '{name}'")

    def build_integrands(
        self,
        phasespace: PhaseSpace,
        flags: int = me.EventGenerator.integrand_flags
    ) -> list[me.Integrand]:
        flavors = [flav["options"][0] for flav in self.meta["flavors"]]
        if self.matrix_element:
            matrix_element = me.MatrixElement(
                self.matrix_element,
                me.Integrand.matrix_element_inputs,
                me.Integrand.matrix_element_outputs,
                True,
            )
        else:
            matrix_element = me.MatrixElement(
                0xBADCAFE,
                self.particle_count,
                me.Integrand.matrix_element_inputs,
                me.Integrand.matrix_element_outputs,
                self.meta["diagram_count"],
                True,
            )
        cross_section = me.DifferentialCrossSection(
            matrix_element=matrix_element,
            cm_energy=self.process.e_cm,
            running_coupling=self.process.running_coupling,
            energy_scale=self.scale,
            pid_options=flavors,
            has_pdf1=True,
            has_pdf2=True,
            pdf_grid1=None if len(flavors) > 1 else self.process.pdf_grid,
            pdf_grid2=None if len(flavors) > 1 else self.process.pdf_grid,
            has_mirror=self.meta["has_mirror_process"],
            input_momentum_fraction=True,
        )
        integrands = []
        for channel in phasespace.channels:
            integrands.append(me.Integrand(
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
