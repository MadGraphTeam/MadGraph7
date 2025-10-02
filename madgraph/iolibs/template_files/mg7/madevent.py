import argparse
import os
import time
from datetime import timedelta
import glob
import json
import subprocess
import logging
from dataclasses import dataclass
from typing import Literal
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
        for group_name, group in groups.items():
            if group_name not in cut_args:
                continue
            group_args = cut_args[group_name]
            for obs_name, obs in observables.items():
                if obs_name not in group_args:
                    continue
                obs_args = group_args[obs_name]
                if "min" in obs_args:
                    self.cut_data.append(
                        me.CutItem(obs, me.Cuts.min, obs_args["min"], group)
                    )
                if "max" in obs_args:
                    self.cut_data.append(
                        me.CutItem(obs, me.Cuts.max, obs_args["max"], group)
                    )
        if "sqrt_s" in cut_args:
            sqs_args = cut_args["sqrt_s"]
            if "min" in sqs_args:
                self.cut_data.append(
                    me.CutItem(me.Cuts.obs_sqrt_s, me.Cuts.min, sqs_args["min"], [])
                )
            if "max" in sqs_args:
                self.cut_data.append(
                    me.CutItem(me.Cuts.obs_sqrt_s, me.Cuts.max, sqs_args["max"], [])
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
        self.event_generator_config = cfg
        self.event_generator = None

    def init_context(self) -> None:
        device_name = self.run_card["run"]["device"]
        if device_name == "cpu":
            device = me.cpu_device()
        elif device_name == "cuda":
            device = me.cuda_device()
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
            phasespaces, "events.npy" if mode is None else f"events_{mode}.npy"
        )

        print()
        if mode is None:
            print("Running survey")
        else:
            print(f"Running survey for {mode} phasespace")
        start_time = get_start_time()
        event_generator.survey()
        print_run_time(start_time)
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
                self.event_generator = self.build_event_generator(self.phasespaces, "events.npy")
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
        self.event_generator = self.build_event_generator(madnis_phasespaces, "events.npy")
        self.event_generator.survey() #TODO: avoid

    def generate_events(self) -> None:
        start_time = get_start_time()
        self.event_generator.generate()
        print_run_time(start_time)

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
        pids_out.append(pid)
    return pids_out


class MadgraphSubprocess:
    def __init__(self, process: MadgraphProcess, meta: dict, subproc_id: int):
        self.process = process
        self.meta = meta
        self.subproc_id = subproc_id

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

        self.me_index = self.process.context.load_matrix_element(
            api_path, self.process.param_card_path
        )

    def build_multichannel_phasespace(self) -> PhaseSpace:
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        diagram_count = self.meta["diagram_count"]
        bw_cutoff = self.process.run_card["phasespace"]["bw_cutoff"]

        amp2_remap = [-1] * diagram_count
        symfact = []
        channels = []
        topologies = []
        all_topologies = []
        permutations = []
        channel_indices = []
        diagram_indices = []
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

            for topo_index, topo in enumerate(chan_topologies):
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
                indices = [
                    symfact_index_first + topo_index + i * topo_count
                    for i in range(len(chan_permutations))
                ]
                channels.append(Channel(
                    phasespace_mapping = mapping,
                    adaptive_mapping = self.build_vegas(mapping, prefix),
                    discrete_before = discrete_before,
                    discrete_after = discrete_after,
                    channel_weight_indices = indices,
                ))
            topologies.append(chan_topologies[0])
            all_topologies.append(chan_topologies)
            permutations.append(chan_permutations)
            channel_indices.append(list(range(channel_index_first, channel_index)))
            diagram_indices.append([d["diagram"] for d in diagrams])

        chan_weight_remap = list(range(len(symfact))) #TODO: only construct if necessary
        if self.process.run_card["phasespace"]["sde_strategy"] == "denominators":
            prop_chan_weights = me.PropagatorChannelWeights(
                topologies, permutations, channel_indices
            )
            indices_for_subchan = channel_indices
        else:
            prop_chan_weights = None
            indices_for_subchan = diagram_indices

        if any(len(topos) > 1 for topos in all_topologies):
            subchan_weights = me.SubchannelWeights(
                all_topologies, permutations, indices_for_subchan
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
        flavors = [flav[0] for flav in self.meta["flavors"]]
        cross_section = me.DifferentialCrossSection(
            flavors,
            self.me_index,
            self.process.running_coupling,
            None if len(flavors) > 1 else self.process.pdf_grid,
            self.process.e_cm,
            self.scale,
            False,
            self.meta["diagram_count"],
            self.meta["has_mirror_process"],
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
