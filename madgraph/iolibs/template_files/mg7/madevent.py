import os
import time
import argparse
from datetime import timedelta
from types import SimpleNamespace
import glob
import json
import subprocess
import logging
from collections import defaultdict
from dataclasses import dataclass
try:
    import tomllib
except ModuleNotFoundError:
    # for versions before 3.11
    import pip._vendor.tomli as tomllib

import madevent7 as me
from models.check_param_card import ParamCard

logger = logging.getLogger('madgraph.madnis')


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
    integrand: me.Integrand
    amp2_remap: list[int] | None
    active_flavors: list[int]
    channel_weight_indices: list[int] | None


class MadgraphProcess:
    def __init__(self):
        with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
            self.run_card = tomllib.load(f)
        self.param_card = ParamCard(os.path.join("Cards", "param_card.dat"))
        with open(os.path.join("SubProcesses", "subprocesses.json")) as f:
            subprocess_data = json.load(f)

        self.init_cuts()
        self.init_generator_config()
        self.subprocesses = [
            MadgraphSubprocess(self, meta, subproc_id)
            for subproc_id, meta in enumerate(subprocess_data)
        ]

    def init_cuts(self):
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

    def init_generator_config(self):
        gen_args = self.run_card["generation"]
        vegas_args = self.run_card["vegas"]
        cfg = me.EventGeneratorConfig()
        cfg.target_count = gen_args["events"]
        cfg.vegas_damping = vegas_args["damping"]
        cfg.max_overweight_fraction = gen_args["max_overweight_fraction"]
        cfg.max_overweight_truncation = gen_args["max_overweight_truncation"]
        cfg.start_batch_size = gen_args["start_batch_size"]
        cfg.max_batch_size = gen_args["max_batch_size"]
        cfg.survey_min_iters = gen_args["survey_min_iters"]
        cfg.survey_max_iters = gen_args["survey_max_iters"]
        cfg.survey_target_precision = gen_args["survey_target_precision"]
        cfg.optimization_patience = vegas_args["optimization_patience"]
        cfg.optimization_threshold = vegas_args["optimization_threshold"]
        self.event_generator_config = cfg

    def run_survey(self):
        phasespace_mode = self.run_card["phasespace"]["mode"]
        if phasespace_mode == "multichannel":
            channels = [
                channel
                for subproc in self.subprocesses
                for channel in subproc.build_multichannel_phasespace(build_flow=False)
            ]
        elif phasespace_mode == "flat":
            channels = [
                channel
                for subproc in self.subprocesses
                for channel in subproc.build_flat_phasespace(build_flow=False)
            ]
        else:
            raise ValueError("Unknown phasespace mode")

        event_generator = MadgraphEventGenerator(args, process, phasespace)

        print()
        print("Running survey")
        start_time = get_start_time()
        event_generator.survey()
        print_run_time(start_time)

        return event_generator

    def get_mass(self, pid: int) -> float:
        return self.param_card.get_value("mass", pid)

    def get_width(self, pid: int) -> float:
        return self.param_card.get_value("width", pid)


def clean_pids(pids: list[int])  -> list[int]:
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

        if not os.path.isfile(self.meta["path"]):
            cwd = os.getcwd()
            api_dir = os.path.dirname(self.meta["path"])
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
        self.cuts = me.Cuts(clean_pids(self.meta["outgoing"]), self.process.cut_data)
        self.differential_xs = me.Integrand(

        )

    def build_multichannel_phasespace(
        self, build_flow: bool
    ) -> tuple[list[Channel], list[int]]:
        t_channel_mode = self.t_channel_mode(
            self.process.run_card["phasespace"]["t_channel"]
        )
        amp2_remap = [-1] * self.meta["diagram_count"]
        symfact = []
        channels = []

        for channel_id, channel in enumerate(self.meta["channels"]):
            propagators = [
                me.Propagator(
                    self.process.get_mass(pid), self.process.get_width(pid)
                )
                for pid in clean_pids(channel["propagators"])
            ]
            vertices = channel["vertices"]
            diagrams = channel["diagrams"]
            permutations = [
                [p - 2 for p in d["permutation"][2:]] for d in diagrams 
            ] #TODO: full perm here in the future
            channel_index = len(symfact)
            amp2_remap[diagrams[0]["diagram"]] = channel_index
            symfact.append(None)
            for d in diagrams[1:]:
                amp2_remap[d["diagram"]] = len(symfact)
                symfact.append(channel_index)

            diag = me.Diagram(
                self.incoming_masses, self.outgoing_masses, propagators, vertices
            )
            topology = me.Topology(diag, me.Topology.all_decays)
            mapping = me.PhaseSpaceMapping(
                topology,
                self.process.e_cm2,
                t_channel_mode=t_channel_mode,
                cuts=self.cuts,
                nu=self.process.run_card["phasespace"]["nu"],
                permutations=permutations[1:], #TODO: probably wrong
            )
            prefix = f"subproc{self.subproc_id}_channel{channel_id}_"
            channels.append(
                Channel(
                    phasespace_mapping = mapping,
                    adaptive_mapping = self.build_adaptive_mapping(mapping, prefix, build_flow),
                    integrand = None,
                    amp2_remap = amp2_remap,
                    active_flavors = channel["active_flavors"],
                    channel_weight_indices = list(
                        range(channel_index, channel_index + len(permutations))
                    ),
                )
            )
        return channels, symfact

    def build_flat_phasespace(self, build_flow: bool = False) -> Channel:
        mapping = me.PhaseSpaceMapping(
            self.incoming_masses + self.outgoing_masses,
            self.process.e_cm2,
            mode=self.t_channel_mode(self.process.run_card["phasespace"]["flat_mode"]),
            cuts=self.cuts
        )
        prefix = f"subproc{self.subproc_id}_flat_"
        return Channel(
            phasespace_mapping = mapping,
            adaptive_mapping = self.build_adaptive_mapping(mapping, prefix, build_flow),
            integrand = None,
            amp2_remap = [0] * self.meta["diagram_count"],
            active_flavors = list(range(len(self.meta["flavors"]))),
            channel_weight_indices = [0],
        )

    def build_adaptive_mapping(
        self, mapping: me.PhaseSpaceMapping, prefix: str, build_flow: bool
    ) -> me.Flow | me.VegasMapping:
        if build_flow:
            madnis_args = self.meta["madnis"]
            return me.Flow(
                input_dim=mapping.random_dim(),
                condition_dim=0,

            )
        else:
            return me.VegasMapping(
                mapping.random_dim(),
                self.process.run_card["vegas"]["bins"],
                prefix,
            )

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


class MadgraphEventGenerator:
    def __init__(
        self,
        args: dict,
        process: MadgraphProcess,
        phasespace: Channel
    ):
        self.process = process
        self.phasespace = phasespace
        #self.n_channels = len(phasespace.mappings)
        self.n_channels = sum(len(gids) for gids in phasespace.group_indices)
        self.context = me.Context()

        self.context.load_pdf(self.process.pdf_name, 0)
        me_id = self.context.load_matrix_element(
            self.process.api_path,
            "Cards/param_card.dat",
            0,
            self.context.pdf_set().alpha_s(self.process.pdf_scale)
        )
        dxs = me.DifferentialCrossSection( #TODO: single process hardcoded for now
            [(process.all_pid_options[0]["options"][0][0], me_id)],
            self.process.e_cm2,
            self.process.pdf_scale,
            self.n_channels,
            [] if self.n_channels == 1 else self.phasespace.amp2_remap,
        )
        integrands = []
        for i, (mapping, group_indices) in enumerate(zip(
            self.phasespace.mappings, self.phasespace.group_indices
        )):
            vegas = me.VegasMapping(
                mapping.random_dim(), args["vegas"]["bins"], f"channel_{i}"
            )
            vegas.initialize_global(self.context)
            integrands.append(
                me.Integrand(
                    mapping,
                    dxs,
                    vegas,
                    me.EventGenerator.integrand_flags,
                    group_indices,
                )
            )
        #print(integrands[0].function())
        #print(integrands[1].function())

        os.makedirs("Events", exist_ok=True)
        existing_run_dirs = glob.glob("Events/run_*")
        run_index = 1
        while f"Events/run_{run_index:02d}" in existing_run_dirs:
            run_index += 1
        while True:
            try:
                run_path = f"Events/run_{run_index:02d}"
                os.mkdir(run_path)
                break
            except FileExistsError:
                run_index += 1

        gen_args = args["generation"]
        vegas_args = args["vegas"]
        cfg = me.EventGeneratorConfig()
        cfg.target_count = gen_args["events"]
        cfg.vegas_damping = vegas_args["damping"]
        cfg.max_overweight_fraction = gen_args["max_overweight_fraction"]
        cfg.max_overweight_truncation = gen_args["max_overweight_truncation"]
        cfg.start_batch_size = gen_args["start_batch_size"]
        cfg.max_batch_size = gen_args["max_batch_size"]
        cfg.survey_min_iters = gen_args["survey_min_iters"]
        cfg.survey_max_iters = gen_args["survey_max_iters"]
        cfg.survey_target_precision = gen_args["survey_target_precision"]
        cfg.optimization_patience = vegas_args["optimization_patience"]
        cfg.optimization_threshold = vegas_args["optimization_threshold"]
        self.event_generator = me.EventGenerator(
            self.context, integrands, os.path.join(run_path, "events.npy"), cfg
        )

    def survey(self):
        self.event_generator.survey()

    def generate(self):
        self.event_generator.generate()



def main():
    process = MadgraphProcess()
    process.run_survey()
    #event_generator = run_survey(args)
    #start_time = get_start_time()
    #event_generator.generate()
    #print_run_time(start_time)

