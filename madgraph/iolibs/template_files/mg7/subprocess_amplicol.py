"""Subprocesses with AmpliCol matrix elements ('[amplicol]' processes).

Each subprocess is one colour ordering (UMAMI channel) of an AmpliCol provider,
integrated on its own at leading colour with a colour-ordered phase-space
mapping. A second pass re- or unweights its events to full colour with the
ratio of the full-colour to the leading-colour matrix element summed over all
colour orderings, at the helicity of the event.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from madgraph.iolibs.template_files.mg7.common import lhe_helicities, ms
from madgraph.iolibs.template_files.mg7.observables import clean_pids
from madgraph.iolibs.template_files.mg7.subprocess_madmatrix import (
    Channel,
    MadgraphSubprocess,
    PhaseSpace,
)

if TYPE_CHECKING:
    from madgraph.iolibs.template_files.mg7.madgraph_process import MadgraphProcess


def is_amplicol(meta: dict) -> bool:
    return meta.get("backend") == "amplicol"


def amplicol_parameters(meta: dict, process: MadgraphProcess) -> dict[str, float]:
    """The values of the mutable model parameters in the param card."""
    parameters = {}
    for name, (block, code) in meta["parameters"].items():
        try:
            parameters[name] = float(
                process.param_card.get_value(block.lower(), tuple(code))
            )
        except (KeyError, ValueError):
            continue
    return parameters


def chain_topology(incoming_masses: list[float], outgoing_masses: list[float]):
    """A t-channel chain without propagators, for the LHE completion."""
    n_out = len(outgoing_masses)
    vertices = [["i0", "p0", "o0"]]
    vertices += [[f"p{i - 1}", f"p{i}", f"o{i}"] for i in range(1, n_out - 1)]
    vertices.append(["i1", f"p{n_out - 2}", f"o{n_out - 1}"])
    propagators = [ms.Propagator() for _ in range(n_out - 1)]
    return ms.Topology(
        ms.Diagram(incoming_masses, outgoing_masses, propagators, vertices)
    )


def amplicol_subproc_args(meta: dict, process: MadgraphProcess) -> ms.SubprocArgs:
    """LHE completion of an AmpliCol subprocess: its events have no diagram."""
    return ms.SubprocArgs(
        topologies=[chain_topology(
            [process.get_mass(pid) for pid in clean_pids(meta["incoming"])],
            [process.get_mass(pid) for pid in clean_pids(meta["outgoing"])],
        )],
        permutations=[[]],
        diagram_indices=[[]],
        diagram_color_indices=[[]],
        color_flows=meta["color_flows"],
        pdg_color_types={
            int(key): value for key, value in meta["pdg_color_types"].items()
        },
        helicities=lhe_helicities(meta, process.run_card),
        pdg_ids=[flavor["options"] for flavor in meta["flavors"]],
    )


class AmplicolSubprocess(MadgraphSubprocess):
    def __init__(self, process, meta, subproc_id, unmerged_meta=None):
        super().__init__(process, meta, subproc_id, unmerged_meta)
        self.check_flavor_blind_cuts()

    def check_flavor_blind_cuts(self):
        """AmpliCol groups the flavours and identical-particle orderings that
        share a matrix element into one channel, so its cuts, built for the
        first flavour, must select the same particles for all of them."""
        options = [
            clean_pids(pdgs) for flavor in self.meta["flavors"]
            for pdgs in flavor["options"]
        ]
        for item in self.process.cut_data:
            for group in item.observable_kwargs["select_pids"]:
                if len({tuple(pid in group for pid in pdgs) for pdgs in options}) > 1:
                    raise RuntimeError(
                        f"the cut {item.observable_kwargs['name']} does not select "
                        f"the same particles in all flavours of the [amplicol] "
                        f"subprocess {self.meta['incoming']} > {self.meta['outgoing']}"
                    )

    def load_matrix_elements(self, all_api_paths, api_path_formats, member_metas):
        run_args = self.process.run_card["run"]
        if any(device != "cpu" for device in self.process.device_types):
            raise RuntimeError("[amplicol] processes are only supported on CPUs")
        if run_args["dummy_matrix_element"]:
            raise RuntimeError("[amplicol] processes need their matrix elements")
        if self.process.run_card["gridpack"]["save_gridpack"]:
            raise RuntimeError("[amplicol] processes do not support gridpacks yet")
        self.supports_alpha_s = self.meta.get("supports_alpha_s", True)
        if not self.supports_alpha_s and not self.process.run_card["beam"]["fixed_ren_scale"]:
            raise RuntimeError(
                "the AmpliCol matrix elements of this model cannot take the running "
                "alpha_s, as the model has no SMINPUTS 3 parameter: use a fixed "
                "renormalization scale"
            )
        parameters = amplicol_parameters(self.meta, self.process)
        self.channel_index = self.meta["umami_channel"]
        self.matrix_elements = [
            self.load(self.meta["me_path"], self.meta["artifact"], parameters)
        ]
        self.surrogates = []
        self.ratio_matrix_element = None
        if self.process.run_card["amplicol"]["color_accuracy"] != "lc":
            self.ratio_matrix_element = self.load(
                self.meta["ratio_path"], self.meta["ratio_artifact"], parameters
            )

    def load(self, path: str, artifact: str, parameters: dict):
        # loaded once for all channels; each load reads the whole pyamplicol artifact
        apis = self.process.amplicol_apis
        if path not in apis:
            for context in self.process.contexts:
                apis[path] = context.load_matrix_element(
                    path, os.path.abspath(artifact), parameters
                )
        return apis[path]

    def build_multi_channel_data(self):
        return None

    def build_multichannel_phasespace(self) -> PhaseSpace:
        mapping = ms.PhaseSpaceMapping(
            self.incoming_masses + self.outgoing_masses,
            self.process.e_cm,
            leptonic=self.process.leptonic,
            invariant_power=self.process.run_card["phasespace"]["invariant_power"],
            mode=ms.PhaseSpaceMapping.color_ordered,
            cuts=self.cuts,
            color_order=self.meta["phasespace_order"],
        )
        prefix = f"subproc{self.subproc_id}.amplicol"
        discrete_sym, discrete_flavor = self.build_discrete(
            1, len(self.meta["flavors"]), prefix
        )
        channel = Channel(
            phasespace_mapping=mapping,
            adaptive_mapping=self.build_vegas(mapping, prefix),
            discrete_sym=discrete_sym,
            discrete_flavor=discrete_flavor,
            channel_weight_indices=[0],
            name="C",
            active_flavors=[],
        )
        return PhaseSpace(mode="multichannel", channels=[channel], symfact=[None])

    def build_flat_phasespace(self) -> PhaseSpace:
        return self.build_multichannel_phasespace()

    def simplify_phasespace(self, multi_phasespace, cross_sections) -> PhaseSpace:
        return multi_phasespace

    def integrand_matrix_elements(self) -> list:
        # without an alpha_s input, the model value of the fixed scale is used
        inputs = [
            ms.MatrixElement.momenta_in,
            *([ms.MatrixElement.alpha_s_in] if self.supports_alpha_s else []),
            ms.MatrixElement.flavor_in,
            ms.MatrixElement.channel_in,
            ms.MatrixElement.random_helicity_in,
        ]
        outputs = [
            ms.MatrixElement.matrix_element_out,
            ms.MatrixElement.color_index_out,
            ms.MatrixElement.helicity_index_out,
        ]
        return [ms.MatrixElement(
            self.matrix_elements[0].index(), self.particle_count, inputs, outputs, 1,
            True, self.process.me_frame, self.process.incoming_count,
            self.channel_index,
        )]

    def has_weighting_steps(self) -> bool:
        return self.ratio_matrix_element is not None

    def weighting_steps(self) -> list:
        """The second pass to full colour."""
        if self.ratio_matrix_element is None:
            return []
        ratio = ms.MatrixElement(
            self.ratio_matrix_element.index(),
            self.particle_count,
            [ms.MatrixElement.momenta_in, ms.MatrixElement.flavor_in,
             ms.MatrixElement.channel_in, ms.MatrixElement.helicity_in],
            [ms.MatrixElement.matrix_element_ratio_out],
            1,
            False,
            self.process.me_frame,
            self.process.incoming_count,
            self.channel_index,
        )
        gen_args = self.process.run_card["generation"]
        return [ms.WeightingStep(
            [ratio],
            mode=self.process.run_card["amplicol"]["mode"],
            max_overweight_truncation=gen_args["max_overweight_truncation"],
            freeze_max_weight_after=gen_args["freeze_max_weight_after"],
        )]
