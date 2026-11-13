"""The fixed-order NLO run of madspace on an aMC@NLO output: the FKS
integrand of madspace (FKSKinematics, FKSCrossSection) with the matrix
elements of the Fortran provider (fks_provider.py) and the PDFs, alpha_s,
scales and cuts of the run card, integrated per subprocess and per Born
channel of MadFKS, with its histograms in the info.json of the run, which
MadBoard reads.

    python -m madgraph.mg7.fks_driver OUTPUT [--name RUN] [--accuracy ACC]

The output is an aMC@NLO output of a process with [QCD] whose libraries are
built, as a fixed-order run of MG5aMC (or its compile FO) leaves them. The
integration follows the fixed-order run of MadFKS (driver_mintFO.f with
mint):

- A channel is a Born configuration of a subprocess (born_conf.inc, its
  Born diagram as a madspace Topology) with the multichannel weight of
  MadFKS, amp2 of its diagram over that of all Born diagrams at the Born.
- The Born of a channel comes from the PhaseSpaceMapping of its topology and
  the radiation from FKSKinematics, with the bound on sqrt(s_hat) the cuts
  on the Born imply, under a VEGAS grid over their random numbers. The FKS
  configuration comes from an adaptive choice (as MC_integer), its
  probability proportional to the root mean square of its integrand.
- The matrix elements come from the provider where the cuts let a
  contribution through. The virtual (MadLoop) is evaluated on a fraction of
  the points of a channel, with the average of V / B per NLO order times
  the Born on all of them (the virtual trick, types 14 and 15 of MadFKS),
  the fraction updated as update_virtual_fraction of mint.
- Every iteration adapts the grids, the choice of the configurations, the
  fraction and the averages to its points, as mint does, with the
  iterations, the shares of the channels and the outputs of the core of
  the NLO runs (nlo_driver.py).

The run writes Events/RUN/info.json as the LO generator of madspace does
(status, process and channels with mean and error, histograms, run_times),
with the LO cross section, the cross sections per subprocess and the calls
of the matrix elements in "fks", and the histograms as HwU in
MADatNLO_madspace.HwU. The histograms are those of --histograms (a TOML file
with [histograms] and [multiparticles] as the run_card.toml of MG7, which
can be one) or, by default, those MG7 writes for the final state (the pt
and eta of each object, the masses of pairs, sqrt_s).

Not covered yet: the resonance-local recoil of a final-state emitter inside
a resonance of the Born, PDF schemes other than MSbar, lepton beams with
densities, user scales, FxFx and the photon cuts.
"""

import argparse
import logging
import math
import os

import numpy as np

from madgraph.mg7.fks_provider import FortranProvider, fks_blocks
from madgraph.mg7.nlo_driver import (  # noqa: F401 (the names the tests import)
    NLORun,
    Runtime,
    born_cuts,
    born_topology,
    default_histograms,
    integer,
    madspace,
    read_histograms,
    real,
    run_cuts,
    scale_parameters,
)

logger = logging.getLogger("madgraph.mg7.fks_driver")

# the kinematics of FKSObservableHistograms and their weights
HISTOGRAM_KINEMATICS = (
    ("momenta", "event_weight"),
    ("soft_momenta", "soft_weight"),
    ("collinear_momenta", "collinear_weight"),
    ("soft_collinear_momenta", "soft_collinear_weight"),
    ("born_momenta", "nbody_weight"),
)

class SubprocessIntegrand:
    """The FKS integrand of one subprocess (a P* directory) of an aMC@NLO
    output, per Born channel of MadFKS, the discrete choice of a channel that
    of its FKS configuration (an integrand of nlo_driver)."""

    choice_key = "configuration_probabilities"

    def __init__(self, directory, run_card, param_card, couplings, context, n_f, n_l):
        ms = madspace()
        from madspace.fks import (
            fks_configuration_mapping,
            fks_cross_section,
            fks_cuts,
            fks_minimal_masses,
            fks_resonances,
            fks_scale,
            fks_splitting_kernels,
        )

        self.name = os.path.basename(os.path.normpath(directory))
        self.context = context
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        self.incoming = self.configurations["nincoming"]
        born_pdgs = self.configurations["born_pdgs"]
        masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in born_pdgs
        ]
        if self.incoming == 1:
            energy, leptonic = masses[0], True
        else:
            energy = 2 * math.sqrt(real(run_card["ebeam1"]) * real(run_card["ebeam2"]))
            leptonic = integer(run_card["lpp1"]) == 0
        beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
        self.beams = (0, 0) if leptonic else beams
        real_pdgs = {
            p for c in self.configurations["configurations"] for p in c["real_pdgs"]
        }
        if 22 in real_pdgs and real(run_card["ptgmin"]) > 0:
            raise NotImplementedError("fks_driver: no isolated photons (ptgmin)")
        self.cuts = born_cuts(born_pdgs, run_card) if self.incoming == 2 else None
        minimal = fks_minimal_masses(
            self.configurations,
            masses,
            **{key: real(run_card[key]) for key in ("ptj", "ptl", "mll", "mll_sf")},
            maxjetflavor=integer(run_card["maxjetflavor"]),
        )
        self.channel_mappings = []
        for channel in self.provider.channels():
            topology = born_topology(masses, self.incoming, channel)
            resonances = fks_resonances(topology, self.configurations, minimal)
            if any(members for entry in resonances for members in entry):
                raise NotImplementedError(
                    f"fks_driver: {self.name} has a resonance-local recoil"
                )
            self.channel_mappings.append(
                ms.PhaseSpaceMapping(
                    topology,
                    energy,
                    leptonic=leptonic,
                    sqrt_s_hat_min_condition=self.cuts is not None,
                )
            )
        born = self.channel_mappings[0]
        self.alpha = 1 / float(param_card["sminputs"]["1"])
        scales = scale_parameters(run_card)
        scale = fks_scale(self.configurations, **scales)
        self.born_scale = Runtime(scale.with_particle_count(len(born_pdgs)), context)
        self.kernels = Runtime(
            fks_splitting_kernels(born, self.configurations), context
        )
        cuts = run_cuts(run_card)
        self.pass_cuts = Runtime(fks_cuts(born, self.configurations, **cuts), context)
        self.cross_section = Runtime(
            fks_cross_section(
                born,
                self.configurations,
                n_l=n_l,
                cuts=cuts,
                scales=scales,
                running_coupling=couplings.running,
                pdf_grid=couplings.pdf if self.beams != (0, 0) else None,
                beams=self.beams,
                sum_flavors=True,
                n_f=n_f,
            ),
            context,
        )
        self.kinematics = [
            Runtime(
                ms.FKSKinematics(
                    fks_configuration_mapping(b, self.configurations, cuts=self.cuts)
                ),
                context,
            )
            for b in self.channel_mappings
        ]
        self.dimension = born.random_dim() + 3
        self.configuration_count = len(self.configurations["configurations"])
        self.nlo_count = len(self.configurations["orders"]["nlo"])
        self.born = born
        self.born_pdgs = born_pdgs
        self.energy = energy

    @property
    def channel_count(self):
        return len(self.channel_mappings)

    def channel_dimension(self, index):
        return self.dimension

    def channel_name(self, index):
        return f"{self.name} channel {index + 1}"

    def choice_count(self, index):
        return self.configuration_count

    def histogram_runtime(self, histograms, jets):
        """The FKSObservableHistograms of the histograms with the jets of the
        run card."""
        from madspace.fks import fks_histograms

        return Runtime(
            fks_histograms(self.born, self.configurations, histograms, **jets),
            self.context,
        )

    def histogram_sums(self, runtime, result, jacobian, fks_index):
        """The histogram weights of a batch (result of evaluate) with the
        Jacobian of its VEGAS grid."""
        outputs = dict(result["outputs"])
        for name in [w for _, w in HISTOGRAM_KINEMATICS]:
            outputs[name] = outputs[name] * jacobian
        sums = runtime.by_name({**result["kinematics"], **outputs}, fks_index=fks_index)
        return list(sums.values())

    def matrix_element_calls(self):
        return self.provider.statistics

    def close(self):
        self.provider.close()

    def evaluate(self, channel, x, fks_index, configuration_weight, trick, rng):
        """The weights of a channel at the points x (n, dimension) of its
        random numbers, the FKS configurations and the weights of their
        choice, with the virtual on the fraction of the points of trick:
        the NLO weight per point, its LO part and its virtual residual (that
        of the points with the virtual), the outputs of FKSKinematics and
        FKSCrossSection by name, the virtual and the Born of the points with
        the virtual."""
        out = self.kinematics[channel](x, fks_index=fks_index)
        kernels = self.kernels(
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            fks_index=fks_index,
        )
        ren_scale = self.born_scale(out["born_momenta"])["ren_scale"]
        passes = {
            name: self.pass_cuts(out[name], fks_index=fks_index)["mask"] != 0
            for name in ("momenta", "soft_momenta", "collinear_momenta")
        }
        born_mask = passes["momenta"] | passes["soft_momenta"]
        born_mask |= passes["collinear_momenta"]
        count = len(fks_index)
        # the n-body part, the virtual with it, passes the cuts where the soft
        # counter-event does
        evaluated = passes["soft_momenta"] & (out["born_weight"] != 0)
        evaluated &= rng.random(count) < trick.fraction
        blocks = fks_blocks(
            self.provider,
            out,
            fks_index,
            kernels["k_perp"],
            ren_scale,
            real_mask=passes["momenta"],
            born_mask=born_mask,
            virtual_mask=evaluated,
        )
        amp2 = blocks["amp2"]
        total = amp2.sum(1)
        channel_weight = np.where(
            total > 0, amp2[:, channel] / np.where(total > 0, total, 1), 0.0
        )
        virtual, approximation = trick.inputs(
            blocks["virtual"], blocks["born"], evaluated
        )
        inputs = dict(out)
        inputs.update(blocks)
        inputs.update(
            virtual=virtual,
            channel_weight=channel_weight,
            configuration_weight=configuration_weight,
            alpha=np.full(count, self.alpha),
        )
        result = self.cross_section.by_name(inputs, fks_index=fks_index)
        inputs["virtual"] = approximation
        without = self.cross_section.by_name(inputs, fks_index=fks_index)
        residual = result["cross_section"] - without["cross_section"]
        return dict(
            nlo=result["cross_section"],
            lo=result["lo_weight"],
            residual=residual,
            outputs=result,
            kinematics=out,
            virtual=blocks["virtual"],
            born=blocks["born"],
            evaluated=evaluated,
        )


class FKSRun(NLORun):
    """A fixed-order NLO run of madspace with the FKS integrand on an aMC@NLO
    output (see the module and NLORun of nlo_driver for the arguments)."""

    scheme = "fks"
    work_directory = "madspace_fks"

    def make_integrands(self, directories, n_f, n_l):
        return [
            SubprocessIntegrand(
                directory,
                self.run_card,
                self.param_card,
                self.couplings,
                self.context,
                n_f,
                n_l,
            )
            for directory in directories
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", help="the aMC@NLO output")
    parser.add_argument("--name", default="madspace_fo", help="the run name")
    parser.add_argument("--accuracy", type=float, help="req_acc_FO of the run")
    parser.add_argument("--grid-points", type=int, help="points per grid iteration")
    parser.add_argument("--grid-iterations", type=int, help="grid iterations")
    parser.add_argument("--points", type=int, help="points per channel and iteration")
    parser.add_argument("--max-iterations", type=int, help="integration iterations")
    parser.add_argument("--seed", type=int, help="the seed (iseed of the run card)")
    parser.add_argument(
        "--histograms", help="a TOML file with [histograms] as run_card.toml"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = FKSRun(
        args.output,
        name=args.name,
        accuracy=args.accuracy,
        grid_points=args.grid_points,
        grid_iterations=args.grid_iterations,
        points=args.points,
        max_iterations=args.max_iterations,
        seed=args.seed,
        histograms=args.histograms,
    )
    result = run.run()
    unit = result["unit"]
    print(f"NLO: {result['nlo']['sigma']:.6e} +- {result['nlo']['error']:.2e} {unit}")
    print(f"LO:  {result['lo']['sigma']:.6e} +- {result['lo']['error']:.2e} {unit}")
    for name, entry in result["subprocesses"].items():
        nlo = entry["nlo"]
        print(f"  {name}: {nlo['sigma']:.6e} +- {nlo['error']:.2e}")
    return result


if __name__ == "__main__":
    main()
