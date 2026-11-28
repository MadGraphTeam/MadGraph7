"""The fixed-order NLO run of madspace on an aMC@NLO output: the FKS
integrand of madspace (FKSKinematics, FKSCrossSection) with the matrix
elements of the Fortran provider (nlo/provider.py) and the PDFs, alpha_s,
scales and cuts of the run card, integrated per subprocess and per Born
channel of MadFKS, with its histograms in the info.json of the run, which
MadBoard reads.

    python -m madgraph.mg7.nlo.fks OUTPUT [--name RUN] [--accuracy ACC]

The output is an aMC@NLO output of a process with [QCD] whose libraries are
built, as a fixed-order run of MG5aMC (or its compile FO) leaves them. The
integration follows the fixed-order run of MadFKS (driver_mintFO.f with
mint):

- A channel is a Born configuration of a subprocess (born_conf.inc, its
  Born diagram as a madspace Topology) with the multichannel weight of
  MadFKS, amp2 of its diagram over that of all Born diagrams at the Born.
- The Born of a channel comes from the PhaseSpaceMapping of its topology and
  the radiation from FKSKinematics, with the bound on sqrt(s_hat) the cuts
  on the Born imply, under a VEGAS grid over their random numbers adapted to
  the absolute weights (as mint). The FKS configuration comes from an
  adaptive choice (as MC_integer), its probability proportional to the mean
  absolute integrand of its event and counter-events.
- The matrix elements come from the provider where the cuts let a
  contribution through. The virtual (MadLoop) is evaluated on a fraction of
  the points of a channel, with the average of V / B per NLO order times
  the Born on all of them (the virtual trick, types 14 and 15 of MadFKS),
  the fraction updated as update_virtual_fraction of mint.
- Every iteration adapts the grids, the choice of the configurations, the
  fraction and the averages to its points, as mint does, with the
  iterations, the shares of the channels and the outputs of the core of
  the NLO runs (nlo/core.py).

The run writes Events/RUN/info.json as the LO generator of madspace does
(status, process and channels with mean and error, histograms, run_times),
with the LO cross section, the cross sections per subprocess and the calls
of the matrix elements in "fks", and the histograms as HwU in
MADatNLO_madspace.HwU. The histograms are those of --histograms (a TOML file
with [histograms] and [multiparticles] as the run_card.toml of MG7, which
can be one) or, by default, those MG7 writes for the final state (the pt
and eta of each object, the masses of pairs, sqrt_s).

Beams of dressed electrons and positrons (lpp 3 and -3) take the densities
of pdlabel among the lep_densities of the output, with their momentum
fractions sampled as generate_x_ee of genps_fks.f and the leptonic Born at
the energy they leave. Beams of elastic photons from protons (lpp 2 and -2)
take the equivalent-photon density of epa_proton of MadFKS.

A final-state emitter inside a resonance of the Born radiates with the
resonance-local recoil, as MadFKS installs it (fks_resonances): the
channel mappings take the resonances of their Born configuration, and the
splitting kernels and the subtraction the momentum of the resonance of each
event, zero where it has none.

The cuts are those of cuts.f with the photon isolation and the
recombination of photons (NLOCuts), the photons that must be isolated those
of fks_isolated_photons, and the n-body part takes them at the n-body
configuration of MadFKS (fks_nbody_cuts).

Mixed QCD and EW corrections take the Borns of their extra counterterms from
the provider, the Born of the other mother (a gluon or a photon) of a quark
pair. With tagged photons and QED corrections the blocks and the QED
coupling take the factors of tagged_factors of the provider (the couplings
of the tagged photons in alpha(0), the others in G_mu).

With --ewsudakov the run gives the Sudakov approximation of the EW
corrections too, for sud_mod 0 and 1 as compute_ewsudakov of fks_singular.f:
EWSudakov of madspace with the matrix elements of the provider per helicity
combination of the Born, an n-body part of its own at the order of the LO1
with two powers of QED more (ew_sudakov of the result, per mode). The NLO
cross section and the histograms leave them out, which the integrand of
MadFKS sums with it (nlo_with_ew_sudakov of the result).

The scale of the user (dynamical_scale_choice 0 or 10, user_dynamical_scale
of dummy_fct.f) comes from the provider at the event and its counter-events.
With reweight_scale the run gives the cross section at the factors of
rw_rscale and rw_fscale of mu_R and mu_F too, and their envelope, as the
reweighting of MadFKS (FKSCrossSection with scale_variations). With
reweight_PDF it gives the cross section with every member of the LHAPDF set
of lhaid (its densities and alpha_s at the scales of the run) and the PDF
uncertainty of LHAPDF (FKSCrossSection with pdf_variations).

Not covered yet: PDF schemes other than MSbar, densities of eMELA, photons
of gamma-UPC and FxFx. The run refuses settings of MadFKS it would
otherwise ignore: cuts by PDG code, functions of the user other than the
scale (custom_fcts), more than one dynamical scale choice or PDF set, and
beams of different energies.
"""

import argparse
import logging
import math
import os

import numpy as np

from madgraph.mg7.nlo.provider import FortranProvider, ew_sudakov_inputs, fks_blocks
from madgraph.mg7.nlo.core import (  # noqa: F401 (the names the tests import)
    LEPTON_POWER,
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
    scale_variations,
)

logger = logging.getLogger("madgraph.mg7.nlo.fks")

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
    of its FKS configuration (an integrand of nlo.core)."""

    choice_key = "configuration_probabilities"
    # the pole check of the setup of the run (pole_check of nlo.core)
    takes_virtual = True

    def __init__(self, directory, run_card, param_card, couplings, context, n_f, n_l):
        ms = madspace()
        from madspace.nlo.fks import (
            fks_configuration_mapping,
            fks_cross_section,
            fks_cuts,
            fks_minimal_masses,
            fks_resonances,
            fks_splitting_kernels,
        )
        from madspace.nlo import nlo_scale

        self.name = os.path.basename(os.path.normpath(directory))
        self.context = context
        self.provider = FortranProvider(directory)
        self.configurations = self.provider.fks_configurations
        # the QED coupling of the model, and the factors of tagged photons
        _, self.alpha, _ = self.provider.couplings()
        self.tagged = self.provider.tagged_factors()
        self.incoming = self.configurations["nincoming"]
        born_pdgs = self.configurations["born_pdgs"]
        masses = [
            float(param_card["mass"].get(str(abs(pdg)), 0.0)) for pdg in born_pdgs
        ]
        # dressed leptons have a leptonic Born at the energy their momentum
        # fractions leave, which the luminosity of FKSKinematics gives
        self.leptons = couplings.leptons if self.incoming == 2 else None
        if self.incoming == 1:
            energy, leptonic = masses[0], True
        else:
            energy = 2 * math.sqrt(real(run_card["ebeam1"]) * real(run_card["ebeam2"]))
            leptonic = integer(run_card["lpp1"]) == 0 or self.leptons is not None
        beams = (integer(run_card["lpp1"]), integer(run_card["lpp2"]))
        self.beams = beams if self.incoming == 2 and beams != (0, 0) else (0, 0)
        self.cuts = born_cuts(born_pdgs, run_card) if self.incoming == 2 else None
        minimal = fks_minimal_masses(
            self.configurations,
            masses,
            **{key: real(run_card[key]) for key in ("ptj", "ptl", "mll", "mll_sf")},
            maxjetflavor=integer(run_card["maxjetflavor"]),
        )
        # s_hat of the Born flat below the bound of the cuts and falling
        # above: the Born of an initial-state emitter has no floor, its real
        # emission reaches the bound from below
        s_hat_shift = None
        if self.cuts is not None and not leptonic:
            floor = ms.fks_sqrt_s_hat_min(self.cuts, masses[self.incoming :])
            s_hat_shift = floor**2 if floor > 0 else None
        # the resonances of the resonance-local recoil of each channel, as
        # MadFKS installs them for its Born configuration
        self.channel_mappings, self.resonances = [], []
        for channel in self.provider.channels():
            topology = born_topology(masses, self.incoming, channel)
            self.resonances.append(
                fks_resonances(topology, self.configurations, minimal)
            )
            self.channel_mappings.append(
                ms.PhaseSpaceMapping(
                    topology,
                    energy,
                    leptonic=leptonic,
                    sqrt_s_hat_min_condition=self.cuts is not None,
                    sqrt_s_hat_condition=self.leptons is not None,
                    s_hat_shift=s_hat_shift,
                )
            )
        born = self.channel_mappings[0]
        self.resonance = any(
            members
            for channel in self.resonances
            for entry in channel
            for members in entry
        )
        scales = scale_parameters(run_card)
        scale = nlo_scale(len(born_pdgs) + 1, self.incoming, **scales)
        self.born_scale = Runtime(scale.with_particle_count(len(born_pdgs)), context)
        # the scale of the user, from the provider
        self.user_scale = scale.user()
        self.kernels = Runtime(
            fks_splitting_kernels(born, self.configurations, self.resonance), context
        )
        cuts = run_cuts(run_card)
        self.pass_cuts = Runtime(fks_cuts(born, self.configurations, **cuts), context)
        # the factors of mu_R and mu_F of the scale variations, and the
        # members of the PDF set
        self.scale_variations = scale_variations(run_card)
        hadrons = abs(self.beams[0]) == 1
        members = couplings.members if hadrons else []
        self.pdf_members = len(members)
        # the Sudakov approximation of the EW corrections, with --ewsudakov,
        # with tagged photons with the factor of the power of e of its order
        self.ew_sudakov, self.ew_sudakov_modes = None, 0
        if self.provider.ew_sudakov_info() is not None:
            self.ew_sudakov = ew_sudakov_runtime(self.provider, context)
            self.ew_sudakov_modes = self.ew_sudakov.function.mode_count()
            self.ew_sudakov_factor = 1.0
            if self.tagged is not None:
                power = ew_sudakov_order(self.configurations)[self.provider.qed_pos]
                _, _, (self.ew_sudakov_factor,) = self.provider.couplings([power])
        self.cross_section = Runtime(
            fks_cross_section(
                born,
                self.configurations,
                n_l=n_l,
                cuts=cuts,
                scales=scales,
                running_coupling=couplings.running,
                pdf_grid=couplings.pdf if hadrons else None,
                beams=self.beams,
                sum_flavors=True,
                lepton_luminosity=None if self.leptons is None else self.leptons.full,
                scale_variations=self.scale_variations,
                pdf_variations=members,
                ew_sudakov_modes=self.ew_sudakov_modes,
                resonance=self.resonance,
                n_f=n_f,
            ),
            context,
        )
        # the momentum fractions of dressed leptons as generate_x_ee of
        # genps_fks.f samples them, with the full Jacobian for their full
        # densities
        luminosity = None
        if self.leptons is not None:
            luminosity = ms.LeptonLuminosity(
                energy**2,
                0.0,
                LEPTON_POWER,
                reduced_density=False,
                s_hat_min_condition=True,
            )
        mappings = [
            fks_configuration_mapping(
                b,
                self.configurations,
                cuts=self.cuts,
                luminosity=luminosity,
                resonances=resonances,
            )
            for b, resonances in zip(self.channel_mappings, self.resonances)
        ]
        self.kinematics = [
            Runtime(ms.FKSKinematics(mapping), context) for mapping in mappings
        ]
        self.dimension = mappings[0].random_dim()
        self.configuration_count = len(self.configurations["configurations"])
        self.nlo_count = len(self.configurations["orders"]["nlo"])
        self.born_count = len(self.configurations["orders"]["born"])
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
        from madspace.nlo.fks import fks_histograms

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

    def channel_integrand(self, channel, context):
        """The C++ integrand of a channel (FKSChannelIntegrand of madspace)
        with the matrix elements of the provider as UMAMI libraries in the
        context (ProviderLibraries of nlo/libraries.py, added once), the same
        integrand as evaluate at the same random numbers, configurations and
        points with the virtual, with the Borns of the extra counterterms, the
        reference scale of the user and the Sudakov approximation of the EW
        corrections. The libraries evaluate at the e of the model, so that
        the blocks are those of the provider."""
        from madgraph.mg7.nlo.libraries import ProviderLibraries

        ms = madspace()
        element = ms.MatrixElement
        if getattr(self, "_libraries", None) is None:
            self._libraries = ProviderLibraries(self.provider, self.name)
            self._apis = self._libraries.add_to(context)
        apis = self._apis
        born_outputs = [
            element.squared_orders_out,
            element.diagram_amp2_out,
            element.spin_correlated_out,
        ]
        if self._libraries.has_color:
            born_outputs.append(element.color_correlated_out)
        born = element(
            apis["born"],
            [
                element.momenta_in,
                element.alpha_s_in,
                element.g_ew_in,
                element.spin_legs_in,
                element.spin_vectors_in,
            ],
            born_outputs,
        )
        reals = [
            element(
                api,
                [element.momenta_in, element.alpha_s_in, element.g_ew_in],
                [element.squared_orders_out],
            )
            for api in apis["reals"]
        ]
        loop = element(
            apis["loop"],
            [
                element.momenta_in,
                element.alpha_s_in,
                element.g_ew_in,
                element.scale_in,
            ],
            [element.loop_out],
        )
        extras = [
            element(
                api,
                [
                    element.momenta_in,
                    element.alpha_s_in,
                    element.g_ew_in,
                    element.spin_legs_in,
                    element.spin_vectors_in,
                ],
                [element.squared_orders_out, element.spin_correlated_out],
            )
            for api in apis["extras"]
        ]
        reference_scale = None
        if self.user_scale:
            reference_scale = ms.CustomFunction(
                apis["scale"],
                ms.CustomFunction.scale,
                self.provider.particle_count,
            )
        tagged = self.tagged or {}
        # the Sudakov approximation of the EW corrections on its blocks
        ew_sudakov = blocks = None
        if self.ew_sudakov is not None:
            ew_sudakov = self.ew_sudakov.function
            blocks = element(
                apis["ew_sudakov"],
                [element.momenta_in, element.alpha_s_in, element.g_ew_in],
                [
                    element.ew_sudakov_diagonal_out,
                    element.ew_sudakov_lo2_out,
                    element.ew_sudakov_derivatives_out,
                    element.ew_sudakov_interference_out,
                ],
            )
        return ms.FKSChannelIntegrand(
            self.kinematics[channel].function,
            self.kernels.function,
            self.pass_cuts.function,
            self.born_scale.function,
            self.cross_section.function,
            born,
            reals,
            loop,
            self._libraries.real_indices(),
            self._libraries.spin_legs(),
            apis["born"].color_representations(),
            channel,
            self._libraries.e,
            self.alpha,
            tagged_nlo=[float(v) for v in tagged.get("nlo", [])],
            tagged_born=[float(v) for v in tagged.get("born", [])],
            tagged_alpha=float(tagged.get("alpha", 1.0)),
            reference_scale=reference_scale,
            extras=extras,
            extra_indices=self._libraries.extra_indices() if extras else [],
            extra_spin_legs=self._libraries.extra_spin_legs() if extras else [],
            ew_sudakov=ew_sudakov,
            ew_sudakov_blocks=blocks,
            ew_sudakov_factor=getattr(self, "ew_sudakov_factor", 1.0),
        )

    def generator_integrand(
        self,
        channel,
        context,
        prefix,
        fraction,
        min_fraction,
        bins,
        histograms,
        jets,
        flow=None,
        discrete=None,
        madnis_training=False,
    ):
        """The FKSIntegrand of a channel for the C++ event generator (see
        nlo.core): channel_integrand with a VEGAS grid of bins bins, the
        virtual trick from fraction down to min_fraction and the histograms
        (with the jets of the run card), its globals with the prefix defined
        in the context. With a flow (and a DiscreteFlow of the
        configurations), the integrand takes them instead of the VEGAS grid
        and the probabilities of the configurations, and keeps the globals
        of the integrand before it, for the MadNIS training with
        madnis_training."""
        ms = madspace()
        integrand = self.channel_integrand(channel, context)
        adaptive_map = flow
        if flow is None:
            adaptive_map = ms.VegasMapping(
                self.channel_dimension(channel), bins, prefix
            )
        observables = None
        if histograms and not madnis_training:
            from madspace.nlo.fks import fks_histograms

            observables = fks_histograms(
                self.born, self.configurations, histograms, **jets
            )
        generator = ms.FKSIntegrand(
            integrand,
            adaptive_map,
            prefix,
            virtual_fraction=fraction,
            min_virtual_fraction=min_fraction,
            histograms=observables,
            discrete=discrete,
            madnis_training=madnis_training,
        )
        if flow is None:
            generator.initialize_globals(context)
        return generator

    def close(self):
        self.provider.close()

    def evaluate(self, channel, x, fks_index, configuration_weight, trick, rng):
        """The weights of a channel at the points x (n, dimension) of its
        random numbers, the FKS configurations and the weights of their
        choice, with the virtual on the fraction of the points of trick:
        the NLO weight per point, its LO part, its n-body part and its
        virtual residual (that of the points with the virtual), the outputs
        of FKSKinematics and
        FKSCrossSection by name, the virtual and the Born of the points with
        the virtual."""
        out = self.kinematics[channel](x, fks_index=fks_index)
        if self.resonance and "resonance_momentum" not in out:
            # a channel without resonances
            out["resonance_momentum"] = np.zeros((len(fks_index), 4))
        resonance = [out["resonance_momentum"]] if self.resonance else []
        kernels = self.kernels(
            out["collinear_momenta"],
            out["p_i_hat"],
            out["born_momenta"],
            *resonance,
            fks_index=fks_index,
        )
        references = {}
        if self.user_scale:
            # the reference scales of the user at the event and its
            # counter-events, the Born at the soft counter-event
            for name, momenta, valid in (
                ("reference_scale", "momenta", out["weight"] != 0),
                (
                    "soft_reference_scale",
                    "soft_momenta",
                    (out["soft_det"] != 0) | (out["born_weight"] != 0),
                ),
                (
                    "collinear_reference_scale",
                    "collinear_momenta",
                    out["collinear_det"] != 0,
                ),
            ):
                references[name] = self.provider.reference_scales(out[momenta], valid)
        user = [references["soft_reference_scale"]] if self.user_scale else []
        ren_scale = self.born_scale(out["born_momenta"], *user)["ren_scale"]
        passes = {
            name: self.pass_cuts(out[name], layout_index=fks_index)["mask"] != 0
            for name in ("momenta", "soft_momenta", "collinear_momenta")
        }
        # the soft counter-event only where it exists, at xi = 0, as
        # FKSCrossSection
        passes["soft_momenta"] &= out["soft_xi"] == 0
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
        alpha = self.alpha
        if self.tagged is not None:
            # the couplings of tagged photons in alpha(0), the others in G_mu
            for name in ("matrix_element", "virtual"):
                blocks[name] = blocks[name] * self.tagged["nlo"]
            for name in ("born", "spin_correlated"):
                blocks[name] = blocks[name] * self.tagged["born"]
            b = self.born_count
            factors = self.tagged["born"][:b, None, None]
            blocks["color_correlated"] = blocks["color_correlated"] * factors
            alpha = alpha * self.tagged["alpha"]
        amp2 = blocks["amp2"]
        total = amp2.sum(1)
        channel_weight = np.where(
            total > 0, amp2[:, channel] / np.where(total > 0, total, 1), 0.0
        )
        # the Born orders, without the Borns of the extra counterterms
        born = blocks["born"][:, : self.born_count]
        virtual, approximation = trick.inputs(blocks["virtual"], born, evaluated)
        inputs = dict(out)
        inputs.update(blocks)
        inputs.update(references)
        inputs.update(
            virtual=virtual,
            channel_weight=channel_weight,
            configuration_weight=configuration_weight,
            alpha=np.full(count, alpha),
        )
        if self.ew_sudakov is not None:
            # at the Born of the points that take it, as compute_ewsudakov
            # with the Born
            nbody = born_mask & (out["born_weight"] != 0)
            inputs["ew_sudakov"] = self.ew_sudakov_factor * ew_sudakov_corrections(
                self.provider,
                self.ew_sudakov,
                out["born_momenta"],
                fks_index,
                ren_scale,
                nbody,
            )
        result = self.cross_section.by_name(inputs, fks_index=fks_index)
        inputs["virtual"] = approximation
        without = self.cross_section.by_name(inputs, fks_index=fks_index)
        residual = result["cross_section"] - without["cross_section"]
        return dict(
            nlo=result["cross_section"],
            lo=result["lo_weight"],
            nbody=result["nbody_weight"],
            residual=residual,
            outputs=result,
            kinematics=out,
            virtual=blocks["virtual"],
            born=born,
            evaluated=evaluated,
            variations=np.hstack(
                [
                    result.get("scale_variations", np.zeros((count, 0))),
                    result.get("pdf_variations", np.zeros((count, 0))),
                ]
            ),
            ew_sudakov=result.get("ew_sudakov_weights", np.zeros((count, 0))),
        )


def ew_sudakov_order(configurations):
    """The split orders of the Sudakov approximation of the EW corrections,
    those of the LO1 with two powers of QED more (of the LO2 with two of QCD
    more), from the ewsudakov entry of fks_configurations.json."""
    info = configurations["ewsudakov"]
    split_orders = configurations["orders"]["split_orders"]
    if info.get("lo1") is not None:
        order, raised = list(info["lo1"]), "QED"
    else:
        order, raised = list(info["lo2"]), "QCD"
    order[split_orders.index(raised)] += 2
    return order


def ew_sudakov_runtime(provider, context):
    """The EWSudakov of the Born of the subprocess of a provider with
    --ewsudakov (sud_mod 0 and 1, as compute_ewsudakov), with the parameters
    of its Fortran, as a Runtime."""
    ms = madspace()
    from madspace.nlo.fks import ew_sudakov

    info = provider.ew_sudakov_info()
    p = info["parameters"]
    scheme = provider.fks_configurations["ewsudakov"]["scheme"]
    parameters = ms.EWSudakovParameters(
        mw=p["mw"],
        mz=p["mz"],
        mh=p["mh"],
        mt=p["mt"],
        mb=p["mb"],
        alpha=p["gal"] ** 2 / (4 * math.pi),
        gf=p["gf"],
        scheme=ms.EWSudakovScheme(scheme),
    )
    return Runtime(ew_sudakov(provider.fks_configurations, parameters), context)


def ew_sudakov_corrections(provider, runtime, momenta, fks_index, ren_scale, mask):
    """The outputs of the EWSudakov runtime (batch, modes) at the Born
    kinematics momenta (batch, n, 4) of the configurations fks_index, with
    the matrix elements of the provider at g = 1 and the LO2 part at
    alpha_s = 1 / (4 pi) and the renormalization scale of the Born, zero
    where mask is false."""
    count = len(momenta)
    out = np.zeros((count, runtime.function.mode_count()))
    points = np.flatnonzero(mask)
    if len(points) == 0:
        return out
    p = np.ascontiguousarray(momenta[points])
    inputs = ew_sudakov_inputs(provider, runtime.function, p, fks_index[points])
    out[points] = runtime(
        p,
        inputs["diagonal"],
        inputs["lo2"],
        inputs["derivatives"],
        inputs["interference"],
        np.full(len(points), 1 / (4 * math.pi)),
        np.asarray(ren_scale)[points],
    )["sudakov"]
    return out


class FKSRun(NLORun):
    """A fixed-order NLO run of madspace with the FKS integrand on an aMC@NLO
    output (see the module and NLORun of nlo.core for the arguments)."""

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
        "--generator",
        action="store_true",
        help="run the iterations in the C++ event generator of madspace",
    )
    parser.add_argument(
        "--threads", type=int, default=-1, help="threads of the event generator"
    )
    parser.add_argument(
        "--madnis-batches",
        type=int,
        default=0,
        help="batches of a MadNIS training per subprocess, with --generator",
    )
    parser.add_argument(
        "--madnis-discrete",
        action="store_true",
        help="train a DiscreteFlow of the FKS configurations as well",
    )
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
        generator=args.generator,
        threads=args.threads,
        madnis_batches=args.madnis_batches,
        madnis_discrete=args.madnis_discrete,
    )
    result = run.run()
    unit = result["unit"]
    print(f"NLO: {result['nlo']['sigma']:.6e} +- {result['nlo']['error']:.2e} {unit}")
    print(f"LO:  {result['lo']['sigma']:.6e} +- {result['lo']['error']:.2e} {unit}")
    if "scale_uncertainty" in result:
        band = result["scale_uncertainty"]
        print(f"scale variation: +{band['up']:.1f}% -{band['down']:.1f}%")
    if "pdf_uncertainty" in result:
        band = result["pdf_uncertainty"]
        print(
            f"PDF variation ({band['name']}, {band['error_type']}): "
            f"+{band['up']:.1f}% -{band['down']:.1f}%"
        )
    for entry in result.get("ew_sudakov", []):
        print(
            f"EW Sudakov (sud_mod {entry['sud_mod']}): "
            f"{entry['sigma']:.6e} +- {entry['error']:.2e} {unit}"
        )
    for name, entry in result["subprocesses"].items():
        nlo = entry["nlo"]
        print(f"  {name}: {nlo['sigma']:.6e} +- {nlo['error']:.2e}")
    return result


if __name__ == "__main__":
    main()
