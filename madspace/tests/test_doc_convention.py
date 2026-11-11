"""Guard the "expose every constructor parameter" documentation convention.

For each documented ``Mapping`` subclass under ``phasespace/``, construct it
with every parameter its docstring names passed **by keyword**. This fails if a
binding drops a constructor argument (e.g. ``has_cut`` of ``DoubleT``) or
renames one away from the header, keeping the rendered Parameters list honest.
"""

import inspect

import pytest

import madspace as ms

# u u~ > e- e+ g, with the gluon i forming FKS pairs with both quarks
FKS_CONFIGURATION = dict(
    i_fks=4,
    j_fks=0,
    pdgs=[2, -2, 11, -11, 21],
    colors=[3, -3, 1, 1, 8],
    charges=[2 / 3, -2 / 3, -1.0, 1.0, 0.0],
    masses=[0.0] * 5,
    fks_j_from_i=[[], [], [], [], [0, 1]],
    splitting_types=["QCD"],
)
FKS_CONFIGURATIONS = [
    ms.FKSConfiguration(**dict(FKS_CONFIGURATION, j_fks=j)) for j in [0, 1]
]
# its coupling orders, alpha_s alpha^2 at the Born and QCD corrections
FKS_COUPLING_ORDERS = dict(
    split_orders=["QCD", "QED"], born=[[2, 4]], nlo=[[4, 4]], max_nlo=[4, 4]
)

# One representative keyword-argument set per in-scope class. Every key here is
# a parameter the class comment documents; the call must accept all of them.
CASES = {
    "Invariant": dict(power=0.8, mass=0.0, width=0.0),
    "Luminosity": dict(
        s_lab=1.0e8,
        s_hat_min=1.0e4,
        s_hat_max=1.0e8,
        invariant_power=1.0,
        mass=0.0,
        width=0.0,
    ),
    "FKSLeptonLuminosity": dict(
        s_lab=1.0e6, beta=0.975, s_hat_min=1.0e4, s_hat_min_condition=False
    ),
    "TwoBodyDecay": dict(com=True),
    "TwoToTwoParticleScattering": dict(
        com=True, invariant_power=0.0, mass=0.0, width=0.0, has_cut=False
    ),
    "DoubleT": dict(
        t1_invariant_power=0.0,
        t1_mass=0.0,
        t1_width=0.0,
        t2_invariant_power=0.0,
        t2_mass=0.0,
        t2_width=0.0,
        has_cut=False,
    ),
    "ThreeBodyDecay": dict(com=True),
    "TwoToThreeParticleScattering": dict(
        t_invariant_power=0.0,
        t_mass=0.0,
        t_width=0.0,
        s_invariant_power=0.0,
        s_mass=0.0,
        s_width=0.0,
        has_cut=False,
    ),
    "ChiliMapping": dict(n_particles=3, y_max=[], pt_min=[]),
    "FKSMapping": dict(
        born=ms.PhaseSpaceMapping(
            [0.0] * 4, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
        ),
        i_fks=4,
        j_fks=2,
        s_hat_min=0.0,
    ),
    "FKSConfigurationMapping": dict(
        born=ms.PhaseSpaceMapping(
            [0.0] * 4,
            1000.0,
            mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
            sqrt_s_hat_min_condition=True,
        ),
        configurations=[(4, 2), (4, 3)],
        s_hat_min=0.0,
        sqrt_s_hat_floor=100.0,
        luminosity=None,
    ),
    "FKSKinematics": dict(
        mapping=ms.FKSMapping(
            ms.PhaseSpaceMapping(
                [0.0] * 4, 1000.0, mode=ms.PhaseSpaceMapping.TChannelMode.rambo
            ),
            4,
            2,
        )
    ),
    "FKSRadiation": dict(
        born_particle_count=5,
        i_fks=5,
        j_fks=4,
        leptonic=False,
        s_hat_min=0.0,
        emitter_mass=0.0,
    ),
    "FKSConfiguration": FKS_CONFIGURATION,
    "FKSCouplingOrders": FKS_COUPLING_ORDERS,
    "FKSEikonals": dict(configurations=FKS_CONFIGURATIONS),
    "FKSSplittingKernels": dict(configurations=FKS_CONFIGURATIONS),
    "FKSLimits": dict(
        configurations=FKS_CONFIGURATIONS,
        orders=ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS),
        collinear_born=False,
    ),
    "FKSSubtraction": dict(
        configurations=FKS_CONFIGURATIONS,
        orders=ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS),
        xicut=0.5,
        xi_s_cut=0.5,
        delta_o=1.0,
        delta_i=1.0,
        delta_s=1.0,
        fks_a=1.5,
        fks_b=1.5,
        use_energy=True,
        use_beta=True,
        h_damp_one=0.01,
        h_damp_power=1.0,
        tiny=1e-6,
        pdf_scheme=ms.FKSSubtraction.PdfScheme.msbar,
        n_f=5.0,
        upc_radii=[],
        collinear_born=False,
    ),
    "FKSSoftVirtual": dict(
        configurations=FKS_CONFIGURATIONS,
        orders=ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS),
        xicut=0.5,
        delta_o=1.0,
        n_f=5.0,
        n_l=3.0,
    ),
    "FKSWeights": dict(
        configurations=FKS_CONFIGURATIONS,
        orders=ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS),
    ),
    "FKSCuts": dict(configurations=FKS_CONFIGURATIONS, ptj=10.0),
    "FKSScale": dict(particle_count=5, dynamical_scale_choice=1),
    "FKSLuminosity": dict(flavors=[[[2, -2], [4, -4]], [[21, -2]]]),
    "FKSCrossSection": dict(
        subtraction=ms.FKSSubtraction(
            FKS_CONFIGURATIONS, ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS)
        ),
        soft_virtual=ms.FKSSoftVirtual(
            FKS_CONFIGURATIONS, ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS)
        ),
        weights=ms.FKSWeights(
            FKS_CONFIGURATIONS, ms.FKSCouplingOrders(**FKS_COUPLING_ORDERS)
        ),
    ),
    "FKSPrefactors": dict(
        configurations=FKS_CONFIGURATIONS,
        xicut=0.5,
        xi_s_cut=0.5,
        delta_o=1.0,
        delta_i=1.0,
        delta_s=1.0,
    ),
    "FKSSectorFunction": dict(
        configurations=FKS_CONFIGURATIONS,
        fks_a=1.5,
        fks_b=1.5,
        use_energy=True,
        use_beta=True,
        h_damp_one=0.01,
        h_damp_power=1.0,
    ),
    "EWSudakovAmplitude": dict(
        type="ipm2", base=0, legs=[0, 2], old_pdgs=[2, 1], new_pdgs=[1, 2]
    ),
    "EWSudakovProcess": dict(
        pdgs=[2, -2, 1, -1],
        helicities=[[-1, 1, -1, 1]],
        goldstone=[],
        interference=[],
        has_lo1=True,
        has_lo2=False,
        lo2_qcd_power=0,
        incoming=2,
    ),
    "EWSudakovParameters": dict(
        mw=80.4,
        mz=91.2,
        mh=125.0,
        mt=173.0,
        alpha=1 / 132.5,
        gf=1.166e-5,
        scheme=ms.EWSudakovScheme.gmu,
        mb=0.0,
    ),
    "EWSudakovMode": dict(sud_mod=1, s_to_rij=False, rij_ge_mw=True),
    "EWSudakov": dict(
        process=ms.EWSudakovProcess(pdgs=[2, -2, 1, -1], helicities=[[-1, 1, -1, 1]]),
        parameters=ms.EWSudakovParameters(
            mw=80.4, mz=91.2, mh=125.0, mt=173.0, alpha=1 / 132.5
        ),
        modes=[ms.EWSudakovMode()],
    ),
    "FastRamboMapping": dict(n_particles=4, massless=True, com=True),
    "TPropagatorMapping": dict(integration_order=[0], invariant_power=0.8, pt_min=[]),
    "ColorOrderedMapping": dict(
        options=ms.ColorOrderedMapping.Options([0, 1, 2, 3]),
        pt_min=[],
        m_inv_min=[],
        dr_min=[],
        y_max=[],
    ),
    "MultiChannelMapping": dict(
        mappings=[ms.TwoBodyDecay(com=True), ms.TwoBodyDecay(com=True)]
    ),
    "VegasMapping": dict(dimension=3, bin_count=64, prefix=""),
    "Flow": dict(
        input_dim=4,
        condition_dim=0,
        prefix="",
        bin_count=10,
        subnet_hidden_dim=32,
        subnet_layers=3,
        subnet_activation=ms.MLP.Activation.leaky_relu,
        invert_spline=True,
    ),
    "DiscreteSampler": dict(option_counts=[3, 4], prefix="", dims_with_prior=[]),
    "DiscreteFlow": dict(
        option_counts=[3, 4],
        prefix="",
        dims_with_prior=[],
        condition_dim=0,
        subnet_hidden_dim=32,
        subnet_layers=3,
        subnet_activation=ms.MLP.Activation.leaky_relu,
    ),
    "MLP": dict(
        input_dim=4,
        output_dim=2,
        hidden_dim=32,
        layers=3,
        activation=ms.MLP.Activation.leaky_relu,
        prefix="",
    ),
    # A few cheaply-constructible FunctionGenerator subclasses.
    "EnergyScale": dict(
        particle_count=4,
        type=ms.EnergyScale.DynamicalScaleType.transverse_energy,
    ),
    "MomentumPreprocessing": dict(particle_count=4),
    "ChannelWeightNetwork": dict(
        channel_count=3,
        particle_count=4,
        hidden_dim=32,
        layers=3,
        activation=ms.MLP.Activation.leaky_relu,
        prefix="",
        include_preprocessing=True,
    ),
    "VegasHistogram": dict(dimension=3, bin_count=64),
    "DiscreteHistogram": dict(option_counts=[3, 4]),
}


@pytest.mark.parametrize("name", ["MultiChannelFunction"])
def test_multi_channel_function_kwargs(name):
    ms.MultiChannelFunction(
        functions=[ms.MLP(2, 2), ms.MLP(2, 2)], return_batch_sizes=False
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_documented_constructor_kwargs_are_accepted(name):
    cls = getattr(ms, name)
    obj = cls(**CASES[name])
    assert isinstance(obj, cls)


def test_phasespace_mapping_both_constructors():
    ms.PhaseSpaceMapping(
        external_masses=[10.0, 10.0, 10.0, 10.0],
        cm_energy=1000.0,
        leptonic=False,
        invariant_power=0.8,
        mode=ms.PhaseSpaceMapping.TChannelMode.rambo,
        cuts=None,
        color_ordered=None,
        beam_rapidity=0.0,
        mirror_beams=False,
        s_hat_power=None,
        sqrt_s_hat_min_condition=True,
        sqrt_s_hat_condition=False,
    )


def test_matrix_element_index_is_not_diagram_count():
    """`matrix_element_index` and `diagram_count` are separate parameters."""
    me = ms.MatrixElement(
        matrix_element_index=3,
        particle_count=4,
        inputs=[ms.MatrixElement.MatrixElementInput.momenta_in],
        outputs=[ms.MatrixElement.MatrixElementOutput.matrix_element_out],
        diagram_count=1,
    )
    assert me.matrix_element_index() == 3
    assert me.diagram_count() == 1


def test_every_documented_class_is_importable():
    for name in [*CASES, "PhaseSpaceMapping"]:
        assert inspect.isclass(getattr(ms, name))
