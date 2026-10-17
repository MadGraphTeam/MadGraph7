Coming from MadEvent
====================

MadEvent is configured with ``Cards/run_card.dat``, a list of ``value = name`` lines. MG7 uses
``Cards/run_card.toml`` instead, which is described in :doc:`run_card`. This page shows where
the MadEvent settings went. The ``param_card.dat`` is unchanged.

Not every MadEvent setting has a counterpart yet. The last section lists what is missing.

Setting values from old scripts
-------------------------------

At the launch question, a few MadEvent names are translated to their MG7 equivalent, so
existing scripts keep working. These are ``nevents``, ``gridpack``, ``fixed_ren_scale``,
``scale``, ``dsqrt_q2fact1``, ``dsqrt_q2fact2``, ``scalefact``, ``bwcutoff``, ``cut_decays``
and ``use_syst``. The shortcuts ``set lhc``, ``set lep``, ``set fixed_scale``,
``set no_parton_cut`` and ``set dynamical_scale_choice HT/n`` work as before, and so does
``set iseed``. The beam settings ``ebeam``, ``ebeam1``, ``ebeam2``, ``lpp``, ``lpp1``,
``lpp2``, ``pdlabel``, ``pdlabel1``, ``pdlabel2`` and ``lhaid`` are translated as in the
table below. MadEvent draws a random seed for ``iseed = 0``. MG7 does the same for
``seed = -1`` and treats ``0`` as an ordinary fixed seed, so ``set iseed 0`` is translated.
All other settings must use their new names.

Beams and PDFs
--------------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``ebeam1``, ``ebeam2``
      - ``beam.ebeam1``, ``beam.ebeam2``
      - One energy per beam. With different energies the events are written, and the
        :math:`\eta` cuts applied, in the lab frame.
    * - ``lpp1``, ``lpp2``
      - ``beam.leptonic``
      - ``false`` for proton beams and ``true`` for lepton beams without PDFs. Other beam
        types, such as elastic photons or antiprotons, are not supported.
    * - ``pdlabel``, ``lhaid``
      - ``beam.pdf1``, ``beam.pdf2``
      - The name of the LHAPDF set instead of its id. For example, ``lhaid = 331900`` becomes
        ``"NNPDF40_lo_as_01180"``. ``set beam.pdf`` sets both beams.
    * - ``pdlabel1``, ``pdlabel2``
      - ``beam.pdf1``, ``beam.pdf2``
      - One set per beam. A beam without PDF exists only for a lepton collider, as
        ``beam.leptonic``.

Scales
------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``fixed_ren_scale``
      - ``beam.fixed_ren_scale``
      -
    * - ``fixed_fac_scale``
      - ``beam.fixed_fact_scale``
      - A single switch for both beams.
    * - ``scale``
      - ``beam.ren_scale``
      -
    * - ``dsqrt_q2fact1``, ``dsqrt_q2fact2``
      - ``beam.fact_scale1``, ``beam.fact_scale2``
      -
    * - ``dynamical_scale_choice``
      - ``beam.dynamical_scale_choice``
      - A name instead of a number. ``1`` is ``"transverse_energy"``, ``2`` is
        ``"transverse_mass"``, ``3`` is ``"half_transverse_mass"`` and ``4`` is
        ``"partonic_energy"``.
    * - ``scalefact``
      - ``beam.scale_factor``
      - Applies to the dynamical scale only.

The MadEvent default ``dynamical_scale_choice = -1`` follows the clustering of the diagram,
as needed for CKKW merging. MG7 does not have it. Its default is ``half_transverse_mass``.

Generation and phase space
--------------------------

.. list-table::
    :header-rows: 1
    :widths: 30 30 40

    * - ``run_card.dat``
      - ``run_card.toml``
      - Notes
    * - ``nevents``
      - ``generation.events``
      -
    * - ``iseed``
      - ``run.seed``
      - ``-1`` instead of ``0`` for a random seed.
    * - ``run_tag``
      - ``run.run_name``
      - Similar, but the run directory is ``Events/<run_name>_NN``.
    * - ``gridpack``
      - ``gridpack.save_gridpack``
      - See ``[gridpack]`` for further options.
    * - ``bwcutoff``
      - ``phasespace.bw_cutoff``
      -
    * - ``cut_decays``
      - ``phasespace.cut_decays``
      - ``false`` by default in both.
    * - ``SDE_strategy``
      - ``phasespace.sde_strategy``
      - ``1`` is ``"diagrams"`` and ``2`` is ``"denominators"``.
    * - ``maxjetflavor``
      - ``[multiparticles]`` ``jet``
      - List the PDG ids of the quarks that count as jets. The default includes the quarks
        up to the charm quark and the gluon. Add ``5`` and ``-5`` for ``maxjetflavor = 5``.
    * - ``use_syst``
      - ``systematics.enable``
      - The scale and PDF variations are computed natively. Choose them with
        ``systematics.mur``, ``muf`` and ``pdf``.
    * - ``time_of_flight``
      - ``postprocessing.time_of_flight``
      -

Technical settings for the MadEvent run engine, such as ``vector_size``, ``nb_warp``,
``job_strategy``, ``survey_splitting``, ``refine_evt_by_job`` and the ``hel_*`` options, have
no counterpart. The integration and the hardware are configured in ``[run]``,
``[generation]``, ``[vegas]``, ``[phasespace]`` and ``[madnis]`` instead.

Cuts
----

MadEvent has one parameter per cut. MG7 has one entry per observable in ``[cuts]``, named
after the particle groups and the observable, as explained in :ref:`the run card page
<run-card-multiparticles>`. A bound that is switched off in MadEvent, such as a minimum of
zero, is simply left out.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - ``run_card.dat``
      - ``run_card.toml``
    * - ``ptj``, ``ptb``, ``pta``, ``ptl`` and the ``max`` versions
      - ``jet-pt``, ``bottom-pt``, ``photon-pt``, ``lepton-pt`` with ``.min`` and ``.max``
    * - ``misset``, ``missetmax``
      - ``missing-pt.min``, ``missing-pt.max``
    * - ``etaj``, ``etab``, ``etaa``, ``etal``
      - ``jet-eta_abs.max``, ``bottom-eta_abs.max``, ``photon-eta_abs.max``,
        ``lepton-eta_abs.max``
    * - ``drjj``, ``drbb``, ``drll``, ``draa`` and the ``max`` versions
      - ``jet-delta_r``, ``bottom-delta_r``, ``lepton-delta_r``, ``photon-delta_r``
    * - ``drbj``, ``draj``, ``drab``, ``drbl``, ``drjl``, ``dral`` and the ``max`` versions
      - ``bottom-jet-delta_r``, ``photon-jet-delta_r``, ``photon-bottom-delta_r``,
        ``bottom-lepton-delta_r``, ``jet-lepton-delta_r``, ``photon-lepton-delta_r``
    * - ``mmjj``, ``mmbb``, ``mmaa`` and the ``max`` versions
      - ``jet-pair_mass``, ``bottom-pair_mass``, ``photon-pair_mass``
    * - ``mmll``, ``mmllmax``
      - ``lepton-sfos_pair_mass.min``, ``lepton-sfos_pair_mass.max``
    * - ``dsqrt_shat``, ``dsqrt_shatmax``
      - ``sqrt_s.min``, ``sqrt_s.max``
    * - ``etajmin``, ``etabmin``, ``etaamin``, ``etalmin``
      - ``jet-eta_abs.min``, ``bottom-eta_abs.min``, ``photon-eta_abs.min``,
        ``lepton-eta_abs.min``
    * - ``ej``, ``eb``, ``ea``, ``el`` and the ``max`` versions
      - ``jet-e``, ``bottom-e``, ``photon-e``, ``lepton-e`` with ``.min`` and ``.max``
    * - ``ptj1min`` to ``ptj4max``, ``ptl1min`` to ``ptl4max``
      - ``jet_1-pt`` to ``jet_4-pt`` and ``lepton_1-pt`` to ``lepton_4-pt`` with ``.min``
        and ``.max``
    * - ``xptj``, ``xptl``
      - ``jet_1-pt.min``, ``lepton_1-pt.min``. At least one particle passes if the hardest one
        does.
    * - ``htjmin``, ``htjmax``
      - ``jet-pt-sum.min``, ``jet-pt-sum.max``
    * - ``ihtmin``, ``ihtmax``
      - ``parton-pt-sum.min``, ``parton-pt-sum.max``, with a group ``parton`` that holds all
        quarks including the b quark and the gluon
    * - ``ht2min``, ``ht3min``, ``ht4min`` and the ``max`` versions
      - ``jet_1-jet_2-pt-sum``, ``jet_1-jet_2-jet_3-pt-sum``,
        ``jet_1-jet_2-jet_3-jet_4-pt-sum``
    * - ``ptllmin``, ``ptllmax``
      - ``alllepton-sum-pt.min``, ``alllepton-sum-pt.max``, with a group ``alllepton`` that
        holds the charged leptons and the neutrinos
    * - ``mmnl``, ``mmnlmax``
      - ``alllepton-sum-mass.min``, ``alllepton-sum-mass.max``
    * - ``ptheavy``
      - ``heavy_1-pt.min``, with a group ``heavy`` that holds the heavy particles of your
        process
    * - ``pt_min_pdg``, ``pt_max_pdg``, ``e_min_pdg``, ``e_max_pdg``, ``eta_min_pdg``,
        ``eta_max_pdg``
      - A group for the particle, for example ``top = [6, -6]``, and ``top-pt``,
        ``top-e`` or ``top-eta_abs`` with ``.min`` and ``.max``
    * - ``mxx_min_pdg``
      - ``top-pair_mass.min`` for the group ``top = [6, -6]``. Use ``top-sfos_pair_mass``
        instead if ``mxx_only_part_antipart`` is true.

The cut ``mmll`` only applies to pairs of same-flavor and opposite-sign leptons. The
observable ``sfos_pair_mass`` does the same.

The groups in the last rows are your own entries in ``[multiparticles]``. The page about the
:ref:`cut syntax <run-card-cuts>` explains how the selections, ``-sum-`` and ``-sum`` are
combined.

Decay widths
------------

``compute_widths`` and the ``auto`` widths of a ``param_card`` integrate the decays into
three bodies or more with madspace, like the rest of MG7, where MadEvent used to be run. The
two-body widths still come from the formulas of the model when it has them. Each decaying
particle is written as a decay output (``output mg7``) in a temporary directory and surveyed:
the integration takes milliseconds and the build is what there is to wait for. A build of at
least four subprocess directories (this holds for any mg7 run) precompiles the standard
headers first (``make PCH=1``, clang), which halves it. For the Higgs of the ``heft`` model up to three bodies, the whole
command takes 6 s, against 26 s with MadEvent.

``alpha_s`` of a decay is run from the ``alpha_s(m_Z)`` of the card up to the mass of the
decaying particle, with the same two-loop running as MadEvent. The widths of the two
integrators agree within the integration error (the ``--precision_channel`` of
``compute_widths``, 1% by default, is the relative precision aimed at for each channel).

The integrator is chosen with the ``auto_width_engine`` option, ``madspace`` by default, or
for one call with ``--engine``::

    set auto_width_engine madevent
    compute_widths t --body_decay=3 --engine=madspace

MG7 goes back to MadEvent on its own, with a warning, if madspace cannot compute the widths
and no engine was asked for with ``--engine``. The complex mass scheme is not available in
madspace: the widths are always computed with MadEvent then.

NLO widths
^^^^^^^^^^

``compute_widths --nlo`` computes the width at NLO in QCD. A model that comes with a dedicated
code for it (``SMWidth``, which is HDECAY, for the Standard Model) uses it. For the others the
two-body decays that have a colour charge, in the decaying particle or in a decay product, are
run at NLO with MC@NLO (aMC@NLO at fixed order, one run per kind of decay, up to four at a time).
The model needs loop information: ``sm`` is replaced by ``loop_sm``, a model that cannot make
loops stops the computation of that decay, which keeps its LO width with a warning. The decays
into three bodies or more are always at LO (madspace, or MadEvent), so that
``--nlo --body_decay=3`` is the two-body decays at NLO and the three-body ones at LO, and
``--nlo --body_decay=2`` the two-body decays alone::

    compute_widths t w+ z h --nlo --body_decay=3

The result depends on the renormalisation scheme of the model: with the pole-mass Yukawa of
``loop_sm``, ``h > b b~`` gets a correction of -35%, which is the known
``1 + 4/3 alpha_s/pi (9/4 + 3/2 log(m_b^2/m_H^2))`` and not what a physical width with a running
Yukawa has.

The options also go in the param_card, after the ``auto`` of a width::

    DECAY  6 auto --nlo --body_decay=3
    DECAY 25 auto --body_decay=2
    DECAY 23 auto@NLO

(``auto@NLO`` is the older spelling of ``auto --nlo``.) The options are those of
``compute_widths``: ``--nlo``, ``--body_decay``, ``--precision_channel``, ``--min_br`` and
``--engine``. The widths with the same options are computed together, and
``set decay 6 auto --nlo --body_decay=3`` writes the same line from the card editor.

Not supported yet
-----------------

The following MadEvent features have no MG7 equivalent. Settings for them are ignored.

* Matching and merging: ``ickkw``, ``xqcut``, ``ktdurham``, ``ptlund``, ``dparameter``,
  ``highestmult``, ``ktscheme``, ``alpsfact``, ``chcluster``, ``pdfwgt``,
  ``asrwgtflavor``, ``clusinfo``, ``auto_ptj_mjj``, ``pdgs_for_merging_cut``.
* Biasing: ``bias_module`` and ``bias_parameters``.
* Beam polarization and heavy-ion or equivalent-photon beams: ``polbeam1``, ``polbeam2``,
  ``nb_proton1``, ``nb_proton2``, ``nb_neutron1``, ``nb_neutron2``, ``mass_ion1``,
  ``mass_ion2``, ``ievo_eva``, ``evaorder``, ``eva_xcut``.
* Fixed factorization scales for the two beams, and the scale parameters
  ``mue_over_ref``, ``mue_ref_fixed`` and ``fixed_extra_scale``.
* Cuts that are not a bound on a single observable: ``cutuse = 1``, which accepts an event if
  any of the ordered jet cuts passes, and the vector-boson-fusion cuts ``xetamin`` and
  ``deltaeta``, which require the two hardest jets to lie in opposite hemispheres.
* Photon isolation: ``ptgmin``, ``r0gamma``, ``xn``, ``epsgamma``, ``isoem``.
* Helicity sampling: ``nhel``, ``limhel``, ``hel_recycling``, ``hel_filtering``,
  ``hel_splitamp``, ``hel_zeroamp``.
* Event output options: ``event_norm``, ``lhe_version``, ``boost_event``, ``me_frame``,
  ``frame_id``, ``systematics_program`` and ``systematics_arguments``.
