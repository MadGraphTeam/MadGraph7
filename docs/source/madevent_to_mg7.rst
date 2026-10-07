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
``set iseed``. MadEvent draws a random seed for ``iseed = 0``. MG7 does the same for
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
      - ``beam.e_cm``
      - The sum of the two energies. Only symmetric beams are supported.
    * - ``lpp1``, ``lpp2``
      - ``beam.leptonic``
      - ``false`` for proton beams and ``true`` for lepton beams without PDFs. Other beam
        types, such as elastic photons or antiprotons, are not supported.
    * - ``pdlabel``, ``lhaid``
      - ``beam.pdf``
      - The name of the LHAPDF set instead of its id. For example, ``lhaid = 331900`` becomes
        ``"NNPDF40_lo_as_01180"``.
    * - ``pdlabel1``, ``pdlabel2``
      - ``beam.pdf``
      - MG7 uses one PDF set for both beams.

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

The cut ``mmll`` only applies to pairs of same-flavor and opposite-sign leptons. The
observable ``sfos_pair_mass`` does the same.

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
* Different PDFs or fixed factorization scales for the two beams, and the scale
  parameters ``mue_over_ref``, ``mue_ref_fixed`` and ``fixed_extra_scale``.
* Cuts on a pseudorapidity minimum (``etajmin`` and similar), on energies (``ej``, ``eb``,
  ``ea``, ``el`` and the ``max`` versions), on ordered particles (``ptj1min`` to
  ``ptl4max``, ``cutuse``), on :math:`H_T` (``htjmin``, ``ihtmin``, ``ht2min`` and
  similar), on summed momenta (``xptj``, ``xptb``, ``xpta``, ``xptl``), on lepton pairs
  (``ptllmin``, ``ptllmax``, ``mmnl``, ``mmnlmax``) and on quarkonia and heavy particles
  (``ptheavy``, ``ptonium``, ``etaonium``).
* Photon isolation: ``ptgmin``, ``r0gamma``, ``xn``, ``epsgamma``, ``isoem``, ``xetamin``,
  ``deltaeta``.
* Cuts for individual particle types: ``pt_min_pdg``, ``pt_max_pdg``, ``e_min_pdg``,
  ``e_max_pdg``, ``eta_min_pdg``, ``eta_max_pdg``, ``mxx_min_pdg`` and
  ``mxx_only_part_antipart``.
* Helicity sampling: ``nhel``, ``limhel``, ``hel_recycling``, ``hel_filtering``,
  ``hel_splitamp``, ``hel_zeroamp``.
* Event output options: ``event_norm``, ``lhe_version``, ``boost_event``, ``me_frame``,
  ``frame_id``, ``systematics_program`` and ``systematics_arguments``.
