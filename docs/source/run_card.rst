Run card
========

The run card ``Cards/run_card.toml`` holds the settings of an MG7 run. It is a
`TOML <https://toml.io>`_ file with one section per topic. This page describes every
section. If you are coming from MadEvent, see :doc:`madevent_to_mg7`.

Editing the run card
--------------------

You can edit the file directly or change values at the launch question with ``set``:

.. code-block:: text

    set events 50000
    set beam.e_cm 14 TeV
    set jet-pt.min 30
    set seed 42

A key name without its section works as long as it is unique. Otherwise write
``section.key``. Energies accept units (``eV`` to ``PeV``) and are stored in GeV. Numeric
values also accept arithmetic expressions with the masses ``mz``, ``mw``, ``mh``, ``mt``,
``mb``, ``mc``, ``mtau`` and ``mmu``, for example ``set fixed_scale mz/2``. ``help <name>`` prints
a short description of a parameter.

Some parameters can be set to ``"auto"``. MG7 then chooses the value during the run. A
number you set explicitly always wins.

The sections ``[multiparticles]``, ``[cuts]`` and ``[histograms]`` are lists rather than
fixed parameters. Their entries are described below.

Hidden option groups
--------------------

To keep the card short, some option groups are left out of it by default: ``[gridpack]``,
``[cluster]`` (with ``[cluster_train]`` and ``[cluster_deploy]``),
``[vegas]``, ``[custom_functions]``, ``[surrogate]``, ``[amplicol]`` (shown for an output with
``[amplicol]`` processes), ``[merging]`` (shown for a merged sample) and every ``[madnis]`` option but ``enable`` and ``train_batches``
(the group ``madnis``). The second line of the card lists the hidden groups. They keep their
default values, and

- ``update <group>`` at the launch prompt, e.g. ``update vegas`` or ``update madnis``, adds a
  group to the card, where it stays from then on (the card lists it as a revealed option group);
  ``update to_full`` adds all of them, together with the hidden parameters of the other sections;
- ``set`` works on a hidden parameter as on any other: the card then shows its line alone, in
  its section, and so do the entries of a hidden group added to the card by hand;
- a card without the group lines at its top, e.g. one written by hand, shows the groups it has
  entries of.

The ``[cluster]`` sections are shown whole once cluster mode is enabled or one of their
settings differs from its default, and ``[custom_functions]`` once a function has parameters.

[run]
-----

``seed`` (``-1``)
    Random seed. The same seed reproduces a run exactly. ``-1`` draws a new seed for each
    run. The seed that was used is recorded in the output.

``device`` (``["cpu"]``)
    List of devices to run on. Each entry is ``cpu``, ``cuda`` or ``hip``, optionally
    followed by a device index such as ``"cuda:1"``. Several devices share the work.

``cpu_mode`` (``"auto"``)
    SIMD width used on ``cpu`` devices. Choose from ``auto``, ``scalar``, ``simd_128``,
    ``simd_256``, ``simd_512`` and ``avx512y``. ``auto`` picks the widest width the host
    supports.

``madspace_cpu_mode`` (``"auto"``)
    SIMD width of the phase-space sampling on ``cpu`` devices. Choose from ``auto``,
    ``scalar``, ``simd_128`` (ARM only), ``simd_256``, ``avx512y`` and ``simd_512``.
    ``auto`` picks ``simd_512``, then ``simd_256`` on x86. On ARM it picks ``scalar``,
    because the vectorized mode is currently slower there.

``precision`` (``"denom64"``)
    Floating-point precision of the matrix elements. ``all64`` and ``all32`` use double and
    single precision throughout. ``color32`` evaluates the color algebra in single precision
    and everything else in double precision. ``denom64``, the default, keeps the momenta and
    propagator denominators in double precision and uses single precision elsewhere.

``precision_check`` (``0``)
    Fraction of the events evaluated again with the matrix elements in ``all64``, to check
    the precision of a run in any other precision. Off by default, because the run then
    also builds an ``all64`` CPU library of each subprocess (``[run] validation``
    reweights the whole sample through MadtRex instead). Once the events are written, the
    two are compared at the momenta of events drawn at random from the output (at most
    1000 of them, hidden entry ``precision_check_max_events``). With
    ``r = |M|^2 / |M|^2_all64 - 1`` per event, the mean of ``r`` (the relative change of the
    cross section, since unweighted events already follow ``|M|^2``), the mean and the
    maximum of ``|r|`` are logged and written to ``Events/<run>/precision_check.json``; the
    run warns when an event deviates by more than ``1e-3`` (hidden entry
    ``precision_check_threshold``) or is not finite, in which case ``all64`` or ``color32``
    is the safer choice for that process.

``validation`` (``""``)
    FPTYPE to check the generated weights against, after the run: the events are
    reweighted to the matrix elements compiled at that precision, and the number of
    significant digits they share with the weights is reported. ``d`` (all64) checks
    every event; ``d:1000`` checks 1000 of them, evenly spaced through the sample.
    ``""`` is off. It needs MadtRex, and is not run for a cluster run or a gridpack.

``cpu_thread_pool_size`` (``-1``), ``gpu_thread_pool_size`` (``1``), ``combine_thread_pool_size`` (``-1``)
    Number of threads for each ``cpu`` device, each GPU device and for combining the
    channel results. ``-1`` chooses based on the number of CPUs.


``verbosity`` (``"auto"``)
    Amount of console output: ``silent``, ``pretty`` or ``log``. ``auto`` is ``pretty`` in
    a terminal and ``log`` otherwise.

``dummy_matrix_element`` (``false``)
    Skip the matrix element evaluation. This is meant for testing the phase-space sampling.
    Hidden: written to the card only once changed.

[gridpack]
----------

Gridpacks are self-contained event generators created after the survey and training.
See :doc:`gridpacks` for details.

``save_gridpack`` (``false``)
    Create a gridpack at the end of the run.

``include_source`` (``false``)
    Ship the process source code instead of the compiled libraries.

``include_madspace`` (``true``)
    Include the installed ``madspace`` package, so the gridpack runs without a separate
    installation.

``include_madspace_source`` (``false``)
    Include the ``madspace`` source code.

``compress`` (``false``)
    Pack the gridpack into ``gridpack.tar.gz`` instead of leaving it as a directory.

``run_mode`` (``"minimal"``)
    How the run that creates the gridpack ends. Only used if ``save_gridpack`` is set.

    - ``regular``: generate the events of the run, then store the gridpack.
    - ``minimal``: no events. The gridpack is stored right after the MadNIS training or,
      with VEGAS, once the grid of every channel has converged.
    - ``fix_max_weight``: no events. Each channel runs until ``freeze_max_weight_after``
      unweighted events are reached. Its maximum weight is then fixed in the gridpack,
      and its integral is used to share the events between the channels. This can be
      slow for channels with a low unweighting efficiency. The gridpack also gets
      ``bin/event_stream.py`` to stream events in memory from Python.

    See :ref:`gridpack-run-mode`.

``temp_output_dir`` (``""``)
    Directory for the temporary npy files of the channels. ``""`` uses the output
    directory. ``"auto"`` creates a new temporary directory for each run, see
    :py:func:`tempfile.mkdtemp`. Its location can be set with the ``TMPDIR``
    environment variable.

``output_dir`` (``""``)
    Directory for the output files. ``""`` creates a new run folder
    ``Events/<run_name>_NN``. If set, no run folder is created.


[cluster]
---------

Cluster mode trains the subprocesses and generates the events in parallel batch jobs.
It is off by default, and then this section, ``[cluster_train]`` and ``[cluster_deploy]``
are a hidden option group, left out of the card until ``set cluster.enable true``,
``update cluster``, or one of their settings differs from the site default in
``input/default_run_card_mg7.toml``, where their defaults come from. See :doc:`cluster` for how it works.

``enable`` (``false``)
    Run on a batch system. Independent of the ``run_mode`` of the MadGraph configuration.

``mode`` (``"full"``)
    ``full`` runs both passes and combines the events. ``gridpack_only`` only trains, and
    the merged gridpack is the result.

``type`` (``"auto"``)
    The batch system: ``condor``, ``pbs``, ``sge``, ``lsf``, ``ge``, ``slurm``,
    ``htcaas``, ``htcaas2``, the batch system of a plugin (``new_cluster``), or
    ``local_parallel`` to run the jobs on this machine. ``auto`` takes ``cluster_type`` of
    the MadGraph configuration.

``parallel_jobs`` (``-1``)
    Jobs running at once with ``local_parallel``. ``-1`` uses every core. The threads of
    each job are divided accordingly if ``cpu_thread_pool_size`` is ``-1``.

``train_queue`` (``""``), ``deploy_queue`` (``""``)
    The queue (partition) of the jobs. ``""`` takes ``cluster_queue`` of the MadGraph
    configuration.

``train_walltime`` (``""``), ``deploy_walltime`` (``""``)
    The walltime of the jobs, in the format of the batch system. Used for ``slurm`` and
    ``condor``.

``train_extra_args`` (``""``), ``deploy_extra_args`` (``""``)
    Further arguments of the submit command, for example ``"--gres=gpu:1"`` for
    ``slurm``, or ``"-l select=1:ngpus=1"`` for ``pbs``. For ``condor`` they are lines of
    the submit file separated by ``;``, for example ``"request_gpus = 1; request_memory =
    8GB"``.

``subprocesses_per_job`` (``1``)
    The number of subprocesses (merge groups with ``merge_subprocesses``) that are
    trained together in one job. Raise it for a process with many small subprocesses.

``fix_max_weight`` (``true``)
    ``true``: the training jobs store the maximum weight of every channel, as for a
    gridpack with ``run_mode = "fix_max_weight"``, and all generation jobs unweight
    against it. This is the statistically correct choice. ``false``: every generation job
    estimates the maximum weights on its own, which saves time in the training.

``events_per_job`` (``100000``)
    The events of one generation job. The ``[generation]`` ``events`` are split into
    ``ceil(events / events_per_job)`` jobs of almost equal size.

A few more options are hidden: ``nb_retry`` (``1``, how often a failed job is submitted
again), ``retry_wait`` (``300``, seconds a job is given to deliver its output),
``status_update`` (``[600, 30]``, seconds between status checks), ``temp_path`` (the
``cluster_temp_path`` of the configuration) and ``keep_job_files`` (``false``, keep the
gridpacks of the training jobs once merged, and the event files of the generation
jobs once combined; their logs and info.json are always kept).

The devices, threads and batch sizes of the training and of the generation jobs are set
in the ``[cluster_train]`` and ``[cluster_deploy]`` sections, for example
``run.device = ["cuda"]`` in ``[cluster_train]`` and ``run.device = ["cpu"]`` in
``[cluster_deploy]`` to train on GPUs and generate on CPUs. See :doc:`cluster`.


[output]
--------

The files of a run: the event file format, its compression, and the name of the run folder.

``run_name`` (``"run"``)
    Name of the run. Events are written to ``Events/<run_name>_NN``, where ``NN`` counts up
    from 01.

``output_format`` (``"lhe_npy"``)
    Format of the event file. ``compact_npy`` and ``lhe_npy`` write ``events.npy`` together
    with a ``header.lhe`` that carries the run and param card and the cross-section
    information. ``lhe`` writes a compressed LHE file directly. Tools that read LHE files,
    such as the shower programs, need ``lhe``. The launch question switches to it when you
    select one of them.

``use_pigz`` (``"auto"``)
    gzip the LHE output with pigz when it is installed. ``auto`` detects it; ``false`` uses gzip.

``lhe_compression`` (``"auto"``)
    Where the LHE output is gzipped. ``auto`` uses a GPU when madspace was built with nvCOMP,
    otherwise the CPU. ``cpu`` always uses pigz or gzip.

``lhe_gzip_algorithm`` (``5``)
    nvCOMP gzip algorithm for the GPU compression: ``0`` is the fastest and ``5`` gives the
    smallest output.

``time_of_flight`` (``-1.0``)
    Threshold in mm below which the invariant livetime is not written. ``-1`` disables the
    step. The step is a post-processing of the LHE file of MadEvent, so enabling it switches
    ``output_format`` to ``"lhe"``.

[beam]
------

``ebeam1`` (``6500.0``), ``ebeam2`` (``6500.0``)
    Energy of each beam in GeV, beam 1 moving along :math:`+z`. Units are accepted, for
    example ``"7 TeV"``. The collision energy is :math:`2\sqrt{E_1 E_2}`. With different
    energies the events are written, and the :math:`\eta` cuts applied, in this lab frame.
    ``e_cm`` can still be read and set: ``set e_cm 13 TeV`` sets both beams to 6.5 TeV.

``leptonic`` (``false``)
    Treat the beams as leptons: no PDFs are used and the partonic energy equals the
    collision energy. This is set automatically for lepton-collider processes, together
    with 500 GeV beams; the jet cuts stay, as in madevent. Without a PDF,
    :math:`\alpha_s` is the 2-loop running of the :math:`\alpha_s(m_Z)` of the param card,
    as madevent does, and ``pdf1``/``pdf2`` are not used (nor downloaded).

``pdf1`` (``"NNPDF40_lo_as_01180"``), ``pdf2`` (``"NNPDF40_lo_as_01180"``)
    Name of the LHAPDF set of each beam. ``set beam.pdf X`` sets both. MG7 tries to
    download a set that is not installed. The default is the 5-flavor NNPDF 4.0 LO set,
    which has 100 error members for the PDF variations. With different sets on the two
    beams the PDF member variations of ``[systematics]`` are dropped.

``me_frame`` (``[]``)
    External particles whose momenta define the rest frame the matrix element is
    evaluated in, for a matrix element that is not Lorentz invariant (a polarised one).
    ``[]`` applies no boost, ``[1, 2]`` is the partonic centre of mass. A polarised
    process gets the default of MadEvent and aMC@NLO, set in the process definition.
    Hidden: written to the card only once it differs from ``[]``.

[generation]
------------

``events`` (``100000``)
    Number of unweighted events to generate.

``survey_min_iters`` (``3``), ``survey_max_iters`` (``3``), ``survey_target_precision`` (``0.1``)
    The survey adapts the sampling before events are generated. These set the minimum and
    maximum number of iterations and the relative precision the survey aims for.

``max_overweight_truncation`` (``0.001``)
    Fraction of the weight distribution's tail that is ignored when the maximum weight for
    unweighting is estimated. Larger values raise the unweighting efficiency at the price of
    more events with a weight above one.

``freeze_max_weight_after`` (``100000``)
    Number of events after which the maximum weight estimate is no longer updated.

``cpu_batch_size`` (``1000``), ``gpu_batch_size`` (``64000``)
    Number of phase-space points per batch on CPU and GPU devices.

``cut_efficiency_threshold`` (``0.7``), ``max_cut_repetitions`` (``1000``)
    When fewer than this fraction of the points in a batch pass the cuts, the batch is
    sampled again, up to the given number of times.

``interference_helicity`` (``"exact"``)
    How the helicity of an interference matrix element is handled, that is of a process
    whose squared split orders keep a cross term (e.g. ``QCD^2==2``) and so can give
    negative weights. Such a cross term can be negative for some helicities and positive
    for others. ``"exact"`` does as madevent: the helicity of each event is drawn with
    probability :math:`|T_i|/\sum_j |T_j|` and the event gets the weight
    :math:`\mathrm{sign}(T_i)\sum_j |T_j|`, so the events carry their exact helicities
    at the price of a larger variance (more negative weights). ``"summed"`` gives each
    event the helicity sum as its weight (smaller variance) and writes its helicities as
    9 (unknown) in the LHE file. Processes without such a cross term are not affected.

[systematics]
-------------

Scale and PDF variation weights are computed while the events are written. They are stored
as ``<rwgt>`` blocks in LHE files and as ``rwgt_<id>`` columns in the npy files.

``enable`` (``true``)
    Compute the variation weights.

``mur`` (``[0.5, 1.0, 2.0]``), ``muf`` (``[0.5, 1.0, 2.0]``)
    Factors applied to the renormalization and the factorization scale. ``1.0`` is the
    nominal value.

``together`` (``true``)
    If true, all combinations of ``mur`` and ``muf`` are evaluated. If false, one scale is
    varied at a time.

``dynamical_scale`` (``[]``)
    Alternative dynamical scale choices, using the same names as ``dynamical_scale_choice``.
    Each is combined with the factors above.

``pdf`` (``["errorset"]``)
    PDF variations. ``"errorset"`` evaluates all members of the nominal set and
    ``"central"`` its central member. LHAPDF set names or ids evaluate all members of that
    set, and ``"<set>@<member>"`` a single one.

``write_inputs`` (``false``)
    Also store :math:`x_1`, :math:`x_2` and the scales of each event, so the weights can be
    recomputed later. In LHE files they go into an ``<mgrwt>`` block.

The scale and PDF variations of ``systematics.py`` are computed by ``[systematics]``
instead. The ``systematics*`` entries of this section in older cards are ignored.

[vegas]
-------

VEGAS grids adapt the phase-space sampling during the survey.

``enable`` (``true``)
    Use VEGAS grids. MadNIS flows start from the VEGAS grid when both are active.

``bins`` (``64``)
    Number of grid bins per dimension.

``damping`` (``0.4``)
    Damping of the grid updates. Raise it if the grid oscillates instead of settling.

``optimization_patience`` (``5``), ``optimization_threshold`` (``0.9``)
    The grid optimization stops when this many iterations in a row fail to improve the
    result by the given relative margin.

``start_batch_size`` (``1000``), ``max_batch_size`` (``32000``)
    Batch size of the first survey iteration and the largest batch size used afterwards.
    Larger batches give a better grid at a proportionally higher cost.

[phasespace]
------------
Only ``merge_subprocesses`` and ``sde_strategy`` are written to the card by default. The other
entries are hidden: they keep their defaults, and are written to the card only once
changed (``set`` takes them as any other).

``mode`` (``"multichannel"``)
    Strategy for the phase space. ``multichannel`` uses one channel per diagram. ``flat``
    uses a single channel described by ``flat_mode``. ``both`` surveys the multichannel
    phase space and then replaces the least important channels with one flat channel, see
    ``combine_channel_threshold``. ``auto`` behaves like ``both`` if MadNIS is enabled and
    like ``multichannel`` otherwise. Decays always use ``multichannel``.

``merge_subprocesses`` (``false``)
    Merge subprocesses with the same diagram topologies into shared channels.

``sde_strategy`` (``"diagrams"``)
    How the multichannel weights are computed. ``diagrams`` uses the squared single
    diagrams, ``denominators`` the product of the propagator denominators.

``t_channel`` (``"propagator"``)
    Parametrization of the t-channel part of the multichannel phase space. ``propagator``
    follows the diagram as a chain of :math:`2 \to 2` steps. ``rambo`` applies RAMBO to the
    t-channel particles and ``chili`` uses collider coordinates.

``flat_mode`` (``"rambo"``)
    The same choice for the flat channel.

``combine_channel_threshold`` (``0.01``)
    In ``both`` mode, the channels that together contribute this fraction of the cross
    section are replaced by the flat channel.

``drop_qcd_s_channel`` (``20``)
    If a subprocess has more channels than this, channels without a QCD s-channel resonance
    are dropped. ``-1`` keeps all channels. This helps with processes that have too many
    channels.

``ew_channel_reduction`` (``20``)
    If a subprocess still has more channels than this, channels that differ only in the
    bosons of some of their off-shell propagators are merged into one, which takes over
    their channel weights, and redundant ones are dropped (see
    ``ew_channel_reduction_mode``). Channels that map a flavor or a (flavor, resonance) pair
    no kept channel does are always kept. ``-1`` keeps all channels. This is the
    counterpart of ``drop_qcd_s_channel`` for processes with many electroweak diagrams.

``ew_channel_reduction_mode`` (``"partial"``)
    How far ``ew_channel_reduction`` goes:

    - ``"t_channel"``: merge channels that differ only in the bosons of their t-channel
      propagators (a photon, Z or W); every s-channel structure keeps its channel.
    - ``"partial"``: merge as ``"t_channel"``, and keep only the channels with at most one
      non-resonant s-channel propagator more than the fewest.
    - ``"full"``: also merge channels that differ in their off-shell s-channel bosons, and
      keep only the channels with the fewest non-resonant s-channel propagators. This
      gives the fewest channels, but trains to lower unweighting efficiencies.

``channel_reduction_with_vegas`` (``false``)
    ``ew_channel_reduction`` is meant for MadNIS, which trains poorly with many channels.
    By default it is skipped when the run is certain to use VEGAS: ``madnis.enable =
    false``, or ``"auto"`` for a 2 -> 2 process. ``true`` applies it then as well.

``invariant_power`` (``0.7``)
    Exponent :math:`p` of the :math:`1/s^p` sampling of time-like invariants.

``adaptive_symmetry_sampling`` (``true``)
    Learn the probabilities of the permutations of identical particles instead of using
    equal ones.

[merging]
---------

MLM and CKKW-L merging. The section is shown in the card once the sample is merged: an MLM
sample (``dynamical_scale_choice = "mlm"`` with ``xqcut > 0``) or a CKKW-L one (``ktdurham``
or ``ptlund`` set). Otherwise it is hidden.

``jet_radius`` (``1.0``)
    Jet radius of the MLM clustering measure, used with ``dynamical_scale_choice = "mlm"``.
    Hidden: written to the card only once changed.

``xqcut`` (``0.0``)
    MLM merging cut on the :math:`k_T` of the matrix-element jets, in GeV, with
    ``dynamical_scale_choice = "mlm"``: an emission below it is left to the parton
    shower, so the event is dropped. ``0`` disables it.

``auto_ptj_mjj`` (``true``)
    With a merging cut in place, raise the jet :math:`p_T` and pair-mass cuts to ``xqcut``,
    which the jets are bounded by anyway, and drop the jet :math:`\Delta R` cuts that
    ``xqcut`` supersedes. As in MadEvent's ``auto_ptj_mjj``.

``ktdurham`` (``0.0``)
    CKKW-L merging with Pythia8: the smallest Durham :math:`k_T` an event may have, taken
    over the :math:`p_T` of each merging parton (hadron colliders only) and the
    :math:`k_T = \min(p_{T,i}, p_{T,j})\,\Delta R_{ij}/D` of each pair. ``0`` disables it.
    Pythia8's ``Merging:TMS`` defaults to this value, and ``Merging:Process`` has to be set
    in the Pythia8 card. Only one of ``ktdurham`` and ``ptlund`` can be set, and neither
    together with ``xqcut``. On hadron colliders the phase-space sampling starts the
    :math:`p_T` of the merging partons at this value.

``ptlund`` (``0.0``)
    CKKW-L merging with Pythia8: the smallest Pythia8 evolution :math:`p_T` over all ISR and
    FSR splittings that can produce a merging parton. ``0`` disables it. An ``[amplicol]``
    subprocess usually groups flavours that split differently (quarks and gluons), which
    this cut cannot tell apart: use ``ktdurham`` there.

``dparameter`` (``0.4``)
    :math:`D` of the hadron-collider ``ktdurham`` measure, Pythia8's ``Merging:Dparameter``.

``merging_pdgs`` (``"auto"``)
    PDG ids of the particles the ``ktdurham`` or ``ptlund`` cut applies to. ``auto`` takes,
    as in MadEvent, all colored particles of the model (including the top and any colored
    BSM particles); only those in the final state matter. As in MadEvent, decay-chain products are never
    merging partons.

[custom_functions]
------------------

User-written functions, created with ``bin/init_custom_func``. Each one is a shared
library (``.so``) or a Python file (``.py``). Paths are relative to the process directory.
An empty string disables the function. See :doc:`custom_functions` for details.

``cut`` (``""``)
    Custom cut, for example ``"lib/custom_cut.so"`` or ``"custom_cut/custom_cut.py"``. It
    is applied on top of the ``[cuts]``.

``scale`` (``""``)
    Custom renormalization and factorization scales. They replace
    ``dynamical_scale_choice``.

``pdf`` (``""``)
    Custom parton density. It replaces the PDF set, but :math:`\alpha_s` still comes from
    ``[beam] pdf``. The systematics are disabled.

``bias`` (``""``)
    Custom event bias. Events are generated with the weight times the bias and written with
    the weight divided by the bias.

``<function>_parameters.<name>``
    A parameter of a custom function, for example ``cut_parameters.pt_min = 30.0``. It replaces
    the default value declared by the function. See :ref:`custom-function-parameters`.

.. _run-card-surrogate:

[surrogate]
-----------

A fast surrogate of the matrix element, for example a neural network. The integrand and
its optimization use the surrogate. The events are then reweighted or unweighted a second
time with the full matrix element, which is only evaluated for the events that pass the
first unweighting. The cross section and the events are those of the full matrix element.
The surrogate does not return channel weights, so it needs ``sde_strategy =
"denominators"``. It is not supported for outputs with ``[amplicol]`` subprocesses.

``function`` (``""``)
    The surrogate. An empty string disables it.

    - A ``.so`` library implementing the UMAMI interface. It gets the momenta,
      :math:`\alpha_s` and the flavor index of the madmatrix libraries as inputs, and
      returns the matrix element. It is loaded once per subprocess: ``{subprocess}`` and
      ``{device}`` in the path are replaced by the subprocess name (e.g. ``P1_gg_ttx``)
      and the device, like in ``lib/libmadmatrix_{subprocess}_{device}.so``. It has to
      be built for the device of the run. If it implements ``umami_set_parameter``, it
      gets the run-time parameters of the madmatrix libraries, such as ``bwcutoff``.
    - A Python file defining ``custom_surrogate(momenta, pdg_ids, alpha_s)``, which returns
      :math:`|M|^2` for a batch of events. ``momenta`` has the shape ``(batch, n, 4)``,
      ``pdg_ids`` the shape ``(batch, n)`` and identifies subprocess and flavor, and
      ``alpha_s`` the shape ``(batch,)``. ``FRAMEWORK = "torch"`` passes torch tensors,
      which also run on GPUs, otherwise numpy arrays are passed. See
      :ref:`custom-function-surrogate`.

``mode`` (``"unweight"``)
    ``"unweight"`` unweights the events a second time with respect to the ratio of the full
    matrix element to the surrogate. The events then have unit weights apart from
    overweights. ``"reweight"`` keeps the ratio in the event weights.

``first_pass_max_overweight_truncation`` (``0.01``), ``first_pass_freeze_max_weight_after`` (``10000``)
    ``max_overweight_truncation`` and ``freeze_max_weight_after`` of the first
    unweighting, with the surrogate, in unweight mode. The ones of ``[generation]`` set
    the second unweighting with the full matrix element, which gives the final events,
    and the unweighting of the subprocesses without a surrogate. In reweight mode, the
    events are unweighted once, with the surrogate, as set by ``[generation]``.

``param_card`` (``""``)
    The param card a ``.so`` surrogate is initialized with. An empty string uses
    ``Cards/param_card.dat``. A madmatrix library with a different param card, for example,
    is a surrogate for testing.

[amplicol]
----------

The ``[amplicol]`` subprocesses (``generate p p > j j j [amplicol]``) take their matrix
elements from AmpliCol instead of Feynman diagrams. They are integrated at leading colour,
one colour ordering at a time, over colour-ordered phase-space mappings. A second pass
then re- or unweights the events to full colour with the ratio of the full-colour to the
leading-colour matrix element, summed over the colour orderings, at the helicity of the
event. These subprocesses run on CPUs only, and MG7 does not turn MLM merging on by itself
for an output that has them.

``color_accuracy`` (``"full"``)
    ``"full"`` corrects the events to full colour in the second pass. ``"lc"`` keeps the
    leading-colour events and skips the second pass.

``mode`` (``"unweight"``)
    ``"reweight"`` keeps the full- to leading-colour ratio in the event weights.
    ``"unweight"`` unweights the events a second time with respect to that ratio. The
    events then have unit weights apart from overweights.

``first_pass_max_overweight_truncation`` (``0.001``), ``first_pass_freeze_max_weight_after`` (``10000``)
    ``max_overweight_truncation`` and ``freeze_max_weight_after`` of the first,
    leading-colour unweighting, in unweight mode. The ones of ``[generation]`` set the
    second unweighting, to full colour, which gives the final events. In reweight mode, the
    events are unweighted once, at leading colour, as set by ``[generation]``.

.. _run-card-multiparticles:

[multiparticles]
----------------

Named groups of particles, given as lists of PDG ids. The names are used in the cuts and
histograms. The defaults are ``jet``, ``bottom``, ``lepton``, ``missing`` and ``photon``:

.. code-block:: toml

    [multiparticles]
    jet = [1, 2, 3, 4, -1, -2, -3, -4, 21]
    bottom = [-5, 5]
    lepton = [11, 13, 15, -11, -13, -15]
    missing = [12, 14, 16, -12, -14, -16]
    photon = [22]

You can add your own groups, for example ``top = [6, -6]``.

.. _run-card-cuts:

[cuts]
------

Each entry has the form ``<selection>-<observable>`` with the bounds ``min`` and ``max``:

.. code-block:: toml

    [cuts]
    jet-pt.min = 20.0
    jet-eta_abs.max = 5.0
    jet-delta_r.min = 0.4
    lepton-pt.min = 10.0
    lepton-eta_abs.max = 2.5
    lepton-delta_r.min = 0.4
    jet-lepton-delta_r.min = 0.4
    sqrt_s.min = 0.0

The selection is one or more groups from ``[multiparticles]``. The observable is one of

* ``pt``, ``mass``, ``e``, ``px``, ``py``, ``pz``, ``p_mag``, ``phi``, ``theta``, ``y``,
  ``y_abs``, ``eta`` and ``eta_abs`` for single particles,
* ``delta_r``, ``delta_eta`` and ``delta_phi`` for pairs of particles,
* ``pair_mass`` and ``sfos_pair_mass`` for the invariant mass of pairs,
* ``sqrt_s`` for the partonic center-of-mass energy. It takes no selection.

A single group applies the observable to each of its particles. ``jet-pt.min = 20``
requires every jet to pass. Two groups, as in ``jet-lepton-delta_r``, form all pairs with
one particle from each group. The pairs of a single group, as in ``jet-delta_r``, are all
distinct pairs within it.

``pair_mass`` is the invariant mass of every pair the groups can form, so
``lepton-pair_mass`` is the mass of every lepton pair. ``sfos_pair_mass`` only keeps
same-flavor opposite-sign pairs such as :math:`e^+ e^-`. It corresponds to MadEvent's
``mmll``.

Add ``-sum-`` before the observable to add up the momenta first. ``lepton-lepton-sum-mass``
is the mass of two leptons taken together, and ``lepton-jet-top-sum-mass`` the mass of a
lepton, a jet and a top. Add ``-sum`` after the observable to add up the values instead, so
``jet-pt-sum`` is the scalar sum of the jet transverse momenta.

An index selects particles by rank. ``jet_1-pt`` is the hardest jet, ``jet_2-pt`` the second
hardest, and so on. The key ``order_by = "pt"`` sets the observable that defines the rank.

By default a bound must hold for every selected particle. Setting ``mode = "any"`` on an
entry relaxes this to at least one.

The selections are flexible because you define the groups. Some examples:

.. code-block:: toml

    [multiparticles]
    alllepton = [11, 13, 15, -11, -13, -15, 12, 14, 16, -12, -14, -16]
    top = [6, -6]

    [cuts]
    jet-pt-sum.min = 200.0             # H_T of the jets
    jet_1-jet_2-pt-sum.min = 150.0     # H_T of the two hardest jets
    jet_2-pt.min = 40.0                # second hardest jet
    alllepton-sum-pt.min = 30.0        # pT of all leptons and neutrinos together
    top-pair_mass.min = 250.0          # mass of every pair of tops
    jet-eta_abs.min = 1.0              # bounds work in both directions

Cuts also restrict the sampling. ``pair_mass`` and ``sqrt_s`` minima bound the
integration region, as do ``pt`` and ``delta_r`` minima through the masses they imply.
``eta_abs`` maxima bound the rapidity of the partonic system and, together with the ``pt``
minima, the scattering angle. This only improves efficiency. The result is the same.

Use ``set no_parton_cut`` to remove all cuts. ``set <selection>-<observable>.min <value>``
changes a single bound. Processes without initial-state particles, such as decays, start
without cuts.

.. _run-card-histograms:

``max_jet_flavor`` (``4``)
    Heaviest quark flavour the jet cuts and the MLM back-clustering treat as a jet: the
    ``jet`` multiparticle follows it (``5`` in a 5-flavour model, where the b is a jet).

``bw_cutoff`` (``15``)
    Breit-Wigner propagators are sampled within this many widths of the mass. The same
    window applies to propagators excluded with ``$``.

``cut_decays`` (``false``)
    Apply the cuts to the decay products of on-shell particles. As in MadEvent, they are
    left uncut by default.

[histograms]
------------
Output settings of the plots first, then the observables:

``weighted_histograms`` (``false``)
    Fill the :ref:`histograms <run-card-histograms>` with weighted events during the
    integration. This costs one observable evaluation per phase-space point.

``postprocessing_histograms`` (``true``)
    Fill the histograms with the final events, including their scale and PDF weights.
    Plots and the HwU file are made from these.

``make_plots`` (``true``)
    Plot the histograms into ``Events/<run>/plots``. This needs ``matplotlib``. Without it
    the numbers are still written to ``info.json``.

``write_hwu`` (``false``)
    Also write the histograms as ``MADatLO.HwU``. This is the format MG5_aMC writes as
    ``MADatNLO.HwU`` for NLO runs, with the same conventions: the cross section per bin
    (not divided by the bin width, unlike the plots), and the same scale-variation
    columns. ``madgraph/various/histograms.py`` reads both files together.

Histograms of observables, filled during the run and written to ``Events/<run>/info.json``.
They use the same observable names as the cuts. Each entry needs a range and a number of
bins:

.. code-block:: toml

    [histograms]
    jet_1-pt.min = 0.0
    jet_1-pt.max = 500.0
    jet_1-pt.bin_count = 50

The entry ``weight`` is not an observable of the momenta. It is the distribution of the
event weight in units of the unit weight :math:`\sigma_\mathrm{abs}`, the integral of
the absolute value of the weights. A fully unweighted sample is a spike at 1, and its
negative weights (from an interference) a spike at -1.

``output`` writes a default set of histograms for the final state of your process. ``set
histograms OFF`` at the launch question removes them and ``set histograms default``
restores them. The settings above, ``weighted_histograms``, ``postprocessing_histograms``,
``make_plots`` and ``write_hwu``, control what is done with them.

[madnis]
--------

`MadNIS <https://arxiv.org/abs/2212.06172>`_ trains neural networks to improve the phase-space
sampling. It needs PyTorch. ``enable`` can be ``true``, ``false`` or ``"auto"``. In auto
mode, MG7 surveys first and then decides for each subprocess. Processes with two outgoing
particles never use MadNIS. Three outgoing particles use it if the survey is noisy, if more
than a million events are requested or if a gridpack is built. Four or more outgoing
particles always use it. Auto mode also sizes the networks and picks the learning rate.
The parameters marked ``"auto"`` below are chosen this way, and setting one yourself
overrides this.

Start tuning with ``train_batches`` and ``batch_size_per_channel``. Next try ``lr`` and the
loss. Change the network shapes last.

Training
^^^^^^^^

``enable`` (``"auto"``)
    Train MadNIS networks.

``train_batches`` (``1000``)
    Number of training batches.

``batch_size_per_channel`` (``"auto"``)
    Minimum number of training events per active channel.

``batch_size_offset`` (``512``)
    Number of events added to the training batch size of every channel.

``log_interval`` (``100``)
    Batches between two entries in the training log.

``uniform_channel_ratio`` (``0.5``)
    Fraction of the training batch spread uniformly over the channels. The rest is
    distributed according to the channel integrals.

``integration_history_length`` (``100``)
    Number of batches for which the mean and variance of each channel are kept.

``drop_zero_integrands`` (``true``)
    Ignore points with a vanishing integrand in the training.

``fixed_cwnet_fraction`` (``"auto"``)
    Fraction of the training during which the channel weight network is frozen. It only
    starts training afterwards.

``softclip_threshold`` (``30.0``)
    Soft clipping threshold of the channel weights. ``0`` disables it.

``compressed_channel_weight_count`` (``50``)
    Number of channel weights kept per event in the multichannel weight loss.

Optimizer
^^^^^^^^^

``lr`` (``"auto"``)
    Learning rate of the Adam optimizer.

``lr_scheduler`` (``"cosine"``)
    ``cosine`` decays the learning rate to zero over the training. ``none`` keeps it
    constant.

``adam_beta1`` (``0.9``), ``adam_beta2`` (``0.999``), ``adam_eps`` (``1e-8``), ``adam_weight_decay`` (``1e-4``)
    Adam parameters.

``grad_clip_threshold`` (``0.03``)
    Maximum gradient norm. ``0`` disables clipping.

Replay buffer
^^^^^^^^^^^^^

Training can reuse earlier samples. This saves matrix element evaluations.

``buffer_capacity`` (``60000``)
    Number of samples kept per channel. ``0`` disables the buffer.

``minimum_buffer_size`` (``10000``)
    Number of buffered samples needed before training on the buffer starts.

``buffered_steps_fraction`` (``"auto"``)
    Fraction of the training steps done on buffered samples. ``auto`` is 0 for
    ``[amplicol]`` subprocesses and 0.8 otherwise.

``buffer_skip_batches`` (``1000``)
    Number of initial batches that are not stored.

``buffer_unweighting_quantile`` (``0.95``)
    Quantile of the weights used as the maximum when buffered samples are unweighted.

Sample generation
^^^^^^^^^^^^^^^^^

``generator_target_size_factor`` (``32``)
    Number of training batches the sample generator keeps buffered for each channel.

``gpu_generator_batch_granularity`` (``1000``)
    Batch sizes for sample generation on GPUs are rounded to a multiple of this.

Channel dropping
^^^^^^^^^^^^^^^^

``channel_dropping_threshold`` (``0.01``), ``channel_dropping_interval`` (``100``)
    Every ``channel_dropping_interval`` batches, channels that contribute less than the
    threshold fraction of the integral are dropped.

``resonance_guard`` (``true``)
    The channel dropping never drops the last channel that maps a resonance (a propagator
    that can go on shell, with its decay products) for a flavor. The last channel of a
    flavor is kept in any case. Without the guard, a resonance with a small share of the cross section, such as
    H -> W W* in a VBS process, can lose its only channel and end up undersampled.

Networks
^^^^^^^^

Three networks are trained. The flow maps the phase-space sampling, the discrete network
learns the probabilities of discrete choices such as permutations and flavors, and the
channel weight network learns the multichannel weights. Each is built from fully connected
subnetworks and has settings for the number of hidden units and layers and the activation
function. The activations are ``relu``, ``leaky_relu``, ``elu``, ``gelu``, ``sigmoid`` and
``softplus``.

``flow_hidden_dim`` (``"auto"``), ``flow_layers`` (``"auto"``), ``flow_activation`` (``"leaky_relu"``)
    Size and activation of the flow subnetworks.

``flow_spline_bins`` (``10``)
    Number of spline bins per flow transformation.

``flow_invert_spline`` (``false``)
    Apply the splines in the inverse direction.

``discrete_hidden_dim`` (``"auto"``), ``discrete_layers`` (``3``), ``discrete_activation`` (``"leaky_relu"``)
    Size and activation of the discrete network.

``cwnet_hidden_dim`` (``"auto"``), ``cwnet_layers`` (``3``), ``cwnet_activation`` (``"leaky_relu"``)
    Size and activation of the channel weight network.

``nn_float_precision`` (``"float32"``)
    Floating-point precision of the weights and the evaluation of all networks,
    ``float32`` or ``float64``. The rest of the computation, including the network
    outputs, softmaxes and Jacobians, stays in double precision.
