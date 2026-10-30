Cluster mode
============

The training of the subprocesses of a process is independent, but a normal run does it one
subprocess after the other. In cluster mode, the subprocesses are trained in parallel on a
batch system, and the events are generated in parallel as well. It is built on
:doc:`gridpacks`, and steered by the ``[cluster]`` section of the :doc:`run card
<run_card>`. It is off by default.

.. code-block:: toml

    [cluster]
    enable = true
    type = "slurm"
    train_device = ["cuda"]
    train_extra_args = "--gres=gpu:1"
    deploy_device = ["cpu"]
    events_per_job = 100000

and start the run as usual, with ``launch`` or ``bin/generate_events``. The command waits
until everything is done.

How it works
------------

1. **Training.** One batch job per subprocess (or merge group, see
   ``merge_subprocesses``) runs the survey and the training, here with the device of
   ``train_device``, and saves the data of a gridpack. Use ``subprocesses_per_job`` to
   train several small subprocesses in one job.
2. **Merging.** The gridpacks of all jobs are merged into one with all channels. The
   libraries of the matrix elements are compiled on the machine that you launch from, for
   the devices of both passes.
3. **Generation.** The merged gridpack is run by one job per ``events_per_job`` events,
   each with another seed, on the devices of ``deploy_device``.
4. **Combining.** The events of the jobs are concatenated and written as an ordinary run
   to ``Events/<run_name>_NN``, in the ``output_format`` of the run card. The cross
   section is the mean of the cross sections of the jobs weighted with their number of
   events, which is what the concatenated events measure, and its error follows from the
   independence of the jobs. The legacy post-processing of the ``[postprocessing]``
   section is run on the result.

With ``mode = "gridpack_only"``, the run stops after the merging and the merged
gridpack is written to ``Events/<run_name>_NN/gridpack``. It is a normal gridpack, see
:doc:`gridpacks`.

Everything of the jobs is kept in ``Events/<run_name>_NN/cluster``, including their logs.

Stopping and continuing
-----------------------

The state of the run is stored in ``cluster/state.json``. If a run is interrupted or a
job fails, launch it again without changing the cards: it continues where it stopped, and
only repeats the jobs that did not finish. After a change of the ``run_card.toml`` or
``param_card.dat`` a new run is started.

The maximum weights
-------------------

With ``fix_max_weight = true`` (default), the training jobs determine the maximum weight
of each channel, and every generation job unweights against it. The events of all jobs are
then unweighted in the same way. With ``false``, every generation job estimates the maximum
weights from its own events. That makes the training shorter, but each job has to
generate enough events to estimate them.

Batch systems
-------------

``type`` is any of the batch systems of the MadGraph5_aMC@NLO cluster mode, or
``multicore`` to run the jobs on the local machine. The latter is useful for testing, and
divides the cores between the jobs. ``type = "auto"`` takes the ``cluster_type`` of the
MadGraph configuration, which also provides the queue, if that is not set in the card, as
well as the retry settings.

The jobs run the code from the MadGraph directory and read and write the process directory,
so both have to be available on the nodes (a shared file system).
The ``cpu_mode`` is resolved where you launch from, and the jobs use the libraries built
for it: set it explicitly for a cluster with different CPUs.

Limitations
-----------

- Scans over the run card or the param card, and shower, MadSpin, Delphes and
  reweighting after the run are not supported. Run them on the events of the cluster run.
- The histograms of the ``[histograms]`` section are not filled, since a gridpack does
  not fill them.

Merging gridpacks by hand
-------------------------

The ``subprocess_selection`` option of the ``[phasespace]`` section lists the subprocesses
(merge groups) that a run integrates. They keep their number, so the gridpacks of runs over
different selections can be merged into one:

.. code-block:: bash

    ./bin/merge_gridpacks -o merged gridpack_a gridpack_b

The gridpacks (directories or ``.tar.gz``) have to come from the same process, cards and
version of ``madspace``, and must not overlap. ``bin/combine_cluster_runs -o Events/combined
Events/job_a Events/job_b`` combines the events of runs of the same gridpack in the way
described above.
