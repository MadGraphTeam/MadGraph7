Calling a MadGraph-generated matrix element
==============================================

MadSpace evaluates matrix elements through the UMAMI interface, a set of C functions every
MadGraph7 output exports. This example loads the matrix element for ``g g > t t~ g``, the
process from the first example, and evaluates it on sampled momenta.

Generating and compiling the process
---------------------------------------

At the MadGraph7 prompt::

    generate g g > t t~ g
    output mg7 PROC_ggttg

Then compile the matrix elements it wrote::

    cd PROC_ggttg/SubProcesses
    make

This produces one shared library per subprocess under ``PROC_ggttg/lib/``, together with
``PROC_ggttg/SubProcesses/subprocesses.json``, which describes every subprocess and points
at its library.

Loading the matrix element
-----------------------------

``subprocesses.json`` gives the library path as a template, with ``{device}`` standing in
for the compute backend the library was built for. A glob picks up whichever backend
``make`` chose:

.. code-block:: python

    import glob
    import json
    import os

    import numpy as np
    import madspace as ms

    proc_dir = "PROC_ggttg"
    meta = json.load(open(os.path.join(proc_dir, "SubProcesses", "subprocesses.json")))[0]
    me_glob = os.path.join(proc_dir, meta["me_path"].format(device="*"))
    me_path = glob.glob(me_glob)[0]
    param_card = os.path.join(proc_dir, "Cards", "param_card.dat")

    ctx = ms.default_context()
    api = ctx.load_matrix_element(me_path, param_card)
    print(f"particle count: {api.particle_count()}, diagrams: {api.diagram_count()}")

:py:meth:`Context.load_matrix_element <madspace.Context.load_matrix_element>` returns a
:py:class:`MatrixElementApi <madspace.MatrixElementApi>`, which reads the parameters from
``param_card.dat`` when it loads the library::

    particle count: 5, diagrams: 16

Building the compute-graph function
--------------------------------------

:py:class:`MatrixElement <madspace.MatrixElement>` wraps the loaded library as a
:py:class:`FunctionGenerator <madspace.FunctionGenerator>`. ``inputs`` and ``outputs`` pick
which quantities to pass in and read back. The last argument lets the matrix element draw its
own random helicity, color and diagram choices, so none of them need to be supplied here:

.. code-block:: python

    MEI = ms.MatrixElement.MatrixElementInput
    MEO = ms.MatrixElement.MatrixElementOutput
    matrix_element = ms.MatrixElement(
        api, [MEI.momenta_in, MEI.alpha_s_in], [MEO.matrix_element_out], True
    )

Evaluating it
----------------

The matrix element is a plain function of momenta and :math:`\alpha_s`, so it takes the
output of a :py:class:`PhaseSpaceMapping <madspace.PhaseSpaceMapping>` directly:

.. code-block:: python

    mapping = ms.PhaseSpaceMapping([0.0, 0.0, 173.0, 173.0, 0.0], 13000.0, mode="rambo")
    rng = np.random.default_rng(0)
    r = rng.random((5, mapping.random_dim()))
    momenta, x1, x2, det = mapping.map_forward([r])
    alpha_s = np.full(5, 0.13)

    print(matrix_element(momenta, alpha_s))

::

    [1.18355914e-04 3.54037768e-04 3.24861340e-05 2.72819425e-03 1.46426371e-05]
