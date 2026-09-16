Fusing RNG, flow, mapping and matrix element into one graph
================================================================

The previous two integrators call the mapping, and then the cross section, as two separate
Python-level steps. MadSpace also carries its own normalizing flow, :py:class:`Flow
<madspace.Flow>`, and its own compute graph builder, :py:class:`FunctionBuilder
<madspace.FunctionBuilder>`. This example uses the builder to fuse everything, from drawing
the random numbers to the final cross-section weight, into a single compiled graph, and trains
the flow's weights with a plain PyTorch optimizer.

Setting up the physics
--------------------------

The mapping, cuts, matrix element, PDF and scale are exactly the same building blocks as the
:doc:`earlier integrator <integrator>`, generated with ``output mg7 PROC_ggttg`` beforehand:

.. code-block:: python

    import glob
    import json
    import os

    import numpy as np
    import torch
    import madspace as ms
    from madspace.torch import FunctionModule

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    E_CM = 13000.0
    masses = [0.0, 0.0, 173.0, 173.0, 0.0]  # g g -> t t~ g
    pids = [21, 21, 6, -6, 21]

    O = ms.Observable
    cuts = ms.Cuts([
        ms.CutItem(O(pids, O.obs_pt, [O.jet_pids]), min=20.0),
        ms.CutItem(O(pids, O.obs_eta_abs, [O.jet_pids]), max=5.0),
        ms.CutItem(O(pids, O.obs_delta_r, [O.jet_pids]), min=0.4),
    ])
    mapping = ms.PhaseSpaceMapping(masses, E_CM, mode="propagator", cuts=cuts)

    proc_dir = "PROC_ggttg"
    meta = json.load(open(os.path.join(proc_dir, "SubProcesses", "subprocesses.json")))[0]
    me_path = glob.glob(os.path.join(proc_dir, meta["me_path"].format(device="*")))[0]
    param_card = os.path.join(proc_dir, "Cards", "param_card.dat")

    ctx = ms.default_context()
    api = ctx.load_matrix_element(me_path, param_card)
    MEI = ms.MatrixElement.MatrixElementInput
    MEO = ms.MatrixElement.MatrixElementOutput
    matrix_element = ms.MatrixElement(
        api, [MEI.momenta_in, MEI.alpha_s_in], [MEO.matrix_element_out], True
    )

    PDF_SET = "NNPDF40_lo_as_01180"
    import lhapdf
    pdf_dir = os.path.join(lhapdf.paths()[0], PDF_SET)
    pdf_grid = ms.PdfGrid(os.path.join(pdf_dir, f"{PDF_SET}_0000.dat"))
    pdf_grid.initialize_globals(ctx)
    alphas_grid = ms.AlphaSGrid(os.path.join(pdf_dir, f"{PDF_SET}.info"))
    alphas_grid.initialize_globals(ctx)
    alpha_s = ms.RunningCoupling(alphas_grid)

    scale = ms.EnergyScale(
        particle_count=5, type=ms.EnergyScale.DynamicalScaleType.half_transverse_mass
    )
    cross_section = ms.DifferentialCrossSection(
        matrix_element=matrix_element,
        cm_energy=E_CM,
        running_coupling=alpha_s,
        energy_scale=scale,
        pid_options=[[21, 21]],
        pdf1=pdf_grid, pdf2=pdf_grid,
        input_momentum_fraction=True,
    )

Declaring the flow
----------------------

A :py:class:`Flow <madspace.Flow>` reshapes ``mapping.random_dim()`` unit-hypercube
coordinates into a learned sampling distribution. Its weights live in the context as trainable
globals, created by ``initialize_globals``:

.. code-block:: python

    flow = ms.Flow(
        input_dim=mapping.random_dim(), prefix="flow", bin_count=10,
        subnet_hidden_dim=32, subnet_layers=2,
    )
    flow.initialize_globals(ctx, seed=0)

Building the fused graph
----------------------------

:py:meth:`FunctionBuilder.random <madspace.FunctionBuilder.random>` draws the random numbers
inside the graph itself, given a symbolic batch size. Every other piece is embedded the same
way each building block embeds any other: ``build_forward`` for a
:py:class:`Mapping <madspace.Mapping>`, ``build_function`` for a
:py:class:`FunctionGenerator <madspace.FunctionGenerator>`. The three Jacobians, from the flow,
the mapping and the cross section itself, multiply into one weight:

.. code-block:: python

    fb = ms.FunctionBuilder(
        ms.NamedTypes([("batch_size", ms.Type([ms.batch_size]))]),
        ms.NamedTypes([("weight", ms.batch_float)]),
    )
    n = fb.input(0)
    r = fb.random(n, ms.Value(mapping.random_dim()))
    flow_out = flow.build_forward(fb, [r], [])
    mapping_out = mapping.build_forward(fb, [flow_out["data"]], [])
    pdf_id = fb.full([ms.Value(0), n])  # only one entry in pid_options
    xsec_out = cross_section.build_function(
        fb, [mapping_out["momenta"], mapping_out["x1"], mapping_out["x2"], pdf_id]
    )
    weight = fb.mul(fb.mul(flow_out["det"], mapping_out["det"]), xsec_out["matrix_element"])
    fb.output(0, weight)
    func = fb.function()

The compiled function takes a single argument, how many events to generate, and returns the
fully differential weight for each of them.

Training
-----------

:py:class:`madspace.torch.FunctionModule` wraps the graph as an ``nn.Module``, exposing every
trainable global as a parameter. Because a normalizing flow always integrates to one, the
average weight over its own samples already equals the cross section however the flow is
trained; minimizing the average of the squared weight reduces its variance without biasing
that estimate, the same principle behind Kleiss and Pittau's multichannel weight optimization:

.. code-block:: python

    module = FunctionModule(func, ctx)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    batch_size = torch.tensor([4096], dtype=torch.int32)

    for step in range(300):
        optimizer.zero_grad()
        weight = module(batch_size)
        loss = weight.square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()

The matrix element has long tails, so a few batches produce a very large gradient; clipping
keeps those from derailing the training.

Checking the result
-----------------------

The trained weights already live in ``ctx``, so evaluating the same graph without going
through PyTorch, with a plain :py:class:`FunctionRuntime <madspace.FunctionRuntime>` and
NumPy, reports the trained integrator's performance:

.. code-block:: python

    def integrate(n_events):
        runtime = ms.FunctionRuntime(func, ctx)
        w = runtime(np.array([n_events], dtype=np.int32))
        return w.mean(), w.std() / np.sqrt(n_events)

    sigma, error = integrate(50000)
    print(f"sigma = {sigma:.2f} +- {error:.2f} pb  (rel. error {error / sigma * 100:.2f}%)")

::

    sigma = 238.64 +- 5.47 pb  (rel. error 2.29%)

This is a shorter, less tuned training loop than either MadGraph7's own event generator or the
external MadNIS package, both of which use more sophisticated losses and training schedules.
Even so, the relative error is roughly half of what an untrained flow gives for the same
number of samples.
