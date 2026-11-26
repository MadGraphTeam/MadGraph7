"""Flavour switch in a weighting step: the integrand's parton_densities output
(both legs, PDG ids -6..6, with the beam of each leg after mirroring) and the
write-back of the switched flavour, color, helicity and diagram."""

import os

import numpy as np
import pytest
from test_event_stream import make_config, make_context

import madspace as ms

SETS = ("NNPDF40MC_lo_as_01180", "NNPDF40_lo_as_01180")
CM_ENERGY = 13000.0
PIDS = [2, 21, 2, 21, 21]


def _grid_files():
    paths = []
    if os.environ.get("LHAPDF_DATA_PATH"):
        paths += os.environ["LHAPDF_DATA_PATH"].split(os.pathsep)
    try:
        import lhapdf

        lhapdf.setVerbosity(0)
        paths += list(lhapdf.paths())
    except ImportError:
        pass
    files = []
    for name in SETS:
        found = [
            os.path.join(path, name, f"{name}_0000.dat")
            for path in paths
            if os.path.isfile(os.path.join(path, name, f"{name}_0000.dat"))
        ]
        if not found:
            return None
        files.append(found[0])
    return files


GRID_FILES = _grid_files()
pytestmark = [
    pytest.mark.skipif(GRID_FILES is None, reason=f"pdf sets {SETS} required"),
    pytest.mark.cpu_only,
]


def python_me(context, function, inputs, outputs):
    api = context.add_matrix_element(
        ms.PythonFunction(function, inputs, outputs, particle_count=len(PIDS))
    )
    return ms.MatrixElement(
        api.index(),
        len(PIDS),
        [getattr(ms.MatrixElement, f"{name}_in") for name in inputs],
        [getattr(ms.MatrixElement, f"{name}_out") for name in outputs],
    )


def generate(run_dir, step_function, step_kwargs, target_count=2000):
    os.makedirs(run_dir, exist_ok=True)
    context, alphas_file = make_context(run_dir, 2)
    grids = [ms.PdfGrid(path) for path in GRID_FILES]
    grids[0].initialize_globals(context)
    grids[1].initialize_globals(context, "beam2")
    mapping = ms.PhaseSpaceMapping(
        [0.0] * len(PIDS),
        CM_ENERGY,
        mirror_beams=True,
        cuts=ms.Cuts(
            [
                ms.CutItem(ms.Observable(PIDS, ms.Observable.obs_pt, [[i]]), min=50.0)
                for i in (2, 3, 4)
            ]
        ),
    )
    diff_xs = ms.DifferentialCrossSection(
        matrix_element=python_me(
            context,
            lambda p: np.ones(len(p)),
            ["momenta"],
            ["matrix_element"],
        ),
        cm_energy=CM_ENERGY,
        running_coupling=None,
        energy_scale=ms.CachedScale(),
        pdf1=ms.CachedPdf(),
        pdf2=ms.CachedPdf(),
        return_matrix_element=True,
    )
    integrand = ms.Integrand(
        mapping,
        [diff_xs],
        pid_options=[PIDS],
        pdf_grid=grids[0],
        pdf_grid2=grids[1],
        running_coupling=ms.RunningCoupling(ms.AlphaSGrid(alphas_file)),
        energy_scale=ms.EnergyScale(len(PIDS), 91.188),
        channel_indices=[0],
        flavor_mirror=[True],
        parton_density_output=True,
    )
    step_me = python_me(
        context,
        step_function,
        ["momenta", "parton_densities", "random_flavor"],
        ["matrix_element_ratio", "color_index", "helicity_index", "flavor_index"],
    )
    config = make_config()
    config.target_count = target_count
    config.cpu_batch_size = 500
    channel = ms.ChannelEventGenerator(
        contexts=[context],
        integrand=integrand,
        event_file=os.path.join(run_dir, "events.0.npy"),
        weight_file=os.path.join(run_dir, "weights.0.npy"),
        config=config,
        subprocess_index=0,
        name="0",
        histograms=None,
    )
    step = ms.WeightingStep([step_me], mode=ms.WeightingStep.reweight, **step_kwargs)
    generator = ms.EventGenerator(
        [context],
        [channel],
        seed=3,
        config=config,
        groups=[ms.WeightingGroup([0], [step])],
    )
    generator.survey()
    generator.generate()
    output_file = os.path.join(run_dir, "combined.npy")
    generator.combine_to_compact_npy(output_file)
    return np.load(output_file), grids


def test_parton_densities_follow_the_beam_of_each_leg(tmp_path):
    seen = []

    def step(p, densities, random):
        seen.append((np.array(p), np.array(densities)))
        n = len(p)
        return (
            np.ones(n),
            np.full(n, -1, np.int32),
            np.full(n, -1, np.int32),
            np.full(n, -1, np.int32),
        )

    generate(str(tmp_path), step, {"flavor_target_count": 1})
    momenta = np.concatenate([p for p, _ in seen])
    densities = np.concatenate([d for _, d in seen])
    import lhapdf

    pdfs = [lhapdf.mkPDF(name, 0) for name in SETS]
    mirrored = momenta[:, 0, 3] < 0
    assert mirrored.any() and not mirrored.all()
    rng = np.random.default_rng(1)
    for k in rng.choice(len(momenta), 200, replace=False):
        for leg in (0, 1):
            beam = leg ^ int(mirrored[k])
            x = 2 * momenta[k, leg, 0] / CM_ENERGY
            for column, pid in enumerate(range(-6, 7)):
                code = 21 if pid == 0 else pid
                expected = 0.0 if abs(pid) == 6 else pdfs[beam].xfxQ(code, x, 91.188)
                assert densities[k, leg, column] == pytest.approx(
                    expected, rel=1e-6, abs=1e-12
                )


def test_switched_events_take_the_target_indices(tmp_path):
    def step(p, densities, random):
        n = len(p)
        target = np.where(random < 0.25, 2, -1).astype(np.int32)
        return (
            np.ones(n),
            np.full(n, 4, np.int32),
            np.full(n, 6, np.int32),
            target,
        )

    events, _ = generate(
        str(tmp_path), step, {"flavor_target_count": 3, "flavor_offset": 1}
    )
    switched = events["flavor_index"] == 3
    assert set(np.unique(events["flavor_index"])) == {0, 3}
    assert 0.2 < switched.mean() < 0.3
    assert (events["color_index"][switched] == 4).all()
    assert (events["helicity_index"][switched] == 6).all()
    assert (events["color_index"][~switched] == -1).all()
    assert (events["helicity_index"][~switched] == -1).all()


def test_switch_rules():
    context = ms.Context(ms.cpu_device(), 1)
    me = python_me(
        context,
        lambda p, d, r: None,
        ["momenta", "parton_densities", "random_flavor"],
        ["matrix_element_ratio", "color_index", "helicity_index", "flavor_index"],
    )
    step = ms.WeightingStep([me], flavor_target_count=2)
    assert step.switches_flavor() and step.needs_parton_densities()
    assert not ms.WeightingStep([me]).switches_flavor()


def test_switch_fields_survive_json():
    context = ms.Context(ms.cpu_device(), 1)
    me = python_me(
        context,
        lambda p, d, r: None,
        ["momenta", "parton_densities", "random_flavor"],
        ["matrix_element_ratio", "color_index", "helicity_index", "flavor_index"],
    )
    step = ms.WeightingStep(
        [me],
        flavor_target_count=2,
        flavor_offset=5,
        switch_diagram_map=[[1, 0], [0, 2]],
    )
    (group,) = ms.weighting_groups_from_json(
        ms.weighting_groups_to_json([ms.WeightingGroup([0], [step])])
    )
    restored = group.steps[0]
    assert restored.flavor_target_count == 2 and restored.flavor_offset == 5
    assert restored.switch_diagram_map == [[1, 0], [0, 2]]
