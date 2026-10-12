import numpy as np
import pytest

import madspace as ms

PDG_IDS = [21, 21, 6, -6]


@pytest.fixture
def context():
    # several threads, so the function is called from worker threads
    return ms.Context(device=ms.cpu_device(), thread_count=4)


@pytest.fixture
def momenta():
    rng = np.random.default_rng(1234)
    return rng.normal(size=(5000, 4, 4)) * 100.0


def pt(momenta, index):
    return np.hypot(momenta[:, index, 1], momenta[:, index, 2])


def run_custom(context, function, kind, momenta, framework="numpy"):
    api = context.add_matrix_element(
        ms.PythonFunction(function, ["momenta", "pdg_ids"], [kind], framework=framework)
    )
    generator = ms.CustomFunction(api, getattr(ms.CustomFunction, kind), 4, [PDG_IDS])
    return ms.FunctionRuntime(generator.function(), context)(momenta)


def test_numpy_cut(context, momenta):
    def cut(p, pdg_ids):
        assert p.shape == (len(p), 4, 4)
        assert (pdg_ids == PDG_IDS).all()
        return (pt(p, 2) > 100.0).astype(np.float64)

    values = run_custom(context, cut, "cut", momenta)
    assert np.array_equal(np.asarray(values), (pt(momenta, 2) > 100.0).astype(float))


def test_numpy_scale(context, momenta):
    def scale(p, pdg_ids):
        return (pt(p, 2) + pt(p, 3)) / 2

    values = run_custom(context, scale, "scale", momenta)
    assert np.allclose(values, (pt(momenta, 2) + pt(momenta, 3)) / 2)


def test_numpy_pdf(context):
    def pdf(beam_index, pdg_id, x, q):
        return np.where(pdg_id == 21, 3.0, 0.5) * np.sqrt(x) * (1 - x) ** 3 + beam_index

    api = context.add_matrix_element(
        ms.PythonFunction(
            pdf, ["beam_index", "pdg_id", "momentum_fraction", "scale"], ["pdf"]
        )
    )
    x = np.array([0.1, 0.5, 0.9])
    for beam_index in (0, 1):
        density = ms.PartonDensity(ms.CustomPdf(api), [21, 2], beam_index)
        values = np.asarray(
            ms.FunctionRuntime(density.function(), context)(x, np.full(3, 100.0))
        )
        toy = np.sqrt(x) * (1 - x) ** 3
        assert np.allclose(values[:, 0], 3 * toy + beam_index)
        assert np.allclose(values[:, 1], 0.5 * toy + beam_index)


def test_multiple_outputs_fixed_particle_count(context, momenta):
    def matrix_element(p):
        return p[:, 2, 0] ** 2, np.full(len(p), 3, dtype=np.int32)

    api = context.add_matrix_element(
        ms.PythonFunction(
            matrix_element,
            ["momenta"],
            ["matrix_element", "helicity_index"],
            particle_count=4,
        )
    )
    generator = ms.MatrixElement(
        api.index(),
        4,
        [ms.MatrixElement.momenta_in],
        [ms.MatrixElement.matrix_element_out, ms.MatrixElement.helicity_index_out],
    )
    me, helicity = ms.FunctionRuntime(generator.function(), context)(momenta)
    assert np.allclose(me, momenta[:, 2, 0] ** 2)
    assert (np.asarray(helicity) == 3).all()


@pytest.mark.parametrize(
    "result, message",
    [
        (lambda p: np.ones(len(p), dtype=np.float32), "must have dtype float64"),
        (lambda p: np.ones(3), "must have shape"),
        (lambda p: list(np.ones(len(p))), "must be a numpy.ndarray"),
    ],
)
def test_result_checks(context, momenta, result, message):
    with pytest.raises(RuntimeError, match=message):
        run_custom(context, lambda p, pdg_ids: result(p), "cut", momenta)


def test_exception_propagates(context, momenta):
    def broken(p, pdg_ids):
        raise KeyError("something went wrong")

    with pytest.raises(
        RuntimeError, match="Error in Python function broken.*something"
    ):
        run_custom(context, broken, "cut", momenta)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown input"):
        ms.PythonFunction(lambda: None, ["momentum"], ["cut"])
    with pytest.raises(ValueError, match="unknown output"):
        ms.PythonFunction(lambda: None, ["momenta"], ["cuts"])
    with pytest.raises(ValueError, match="framework"):
        ms.PythonFunction(lambda: None, ["momenta"], ["cut"], framework="jax")


def test_release_inputs_on_worker_threads(context, momenta):
    # numpy-backed inputs are freed by madspace, possibly on a worker thread
    def cut(p, pdg_ids):
        return np.ones(len(p))

    api = context.add_matrix_element(
        ms.PythonFunction(cut, ["momenta", "pdg_ids"], ["cut"])
    )
    generator = ms.CustomFunction(api, ms.CustomFunction.cut, 4, [PDG_IDS])
    runtime = ms.FunctionRuntime(generator.function(), context)
    for _ in range(20):
        runtime(momenta.copy())
        runtime.release_inputs()


def test_torch_cpu(context, momenta):
    torch = pytest.importorskip("torch")

    def scale(p, pdg_ids):
        assert isinstance(p, torch.Tensor)
        return torch.hypot(p[:, 2:, 1], p[:, 2:, 2]).sum(dim=1) / 2

    expected = (pt(momenta, 2) + pt(momenta, 3)) / 2
    values = run_custom(context, scale, "scale", momenta, framework="torch")
    assert np.allclose(values, expected)
    values = run_custom(
        context, scale, "scale", torch.from_numpy(momenta), framework="torch"
    )
    assert torch.allclose(values, torch.from_numpy(expected))


def test_torch_wrong_type(context, momenta):
    pytest.importorskip("torch")
    with pytest.raises(RuntimeError, match="must be a torch.Tensor"):
        run_custom(context, lambda p, ids: np.ones(len(p)), "cut", momenta, "torch")
