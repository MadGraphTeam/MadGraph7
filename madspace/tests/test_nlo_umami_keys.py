"""The provisional NLO keys of UMAMI (umami.h) in MatrixElement and
PythonFunction: the outputs per squared coupling order (squared_orders,
color_correlated over the pairs of coloured legs, spin_correlated, loop), the
inputs of the spin correlation (spin_legs, spin_vectors) and the electroweak
coupling (g_ew), the outputs of the Sudakov approximation of the EW
corrections per helicity combination, and the counts and colour
representations a PythonFunction reports as metadata.
The function of the tests returns known arrays of its inputs, on several
threads, so that the strides and offsets of the batch are those of the
runtime."""

import numpy as np
import pytest

import madspace as ms

ORDERS = 2
PARTICLES = 4
DIAGRAMS = 3
# three coloured legs, so three pairs
COLORS = [3, -3, 1, 8]
PAIRS = 3
BATCH = 5000
INPUTS = ["momenta", "alpha_s", "spin_legs", "spin_vectors", "scale", "g_ew"]
OUTPUTS = [
    "matrix_element",
    "diagram_amp2",
    "squared_orders",
    "color_correlated",
    "spin_correlated",
    "loop",
]


@pytest.fixture
def context():
    return ms.Context(device=ms.cpu_device(), thread_count=4)


@pytest.fixture
def inputs():
    rng = np.random.default_rng(42)
    return (
        rng.normal(size=(BATCH, PARTICLES, 4)),
        rng.uniform(0.05, 0.2, BATCH),
        rng.integers(-1, PARTICLES, (BATCH, 2)).astype(np.int32),
        rng.normal(size=(BATCH, 2, 4)),
        rng.uniform(10.0, 1000.0, BATCH),
        rng.uniform(0.2, 0.4, BATCH),
    )


def expected(momenta, alpha_s, legs, vectors, scale, g_ew):
    """The outputs of the toy matrix element, each a different function of its
    indices and inputs."""
    o = np.arange(ORDERS)
    orders = (o + 1) * (momenta[:, 0, 0] * alpha_s * g_ew)[:, None]
    amp2 = np.arange(1, DIAGRAMS + 1) * momenta[:, 1, 0][:, None]
    color = (
        100.0 * (o + 1)[None, :, None]
        + 10.0 * np.arange(PAIRS)[None, None, :]
        + momenta[:, 2, 0][:, None, None]
    )
    spin = (legs * vectors[:, :, 0] + vectors[:, :, 3]).sum(1)[:, None] + o
    loop = scale[:, None, None] * np.arange(1, 4) + 0.5 * o[:, None]
    return dict(
        matrix_element=orders.sum(1),
        diagram_amp2=amp2,
        squared_orders=orders,
        color_correlated=color,
        spin_correlated=spin,
        loop=loop,
    )


def toy(momenta, alpha_s, legs, vectors, scale, g_ew):
    values = expected(
        np.asarray(momenta),
        np.asarray(alpha_s),
        np.asarray(legs),
        np.asarray(vectors),
        np.asarray(scale),
        np.asarray(g_ew),
    )
    return tuple(np.ascontiguousarray(values[key]) for key in OUTPUTS)


def toy_function(**options):
    return ms.PythonFunction(
        toy,
        INPUTS,
        OUTPUTS,
        particle_count=PARTICLES,
        diagram_count=DIAGRAMS,
        squared_order_count=ORDERS,
        color_representations=COLORS,
        **options,
    )


ME_INPUTS = [
    ms.MatrixElement.momenta_in,
    ms.MatrixElement.alpha_s_in,
    ms.MatrixElement.spin_legs_in,
    ms.MatrixElement.spin_vectors_in,
    ms.MatrixElement.scale_in,
    ms.MatrixElement.g_ew_in,
]
ME_OUTPUTS = [
    ms.MatrixElement.matrix_element_out,
    ms.MatrixElement.diagram_amp2_out,
    ms.MatrixElement.squared_orders_out,
    ms.MatrixElement.color_correlated_out,
    ms.MatrixElement.spin_correlated_out,
    ms.MatrixElement.loop_out,
]


def test_outputs_by_order(context, inputs):
    """Every output with its shape, from the counts of the metadata."""
    api = context.add_matrix_element(toy_function())
    assert api.particle_count() == PARTICLES
    assert api.diagram_count() == DIAGRAMS
    assert api.squared_order_count() == ORDERS
    assert api.color_representations() == COLORS
    assert api.color_pair_count() == PAIRS
    element = ms.MatrixElement(api, ME_INPUTS, ME_OUTPUTS)
    assert element.squared_order_count() == ORDERS
    assert element.color_pair_count() == PAIRS
    assert list(element.arg_types().keys()) == INPUTS
    assert list(element.return_types().keys()) == OUTPUTS
    outputs = ms.FunctionRuntime(element.function(), context)(*inputs)
    for key, value in zip(OUTPUTS, outputs):
        reference = expected(*inputs)[key]
        value = np.asarray(value)
        assert value.shape == reference.shape, key
        np.testing.assert_array_equal(value, reference, err_msg=key)


def test_torch(context, inputs):
    """The same outputs from torch tensors."""
    torch = pytest.importorskip("torch")

    def function(*arrays):
        values = toy(*[a.numpy() for a in arrays])
        return tuple(torch.from_numpy(v) for v in values)

    api = context.add_matrix_element(
        ms.PythonFunction(
            function,
            INPUTS,
            OUTPUTS,
            framework="torch",
            particle_count=PARTICLES,
            diagram_count=DIAGRAMS,
            squared_order_count=ORDERS,
            color_representations=COLORS,
        )
    )
    element = ms.MatrixElement(api, ME_INPUTS, ME_OUTPUTS)
    outputs = ms.FunctionRuntime(element.function(), context)(*inputs)
    for key, value in zip(OUTPUTS, outputs):
        np.testing.assert_array_equal(
            np.asarray(value), expected(*inputs)[key], err_msg=key
        )


def test_counts_by_argument(context, inputs):
    """The constructor with the index takes the counts as arguments, one order
    by default."""
    api = context.add_matrix_element(toy_function())
    element = ms.MatrixElement(
        api.index(),
        PARTICLES,
        ME_INPUTS,
        [
            ms.MatrixElement.squared_orders_out,
            ms.MatrixElement.color_correlated_out,
            ms.MatrixElement.loop_out,
        ],
        squared_order_count=ORDERS,
        color_pair_count=PAIRS,
    )
    orders, color, loop = ms.FunctionRuntime(element.function(), context)(*inputs)
    reference = expected(*inputs)
    np.testing.assert_array_equal(np.asarray(orders), reference["squared_orders"])
    np.testing.assert_array_equal(np.asarray(color), reference["color_correlated"])
    np.testing.assert_array_equal(np.asarray(loop), reference["loop"])
    plain = ms.MatrixElement(api.index(), PARTICLES, ME_INPUTS, ME_OUTPUTS[:1])
    assert plain.squared_order_count() == 1


def test_one_order_without_metadata(context):
    """A library without the count of squared orders has one, and no colour
    representations."""
    api = context.add_matrix_element(
        ms.PythonFunction(
            lambda p: np.ones(len(p)), ["momenta"], ["matrix_element"], particle_count=4
        )
    )
    assert api.squared_order_count() == 1
    assert api.color_representations() == []
    with pytest.raises(RuntimeError, match="colour representations"):
        api.color_pair_count()


def test_errors(context):
    with pytest.raises(ValueError, match="needs squared_order_count"):
        ms.PythonFunction(toy, INPUTS, ["loop"], particle_count=PARTICLES)
    with pytest.raises(ValueError, match="needs diagram_count"):
        ms.PythonFunction(toy, INPUTS, ["diagram_amp2"], particle_count=PARTICLES)
    with pytest.raises(ValueError, match="needs color_representations"):
        ms.PythonFunction(
            toy,
            INPUTS,
            ["color_correlated"],
            particle_count=PARTICLES,
            squared_order_count=ORDERS,
        )
    with pytest.raises(ValueError, match="one value per particle"):
        ms.PythonFunction(toy, INPUTS, ["matrix_element"], color_representations=COLORS)
    api = context.add_matrix_element(toy_function())
    with pytest.raises(ValueError, match="boost of the momenta"):
        ms.MatrixElement(api, ME_INPUTS, ME_OUTPUTS, me_frame=[1, 2])
    with pytest.raises(ValueError, match="at least one squared order"):
        ms.MatrixElement(
            api.index(), PARTICLES, ME_INPUTS, ME_OUTPUTS, squared_order_count=0
        )
    with pytest.raises(ValueError, match="at least two coloured legs"):
        ms.MatrixElement(
            api.index(),
            PARTICLES,
            ME_INPUTS,
            [ms.MatrixElement.color_correlated_out],
            squared_order_count=ORDERS,
        )


def test_wrong_shape(context, inputs):
    """A function that returns another shape stops the call."""

    def wrong(momenta, alpha_s, legs, vectors, scale, g_ew):
        values = toy(momenta, alpha_s, legs, vectors, scale, g_ew)
        return values[:4] + (values[4][:, :1],) + values[5:]

    api = context.add_matrix_element(
        ms.PythonFunction(
            wrong,
            INPUTS,
            OUTPUTS,
            particle_count=PARTICLES,
            diagram_count=DIAGRAMS,
            squared_order_count=ORDERS,
            color_representations=COLORS,
        )
    )
    element = ms.MatrixElement(api, ME_INPUTS, ME_OUTPUTS)
    with pytest.raises(RuntimeError, match="spin_correlated"):
        ms.FunctionRuntime(element.function(), context)(*inputs)


# the helicity combinations and interference amplitudes of the outputs of the
# Sudakov approximation of the EW corrections
HELICITIES = 3
INTERFERENCES = 2
SUDAKOV_OUTPUTS = [
    "ew_sudakov_diagonal",
    "ew_sudakov_lo2",
    "ew_sudakov_derivatives",
    "ew_sudakov_interference",
]


def sudakov_expected(momenta, alpha_s, g_ew):
    h = np.arange(HELICITIES)
    base = (momenta[:, 0, 0] * alpha_s * g_ew)[:, None]
    diagonal = (h + 1) * base
    lo2 = (h + 2) * base
    derivatives = diagonal[:, :, None] * np.arange(1, 7)
    interference = (
        diagonal[:, :, None, None]
        + 10.0 * np.arange(INTERFERENCES)[None, None, :, None]
        + 100.0 * np.arange(2)
    )
    return dict(
        ew_sudakov_diagonal=diagonal,
        ew_sudakov_lo2=lo2,
        ew_sudakov_derivatives=derivatives,
        ew_sudakov_interference=interference,
    )


def test_ew_sudakov_outputs(context, inputs):
    """The outputs of the Sudakov approximation per helicity combination,
    with the helicity and interference counts of the metadata, also through
    the constructor with the counts as arguments, and their errors."""
    momenta, alpha_s, *_, g_ew = inputs

    def blocks(p, a, g):
        values = sudakov_expected(np.asarray(p), np.asarray(a), np.asarray(g))
        return tuple(np.ascontiguousarray(values[key]) for key in SUDAKOV_OUTPUTS)

    api = context.add_matrix_element(
        ms.PythonFunction(
            blocks,
            ["momenta", "alpha_s", "g_ew"],
            SUDAKOV_OUTPUTS,
            particle_count=PARTICLES,
            helicity_count=HELICITIES,
            interference_count=INTERFERENCES,
        )
    )
    assert api.helicity_count() == HELICITIES
    assert api.ew_sudakov_interference_count() == INTERFERENCES
    element_inputs = [
        ms.MatrixElement.momenta_in,
        ms.MatrixElement.alpha_s_in,
        ms.MatrixElement.g_ew_in,
    ]
    element_outputs = [
        ms.MatrixElement.ew_sudakov_diagonal_out,
        ms.MatrixElement.ew_sudakov_lo2_out,
        ms.MatrixElement.ew_sudakov_derivatives_out,
        ms.MatrixElement.ew_sudakov_interference_out,
    ]
    reference = sudakov_expected(momenta, alpha_s, g_ew)
    for element in (
        ms.MatrixElement(api, element_inputs, element_outputs),
        ms.MatrixElement(
            api.index(),
            PARTICLES,
            element_inputs,
            element_outputs,
            helicity_count=HELICITIES,
            interference_count=INTERFERENCES,
        ),
    ):
        assert element.helicity_count() == HELICITIES
        assert element.interference_count() == INTERFERENCES
        assert list(element.return_types().keys()) == SUDAKOV_OUTPUTS
        outputs = ms.FunctionRuntime(element.function(), context)(
            momenta, alpha_s, g_ew
        )
        for key, value in zip(SUDAKOV_OUTPUTS, outputs):
            value = np.asarray(value)
            assert value.shape == reference[key].shape, key
            np.testing.assert_array_equal(value, reference[key], err_msg=key)
    # the counts of a library without them
    plain = context.add_matrix_element(toy_function())
    assert plain.ew_sudakov_interference_count() == 0
    with pytest.raises(ValueError, match="needs helicity_count"):
        ms.PythonFunction(
            blocks, ["momenta"], ["ew_sudakov_diagonal"], particle_count=PARTICLES
        )
    with pytest.raises(ValueError, match="needs interference_count"):
        ms.PythonFunction(
            blocks,
            ["momenta"],
            ["ew_sudakov_interference"],
            particle_count=PARTICLES,
            helicity_count=HELICITIES,
        )
    with pytest.raises(ValueError, match="need helicity combinations"):
        ms.MatrixElement(api.index(), PARTICLES, element_inputs, element_outputs[:1])
    with pytest.raises(ValueError, match="need interference amplitudes"):
        ms.MatrixElement(
            api.index(),
            PARTICLES,
            element_inputs,
            element_outputs[3:],
            helicity_count=HELICITIES,
        )
