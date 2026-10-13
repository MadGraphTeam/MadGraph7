"""Single-precision NN instructions: forward and backward against torch."""

import pytest
import torch

import madspace as ms
from madspace.torch import FunctionModule

# a local generator, so that the global RNG state seen by other tests is unchanged
GEN = torch.Generator().manual_seed(1234)

N = 103
DIMS_IN = 7
DIMS_OUT = 5

ACTIVATIONS = {
    "relu": torch.relu,
    "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, 0.01),
    "elu": torch.nn.functional.elu,
    "gelu": torch.nn.functional.gelu,
    "sigmoid": torch.sigmoid,
    "softplus": torch.nn.functional.softplus,
    "scaled_tanh": lambda x: 4 * torch.tanh(x / 4),
}

DTYPES = {
    "float32": (ms.DataType.float32, torch.float32, 1e-5),
    "float64": (ms.DataType.float64, torch.float64, 1e-12),
}


def batch_type(dtype, shape):
    return ms.Type(dtype, ms.batch_size, shape)


@pytest.mark.parametrize("precision", DTYPES.keys())
@pytest.mark.parametrize("name", ACTIVATIONS.keys())
def test_activation(name, precision):
    ms_dtype, torch_dtype, tol = DTYPES[precision]
    io_type = batch_type(ms_dtype, [DIMS_IN])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type)]), ms.NamedTypes([("out", io_type)])
    )
    fb.output(0, getattr(fb, name)(fb.input(0)))
    module = FunctionModule(fb.function())

    x = (
        3 * torch.randn((N, DIMS_IN), dtype=torch_dtype, generator=GEN)
    ).requires_grad_()
    out = module(x)
    assert out.dtype == torch_dtype
    out_grad = torch.randn(out.shape, dtype=out.dtype, generator=GEN)
    (out * out_grad).sum().backward()

    x_ref = x.detach().clone().requires_grad_()
    out_ref = ACTIVATIONS[name](x_ref)
    (out_ref * out_grad).sum().backward()
    torch.testing.assert_close(out, out_ref, rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=tol, atol=tol)


@pytest.mark.parametrize("precision", DTYPES.keys())
def test_matmul(precision):
    ms_dtype, torch_dtype, tol = DTYPES[precision]
    fb = ms.FunctionBuilder(
        ms.NamedTypes(
            [
                ("x", batch_type(ms_dtype, [DIMS_IN])),
                ("w", ms.Type(ms_dtype, ms.BatchSize.one, [DIMS_OUT, DIMS_IN])),
                ("b", ms.Type(ms_dtype, ms.BatchSize.one, [DIMS_OUT])),
            ]
        ),
        ms.NamedTypes([("y", batch_type(ms_dtype, [DIMS_OUT]))]),
    )
    fb.output(0, fb.matmul(fb.input(0), fb.input(1), fb.input(2)))
    module = FunctionModule(fb.function())

    x = torch.randn((N, DIMS_IN), dtype=torch_dtype, generator=GEN, requires_grad=True)
    # single inputs are passed without batch dimension
    w = torch.randn(
        (DIMS_OUT, DIMS_IN), dtype=torch_dtype, generator=GEN, requires_grad=True
    )
    b = torch.randn((DIMS_OUT,), dtype=torch_dtype, generator=GEN, requires_grad=True)
    y = module(x, w, b)
    assert y.dtype == torch_dtype
    y_grad = torch.randn(y.shape, dtype=y.dtype, generator=GEN)
    (y * y_grad).sum().backward()

    refs = [t.detach().clone().requires_grad_() for t in (x, w, b)]
    y_ref = refs[0] @ refs[1].T + refs[2]
    (y_ref * y_grad).sum().backward()
    torch.testing.assert_close(y, y_ref, rtol=tol, atol=tol)
    for t, t_ref in zip((x, w, b), refs):
        torch.testing.assert_close(t.grad, t_ref.grad, rtol=tol, atol=tol)


def test_convert_roundtrip():
    in_type = batch_type(ms.DataType.float64, [3])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", in_type)]),
        ms.NamedTypes(
            [("single", batch_type(ms.DataType.float32, [3])), ("out", in_type)]
        ),
    )
    single = fb.convert_float32(fb.input(0))
    fb.output(0, single)
    fb.output(1, fb.convert_float64(fb.relu(single)))
    module = FunctionModule(fb.function())

    x = torch.randn((N, 3), dtype=torch.float64, generator=GEN, requires_grad=True)
    single_out, out = module(x)
    assert single_out.dtype == torch.float32
    assert out.dtype == torch.float64
    torch.testing.assert_close(single_out, x.detach().float())
    torch.testing.assert_close(out, torch.relu(x.detach().float()).double())
    out.sum().backward()
    torch.testing.assert_close(x.grad, (x.detach() > 0).double())


def test_literal_converted_for_float32():
    io_type = batch_type(ms.DataType.float32, [1])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([("in", io_type)]), ms.NamedTypes([("out", io_type)])
    )
    # float64 literals are converted when the instruction dispatches to float32
    fb.output(0, fb.matmul(fb.input(0), ms.Value([2.0], [1, 1]), ms.Value([1.0])))
    module = FunctionModule(fb.function())
    x = torch.randn((N, 1), dtype=torch.float32, generator=GEN)
    torch.testing.assert_close(module(x), 2 * x + 1)


def test_flow_float32_subnets(tmp_path):
    """A flow with float32 subnetworks matches the float64 one with the same
    (rounded) weights; the Jacobian and the output stay in float64."""
    ctx = ms.default_context()
    flows = {
        dtype: ms.Flow(
            4, prefix=f"fp32_test_{name}", subnet_precision=dtype, invert_spline=False
        )
        for name, dtype in [("f32", ms.DataType.float32), ("f64", ms.DataType.float64)]
    }
    for flow in flows.values():
        flow.initialize_globals(ctx, 42)
    # randomize all weights (the last layers start at zero) and round to float32
    names_f64 = [n for n in ctx.global_names() if n.startswith("fp32_test_f64")]
    for name in names_f64:
        weights = ctx.get_global(name).torch()
        weights[:] = (
            0.3
            * torch.randn(weights.shape, generator=GEN, dtype=torch.float32).double()
        )
        ctx.get_global(name.replace("f64", "f32")).torch()[:] = weights.float()

    r = torch.rand((N, 4), dtype=torch.float64, generator=GEN)
    out32, det32 = flows[ms.DataType.float32].map_forward([r])
    out64, det64 = flows[ms.DataType.float64].map_forward([r])
    assert out32.dtype == det32.dtype == torch.float64
    torch.testing.assert_close(out32, out64, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(det32, det64, rtol=1e-4, atol=1e-5)

    # float32 globals survive saving and loading
    ctx.save_globals(str(tmp_path))
    loaded = ms.Context()
    loaded.load_globals(str(tmp_path))
    for name in names_f64:
        name32 = name.replace("f64", "f32")
        global32 = loaded.get_global(name32).torch()
        assert global32.dtype == torch.float32
        torch.testing.assert_close(global32, ctx.get_global(name32).torch())
