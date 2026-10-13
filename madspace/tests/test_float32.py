"""Single-precision NN instructions: forward and backward against torch."""

import pytest
import torch

import madspace as ms
from madspace.torch import FunctionModule

torch.manual_seed(1234)

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

    x = (3 * torch.randn((N, DIMS_IN), dtype=torch_dtype)).requires_grad_()
    out = module(x)
    assert out.dtype == torch_dtype
    out_grad = torch.randn_like(out)
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

    x = torch.randn((N, DIMS_IN), dtype=torch_dtype, requires_grad=True)
    # single inputs are passed without batch dimension
    w = torch.randn((DIMS_OUT, DIMS_IN), dtype=torch_dtype, requires_grad=True)
    b = torch.randn((DIMS_OUT,), dtype=torch_dtype, requires_grad=True)
    y = module(x, w, b)
    assert y.dtype == torch_dtype
    y_grad = torch.randn_like(y)
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

    x = torch.randn((N, 3), dtype=torch.float64, requires_grad=True)
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
    x = torch.randn((N, 1), dtype=torch.float32)
    torch.testing.assert_close(module(x), 2 * x + 1)
