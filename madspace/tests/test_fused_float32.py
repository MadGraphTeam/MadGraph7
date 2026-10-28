"""Fused single-precision instructions against their unfused equivalents."""

import re

import pytest
import torch
from madspace_testing import generator, new_context, rand, randn

import madspace as ms
from madspace.torch import FunctionModule

N = 211
DIMS_IN = 7
DIMS_OUT = 5


def batch_type(dtype, shape):
    return ms.Type(dtype, ms.batch_size, shape)


def matmul_module(fused):
    f32, f64 = ms.DataType.float32, ms.DataType.float64
    fb = ms.FunctionBuilder(
        ms.NamedTypes(
            [
                ("x", batch_type(f32, [DIMS_IN])),
                ("w", ms.Type(f32, ms.BatchSize.one, [DIMS_OUT, DIMS_IN])),
                ("b", ms.Type(f32, ms.BatchSize.one, [DIMS_OUT])),
            ]
        ),
        ms.NamedTypes([("y", batch_type(f64, [DIMS_OUT]))]),
    )
    x, w, b = fb.input(0), fb.input(1), fb.input(2)
    if fused:
        y = fb.matmul_to_float64(x, w, b)
    else:
        y = fb.convert_float64(fb.matmul(x, w, b))
    fb.output(0, y)
    return FunctionModule(fb.function(), new_context())


def test_matmul_to_float64():
    g = generator(77)
    x = randn((N, DIMS_IN), generator=g, dtype=torch.float32, requires_grad=True)
    w = randn((DIMS_OUT, DIMS_IN), generator=g, dtype=torch.float32, requires_grad=True)
    b = randn((DIMS_OUT,), generator=g, dtype=torch.float32, requires_grad=True)
    y_grad = randn((N, DIMS_OUT), generator=g, dtype=torch.float64)

    results = []
    for fused in (True, False):
        xs, ws, bs = (t.detach().clone().requires_grad_() for t in (x, w, b))
        y = matmul_module(fused)(xs, ws, bs)
        assert y.dtype == torch.float64
        (y * y_grad).sum().backward()
        results.append((y.detach(), xs.grad, ws.grad, bs.grad))

    y_ref = (x @ w.T + b).double()
    torch.testing.assert_close(results[0][0], y_ref, rtol=1e-5, atol=1e-5)
    # the gradients through the fused and the unfused graph agree
    for fused_t, unfused_t in zip(results[0], results[1]):
        torch.testing.assert_close(fused_t, unfused_t, rtol=1e-4, atol=1e-4)

    g_ref = y_grad.float()
    torch.testing.assert_close(results[0][1], g_ref @ w.detach(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        results[0][2], g_ref.T @ x.detach(), rtol=1e-4, atol=1e-3
    )
    torch.testing.assert_close(results[0][3], g_ref.sum(dim=0), rtol=1e-4, atol=1e-3)


FUSED_NAMES = (
    "convert_float32",
    "convert_float64",
    "matmul_float32",
    "matmul_to_float64",
    "select_with_float32",
    "rqs_forward_with_float32",
    "rqs_inverse_with_float32",
)


def instruction_counts(function):
    names = "|".join(FUSED_NAMES)
    counts = {}
    for name in re.findall(rf"\b({names})\b", str(function)):
        counts[name] = counts.get(name, 0) + 1
    return counts


def make_flow(prefix):
    return ms.Flow(
        4,
        prefix=prefix,
        subnet_precision=ms.DataType.float32,
        subnet_hidden_dim=16,
        subnet_layers=3,
        invert_spline=False,
    )


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_conversion_lowering_graph(direction):
    flow = make_flow("lowering")
    function = getattr(flow, f"{direction}_function")()
    before = instruction_counts(function)
    after = instruction_counts(ms.fuse_conversions(function))
    spline = "rqs_forward" if direction == "forward" else "rqs_inverse"
    # every MLP of the flow converts its input and its output
    assert before["convert_float32"] == before["convert_float64"] == 4
    assert "convert_float32" not in after and "convert_float64" not in after
    assert after["matmul_to_float64"] == 4
    assert after["select_with_float32"] + after[f"{spline}_with_float32"] == 4
    # the unfused matmuls are the hidden layers
    assert after["matmul_float32"] == before["matmul_float32"] - 4
    # the original function is not modified, and the pass is idempotent
    assert instruction_counts(function) == before
    assert instruction_counts(ms.fuse_conversions(ms.fuse_conversions(function))) == after


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_conversion_lowering_numerics(direction, monkeypatch):
    """The lowered flow agrees with the unlowered one in values and gradients. On a
    GPU the reference runs with the lowering disabled."""
    g = generator(2024)
    flow = make_flow(f"lowering_{direction}")
    ctx = new_context()
    flow.initialize_globals(ctx, 42)
    # randomize all weights (the last layers start at zero)
    for name in ctx.global_names():
        weights = ctx.get_global(name).torch()
        weights[:] = 0.3 * randn(weights.shape, generator=g, dtype=torch.float32).to(
            weights.dtype
        )

    r = rand((137, 4), generator=g, dtype=torch.float64)
    function = getattr(flow, f"{direction}_function")()
    results = {}
    for lowered in (True, False):
        with monkeypatch.context() as mp:
            if not lowered:
                mp.setenv("MADSPACE_NO_FUSION", "1")
            module = FunctionModule(
                ms.fuse_conversions(function) if lowered else function, ctx
            )
            r_grad = r.clone().requires_grad_()
            out, det = module(r_grad)
            (out.sum() + det.log().sum()).backward()
            results[lowered] = (
                out.detach(),
                det.detach(),
                r_grad.grad,
                {n: p.grad.clone() for n, p in module.global_params.items()},
            )
    for lowered_t, ref_t in zip(results[True][:3], results[False][:3]):
        torch.testing.assert_close(lowered_t, ref_t, rtol=1e-12, atol=1e-12)
    for name, grad in results[True][3].items():
        torch.testing.assert_close(
            grad, results[False][3][name], rtol=1e-4, atol=1e-6
        )

