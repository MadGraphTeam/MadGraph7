import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytest import approx

import madspace as ms
from madspace.torch import FunctionModule

torch.set_default_dtype(torch.float64)
torch.manual_seed(3210)


@pytest.fixture
def mlp():
    ctx = ms.Context()
    mlp = ms.MLP(10, 1, 32, 3, ms.MLP.leaky_relu, "")
    mlp.initialize_globals(ctx)
    return FunctionModule(mlp.function(), ctx)


def test_properties():
    mlp = ms.MLP(10, 1, 32, 3, ms.MLP.leaky_relu, "")
    assert mlp.input_dim() == 10
    assert mlp.output_dim() == 1


def test_initialization(mlp):
    assert torch.all(mlp.global_params["layer1:weight"] != 0)
    assert torch.all(mlp.global_params["layer1:bias"] != 0)
    assert torch.all(mlp.global_params["layer2:weight"] != 0)
    assert torch.all(mlp.global_params["layer2:bias"] != 0)
    assert torch.all(mlp.global_params["layer3:weight"] == 0)
    assert torch.all(mlp.global_params["layer3:bias"] == 0)


@pytest.fixture(params=["relu", "leaky_relu", "elu", "gelu", "sigmoid", "softplus"])
def activation(request):
    return request.param


def test_activation(mlp, activation):
    fb = ms.FunctionBuilder([ms.batch_float_array(10)], [ms.batch_float_array(10)])
    fb.output(0, getattr(fb, activation)(fb.input(0)))
    func = FunctionModule(fb.function())
    x = 10 * torch.randn((1000, 10))
    x.requires_grad = True

    y_ms = func(x)
    y_ms.sum().backward()
    grad_me = x.grad
    x.grad = None

    y_torch = getattr(F, activation)(x)
    y_torch.sum().backward()
    grad_torch = x.grad

    assert y_ms.detach() == approx(y_torch.detach())
    assert grad_me == approx(grad_torch)


def test_training(mlp):
    mlp_torch = nn.Sequential(
        nn.Linear(10, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1),
    )

    with torch.no_grad():
        for i in range(3):
            mlp_torch[2 * i].weight[:] = mlp.global_params[f"layer{i+1}:weight"][0]
            mlp_torch[2 * i].bias[:] = mlp.global_params[f"layer{i+1}:bias"][0]

    opt_ms = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    opt_torch = torch.optim.Adam(mlp_torch.parameters(), lr=1e-3)

    for i in range(10):
        x = torch.randn((128, 10))

        loss_ms = (x**2 - mlp(x)).square().mean()
        opt_ms.zero_grad()
        loss_ms.backward()
        opt_ms.step()

        loss_torch = (x**2 - mlp_torch(x)).square().mean()
        opt_torch.zero_grad()
        loss_torch.backward()
        opt_torch.step()

        with torch.no_grad():
            for i in range(3):
                assert mlp.global_params[f"layer{i+1}:weight"][0].numpy() == approx(
                    mlp_torch[2 * i].weight.numpy()
                )
                assert mlp.global_params[f"layer{i+1}:bias"][0].numpy() == approx(
                    mlp_torch[2 * i].bias.numpy()
                )
                assert mlp.global_params[f"layer{i+1}:weight"].grad[
                    0
                ].numpy() == approx(mlp_torch[2 * i].weight.grad.numpy())
                assert mlp.global_params[f"layer{i+1}:bias"].grad[0].numpy() == approx(
                    mlp_torch[2 * i].bias.grad.numpy()
                )
