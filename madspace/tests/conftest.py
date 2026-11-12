"""Runs the tests on the device set by MADSPACE_TEST_DEVICE (cpu, cuda or hip).

Tests marked cpu_only (host-side code, e.g. LHE files) are skipped on a GPU.
The FKS tests also import the MG7 NLO driver (madgraph/mg7) of the repository.
"""

import os
import sys

import pytest
import torch
from madspace_testing import BACKEND, DEVICE, ON_GPU

torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)

REPOSITORY = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPOSITORY not in sys.path:
    sys.path.append(REPOSITORY)


def pytest_configure(config):
    config.addinivalue_line("markers", "cpu_only: host-side test, skipped on a GPU")
    if ON_GPU and not torch.cuda.is_available():
        # a GPU run without a GPU must fail, not pass with every test on the CPU
        pytest.exit(
            f"MADSPACE_TEST_DEVICE={BACKEND}, but torch {torch.__version__} "
            "has no GPU available",
            returncode=1,
        )


def pytest_report_header(config):
    name = torch.cuda.get_device_name(DEVICE) if ON_GPU else "cpu"
    simd_mode = os.environ.get("MADSPACE_SIMD_MODE", "auto")
    return [
        f"madspace test device: {BACKEND} ({name})",
        f"madspace SIMD mode: {simd_mode}",
    ]


def pytest_collection_modifyitems(config, items):
    if not ON_GPU:
        return
    skip = pytest.mark.skip(reason=f"cpu_only, test device is {BACKEND}")
    for item in items:
        if "cpu_only" in item.keywords:
            item.add_marker(skip)
