# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import socket
import subprocess
import sys

_RESTRICTED_MODE_MRE = r"""
from symbolica import Expression, is_licensed
from symbolica.community.spenso import (
    LibraryTensor,
    Representation,
    TensorLibrary,
    TensorName,
    TensorNetwork,
)

assert not is_licensed()
library = TensorLibrary.hep_lib_atom()
minkowski = Representation.mink(4)
mu = minkowski("ufo_l_1_3")
nu = minkowski("ufo_l_2_3")
rho = minkowski("ufo_l_1_4")
sigma = minkowski("ufo_l_2_4")
a = TensorName("spenso_restricted_mode_test::A")
b = TensorName("spenso_restricted_mode_test::B")
library.register(
    LibraryTensor.dense(
        a(minkowski, minkowski),
        tuple(Expression.parse(f"a{i}") for i in range(16)),
    )
)
library.register(
    LibraryTensor.dense(
        b(minkowski, minkowski),
        tuple(Expression.parse("1" if i == 5 else "0") for i in range(16)),
    )
)
metric = TensorName.g()
expression = (
    -2 * metric(nu, mu).to_expression() * metric(sigma, rho).to_expression()
    + 2 * metric(mu, rho).to_expression() * metric(nu, sigma).to_expression()
    + 2 * metric(nu, rho).to_expression() * metric(sigma, mu).to_expression()
) * a(mu, nu).to_expression() * b(rho, sigma).to_expression()

network = TensorNetwork(expression, library)
network.execute(library=library)
result = network.result_tensor(library)
result.to_dense()
print(result[0])
"""


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_spenso_sequential_execution_respects_symbolica_restricted_mode() -> None:
    environment = dict(os.environ)
    environment.pop("SYMBOLICA_LICENSE", None)
    environment.pop("SYMBOLICA_MASTER_LICENSE", None)
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    environment["SYMBOLICA_PORT"] = str(_free_local_port())
    result = subprocess.run(
        [sys.executable, "-c", _RESTRICTED_MODE_MRE],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2.00000000000000*a0+2*a5-2*a10-2*a15" in result.stdout
