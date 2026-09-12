"""Run with Symbolica's SymJIT backend; 63 works, 64/65 abort before the fix."""
import argparse

import numpy as np
from symbolica import Expression, S


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--arity", type=int, default=64)
parser.add_argument("--no-compress", action="store_true")
args = parser.parse_args()
n = args.arity
parameters = list(S(*[f"x{i}" for i in range(3 * n)]))
outputs = []
for group in range(3):
    value = parameters[group * n]
    for j in range(1, n):
        x = parameters[group * n + j]
        value = value * x if j % 2 else value + x
    outputs.append(value)
options = {"compress": "false" if args.no_compress else "true"}
evaluator = Expression.evaluator_multiple(
    outputs, parameters, iterations=0, cpe_iterations=0, n_cores=1,
    jit_compile=True, jit_optimization_level=2, jit_options=options,
)
evaluator.jit_compile(True, optimization_level=2, options=options)
result = evaluator.evaluate_complex(np.ones((1, 3 * n), dtype=np.complex128))
assert np.array_equal(result, np.full((1, 3), (n + 1) // 2, dtype=np.complex128))
print(result)
