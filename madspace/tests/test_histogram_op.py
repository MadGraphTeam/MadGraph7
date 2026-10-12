"""The histogram instruction: n_bins interior bins plus an underflow and an overflow bin.

Its outputs hold n_bins + 2 values, in that order: underflow, the interior bins, overflow.
The GPU runtime reduces the events to these n_bins + 2 keys (and used to do so into a
scratch buffer of n_bins); .github/gpu_ci/madspace_checks.py compares it with this CPU
result on a cluster GPU.
"""

import numpy as np

import madspace as ms

N_BINS = 5


def run_histogram(x, w, lo, hi):
    out_type = ms.Type(ms.DataType.float, ms.BatchSize.one, [N_BINS + 2])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(name, ms.batch_float) for name in ("x", "w", "lo", "hi")]),
        ms.NamedTypes([("values", out_type), ("square_values", out_type)]),
    )
    values, square_values = fb.histogram(*(fb.input(i) for i in range(4)), N_BINS)
    fb.output(0, values)
    fb.output(1, square_values)
    runtime = ms.FunctionRuntime(fb.function(), ms.default_context())
    return [np.asarray(ms.Tensor.numpy(t))[0] for t in runtime.call([x, w, lo, hi])]


def test_histogram_underflow_bins_overflow():
    x = np.array([-1.0, 0.1, 0.5, 0.95, 1.5, 2.0, 0.3])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    n = len(x)
    values, square_values = run_histogram(x, w, np.zeros(n), np.ones(n))
    # underflow: -1; bins of width 0.2: 0.1, 0.3, 0.5, -, 0.95; overflow: 1.5 and 2
    expected = np.array([1.0, 2.0, 7.0, 3.0, 0.0, 4.0, 5.0 + 6.0])
    np.testing.assert_allclose(values, expected)
    np.testing.assert_allclose(square_values, [1.0, 4.0, 49.0, 9.0, 0.0, 16.0, 25.0 + 36.0])


def test_histogram_range_per_event():
    """min and max are per-event inputs: the same value falls in different bins."""
    x = np.full(3, 0.5)
    w = np.ones(3)
    values, _ = run_histogram(x, w, np.array([0.0, 0.4, 1.0]), np.array([1.0, 1.4, 2.0]))
    # 0.5 in [0, 1): bin 2 (index 3); in [0.4, 1.4): bin 0 (index 1); below [1, 2): underflow
    np.testing.assert_allclose(values, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
