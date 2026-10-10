#!/usr/bin/env python3
"""madspace GPU runtime operations against the CPU runtime, on the same inputs.

Run by madspace_ops_gpu.sh (job madspace_ops of gpu_runner_ci.yml) in its run
directory. The inputs go to the GPU as torch tensors (dlpack), so that madspace runs
the function on its GPU runtime; the same function on numpy inputs runs on the CPU.

1. histogram: a weighted histogram with underflow and overflow bins (the GPU reduction
   used to write its n_bins + 2 keys into a scratch buffer of n_bins);
2. rank4_add: an elementwise kernel on a rank-4 tensor, batch x 2 x 3 x 4 (the GPU
   launcher computed only the first slice of the last dimension).

Writes summary.txt (key=value lines) in the current directory; exit status 1 if a
check fails.
"""

import os
import sys
import time

import numpy as np

import madspace as ms

results = {}


def check(name, ok, detail):
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    results[name] = (ok, detail)


def to_numpy(tensor, torch):
    return torch.from_dlpack(tensor).cpu().numpy()


def check_histogram(torch, device, n=100000, n_bins=40):
    out_type = ms.Type(ms.DataType.float, ms.BatchSize.one, [n_bins + 2])
    fb = ms.FunctionBuilder(
        ms.NamedTypes([(name, ms.batch_float) for name in ('x', 'w', 'lo', 'hi')]),
        ms.NamedTypes([('values', out_type), ('square_values', out_type)]),
    )
    values, square_values = fb.histogram(*(fb.input(i) for i in range(4)), n_bins)
    fb.output(0, values)
    fb.output(1, square_values)
    function = fb.function()
    rng = np.random.default_rng(17)
    inputs = [rng.uniform(-0.5, 1.5, n), rng.normal(size=n), np.zeros(n), np.ones(n)]
    cpu = [np.asarray(ms.Tensor.numpy(t))
           for t in ms.FunctionRuntime(function, ms.default_context()).call(inputs)]
    gpu = [to_numpy(t, torch) for t in ms.FunctionRuntime(function).call(
        [torch.from_numpy(a).to(device) for a in inputs])]
    worst = max(float(np.max(np.abs(g - c) / np.maximum(np.abs(c), 1e-300)))
                for g, c in zip(gpu, cpu))
    under, over = cpu[0][0][0], cpu[0][0][-1]
    check('histogram', worst < 1e-9 and under != 0 and over != 0,
          '%d bins + underflow/overflow, max rel diff GPU vs CPU %.1e '
          '(underflow %.4g, overflow %.4g)' % (n_bins, worst, under, over))


def check_rank4_add(torch, device, n=4096):
    rank4 = ms.Type(ms.DataType.float, ms.batch_size, [2, 3, 4])
    fb = ms.FunctionBuilder(ms.NamedTypes([('x', rank4)]), ms.NamedTypes([('y', rank4)]))
    x = fb.input(0)
    fb.output(0, fb.add(x, x))
    a = np.random.default_rng(23).normal(size=(n, 2, 3, 4))
    (y,) = ms.FunctionRuntime(fb.function()).call([torch.from_numpy(a).to(device)])
    y = to_numpy(y, torch)
    wrong = int(np.sum(y != 2 * a))
    check('rank4_add', wrong == 0, '%d of %d elements of x + x differ from 2 x'
          % (wrong, a.size))


def main():
    start = time.time()
    try:
        import torch
    except ImportError as error:
        check('torch', False, 'cannot import torch (%s): no GPU inputs' % error)
        torch = None
    if torch is not None and not torch.cuda.is_available():
        check('torch', False, 'torch sees no GPU')
        torch = None
    if torch is not None:
        for name, step in (('histogram', check_histogram), ('rank4_add', check_rank4_add)):
            try:
                step(torch, 'cuda')  # also the ROCm devices in torch
            except Exception as error:  # one broken step must not hide the others
                check(name, False, 'error: %s' % error)
    failures = [name for name, (ok, _detail) in results.items() if not ok]
    with open('summary.txt', 'w') as f:
        for name, (ok, detail) in results.items():
            f.write('%s=%s (%s)\n' % (name, 'ok' if ok else 'FAILED', detail))
        f.write('node=%s\ngpu=%s\nwalltime=%ds\n' % (
            os.uname().nodename, os.environ.get('GPU_NAME', 'none'), time.time() - start))
    print('\n%d check(s) failed: %s' % (len(failures), ', '.join(failures)) if failures
          else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
