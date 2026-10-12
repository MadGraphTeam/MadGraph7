"""TEMP diagnostic: ColorOrderedMapping with a pt cut, every intermediate value of the
forward and inverse functions on CUDA against the CPU, on the same inputs."""

import json
import os
import tempfile

import torch

import madspace as ms

GPU = os.environ.get("DIAG_DEVICE", "cuda")
N = int(os.environ.get("DIAG_N", 200_000))
ORDER = [0, 2, 3, 4, 5, 1]


def with_all_outputs(function):
    """The function with every batch float local an instruction makes as an output."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.json")
        function.save(path)
        data = json.load(open(path))
        known = {o["local"] for o in data["outputs"]}
        extra = []
        for instr in function.instructions:
            for out in instr.outputs:
                t = out.type
                if (
                    out.local_index not in known
                    and str(t.dtype).endswith("float")
                    and str(t.batch_size) == "batch_size"
                ):
                    extra.append(out.local_index)
                    known.add(out.local_index)
                    data["outputs"].append(
                        {
                            "dtype": "float",
                            "local": out.local_index,
                            "name": f"l{out.local_index}",
                            "shape": list(t.shape),
                        }
                    )
        json.dump(data, open(path, "w"))
        return ms.Function.load(path), len(function.outputs), extra


def rel_diff(a, b):
    a, b = a.double().reshape(a.shape[0], -1), b.double().reshape(b.shape[0], -1)
    scale = torch.maximum(a.abs(), b.abs()).amax(1, keepdim=True).clamp(min=1e-300)
    d = ((a - b).abs() / scale).amax(1)
    return d[torch.isfinite(d)]


def report(function, inputs, label):
    print(f"\n===== {label}")
    full, n_out, extra = with_all_outputs(function)
    outs = {}
    for dev in ("cpu", GPU):
        rt = ms.FunctionRuntime(full)
        outs[dev] = [torch.from_dlpack(t).cpu() for t in rt.call([x.to(dev) for x in inputs])]
    diff = {}
    for k, local in enumerate(extra):
        d = rel_diff(outs["cpu"][n_out + k], outs[GPU][n_out + k])
        diff[local] = d
    in_diff = {}
    for i, x in enumerate(inputs):
        in_diff[function.inputs[i].local_index] = torch.zeros(1)

    def fmt(d):
        if d is None or d.numel() == 0:
            return "      -    "
        return f"{torch.quantile(d, .999).item():.1e}/{d.max().item():.1e}"

    print("instruction: q999/max of the scaled CUDA-CPU difference, outputs <- inputs")
    for instr in function.instructions:
        o = [fmt(diff.get(v.local_index)) for v in instr.outputs]
        i = [fmt(diff.get(v.local_index, in_diff.get(v.local_index))) for v in instr.inputs]
        print(f"  {str(instr)[:70]:70s} out {' '.join(o)}  in {' '.join(i)}")
    return outs["cpu"][:n_out], outs[GPU][:n_out]


mapping = ms.ColorOrderedMapping(ORDER, pt_min=[20.0] * 4)
g = torch.Generator().manual_seed(11)
r = torch.rand((N, mapping.random_dim()), generator=g, dtype=torch.float64)
d = torch.randint(0, 2, (N, mapping.discrete_dim()), generator=g, dtype=torch.int32)
inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
cond = [torch.full((N,), 1000.0, dtype=torch.float64)] + [torch.zeros(N, dtype=torch.float64)] * 4
cpu_out, _ = report(mapping.forward_function(), inputs + cond, "forward, pt_min")
momenta = cpu_out[: len(ORDER)]
report(mapping.inverse_function(), momenta + cond, "inverse of the CPU momenta, pt_min")
