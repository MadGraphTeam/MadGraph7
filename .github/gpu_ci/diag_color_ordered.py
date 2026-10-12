"""TEMP diagnostic: ColorOrderedMapping with pair cuts, CUDA against CPU on the same inputs."""

import os

import torch

import madspace as ms

GPU = os.environ.get("DIAG_DEVICE", "cuda")

N = int(os.environ.get("DIAG_N", 200_000))
CUTS = dict(
    pt_min=[20.0] * 4,
    m_inv_min=[[0.0 if i == j else 10.0 for j in range(4)] for i in range(4)],
    dr_min=[[0.0 if i == j else 0.4 for j in range(4)] for i in range(4)],
)
ORDER = [0, 2, 3, 4, 5, 1]


def q(x, p):
    return torch.quantile(x.double().cpu(), p).item() if x.numel() else float("nan")


def summary(name, x):
    print(f"{name}: n={x.numel()} q50={q(x, .5):.2e} q99={q(x, .99):.2e} "
          f"q999={q(x, .999):.2e} max={x.max().item() if x.numel() else float('nan'):.2e}")


def run(cuts, label):
    print(f"\n===== {label}")
    mapping = ms.ColorOrderedMapping(ORDER, **cuts)
    g = torch.Generator().manual_seed(11)
    r = torch.rand((N, mapping.random_dim()), generator=g, dtype=torch.float64)
    d = torch.randint(0, 2, (N, mapping.discrete_dim()), generator=g, dtype=torch.int32)
    inputs = [r[:, i].contiguous() for i in range(r.shape[1])]
    inputs += [d[:, j].contiguous() for j in range(d.shape[1])]
    cond = [torch.full((N,), 1000.0, dtype=torch.float64)] + [
        torch.zeros(N, dtype=torch.float64)
    ] * 4
    res = {}
    for dev in ("cpu", GPU):
        inp = [t.to(dev) for t in inputs]
        cnd = [t.to(dev) for t in cond]
        *mom, det = mapping.map_forward(inp, cnd)
        res[dev] = ([m.cpu() for m in mom], det.cpu(), inp, cnd)
    (mc, dc, _, cc), (mg, dg, _, cg) = res["cpu"], res[GPU]
    ok = (dc > 0) & (dg > 0)
    print(f"det>0: cpu {(dc > 0).sum().item()} cuda {(dg > 0).sum().item()} both {ok.sum().item()}")
    summary("forward det rel diff cuda vs cpu", ((dg - dc).abs() / dc.abs())[ok])
    pc, pg = torch.stack(mc, 1), torch.stack(mg, 1)
    summary("forward momenta max abs diff / E_cm", ((pg - pc).abs().amax((1, 2)) / 1000.0)[ok])

    def inverse(mom, dev):
        out = mapping.map_inverse([m.to(dev) for m in mom], [c.to(dev) for c in cond])
        return torch.stack(out[: mapping.random_dim()], 1).cpu(), out[-1].cpu()

    for mom_name, mom in (("cpu", mc), ("gpu", mg)):
        for dev in ("cpu", GPU):
            r_inv, det_inv = inverse(mom, dev)
            det_fwd = dc if mom_name == "cpu" else dg
            rt = (det_fwd * det_inv - 1).abs()[ok]
            rerr = (r_inv - r).abs().amax(1)[ok]
            summary(f"round trip |det*det_inv-1|, momenta {mom_name}, inverse {dev}", rt)
            print(f"   frac r_err>1e-6: {(rerr > 1e-6).double().mean().item():.2e}")
    # worst cuda round-trip points: which random dims are off
    r_inv, det_inv = inverse(mg, GPU)
    rt = (dg * det_inv - 1).abs()
    rt[~ok] = 0
    worst = torch.argsort(rt, descending=True)[:10]
    torch.set_printoptions(precision=3, linewidth=200, sci_mode=True)
    print("worst cuda points: rt, |det_cuda/det_cpu-1|, per-dim |r_inv-r|")
    for i in worst.tolist():
        print(f"  {rt[i].item():.2e} {abs(dg[i] / dc[i] - 1).item():.2e} {(r_inv[i] - r[i]).abs()}")
    print("  r of worst:", r[worst[:3]])


run({}, "no cuts")
run(CUTS, "with cuts")
run(dict(pt_min=[20.0] * 4), "pt_min only")
run(dict(m_inv_min=CUTS["m_inv_min"]), "m_inv_min only")
run(dict(dr_min=CUTS["dr_min"]), "dr_min only")
