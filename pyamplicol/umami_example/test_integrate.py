"""Integrate a generated UMAMI provider with MadSpace and VEGAS/MadNIS.

Adapted from R. Frederix and T. Vitos's AmpliCol madspace_interface example:
https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_integrate.py
Its seven-section structure, channel integration and optional MadNIS/PDF workflow
are retained. README.md describes the pyAmpliCol and current-MadSpace adaptations.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import madspace as ms
import torch
import torch.nn as nn
from _metadata import load_metadata


# =============================================================================
# 1. Config / cuts
# =============================================================================
@dataclass
class Cuts:
    pt_min: float = 30.0
    eta_max: float = 6.0
    dr_min: float = 0.4
    # None means all outgoing species; no built-in particle classification.
    pdgs: list[int] | None = None


@dataclass
class Config:
    sqrts: float = 1000.0
    pdfset: str = "NNPDF23_nlo_as_0119_qed"
    pdf_dir: Path | None = None
    scale_ref: float | None = None
    t_invariant_power: float = 0.3
    cuts: Cuts = field(default_factory=Cuts)


# =============================================================================
# 2. Channel loader (generated converted metadata)
# =============================================================================
@dataclass
class Channel:
    index: int
    color_order: list[int]
    init_states: list[tuple[int, int]]
    pdg_final: list[int]
    masses: list[float]
    processes: list[list[tuple[float, int, int, tuple]]] = field(default_factory=list)


def load_channels(path, provider=None):
    data = load_metadata(path, provider)
    channels = []
    for ci, ch in enumerate(data["channels"]):
        inits, finals, procs = set(), [], []
        for proc in ch["processes"]:
            partners = proc.get("multichannels", [ci])
            if partners != [ci]:
                raise ValueError(
                    "this example requires one nonduplicated map per contribution"
                )
            cfgs = []
            for me in proc["matrix_elements"]:
                pdg = list(data["pdg_ids"][me["pdg_ids"]])
                factor = float(me["factor"])
                cfgs.append((factor, pdg[0], pdg[1], tuple(pdg[2:])))
                inits.add((pdg[0], pdg[1]))
                if not finals:
                    finals = pdg[2:]
            procs.append(cfgs)
        channels.append(
            Channel(
                ci,
                list(ch["phasespace_order"]),
                sorted(inits),
                finals,
                list(data["provider"]["masses"]),
                procs,
            )
        )
    return channels, data["provider"]["particle_count"] - 2, data


# =============================================================================
# 3. Cuts -> ms.Cuts (applied INSIDE PhaseSpaceMapping)
# =============================================================================
def build_ms_cuts(cfg: Config, ch: Channel):
    c = cfg.cuts
    a, b = ch.init_states[0]
    pids = [a, b, *ch.pdg_final]
    selected = c.pdgs if c.pdgs is not None else sorted(set(ch.pdg_final))
    # A map is shared only when every physical flavour gets the same cut mask.
    expected = [p in selected for p in ch.pdg_final]
    for configs in ch.processes:
        for _factor, _a, _b, final in configs:
            actual_selected = c.pdgs if c.pdgs is not None else set(final)
            if [p in actual_selected for p in final] != expected:
                raise ValueError(
                    "flavour-dependent cuts require separate phase-space maps"
                )
    obs_type = ms.Observable
    items = []

    def add(obs, **kwargs):
        items.append(ms.CutItem(obs_type(pids, obs, [selected]), **kwargs))

    if c.pt_min > 0:
        add(obs_type.obs_pt, min=c.pt_min)
    if c.eta_max > 0:
        add(obs_type.obs_eta_abs, max=c.eta_max)
    if c.dr_min > 0:
        add(obs_type.obs_delta_r, min=c.dr_min)
    return ms.Cuts(items) if items else None


# =============================================================================
# 4. PDFs & alpha_s + matrix element (MadSpace primitives, train.py style)
# =============================================================================
def load_pdf_and_coupling(cfg: Config, ctx):
    if cfg.pdf_dir is None:
        import lhapdf

        lhapdf.setVerbosity(0)
        candidates = [Path(directory) / cfg.pdfset for directory in lhapdf.paths()]
        base = next((path for path in candidates if path.is_dir()), None)
        if base is None:
            raise FileNotFoundError(f"PDF set {cfg.pdfset!r} is not installed")
    else:
        base = cfg.pdf_dir
    pdf_grid = ms.PdfGrid(str(base / f"{cfg.pdfset}_0000.dat"))
    pdf_grid.initialize_globals(ctx)
    ag = ms.AlphaSGrid(str(base / f"{cfg.pdfset}.info"))
    run_coupl = ms.RunningCoupling(ag)
    ag.initialize_globals(ctx)
    return pdf_grid, run_coupl


def load_matrix_element(ctx, library, artifact, use_running_alpha_s=False):
    api = ctx.load_matrix_element(
        str(Path(library).resolve()), str(Path(artifact).resolve())
    )
    inputs = [ms.MatrixElement.momenta_in]
    # No-PDF default deliberately omits alpha_s: UMAMI uses artifact defaults.
    if use_running_alpha_s:
        inputs.append(ms.MatrixElement.alpha_s_in)
    inputs += [
        ms.MatrixElement.flavor_in,
        ms.MatrixElement.channel_in,
        ms.MatrixElement.random_helicity_in,
    ]
    return ms.MatrixElement(
        api,
        inputs,
        [ms.MatrixElement.matrix_element_out, ms.MatrixElement.helicity_index_out],
    )


# =============================================================================
# 5. One color-ordered channel: randoms -> weight
# =============================================================================
class ColorOrderedChannel:
    """One independent additive contribution, with an optional PDF convolution."""

    def __init__(
        self,
        ch: Channel,
        n_out,
        cfg: Config,
        ctx,
        me_func=None,
        run_coupl=None,
        pdf_grid=None,
    ):
        self.ch, self.cfg, self.ctx = ch, cfg, ctx
        self.psmap = ms.PhaseSpaceMapping(
            ch.masses,
            cfg.sqrts,
            leptonic=pdf_grid is None,
            invariant_power=cfg.t_invariant_power,
            mode=ms.PhaseSpaceMapping.color_ordered,
            color_order=ch.color_order,
            cuts=build_ms_cuts(cfg, ch),
        )
        self.random_dim = self.psmap.random_dim()
        self.discrete_dim = self.psmap.discrete_dim()
        self.dim = self.random_dim
        self.cfg_proc, self.cfg_factor, pid_options = [], [], []
        for iproc, cfgs in enumerate(ch.processes):
            for fac, a, b, final in cfgs:
                self.cfg_proc.append(iproc)
                self.cfg_factor.append(fac)
                pid_options.append([a, b, *final])
        self.n_config = len(pid_options)
        self.dcs = None
        if me_func is not None:
            escale = None
            if run_coupl is not None:
                escale = ms.EnergyScale(
                    2 + n_out,
                    ms.EnergyScale.half_transverse_mass
                    if cfg.scale_ref is None
                    else float(cfg.scale_ref),
                )
            # UMAMI already supplies spin/colour averaging and identical-particle
            # normalization; DCS adds only flux, optional PDFs and the pb conversion.
            self.dcs = ms.DifferentialCrossSection(
                me_func,
                cfg.sqrts,
                run_coupl,
                escale,
                pid_options,
                pdf_grid,
                pdf_grid,
                True,
                False,
            )

    def weight(self, r, disc=None):
        """Keep upstream's ownership of discrete phase-space branch weights."""
        dev, dt = r.device, r.dtype
        mult = 1.0
        if self.discrete_dim:
            if disc is None:
                disc = torch.randint(
                    0, 2, (r.shape[0], self.discrete_dim), device=dev, dtype=torch.int32
                )
                mult = float(2**self.discrete_dim)
            else:
                disc = disc.to(torch.int32)
            inputs = [r, disc]
        else:
            inputs = [r]
        p, x1, x2, det = self.psmap.map_forward(inputs, context=self.ctx)
        good = torch.isfinite(det) & (det > 0)
        out = torch.zeros(r.shape[0], dtype=det.dtype, device=dev)
        if self.dcs is None:
            return torch.where(good, det * mult, out)
        if not bool(good.any()):
            return out
        # Fixed partonic energy returns singleton x1/x2; broadcast before masking.
        x1, x2 = x1.expand(r.shape[0]), x2.expand(r.shape[0])
        pg, x1g, x2g, detg = p[good], x1[good], x2[good], det[good]
        ng = len(detg)
        chan_id = torch.full((ng,), self.ch.index, device=dev, dtype=torch.int32)
        total = torch.zeros(ng, dtype=detg.dtype, device=dev)
        for c in range(self.n_config):
            flavor = torch.full((ng,), self.cfg_proc[c], device=dev, dtype=torch.int32)
            pdf_id = torch.full((ng,), c, device=dev, dtype=torch.int32)
            rnd_hel = torch.rand(ng, device=dev, dtype=dt)
            res = self.dcs(
                pg.contiguous(),
                flavor,
                chan_id,
                rnd_hel,
                x1g,
                x2g,
                pdf_id,
                context=self.ctx,
            )
            dxs = res[0] if isinstance(res, (tuple, list)) else res
            if not bool(torch.isfinite(dxs).all()):
                raise FloatingPointError("nonfinite UMAMI differential cross section")
            total = total + dxs * self.cfg_factor[c]
        out[good] = total * detg * mult
        return out


# =============================================================================
# 6. Per-ordering integration: one independent integrator per colour ordering
# =============================================================================
def integrate_per_channel(
    channels,
    n_out,
    cfg,
    ctx,
    me_library=None,
    param_card="",
    use_pdf=False,
    batch_size=256,
    epochs=200,
    n_points=2048,
    seed=0,
    vegas_only=True,
    learn_discrete=False,
    training_points=(256, 512),
):
    from madnis.integrator import (
        Integrand,
        Integrator,
        VegasPreTraining,
        stratified_variance_softclip,
    )

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(seed)
    me_func = (
        None
        if me_library is None
        else load_matrix_element(
            ctx, me_library, param_card, use_running_alpha_s=use_pdf
        )
    )
    pdf_grid = run_coupl = None
    if use_pdf and me_func is not None:
        pdf_grid, run_coupl = load_pdf_and_coupling(cfg, ctx)

    results = []
    total, var = 0.0, 0.0
    for c in channels:
        ch = ColorOrderedChannel(c, n_out, cfg, ctx, me_func, run_coupl, pdf_grid)
        dd = ch.discrete_dim
        if learn_discrete and dd:

            def fn(x, ch=ch, nd=dd):
                return ch.weight(
                    x[:, nd:].contiguous().double(), disc=x[:, :nd]
                ).double()

            integrand = Integrand(
                fn, input_dim=ch.random_dim + dd, discrete_dims=[2] * dd
            )
        else:

            def fn(x, ch=ch):
                return ch.weight(x.double()).double()

            integrand = Integrand(fn, input_dim=ch.dim)

        def vegas_callback(status, channel_index=c.index):
            print(
                f"[VEGAS channel {channel_index}] Iteration {status.step + 1}: "
                f"loss={status.variance:.6f}",
                flush=True,
            )

        def callback(status):
            if (status.step + 1) % max(1, epochs // 10) == 0:
                print(
                    f"[MadNIS] Iteration {status.step + 1:4d}: loss={status.loss:.6f}"
                )

        flow_dict = dict(layers=3, units=64, bins=10, activation=nn.LeakyReLU)
        integrator = Integrator(
            integrand,
            flow_kwargs=flow_dict,
            batch_size=batch_size,
            dtype=torch.float64,
            optimizer=lambda p: torch.optim.Adam(p, lr=1e-3),
            scheduler=lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(
                o, T_max=epochs
            ),
            loss=stratified_variance_softclip,
        )
        vegas = VegasPreTraining(integrator, bins=16, damping=0.7)
        vegas.train(list(training_points), callback=vegas_callback)
        if vegas_only:
            Ii, ei = vegas.integrate(n_points)
        else:
            vegas.initialize_integrator()
            integrator.train(epochs, callback=callback)
            Ii, ei = integrator.integrate(n_points)
        if not (math.isfinite(Ii) and math.isfinite(ei) and ei >= 0):
            raise FloatingPointError(
                "integration returned invalid estimate/uncertainty"
            )
        rsd = (ei / Ii) * math.sqrt(n_points) if Ii else None
        results.append((c.index, c.color_order, Ii, ei, rsd))
        total += Ii
        var += ei * ei
        rsd_text = f"{rsd:.3f}" if rsd is not None else "undefined (zero integral)"
        print(
            f"[channel {c.index}] color_order={c.color_order}: "
            f"{Ii:.6e} +/- {ei:.2e}  RSD={rsd_text}",
            flush=True,
        )
    return total, math.sqrt(var), results


# =============================================================================
# 7. CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json", type=Path)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument(
        "--param-card", "--artifact", dest="artifact", type=Path, required=True
    )
    ap.add_argument("--provider")
    ap.add_argument(
        "--pdf", action="store_true", help="optional hadronic run; needs PDF data"
    )
    ap.add_argument("--no-pdf", dest="pdf", action="store_false")
    ap.add_argument("--pdfset", default=Config.pdfset)
    ap.add_argument("--pdf-dir", type=Path)
    ap.add_argument("--sqrts", type=float, default=1000.0)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--n", type=int, default=2048, help="integration points per contribution"
    )
    ap.add_argument("--training", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--madnis", action="store_true", help="train a neural flow after VEGAS"
    )
    ap.add_argument("--vegas-only", dest="madnis", action="store_false")
    ap.add_argument("--learn-discrete", action="store_true")
    ap.add_argument("--pt-min", type=float, default=30.0)
    ap.add_argument("--eta-max", type=float, default=6.0)
    ap.add_argument("--dr-min", type=float, default=0.4)
    ap.add_argument("--cut-pdg", type=int, action="append")
    ap.add_argument(
        "--selftest", action="store_true", help="phase-space volume, without ME or PDFs"
    )
    ap.add_argument("--result", type=Path, help="optional machine-readable result")
    args = ap.parse_args()
    if min(args.n, args.batch_size, args.epochs, *args.training) < 2:
        ap.error("sample counts, batch size and epochs must be at least two")
    channels, n_out, data = load_channels(args.json, args.provider)
    if data["grouping"]["mode"] == "flavour_blind_observables" and args.cut_pdg:
        ap.error("compressed integration assumes flavour-blind cuts; omit --cut-pdg")
    if args.pdf and data["provider"]["alpha_s_parameter"] is None:
        ap.error("this provider has no unambiguous running-alpha_s input")
    cfg = Config(
        sqrts=args.sqrts,
        pdfset=args.pdfset,
        pdf_dir=args.pdf_dir,
        scale_ref=args.scale,
        cuts=Cuts(args.pt_min, args.eta_max, args.dr_min, args.cut_pdg),
    )
    print(
        f"Loaded {len(channels)} independent contributions, n_out={n_out}; "
        f"{data['provider']['color_accuracy']} colour, "
        f"{data['grouping']['mode']} grouping"
    )
    ctx = ms.Context(1)
    torch.set_num_threads(1)
    t0 = time.time()
    sigma, err, per_channel = integrate_per_channel(
        channels,
        n_out,
        cfg,
        ctx,
        None if args.selftest else args.library,
        args.artifact,
        args.pdf and not args.selftest,
        args.batch_size,
        args.epochs,
        args.n,
        args.seed,
        not args.madnis,
        args.learn_discrete,
        args.training,
    )
    if not args.selftest and sigma <= 0:
        raise AssertionError("the demonstration must produce a positive cross section")
    result = dict(
        integral=sigma,
        uncertainty=err,
        unit="" if args.selftest else "pb",
        points_per_channel=args.n,
        channels=len(channels),
        elapsed_seconds=time.time() - t0,
        per_channel=per_channel,
        provider=data["provider"]["id"],
        seed=args.seed,
    )
    print(
        f"\n{'MadNIS' if args.madnis else 'VEGAS'} integral: "
        f"{sigma:.6e} +/- {err:.2e} {result['unit']} "
        f"({result['elapsed_seconds']:.1f}s)"
    )
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
