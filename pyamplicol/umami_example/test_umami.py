"""Batched MadSpace loading example, with native pyAmpliCol parity checks.

Adapted from R. Frederix and T. Vitos's AmpliCol madspace_interface example:
https://github.com/rikkert-frederix/AmpliCol/blob/madspace_interface/test_umami.py
See README.md for the small, necessary changes to that workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import madspace as ms
import numpy as np
from _metadata import load_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--metadata", type=Path)
    ap.add_argument("--provider")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sqrts", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n must be positive")
    artifact = args.artifact.resolve()
    data = load_metadata(
        args.metadata or artifact / "API/umami/metadata.json", args.provider
    )

    # Same load -> phase space -> batched MatrixElement sequence as upstream.
    np.random.seed(args.seed)
    ctx = ms.Context(1)
    me = ctx.load_matrix_element(str(args.library.resolve()), str(artifact))
    n = args.n
    psmap = ms.PhaseSpaceMapping(data["provider"]["masses"], args.sqrts, leptonic=True)
    p_ext, _x1, _x2, det = psmap.map_forward(
        [np.random.rand(n, psmap.random_dim())], context=ctx
    )
    if not np.all(np.isfinite(det) & (det > 0)):
        raise AssertionError("phase-space mapping returned a degenerate point")
    inputs = [ms.MatrixElement.momenta_in]
    alpha_name = data["provider"]["alpha_s_parameter"]
    if alpha_name is not None:
        inputs.append(ms.MatrixElement.alpha_s_in)
    inputs += [
        ms.MatrixElement.flavor_in,
        ms.MatrixElement.channel_in,
        ms.MatrixElement.random_helicity_in,
    ]
    func = ms.MatrixElement(
        me,
        inputs,
        [ms.MatrixElement.matrix_element_out, ms.MatrixElement.helicity_index_out],
    )

    from pyamplicol import Runtime

    runtimes = {}
    checked = 0
    maximum_error = 0.0
    for channel_index, channel in enumerate(data["channels"]):
        for flavor_index, process in enumerate(channel["processes"]):
            spec = process["runtime"]
            process_id = spec["process_id"]
            if process_id not in runtimes:
                runtimes[process_id] = Runtime.load(artifact, process=process_id)
            runtime = runtimes[process_id]
            buffers = [p_ext]
            alpha_s = None
            if alpha_name is not None:
                default = next(
                    p.default_real
                    for p in runtime.physics.model_parameters
                    if p.name == alpha_name
                )
                # Exercise actual dependent-coupling updates, not just a fixed alpha_s.
                alpha_s = default * np.linspace(0.9, 1.1, n)
                buffers.append(alpha_s)
            flavors = np.full(n, flavor_index, dtype=np.int32)
            channels = np.full(n, channel_index, dtype=np.int32)
            rnd_hel = np.random.rand(n)
            amp2, hel = func(*buffers, flavors, channels, rnd_hel, context=ctx)

            native = []
            color_id = spec["color_id"]
            for i, point in enumerate(p_ext):
                if alpha_s is not None:
                    runtime.set_model_parameter(alpha_name, float(alpha_s[i]))
                mapped = point[spec["permutation"]]
                value = runtime.evaluate(
                    (mapped,), color_flows=None if color_id is None else (color_id,)
                )[0]
                native.append(float(complex(value).real) * float(spec["factor"]))
            native = np.asarray(native)
            np.testing.assert_allclose(amp2, native, rtol=2e-10, atol=1e-25)
            if not np.all(np.isfinite(amp2)) or not np.all(amp2 >= 0):
                raise AssertionError("invalid squared matrix element")
            runtime_meta = data["runtime_processes"][spec["process_index"]]
            helicity_count = len(runtime_meta["helicity_ids"])
            if not np.all((hel >= 0) & (hel < helicity_count)):
                raise AssertionError("sampled helicity does not index process metadata")
            scale = np.maximum(np.abs(native), np.finfo(float).tiny)
            maximum_error = max(
                maximum_error, float(np.max(np.abs(amp2 - native) / scale))
            )
            checked += n
    print(
        f"MadSpace/UMAMI: {checked} values match native pyAmpliCol; "
        f"max relative difference {maximum_error:.3g}"
    )


if __name__ == "__main__":
    main()
