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
    # Bind the explicit context: v0.2.1 convenience calls use the global context.
    ps_runtime = ms.FunctionRuntime(psmap.forward_function(), ctx)
    p_ext, _x1, _x2, det = ps_runtime(np.random.rand(n, psmap.random_dim()))
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
    # The convenience overload queries diagram_count even without diagram
    # outputs. This recursion provider intentionally has no diagram metadata.
    func = ms.MatrixElement(
        me.index(),
        me.particle_count(),
        inputs,
        [ms.MatrixElement.matrix_element_out, ms.MatrixElement.helicity_index_out],
        diagram_count=0,
    )
    evaluator = ms.FunctionRuntime(func.function(), ctx)

    from pyamplicol import Runtime

    runtimes = {}
    physical_sums = {}
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
            amp2, hel = evaluator(*buffers, flavors, channels, rnd_hel)

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
            for member in process["members"]:
                # Every mode groups identical-particle integration orbits.
                # Compose the maps to evaluate each member at the original
                # labelled point, not at the representative's labelled point.
                mapped_points = np.empty_like(p_ext)
                mapped_points[:, spec["permutation"], :] = p_ext[
                    :, member["runtime"]["permutation"], :
                ]
                member_buffers = [mapped_points]
                if alpha_s is not None:
                    member_buffers.append(alpha_s)
                member_amp2, _member_hel = evaluator(
                    *member_buffers, flavors, channels, rnd_hel
                )
                original_id = member["process_id"]
                if original_id not in runtimes:
                    runtimes[original_id] = Runtime.load(artifact, process=original_id)
                physical_runtime = runtimes[original_id]
                physical_color = member["color_id"]
                selected = None if physical_color is None else (physical_color,)
                physical_native = []
                for i, point in enumerate(p_ext):
                    if alpha_s is not None:
                        physical_runtime.set_model_parameter(
                            alpha_name, float(alpha_s[i])
                        )
                    value = physical_runtime.evaluate((point,), color_flows=selected)[0]
                    physical_native.append(float(complex(value).real))
                physical_native = np.asarray(physical_native)
                np.testing.assert_allclose(
                    member_amp2, physical_native, rtol=2e-10, atol=1e-25
                )
                if not np.all(np.isfinite(member_amp2)) or not np.all(member_amp2 >= 0):
                    raise AssertionError("invalid physical-member matrix element")
                scale = np.maximum(np.abs(physical_native), np.finfo(float).tiny)
                maximum_error = max(
                    maximum_error,
                    float(np.max(np.abs(member_amp2 - physical_native) / scale)),
                )
                checked += n
                if data["provider"]["color_accuracy"] == "lc":
                    total, colors = physical_sums.setdefault(
                        original_id, (np.zeros(n), [])
                    )
                    # Only the physical member weight belongs in a pointwise
                    # sum; matrix_elements factors are integration weights.
                    total += member_amp2 * float(member["factor"])
                    colors.append(physical_color)
    reconstructed = 0
    for process_id, (total, colors) in physical_sums.items():
        if process_id not in runtimes:
            runtimes[process_id] = Runtime.load(artifact, process=process_id)
        runtime = runtimes[process_id]
        if runtime.physics.color_coverage != "complete":
            continue
        if len(colors) != len(set(colors)) or set(colors) != set(
            runtime.physics.color_flow_ids
        ):
            raise AssertionError(
                "UMAMI contributions do not cover physical LC flows once"
            )
        default = (
            None
            if alpha_name is None
            else next(
                p.default_real
                for p in runtime.physics.model_parameters
                if p.name == alpha_name
            )
        )
        native_total = []
        alpha_factors = np.linspace(0.9, 1.1, n)
        for i, point in enumerate(p_ext):
            if default is not None:
                runtime.set_model_parameter(
                    alpha_name, default * float(alpha_factors[i])
                )
            native_total.append(float(complex(runtime.evaluate((point,))[0]).real))
        np.testing.assert_allclose(total, native_total, rtol=2e-10, atol=1e-25)
        reconstructed += n
    print(
        f"MadSpace/UMAMI: {checked} values match native pyAmpliCol; "
        f"max relative difference {maximum_error:.3g}"
    )
    if reconstructed:
        print(f"LC channel sums reconstruct {reconstructed} native total values")


if __name__ == "__main__":
    main()
