#!/usr/bin/env python3
"""Bind the source-only trace design; never compile or execute native code."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SP = Path('.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/P0_uux_zgg')


def record(path):
    data = (ROOT / path).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    paths = [SP / name for name in (
        'CT_interface.f', 'TIRUST_interface.f', 'TIR_interface.f',
        'loop_CT_calls_1.f', 'mp_helas_calls_ampb_1.f', 'helas_calls_ampb_1.f',
        'loop_matrix.f', 'mp_compute_loop_coefs.f', 'improve_ps.f',
    )]
    paths.extend(Path(p) for p in (
        'docs/benchmarkfolder/experiments/physical-stress/points/qcd_n3_uux_zgg/selected.dat',
        'docs/benchmarkfolder/experiments/physical-stress/inputs/candidate/qcd_n3_uux_zgg/param_card.dat',
        'docs/benchmarkfolder/experiments/physical-stress/runs/qcd_n3_uux_zgg/candidate-8-auto/MadLoopParams.dat',
        'docs/benchmarkfolder/experiments/physical-stress/runs/qcd_n3_uux_zgg/candidate-9-auto/MadLoopParams.dat',
        'docs/benchmarkfolder/experiments/qcd-born-cancellation-audit/evidence-index.json',
        'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/evidence-index.json',
        'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/amplitude-native-1/adjudication.json',
    ))
    anchor_specs = {
        'CT_interface.f': [
            'SUBROUTINE ML5_0_MP_BUILD_KINEMATIC_MATRIX',
            'SUBROUTINE ML5_0_MP_NINJARS_LOOP',
            'SUBROUTINE ML5_0_LOOP_2(',
        ],
        'TIRUST_interface.f': [
            'SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC',
            'SUBROUTINE ML5_0_MP_TIRUST_LOOP',
        ],
        'TIR_interface.f': ['SUBROUTINE ML5_0_MP_SWITCH_ORDER'],
        'loop_matrix.f': ['SUBROUTINE ML5_0_SET_MP_PS'],
    }
    anchors = []
    for name, needles in anchor_specs.items():
        lines = (ROOT / SP / name).read_text().splitlines()
        for needle in needles:
            matches = [i for i, line in enumerate(lines, 1) if needle in line]
            if len(matches) != 1:
                raise SystemExit(f'Anchor must be unique: {name}: {needle}: {matches}')
            anchors.append({'path': str(SP / name), 'line': matches[0], 'text': needle})
    out = {
        'schema': 1, 'status': 'source-only observational proposal',
        'native_executed': False, 'production_modified': False,
        'source_inputs': {str(path): record(path) for path in paths},
        'source_anchors': anchors,
        'limitations': [
            'No observed per-group clamp records exist yet.',
            'Original-input Born arithmetic does not replace actual repaired QP loop kinematics.',
            'A snap or S/q mismatch alone does not establish the rejected amplitude cause.',
        ],
    }
    (HERE / 'source-bindings.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'source_files_bound': len(paths), 'anchors': len(anchors), 'native_executed': False}))


if __name__ == '__main__':
    main()
