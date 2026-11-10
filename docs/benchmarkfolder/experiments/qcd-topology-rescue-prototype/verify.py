#!/usr/bin/env python3
"""Verify frozen private source variants; never compile or run numerical code."""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_record(entry):
    path = ROOT / entry['path']
    require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from MadGraph root')
    inventory = json.loads((HERE / 'evidence-index.json').read_text())
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}
    require(actual == set(inventory['files']) | {'evidence-index.json'}, 'Unexpected/missing portable files')
    for name, entry in inventory['files'].items():
        path = HERE / name
        require(not path.is_symlink() and '..' not in Path(name).parts, name)
        raw = path.read_bytes()
        require(not raw.startswith(b'\x7fELF') and path.suffix not in ('.pyc', '.o', '.mod', '.a', '.so'), name)
        require(len(raw) == entry['bytes'] and sha(path) == entry['sha256'], name)
    arm_plan = json.loads((HERE / 'arms.json').read_text())
    require(not arm_plan['compiler_executed'] and not arm_plan['native_executed']
            and not arm_plan['production_modified'], 'Source-only boundary')
    require(set(arm_plan['arms']) == {'original-b', 'topology-b', 'original-c', 'topology-c'}, 'Arm set')
    for arm in arm_plan['arms'].values():
        require(not arm['runtime_evaluated'], 'Unexecuted arm')
        for entry in arm['sources'].values():
            check_record(entry)
    for field in ('historical_control', 'historical_build', 'source_preparation', 'c_derivation', 'script'):
        check_record(arm_plan[field])
    topo = module(HERE / 'prepare_topology.py', 'qcd_topology_source_verification')
    ct = (topo.DESIGN / 'sources/CT_interface.f').read_text()
    tir = (topo.DESIGN / 'sources/TIRUST_interface.f').read_text()
    expected_ct = ct
    for old, new in (('ML5_0_MP_NINJARS_LOOP', 'QK_MP_NINJARS_CLASSIFIED'),
                     ('ML5_0_MP_TIRUST_LOOP', 'QK_MP_TIRUST_CLASSIFIED')):
        expected_ct = topo.replace_statement(expected_ct, 'CALL ' + old + '(',
            lambda s, old=old, new=new: topo.append_context(s, old, new, 'LOOPNUM,CTMODE'), expected=7)
    expected_ct += '\nC     BEGIN PRIVATE CLASSIFIED APPENDIX\n' + topo.classified_builder(ct) + '\n' + topo.classified_nrs(ct)
    expected_tir = tir + '\nC     BEGIN PRIVATE CLASSIFIED APPENDIX\n' + topo.classified_tirust(tir)
    require((HERE / 'topology-design/CT_interface.f').read_text() == expected_ct, 'Classified CT source derivation')
    require((HERE / 'topology-design/TIRUST_interface.f').read_text() == expected_tir, 'Classified TIR source derivation')
    for name in ('ML5_0_NINJA_LOOP', 'ML5_0_NINJARS_LOOP', 'ML5_0_MP_NINJA_LOOP',
                 'ML5_0_BUILD_KINEMATIC_MATRIX', 'ML5_0_MP_BUILD_KINEMATIC_MATRIX', 'ML5_0_MP_NINJARS_LOOP'):
        require(topo.routine(ct, name) == topo.routine(expected_ct, name), 'Original entry point changed: ' + name)
    for name in ('ML5_0_TIRUST_TENSOR_DIAGNOSTIC', 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC',
                 'ML5_0_TIRUSTLOOP', 'ML5_0_MP_TIRUST_LOOP'):
        require(topo.routine(tir, name) == topo.routine(expected_tir, name), 'Original diagnostic changed: ' + name)
    metadata = json.loads((HERE / 'topology-design/table-metadata.json').read_text())
    table, counts = topo.table_source(metadata['groups_by_mode'])
    require(table == (HERE / 'topology-design/shell_classes.f').read_text(), 'Table derivation')
    require(counts == metadata['counts'], 'Table counts')
    for mode, groups in metadata['groups_by_mode'].items():
        require([g['group'] for g in groups] == list(range(1, 117)), 'Group coverage')
        pairs = [p for group in groups for p in group['pairs']]
        require(len(pairs) == 477 and sum(p['classification'] == 'composite' for p in pairs) == 235,
                'Pair coverage')
        require(sum(p['target_p1_minus_p4'] for p in pairs) == 19, 'Target coverage')
    c = module(HERE / 'born-c-design/derive.py', 'qcd_born_c_source_verification')
    patch, bindings = c.artifacts()
    require(patch == (HERE / 'born-c-design/born-c-handoff.patch').read_text(), 'C patch')
    require(bindings == json.loads((HERE / 'born-c-design/source-bindings.json').read_text()), 'C bindings')
    _, sources = c.derive_sources()
    for name, text in sources.items():
        require((HERE / 'source-variants/born-c' / name).read_text() == text, 'C source derivation: ' + name)
    prepared = json.loads((HERE / 'topology-design/prepared.json').read_text())
    require(prepared['script_sha256'] == sha(HERE / 'prepare_topology.py'), 'Topology generator hash')
    for name, digest in prepared['inputs'].items():
        require(sha(ROOT / name) == digest, name)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    require(not protocol['native_authorized'] and not protocol['compiler_executed']
            and not protocol['native_executed'], 'Protocol is source-only')
    for entry in protocol['frozen_inputs'].values():
        check_record(entry)
    live_count = 0
    if args.live:
        graph = module(topo.ROUTING / 'audit_routing.py', 'qcd_live_graph_verification')
        require(graph.audit(topo.PROCESS) == json.loads((topo.ROUTING / 'routing.json').read_text()), 'Live producer graph')
        directions = module(topo.DESIGN / 'derive_routing.py', 'qcd_live_direction_verification')
        require(directions.derive() == json.loads((topo.DESIGN / 'routing-modes.json').read_text()), 'Live directions')
        for name, digest in protocol['historical_protected_inputs'].items():
            require(sha(ROOT / name) == digest, name)
            live_count += 1
    print(json.dumps({'verification': 'pass', 'portable_files': len(inventory['files']), 'source_arms': 4,
                      'groups_per_direction': 116, 'pairs_per_direction': 477,
                      'live_historical_pins': live_count, 'compiler_executed': False, 'native_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
