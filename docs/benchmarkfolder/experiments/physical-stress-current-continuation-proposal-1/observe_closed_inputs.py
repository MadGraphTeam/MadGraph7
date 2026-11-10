"""Explicit root-authorized, read-only post-closure input observation.

This writes a new audit observation only. It does not admit execution or invent
the future terminal capsule hash. Importing it never reads any execution root.
"""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def observe(controller, *, root_terminal_observed):
    r = controller; links = r.links
    r.require(root_terminal_observed is True, 'Root terminal confirmation required before any closed-root read')
    stage = r.read(r.OLD_REPORT / 'stage-run.json')
    r.require(stage['status'] == 'failed' and stage['exception'] == "TimeoutError('Lock wait expired before execution')",
              'Unexpected original terminal stage')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Enumerate only the six explicit service directory handles from the
    # historical generated graph, then their declared directory aliases.
    roots = set(); masters = {}; references = {}

    def bind(path, checksum, origin):
        canonical = str(Path(path).resolve()); observed = r.sha(canonical)
        r.require(observed == checksum, 'Historically used leaf changed: ' + str(path))
        references.setdefault(canonical, {'observed_sha256': observed, 'all_recorded_bindings_match': True, 'references': []})['references'].append(
            {'logical_path': str(path), 'sha256': checksum, 'origin': origin})

    for entry in r.PLAN['cases']:
        for source in ('baseline', 'candidate'):
            name = source + '/' + entry['case']['id']; master = r.OLD_WORK / 'exports' / name
            snapshot = links.tree_snapshot(master); masters[name] = snapshot
            for item in snapshot['entries'].values():
                if item['kind'] == 'symlink' and item['resolved_type'] == 'directory' and not Path(item['resolved']).is_relative_to(master):
                    roots.add(item['resolved'])
            build = r.base.build_record(source, entry)
            preflight_path = r.OLD_REPORT / 'builds' / (source + '-' + entry['case']['id']) / 'preflight.json'
            preflight = r.read(preflight_path)
            r.require(preflight['build_sha256'] == r.sha(preflight_path.parent / 'build.json'), 'Build/preflight identity differs')
            for path, checksum in build['bound_inputs_before_after'].items(): bind(path, checksum, name + '/build')
            for path, checksum in preflight['runtime']['files'].items(): bind(path, checksum, name + '/preflight-runtime')
    r.require(len(roots) == 6, 'Expected exactly six explicit external directory handles')
    envelopes = {root: links.tree_snapshot(root) for root in sorted(roots)}; aliases = {}
    for inventory in list(envelopes.values()):
        for item in inventory['entries'].values():
            if item['kind'] == 'symlink' and item['resolved_type'] == 'directory' and item['resolved'] not in roots:
                target = item['resolved']
                aliases[target] = links.tree_snapshot(target)
    observations = {}; transitive = {}
    for record in references.values():
        for reference in record['references']:
            logical = reference['logical_path']; observations[logical] = links._describe(logical)
    for inventory in list(masters.values()) + list(envelopes.values()) + list(aliases.values()):
        for name, item in inventory['entries'].items():
            if item['kind'] == 'symlink': observations[str(Path(inventory['root']) / name)] = item
    for item in list(observations.values()):
        for hop in item.get('symlink_chain', []):
            path = hop['logical_path']; transitive[path] = {'logical_path': path, **links._describe(path)}
    # This is the helper's mechanical external-graph schema, with no CMS
    # numerical observations or unrelated CMS master inventories inherited.
    graph = {'schema': 'cms-replay-link-closure-audit-v1', 'external_directory_envelopes': envelopes,
             'external_directory_alias_target_envelopes': aliases, 'historical_leaf_bindings': references,
             'logical_bound_leaf_observations': observations, 'transitive_symlink_nodes': transitive}
    links.verify_external_graph(graph)
    for snapshot in masters.values(): links.verify_snapshot(snapshot)
    r.base.frozen()
    closed_report = links.tree_snapshot(r.OLD_REPORT)
    return {'schema': 'physical-stress-closed-input-audit-v1', 'observed_only_after_root_terminal': True,
            'root_terminal': {'session_id': 29078, 'exit_code': 1}, 'ancestor_index_sha256': None,
            'observation_started_utc': started, 'observation_finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'ambient_membership_claim': 'new post-closure observation; not historical build-time freezing',
            'historically_used_leaves_claim': 'every original build/preflight binding matches the current canonical leaf',
            'masters': masters, 'external_graph': graph, 'closed_report': closed_report,
            'executed_compilers_loaders_mass_probes_or_workers': 0, 'execution_admission': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-observed-terminal-session', type=int, required=True)
    parser.add_argument('--root-observed-exit-code', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output = args.output.absolute()
    r = load('stress5_closed_observation_controller', HERE / 'run.py')
    r.require(Path.cwd() == r.ROOT, 'Run from repository root')
    r.require(args.root_observed_terminal_session == 29078 and args.root_observed_exit_code == 1, 'Unexpected terminal confirmation')
    r.require(not args.output.exists() and args.output.parent == HERE, 'Fresh observation inside proposal namespace required')
    result = observe(r, root_terminal_observed=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'observation': str(args.output), 'sha256': r.sha(args.output), 'masters': len(result['masters']),
        'historically_bound_canonical_leaves': len(result['external_graph']['historical_leaf_bindings']),
        'external_directory_envelopes': len(result['external_graph']['external_directory_envelopes']),
        'execution_admission': False}, indent=2))


if __name__ == '__main__': main()
