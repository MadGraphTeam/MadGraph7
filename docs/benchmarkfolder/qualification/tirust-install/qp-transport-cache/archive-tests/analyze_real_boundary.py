"""Classify the already executed single paired native boundary diagnostic."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
REPORT = ROOT / 'docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/archive-tests'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path):
    frames = []
    final = None
    for line in path.read_text().splitlines():
        key, _, value = line.partition(' ')
        if key == 'CASE':
            frames.append({'case': value})
        elif key == 'FINALIZE_STATUS':
            final = int(value)
        else:
            require(bool(frames), 'Record before first case')
            require(key not in frames[-1], 'Duplicate diagnostic field')
            frames[-1][key] = value.strip()
    require(len(frames) == 3 and [f['case'] for f in frames] ==
            ['ordinary-seed', 'large-mass', 'large-mass'] and final == 0,
            'Missing/incomplete diagnostic frames')
    for frame in frames:
        frame['PUBLIC_STATUS'] = int(frame['PUBLIC_STATUS'])
        frame['CACHE_STATUS'] = int(frame['CACHE_STATUS'])
        frame['QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'] = [
            int(x) for x in frame['QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'].split()]
        frame['IEEE_INVALID_DIVZERO_OVERFLOW_UNDERFLOW_INEXACT'] = frame[
            'IEEE_INVALID_DIVZERO_OVERFLOW_UNDERFLOW_INEXACT'].split()
    return frames


results = json.loads((REPORT / 'results.json').read_text())
frames = {}
for label in ('old095e', 'new059b'):
    result = results['results'][label + '-real-provider']
    require(result['returncode'] == 0 and not result['timed_out'], 'Optional native run incomplete')
    path = REPORT / (label + '-real-provider.log')
    command = [item for item in results['commands'] if item['log'].endswith(path.name)]
    require(len(command) == 1 and command[0]['native_shared_lock'] and
            command[0]['timeout_seconds_after_lock'] == 180 and
            command[0]['log_sha256'] == sha(path), 'Native execution/log provenance mismatch')
    frames[label] = parse(path)
    seed, first, repeat = frames[label]
    require(seed['PUBLIC_STATUS'] == seed['CACHE_STATUS'] == 0 and
            seed['CLASSIFICATION'] == 'public-success' and
            seed['QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'][0:3] == [0, 1, 1],
            'Ordinary valid cache entry was not seeded')
    for frame in (first, repeat):
        require(frame['PUBLIC_STATUS'] == 2 and frame['CACHE_STATUS'] == 0 and
                frame['CLASSIFICATION'] == 'public-transport-rejection' and
                frame['ERROR_MESSAGE'] == 'TIRust returned an invalid or nonfinite binary128 output record.' and
                frame['OUTPUTS_ALL_ZERO'] == 'T', 'Native run did not reach the required public rejection')
        require(frame['IEEE_INVALID_DIVZERO_OVERFLOW_UNDERFLOW_INEXACT'] == ['F', 'F', 'T', 'F', 'T'],
                'Unexpected IEEE flags')
old = frames['old095e']
new = frames['new059b']
for left, right in zip(old, new):
    require({k: v for k, v in left.items() if k != 'QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'} ==
            {k: v for k, v in right.items() if k != 'QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'},
            'Input/status/output/flag observations changed beyond cache invalidation')
stats = 'QP_CACHE_HITS_MISSES_ENTRIES_BYTES_REFUSED_BYPASSED'
require(old[1][stats] == [0, 2, 2, 2592, 0, 0] and old[2][stats] == [1, 2, 2, 2592, 0, 0],
        'Old archive does not demonstrate retained-entry hit')
require(new[1][stats] == new[2][stats] == [0] * 6, 'Fixed archive retained numerical cache state')
summary = {'schema_version': 1, 'state': 'complete', 'passed': True, 'performance_eligible': False,
    'execution_count_per_archive': 1, 'calls_per_execution': 3, 'timeout_seconds_per_execution': 180,
    'native_boundary_reached': True, 'old_retained_entry_repeat_hit_observed': True,
    'fixed_cache_empty_after_both_rejections': True,
    'inputs_public_status_zero_outputs_and_ieee_flags_identical': True,
    'clear_resets_counters_limit': 'Zero counters after clear do not by themselves count re-evaluations; '
                                   'the separate independent mock-provider test proves retry misses/re-evaluation.',
    'trap_policy': 'Only this diagnostic disables traps and restores original IEEE settings on exit.',
    'identities': results['identities'], 'frames': frames,
    'source_sha256': sha(Path(__file__)),
    'input_artifacts': {name: sha(REPORT / name) for name in
        ('results.json', 'real-provider.f90', 'old095e-real-provider.log', 'new059b-real-provider.log')}}
(REPORT / 'real-boundary-analysis.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps({k: v for k, v in summary.items() if k not in ('frames', 'identities')}, indent=2))
