"""Validate the prospective host-only affinity change, without numerical work."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_addendum():
    data = json.loads((HERE / 'execution-affinity.json').read_text())
    identity = data.pop('identity')
    assert hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == identity
    assert data['schema'] == 'pure-tensor-execution-affinity-v1'
    assert data['prior_cpu'] == 212 and data['execution_cpu'] == 234
    for value in data['bindings'].values():
        path = ROOT / value['path']
        assert sha(path) == value['sha256'] and path.stat().st_size == value['bytes'], path
    plan = json.loads((HERE / 'plan.json').read_text())
    assert plan['identity'] == data['plan_identity']
    assert plan['cpu'] == 212 and plan['pairs'] == 21
    survey = json.loads((HERE / 'survey-all-1.json').read_text())
    assert survey['source_sha256'] == sha(HERE / 'survey_all.py')
    assert survey['numerical_execution'] is False
    assert survey['interval_count'] == 3 and survey['interval_seconds'] == 3
    assert survey['maximum_busy_fraction'] == .05 and 207 not in survey['cpus']
    assert len(survey['samples']) == 3
    for row in survey['samples']:
        for cpu in survey['cpus']:
            ticks = row[str(cpu)]
            delta = [a - b for a, b in zip(ticks['ticks_after'], ticks['ticks_before'])]
            assert len(delta) == 8 and min(delta) >= 0 and sum(delta) > 0
            assert ticks['busy_fraction'] == 1 - (delta[3] + delta[4]) / sum(delta)
    eligible = [cpu for cpu in survey['cpus']
                if survey['topology'][str(cpu)]['thread_siblings_list'] == str(cpu)
                and all(row[str(cpu)]['busy_fraction'] <= .05 for row in survey['samples'])]
    assert survey['eligible'] == eligible
    selected = min(eligible, key=lambda cpu: (
        max(row[str(cpu)]['busy_fraction'] for row in survey['samples']),
        sum(row[str(cpu)]['busy_fraction'] for row in survey['samples']), cpu))
    assert selected == data['execution_cpu'] == 234
    original = (HERE / 'run.py').read_text()
    replacements = [
        ('from pathlib import Path\n', 'from pathlib import Path\nfrom affinity_addendum import verify_addendum\n'),
        ("choices=['capture', 'kernel']", "choices=['kernel']"),
        ('    os.sched_setaffinity(0, {212})', '    verify_addendum()\n    os.sched_setaffinity(0, {234})'),
        ('/cpu212/topology/', '/cpu234/topology/'),
        ("              'plan': binding(HERE / 'plan.json'), 'controller': binding(Path(__file__).resolve()),",
         "              'plan': binding(HERE / 'plan.json'), 'controller': binding(Path(__file__).resolve()),\n              'execution_affinity': binding(HERE / 'execution-affinity.json'),"),
    ]
    for before, after in replacements:
        assert original.count(before) == 1
        original = original.replace(before, after)
    assert original == (HERE / 'run_cpu234.py').read_text()
    return dict(data, identity=identity)


if __name__ == '__main__':
    print(json.dumps({'status': 'passed', 'identity': verify_addendum()['identity']}))
