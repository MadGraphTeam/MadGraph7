#!/usr/bin/env python3
"""Additional offline work-weighted interpretation; never executes reducers."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def analyze_physical(raw, summary, source_sha256):
    assert raw['uniqueness_complete'] and raw['dropped_events'] == 0
    assert summary['original_off_on_exact_parity'] and summary['off_on_mpfr_flag_parity']
    rows = {p['parent']: p for p in summary['parents']}
    parents = {e['parent']: e for e in raw['events'] if e['kind'] == 'parent_begin'}
    ends = {e['id']: e for e in raw['events'] if e['kind'] == 'scalar_end'}
    requests = defaultdict(list)
    for event in raw['events']:
        if event['kind'] == 'scalar_begin': requests[event['parent']].append(event)
    observed = set()
    committed = set()
    scopes = defaultdict(Counter)
    families = defaultdict(Counter)
    failures = []
    for pid, p in parents.items():
        row = rows[pid]
        if p['synthetic']: continue
        labels = ('physical', 'ordinary' if p['point'] == 1 else 'threshold')
        good = row['bridge_status'] == row['public_status'] == 0
        for label in labels:
            scopes[label]['parents'] += 1
            scopes[label]['successful_parents' if good else 'failed_parents'] += 1
        if not good:
            failures.append({'parent': pid, 'point': p['point'], 'group': p['group'],
                             'status': row['public_status'], 'scalar_requests': len(requests[pid])})
        base = (p['context'], row['epoch'], p['finite'], p['poles'], p['check'], p['synthetic'])
        within = set()
        pending = set()
        recomputed = Counter()
        for b in requests[pid]:
            e = ends[b['id']]
            key = (*base, b['key'])
            kind = 'within_parent_repeat' if key in within else ('across_parent_repeat' if key in observed else 'new_exact_key')
            within.add(key); observed.add(key)
            valid = b['input_finite'] and e['success'] and e['output_finite']
            eligible = good and valid and p['check'] == 0 and p['cache'] == 1
            prior = eligible and key in committed
            values = {'requests':1, 'key_ns':b['key_ns'], 'kernel_ns':e['kernel_ns'],
                      kind+'_requests':1, kind+'_kernel_ns':e['kernel_ns']}
            if prior:
                values['requests_matching_prior_eligible_parent'] = 1
                values['kernel_ns_matching_prior_eligible_parent'] = e['kernel_ns']
            if e['flags_after_kernel'] != b['flags_after_key']:
                values['kernel_sticky_flag_changes'] = 1
            for label in labels: scopes[label].update(values)
            families[str(b['family'])].update(values)
            recomputed['within_parent_repeats' if kind=='within_parent_repeat' else
                       'across_parent_repeats' if kind=='across_parent_repeat' else 'new_exact_keys'] += 1
            if prior: recomputed['repeat_of_prior_successful_eligible_parent'] += 1
            if eligible: pending.add(key)
        for name, count in recomputed.items(): assert count == row['counts'].get(name, 0), (pid,name)
        # All physical requests have sufficient recorded budget under the
        # pre-existing offline capacity model. This is not an allocated store.
        assert row['counts'].get('simulated_budget_refusals',0) == 0
        committed.update(pending)
    for c in scopes.values():
        c['fraction_observed_kernel_ns_matching_prior_eligible_parent'] = c['kernel_ns_matching_prior_eligible_parent']/c['kernel_ns']
    result = {'scope':'Observed physical scalar kernel work only; no scalar reuse, value-equivalence proof or speedup claim',
              'source_sha256':source_sha256,
              'physical_scopes':{k:dict(v) for k,v in scopes.items()},
              'physical_families':{k:{'name':('tadpole','bubble','triangle','box')[int(k)],**dict(v)} for k,v in families.items()},
              'retained_geometry_failures':failures,
              'synthetic_only_counts':dict(sum((Counter(p['counts']) for p in rows.values() if p['synthetic']),Counter())),
              'parent_wall_time_recorded':False,
              'limitations':['Kernel clocks are instrumented one-shot observations under shared host load.',
                             'Parent and amplitude wall times were not recorded; process launch elapsed times are not attribution denominators.',
                             'Equal exact input keys do not establish equal returned scalar values: output values were not journaled.',
                             'Prior eligible key matches are candidate opportunities, not cache hits, allocated memory or achievable savings.',
                             'Physical input subset covers only two EW3 points and groups1-12 in CTMODE4, not n4 warmups or reverse checks.',
                             'Failed physical parents rejected in geometry before any scalar call, so this native corpus does not exercise discarding completed scalars from a subsequently failing parent.',
                             'Future reuse must preserve MPFR sticky-flag effects and independent stability-check stores.']}
    return result

def main():
    raw = json.loads((HERE/'results/on-shadow.json').read_text())
    summary = json.loads((HERE/'results/summary.json').read_text())
    source = {n:hashlib.sha256((HERE/'results'/n).read_bytes()).hexdigest()
              for n in ('on-shadow.json','summary.json','execution.json')}
    result = analyze_physical(raw, summary, source)
    (HERE/'results/physical-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['physical_scopes'],indent=2))

if __name__ == '__main__': main()
