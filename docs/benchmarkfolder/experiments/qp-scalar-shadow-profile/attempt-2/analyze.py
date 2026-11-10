#!/usr/bin/env python3
"""Audit shadow observations offline; no scalar result is ever reused."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

def analyze(data):
    if data.get('mpfr_reader_available') is not True:
        raise ValueError('MPFR flag reader unavailable; parity/opportunity comparison refused')
    parents = {}; beginnings = {}; endings = {}; epochs = defaultdict(int)
    complete = data['uniqueness_complete']
    orphan_events = 0
    for event in data['events']:
        for name,value in event.items():
            if name.startswith('flags_') or name=='mpfr_flags':
                if not isinstance(value,int) or not 0<=value<=63:
                    raise ValueError('Unavailable or incompatible MPFR flag observation')
        kind = event['kind']
        if kind not in ('lifecycle','parent_begin') and event.get('parent') not in parents:
            assert not complete, 'Orphan event in a purportedly complete journal'
            orphan_events += 1
            continue
        if kind == 'lifecycle':
            assert event['action'] in ('clear','budget_changed','destroy')
            epochs[event['context']] += 1
        elif kind == 'parent_begin':
            pid = event['parent']; assert pid not in parents
            parents[pid] = {**event,'epoch':epochs[event['context']], 'requests':[], 'notes':[]}
        elif kind == 'parent_end':
            assert 'status' not in parents[event['parent']]
            parents[event['parent']]['status'] = event['status']
        elif kind == 'store_state':
            parents[event['parent']]['store'] = event
        elif kind == 'parent_note':
            parents[event['parent']]['notes'].append(event['action'])
        elif kind == 'scalar_begin':
            sid = event['id']; assert sid not in beginnings
            assert len(bytes.fromhex(event['key'])) == event['key_bytes']
            beginnings[sid] = event
            parents[event['parent']]['requests'].append(sid)
        elif kind == 'scalar_end':
            sid = event['id']; assert sid not in endings
            endings[sid] = event
        else:
            raise ValueError('Unknown shadow event '+kind)
    if complete:
        assert set(beginnings) == set(endings)
        assert len(beginnings) == data['scalar_requests'] == data['scalar_completed']
        assert all('status' in p for p in parents.values())
    totals = Counter(); family = defaultdict(Counter); rows = []
    observed = set(); committed = {}; modeled_bytes = defaultdict(int)
    for pid,p in parents.items():
        p['public_status'] = data.get('public_status_by_parent',{}).get(str(pid),p.get('status'))
        totals['parents'] += 1
        totals['successful_parents' if p.get('status') == p['public_status'] == 0 else 'failed_or_unfinished_parents'] += 1
        if p['synthetic']: totals['synthetic_parents'] += 1
        if 'whole_tensor_cache_hit' in p['notes']: totals['whole_tensor_cache_hits'] += 1
        within = set(); pending = {}; stats = Counter()
        store = p.get('store',{})
        base = (p['context'],p['epoch'],p['finite'],p['poles'],p['check'],p['synthetic'])
        budget_namespace = (p['context'],p['epoch'])
        # Synthetic suffixes are deliberately isolated from physical hit-rate statistics.
        mode_allowed = p['check'] == 0 and p['cache'] == 1
        success = p.get('status') == p['public_status'] == 0
        for sid in p['requests']:
            b = beginnings[sid]; e = endings.get(sid)
            stats['requests_observed'] += 1
            stats['key_ns'] += b['key_ns']
            if e: stats['kernel_ns'] += e['kernel_ns']
            key = (*base,b['key'])
            if key in within: stats['within_parent_repeats'] += 1
            elif key in observed: stats['across_parent_repeats'] += 1
            else: stats['new_exact_keys'] += 1
            within.add(key); observed.add(key)
            if not e:
                stats['incomplete_scalar_events'] += 1; continue
            valid = b['input_finite'] and e['success'] and e['output_finite']
            if not valid:
                stats['scalar_error_or_nonfinite_excluded'] += 1; continue
            stats['locally_finite_scalar_successes'] += 1
            if not success:
                stats['failed_parent_candidates_discarded'] += 1; continue
            if not mode_allowed:
                stats['check_or_cacheoff_candidates_excluded'] += 1; continue
            stats['successful_primary_candidate_requests'] += 1
            if key in committed:
                stats['repeat_of_prior_successful_eligible_parent'] += 1
            # Only a shadow capacity estimate, never a claim that an actual
            # cache would charge or allocate these precise bytes.
            cost = b['key_bytes'] + e['output_payload_estimate'] + 256
            pending.setdefault(key,cost)
        for key,cost in pending.items():
            if key in committed: continue
            usage = sum(store.get(k,0) for k in ('qp_bytes','dp_bytes','pole_bytes'))
            free = max(0,store.get('budget',0)-usage-modeled_bytes[budget_namespace])
            if cost > free:
                stats['simulated_budget_refusals'] += 1
            else:
                committed[key] = pid; modeled_bytes[budget_namespace] += cost
                stats['simulated_committed_unique_keys'] += 1
        # All parent statuses and exclusions remain visible, including parents
        # which fail in geometry before requesting any scalar kernel.
        totals.update(stats)
        for sid in p['requests']:
            b=beginnings[sid]; e=endings.get(sid)
            f=family[str(b['family'])]; f['requests']+=1
            if e: f['kernel_ns']+=e['kernel_ns']
        rows.append({k:p[k] for k in ('parent','point','group','synthetic','n','rank','finite','poles','check','cache','epoch')} |
                    {'bridge_status':p.get('status'),'public_status':p['public_status'],'counts':dict(stats),'notes':p['notes']})
    derived = ('within_parent_repeats','across_parent_repeats','new_exact_keys',
               'repeat_of_prior_successful_eligible_parent','simulated_budget_refusals',
               'simulated_committed_unique_keys')
    if not complete:
        # A missing lifecycle or status event can invalidate more than a
        # simple count: do not publish hit opportunities or budget estimates.
        for name in derived: totals[name] = None
        for row in rows:
            for name in derived: row['counts'][name] = None
    return {'scope':'Bounded scalar shadow observations; no reuse or whole-amplitude timing claim',
        'uniqueness_complete':complete,'raw_total_scalar_requests':data['scalar_requests'],
        'raw_completed_scalar_requests':data['scalar_completed'],'dropped_events':data['dropped_events'],
        'orphan_events_from_cap':orphan_events,
        'opportunity_assessment_allowed':complete,
        'public_transport_status_bound':'public_status_by_parent' in data,
        'derived_counts_suppressed_if_incomplete':list(derived),
        'observation_flag_changes':{'key':data['key_flag_changes'],'output':data['output_flag_changes']},
        'counts':dict(totals),'family':{k:dict(v) for k,v in family.items()},'parents':rows,
        'budget_limitation':'Simulation uses estimated MPFR payload and metadata sizes. Repeated successful keys are opportunities, not qualified usable hits.',
        'incompleteness_limitation':'When capped, global request/time totals continue; key uniqueness and all derived repetition counts are incomplete.'}

if __name__ == '__main__':
    src=Path(sys.argv[1]); result=analyze(json.loads(src.read_text()))
    dst=Path(sys.argv[2]); dst.write_text(json.dumps(result,indent=2)+'\n')
