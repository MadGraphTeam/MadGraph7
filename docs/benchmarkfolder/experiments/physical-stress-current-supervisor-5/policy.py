"""Pure conditional admission and disjoint coverage; no filesystem or execution."""
import copy


def require(value, message):
    if not value:
        raise ValueError(message)


def key(request):
    return request['case'] + '/' + request['label']


def select(plan, summary, stage, accounting, aborted, launched_request_ids):
    """Admit only the exact four-completed/request5-lock-deferred predecessor.

    The caller first verifies the portable capsule and every original closed
    report byte. ``launched_request_ids`` is independently derived from its
    authenticated specification/execution inventory, never saved success flags.
    """
    requests = plan['requests']; keys = [key(r) for r in requests]
    require(len(keys) == len(set(keys)) == 24, 'Expected 24 unique original requests')
    rows = summary['request_rows']
    require([r['request_id'] for r in rows] == keys, 'Ancestor request identities/order differ')
    require(stage['status'] == summary['stage_status'] == 'failed', 'Ancestor was not a failed terminal stage')
    require(stage['exception'] == "TimeoutError('Lock wait expired before execution')", 'Unexpected ancestor failure')
    require(aborted == {'request': requests[4], 'state': 'deferred-before-launch',
                       'exception': "TimeoutError('Lock wait expired before execution')",
                       'numerically_qualified': False}, 'Ancestor was not request5 deferred before launch')
    require(accounting['requested'] == 24 and accounting['completed'] == 4
            and accounting['not_completed'] == requests[4:]
            and accounting['native_stage_failure'] == stage['exception'], 'Ancestor accounting differs')
    require(launched_request_ids == keys[:4], 'Ancestor launch inventory differs')
    require(summary['completed_orchestration'] == summary['host_checked_complete_streams'] == 4,
            'Ancestor completion counts differ')
    for i, (request, row) in enumerate(zip(requests, rows)):
        require(all(row[k] == request[k] for k in ('case', 'label', 'source', 'backend')),
                'Ancestor row identity differs')
        expected = 'host_checked_complete' if i < 4 else ('deferred_before_launch' if i == 4 else 'not_started')
        require(row['state'] == expected and row['host_checked_complete_stream'] is (i < 4),
                'Ancestor is not the exact admitted completed/deferred/not-started prefix')
        require(row['qualified_final_points'] == ([1, 2] if i < 4 else []), 'Ancestor final-point qualification differs')
        if i >= 4:
            require(row['raw_final_points_observed'] == [] and row['warmup_calls_observed'] == 0,
                    'Neverlaunched request has numerical observations')
    selected = copy.deepcopy(requests[4:])
    require(selected[0]['backend'] == 8 and selected[1]['backend'] == 9
            and len(selected) == 20, 'Unexpected continuation suffix')
    return selected


def combine(plan, earlier_rows, later_rows):
    """Provenance-independent disjointness gate, not numerical qualification.

    A future aggregate binder must verify both portable proofs before calling
    this helper and recompute all coefficient comparisons from retained rows.
    This function never upgrades a failed stream or declares agreement.
    """
    expected = [key(r) for r in plan['requests']]
    require(len(expected) == len(set(expected)) == 24, 'Expected 24 unique original requests')
    left = [r['request_id'] for r in earlier_rows]; right = [r['request_id'] for r in later_rows]
    require(left == expected[:4] and right == expected[4:], 'Coverage is not the exact disjoint prefix and suffix')
    require(not set(left) & set(right), 'Repeated request cannot enter aggregate coverage')
    return {'request_rows': copy.deepcopy(earlier_rows + later_rows), 'coverage_count': 24,
            'proofs_must_be_independently_verified': True, 'coefficient_agreement': None,
            'independent_physical_accuracy_qualified': False, 'performance_eligible': False}
