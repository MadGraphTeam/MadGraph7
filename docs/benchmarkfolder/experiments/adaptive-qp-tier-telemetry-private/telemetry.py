"""Read additive telemetry records without inferring tiers from return code323."""
import json
import math
import re
import struct

FIELDS = ('evaluations_256', 'eligible_refusals_256', 'evaluations_512',
          'successes_512', 'failures_512', 'cache_hits_256', 'cache_hits_512',
          'check_evaluations_256', 'check_evaluations_512')


def require(value, message):
    if not value:
        raise ValueError(message)


def parse_snapshots(text):
    lines = iter(text.splitlines())
    result = {}
    for line in lines:
        tokens = line.split()
        require(len(tokens) == 18 and tokens[0] == 'TIRUST_QP_TIER_V1', 'Expected exactly eleven telemetry words')
        point, phase = int(tokens[1]), tokens[2]
        available, generation, status, count = map(int, tokens[3:7])
        require(point >= 0 and phase in ('BEFORE', 'AFTER'), 'Invalid snapshot identity')
        require(available in (0, 1) and generation >= -1 and status in (0, 1, 2, 3) and count == 11, 'Invalid snapshot metadata')
        require(all(re.fullmatch('[0-9A-Fa-f]{16}', x) for x in tokens[7:]), 'Invalid word encoding')
        words = [int(x, 16) for x in tokens[7:]]
        error = next(lines, '').split()
        require(len(error) == 4 and error[:3] == ['TIRUST_QP_ERROR_V1', str(point), phase], 'Missing matching error record')
        require(re.fullmatch('[0-9A-Fa-f]{2048}', error[3]), 'Invalid fixed error transport')
        raw = bytes.fromhex(error[3])
        condition = struct.unpack('>d', words[10].to_bytes(8, 'big'))[0]
        if status == 0:
            require(available == 1 and generation >= 1, 'Success without capability/live context')
            require(words[9] in (0, 256, 512) and math.isfinite(condition) and condition >= 0, 'Invalid last-dispatch metadata')
        else:
            require(not any(words), 'Failed snapshot must not expose partial words')
        key = (point, phase)
        require(key not in result, 'Duplicate snapshot')
        result[key] = {'point': point, 'phase': phase, 'available': bool(available), 'generation': generation,
                       'status': status, 'words_hex': [f'{word:016x}' for word in words],
                       'cumulative': dict(zip(FIELDS, words[:9])), 'last_output_bits': words[9],
                       'last_eligible_condition_bits': f'{words[10]:016x}', 'last_eligible_condition': condition,
                       'native_error_before_query': raw[:512].decode('utf8', 'replace').rstrip(' '),
                       'telemetry_error': raw[512:].decode('utf8', 'replace').rstrip(' ')}
    return result


def point_record(before, after, *, return_code, use_qp_integrand):
    require(type(return_code) is int and type(use_qp_integrand) is bool, 'Typed host metadata required')
    require(before['point'] == after['point'] and before['phase'] == 'BEFORE' and after['phase'] == 'AFTER', 'Wrong point boundary pair')
    reason = None
    if not before['available'] or not after['available']:
        reason = 'getter_unavailable'
    elif before['status'] or after['status']:
        reason = 'snapshot_failed_or_context_absent'
    elif before['generation'] != after['generation']:
        reason = 'context_generation_changed'
    elif any(after['cumulative'][key] < before['cumulative'][key] for key in FIELDS):
        reason = 'counter_reset_or_wrap'
    delta = None if reason else {key: after['cumulative'][key] - before['cumulative'][key] for key in FIELDS}
    return {'schema': 'madloop-private-adaptive-qp-tiers-v1', 'point': before['point'],
            'host_return_code': return_code, 'use_qp_integrand': use_qp_integrand,
            'before': before, 'after': after, 'delta': delta, 'attribution_gap': reason,
            'successes_256': None, 'successes_256_reason': 'The eleven-word ABI has no counter for 256-bit successes or ineligible failures.',
            'accepted_amplitude_tier': None,
            'tier_scope': 'Integral working precision; cumulative attempts include checks. Last bits describe only the last dispatched QP tensor, may predate an invalid foreign call, and do not classify the accepted amplitude or numerator.',
            'boundary_contract': 'One isolated point between snapshots in the same context generation; cache clear/budget changes do not reset precision counters.'}


def serialize(record):
    return json.dumps(record, sort_keys=True, allow_nan=False)
