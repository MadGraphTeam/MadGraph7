from copy import deepcopy
import json
import os
from pathlib import Path
import pytest
from telemetry import parse_snapshots, point_record, serialize


def lines(words=None, *, point=1, phase='BEFORE', generation=1, status=0, available=1):
    words = [0] * 11 if words is None else words
    return ('TIRUST_QP_TIER_V1 %d %s %d %d %d 11 ' % (point, phase, available, generation, status)
            + ' '.join('%016X' % x for x in words) + '\n'
            + 'TIRUST_QP_ERROR_V1 %d %s ' % (point, phase) + ('20' * 1024) + '\n')


def pair(before=None, after=None, *, generation=1):
    parsed = parse_snapshots(lines(before) + lines(after, phase='AFTER', generation=generation))
    return parsed[(1, 'BEFORE')], parsed[(1, 'AFTER')]


def test_exact_u64_words_and_condition_bits_survive_json_without_float_counter_loss():
    values = [(1 << 63) + 1] * 9 + [512, 0x3FF0000000000001]
    before, after = pair(values, values)
    result = json.loads(serialize(point_record(before, after, return_code=323, use_qp_integrand=False)))
    assert result['after']['cumulative']['evaluations_256'] == (1 << 63) + 1
    assert result['after']['last_eligible_condition_bits'] == '3ff0000000000001'
    assert result['after']['last_eligible_condition'] == 1.0000000000000002
    assert result['host_return_code'] == 323 and result['accepted_amplitude_tier'] is None
    assert result['successes_256'] is None and result['use_qp_integrand'] is False


def test_counter_deltas_report_attempts_checks_and_cache_separately():
    before, after = pair([4] * 9 + [256, 0], [7, 5, 5, 5, 4, 6, 5, 5, 5, 512, 0])
    record = point_record(before, after, return_code=323, use_qp_integrand=True)
    assert list(record['delta'].values()) == [3, 1, 1, 1, 0, 2, 1, 1, 1]
    assert record['attribution_gap'] is None


@pytest.mark.parametrize('return_code', [219, 223, 323, 420, 120])
def test_historical_status_is_not_a_precision_detector(return_code):
    before, after = pair()
    record = point_record(before, after, return_code=return_code, use_qp_integrand=True)
    assert record['host_return_code'] == return_code
    assert record['accepted_amplitude_tier'] is None and record['after']['last_output_bits'] == 0


@pytest.mark.parametrize('bad', [[], [0] * 10, [0] * 12])
def test_short_or_extra_word_responses_are_rejected(bad):
    with pytest.raises(ValueError, match='eleven'):
        parse_snapshots(lines(bad))


@pytest.mark.parametrize('mutate', [
    lambda text: text.replace(' 0 11 ', ' 0 10 ', 1),
    lambda text: text.replace('0000000000000000', 'garbage', 1),
    lambda text: text + text,
    lambda text: text.splitlines()[0] + '\n',
    lambda text: text.replace('TIRUST_QP_ERROR_V1 1 BEFORE', 'TIRUST_QP_ERROR_V1 2 BEFORE'),
])
def test_malformed_truncated_duplicate_or_misattributed_records_fail(mutate):
    with pytest.raises(ValueError):
        parse_snapshots(mutate(lines()))


@pytest.mark.parametrize('last,condition', [(257, 0), (0, 0x7FF0000000000000), (512, 0xFFF0000000000000)])
def test_invalid_last_dispatch_metadata_fails(last, condition):
    with pytest.raises(ValueError):
        parse_snapshots(lines([0] * 9 + [last, condition]))


@pytest.mark.parametrize('change,expected', [
    ({'available': False, 'status': 3}, 'getter_unavailable'),
    ({'generation': -1, 'status': 2}, 'snapshot_failed_or_context_absent'),
    ({'generation': 2}, 'context_generation_changed'),
])
def test_missing_context_or_reset_never_manufactures_a_point_delta(change, expected):
    before, after = pair()
    after.update(change)
    result = point_record(before, after, return_code=323, use_qp_integrand=True)
    assert result['delta'] is None and result['attribution_gap'] == expected


def test_counter_reset_or_wrap_is_explicit_even_if_epoch_was_not_changed():
    before, after = pair([8] * 9 + [512, 0], [0] * 11)
    result = point_record(before, after, return_code=420, use_qp_integrand=True)
    assert result['delta'] is None and result['attribution_gap'] == 'counter_reset_or_wrap'


def test_wrong_phase_or_point_is_not_silently_joined():
    before, after = pair()
    after['point'] = 2
    with pytest.raises(ValueError, match='boundary'):
        point_record(before, after, return_code=323, use_qp_integrand=True)


def test_compiled_fortran_and_optional_c_getter_transport():
    directory = Path(os.environ['TELEMETRY_MOCK_OUTPUT'])
    present = parse_snapshots((directory / 'present.log').read_text())
    normal = point_record(present[(7, 'BEFORE')], present[(7, 'AFTER')], return_code=323, use_qp_integrand=True)
    assert list(normal['delta'].values()) == [3, 1, 1, 1, 0, 2, 1, 1, 1]
    assert normal['after']['cumulative']['evaluations_256'] == (1 << 63) + 4
    reset = point_record(present[(8, 'BEFORE')], present[(8, 'AFTER')], return_code=219, use_qp_integrand=False)
    assert reset['delta'] is None and reset['attribution_gap'] == 'context_generation_changed'
    absent = parse_snapshots((directory / 'absent.log').read_text())
    missing = point_record(absent[(1, 'BEFORE')], absent[(1, 'AFTER')], return_code=323, use_qp_integrand=True)
    assert missing['delta'] is None and missing['attribution_gap'] == 'getter_unavailable'
    assert all(row['status'] == 3 for row in absent.values())
    (directory / 'example-point.json').write_text(serialize(normal) + '\n')
