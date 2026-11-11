"""Pure protocol, exact-word, environment and fixed-history timing checks."""
import copy
import importlib.util
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bulk_physical_analysis_tests',HERE/'analyze.py')
analyze=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyze)


def parity_fixture():
    words=['META 1 100 1 1 1'];ieee=list(words)
    occurrences=[(1,1,1),(2,2,16)]+[(i+2,3,i) for i in range(1,101)]
    for occurrence,phase,point in occurrences:
        status=323 if point==16 else 219
        for order in range(2):
            for component in range(4):
                words.append(f'ANS 1 {occurrence} {phase} {point} {order} {component} {status} 3FF0000000000000')
        for order in range(2):words.append(f'ACC 1 {occurrence} {phase} {point} {order} {status} 3CB0000000000000')
        for edge in (0,1):ieee.append('IEEE '+' '.join(map(str,[1,occurrence,phase,point,edge,0,*([0]*10)])))
    return '\n'.join(words)+'\n','\n'.join(ieee)+'\n',dict(points=100,ndim=1,nso=1,nborn=1)


def test_complete_full100_and_two_anchors():
    parsed=analyze.parse_parity(*parity_fixture())
    assert parsed['occurrences']==102 and len(parsed['values'])==102
    assert parsed['values']['1']['status']==219 and parsed['values']['2']['status']==323
    assert [parsed['values'][str(i)]['point'] for i in range(3,103)]==list(range(1,101))


@pytest.mark.parametrize('which',['missing-word','wrong-backend','nan','wrong-order','changed-ieee'])
def test_parity_corruption_rejected(which):
    words,ieee,ready=parity_fixture()
    if which=='missing-word':words=words.replace('ANS 1 1 1 1 0 0 219 3FF0000000000000\n','')
    if which=='wrong-backend':words=words.replace(' 219 ',' 216 ',1)
    if which=='nan':words=words.replace('3FF0000000000000','7FF8000000000000',1)
    if which=='wrong-order':words=words.replace('ANS 1 1 1 1 0 0','ANS 1 1 1 1 1 0',1)
    if which=='changed-ieee':ieee=ieee.replace('IEEE 1 1 1 1 1 0','IEEE 1 1 1 1 1 1',1)
    with pytest.raises(ValueError):analyze.parse_parity(words,ieee,ready)


def test_single_changed_accuracy_word_blocks_pair_parity():
    parsed=analyze.parse_parity(*parity_fixture());different=copy.deepcopy(parsed)
    different['values']['102']['accuracy_words']['0']='3cb0000000000001'
    with pytest.raises(ValueError):analyze.compare_parity(parsed,different)


def test_closed_anchor_schema_includes_backend_precision_and_float_mirrors():
    parsed=analyze.parse_parity(*parity_fixture())
    expected=[]
    for point,status,precision in [(1,219,'DP'),(16,323,'QP')]:
        for order in (0,1):expected.append(dict(point=point,order=order,status=status,actual_backend=9,precision=precision,
            coefficients=[1.]*4,accuracy=2.**-52,coefficient_binary64_big_endian_hex=['3ff0000000000000']*4,
            accuracy_binary64_big_endian_hex='3cb0000000000000'))
    analyze.closed_anchor_words(expected,parsed)
    expected[-1]['accuracy_binary64_big_endian_hex']='3cb0000000000001'
    with pytest.raises(ValueError):analyze.closed_anchor_words(expected,parsed)


def wire_fixture(poles=0,halting=0,status=0):
    text=['ENVCONST 0 61',f'CALL 1 1 0 {poles} 1 0 1 {status} 512 0 {halting} 32']
    if status==0:
        for i in range(8):text.append(f'RECORD {i+1} '+b'0'.ljust(64,b' ').hex())
    text.append('END 1')
    return '\n'.join(text)+'\n',f'0 1 {8 if status==0 else 0}'


def test_real_wire_eligibility_uses_whole_batch_and_actual_environment():
    assert analyze.parse_wire(*wire_fixture())['whole_batch_eligible']==1
    assert analyze.parse_wire(*wire_fixture(poles=1))['whole_batch_eligible']==0
    assert analyze.parse_wire(*wire_fixture(halting=32))['whole_batch_eligible']==0
    assert analyze.parse_wire(*wire_fixture(status=1))['successful_native_QP_calls']==0
    text,health=wire_fixture();text=text.replace(b'0'.ljust(64,b' ').hex(),b'1.25'.ljust(64,b' ').hex(),1)
    assert analyze.parse_wire(text,health)['whole_batch_eligible']==0


@pytest.mark.parametrize('health',['1 1 8','0 2 8','0 1 7'])
def test_wire_io_or_count_failure_is_not_eligible_success(health):
    text,_=wire_fixture()
    with pytest.raises(ValueError):analyze.parse_wire(text,health)


def test_calibration_counts_shared_and_predeclared():
    assert analyze.common_pilot(.001,.002)==500
    assert analyze.common_fixed(500,.5,1.)==1000
    assert analyze.common_pilot(2.,3.)==1
    assert analyze.common_fixed(1,2.,3.)==1


def test_negative_benefit_retained_without_promotion():
    pairs=[]
    for i in range(7):
        output=dict(point=1,phase=4,status=219,ans_words={},accuracy_words={})
        pairs.append({'order':['original','candidate'] if i%2==0 else ['candidate','original'],
                      'original':dict(point=1,loops=2,seconds=2.,seconds_per_call=1.,last_output=output),
                      'candidate':dict(point=1,loops=2,seconds=3.,seconds_per_call=1.5,last_output=output)})
    summary=analyze.summarize_pairs(pairs)
    assert summary['median_paired_ratio']==1.5 and summary['observed_percent_change']==50.
    assert not summary['candidate_promoted'] and not summary['performance_acceptance']
    pairs[-1]['order'].reverse()
    with pytest.raises(ValueError):analyze.summarize_pairs(pairs)


def test_wire_wrapper_never_substitutes_native_result():
    source=(HERE/'wire_capture.c').read_text()
    function=source[source.index('int __wrap_'):source.index('void qp_wire_end_')]
    assert function.count('__real_ml_tirust_evaluate_mp(')==1
    assert 'return status;' in function and 'return failure' not in function
    assert 'fegetround()' in function and 'fegetexcept()' in function
    assert 'feset' not in source and 'feclear' not in source
