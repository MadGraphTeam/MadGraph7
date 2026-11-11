"""Pure source/lexical checks only: no Fortran/compiler/native execution."""
from pathlib import Path
import importlib.util
import re
import hashlib
import json
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bulk_source_prepare',HERE/'prepare.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)

def lexical_model(record):
    if len(record)!=64: return False
    token=record.rstrip(b' ')
    if token in (b'0',b'-0'): return True
    m=re.fullmatch(rb'-?[1-9]\.[0-9]{38}e(-?)(0|[1-9][0-9]{0,3})',token)
    return bool(m and int(m[2])<=4900 and not(m[1] and m[2]==b'0'))

def field(s): return s.encode().ljust(64,b' ')
NORMAL='1.'+'2345678901'*3+'23456789'

@pytest.mark.parametrize('s',['0','-0',NORMAL+'e0',NORMAL+'e4900',NORMAL+'e-4900','-'+NORMAL+'e17'])
def test_eligible_model(s): assert lexical_model(field(s))

@pytest.mark.parametrize('s',[
    '', ' ', '+0','-0e0',NORMAL+'e4901',NORMAL+'e-4901',NORMAL+'e0001',NORMAL+'e+1',NORMAL+'e-0',
    NORMAL+'E1',' '+NORMAL+'e0',NORMAL+'e0\t',NORMAL+'e0\x00',NORMAL+'e0,2',NORMAL+'e0/',
    '2*'+NORMAL+'e0','NaN','Inf','-Infinity',NORMAL+'e0\n1','1.0e0','0.'+'0'*38+'e0',
    '1.2 3e0','1.2\te0','1.2\re0',',','/','2*','\x00','\u00e9',NORMAL+'e9999999999999'])
def test_ineligible_model(s): assert not lexical_model(field(s))

def block(source,name):
    match=re.search(r'^  subroutine '+name+r'\(.*?^  end subroutine$',source,re.M|re.S)
    assert match
    return match.group(0)

def test_exact_source_derivation_and_scalar_preservation():
    original=(HERE/'original-mtirust.f90').read_text();candidate=(HERE/'candidate-mtirust.f90').read_text()
    assert prepare.transform(original,(HERE/'helper.f90').read_text())==candidate
    for name in ['unpack','pack','tensor_dp','ensure_context','tirust_clear_cache','tirust_error_message']:
        assert block(original,name)==block(candidate,name)
    assert original.split('  interface\n',1)[1].split('contains\n',1)[0]==candidate.split('  interface\n',1)[1].split('contains\n',1)[0]
    assert candidate.count('read(text,*,iostat=io_status) values')==1
    assert 'finite == 1_c_int .and. poles == 0_c_int' in candidate
    assert 'ieee_set_' not in candidate
    assert 'call unpack(records(:,k),values(k),io_status)' in candidate
    before=block(original,'tensor_mp');after=block(candidate,'tensor_mp')
    mapping='    do j=1,4\n      do i=1,count\n'
    assert before.split(mapping)[1]==after.split(mapping)[1]
    assert before.count('ios = clear_cache(context)')==after.count('ios = clear_cache(context)')==1
    error="local_error = 'TIRust returned an invalid or nonfinite binary128 output record.'"
    assert before.count(error)==after.count(error)==1

def test_bindings_and_source_only_scope():
    d=json.loads((HERE/'bindings.json').read_text())
    assert d['status']=='source-only-uncompiled-unqualified'
    assert d['compiler_executions']==d['native_executions']==0
    assert d['runtime_changes'] is d['performance_acceptance'] is False
    assert hashlib.sha256((HERE/'candidate-mtirust.f90').read_bytes()).hexdigest()==d['source_preview_sha256']
    assert hashlib.sha256((HERE/'original-mtirust.f90').read_bytes()).hexdigest()==d['source_original']['sha256']
    assert all(len(s)<=132 for s in (HERE/'candidate-mtirust.f90').read_text().splitlines())
