#!/usr/bin/env python3
"""Source-only, insertion-only private observer derivation; never build or run.

Pinned numerical-campaign inputs are read, never modified. Derived sources and
diffs are written once under this new experiment's prepared/ directory.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'docs/benchmarkfolder/campaigns/split-dp-terminal-1'
OUTPUTS = ROOT / '.benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate'
CASES = {'qcd_n3_uux_zgg': 'P0_uux_zgg', 'ew_n3_epem_ttxa': 'P0_epem_ttxa'}
MANIFEST = '1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31'
COUNTER = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(code):
    global COUNTER
    COUNTER += 1
    return 'C DCO_BEGIN %d\n%s\nC DCO_END %d\n' % (COUNTER, code.rstrip(), COUNTER)


def strip(text):
    return re.sub(r'(?ms)^C DCO_BEGIN (\d+)\n.*?^C DCO_END \1\n', '', text)


def add(text, anchor, code, after=False):
    pattern = r'(?m)^'+re.escape(anchor)
    matches = list(re.finditer(pattern,text))
    if len(matches) != 1:
        raise ValueError(('nonunique observer anchor', anchor, len(matches)))
    return re.sub(pattern,lambda m:m[0]+block(code) if after else block(code)+m[0],text)


def scoped(text, name, transform):
    pattern = r'(?m)^      SUBROUTINE '+re.escape(name)+r'\('
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(('nonunique routine', name))
    start = matches[0].start()
    end = text.index('\n      END\n', start)+len('\n      END\n')
    return text[:start]+transform(text[start:end])+text[end:]


def use(text):
    # Complete fixed-form signature precedes the first USE/declaration.
    anchor = next(line+'\n' for line in text.splitlines()
                  if re.match(r'^      (USE |IMPLICIT |INTEGER |REAL\*|COMPLEX\*|LOGICAL )', line))
    return add(text, anchor, '      USE DP_CAUSE_OBSERVER')


def emit(tag, expr):
    return '      CALL DCO_EMIT(%d,%s)' % (tag, expr)


def ints(tag, values):
    return emit(tag, 'INT((/'+values+'/),8)')


def bits(tag, value, count):
    return emit(tag, 'TRANSFER('+value+',(/0_8/),'+count+')')


def diagnostic(text, precision):
    text = use(text)
    text = add(text, '      STABLE=.FALSE.\n', ints(30, f'{precision},CTMODE,NLOOPLINE,RANK'))
    text = add(text, '      IF(.NOT.TIRUST_SUPPORTS(NLOOPLINE,RANK,TIRUSTCOMPUTEFINITE))\n', ints(31, '2'))
    text = add(text, '      CALL ML5_0_INITIALIZE_TIRUST(STATUS)\n', ints(31, '3'))
    text = add(text, '      CALL ML5_0_INITIALIZE_TIRUST(STATUS)\n', ints(34, '4,STATUS'), after=True)
    text = add(text, '     $ ,STATUS)\n', ints(34, '5,STATUS'), after=True)
    text = add(text, '      IF(REF_P.GT.1D-8) THEN\n', ints(31, '6'))
    text = add(text, 'C     Accumulate ordered prefixes once, preserving the zero-seeded sum.\n', ints(31, '7'))
    text = add(text, '      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS\n', ints(31, '8'))
    text = add(text, '     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)\n', ints(32, 'STATUS'), after=True)
    text = add(text, '      DO I=0,NC-1\n', ints(31, '9'))
    text = add(text, '      STABLE=.TRUE.\n', ints(33, f'{precision}'))
    return text


def tir(text):
    def routine(body):
        body = use(body)
        anchor = ('        CALL ML5_0_TIRUSTLOOP(CTMODE,NLOOPLINE,RANK,PL,M2L,TIRCOEFS\n'
                  '     $   ,STABLE)\n')
        body = add(body, anchor, ints(20, 'CTMODE,I_SQSO,I_LOOPGROUP,NLOOPLINE,RANK'))
        body = add(body, anchor, ints(22, 'MERGE(1,0,STABLE),MERGE(1,0,FPE_IN_DP_REDUCTION)'), after=True)
        body = add(body, '          RETURN\n', ints(21, 'MERGE(1,0,STABLE),MERGE(1,0,FPE_IN_DP_REDUCTION)'))
        body = add(body, '      END\n', '      IF(MLREDUCTIONLIB(I_LIB).EQ.9) THEN\n'+
                   ints(21, 'MERGE(1,0,STABLE),MERGE(1,0,FPE_IN_DP_REDUCTION)')+'\n      ENDIF')
        return body
    return scoped(text, 'ML5_0_TIRLOOP', routine)


def host(text):
    # Only this real routine has DP_RES and the acceptance policy below.
    start = text.index('      SUBROUTINE ML5_0_SLOOPMATRIX_RETRY_IMPL(')
    end = text.index('\n      END\n', start)+len('\n      END\n')
    body = use(text[start:end])
    anchor = '          CALL ML5_0_COMPUTE_ACCURACY(DP_RES,N_DP_EVAL,ACC,ANS)\n'
    code = '\n'.join([
        ints(40, 'MLREDUCTIONLIB(I_LIB),N_DP_EVAL,NSQUAREDSO,MERGE(1,0,FPE_IN_DP_REDUCTION)'),
        bits(41, 'ACC', 'NSQUAREDSO+1'),
        bits(42, 'DP_RES(:,:,1:N_DP_EVAL)', '3*(NSQUAREDSO+1)*N_DP_EVAL'),
    ])
    body = add(body, anchor, code, after=True)
    # This comment immediately follows the FPE ACC=1 override, before any retry.
    anchor = 'C         One explicit compensated DP retry with the same stability checks.\n'
    if anchor not in body:
        anchor = 'C         One explicit compensated DP retry with the same stability\nC          checks.\n'
    body = add(body, anchor, '\n'.join([
        ints(43, 'MLREDUCTIONLIB(I_LIB),MERGE(1,0,FPE_IN_DP_REDUCTION)'),
        bits(44, 'ACC', 'NSQUAREDSO+1'), bits(49, 'MLSTABTHRES', '1')]))
    body = add(body, '              I_LIB=INDEX_QP_TOOLS(1)\n', ints(45,
               'I_LIB,MLREDUCTIONLIB(I_LIB-1),MLREDUCTIONLIB(I_LIB)'))
    anchor = '          CALL ML5_0_COMPUTE_ACCURACY(QP_RES,N_QP_EVAL,ACC,ANS)\n'
    body = add(body, anchor, '\n'.join([
        ints(46, 'MLREDUCTIONLIB(I_LIB),N_QP_EVAL,NSQUAREDSO,MERGE(1,0,FPE_IN_QP_REDUCTION)'),
        bits(47, 'ACC', 'NSQUAREDSO+1')]), after=True)
    return text[:start]+body+text[end:]


def driver(text):
    end=text.index('\n      END\n')+len('\n      END\n')
    tail=text[end:]
    text=text[:end]
    text = use(text)
    text = add(text, "      CALL SETPARA('param_card.dat')\n", '      CALL DCO_SETUP()')
    calls = list(re.finditer(r'(?m)^( +)CALL ML5_0_SLOOPMATRIX_THRES\(P,ANS,-1D0,ACC,CODE\)\n', text))
    if len(calls) != 3:
        raise ValueError('expected bootstrap, warmup and observation call sites')
    for match, metadata in reversed(list(zip(calls, ('0,1,0,0', 'K,WARM_POINT,1,0', 'SEQUENCE,K,2,L')))):
        before = ('      CALL DCO_BEGIN('+metadata+',\n'
                  '     $ (/TRANSFER(P,(/0_8/),4*NEXTERNAL),\n'
                  '     $ TRANSFER(MU_R,0_8)/))')
        after = ('      CALL DCO_END(CODE,TRANSFER(ANS,(/0_8/),4*(NDIM+1)),\n'
                 '     $ TRANSFER(ACC,(/0_8/),NSO+1))')
        text = text[:match.start()]+block(before)+match[0]+block(after)+text[match.end():]
    text = add(text, '      CLOSE(74)\n', '      CALL DCO_CLOSE()')
    return text+tail


def main():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('work from the MadGraph root')
    if sha(CAMPAIGN/'manifest.json') != MANIFEST:
        raise ValueError('registered manifest drift')
    out = HERE/'prepared'
    if out.exists():
        raise ValueError('refusing to replace source preparation')
    records = {}
    staged = []
    for case, sub in CASES.items():
        sp = OUTPUTS/case/'SubProcesses'/sub
        evidence = CAMPAIGN/'experiments/numerical'/('full100-warmup-'+case+'-1')/'candidate-9/candidate-inputs.json'
        prior = json.loads(evidence.read_text())
        rendered = prior['compilation']['sources']
        files = {'TIR_interface.f':tir, 'TIRUST_interface.f':lambda text: scoped(scoped(text,
                 'ML5_0_TIRUST_TENSOR_DIAGNOSTIC',lambda b:diagnostic(b,53)),
                 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC',lambda b:diagnostic(b,113)),
                 'loop_matrix.f':host, 'check_sa.f':driver}
        for name, transform in files.items():
            source = sp/name
            original = source.read_text()
            expected = rendered.get(name)
            if expected is None:
                expected = rendered.get(str(source.relative_to(OUTPUTS/case)))
            # Driver has its own explicit hash; numerical-source map may nest.
            if name == 'check_sa.f':
                expected = prior['compilation']['driver_sha256']
            if isinstance(expected, dict):
                expected = expected.get('sha256')
            if expected != sha(source):
                raise ValueError(('compiled source identity missing/drift',case,name,expected,sha(source)))
            changed = transform(original)
            if strip(changed) != original:
                raise ValueError(('not insertion-only',case,name))
            if any(len(line)>132 for line in changed.splitlines() if not line.startswith('C')):
                raise ValueError(('fixed-form line exceeds132',case,name))
            patch = ''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
                         fromfile='original/'+name,tofile='observed/'+name))
            staged.extend([(case+'/'+name+'.original', original),
                           (case+'/'+name, changed), (case+'/'+name+'.patch',patch)])
            records[case+'/'+name] = {'input':str(source.relative_to(ROOT)),
                'input_sha256':sha(source),'output_sha256':hashlib.sha256(changed.encode()).hexdigest(),
                'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(), 'insertion_only':True}
        records[case+'/evidence'] = {'path':str(evidence.relative_to(ROOT)),'sha256':sha(evidence)}
    # No output is created until every source/anchor/insertion check succeeds.
    out.mkdir()
    for rel,contents in staged:
        target=out/rel; target.parent.mkdir(exist_ok=True); target.write_text(contents)
    (out/'preparation.json').write_text(json.dumps({'schema':1,'source_only':True,
        'compiled':False,'native_executed':False,'manifest_sha256':MANIFEST,
        'prepare_sha256':sha(Path(__file__)),'files':records},indent=2,sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
