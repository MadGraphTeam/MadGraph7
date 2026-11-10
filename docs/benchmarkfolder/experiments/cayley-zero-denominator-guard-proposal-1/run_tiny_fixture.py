#!/usr/bin/env python3
"""Compile and run ONLY the extracted quotient/filter; never a reducer."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/cayley-zero-denominator-guard-proposal-1/attempt1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))

def main():
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    assert platform.libc_ver()[0] == 'glibc'
    spec = json.loads((HERE / 'spec.json').read_text())
    upstream = ROOT.parent / 'references/TIRust/crates/iregi-rust/src/direct/exec/gram.rs'
    assert sha(upstream) == spec['upstream_gram_sha256']
    assert upstream.read_bytes() == (HERE / 'source/gram.rs.baseline').read_bytes()
    WORK.mkdir(parents=True, exist_ok=False)
    raw = WORK / 'raw'
    raw.mkdir()
    start = time.monotonic()
    deadline = start + 300
    receipt = {'schema': 'cayley-zero-guard-tiny-execution-v1', 'complete': False,
               'qualification': spec['qualification'], 'commands': [],
               'platform': platform.platform(), 'libc': platform.libc_ver(),
               'sources': {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()},
               'upstream_gram_before': sha(upstream)}

    def invoke(label, argv, stdin='', timeout=30):
        remain = deadline - time.monotonic()
        assert remain > 0, 'overall fixture deadline exceeded'
        began = time.monotonic()
        child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True, preexec_fn=limits)
        try:
            stdout, stderr = child.communicate(stdin, timeout=min(timeout, remain))
        except BaseException:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
            raise
        rec = {'label': label, 'argv': [str(v) for v in argv], 'returncode': child.returncode,
               'elapsed_seconds': time.monotonic() - began, 'stdout': stdout, 'stderr': stderr,
               'reaped': True}
        receipt['commands'].append({k: v for k, v in rec.items() if k not in ('stdout', 'stderr')})
        (WORK / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return rec

    rustc = Path(shutil.which('rustc')).resolve()
    receipt['rustc'] = {'path': str(rustc), 'sha256': sha(rustc)}
    version = invoke('rustc-version', [str(rustc), '-vV'])
    assert version['returncode'] == 0
    (raw / 'rustc-version.txt').write_text(version['stdout'] + version['stderr'])
    for opt in spec['optimization_levels']:
        binary = WORK / f'fixture-opt{opt}'
        build = invoke(f'compile-opt{opt}', [str(rustc), '--edition=2024', '--crate-name',
                       f'cayley_zero_guard_opt{opt}', str(HERE / 'source/fixture.rs'), '-C',
                       f'opt-level={opt}', '--emit=link,asm', '-o', str(binary)], timeout=60)
        (raw / f'opt{opt}-build.txt').write_text(build['stdout'] + build['stderr'])
        assert build['returncode'] == 0
        receipt.setdefault('artifacts', []).append({'path': str(binary.relative_to(ROOT)),
                                                   'sha256': sha(binary), 'bytes': binary.stat().st_size})
        assembly = binary.with_suffix('.s')
        assert assembly.is_file()
        with gzip.open(raw / f'opt{opt}-assembly.s.gz', 'wb') as out:
            out.write(assembly.read_bytes())
        requests = []
        for n in spec['numerator_and_denominator_bit_patterns']:
            for d in spec['numerator_and_denominator_bit_patterns']:
                for rounding in spec['rounding_modes'].values():
                    for seed in spec['initial_flag_seeds']:
                        identity = f'm{len(requests)//2}'
                        for mode in ('baseline', 'candidate'):
                            requests.append(f'{identity} {mode} {n} {d} {seed} {rounding} 0 0')
        masked = invoke(f'masked-opt{opt}', [str(binary)], '\n'.join(requests) + '\n')
        assert masked['returncode'] == 0 and masked['stderr'] == ''
        with gzip.open(raw / f'opt{opt}-masked.txt.gz', 'wt') as out:
            out.write(masked['stdout'])
        traps = []
        for case in spec['trap_cases']:
            for mode in ('baseline', 'candidate'):
                request = f"{case['id']} {mode} {case['numerator']} {case['denominator']} 0 0 {spec['trap_mask']} 1\n"
                rec = invoke(f"trap-opt{opt}-{case['id']}-{mode}", [str(binary)], request, timeout=5)
                traps.append({'id': case['id'], 'mode': mode, **{k: rec[k] for k in ('returncode', 'stdout', 'stderr')}})
        (raw / f'opt{opt}-traps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in traps))
    module_spec = importlib.util.spec_from_file_location('tiny_analysis', HERE / 'analyze.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result = module.analyze(raw)
    (WORK / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    receipt['upstream_gram_after'] = sha(upstream)
    assert receipt['upstream_gram_before'] == receipt['upstream_gram_after']
    receipt.update(complete=True, passed=result['passed'], elapsed_seconds=time.monotonic() - start,
                   results_sha256=sha(WORK / 'results.json'))
    (WORK / 'execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'complete': True, 'passed': result['passed'], 'work': str(WORK),
                      'results': result}, indent=2))

if __name__ == '__main__':
    main()
