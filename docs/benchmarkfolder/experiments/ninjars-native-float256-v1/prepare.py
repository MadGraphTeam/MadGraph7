#!/usr/bin/env python3
"""Copy the sealed corrected-transport crate and add private test policies only."""
import sys
sys.dont_write_bytecode = True
import difflib
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRANSPORT = HERE.parent / 'ninjars-mp-transport-audit-v1'
WORK = ROOT / '.benchmark-work/experiments/ninjars-native-float256-v1'
BRIDGE = WORK / 'bridge'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    for name, digest in json.loads((HERE/'plan-seal.json').read_text())['files'].items():
        assert sha(HERE/name) == digest, name
    seal = json.loads((TRANSPORT/'manifest.json').read_text())
    for name, record in seal['files'].items():
        assert sha(TRANSPORT/name) == record['sha256'], name
    proposal = json.loads((TRANSPORT/'bridge-prepared.json').read_text())
    BRIDGE.mkdir(parents=True, exist_ok=False)
    baseline = {}
    for name, digest in proposal['proposed_files'].items():
        source = TRANSPORT/'proposed'/name
        assert sha(source) == digest and sha(ROOT/'vendor/ninjars'/name) == digest, name
        target = BRIDGE/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        baseline[name] = digest
    path = BRIDGE/'Cargo.toml'
    text = path.read_text()
    old = 'features = ["native-masters"]'
    assert text.count(old) == 1
    path.write_text(text.replace(old, 'features = ["native-masters", "native-kernels"]'))
    path = BRIDGE/'src/lib.rs'
    text = path.read_text()
    def replace(old, new):
        nonlocal text
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    replace('struct MpEntry {\n', '''#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StudyKernelPolicy {
    Interpreted,
    NativeDenominators,
    FullNative,
}
#[cfg(test)]
impl StudyKernelPolicy {
    const ALL: [Self; 3] = [Self::Interpreted, Self::NativeDenominators, Self::FullNative];
}

struct MpEntry {
    #[cfg(test)]
    study_backends: [ninjars_symbolica::KernelBackendSelection; 8],
''')
    replace('struct MpState {\n', '''struct MpState {
    #[cfg(test)]
    study_policy: StudyKernelPolicy,
''')
    replace('''        Ok(Self {
            precision,
            entries: HashMap::new(),''', '''        Ok(Self {
            #[cfg(test)]
            study_policy: StudyKernelPolicy::Interpreted,
            precision,
            entries: HashMap::new(),''')
    replace('''    fn evaluate(&mut self, request: &MpRequest) -> Result<(i32, Vec<u8>)> {''', '''    #[cfg(test)]
    fn with_study_policy(precision: u32, policy: StudyKernelPolicy) -> Result<Self> {
        let mut state = Self::with_precision(precision)?;
        state.study_policy = policy;
        Ok(state)
    }
    fn evaluate(&mut self, request: &MpRequest) -> Result<(i32, Vec<u8>)> {''')
    replace('''            self.entries.insert(
                (n, rank),
                MpEntry {
                    amplitude:''', '''            #[cfg(test)]
            let kernels = match self.study_policy {
                StudyKernelPolicy::Interpreted => kernels,
                StudyKernelPolicy::NativeDenominators => kernels.with_native_denominators(),
                StudyKernelPolicy::FullNative => kernels.with_native_kernels(),
            };
            #[cfg(test)]
            let study_backends = kernels.kernel_backends();
            self.entries.insert(
                (n, rank),
                MpEntry {
                    #[cfg(test)]
                    study_backends,
                    amplitude:''')
    text += '\n#[cfg(test)]\nmod native_study_tests;\n'
    path.write_text(text)
    (BRIDGE/'src/native_study_tests.rs').write_bytes((HERE/'native_study_tests.rs').read_bytes())
    snapshot = HERE/'source'
    snapshot.mkdir()
    actual = {}
    diff = []
    for path in sorted(BRIDGE.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(BRIDGE).as_posix()
        actual[name] = sha(path)
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        previous = TRANSPORT/'proposed'/name
        if not previous.exists() or previous.read_bytes() != path.read_bytes():
            diff.extend(difflib.unified_diff(previous.read_text().splitlines(True) if previous.exists() else [],
                                           path.read_text().splitlines(True), fromfile='baseline/'+name,
                                           tofile='study/'+name))
    assert {k for k in baseline if baseline[k] != actual[k]} == {'Cargo.toml', 'src/lib.rs'}
    assert set(actual)-set(baseline) == {'src/native_study_tests.rs'}
    assert actual['src/tensor.rs'] == baseline['src/tensor.rs']
    assert actual['Cargo.lock'] == baseline['Cargo.lock']
    (HERE/'study.patch').write_text(''.join(diff))
    capture = proposal['capture']
    compressed = ROOT/capture['path']
    assert sha(compressed) == capture['sha256']
    with gzip.open(compressed, 'rb') as source, (WORK/'actual-mp-requests.jsonl').open('xb') as target:
        while block := source.read(1024*1024):
            target.write(block)
    assert sha(WORK/'actual-mp-requests.jsonl') == capture['uncompressed_sha256']
    result = {'complete': True, 'compiled': False, 'executed': False, 'production_modified': False,
              'prepare_sha256': sha(Path(__file__)), 'transport_seal_sha256': sha(TRANSPORT/'manifest.json'),
              'plan_seal_sha256': sha(HERE/'plan-seal.json'), 'baseline_files': baseline, 'study_files': actual,
              'source_delta_sha256': sha(HERE/'study.patch'), 'capture': capture,
              'policies': ['Interpreted', 'NativeDenominators', 'FullNative'],
              'default_and_public_policy': 'Interpreted', 'backend_report_observed_before_into_amplitude': True}
    with (HERE/'prepared.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
