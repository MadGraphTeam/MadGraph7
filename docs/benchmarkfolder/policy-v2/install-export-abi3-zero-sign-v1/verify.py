#!/usr/bin/env python3
"""Check retained installation/export evidence without executing a library."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def coefficients(text):
    values = {}
    for key, label in (('born', 'Born'), ('finite', 'Finite'),
                       ('single_pole', '1EPS'), ('double_pole', '2EPS')):
        matches = re.findall(rf'(?im)^\s*Matrix element {label}\s*=\s*([+-]?[0-9.]+(?:[dDeE][+-]?\d+)?)', text)
        assert matches, key
        values[key] = float(matches[-1].replace('D', 'E').replace('d', 'e'))
    values['return_code'] = int(re.findall(r'MadLoop return code\s*=\s*(\d+)', text)[-1])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Also rehash preserved installation and executables')
    args = parser.parse_args()
    manifest = read('manifest.json')
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and '__pycache__' not in p.parts and p.name != 'manifest.json'}
    assert actual == set(manifest['files'])
    for name, record in manifest['files'].items():
        p = HERE / name
        assert not p.is_symlink() and p.stat().st_size == record['bytes']
        assert sha(p) == record['sha256'], name
    install = read('installation.json')
    assert install['revision'] == '9f13d10cca5ab6ae5410b6db1f146fa8eddba9b3'
    assert install['passed'] and install['compiled_smoke_passed'] and install['exit_code'] == 0
    assert install['abi_version'] == 3 and install['cargo_jobs'] == 1
    assert install['cli_log_sha256'] == sha(HERE / 'install-cli.log')
    assert install['cli_card_sha256'] == sha(HERE / 'install.mg5')
    assert 'NinjaRS ABI3 native/compensated DP and genuine binary128-input rescue passed' in (HERE / 'install.log').read_text()
    for name in ('install.log', 'Cargo.lock', 'Cargo.toml'):
        assert sha(HERE / name) == install['files'][name]
    exports = read('exports.json')
    assert exports['source_revision'] == install['revision']
    assert exports['installation_sha256'] == sha(HERE / 'installation.json')
    assert set(exports['exports']) == {'standalone', 'nlo'}
    for name, record in exports['exports'].items():
        assert record['exit_code'] == 0
        assert record['card_sha256'] == sha(HERE / (name + '.mg5'))
        assert record['log_sha256'] == sha(HERE / (name + '-export.log'))
        assert 'set output_dependencies internal' in (HERE / (name + '.mg5')).read_text()
    relocation = read('relocation.json')
    hidden = read('installation-hidden.json')
    assert relocation['original_output_removed'] and relocation['original_nlo_output_removed']
    assert hidden['original_installation_path_absent']
    assert set(relocation['copied_files']) == {'libninjars.a', 'mninjars.mod', 'ninjars.h',
        'mninjars.f90', 'ninjars-config.json', 'ninjars-link-before.txt', 'ninjars-native-libs.txt'}
    assert all(r['regular_file'] for r in relocation['copied_files'].values())
    assert relocation['copied_files']['libninjars.a']['sha256'] == install['files']['lib/libninjars.a']
    observed = read('standalone-results.json')
    reference = read('inputs/cuttools-reference.json')
    for label, status in (('dp', 110), ('qp', 120), ('stable', 218)):
        assert coefficients((HERE / f'standalone-{label}.log').read_text()) == observed[label]
        assert observed[label]['return_code'] == status
        for name, old in (('born', 'born'), ('finite', 'finite'),
                          ('single_pole', '1eps'), ('double_pole', '2eps')):
            assert abs(observed[label][name] - reference[old]) <= 1e-11 * max(1, abs(reference[old]))
    runtime = read('runtime-results.json')
    assert runtime['standalone'] == observed
    assert runtime['nlo_cli_success'] and runtime['pole_checks_passed'] == runtime['pole_checks_tested'] == 20
    assert 'Poles successfully cancel for 20 points over 20' in (HERE / 'nlo-compile.log').read_text()
    assert 'Fraction of failures: 0.00' in (HERE / 'nlo-test_ME.log').read_text()
    assert 'FAILED' not in (HERE / 'nlo-test_ME.log').read_text()
    assert re.search(r'>\s*MLReductionLib\s*=\s*8\b', (HERE / 'nlo-check_poles.log').read_text())
    for name in ('standalone', 'nlo-check_poles', 'nlo-madevent_mintFO'):
        nm = (HERE / (name + '-nm.txt')).read_text()
        assert re.search(r'\bU __(?:add|sub|mul|div)tf3@GCC_', nm)
        linked = (HERE / (name + '-ldd.txt')).read_text()
        assert 'libgcc_s.so' in linked and 'not found' not in linked
    focused = read('focused-unit-results.json')
    assert focused['passed'] and focused['python_tests'] == 16
    assert focused['log_sha256'] == sha(HERE / 'focused-unit-tests.log')
    proof = focused['corrected_transport_suite']
    assert (proof['rust_tests_passed'], proof['requests_replayed'], proof['unchanged_output_fields']) == (18, 1392, 11136)
    assert proof['tested_and_archived_build_input_sha256'] == install['archived_bridge_source_sha256']
    assert proof['installation_sha256'] == sha(HERE / 'installation.json')
    if args.live:
        work = ROOT / '.benchmark-work/policy-v2/install-export-abi3-zero-sign-v1'
        assert not (work / 'source/HEPTools/ninjars').exists()
        source = work / 'source/HEPTools/ninjars-unavailable'
        for name, expected in install['files'].items():
            assert sha(source / name) == expected, name
        for output in ('standalone-relocated', 'nlo-relocated'):
            archive = work / output / 'lib/libninjars.a'
            assert archive.is_file() and not archive.is_symlink()
            assert sha(archive) == install['files']['lib/libninjars.a']
        assert sha(work / 'standalone-relocated/SubProcesses/P0_uux_z/check') == relocation['standalone_symbols']['executable_sha256']
        for name, record in runtime['nlo_symbols'].items():
            assert sha(work / 'nlo-relocated/SubProcesses/P0_uux_z' / name) == record['executable_sha256']
    print(json.dumps({'passed': True, 'files': len(actual), 'live': args.live,
                      'standalone_statuses': [110, 120, 218], 'nlo_poles': 20,
                      'compiled_smoke': True, 'engine_executions': 0}))


if __name__ == '__main__':
    main()
