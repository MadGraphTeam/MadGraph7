"""Compose private classifier and corrected mass/caller sources, no compiler."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-consumer'
MASS = ROOT / '.tirust-build/qp-mass-general-implementation'
IDENTITY = ROOT / '.tirust-build/general-kinematics-classification'
TEMPLATE = Path('madgraph/iolibs/template_files/loop_optimized')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main():
    assert sha(IDENTITY / 'source-index.json') == '97f5100a4f7a679e53a6508f669b54d4f71aa2a40868f90e435f4d932cced533'
    index = json.loads((IDENTITY / 'source-index.json').read_text())
    for name, item in index['files'].items():
        assert sha(IDENTITY / name) == item['sha256']
    assert sha(MASS / 'format-correction/loop_exporters.py') == 'f84f3aa247ec3a601e11ba2555af4f584450464c1ef46a9f135b58676326e327'
    assert sha(MASS / 'caller-guard-proposal/evidence-index.json') == '55011f0327790a36e743cebe6e4154318114509461462e9961fa629718bc5b58'
    inputs = {str((IDENTITY / 'source-index.json').relative_to(ROOT)): sha(IDENTITY / 'source-index.json')}
    for name, item in index['files'].items():
        inputs[str((IDENTITY / name).relative_to(ROOT))] = item['sha256']
    for package in ('attempt1', 'caller-guard-proposal'):
        inventory = MASS / package / 'evidence-index.json'
        record = json.loads(inventory.read_text())
        inputs[str(inventory.relative_to(ROOT))] = sha(inventory)
        for name, digest in record['files'].items():
            path = MASS / package / name
            assert sha(path) == digest, str(path)
            inputs[str(path.relative_to(ROOT))] = digest

    def copy(source, relative):
        target = HERE / 'src' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        inputs[str(source.relative_to(ROOT))] = sha(source)

    for path in (ROOT / TEMPLATE).iterdir():
        if path.is_file():
            copy(path, TEMPLATE / path.name)
    for path in (MASS / 'attempt1/src' / TEMPLATE).iterdir():
        if path.is_file():
            copy(path, TEMPLATE / path.name)
    copy(MASS / 'caller-guard-proposal/src' / TEMPLATE / 'helas_calls_split.inc',
         TEMPLATE / 'helas_calls_split.inc')
    copy(MASS / 'caller-guard-proposal/src/madgraph/iolibs/helas_call_writers.py',
         Path('madgraph/iolibs/helas_call_writers.py'))
    copy(IDENTITY / 'src/madgraph/loop/kinematic_identity.py', Path('madgraph/loop/kinematic_identity.py'))
    base_path = MASS / 'format-correction/loop_exporters.py'
    inputs[str(base_path.relative_to(ROOT))] = sha(base_path)
    original = base_path.read_text()
    source = once(original,
        "        return ''.join(interfaces)\n\n    def write_CT_interface",
        "        from madgraph.loop import kinematic_consumer\n"
        "        mapping = dict(replace_dict, qp_mass_suffix='_QPMASS_KIN',\n"
        "                       qp_mass_format=replace_dict['mass_mp_format'])\n"
        "        interfaces.append(misc.apply_template(\n"
        "            kinematic_consumer.adapter_template(template, template_name), mapping))\n"
        "        return ''.join(interfaces)\n\n    def write_CT_interface")
    begin = source.index('    def write_CT_interface(')
    end = source.index('    # Helper function to split HELAS CALLS', begin)
    method = source[begin:end]
    method = once(method, '                files.append(misc.apply_template(file, qp))',
        '                from madgraph.loop import kinematic_consumer\n'
        '                files.append(kinematic_consumer.connect_paired_helper(\n'
        '                    misc.apply_template(file, qp)))')
    method = once(method, '        file="\\n".join(files)\n',
        '        file="\\n".join(files)\n'
        '        if optimized_output:\n'
        '            if self.opt[\'vector_size\']:\n'
        '                raise ValueError("Private kinematic consumer requires scalar export")\n'
        '            from madgraph.loop import kinematic_consumer\n'
        '            file += kinematic_consumer.render(matrix_element, self.group_loops,\n'
        '                matrix_element.rep_dict[\'proc_prefix\'])\n')
    source = source[:begin] + method + source[end:]
    target = HERE / 'src/madgraph/loop/loop_exporters.py'
    target.write_text(source)
    (HERE / 'base').mkdir(exist_ok=True)
    (HERE / 'base/loop_exporters.py').write_text(original)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), source.splitlines(True),
                        fromfile='a/madgraph/loop/loop_exporters.py', tofile='b/madgraph/loop/loop_exporters.py'))
    helper = HERE / 'src/madgraph/loop/kinematic_consumer.py'
    patch += ''.join(difflib.unified_diff([], helper.read_text().splitlines(True),
                        fromfile='/dev/null', tofile='b/madgraph/loop/kinematic_consumer.py'))
    (HERE / 'consumer.patch').write_text(patch)
    record = {'source_only': True, 'compiler_executed': False, 'native_executed': False,
              'production_modified': False, 'inputs': inputs,
              'derived': {'exporter': sha(target), 'consumer': sha(helper), 'patch': sha(HERE / 'consumer.patch')},
              'composition': 'Corrected paired mass exporter+attempt1 templates+guarded caller; append private KIN variants and integer metadata; original public entries retained.'}
    (HERE / 'preparation.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record['derived']))


if __name__ == '__main__':
    main()
