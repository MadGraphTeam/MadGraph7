"""Prepare an append-only private exporter hook; never edit production files."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-classification'
REL = Path('madgraph/loop/loop_exporters.py')
raw = (ROOT / REL).read_bytes()
source = raw.decode()
start = source.index('    def write_TIR_interface(self, writer, matrix_element):')
end = source.index('    def write_TIRUST_interface', start)
routine = source[start:end]
anchor = '        FPR = q_polynomial.FortranPolynomialRoutines(\n'
assert routine.count(anchor) == 1
addition = """        # Private classification-only export: no reducer/host call is changed.
        from madgraph.loop import kinematic_identity
        identity_table = kinematic_identity.build(matrix_element, self.group_loops)
        file += '\\n' + kinematic_identity.render(
            identity_table, replace_dict['proc_prefix'])

"""
derived = source[:start] + routine.replace(anchor, addition + anchor, 1) + source[end:]
for path, text in ((HERE / 'base' / REL, source), (HERE / 'src' / REL, derived)):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text() == text, 'refusing to replace changed private source'
    else:
        path.write_text(text)
patch = ''.join(difflib.unified_diff(source.splitlines(True), derived.splitlines(True),
                                    fromfile='a/' + str(REL), tofile='b/' + str(REL)))
helper = Path('madgraph/loop/kinematic_identity.py')
text = (HERE / 'src' / helper).read_text()
patch += ''.join(difflib.unified_diff([], text.splitlines(True),
                                    fromfile='/dev/null', tofile='b/' + str(helper)))
(HERE / 'implementation.patch').write_text(patch)
record = {'source_only': True, 'production_modified': False, 'compiler_executed': False,
          'native_executed': False, 'base': {str(REL): hashlib.sha256(raw).hexdigest()},
          'derived': {str(path): hashlib.sha256((HERE / 'src' / path).read_bytes()).hexdigest()
                      for path in (REL, helper)},
          'patch_sha256': hashlib.sha256(patch.encode()).hexdigest()}
(HERE / 'preparation.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps(record, sort_keys=True))
