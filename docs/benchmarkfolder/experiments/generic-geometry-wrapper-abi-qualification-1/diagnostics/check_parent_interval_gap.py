"""Pure source counterexample. No Fortran/environment/native setter executes."""
import hashlib
import json
from pathlib import Path
import sys
import types


def f(text):
    return '\n'.join('      ' + line for line in text.strip().splitlines()) + '\n'


def check(root):
    root = Path(root)
    here = root / 'diagnostics'
    pins = json.loads((root / 'source3/parent-pins.json').read_bytes())
    expected = pins['.tirust-build/generic-invariant-provenance-proposal-2/source-index.json']
    data = (here / 'source2-index.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('source2 index mismatch')
    index = json.loads(data)
    for name in ('invariant_environment_source.py', 'invariant_window.py'):
        data = (here / name).read_bytes()
        pin = index['files']['src/madgraph/loop/' + name]
        if {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} != pin:
            raise ValueError('source2 parser changed')
    names = ('madgraph', 'madgraph.loop', 'madgraph.loop.invariant_environment_source',
             'madgraph.loop.invariant_window')
    old = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:2]:
            module = types.ModuleType(name); module.__path__ = []
            sys.modules[name] = module
        def load(name, filename):
            module = types.ModuleType(name)
            module.__file__ = str(here / filename)
            module.__package__ = name.rsplit('.', 1)[0]
            sys.modules[name] = module
            exec(compile((here / filename).read_bytes(), module.__file__, 'exec'), module.__dict__)
            return module
        env = load(names[2], 'invariant_environment_source.py')
        window = load(names[3], 'invariant_window.py')
        raw = {
            'Source/DHELAS/objects.f': f('''MODULE OBJ
TYPE NODE
REAL P(0:3),W(4)
END TYPE
END MODULE
SUBROUTINE MAKE(P,O)
USE OBJ
REAL P(0:3)
TYPE(NODE) O
O%P(:)=P(:)
END'''),
            'SubProcesses/P0/b1.f': f('''SUBROUTINE B1(P,W)
USE OBJ
REAL P(0:3)
TYPE(NODE) W(2)
CALL MAKE(P,W(1))
END'''),
            'SubProcesses/P0/b2.f': f('''SUBROUTINE B2(P,W)
USE OBJ
REAL P(0:3)
TYPE(NODE) W(2)
CALL MAKE(P,W(2))
END'''),
            'SubProcesses/P0/parent.f': f('''SUBROUTINE PARENT(P,W)
USE OBJ
REAL P(0:3)
TYPE(NODE) W(2)
CALL B1(P,W)
CALL ROUND_AWAY()
CALL B2(P,W)
CALL ROUND_BACK()
END
SUBROUTINE ROUND_AWAY()
CALL FESETROUND(2048)
END
SUBROUTINE ROUND_BACK()
CALL FESETROUND(0)
END'''),
        }
        batches = [(name, raw['SubProcesses/P0/' + name]) for name in ('b1.f', 'b2.f')]
        receipts = [{'caller': name, 'line': line, 'call': statement}
                    for name, text in batches
                    for _, line, statement in env.fixed(text, name)
                    if statement.startswith('CALLMAKE')]
        batch_result = window.momentum_window(raw, batches, receipts)
        parent_result = env.Library(raw).closure(['PARENT'])
        if batch_result['complete'] is not True or parent_result['complete'] is not False:
            raise ValueError('expected source-scope distinction not reproduced')
        if not any('mutator' in row['reason'] for row in parent_result['unknown']):
            raise ValueError('parent setter rejection missing')
        overwrite = dict(raw)
        overwrite['SubProcesses/P0/parent.f'] = f('''SUBROUTINE PARENT(P,W)
USE OBJ
REAL P(0:3)
TYPE(NODE) W(2)
CALL B1(P,W)
CALL B2(P,W)
W(1)%P(0)=99.
END''')
        overwrite_result = window.momentum_window(overwrite, batches, receipts)
        rows = [s for _, _, s in env.fixed(overwrite['SubProcesses/P0/parent.f'], 'parent')]
        if overwrite_result['complete'] is not True or rows.index('W(1)%P(0)=99.') < rows.index('CALLB2(P,W)'):
            raise ValueError('post-batch parent overwrite counterexample missing')
        return dict(schema='source-parent-interval-gap-diagnostic-v1',
                    status='pass: known scope gap reproduced',
                    batch_only_window_complete=True, parent_root_complete=False,
                    full_geometry_window_qualified=False,
                    parent_producer_to_consumer_no_overwrite_qualified=False,
                    batch_only_window_with_parent_post_write_complete=True,
                    synthetic_parent_post_batch_statement='W(1)%P(0)=99.',
                    native_or_environment_setter_executions=0,
                    source2_index_sha256=expected,
                    batch_result=batch_result, parent_result=parent_result)
    finally:
        for name, module in old.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == '__main__':
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
