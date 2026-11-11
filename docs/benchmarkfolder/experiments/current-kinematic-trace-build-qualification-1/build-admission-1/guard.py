#!/usr/bin/env python3
"""One additive admission for build plus four loader queries; never amplitudes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]
SOURCE = BASE/'source'
RESULTS = BASE/'results'
OUT = RESULTS/'build-admission-1'
SOURCE_SHA = '7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed'
DERIVED_SHA = '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184'
GENERATION_SHA = '73d32bc26ee0641faaf90c7b1060471c2339a64d9dbc33d6f064356b106a093a'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    raw = Path(path).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def verify(path, pin):
    raw = Path(path).read_bytes()
    require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()} ==
            {k:pin[k] for k in ('bytes','sha256')}, 'Pinned bytes differ: '+str(path))
    return raw


def seal(path, expected):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Reviewed index differs: '+str(path))
    data = json.loads(raw)
    for name,pin in data['files'].items():
        p = Path(name)
        require(not p.is_absolute() and '..' not in p.parts, 'Unsafe index path')
        verify(Path(path).parent/p,pin)
    return data


def save(path, data):
    with Path(path).open('x') as out:
        json.dump(data,out,sort_keys=True,indent=2,allow_nan=False)
        out.write('\n')


def admit_generation(generation_path, derived, cms):
    raw = verify(generation_path,derived['generated_source_receipt'])
    require(hashlib.sha256(raw).hexdigest() == GENERATION_SHA, 'Unreviewed generation receipt')
    generation = json.loads(raw)
    require(generation['raw_sources'] == cms['files'], 'CMS source bindings differ from generation')
    return generation


def compiler_pins(bindings):
    for name,pin in bindings['toolchain'].items():
        verify(name,pin)
    declaration = Path(bindings['dynamic_linker_declaration'])
    loader = Path(declaration.read_text().strip())
    require(loader.is_absolute() and str(loader.resolve()) == bindings['loader'],
            'Compiler dynamic-linker declaration differs')
    return loader.resolve()


def providers(runtime, bound, supplements):
    allowed = dict(bound)
    for name,pin in supplements.items():
        require(name not in allowed or allowed[name] == pin['sha256'], 'Conflicting supplemental provider')
        allowed[name] = pin['sha256']
    missing = {name:checksum for name,checksum in runtime['files'].items() if name not in allowed}
    wrong = {name:checksum for name,checksum in runtime['files'].items()
             if name in allowed and allowed[name] != checksum}
    require(not missing and not wrong, 'Unadmitted runtime providers: '+repr({'missing':missing,'wrong':wrong}))


def prepare(expected_guard):
    seal(HERE/'index.json',expected_guard)
    seal(SOURCE/'index.json',SOURCE_SHA)
    derived = seal(RESULTS/'derived/index.json',DERIVED_SHA)
    bindings = json.loads((HERE/'bindings.json').read_bytes())
    compiler_pins(bindings)
    for name,pin in bindings['supplemental_providers'].items():
        verify(name,pin)
    for name,pin in bindings['evidence'].items():
        verify(ROOT/name,pin)
    cms = json.loads((RESULTS/'derived/cms-source-bindings.json').read_bytes())
    generation = admit_generation(RESULTS/'generation.json',derived,cms)
    # Import only after authentication of all source and inherited helper pins.
    sys.path.insert(0,str(SOURCE))
    import admission
    admission.initial_admission()
    r = admission.load('ktrace_build_guard_controller',SOURCE/'run.py')
    r.verify_context()
    ewraw = verify(ROOT/r.CONTEXT['ew_compilation']['path'],r.CONTEXT['ew_compilation'])
    expected = json.loads(ewraw)
    compilers = {(c['path'],c['sha256']) for scope in expected['effective_compiler_flags']['scopes'].values()
                 for recipe in scope['recipes'].values() for c in recipe['compiler']}
    require(compilers == {(bindings['compiler'],bindings['toolchain'][bindings['compiler']]['sha256'])},
            'Actual retained compiler differs from admitted dynamic-linker declaration')
    r.exact_inventory(BASE/'work/exports/cms',cms['files'])
    ew = json.loads((SOURCE/'ew-source-bindings.json').read_bytes())
    r.exact_inventory(ROOT/r.CONTEXT['ew_donor'],ew['files'])
    modules = json.loads((SOURCE/'ew-module-inputs.json').read_bytes())
    for name,pin in modules['files'].items():
        verify(ROOT/modules['donor']/name,pin)
    for name,pin in generation['shared_configuration'].items():
        verify(name,pin)
    for name,pin in generation['dependencies_before_after'].items():
        verify(name,pin)
    for name,checksum in expected['link_inputs_after']['archives'].items():
        require(record(name)['sha256'] == checksum, 'Original EW archive changed: '+name)
    return r,bindings,{'source_index':record(SOURCE/'index.json'),
        'derived_index':record(RESULTS/'derived/index.json'),'guard_index':record(HERE/'index.json'),
        'generation':record(RESULTS/'generation.json'),'ew_compilation':record(ROOT/r.CONTEXT['ew_compilation']['path']),
        'cms_sources':len(cms['files']),'ew_sources':len(ew['files']),
        'shared_configuration':generation['shared_configuration'],
        'installed_dependencies':generation['dependencies_before_after'],
        'toolchain':bindings['toolchain'],'supplemental_providers':bindings['supplemental_providers']}


def install_hook(r, bindings, out):
    original = r.L.runtime
    expected = {str((BASE/'work/builds'/key/variant/'SubProcesses'/case[1]/'check').resolve()):key+'-'+variant
                for key,case in r.CASES.items() for variant in ('unmodified','instrumented')}
    seen = set()
    def guarded(executable, deadline=None):
        executable = Path(executable).resolve()
        require(str(executable) in expected, 'Runtime query outside four admitted build outputs')
        label = expected[str(executable)]
        require(label not in seen, 'Repeated loader query is not admitted')
        seen.add(label)
        receipt = {'executable':str(executable),'executable_pin':record(executable),'passed':False}
        error = None
        try:
            loader = compiler_pins(bindings)
            # Pure ELF parsing must establish PT_INTERP before the original --list.
            actual = r.L.interpreter(executable)
            receipt['actual_interpreter'] = str(actual)
            require(actual == loader, 'PT_INTERP differs from compiler-declared loader')
            path = RESULTS/'builds'/label/'prelink.json'
            raw = path.read_bytes()
            receipt['prelink_pin'] = record(path)
            prelink = json.loads(raw)
            require(prelink['completed_before_link'] is True, 'Prelink admission incomplete')
            bound = prelink['bound']
            r.L.q.verify_hashes(bound)
            for name,pin in bindings['supplemental_providers'].items():verify(name,pin)
            receipt['admitted_supplements'] = bindings['supplemental_providers']
            runtime = original(executable,deadline)
            receipt['runtime'] = runtime  # Retain rejected providers; never extend admission.
            require(runtime['loader'] == str(loader) and runtime['loader_sha256'] == record(loader)['sha256'],
                    'Returned loader binding differs')
            providers(runtime,bound,bindings['supplemental_providers'])
            r.L.q.verify_hashes(bound)
            for name,pin in bindings['supplemental_providers'].items():verify(name,pin)
            compiler_pins(bindings)
            require(record(executable) == receipt['executable_pin'], 'Executable changed during loader query')
            require(record(path) == receipt['prelink_pin'], 'Prelink receipt changed during loader query')
            receipt['passed'] = True
            return runtime
        except BaseException as exc:
            error = exc
            raise
        finally:
            receipt['failure'] = None if error is None else repr(error)
            save(out/(label+'-runtime.json'),receipt)
    r.L.runtime = guarded
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-guard-index',required=True)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    require(Path.cwd() == ROOT, 'Run from repository root')
    r,bindings,before = prepare(args.reviewed_guard_index)
    if not args.execute:
        print(json.dumps({'passed':True,'native_commands':0,'before':before},sort_keys=True,indent=2))
        return
    require(not OUT.exists(), 'Append-only guard invocation; no retry')
    require(not (RESULTS/'stage-build.json').exists(), 'Build already attempted; no retry')
    OUT.mkdir()
    save(OUT/'before.json',before)
    setup = r.setup_owned_helpers
    seen = set()
    def guarded_setup():
        nonlocal seen
        setup()
        seen = install_hook(r,bindings,OUT)
    r.setup_owned_helpers = guarded_setup
    oldargv = sys.argv
    failure = postfailure = None
    try:
        sys.argv = [str(SOURCE/'run.py'),'build','--execute','--reviewed-source-index',SOURCE_SHA,
                    '--reviewed-derived-index',DERIVED_SHA]
        r.main()
        require(len(seen) == 4,'Build did not perform exactly four admitted loader queries')
    except BaseException as exc:
        failure = exc
    finally:
        sys.argv = oldargv
        try:
            _,_,after = prepare(args.reviewed_guard_index)
            require(after == before, 'Protected inputs changed during build')
            save(OUT/'after.json',after)
        except BaseException as exc:
            postfailure = exc
        save(OUT/'result.json',{'passed':failure is None and postfailure is None,
            'failure':None if failure is None else repr(failure),
            'postcheck_failure':None if postfailure is None else repr(postfailure),
            'loader_queries':sorted(seen),'amplitude_evaluations':0,
            'ordinary_or_stress_admitted':False,'source_index_sha256':SOURCE_SHA,
            'derived_index_sha256':DERIVED_SHA,'guard_index_sha256':args.reviewed_guard_index})
    if failure is not None:raise failure
    if postfailure is not None:raise postfailure


if __name__ == '__main__':main()
