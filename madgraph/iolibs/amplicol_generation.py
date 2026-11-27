################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""Generation of the '[amplicol]' artifacts with pyamplicol, run in the process
of export_amplicol. pyamplicol (and Symbolica) are imported here, after
bootstrap.ensure_pyamplicol put them on sys.path."""

import json
import logging


# least recently used compiled models beyond this many leave the shared cache
SHARED_MODELS = 8


def model_source(pyamplicol, model):
    """The model compiled for pyamplicol, cached in the output (model['cache'])
    under the key model['key'], taken from the shared cache of the outputs
    (model['shared_cache'], see export_amplicol.model_job) or compiled there."""
    import os
    try:
        with open(model['key_file']) as stream:
            cached = json.load(stream).get('key')
    except (OSError, ValueError, AttributeError):
        cached = None
    if cached != model['key'] or not os.path.exists(model['cache']):
        if os.path.exists(model['key_file']):
            os.remove(model['key_file'])
        _remove(model['cache'])
        shared = None
        if model.get('shared_cache'):
            try:
                shared = _shared_model(pyamplicol, model, model['shared_cache'])
            except OSError as error:
                logging.getLogger('madgraph.amplicol').warning(
                    'Cannot use the compiled-model cache %s: %s',
                    model['shared_cache'], error)
        if shared:
            _copy_model(shared, model['cache'])
        else:
            _compile_model(pyamplicol, model, model['cache'])
        with open(model['key_file'], 'w') as stream:
            json.dump({'key': model['key'], 'sources': model['sources']}, stream,
                      indent=2)
    return pyamplicol.ModelSource.from_path(model['cache'])


def _compile_model(pyamplicol, model, output):
    from pyamplicol.config import EvaluatorConfig, JITConfig
    source = pyamplicol.ModelSource.from_path(model['path'],
                                              restriction=model['restriction'])
    level = model.get('jit_optimization_level', 2)
    source.compile(prepared_output=output,
                   evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=level)))


def _shared_model(pyamplicol, model, cache_dir):
    """The compiled model of model['key'] in the shared cache, compiled and
    published there (model file, then its sidecar json, each renamed into
    place) unless a valid one is there already."""
    import os
    import tempfile
    from madgraph.iolibs.export_amplicol import shared_model
    key = model['key']
    path = shared_model(cache_dir, key)
    if path:
        try:
            os.utime(os.path.join(cache_dir, key + '.json'))
        except OSError:
            pass  # a read-only cache
        return path
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, key + '.pyamplicol-model')
    fd, temporary = tempfile.mkstemp(prefix='.%s.' % key, suffix='.pyamplicol-model',
                                     dir=cache_dir)
    os.close(fd)
    try:
        _compile_model(pyamplicol, model, temporary)
        os.replace(temporary, path)
        record = {'key': key, 'sources': model['sources'],
                  'size': os.path.getsize(path)}
        fd, temporary = tempfile.mkstemp(prefix='.%s.' % key, suffix='.json',
                                         dir=cache_dir)
        with os.fdopen(fd, 'w') as stream:
            json.dump(record, stream, indent=2)
        os.replace(temporary, os.path.join(cache_dir, key + '.json'))
    finally:
        _remove(temporary)
    _prune(cache_dir)
    return path


def _prune(cache_dir):
    """Remove the least recently used compiled models beyond SHARED_MODELS."""
    import os
    try:
        records = sorted((entry for entry in os.scandir(cache_dir)
                          if entry.name.endswith('.json')
                          and not entry.name.startswith('.')),
                         key=lambda entry: entry.stat().st_mtime, reverse=True)
    except OSError:
        return
    for entry in records[SHARED_MODELS:]:
        _remove(entry.path)
        _remove(entry.path[:-len('.json')] + '.pyamplicol-model')


def _copy_model(source, destination):
    """A hard link to the shared compiled model, a copy where links fail."""
    import os
    import shutil
    try:
        os.link(source, destination)
    except OSError:
        temporary = destination + '.tmp'
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)


def _remove(path):
    import os
    import shutil
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def generate_artifact(pyamplicol, artifact, accuracy, process, model,
                      flavor_unions=True, workers=None, flavor_union_crossing=False,
                      selector_only_full_colour=False):
    """Generate one artifact. flavor_union_crossing is off unless a job sets it;
    the MG default is in MadGraphCmd.options_madgraph."""
    from pyamplicol.config import resolve_config
    # workers and cores are left to pyamplicol, which clamps them without a
    # license, unless ``workers`` splits the CPUs between concurrent artifacts
    options = {'multiparticles': process['multiparticles'],
               'flavor_blind_strict': True,
               'coupling_order_policy': 'explicit',
               'max_coupling_orders': process['orders']}
    if flavor_unions:
        # one pattern-resolved template per quark-flavour union
        options['flavor_unions'] = True
    if flavor_union_crossing:
        # unions by all-outgoing content (pyamplicol: lc only, needs flavor_unions)
        options['flavor_union_crossing'] = True
    config = {
        'schema_version': 1,
        'action': 'generate',
        'color': {'accuracy': accuracy},
        'process': options,
        # pack_payloads: one payloads.pacbin instead of files per process
        # no numerical check of the generated artifact: its cost grows with the
        # number of colour orderings (hours for p p > 7j)
        'generation': {'emit_api_bundle': True, 'pack_payloads': True,
                       'relation_discovery': {'mode': 'off'},
                       'validation': {'enabled': False}},
        'evaluator': {'backend': 'jit', 'execution_mode': 'recurrence',
                      'jit': {'optimization_level': 2}},
    }
    if selector_only_full_colour:
        # full colour without the main schedule: one helicity per point only
        config['generation']['selector_only_full_colour'] = True
    if workers is not None:
        config['generation']['workers'] = workers
        config['evaluator']['optimization'] = {'cores': 1}
    pyamplicol.Generator(resolve_config(config)).generate(
        process['expression'], artifact, model=model, mode='replace')


# runs generate_child in a new interpreter, the job given as json
_CHILD = ('import json, sys; job = json.loads(sys.argv[1]); sys.path[:0] = job["sys_path"]; '
          'from madgraph.iolibs.amplicol_generation import generate_child; '
          'generate_child(job)')


def generate_child(job):
    """The artifact of a job of start_artifact, in a process of its own."""
    import pyamplicol
    # only when set, so that the call is unchanged without it
    extra = {'selector_only_full_colour': True} if job.get('selector_only_full_colour') else {}
    generate_artifact(pyamplicol, job['artifact'], job['accuracy'], job['process'],
                      pyamplicol.ModelSource.from_path(job['model']),
                      workers=job['workers'],
                      flavor_unions=job.get('flavor_unions', True),
                      flavor_union_crossing=job.get('flavor_union_crossing', False),
                      **extra)


class ArtifactProcess(object):
    """An artifact generated by generate_child in a child process: its Python
    work does not share the GIL of this one."""

    def __init__(self, artifact, accuracy, process, model_path, workers,
                 flavor_unions=True, flavor_union_crossing=False,
                 selector_only_full_colour=False):
        import subprocess
        import sys
        import tempfile
        job = {'sys_path': sys.path, 'artifact': artifact, 'accuracy': accuracy,
               'process': process, 'model': model_path, 'workers': workers,
               'flavor_unions': bool(flavor_unions),
               'flavor_union_crossing': bool(flavor_union_crossing),
               'selector_only_full_colour': bool(selector_only_full_colour)}
        self.artifact = artifact
        # a file, not a pipe, which would fill up while this process generates
        self.log = tempfile.TemporaryFile(mode='w+')
        self.child = subprocess.Popen([sys.executable, '-c', _CHILD, json.dumps(job)],
                                      stdout=self.log, stderr=subprocess.STDOUT)

    def wait(self):
        code = self.child.wait()
        self.log.seek(0)
        output = self.log.read().strip()
        self.log.close()
        if code != 0:
            raise RuntimeError('the generation of %s failed (exit code %d):\n%s'
                               % (self.artifact, code, output[-5000:]))
        if output:
            logging.getLogger('madgraph.amplicol').info(output)

    def kill(self):
        if self.child.poll() is None:
            self.child.kill()
            self.child.wait()
        self.log.close()


def generate(job):
    """Generate the artifacts of the processes of a job (see
    export_amplicol.export_amplicol_processes) and return the reweight providers
    of each of them. With a license the full-colour artifact is generated in a
    child process while this one generates the leading-colour one, sharing the
    CPUs."""
    import os
    import pyamplicol
    from pyamplicol import licensing
    from pyamplicol.artifacts import umami_reweight
    licensed = licensing.detect_symbolica_license(suggest=False).licensed
    if not licensed:
        logging.getLogger('madgraph.amplicol').warning(
            'Symbolica does not accept the license key of pyamplicol (%s): '
            'generating in its restricted mode', pyamplicol.__file__)
    model = model_source(pyamplicol, job['model'])
    # restricted mode: one generation at a time
    budget = licensing._cpu_budget()
    concurrent = licensed and budget > 1
    providers = []
    for process in job['processes']:
        path = process['path']
        flavor_unions = job.get('flavor_unions', True)
        # crossing for lc only until full-colour crossing exists
        crossing = {'lc': job.get('flavor_union_crossing', False), 'full': False}
        # the reweight step evaluates full colour one helicity per point only
        selector = {'lc': False, 'full': job.get('selector_only_full_colour', False)}
        if concurrent:
            full = ArtifactProcess(os.path.join(path, 'full'), 'full', process,
                                   job['model']['cache'], budget // 2,
                                   flavor_unions=flavor_unions,
                                   flavor_union_crossing=crossing['full'],
                                   selector_only_full_colour=selector['full'])
            try:
                generate_artifact(pyamplicol, os.path.join(path, 'lc'), 'lc', process,
                                  model, flavor_unions=flavor_unions,
                                  workers=budget - budget // 2,
                                  flavor_union_crossing=crossing['lc'],
                                  selector_only_full_colour=selector['lc'])
                full.wait()
            except BaseException:
                full.kill()
                raise
        else:
            for accuracy in ('lc', 'full'):
                generate_artifact(pyamplicol, os.path.join(path, accuracy), accuracy,
                                  process, model, flavor_unions=flavor_unions,
                                  flavor_union_crossing=crossing[accuracy],
                                  selector_only_full_colour=selector[accuracy])
        # the flavour switch of lc_flavors = two_lines_resample
        providers.append(umami_reweight.write_reweight_provider(
            os.path.join(path, 'lc'), os.path.join(path, 'full'),
            os.path.join(path, 'reweight'),
            switch_lines=process.get('switch_lines'),
            initial_legs=process.get('initial_legs')))
    return providers
