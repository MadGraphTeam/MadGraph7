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
"""mg7 output of the processes computed by AmpliCol ('[amplicol]').

pyamplicol generates the colour-ordered matrix elements of a process definition
in SubProcesses/A<n>_<name>: a leading-colour artifact (lc), a full-colour one
(full), and the sources of the UMAMI library returning their ratio (reweight).
Its makefile builds the libraries into lib/. Every colour ordering (UMAMI
channel) of every provider becomes one entry of subprocesses.json, integrated
on its own with a colour-ordered phase-space mapping.
"""

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import madgraph

logger = logging.getLogger('madgraph.export_amplicol')
pjoin = os.path.join

MODEL_FILE = 'amplicol_model.pyamplicol-model'

MAKEFILE = """\
# Builds the AmpliCol matrix elements of this process into ../../lib.
RUSTICOL_CONFIG ?= %(rusticol_config)s
BUILD := $(CURDIR)/build

all:
\t$(MAKE) -C lc/API/umami shared BUILD_DIR=$(BUILD)/lc RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)
\t$(MAKE) -C reweight shared BUILD_DIR=$(BUILD)/reweight RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)
%(copies)s
clean:
\trm -rf $(BUILD)
"""


WORKER = pjoin(os.path.dirname(os.path.abspath(__file__)), 'amplicol_worker.py')
# printed by Symbolica when another process holds its per-user thread
SYMBOLICA_BUSY = 'Cannot start another restricted Symbolica thread'


def check_pyamplicol():
    """pyamplicol is only imported by the worker, see amplicol_worker."""
    if importlib.util.find_spec('pyamplicol') is None:
        raise madgraph.InvalidCmd(
            'The [amplicol] processes need pyamplicol. Install it in the Python '
            'environment of MadGraph.')


def run_worker(job):
    """Run amplicol_worker on a job, waiting while Symbolica is busy elsewhere."""
    env = dict(os.environ, SYMBOLICA_HIDE_BANNER='1',
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    waiting = False
    while True:
        worker = subprocess.Popen([sys.executable, WORKER, job], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True)
        busy = False
        for line in worker.stdout:
            if SYMBOLICA_BUSY in line:
                busy = True
            elif not line.startswith(('\u250c', '\u2502', '\u2514', '\u251c')):
                sys.stdout.write(line)
        if worker.wait() == 0:
            return
        if not busy:
            raise madgraph.InvalidCmd('pyamplicol failed to generate the [amplicol] '
                                      'matrix elements, see the messages above')
        if 'symbolica' in sys.modules:
            raise madgraph.InvalidCmd(
                'Symbolica is imported in this MadGraph process, so the pyamplicol '
                'worker cannot start: do not import pyamplicol or symbolica here')
        if not waiting:
            logger.info('Symbolica is used by another process: waiting for it')
            waiting = True
        time.sleep(10)


def load_metadata(artifact):
    """The UMAMI metadata documents of an artifact, one per provider."""
    root = pjoin(artifact, 'API', 'umami')
    with open(pjoin(root, 'metadata.json')) as stream:
        data = json.load(stream)
    if data.get('kind') != 'pyamplicol-umami-provider-index':
        return [data]
    documents = []
    for provider in data['providers']:
        with open(pjoin(root, provider['metadata'])) as stream:
            documents.append(json.load(stream))
    return documents


def leg_pdgs(leg, model):
    """The pdg ids of a leg: the flavours MG merged into a group id, or the ids
    with the group ids expanded."""
    if leg.get('flavor'):
        return list(leg.get('flavor'))
    merged = model.get('merged_particles')
    pdgs = []
    for pid in leg.get('ids'):
        members = merged.get(abs(pid), [abs(pid)])
        pdgs.extend(member if pid > 0 else -member for member in members)
    return pdgs


def process_name(procdef, model, multiparticles):
    """A directory name for a process definition, e.g. 'pp_jjj', with the
    multiparticle labels of the interface."""
    particles = model.get('particle_dict')
    labels = []
    for leg in procdef.get('legs'):
        pdgs = leg_pdgs(leg, model)
        matches = [label for label, ids in multiparticles.items()
                   if sorted(ids) == sorted(pdgs)]
        preferred = 'j' if leg.get('state') else 'p'
        label = preferred if preferred in matches else next(iter(matches), None)
        if label is None:
            label = ''.join(particles[pid].get_name() for pid in pdgs)
        labels.append((leg.get('state'), label))
    name = ''.join(l for state, l in labels if not state) + '_' + \
        ''.join(l for state, l in labels if state)
    return re.sub(r'[^A-Za-z0-9_]', '',
                  name.replace('~', 'x').replace('+', 'p').replace('-', 'm'))


def process_expression(procdef, model):
    """The process in pyamplicol syntax, and its multiparticle definitions."""
    particles = model.get('particle_dict')
    multiparticles = {}
    labels = []
    for leg in procdef.get('legs'):
        names = [particles[pid].get_name() for pid in leg_pdgs(leg, model)]
        if len(names) == 1:
            labels.append(names[0])
            continue
        label = next((key for key, value in multiparticles.items()
                      if value == names), None)
        if label is None:
            label = 'mp%d' % len(multiparticles)
            multiparticles[label] = names
        labels.append(label)
    initial = [label for leg, label in zip(procdef.get('legs'), labels)
               if not leg.get('state')]
    final = [label for leg, label in zip(procdef.get('legs'), labels)
             if leg.get('state')]
    return '%s > %s' % (' '.join(initial), ' '.join(final)), multiparticles


def coupling_orders(procdef, model):
    """The largest orders of the couplings above the lowest hierarchy, as MG
    selects them by default: the minimal WEIGHTED order of the definition,
    with n - 2 vertices at tree level."""
    import madgraph.core.diagram_generation as diagram_generation
    weighted = diagram_generation.MultiProcess.find_optimal_process_orders(
        procdef)['WEIGHTED']
    hierarchy = model.get('order_hierarchy')
    lowest = min(hierarchy.values())
    vertices = len(procdef.get('legs')) - 2
    return {order: max((weighted - vertices * lowest) // (weight - lowest), 0)
            for order, weight in hierarchy.items() if weight > lowest}


def model_job(model, cache_path):
    """The UFO model of the output with its restriction, compiled for
    pyamplicol once per output."""
    path = model.get('modelpath')
    reference = os.path.basename(model.get('modelpath+restriction'))
    restriction = 'default'
    if '-' in reference:
        suffix = reference.split('-', 1)[1]
        if suffix == 'full':
            restriction = 'none'
        else:
            restriction = pjoin(path, 'restrict_%s.dat' % suffix)
    if not os.path.exists(cache_path):
        logger.info('Compiling the model %s for AmpliCol' % reference)
    return {'path': path, 'restriction': restriction, 'cache': cache_path}


def representative_processes(proc_defs, model):
    """Processes standing for the [amplicol] definitions where the output needs
    MG processes (run card defaults, proc_characteristics): every initial state,
    with the first particle of each final-state leg."""
    import itertools
    import madgraph.core.base_objects as base_objects
    processes = []
    for procdef in proc_defs:
        legs = procdef.get('legs')
        initial = [leg_pdgs(leg, model) for leg in legs if not leg.get('state')]
        final = [leg_pdgs(leg, model)[0] for leg in legs if leg.get('state')]
        for ids in itertools.product(*initial):
            pdgs = list(ids) + final
            processes.append(base_objects.Process({
                'legs': base_objects.LegList([
                    base_objects.Leg({'id': pdg, 'number': number + 1,
                                      'state': number >= len(ids)})
                    for number, pdg in enumerate(pdgs)]),
                'model': model,
                'id': procdef.get('id'),
            }))
    return processes


def color_types(model, pdgs):
    types = {}
    for pdg in pdgs:
        particle = model.get_particle(pdg)
        types[pdg] = particle.get_color()
        types[-pdg] = model.get_particle(particle.get_anti_pdg_code()).get_color()
    return types


def mirrored(pdgs, incoming, initial_legs, exported):
    """Whether the process with swapped beams is requested but not exported:
    pyamplicol keeps one process per unordered initial state."""
    if incoming != 2 or pdgs[0] == pdgs[1]:
        return False
    swapped = ((pdgs[1], pdgs[0]), tuple(sorted(pdgs[2:])))
    return (pdgs[1] in initial_legs[0] and pdgs[0] in initial_legs[1]
            and swapped not in exported)


def subprocess_entries(metadata, model, rel_dir, me_path, ratio_path, qcd_power,
                       initial_legs):
    """One subprocesses.json entry per channel of a UMAMI provider; initial_legs
    are the pdg ids allowed for each initial leg."""
    pdg_ids = metadata['pdg_ids']
    provider = metadata['provider']
    incoming = provider['incoming_count']
    exported = {(tuple(pdgs[:incoming]), tuple(sorted(pdgs[incoming:])))
                for pdgs in pdg_ids}
    entries = []
    for channel_index, channel in enumerate(metadata['channels']):
        flavors = []
        for flavor_index, process in enumerate(channel['processes']):
            if process['multichannels'] != [channel_index]:
                raise madgraph.MadGraph5Error(
                    'AmpliCol contributions shared by several channels are not '
                    'supported yet')
            for term in process['matrix_elements']:
                pdgs = pdg_ids[term['pdg_ids']]
                flavors.append({
                    'index': flavor_index,
                    'options': [pdgs],
                    'factor': term['factor'],
                    'mirror': mirrored(pdgs, incoming, initial_legs, exported),
                })
        first = flavors[0]['options'][0]
        entries.append({
            'backend': 'amplicol',
            'incoming': first[:incoming],
            'outgoing': first[incoming:],
            'channels': [],
            'path': rel_dir,
            'me_path': me_path,
            'ratio_path': ratio_path,
            'artifact': pjoin(rel_dir, 'lc'),
            'ratio_artifact': pjoin(rel_dir, 'reweight'),
            'umami_channel': channel_index,
            'phasespace_order': channel['phasespace_order'],
            'masses': provider['masses'],
            'supports_alpha_s': provider['supports_alpha_s'],
            'flavors': flavors,
            'qcd_power': qcd_power,
            'color_flows': metadata['color_flows'],
            'pdg_color_types': color_types(
                model, {abs(pdg) for pdgs in pdg_ids for pdg in pdgs}),
            'helicities': metadata['helicities'],
            'diagram_count': 0,
            'interference': False,
            'parameters': parameter_references(metadata),
        })
    return entries


def parameter_references(metadata):
    """The parameter card entry of every mutable runtime parameter."""
    references = {}
    for process in metadata['runtime_processes']:
        for parameter in process['parameters']:
            if parameter.get('mutable') and parameter.get('lhablock'):
                references[parameter['name']] = [parameter['lhablock'],
                                                 parameter['lhacode']]
    return references


def export_amplicol_processes(dir_path, proc_defs, model, multiparticles=None):
    """Write SubProcesses/A<n>_<name> for every '[amplicol]' process definition
    and return the subprocesses.json entries."""
    check_pyamplicol()
    subproc_dir = pjoin(dir_path, 'SubProcesses')
    rusticol_config = shutil.which('rusticol-config') or 'rusticol-config'
    processes = []
    for number, procdef in enumerate(proc_defs, start=1):
        name = 'A%d_%s' % (number, process_name(procdef, model, multiparticles or {}))
        path = pjoin(subproc_dir, name)
        expression, process_multiparticles = process_expression(procdef, model)
        os.makedirs(path, exist_ok=True)
        logger.info('Generating the AmpliCol matrix elements of %s in %s' %
                    (procdef.nice_string(prefix=False), path))
        processes.append({'name': name, 'path': path, 'expression': expression,
                          'multiparticles': process_multiparticles,
                          'orders': coupling_orders(procdef, model)})
    with tempfile.TemporaryDirectory(prefix='amplicol_job_') as scratch:
        job = pjoin(scratch, 'job.json')
        with open(job, 'w') as stream:
            json.dump({'model': model_job(model, pjoin(subproc_dir, MODEL_FILE)),
                       'processes': processes,
                       'result': pjoin(scratch, 'providers.json')}, stream)
        run_worker(job)
        with open(pjoin(scratch, 'providers.json')) as stream:
            providers = json.load(stream)
    entries = []
    for procdef, process, provider_ids in zip(proc_defs, processes, providers):
        name, path, orders = process['name'], process['path'], process['orders']
        rel_dir = pjoin('SubProcesses', name)
        # every vertex carries one coupling: the rest are powers of g_s
        qcd_power = (len(procdef.get('legs')) - 2 - sum(orders.values())
                     if len(orders) == 1 else -1)
        initial_legs = [set(leg_pdgs(leg, model))
                        for leg in procdef.get('legs') if not leg.get('state')]
        copies = []
        for provider_id, metadata in zip(provider_ids, load_metadata(pjoin(path, 'lc'))):
            library = metadata['provider']['library']
            me_path = pjoin('lib', 'libamplicol_%s_%s.so' % (name, provider_id))
            ratio_path = pjoin('lib', 'libamplicol_%s_%s_ratio.so' %
                               (name, provider_id))
            copies.append('\tcp $(BUILD)/lc/%s ../../%s' % (library, me_path))
            copies.append('\tcp $(BUILD)/reweight/libreweight_%s.so ../../%s' %
                          (provider_id, ratio_path))
            entries.extend(subprocess_entries(metadata, model, rel_dir, me_path,
                                              ratio_path, qcd_power, initial_legs))
        with open(pjoin(path, 'makefile'), 'w') as stream:
            stream.write(MAKEFILE % {'rusticol_config': rusticol_config,
                                     'copies': '\n'.join(copies)})
        with open(pjoin(path, 'process.json'), 'w') as stream:
            json.dump({'process': procdef.nice_string(prefix=False),
                       'expression': process['expression'],
                       'multiparticles': process['multiparticles'],
                       'max_coupling_orders': orders}, stream, indent=2)
    return entries
