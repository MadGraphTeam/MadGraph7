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

import json
import logging
import os
import re
import shutil

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


def import_pyamplicol():
    try:
        import pyamplicol
        from pyamplicol.artifacts import umami_reweight
    except ImportError as error:
        raise madgraph.InvalidCmd(
            'The [amplicol] processes need pyamplicol (%s). Install it in the '
            'Python environment of MadGraph.' % error)
    return pyamplicol, umami_reweight


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


def model_source(pyamplicol, model, cache_path):
    """The UFO model of the output with its restriction, compiled for
    pyamplicol once per output."""
    if os.path.exists(cache_path):
        return pyamplicol.ModelSource.from_path(cache_path)
    path = model.get('modelpath')
    reference = os.path.basename(model.get('modelpath+restriction'))
    restriction = 'default'
    if '-' in reference:
        suffix = reference.split('-', 1)[1]
        if suffix == 'full':
            restriction = 'none'
        else:
            restriction = pjoin(path, 'restrict_%s.dat' % suffix)
    from pyamplicol.config import EvaluatorConfig, JITConfig
    source = pyamplicol.ModelSource.from_path(path, restriction=restriction)
    logger.info('Compiling the model %s for AmpliCol' % reference)
    source.compile(prepared_output=cache_path,
                   evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=2)))
    return pyamplicol.ModelSource.from_path(cache_path)


def generate_artifact(pyamplicol, artifact, accuracy, expression, multiparticles,
                      orders, model):
    from pyamplicol.config import resolve_config
    # one worker: Symbolica's restricted mode allows a single thread
    config = resolve_config({
        'schema_version': 1,
        'action': 'generate',
        'color': {'accuracy': accuracy},
        'process': {'multiparticles': multiparticles,
                    'coupling_order_policy': 'explicit',
                    'max_coupling_orders': orders},
        'generation': {'workers': 1, 'emit_api_bundle': True,
                       'relation_discovery': {'mode': 'off'}},
        'evaluator': {'backend': 'jit', 'execution_mode': 'recurrence',
                      'optimization': {'cores': 1},
                      'jit': {'optimization_level': 2}},
    })
    pyamplicol.Generator(config).generate(expression, artifact, model=model,
                                          mode='replace')


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


def subprocess_entries(metadata, model, rel_dir, me_path, ratio_path, qcd_power):
    """One subprocesses.json entry per channel of a UMAMI provider."""
    pdg_ids = metadata['pdg_ids']
    provider = metadata['provider']
    incoming = provider['incoming_count']
    entries = []
    for channel_index, channel in enumerate(metadata['channels']):
        flavors = []
        for flavor_index, process in enumerate(channel['processes']):
            if process['multichannels'] != [channel_index]:
                raise madgraph.MadGraph5Error(
                    'AmpliCol contributions shared by several channels are not '
                    'supported yet')
            for term in process['matrix_elements']:
                flavors.append({
                    'index': flavor_index,
                    'options': [pdg_ids[term['pdg_ids']]],
                    'factor': term['factor'],
                    'mirror': False,
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
    pyamplicol, umami_reweight = import_pyamplicol()
    subproc_dir = pjoin(dir_path, 'SubProcesses')
    source = model_source(pyamplicol, model, pjoin(subproc_dir, MODEL_FILE))
    rusticol_config = shutil.which('rusticol-config') or 'rusticol-config'
    entries = []
    for number, procdef in enumerate(proc_defs, start=1):
        name = 'A%d_%s' % (number, process_name(procdef, model, multiparticles or {}))
        path = pjoin(subproc_dir, name)
        rel_dir = pjoin('SubProcesses', name)
        expression, multiparticles = process_expression(procdef, model)
        orders = coupling_orders(procdef, model)
        # every vertex carries one coupling: the rest are powers of g_s
        qcd_power = (len(procdef.get('legs')) - 2 - sum(orders.values())
                     if len(orders) == 1 else -1)
        logger.info('Generating the AmpliCol matrix elements of %s in %s' %
                    (procdef.nice_string(prefix=False), path))
        os.makedirs(path, exist_ok=True)
        for accuracy in ('lc', 'full'):
            generate_artifact(pyamplicol, pjoin(path, accuracy), accuracy,
                              expression, multiparticles, orders, source)
        provider_ids = umami_reweight.write_reweight_provider(
            pjoin(path, 'lc'), pjoin(path, 'full'), pjoin(path, 'reweight'))
        documents = umami_reweight.load_umami_metadata(pjoin(path, 'lc'))
        copies = []
        for provider_id, metadata in zip(provider_ids, documents):
            library = metadata['provider']['library']
            me_path = pjoin('lib', 'libamplicol_%s_%s.so' % (name, provider_id))
            ratio_path = pjoin('lib', 'libamplicol_%s_%s_ratio.so' %
                               (name, provider_id))
            copies.append('\tcp $(BUILD)/lc/%s ../../%s' % (library, me_path))
            copies.append('\tcp $(BUILD)/reweight/libreweight_%s.so ../../%s' %
                          (provider_id, ratio_path))
            entries.extend(subprocess_entries(metadata, model, rel_dir, me_path,
                                              ratio_path, qcd_power))
        with open(pjoin(path, 'makefile'), 'w') as stream:
            stream.write(MAKEFILE % {'rusticol_config': rusticol_config,
                                     'copies': '\n'.join(copies)})
        with open(pjoin(path, 'process.json'), 'w') as stream:
            json.dump({'process': procdef.nice_string(prefix=False),
                       'expression': expression,
                       'multiparticles': multiparticles,
                       'max_coupling_orders': orders}, stream, indent=2)
    return entries
