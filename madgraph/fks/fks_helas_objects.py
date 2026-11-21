################################################################################
#
# Copyright (c) 2009 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Definitions of the Helas objects needed for the implementation of MadFKS 
from born"""


from __future__ import absolute_import
import madgraph
import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_common as fks_common
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.various.misc as misc
import copy
import logging
import array
import shutil
import tempfile
import pickle 
import itertools
import os
import sys
from madgraph import MG5DIR
pjoin = os.path.join
logger = logging.getLogger('madgraph.fks_helas_objects')
if madgraph.ordering:
    set = misc.OrderedSet


def make_unique_couplings(couplings):
    """Deduplicate coupling references without hashing mutable flavor tables.

    ``misc.make_unique`` uses ``dict.fromkeys``.  That is appropriate for
    coupling-name strings but not for ``FLV_Coupling`` (a mutable
    ``PhysicsObject``).  Equality here intentionally includes the generated
    coupling name as well as its table, because every distinct name referenced
    by HELAS calls must remain available to model export.
    """
    unique = []
    seen = set()
    for coupling in couplings:
        if isinstance(coupling, MG.FLV_Coupling):
            key = ('flavor', coupling.get('name'),
                   coupling.get_canonical_key())
        else:
            key = ('ordinary', coupling)
        if key in seen:
            continue
        seen.add(key)
        unique.append(coupling)
    return unique


def find_real_amplitude(real_amp_list, amplitude):
    """The position in real_amp_list of the amplitude itself, which
    FKSProcess.generate_real_amplitudes gave to an FKSRealProcess, rather than
    of the first equal one: Amplitude equality compares the processes by
    their legs only (Process.__eq__), not their orders, and would take the
    real of another process definition."""
    for index, real_amp in enumerate(real_amp_list):
        if real_amp is amplitude:
            return index
    raise fks_common.FKSProcessError(
            '%s: this real amplitude is not in the list of real amplitudes' % \
            amplitude['process'].nice_string(print_weighted=False))


def get_initial_states(me):
    """The initial states of the born and real processes of the
    FKSHelasProcess me, for the initial_states_map.dat"""
    initial_states = []
    for processes in [me.born_me.get('processes')] + \
            [fksreal.matrix_element.get('processes') for fksreal in me.real_processes]:
        # Pick out all initial state particles for the two beams (None for
        # the second one of a decay)
        initial_states.append(sorted(list(set((p.get_initial_pdg(1),p.get_initial_pdg(2)) for \
                                              p in processes))))
    return initial_states


def merge_initial_states(initial_states_per_me):
    """The initial states of all the matrix elements, in order, without the
    doubles"""
    initial_states = []
    for me_initial_states in initial_states_per_me:
        for state in me_initial_states:
            if state not in initial_states:
                initial_states.append(state)
    return initial_states


def definition_indices(born_procs):
    """The index of the process definition of each of the FKSProcesses
    born_procs (in the order of their first born): the borns of a process
    definition share their born_pdg_list (see FKSMultiProcess.generate_all)"""
    indices = {}
    return [indices.setdefault(id(born.born_pdg_list), len(indices)) \
            for born in born_procs]


#functions to be used in the ncores_for_proc_gen mode
# (low_mem_multicore_nlo_generation). The matrix elements go from one step to
# the next through pickle files in a temporary directory, so that the parent
# process never holds all of them. These steps must give the same matrix
# elements, in the same order, as the serial mode (FKSMultiProcess.generate_all,
# then FKSHelasMultiProcess.generate_matrix_elements_fks).

def load_tmpfile(path):
    """Unpickle a file written by dump_to_tmpfile"""
    with open(path, 'rb') as infile:
        return pickle.load(infile)


def dump_to_tmpfile(data, tmpdir):
    """Pickle data to a new file in tmpdir, returns its name"""
    output = tempfile.NamedTemporaryFile(delete = False, dir = tmpdir)
    pickle.dump(data, output, protocol=2)
    output.close()
    return output.name


def load_matrix_element(path, uid):
    """Load a matrix element written by the low_mem_multicore_nlo_generation
    mode, with the uid it has in the serial mode: its position (from 1) in
    the list of matrix elements (Process.shell_string uses it for long names)"""
    me = load_tmpfile(path)
    me.get('processes')[0].set('uid', uid)
    return me


def async_generate_real(args):
    real_amp, tmpdir = args
    #amplitude generation
    amplitude = real_amp.generate_real_amplitude()
    # check that the amplitude has diagrams, otherwise quit here
    # and return an empty list
    if not amplitude['diagrams']:
        msg = "Discarding amplitude with no diagrams%s" % \
              (amplitude['process'].nice_string(print_weighted=False).replace('Process ', ''))
        logger.debug(msg)
        return []

    logger.info('Generating real %s' % \
            real_amp.process.nice_string(print_weighted=False).replace('Process', 'process'))
    # its matrix element, as the serial mode makes it
    helasreal = FKSHelasMultiProcess.generate_matrix_elements(
                    diagram_generation.AmplitudeList([amplitude]),
                    combine_matrix_elements=False,
                    matrix_element_opts={'enumerate_all_flavors': True})[0]

    return {'file': dump_to_tmpfile([amplitude, helasreal], tmpdir),
            'ndiagrams': len(amplitude['diagrams'])}


def async_generate_born(args):
    # realfiles: the files of the real amplitudes of the process definition
    # of the born, by external_process_identity
    born, realfiles, loop_optimized, tmpdir = args

    logger.info('Generating born %s' % \
            born.born_amp['process'].nice_string(print_weighted=False).replace('Process', 'process'))
    #load information on reals from temp files
    helasreal_list = []
    amp_to_remove = []
    for amp in born.real_amps:
        # If the complete state identity is absent, it has been removed
        # because there are no diagrams
        try:
            infilename = realfiles[fks_common.external_process_identity(amp.process)]
        except KeyError:
            logger.debug('Removing amplitude: %s' % amp.process.nice_string())
            amp_to_remove.append(amp)
            continue
        amp.amplitude, helasreal = load_tmpfile(infilename)
        helasreal_list.append(helasreal)

    for amp in amp_to_remove:
        born.real_amps.remove(amp)
        
    born.link_born_reals()
        
    # as FKSMultiProcess.generate_all does in the serial mode
    for amp in born.real_amps:
        amp.find_fks_j_from_i(born.born_pdg_list)
    born.generate_virtual_amplitude()
    has_loops = born.virt_amp is not None

    helasfull = FKSHelasProcess(born, helasreal_list,
                                loop_optimized = loop_optimized,
                                decay_ids=[],
                                gen_color=False)

    metag = helas_objects.IdentifyMETag.create_tag(helasfull.born_me.get('base_amplitude'))
    
    return {'file': dump_to_tmpfile(helasfull, tmpdir),
            'tag': metag,
            'has_loops': has_loops,
            'n_loop_diagrams': len(born.virt_amp.get('loop_diagrams')) if has_loops else 0,
            'ndiagrams': len(born.born_amp.get('diagrams'))}


def async_finalize_matrix_elements(args):
    # me_record: see FKSHelasMultiProcess.__init__
    mefile, duplist, tmpdir, me_record = args
    me = load_tmpfile(mefile)

    # Always create an empty color basis, and the
    # list of raw colorize objects (before
    # simplification) associated with amplitude
    col_basis = color_amp.ColorBasis()
    new_amp = me.born_me.get_base_amplitude()
    me.born_me.set('base_amplitude', new_amp)
    colorize_obj = col_basis.create_color_dict_list(new_amp)

    col_basis.build()
    col_matrix = color_amp.ColorMatrix(col_basis)

    me.born_me.set('color_basis',col_basis)
    me.born_me.set('color_matrix',col_matrix)

    cannot_combine = []
    
    for othermefile in duplist:
        otherme = load_tmpfile(othermefile)
        # before entering this function, only the born
        # processes were compared. Now compare the
        # full ME (born/real/virtual)
        if otherme == me:
            me.add_process(otherme)
        else:
            cannot_combine.append(othermefile)
        
    me.set_color_links()    
        
    # the born processes, without the copy of the model which each matrix
    # element unpickled here brings (the parent has the model)
    processes = []
    for process in me.born_me.get('processes'):
        process = copy.copy(process)
        process['model'] = None
        processes.append(process)
    
    #data to be returned to parent process (filename plus small objects only)
    return {'file': dump_to_tmpfile(me, tmpdir),
            'cannot_combine': cannot_combine,
            'initial_states': get_initial_states(me),
            'used_lorentz': me.get_used_lorentz(),
            'used_couplings': me.get_used_couplings(),
            'processes': processes,
            'record': me_record(me) if me_record else None}


class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
    """class to generate the helas calls for a FKSMultiProcess"""

    def get_sorted_keys(self):
        """Return particle property names as a nicely sorted list."""
        keys = super(FKSHelasMultiProcess, self).get_sorted_keys()
        keys += ['real_matrix_elements', 'has_isr', 'has_fsr', 'ewsudakov', 
                 'used_lorentz', 'used_couplings', 'processes']
        return keys

    def filter(self, name, value):
        """Filter for valid leg property values."""

        if name == 'real_matrix_elements':
            if not isinstance(value, helas_objects.HelasMultiProcess):
                raise self.PhysicsObjectError("%s is not a valid list for real_matrix_element " % str(value))                             
    
    def __init__(self, fksmulti, loop_optimized = False, gen_color =True, decay_ids =[],
                 me_record = None):
        """Initialization from a FKSMultiProcess. me_record, a function of a
        matrix element (e.g. the born_support_record of the exporter), is the
        low_mem_multicore_nlo_generation mode only: the processes which
        finalize the matrix elements compute it, in self['me_records']"""

        #swhich the other loggers off
        loggers_off = [logging.getLogger('madgraph.diagram_generation'),
                       logging.getLogger('madgraph.helas_objects')]
        old_levels = [logg.level for logg in loggers_off]
        for logg in loggers_off:
            logg.setLevel(logging.WARNING)
        try:
            self.generate_all(fksmulti, loop_optimized, gen_color, decay_ids,
                              me_record)
        finally:
            for i, logg in enumerate(loggers_off):
                logg.setLevel(old_levels[i])


    def generate_all(self, fksmulti, loop_optimized, gen_color, decay_ids,
                     me_record):
        """The matrix elements of fksmulti, see __init__ which is the only
        caller"""

        self.loop_optimized = loop_optimized

        self['used_lorentz'] = []
        self['used_couplings'] = []
        self['processes'] = []

        if not fksmulti['ncores_for_proc_gen']:
            # generate the real ME's if they are needed.
            # note that it may not be always the case, e.g. it the NLO_mode is LOonly
            if fksmulti['real_amplitudes']:
                logger.info('Generating real emission matrix-elements...')
                self['real_matrix_elements'] = self.generate_matrix_elements(
                        copy.copy(fksmulti['real_amplitudes']), combine_matrix_elements=False,
                        matrix_element_opts={'enumerate_all_flavors': True})
            else:
                self['real_matrix_elements'] = helas_objects.HelasMatrixElementList()

            self['matrix_elements'] = self.generate_matrix_elements_fks(
                                    fksmulti, 
                                    gen_color, decay_ids)
            # a unique id for each process (its position, as in the
            # low_mem_multicore_nlo_generation mode) and the possible PDF
            # combinations
            for uid, me in enumerate(self['matrix_elements'], 1):
                me.get('processes')[0].set('uid', uid)
            self['initial_states'] = merge_initial_states(
                    get_initial_states(me) for me in self['matrix_elements'])
            self['has_loops'] = len(self.get_virt_matrix_elements()) > 0 

        else: 
            # low_mem_multicore_nlo_generation
            self.generate_matrix_elements_multicore(fksmulti, loop_optimized,
                                                    me_record)
                        
        self['has_isr'] = fksmulti['has_isr']
        self['has_fsr'] = fksmulti['has_fsr']
        self['ewsudakov'] = fksmulti['ewsudakov']

        logger.info('... Done')


    def generate_matrix_elements_multicore(self, fksmulti, loop_optimized,
                                           me_record=None):
        """The low_mem_multicore_nlo_generation mode: generate the matrix
        elements of the serial mode, in the same order, with
        ncores_for_proc_gen processes at a time, the parent process keeping
        only pickle files of them, in the temporary directory self['tmpdir']
        (see remove_tmpdir). The real matrix elements of each process
        definition are generated once each, then the FKSHelasProcess (with its
        virtual) of each born, and the identical ones are combined, as
        generate_matrix_elements_fks does."""

        born_procs = fksmulti.get('born_processes')
        ncores = fksmulti['ncores_for_proc_gen']
        self['has_loops'] = False
        self['matrix_elements'] = []
        self['initial_states'] = []
        self['me_processes'] = []
        self['me_records'] = []
        if not born_procs:
            return

        # the real amplitudes, each generated once for each process definition
        # (as the serial mode generates those of a definition with it: a real
        # amplitude depends on the definition, e.g. through its NLO mode or its
        # diagram filter)
        definitions = definition_indices(born_procs)
        reals = {}
        for born, definition in zip(born_procs, definitions):
            for amp in born.real_amps:
                reals.setdefault((definition,
                        fks_common.external_process_identity(amp.process)), amp)

        self['tmpdir'] = tempfile.mkdtemp(prefix='nlo_low_mem_')
        tmpdir = self['tmpdir']
        try:
            logger.info('Generating real matrix elements...')
            modelpath = born_procs[0].born_amp['process']['model'].get('modelpath')
            with misc.TMP_variable(sys, 'path', sys.path + [pjoin(MG5DIR, 'models'), modelpath]):
                realmapout = fks_common.low_mem_map(async_generate_real,
                        [(amp, tmpdir) for amp in reals.values()], ncores)
            # the files of the real amplitudes of each definition (an
            # amplitude without diagrams gives an empty output)
            realfiles = [{} for definition in range(max(definitions) + 1)]
            for (definition, identity), rout in zip(reals, realmapout):
                if rout:
                    realfiles[definition][identity] = rout['file']
            realmapout = [rout for rout in realmapout if rout]

            logger.info('Generating born and virtual matrix elements...')
            bornmapout = fks_common.low_mem_map(async_generate_born,
                    [(born, realfiles[definition], loop_optimized, tmpdir) \
                     for born, definition in zip(born_procs, definitions)], ncores)
            for files in realfiles:
                for realfile in files.values():
                    os.remove(realfile)

            # (as FKSMultiProcess.generate_all in the serial mode, which also
            # checks that each process definition has some correction: in this
            # mode, it checks it with its own generation, see has_corrections)
            logger.info(('Generated %d subprocesses with %d real emission diagrams, ' + \
                         '%d born diagrams and %d virtual diagrams') % \
                        (len(born_procs), sum(rout['ndiagrams'] for rout in realmapout),
                         sum(bout['ndiagrams'] for bout in bornmapout),
                         sum(bout['n_loop_diagrams'] for bout in bornmapout)))

            memapout = self.combine_matrix_elements_multicore(bornmapout,
                                                    ncores, tmpdir, me_record)
        except BaseException:
            self.remove_tmpdir()
            raise

        #set final list of matrix elements (paths to temp files)
        self['matrix_elements'] = [meout['file'] for meout in memapout]
        self['me_records'] = [meout['record'] for meout in memapout]
        self['has_loops'] = any(bout['has_loops'] for bout in bornmapout)

        #cache information needed for output which will not be available from
        #the matrix elements later, as the serial mode computes it
        self['initial_states'] = merge_initial_states(
                meout['initial_states'] for meout in memapout)

        # the born processes of each matrix element, with the model of the
        # process definitions and the uid of the serial mode (see
        # load_matrix_element)
        model = born_procs[0].born_amp['process']['model']
        for ime, meout in enumerate(memapout):
            for proc in meout['processes']:
                proc['model'] = model
            meout['processes'][0].set('uid', ime + 1)
            self['me_processes'].append(meout['processes'])
        self['processes'] = [proc for procs in self['me_processes'] for proc in procs]

        helas_list = []
        coupling_list = []
        for meout in memapout:
            helas_list.extend(meout['used_lorentz'])
            coupling_list.extend([c for l in meout['used_couplings'] for c in l])
        self['used_lorentz'] = misc.make_unique(helas_list)
        self['used_couplings'] = make_unique_couplings(coupling_list)

        # Only the real amplitudes with diagrams survived the workers.
        # Do not report radiation from an empty/restricted amplitude.
        for born, definition in zip(born_procs, definitions):
            born.update_radiation_flags([
                real for real in born.real_amps
                if fks_common.external_process_identity(real.process)
                in realfiles[definition]])
        fksmulti['has_isr'] = any(born.isr for born in born_procs)
        fksmulti['has_fsr'] = any(born.fsr for born in born_procs)


    def remove_tmpdir(self):
        """The low_mem_multicore_nlo_generation mode: remove the temporary
        files of the matrix elements, which an output reads (the next one
        generates them again)"""
        tmpdir = self.pop('tmpdir', None)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        self['matrix_elements'] = []
        self['me_records'] = []


    @staticmethod
    def combine_matrix_elements_multicore(bornmapout, ncores, tmpdir, me_record):
        """Combine the identical FKSHelasProcesses (born, reals and virtual, see
        FKSHelasProcess.__eq__) of the borns, written by async_generate_born,
        as generate_matrix_elements_fks does in the serial mode: a born joins
        the first matrix element it is identical to, and the matrix elements
        are in the order of their first born. In each round, the first
        remaining born with a given born tag is compared with the other ones
        with this tag; those which cannot be combined with it go to the next
        round. Returns the outputs of async_finalize_matrix_elements, in the
        order of the matrix elements."""

        remaining = list(range(len(bornmapout)))
        finalized = []
        while remaining:
            logger.info('Collecting infos and finalizing matrix elements, %d left...' \
                        % len(remaining))
            firsts = []
            others = []
            for i in remaining:
                for first, other in zip(firsts, others):
                    if bornmapout[i]['tag'] == bornmapout[first]['tag']:
                        other.append(i)
                        break
                else:
                    firsts.append(i)
                    others.append([])

            memapout = fks_common.low_mem_map(async_finalize_matrix_elements,
                    [(bornmapout[first]['file'],
                      [bornmapout[i]['file'] for i in other], tmpdir, me_record) \
                     for first, other in zip(firsts, others)], ncores)

            # the borns which have the born tag of a matrix element but could
            # not be combined with it
            not_combined = set()
            for first, meout in zip(firsts, memapout):
                finalized.append((first, meout))
                not_combined.update(meout['cannot_combine'])

            #remove born+virtual temp files
            for i in remaining:
                if bornmapout[i]['file'] not in not_combined:
                    os.remove(bornmapout[i]['file'])
            remaining = [i for i in remaining if bornmapout[i]['file'] in not_combined]

        finalized.sort(key=lambda first_meout: first_meout[0])
        return [meout for first, meout in finalized]
            
        
    def get_used_lorentz(self):
        """Return a list of (lorentz_name, conjugate, outgoing) with
        all lorentz structures used by this HelasMultiProcess."""

        if not self['used_lorentz']:
            helas_list = []
            for me in self.get('matrix_elements'):
                helas_list.extend(me.get_used_lorentz())
            self['used_lorentz'] = misc.make_unique(helas_list)

        return self['used_lorentz']


    def get_used_couplings(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        if not self['used_couplings']:
            coupling_list = []
            for me in self.get('matrix_elements'):
                coupling_list.extend([c for l in me.get_used_couplings() for c in l])
            self['used_couplings'] = make_unique_couplings(coupling_list)

        return self['used_couplings']


    def get_processes(self):
        """Return a list with all couplings used by this
        HelasMatrixElement."""

        if not self['processes']:
            process_list = []
            for me in self.get('matrix_elements'):
                process_list.extend(me.born_me.get('processes'))
            self['processes'] = process_list

        return self['processes']
    

    def get_matrix_elements(self):
        """Extract the list of matrix elements"""
        return self.get('matrix_elements')        


    def get_virt_matrix_elements(self):
        """Extract the list of virtuals matrix elements"""
        return [me.virt_matrix_element for me in self.get('matrix_elements') \
                if me.virt_matrix_element]        
        

    def generate_matrix_elements_fks(self, fksmulti, gen_color = True,
                                 decay_ids = []):
        """Generate the HelasMatrixElements for the amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__. Returns a
        HelasMatrixElementList and an amplitude map (used by the
        SubprocessGroup functionality). decay_ids is a list of decayed
        particle ids, since those should not be combined even if
        matrix element is identical."""

        fksprocs = fksmulti['born_processes']
        assert isinstance(fksprocs, fks_base.FKSProcessList), \
                  "%s is not valid FKSProcessList" % \
                   repr(fksprocs)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        list_color_links = []
        list_color_basis = []
        list_color_matrices = []
        real_me_list = []
        me_id_list = []

        matrix_elements = FKSHelasProcessList()

        for i, proc in enumerate(fksprocs):
            logger.info("Generating Helas calls for FKS %s (%d / %d)" % \
              (proc.get_born_nice_string().\
                                    replace('Process', 'process'),
                        i + 1, len(fksprocs)))
            matrix_element_list = [\
                    FKSHelasProcess(proc, self['real_matrix_elements'],
                    [amp for amp in fksmulti['real_amplitudes'] if amp['diagrams']],
                    loop_optimized = self.loop_optimized,
                    decay_ids=decay_ids,
                    gen_color=False)]

            for matrix_element in matrix_element_list:
                assert isinstance(matrix_element, FKSHelasProcess), \
                          "Not a FKSHelasProcess: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other = \
                          matrix_elements[matrix_elements.index(matrix_element)]
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.
                    if matrix_element.born_me.get('processes') and \
                       matrix_element.born_me.get('diagrams'):
                        matrix_elements.append(matrix_element)

                        if not gen_color:
                            continue

                        # Always create an empty color basis, and the
                        # list of raw colorize objects (before
                        # simplification) associated with amplitude
                        col_basis = color_amp.ColorBasis()
                        new_amp = matrix_element.born_me.get_base_amplitude()
                        matrix_element.born_me.set('base_amplitude', new_amp)
                        colorize_obj = col_basis.create_color_dict_list(new_amp)

                        try:
                            # If the color configuration of the ME has
                            # already been considered before, recycle
                            # the information
                            col_index = list_colorize.index(colorize_obj)
                            logger.info(\
                              "Reusing existing color information for %s" % \
                              matrix_element.born_me.get('processes')\
                              [0].nice_string(print_weighted=False).\
                                                 replace('Process', 'process'))
                        except ValueError:
                            # If not, create color basis and color
                            # matrix accordingly
                            list_colorize.append(colorize_obj)
                            col_basis.build()
                            list_color_basis.append(col_basis)
                            col_matrix = color_amp.ColorMatrix(col_basis)
                            list_color_matrices.append(col_matrix)
                            col_index = -1

                            logger.info(\
                              "Processing color information for %s" % \
                              matrix_element.born_me.get('processes')[0].\
                              nice_string(print_weighted=False).\
                                             replace('Process', 'process'))
                        matrix_element.born_me.set('color_basis', list_color_basis[col_index])
                        matrix_element.born_me.set('color_matrix', list_color_matrices[col_index])                    
                else:
                    # this is in order not to handle valueErrors coming from other plaeces,
                    # e.g. from the add_process function
                    other.add_process(matrix_element)

        for me in matrix_elements:
            me.set_color_links()
        return matrix_elements    


class FKSHelasProcessList(MG.PhysicsObjectList):
    """class to handle lists of FKSHelasProcesses"""
    
    def is_valid_element(self, obj):
        """Test if object obj is a valid FKSProcess for the list."""
        return isinstance(obj, FKSHelasProcess)
    
    
class FKSHelasProcess(object):
    """class to generate the Helas calls for a FKSProcess. Contains:
    -- born ME
    -- list of FKSHelasRealProcesses
    -- color links
    -- charges
    -- extra MEs used as counterterms
    """
    
    def __init__(self, fksproc=None, real_me_list =[], real_amp_list=[], 
            loop_optimized = False, **opts):#test written
        """ constructor, starts from a FKSProcess, 
        sets reals and color links. Real_me_list and real_amp_list are the lists of pre-genrated
        matrix elements in 1-1 correspondence with the amplitudes"""
        
        self._fks_flavor_map_cache = {}
        if fksproc != None:
            self.born_me = helas_objects.HelasMatrixElement(fksproc.born_amp, **opts)

            self.real_processes = []
            self.extra_cnt_me_list = []
            self.perturbation = fksproc.perturbation
            self.charges_born = fksproc.get_charges()
            # user numbering of the born legs, for me_frame (frame_info.inc)
            self.user_leg_order = fksproc.user_leg_order
            real_amps_new = []

            for extra_cnt in fksproc.extra_cnt_amp_list:
                self.extra_cnt_me_list.append(helas_objects.HelasMatrixElement(extra_cnt,gen_color=True))

            # combine for example u u~ > t t~ and c c~ > t t~
            if fksproc.ncores_for_proc_gen:
                # new NLO (multicore) generation mode 
                for real_me, proc in zip(real_me_list,fksproc.real_amps):
                    fksreal_me = FKSHelasRealProcess(proc, real_me, **opts)
                    try:
                        other = self.real_processes[self.real_processes.index(fksreal_me)]
                        other.matrix_element.get('processes').extend(\
                                fksreal_me.matrix_element.get('processes') )
                    except ValueError:
                        if fksreal_me.matrix_element.get('processes') and \
                                fksreal_me.matrix_element.get('diagrams'):
                            self.real_processes.append(fksreal_me)
                            real_amps_new.append(proc)
            else:
                #old mode
                for proc in fksproc.real_amps:
                    if proc.amplitude['diagrams']:
                        fksreal_me = FKSHelasRealProcess(proc, real_me_list, real_amp_list, **opts)
                        try:
                            other = self.real_processes[self.real_processes.index(fksreal_me)]
                            other.matrix_element.get('processes').extend(\
                                    fksreal_me.matrix_element.get('processes') )
                        except ValueError:
                            if fksreal_me.matrix_element.get('processes') and \
                                    fksreal_me.matrix_element.get('diagrams'):
                                self.real_processes.append(fksreal_me)
                                real_amps_new.append(proc)

            fksproc.real_amps = real_amps_new
            if fksproc.virt_amp:
                self.virt_matrix_element = \
                  loop_helas_objects.LoopHelasMatrixElement(fksproc.virt_amp, 
                          optimized_output = loop_optimized)
            else: 
                self.virt_matrix_element = None

            self.sudakov_matrix_elements = []
            self.ewsudakov = fksproc.ewsudakov
            for amp in fksproc.sudakov_amps:
                sudakov_dict = {}
                for key in amp.keys():
                    if key == 'amplitude': 
                        continue
                    sudakov_dict[key] = amp[key]
                sudakov_dict['matrix_element'] = helas_objects.HelasMatrixElement(amp['amplitude'], gen_color=True)

                self.sudakov_matrix_elements.append(sudakov_dict)

                ##amp.pop('amplitude')

                ##col_basis = color_amp.ColorBasis()
                ##new_amp = amp['matrix_element'].get_base_amplitude()
                ##amp['matrix_element'].set('base_amplitude', new_amp)
                ##colorize_obj = col_basis.create_color_dict_list(new_amp)

                ##col_basis.build()
                ##col_matrix = color_amp.ColorMatrix(col_basis)
                ##amp['matrix_element'].set('color_basis', list_color_basis[col_index])
                ##amp['matrix_element'].set('color_matrix', list_color_matrices[col_index])                    

            self.color_links = []



    def set_color_links(self, force=False):
        """this function computes and returns the color links, it should be called
        after the initialization and the setting of the color basis"""
        color_basis = self.born_me.get('color_basis')
        if force or not self.color_links:
            legs = self.born_me.get('base_amplitude').get('process').get('legs')
            model = self.born_me.get('base_amplitude').get('process').get('model')
            color_links_info = fks_common.find_color_links(fks_common.to_fks_legs(legs, model),
                        symm = True, pert = self.perturbation)
            self.color_links = fks_common.insert_color_links(color_basis,
                                color_basis.create_color_dict_list(
                                    self.born_me.get('base_amplitude')),
                                color_links_info)

    def get_fks_info_list(self, resolve_virtual=True):
        """Returns the list of the fks infos for all processes in the format
        {n_me, pdgs, fks_info}, where n_me is the number of real_matrix_element the configuration
        belongs to.

        For a grouped model, one exported FKS configuration is one physical
        flavor class.  Repeating a grouped topology here is intentional: all
        existing runtime metadata is already indexed by ``NFKSPROCESS``, so
        this keeps physical channels separate until matrix elements, FKS
        factors, normalization, and PDFs have been applied.
        """
        model = self.born_me.get('processes')[0].get('model')
        if model.get('merged_particles'):
            return [
                {'n_me': entry['n_me'],
                 'pdgs': entry['real_pdgs'],
                 'fks_info': entry['fks_info'],
                 'flavor_class': entry}
                for entry in self.get_fks_flavor_map(
                    resolve_virtual=resolve_virtual)]

        info_list = []
        for n, real in enumerate(self.real_processes):
            pdgs = [l['id'] for l in real.matrix_element.get_base_amplitude()['process']['legs']]
            for info in real.fks_infos:
                info_list.append({'n_me' : n + 1,'pdgs' : pdgs, 'fks_info' : info})
        return info_list

    def get_fks_flavor_map(self, resolve_virtual=True):
        """Return physical flavor classes for every FKS configuration.

        Each entry translates one signed physical real-emission PDG row to the
        independently resolved local rows of its real and underlying-Born
        matrix elements.  The FKS configuration index follows
        :meth:`get_fks_info_list` order and is one-based, as are all local
        Fortran flavor indices.

        A topology-level merged process can cover real rows that do not belong
        to every one of its FKS configurations.  Such rows are omitted when
        they cannot map to a valid underlying Born row.  A configuration with
        no physical rows is an error.
        """
        cache_key = bool(resolve_virtual)
        if cache_key in self._fks_flavor_map_cache:
            return self._fks_flavor_map_cache[cache_key]

        flavor_map = []
        config_index = 0
        initial_born_color_basis = self.born_me.get('color_basis')
        merged = self.born_me.get('processes')[0].get('model').get(
            'merged_particles')

        def physical_properties(pdgs, model):
            charges = []
            colors = []
            for pdg in pdgs:
                if abs(pdg) in merged:
                    raise fks_common.FKSProcessError(
                        'Merged pseudo-PDG %s leaked into a physical FKS row'
                        % pdg)
                particle = model.get_particle(pdg)
                if particle is None:
                    raise fks_common.FKSProcessError(
                        'Physical PDG %s is not present in the model' % pdg)
                charge = particle.get_charge()
                if isinstance(charge, tuple):
                    raise fks_common.FKSProcessError(
                        'Physical PDG %s has unresolved merged charge %s'
                        % (pdg, charge))
                charges.append(charge)
                colors.append(particle.get_color())
            return charges, colors

        for n_me, real in enumerate(self.real_processes, start=1):
            real_me = real.matrix_element
            model = real_me.get('processes')[0].get('model')
            real_flavors, real_pdg_rows = \
                real_me.get_external_flavors(return_pdgs=True)
            for info in real.fks_infos:
                config_entries = []
                for real_flavor, real_pdgs in zip(real_flavors,
                                                  real_pdg_rows):
                    try:
                        born_pdgs = fks_common.map_real_to_born_pdgs(
                            real_pdgs, info, model)
                    except fks_common.FKSFlavorNotInTopology:
                        # This physical real row is outside the union member
                        # represented by this grouped FKS topology.
                        continue

                    real_index = real_me.get_external_flavor_index(
                        real_pdgs, from_pdgs=True)
                    born_index = self.born_me.get_external_flavor_index(
                        born_pdgs, from_pdgs=True)
                    if real_index == 0 or born_index == 0:
                        continue

                    real_charges, real_colors = physical_properties(
                        real_pdgs, model)
                    born_charges, born_colors = physical_properties(
                        born_pdgs, model)

                    virtual_index = 0
                    if resolve_virtual and self.virt_matrix_element is not None:
                        virtual_index = self.virt_matrix_element.\
                            get_external_flavor_index(
                                born_pdgs, from_pdgs=True)
                        if virtual_index == 0:
                            raise fks_common.FKSProcessError(
                                'No virtual flavor row for physical Born %s'
                                % born_pdgs)

                    extra_cnt_index = 0
                    extra_cnt_pdgs = None
                    extra_cnt_charges = None
                    extra_cnt_colors = None
                    if info['extra_cnt_index'] != -1:
                        try:
                            extra_cnt_pdgs = \
                                fks_common.map_real_to_born_pdgs(
                                    real_pdgs, info, model,
                                    underlying_index=1)
                        except fks_common.FKSProcessError as error:
                            raise fks_common.FKSProcessError(
                                'Cannot map extra counterterm for physical '
                                'real row %s: %s' % (real_pdgs, error))
                        extra_me = self.extra_cnt_me_list[
                            info['extra_cnt_index']]
                        extra_cnt_index = \
                            extra_me.get_external_flavor_index(
                                extra_cnt_pdgs, from_pdgs=True)
                        if extra_cnt_index == 0:
                            raise fks_common.FKSProcessError(
                                'No extra-counterterm flavor row for physical '
                                'PDGs %s' % extra_cnt_pdgs)
                        extra_cnt_charges, extra_cnt_colors = \
                            physical_properties(extra_cnt_pdgs, model)

                    entry = {
                        'n_me': n_me,
                        'fks_info': info,
                        'real_pdgs': list(real_pdgs),
                        'born_pdgs': born_pdgs,
                        'real_flavor': real_flavor,
                        'born_flavor': self.born_me.get_external_flavor(
                            born_index),
                        'real_flavor_index': real_index,
                        'born_flavor_index': born_index,
                        'extra_cnt_pdgs': extra_cnt_pdgs,
                        'extra_cnt_charges': extra_cnt_charges,
                        'extra_cnt_colors': extra_cnt_colors,
                        'extra_cnt_flavor_index': extra_cnt_index,
                        'virtual_flavor_index': virtual_index,
                        'real_charges': real_charges,
                        'born_charges': born_charges,
                        'real_colors': real_colors,
                        'born_colors': born_colors,
                    }
                    config_entries.append(entry)

                if not config_entries:
                    # A grouped topology can expose a structural splitting for
                    # which none of the matrix element's physical rows has the
                    # required i/j flavor relation. It is not an executable FKS
                    # class and therefore must not enter generated tables.
                    continue
                config_index += 1
                for entry in config_entries:
                    entry['fks_config_index'] = config_index
                flavor_map.extend(config_entries)

        # Resolving a restricted physical Born row can trim diagrams and rebuild
        # its color basis. Refresh linked Born bases before exporters consume
        # them; stale link bases still refer to the pre-trim diagram positions.
        if self.born_me.get('color_basis') is not initial_born_color_basis:
            self.set_color_links(force=True)
        self._fks_flavor_map_cache[cache_key] = flavor_map
        return flavor_map

    def clear_fks_flavor_map_cache(self):
        """Invalidate cached physical-row mappings after metadata mutation.

        Normal generation finishes constructing Born, real, virtual, and
        extra-counterterm metadata before requesting this map.  The explicit
        hook is for low-level callers and tests that intentionally alter those
        public objects afterwards.
        """
        self._fks_flavor_map_cache.clear()

    def get_fks_flavor_signature(self):
        """Return an immutable identity for grouped physical FKS mappings.

        Local row numbers are included deliberately: two objects cannot share
        generated wrappers when the same physical channel selects different
        local Born, real, counterterm, or virtual rows.
        """
        signature = []
        for entry in self.get_fks_flavor_map():
            info = entry['fks_info']
            signature.append((
                tuple(entry['real_pdgs']),
                tuple(entry['born_pdgs']),
                tuple(entry['extra_cnt_pdgs'] or ()),
                entry['real_flavor_index'],
                entry['born_flavor_index'],
                entry['extra_cnt_flavor_index'],
                entry['virtual_flavor_index'],
                info['i'], info['j'], info['ij'], info['ij_id'],
                tuple(info['splitting_type']),
                info['need_color_links'], info['need_charge_links'],
                tuple(entry['real_charges']),
                tuple(entry['born_charges']),
                tuple(entry['real_colors']),
                tuple(entry['born_colors'])))
        return tuple(sorted(signature))
        

    def get_lh_pdg_string(self):
        """Returns the pdgs of the legs in the form "i1 i2 -> f1 f2 ...", which may
        be useful (eg. to be written in a B-LH order file)"""

        initial = ''
        final = ''
        for leg in self.born_me.get('processes')[0].get('legs'):
            if leg.get('state'):
                final += '%d ' % leg.get('id')
            else:
                initial += '%d ' % leg.get('id')
        return initial + '-> ' + final


    def get(self, key):
        """the get function references to the born
        matrix element
        """
        return self.born_me.get(key)

    
    def get_used_lorentz(self):
        """the get_used_lorentz function references to born, reals
        and virtual matrix elements"""
        lorentz_list = self.born_me.get_used_lorentz()
        for real in self.real_processes:
            lorentz_list.extend(real.matrix_element.get_used_lorentz())
        if self.virt_matrix_element:
            lorentz_list.extend(self.virt_matrix_element.get_used_lorentz())
        for sud_me in self.sudakov_matrix_elements:
            lorentz_list.extend(sud_me['matrix_element'].get_used_lorentz())

        return misc.make_unique(lorentz_list)
    
    def get_used_couplings(self):
        """the get_used_couplings function references to born, reals
        and virtual matrix elements"""
        coupl_list = self.born_me.get_used_couplings()
        for real in self.real_processes:
            coupl_list.extend([c for c in\
                        real.matrix_element.get_used_couplings()])
        if self.virt_matrix_element:
            coupl_list.extend(self.virt_matrix_element.get_used_couplings())
        for sud_me in self.sudakov_matrix_elements:
            coupl_list.extend(sud_me['matrix_element'].get_used_couplings())
        return coupl_list    

    def get_nexternal_ninitial(self):
        """the nexternal_ninitial function references to the real emissions if they have been
        generated, otherwise to the born"""
        if self.real_processes:
            (nexternal, ninitial) = self.real_processes[0].matrix_element.get_nexternal_ninitial()
        else:
            (nexternal, ninitial) = self.born_me.get_nexternal_ninitial()
            nexternal += 1
        return (nexternal, ninitial)
    
    def __eq__(self, other):
        """the equality between two FKSHelasProcesses is defined up to the
        color links"""
        #first compare the born
        selfamp = self.born_me.get('base_amplitude')
        otheramp = other.born_me.get('base_amplitude')
        selftag = helas_objects.IdentifyMETag.create_tag(selfamp)
        othertag = helas_objects.IdentifyMETag.create_tag(otheramp)

        # MZ: if EW sudakov are included, do not combine.
        # This is not 100% ideal, as it is quite inefficient, but it is the safest option
        # (whichever of the two has them: the serial mode compares an existing
        # matrix element with a new one, the low_mem_multicore_nlo_generation
        # one the other way round)
        if self.ewsudakov or other.ewsudakov:
            logger.warning('With --ewsudakov, matrix elements will not be combined')
            return False

        if selftag != othertag:
            return False

        # now the virtuals
        if self.virt_matrix_element and other.virt_matrix_element: 
            virttag = helas_objects.IdentifyMETag.\
                        create_tag(self.virt_matrix_element.get('base_amplitude'))
            othertag = helas_objects.IdentifyMETag.\
                        create_tag(other.virt_matrix_element.get('base_amplitude'))
            if virttag != othertag: 
                return False
        elif self.virt_matrix_element !=  other.virt_matrix_element: 
            return False

        # now the reals
        reals2 = copy.copy(other.real_processes)

        for real in  self.real_processes:
            try:
                reals2.remove(real)
            except (ValueError, fks_common.FKSProcessError):
                # A malformed physical mapping cannot be equivalent. Keep
                # explicit mapping/merge operations responsible for reporting
                # that error; equality must also work for copied candidates.
                return False  

        self_grouped = bool(self.born_me.get('processes')[0].get(
            'model').get('merged_particles'))
        other_grouped = bool(other.born_me.get('processes')[0].get(
            'model').get('merged_particles'))
        if self_grouped != other_grouped:
            return False
        if self_grouped and self.get_fks_flavor_signature() != \
                other.get_fks_flavor_signature():
            return False
                
        if not reals2:
            return True
        else: 
            return False


    def __ne__(self, other):
        """Inequality operator
        """
        return not self.__eq__(other)

    
    def add_process(self, other): #test written, ppwj
        """adds processes from born and reals of other to itself. Note that 
        corresponding real processes may not be in the same order. This is 
        taken care of by constructing the list of self_reals.
        """
        grouped = bool(self.born_me.get('processes')[0].get(
            'model').get('merged_particles'))
        if grouped and self.get_fks_flavor_signature() != \
                other.get_fks_flavor_signature():
            raise fks_common.FKSProcessError(
                'add_process: incompatible physical FKS flavor mappings')

        # first add the born process
        #need to store pdg lists rather than processes in order to keep mirror processes different
        this_pdgs = [[leg['id'] for leg in proc['legs']] \
                for proc in self.born_me['processes']]
        for oth_proc in other.born_me['processes']:
            oth_pdgs = [leg['id'] for leg in oth_proc['legs']]
            if oth_pdgs not in this_pdgs:
                self.born_me['processes'].append(oth_proc)
                this_pdgs.append(oth_pdgs)

        # then the virtuals (if generated)
        if self.virt_matrix_element and other.virt_matrix_element:
            self.virt_matrix_element.get('processes').extend(
                    other.virt_matrix_element.get('processes'))

        # finally the reals
        self_reals = [real.matrix_element for real in self.real_processes]
        for oth_real in other.real_processes:

            try:
                #there should be a 1to1 correspondence between real emission
                ####this_real = self.real_processes[self_reals.index(oth_real.matrix_element)]
                this_real = self.real_processes[self.real_processes.index(oth_real)]
            except ValueError:
                raise fks_common.FKSProcessError('add_process: error in combination of real MEs')
            #need to store pdg lists rather than processes in order to keep mirror processes different
            this_pdgs = [[leg['id'] for leg in proc['legs']] \
                    for proc in this_real.matrix_element['processes']]
            for oth_proc in oth_real.matrix_element['processes']:
                oth_pdgs = [leg['id'] for leg in oth_proc['legs']]
                if oth_pdgs not in this_pdgs:
                    this_real.matrix_element['processes'].append(oth_proc)
                    this_pdgs.append(oth_pdgs)
        self.clear_fks_flavor_map_cache()

            
    
class FKSHelasRealProcess(object): #test written
    """class to generate the Helas calls for a FKSRealProcess
    contains:
    -- colors
    -- charges
    -- particle_tags
    -- i/j/ij fks, ij refers to the born leglist
    -- ijglu
    -- need_color_links
    -- fks_j_from_i
    -- matrix element
    -- is_to_integrate
    -- leg permutation<<REMOVED"""
    
    def __init__(self, fksrealproc=None, real_me_list = [], real_amp_list =[], **opts):
        """constructor, starts from a fksrealproc and then calls the
        initialization for HelasMatrixElement.
        Sets i/j fks and the permutation.
        real_me_list and real_amp_list are the lists of pre-generated matrix elements in 1-1 
        correspondence with the amplitudes"""
        
        if fksrealproc != None:
            self.isfinite = False
            self.colors = fksrealproc.colors
            self.particle_tags = fksrealproc.particle_tags
            self.charges = fksrealproc.charges
            self.fks_infos = fksrealproc.fks_infos
            self.is_to_integrate = fksrealproc.is_to_integrate

            # real_me_list is a list in the old NLO generation mode;
            # in the new one it is a matrix element
            if type(real_me_list) == list and len(real_me_list) != len(real_amp_list):
                raise fks_common.FKSProcessError(
                        'not same number of amplitudes and matrix elements: %d, %d' % \
                                (len(real_amp_list), len(real_me_list)))
            if type(real_me_list) == list and real_me_list and real_amp_list:
                self.matrix_element = copy.deepcopy(real_me_list[find_real_amplitude(real_amp_list, fksrealproc.amplitude)])
                self.matrix_element['processes'] = copy.deepcopy(self.matrix_element['processes'])

            elif type(real_me_list) == helas_objects.HelasMatrixElement: 
                #new NLO generation mode
                assert fksrealproc.process in real_me_list['processes'], \
                       "Inconsistent input in FKSHelasRealProcess\nfksrealproc: %s\nME: %s" % \
                               (fksrealproc.process.nice_string(), 
                                ' - '.join([p.nice_string() for p in real_me_list['processes']]))
                self.matrix_element = real_me_list

            else:

                if real_me_list and real_amp_list:
                    self.matrix_element = copy.deepcopy(real_me_list[find_real_amplitude(real_amp_list, fksrealproc.amplitude)])
                    self.matrix_element['processes'] = copy.deepcopy(self.matrix_element['processes'])
                else:
                    logger.info('generating matrix element...')
                    self.matrix_element = helas_objects.HelasMatrixElement(
                                                      fksrealproc.amplitude,
                                                      **dict(opts, enumerate_all_flavors=True))
                    #generate the color for the real
                    self.matrix_element.get('color_basis').build(
                                        self.matrix_element.get('base_amplitude'))
                    self.matrix_element.set('color_matrix',
                                     color_amp.ColorMatrix(
                                        self.matrix_element.get('color_basis')))
            #self.fks_j_from_i = fksrealproc.find_fks_j_from_i()
            self.fks_j_from_i = fksrealproc.fks_j_from_i

    def get_nexternal_ninitial(self):
        """Refers to the matrix_element function"""
        return self.matrix_element.get_nexternal_ninitial()

    def get_fks_flavor_signature(self):
        """Return grouped real-to-underlying-Born validity metadata.

        Ungrouped equality intentionally keeps its historical behavior so that
        ordinary subprocesses with equivalent matrix elements can still be
        combined.  A merged matrix element must additionally agree on every
        physical real row represented by each FKS configuration.
        """
        model = self.matrix_element.get('processes')[0].get('model')
        if not model.get('merged_particles'):
            return None

        signature = []
        _, real_pdg_rows = self.matrix_element.get_external_flavors(
            return_pdgs=True)
        for info in self.fks_infos:
            config_entries = []
            for real_pdgs in real_pdg_rows:
                try:
                    born_pdgs = fks_common.map_real_to_born_pdgs(
                        real_pdgs, info, model)
                except fks_common.FKSFlavorNotInTopology:
                    continue
                extra_pdgs = ()
                if info['extra_cnt_index'] != -1:
                    extra_pdgs = tuple(fks_common.map_real_to_born_pdgs(
                        real_pdgs, info, model, underlying_index=1))
                config_entries.append((
                    tuple(real_pdgs), tuple(born_pdgs), extra_pdgs,
                    self.matrix_element.get_external_flavor_index(
                        real_pdgs, from_pdgs=True)))
            signature.append((
                info['i'], info['j'], info['ij'], info['ij_id'],
                tuple(info['splitting_type']),
                info['need_color_links'], info['need_charge_links'],
                tuple(sorted(config_entries))))
        return tuple(sorted(signature))
    
    def __eq__(self, other):
        """Equality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""

        for key in [k for k in self.__dict__.keys() if k not in ['fks_infos', 'charges']]:
            if self.__dict__[key] != other.__dict__[key]:
                return False

        # special care for the fks_infos, ignore the various PDG ids
        if (len(self.fks_infos) != len(other.fks_infos)):
            return False

        tocheck_info = [k for k in self.fks_infos[0].keys() if k not in ['ij_id', 'underlying_born']]
        for selfinfo, otherinfo in zip(self.fks_infos, other.fks_infos):
            if len(selfinfo['underlying_born']) != len(otherinfo['underlying_born']):
                return False
            for key in tocheck_info:
                if selfinfo[key] != otherinfo [key]:
                    return False

        if self.get_fks_flavor_signature() != \
                other.get_fks_flavor_signature():
            return False

        return True
    

    def __ne__(self, other):
        """Inequality operator:
        compare two FKSHelasRealProcesses by comparing their dictionaries"""
        return not self.__eq__(other)

