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
"""A user friendly command line interface to access all MadGraph7 features.
   Uses the cmd package for command interpretation and tab completion.
"""

from __future__ import absolute_import
import os
import logging
import pydoc
import sys
import time
import optparse
import subprocess
import shutil
import copy
import functools
import tempfile
import itertools
import os
import re


import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.extended_cmd as cmd
import madgraph.interface.madgraph_interface as mg_interface
import madgraph.interface.madevent_interface as me_interface
import madgraph.interface.extended_cmd as extended_cmd
import madgraph.interface.amcatnlo_run_interface as run_interface
import madgraph.interface.launch_ext_program as launch_ext
import madgraph.interface.loop_interface as Loop_interface
import madgraph.fks.fks_common as fks_common
import madgraph.fks.fks_base as fks_base
import madgraph.fks.fks_helas_objects as fks_helas
import madgraph.iolibs.export_fks as export_fks
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects

import madgraph.various.cluster as cluster
import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod
import madgraph.models.model_reader as model_reader
#useful shortcut
pjoin = os.path.join


logger = logging.getLogger('cmdprint') # -> stdout
logger_stderr = logging.getLogger('fatalerror') # ->stderr

# What the exporter accumulates in proc_characteristic while it writes the
# directories, which the serial mode writes one after the other: how
# aMCatNLOInterface.write_directories_multicore merges the values with which
# the workers end, in the order of the directories (each worker starts from
# the state of the parent). splitting_types is merged with those of each
# directory (export_fks.add_splitting_types, whose order depends on the
# history). Any other value must be the same in all the workers.
PROC_CHARACTERISTIC_MERGES = {
    'nexternal': max,
    'max_n_matched_jets': max,
    'single_color': lambda old, new: old and new,
    'limitations': lambda old, new: old + [v for v in new if v not in old]}
        
def merge_proc_characteristics(proc_charac, worker_characs, splitting_types):
    """Set in proc_charac the merge of worker_characs, the proc_characteristic
    of the workers, in the order of the directories, with splitting_types,
    the splitting types of each directory (see PROC_CHARACTERISTIC_MERGES)"""
    if not worker_characs:
        return
    for key in worker_characs[0]:
        if key == 'splitting_types':
            value = proc_charac['splitting_types']
            for split_types in splitting_types:
                value = export_fks.add_splitting_types(value, split_types)
        elif key in PROC_CHARACTERISTIC_MERGES:
            value = functools.reduce(PROC_CHARACTERISTIC_MERGES[key],
                                     [charac[key] for charac in worker_characs])
        else:
            values = [charac[key] for charac in worker_characs]
            if any(other != values[0] for other in values[1:]):
                raise MadGraph5Error(
                    'The directories have different %s: %s (see ' % (key,
                    ', '.join(misc.make_unique([str(v) for v in values]))) + \
                    'amcatnlo_interface.PROC_CHARACTERISTIC_MERGES)')
            value = values[0]
        proc_charac[key] = value
    
def aloha_cached_infos(exporter):
    """The analytic information (keys) of the loop wavefunctions computed so
    far, in that order, by the aloha model of the (loop) exporter"""
    aloha_model = getattr(exporter, 'aloha_model', None)
    return list(getattr(aloha_model, 'cached_interaction_infos', {}).keys())
    
def generate_directories_fks_async(args):
    """The low_mem_multicore_nlo_generation mode: in a process forked from
    the parent (see fks_common.low_mem_map), with its own copy of the
    exporter, write the directory of the matrix element of the file mefile
    (written by fks_helas_objects), the ime-th one, as the serial mode does,
    and return what the serial mode accumulates in the exporter meanwhile,
    for the parent to merge in the order of the matrix elements"""

    curr_exporter, mefile, curr_fortran_model, ime, nme, path, olpopts = args[:7]
    second_exporter = args[7] if len(args) > 7 else None
    second_helas = args[8] if len(args) > 8 else None

    me = fks_helas.load_matrix_element(mefile, ime + 1)

    aloha_infos = set(aloha_cached_infos(curr_exporter))
    dp_terminal = set(getattr(curr_exporter, '_dp_terminal_pending', {}))
    
    calls, splitorders = curr_exporter.generate_directories_fks(
        me, curr_fortran_model, ime, nme, path, olpopts,
        second_exporter=second_exporter, second_helas=second_helas)

    #only available after export has been done, so has to be returned from here
    max_loop_vertex_rank = None
    if me.virt_matrix_element:
        max_loop_vertex_rank = me.virt_matrix_element.get_max_loop_vertex_rank()  

    return {'second_dependencies': (second_exporter.nlo_tree_dependencies()
                                    if second_exporter is not None else None),
            'calls': calls,
            'splitorders': splitorders,
            'max_loop_vertex_rank': max_loop_vertex_rank,
            'borndir': export_fks.born_dir_name(me.born_me.get('processes')[0]),
            # (a list: a set does not keep its order when pickled)
            'splitting_types': export_fks.get_splitting_types(me),
            # see merge_proc_characteristics
            'proc_characteristic': dict(curr_exporter.proc_characteristic),
            # the aloha computations this worker added, to replay them
            'aloha_infos': [info for info in aloha_cached_infos(curr_exporter) \
                            if info not in aloha_infos],
            # the loop directories whose private DP terminal updaters the
            # parent's finalize must prove (see finalize_dp_terminal_updates)
            'dp_terminal_pending': {directory: entry for directory, entry in \
                getattr(curr_exporter, '_dp_terminal_pending', {}).items() \
                if directory not in dp_terminal}}

class CheckFKS(mg_interface.CheckValidForCmd):


    def check_display(self, args):
        """ Check the arguments of the display diagrams command in the context
        of the Loop interface."""
        
        mg_interface.MadGraphCmd.check_display(self,args)

        # The output options of the LO 'display diagrams' are not supported here:
        # the born, real and loop amplitudes are handled by separate calls, so
        # --merge would have each diagram type overwrite the previous one, and
        # 'diagrams_text' has its own (pager only) implementation which would
        # silently ignore both the directory and the flags.
        forbidden = [a for a in args[1:] \
                     if a in ['--no_open','-no_open','--merge','-merge']]
        if forbidden and args[0] in ['diagrams', 'diagrams_text']:
            raise self.InvalidCmd("%s: option not supported for NLO processes."%\
                                                          ' '.join(forbidden))

        if args[0] == 'diagrams_text' and len(args)>=2 \
                and args[1] not in ['born','loop','virt','real']:
            raise self.InvalidCmd("Can only display born, loop (virt) or real diagrams, not %s."%args[1])

        if args[0] in ['diagrams', 'processes'] and len(args)>=3 \
                and args[1] not in ['born','loop','virt','real']:
            raise self.InvalidCmd("Can only display born, loop (virt) or real diagrams, not %s."%args[1])
        # rename args[1] if it is 'virt'
        if len(args) > 1:
            if args[1] == 'virt':
                args[1] = 'loop' 

    def check_add(self, args):

        if "--ewsudakov" in args:
            self.ewsudakov = True
            # this is for the Sudakov approximation of EW corrections
            logger.warning('Generating a process including the Sudakov approximation of EW corrections.\n' + \
                           'Please cite arxiv:2110.03714, arxiv:2309.00452')
            args.remove('--ewsudakov')
        
        super(CheckFKS, self).check_add(args)        
        if '$' in args:
            raise self.InvalidCmd('$ syntax not valid for aMC@NLO. $$ syntax is on the other hand a valid syntax.')

    def check_tutorial(self, args):
        """check the validity of the line"""
        if len(args) == 0:
            # a bare 'tutorial' still opens the menu, but from the aMC@NLO
            # interface the NLO tutorial is the natural default
            args.append(self.ask_tutorial(default='nlo'))
        return mg_interface.CheckValidForCmd.check_tutorial(self, args)

    def check_output(self, args):
        """ check the validity of the line"""

        # 'pythia8' is not an output any more; without this it would be taken
        # as the name of the NLO output directory.
        if args and args[0] == 'pythia8':
            raise self.InvalidCmd(mg_interface.PYTHIA8_OUTPUT_DROPPED)

        # '--fks' requests the FKS counter-term standalone output: a lightweight
        # directory exposing the Born building blocks (Born, color-/charge-linked
        # Born, spin-correlated Born) for a single phase-space point.
        self._fks_standalone = '--fks' in args
        if self._fks_standalone:
            args.remove('--fks')
        # '--limits' additionally keeps the real-emission code so that 'launch'
        # can also run the soft/collinear limit test (test_soft_col_limits)
        self._fks_limits = '--limits' in args
        if self._fks_limits:
            args.remove('--limits')
            if not self._fks_standalone:
                raise self.InvalidCmd('"--limits" is only available for the FKS '
                    'standalone output: "output standalone_fortran --fks --limits".')

        if args and args[0] == 'ewsudakovsa':
            self._export_format = 'ewsudsa'
            args.pop(0)
        elif self._fks_standalone:
            # 'standalone' names the MadMatrix export. The FKS oracle extends
            # the Fortran standalone command instead.
            if args and args[0] == 'standalone':
                raise self.InvalidCmd('The FKS Born building-block output is a '
                    'Fortran output: use "output standalone_fortran --fks".')
            if args and args[0] == 'standalone_fortran':
                args.pop(0)
            self._export_format = 'NLO_SA'
        else:
            self._export_format = 'NLO'

        # Elsewhere `standalone` names the MadMatrix (C++/CUDA) export. Here it
        # is accepted as an alias for `standalone_fortran`, so that it is
        # reported as a forbidden *format* below rather than being silently
        # taken as the output path.
        if args and args[0] == 'standalone':
            args[0] = 'standalone_fortran'

        forbidden_formats = ['madevent', 'standalone_fortran']
        

        if not hasattr(self, '_fks_multi_proc') or not self._fks_multi_proc:
            text = 'No processes generated. Please generate a process first.'
            raise self.InvalidCmd(text)

        if not self._curr_model:
            text = 'No model found. Please import a model first and then retry.'
            raise self.InvalidCmd(text)

        if args and args[0][0] != '-':
            if args[0] in forbidden_formats:
                text = 'You generated a NLO process, which cannot be exported in %s mode.\n' % args[0]
                text+= 'Please use the command "output DIR_NAME".\n'
                if args[0] == 'standalone_fortran':
                    text+= 'For the FKS Born building blocks, use "output standalone_fortran --fks".\n'
                raise self.InvalidCmd(text)

            # This is a path
            path = args.pop(0)
            # Check for special directory treatment
            if path == 'auto':
                self.get_default_path()
            elif path != 'auto':
                self._export_dir = path
        else:
            # No valid path
            self.get_default_path()

        self._export_dir = os.path.realpath(self._export_dir)

                
    def check_launch(self, args, options):
        """check the validity of the line. args are DIR and MODE
        MODE being LO, NLO, aMC@NLO or aMC@LO. If no mode is passed, aMC@NLO is used"""
        # modify args in order to be DIR 
        # mode being either standalone or madevent
        
        if not args:
            if self._done_export:
                args.append(self._done_export[0])
                args.append('auto')

                return
            else:
                self.help_launch()
                raise self.InvalidCmd('No default location available, please specify location.')
        
        if len(args) > 2:
            self.help_launch()
            return self.InvalidCmd, 'Invalid Syntax: Too many argument'

        elif len(args) == 2:
            if not args[1] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO', 'auto']:
                raise self.InvalidCmd('%s is not a valid mode, please use "LO", "NLO", "aMC@NLO" or "aMC@LO"' % args[1])
        else:
            #check if args[0] is path or mode
            if args[0] in ['LO', 'NLO', 'aMC@NLO', 'aMC@LO', 'auto'] and self._done_export:
                args.insert(0, self._done_export[0])
            elif os.path.isdir(args[0]) or os.path.isdir(pjoin(MG5DIR, args[0]))\
                    or os.path.isdir(pjoin(MG4DIR, args[0])):
                args.append('auto')
            else:
                self.help_launch()
                raise self.InvalidCmd('%s is not a valid process directory nor run mode' % args[0])

        mode = args[1]
        
        # search for a valid path
        if os.path.isdir(args[0]):
            path = os.path.realpath(args[0])
        elif os.path.isdir(pjoin(MG5DIR,args[0])):
            path = pjoin(MG5DIR,args[0])
        elif  MG4DIR and os.path.isdir(pjoin(MG4DIR,args[0])):
            path = pjoin(MG4DIR,args[0])
        else:    
            raise self.InvalidCmd('%s is not a valid directory' % args[0])
        args[0] = path
                
        # inform where we are for future command
        self._done_export = [path, mode]

        # check for incompatible options/modes
        if options['multicore'] and options['cluster']:
            raise self.InvalidCmd('options -m (--multicore) and -c (--cluster)' + \
                    ' are not compatible. Please choose one.')
        if mode == 'NLO' and options['reweightonly']:
            raise self.InvalidCmd('option -r (--reweightonly) needs mode "aMC@NLO" or "aMC@LO"')


class CheckFKSWeb(mg_interface.CheckValidForCmdWeb, CheckFKS):
    pass

class CompleteFKS(mg_interface.CompleteForCmd):
    
    def complete_display(self, text, line, begidx, endidx):
        """Complete the display command in the context of the FKS interface"""

        args = self.split_arg(line[0:begidx])

        if len(args) == 2 and args[1] in ['diagrams', 'processes']:
            return self.list_completion(text, ['born', 'loop', 'virt', 'real'])
        else:
            return mg_interface.MadGraphCmd.complete_display(self, text, line,
                                                                 begidx, endidx)


    def complete_output(self, text, line, begidx, endidx):
        """Complete the output command in the context of the FKS interface"""
        #don't propose directory use by MG_ME
        forbidden_names = ['MadGraphII', 'Template', 'pythia-pgs', 'CVS',
                            'Calculators', 'MadAnalysis', 'SimpleAnalysis',
                            'mg5', 'DECAY', 'EventConverter', 'Models',
                            'ExRootAnalysis', 'HELAS', 'Transfer_Fct', 'aloha', 
                            'madgraph', 'bin', 'tests', 'input', 'vendor', 'models']
        
        #name of the run =>proposes old run name
        args = self.split_arg(line[0:begidx])
        if len(args) >= 1: 
            if len(args) > 1 and args[1] == 'aloha':
                try:
                    return self.aloha_complete_output(text, line, begidx, endidx)
                except Exception as error:
                    print(error)
            # Directory continuation
            if args[-1].endswith(os.path.sep):
                return [name for name in self.path_completion(text,
                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                        only_dirs = True) if name not in forbidden_names]

            # directory names
            content = [name for name in self.path_completion(text, '.', only_dirs = True) \
                       if name not in forbidden_names]
            return self.list_completion(text, content)


    def complete_launch(self, text, line, begidx, endidx, formatting=True):
        """ complete the launch command"""
        args = self.split_arg(line[0:begidx])

        # Directory continuation
        if args[-1].endswith(os.path.sep):
            return self.path_completion(text,
                                        pjoin(*[a for a in args if a.endswith(os.path.sep)]),
                                        only_dirs = True)
        # Format
        if len(args) == 1:
            out = {'Path from ./': self.path_completion(text, '.', only_dirs = True)}
            if MG5DIR != os.path.realpath('.'):
                out['Path from %s' % MG5DIR] =  self.path_completion(text,
                                     MG5DIR, only_dirs = True, relative=False)
            if MG4DIR and MG4DIR != os.path.realpath('.') and MG4DIR != MG5DIR:
                out['Path from %s' % MG4DIR] =  self.path_completion(text,
                                     MG4DIR, only_dirs = True, relative=False)

        if len(args) == 2:
            modes = ['aMC@NLO', 'NLO', 'aMC@LO', 'LO']
            return self.list_completion(text, modes, line)
            
        #option
        if len(args) >= 3:
            out={}

        if line[0:begidx].endswith('--laststep='):
            opt = ['parton', 'pythia', 'pgs','delphes','auto']
            out['Options'] = self.list_completion(text, opt, line)
        else:

            opt = ['-f', '-c', '-m', '-i', '-x', '-r', '-p', '-o', '-n', 'a',
                    '--force', '--cluster', '--multicore', '--interactive',
                    '--nocompile', '--reweightonly', '--parton', '--only_generation', '--name', '--appl_start_grid']
            out['Options'] = self.list_completion(text, opt, line)
        

        return self.deal_multiple_categories(out, formatting)

class HelpFKS(mg_interface.HelpToCmd):

    def help_display(self):   
        mg_interface.MadGraphCmd.help_display(self)
        logger.info("   In aMC@NLO5, after display diagrams, the user can add the option")
        logger.info("   \"born\", \"virt\" or \"real\" to display only the corresponding diagrams.")

    def help_launch(self):
        """help for launch command"""
        _launch_parser.print_help()

class aMCatNLOInterface(CheckFKS, CompleteFKS, HelpFKS, Loop_interface.CommonLoopInterface):
    
    _fks_display_opts = ['real_diagrams', 'born_diagrams', 'virt_diagrams', 
                         'real_processes', 'born_processes', 'virt_processes']

    _nlo_modes_for_completion = ['all','real']
    display_expansion = False

    def __init__(self, mgme_dir = '', *completekey, **stdin):
        """ Special init tasks for the Loop Interface """
        mg_interface.MadGraphCmd.__init__(self, mgme_dir = '', *completekey, **stdin)
        self.setup()

    def setup(self):
        """ Special tasks when switching to this interface """

        # Refresh all the interface stored value as things like generated
        # processes and amplitudes are not to be reused in between different
        # interfaces
        # Clear history, amplitudes and matrix elements when a model is imported
        # Remove previous imports, generations and outputs from history
        self.history.clean(remove_bef_last='import',
                           to_keep=['set','load','import', 'define'])
        # Reset amplitudes and matrix elements
        self._done_export=False
        self._curr_amps = diagram_generation.AmplitudeList()
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()
        self._v4_export_formats = []
        self._nlo_modes_for_completion = ['all','real']
        self._export_formats = [ 'madevent', 'aloha' ]
        # Do not force NLO model as the user might have asked for reals only.
        # It will anyway be forced later if he attempts virt= or all=.
        self.validate_model(loop_type='real_init', stop=False)
        # Set where to look for CutTools installation.
        # In further versions, it will be set in the same manner as _mgme_dir so that
        # the user can chose its own CutTools distribution.
        self._cuttools_dir=str(pjoin(self._mgme_dir,'vendor','CutTools'))
        if not os.path.isdir(pjoin(self._cuttools_dir, 'src','cts')):
            logger.warning(('Warning: Directory %s is not a valid CutTools directory.'+\
                           'Using default CutTools instead.') % \
                             self._cuttools_dir)
            self._cuttools_dir=str(pjoin(self._mgme_dir,'vendor','CutTools'))
        # Set where to look for IREGI installation
        self._iregi_dir=str(os.path.join(self._mgme_dir,'vendor','IREGI','src'))
        if not os.path.isdir(self._iregi_dir):
            logger.warning(('Warning: Directory %s is not a valid IREGI directory.'+\
                            'Using default IREGI instead.')%\
                           self._iregi_dir)
            self._iregi_dir=str(os.path.join(self._mgme_dir,'vendor','IREGI','src'))

    def do_display(self, line, output=sys.stdout):
        # if we arrive here it means that a _fks_display_opts has been chosen
        args = self.split_arg(line)
        #check the validity of the arguments
        self.check_display(args)

        if args[0] in ['diagrams', 'processes', 'diagrams_text']:
            get_amps_dict = {'real': self._fks_multi_proc.get_real_amplitudes,
                             'born': self._fks_multi_proc.get_born_amplitudes,
                             'loop': self._fks_multi_proc.get_virt_amplitudes}
            if args[0] == 'diagrams':
                if len(args)>=2 and args[1] in list(get_amps_dict.keys()):
                    get_amps = get_amps_dict[args[1]]
                    self._curr_amps = get_amps()
                    #check that if one requests the virt diagrams, there are virt_amplitudes
                    if args[1] == 'loop' and len(self._curr_amps) == 0:
                        raise self.InvalidCmd('No virtuals have been generated')
                    self.draw(' '.join(args[2:]), Dtype = args[1])
                else:
                    for diag_type, get_amps in get_amps_dict.items():
                        self._curr_amps = get_amps()
                        if self._curr_amps:
                            self.draw(' '.join(args[1:]), Dtype=diag_type)

                # set _curr_amps back to empty
                self._curr_amps = diagram_generation.AmplitudeList()

            if args[0] == 'diagrams_text':
                if len(args)>=2 and args[1] in list(get_amps_dict.keys()):
                    get_amps = get_amps_dict[args[1]]
                    self._curr_amps = get_amps()
                    #check that if one requests the virt diagrams, there are virt_amplitudes
                    if args[1] in ['virt', 'loop'] and len(self._curr_amps) == 0:
                        raise self.InvalidCmd('No virtuals have been generated')
                    text = "\n".join([amp.nice_string() for amp in self._curr_amps])
                else:
                    text = 'Born diagrams:\n' 
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['born']())
                    text += '\n\nReal diagrams:'
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['real']())
                    text += '\n\nLoop diagrams:\n'
                    text += '\n'.join(amp.nice_string() for amp in get_amps_dict['loop']())
                pydoc.pager(text)

                # set _curr_amps back to empty
                self._curr_amps = diagram_generation.AmplitudeList()
                    
            elif args[0] == 'processes':
                if len(args)>=2 and args[1] in list(get_amps_dict.keys()):
                    get_amps = get_amps_dict[args[1]]
                    self._curr_amps = get_amps()
                    #check that if one requests the virt diagrams, there are virt_amplitudes
                    if args[1] in ['virt', 'loop'] and len(self._curr_amps) == 0:
                        raise self.InvalidCmd('No virtuals have been generated')
                    print('\n'.join(amp.nice_string_processes() for amp in self._curr_amps))
                else:
                    print('Born processes:')
                    print('\n'.join(amp.nice_string_processes() for amp in get_amps_dict['born']()))
                    print('Real processes:')
                    print('\n'.join(amp.nice_string_processes() for amp in get_amps_dict['real']()))
                    print('Loop processes:')
                    print('\n'.join(amp.nice_string_processes() for amp in get_amps_dict['loop']()))
                # set _curr_amps back to empty
                self._curr_amps = diagram_generation.AmplitudeList()

        else:
            mg_interface.MadGraphCmd.do_display(self,line,output)

    @staticmethod
    def get_parameter_values(model):
        """Return the numerical values of the parameters of model (for the
        default param_card of the model), or None if they cannot be
        evaluated"""

        if isinstance(model, model_reader.ModelReader) and \
                                                  model.get('parameter_dict'):
            return model.get('parameter_dict')
        try:
            reader = model_reader.ModelReader(model)
            reader.set_parameters_and_couplings()
        except Exception as error:
            logger.debug('Cannot evaluate the model parameters: %s' % error)
            return None
        return reader.get('parameter_dict')

    @staticmethod
    def find_unstable_s_channels(fksproc, coloured_only=False):
        """Return the names of the particles with a non-zero width that
        appear as s-channel propagators in the Born diagrams of fksproc
        and can go on shell, i.e. that are heavier than the final-state
        particles they are built from. This excludes e.g. the Z in
        u u~ > t t~, or the top propagator in t* > t g for u u~ > t t~ g.
        If coloured_only, only the colour-charged ones are returned."""

        resonances = set()
        values = {}
        for amp in fksproc.get_born_amplitudes():
            model = amp['process']['model']
            if id(model) not in values:
                values[id(model)] = \
                      aMCatNLOInterface.get_parameter_values(model)
            parameters = values[id(model)]

            def get_value(name):
                """abs of the value of a mass/width, None if unknown"""
                if name.upper() == 'ZERO':
                    return 0.
                if parameters is None or name not in parameters:
                    return None
                return abs(parameters[name].real)

            external = amp['process']['legs']
            final_masses = dict((l['number'],
                                 get_value(model.get_particle(l['id'])['mass']))
                                for l in external if l['state'])

            for diagram in amp['diagrams']:
                # for each leg, the final-state legs it is built from and the
                # number of initial-state legs in it
                content = dict((l['number'],
                                (set([l['number']]) if l['state'] else set(),
                                 0 if l['state'] else 1)) for l in external)
                # the last vertex has no propagator; the other vertices
                # produce one leg, which is an s-channel propagator if it
                # does not contain exactly one initial-state leg (state=True)
                for vertex in diagram['vertices'][:-1]:
                    finals, ninitial = set(), 0
                    for leg in vertex['legs'][:-1]:
                        finals |= content[leg['number']][0]
                        ninitial += content[leg['number']][1]
                    leg = vertex['legs'][-1]
                    content[leg['number']] = (finals, ninitial)
                    if not leg['state']:
                        continue
                    part = model.get_particle(leg['id'])
                    # fake propagators of multi-leg vertices, stable particles
                    if not part or get_value(part.get('width')) == 0.:
                        continue
                    if coloured_only and part.get('color') == 1:
                        continue
                    # the propagator momentum is the sum of the momenta of
                    # these final-state particles: it can only be resonant if
                    # it is heavier than their masses
                    if ninitial:
                        finals = set(final_masses.keys()) - finals
                    mass = get_value(part.get('mass'))
                    threshold = [final_masses[n] for n in finals]
                    if mass is not None and None not in threshold and \
                              mass <= sum(threshold) * (1. + 1e-10):
                        continue
                    resonances.add(part.get('name'))
        return sorted(resonances)

    def warn_resonances_without_cms(self, fksproc, orders):
        """NLO virtual corrections to a process with a resonant propagator
        are not gauge invariant and are dominated by the resonant
        self-energy near the pole unless the complex-mass scheme is used.
        The result can then be wrong by O(100%) (e.g. -90% for
        p p > e+ e- [QED] with m_ll > 30 GeV), without any other sign than
        IR-pole miscancellations during the run.
        With EW corrections (QED in orders) every unstable s-channel
        propagator is concerned; with QCD corrections only the
        colour-charged ones (e.g. an intermediate top quark) receive a
        self-energy correction."""

        if 'QED' in orders:
            resonances = self.find_unstable_s_channels(fksproc)
            corr = 'EW'
        elif 'QCD' in orders:
            resonances = self.find_unstable_s_channels(fksproc,
                                                       coloured_only=True)
            corr = 'QCD'
        else:
            return
        if not resonances:
            return
        logger.warning(
"""NLO %s corrections are requested for a process with s-channel
propagator(s) of unstable particle(s): %s, while the complex-mass scheme is
not used. Close to the resonance the virtual corrections are then not gauge
invariant and are dominated by the resonant self-energy: the NLO cross
section can be wrong by O(100%%). Use 'set complex_mass_scheme True' before
generating the process (see arXiv:1804.10017).""" % (corr, ', '.join(resonances)))

    def do_add(self, line, *args,**opt):
        
        args = self.split_arg(line)
        # Check the validity of the arguments
        self.ewsudakov = False

        # the model imports of check_add (when no model is active yet) and of
        # validate_model reset self._generate_info: keep it
        geninfo = self._generate_info
        self.check_add(args)

        if args[0] == 'model':
            return self.add_model(args[1:])
        elif args[0] != 'process': 
            raise self.InvalidCmd("The add command can only be used with process or model")
        else:
            # if the new nlo process generation mode is enabled, the number of cores to be
            # used has to be passed
            # ncores_for_proc_gen has the following meaning
            #   0 : do things the old way
            #   > 0 use ncores_for_proc_gen
            #   -1 : use all cores
            if self.options['low_mem_multicore_nlo_generation']:
                if self.options['nb_core']:
                    ncores_for_proc_gen = int(self.options['nb_core'])
                else:
                    ncores_for_proc_gen = -1
            else:
                ncores_for_proc_gen = 0
            # the processes of an output are all generated in the same mode
            # (see FKSMultiProcess.add): check it before generating this one
            if hasattr(self, '_fks_multi_proc'):
                try:
                    fks_base.check_generation_mode(self._fks_multi_proc,
                                                   ncores_for_proc_gen)
                except fks_common.FKSProcessError as error:
                    raise self.InvalidCmd(str(error))

            if any ([a.startswith("--loop_filter=") for a in args]):
                for a in args:
                    if a.startswith("--loop_filter="):
                        self._fks_multi_proc['loop_filter'] = a.split('=',1)[1]
                        args.remove(a)
                        break

            if any(a.startswith('--diagram_filter') for a in args):
                raise self.InvalidCmd("--diagram_filter is not supported for "
                    "NLO (FKS) generation: use --tree_diagram_filter (Born "
                    "and real emission diagrams) and/or --loop_diagram_filter "
                    "(loop diagrams).")
            tree_diagram_filter = self.extract_diagram_filter(args,
                                                       '--tree_diagram_filter')
            loop_diagram_filter = self.extract_diagram_filter(args,
                                                       '--loop_diagram_filter')

            line = ' '.join(args[1:])

        # convert the single $ to $$ automatically
        if re.search(r"\b\$\b", line):
            raise MadGraph5Error("Single $ syntax is not supported at NLO, please use $$")

        proc_type=self.extract_process_type(line)
        if proc_type[1] not in ['real', 'LOonly']:
            run_interface.check_compiler(self.options, block=False)
        
        if not aMCatNLOInterface.display_expansion:
            if proc_type[2] != ['QCD'] and proc_type[1] == 'all':
                aMCatNLOInterface.display_expansion = True
                if 'QED' in proc_type[2]:
                    logger.info(
"""------------------------------------------------------------------------
This computation involves NLO EW corrections.
Please also cite ref. 'arXiv:1804.10017' when using results from this code.
------------------------------------------------------------------------
""", '$MG:BOLD')
                else:
                    logger.info(
"""------------------------------------------------------------------------
This computation involve not SM-QCD corrections at NLO.
Please also cite ref. 'arXiv:1804.10017' when using results from this code.
------------------------------------------------------------------------
""", '$MG:BOLD')


        # this warning is to remind us that for the moment adding EW sudakov
        # corrections lacks the LO2 contributions unless EW corrections are
        # also computed
        if self.ewsudakov and not 'QED' in proc_type[2]:
            logger.warning('The EW Sudakov corrections might be incomplete.' + \
                           'They will not include contributions of QCD origin ' + \
                           'stemming from the LO2, if it exists.')

        self.validate_model(proc_type[1], coupling_type=proc_type[2])
        self._generate_info = geninfo

        #now generate the amplitudes as usual
        #self.options['group_subprocesses'] = 'False'
        collect_mirror_procs = False
        ignore_six_quark_processes = self.options['ignore_six_quark_processes']
        if ',' in line:
            myprocdef, line = mg_interface.MadGraphCmd.extract_decay_chain_process(self,line)
            if myprocdef.are_decays_perturbed():
                raise MadGraph5Error("Decay processes cannot be perturbed")
        else:
            myprocdef = mg_interface.MadGraphCmd.extract_process(self,line)

        self.proc_validity(myprocdef,'aMCatNLO_%s'%proc_type[1])

        # set the orders
        # if some orders have been set by the user,
        # check that all the orders of the model have been specified
        # set to default those which have not been specified and warn the user
        if myprocdef['orders'] and not all([o in list(myprocdef['orders'].keys()) for o in myprocdef['model'].get_coupling_orders()]):
            for o in myprocdef['model'].get_coupling_orders():
                if o not in list(myprocdef['orders'].keys()):
                    myprocdef['orders'][o] = self.options['default_unset_couplings']
                    logger.warning(('%s order is missing in the process definition. It will be set to "default user couplings": %s\n' + \
                                   'If this is not what you need, please regenerate with the correct orders.') % (o,myprocdef['orders'][o]))

        # raise a warning if squared-orders constraints are not of the '<=' kind
        if myprocdef['sqorders_types'] and any([v != '<=' for v in  myprocdef['sqorders_types'].values()]):
            raise MadGraph5Error('The squared-order constraints passed are not \'<=\'.\n Other kind of squared-order constraints are not supported at NLO')

        # this is in case no orders have been passed
        if not myprocdef['squared_orders'] and not myprocdef['orders']:
            # find the minimum weighted order, then extract the values for the various
            # couplings in the model
            weighted = diagram_generation.MultiProcess.find_optimal_process_orders(myprocdef,
                                                           tree_diagram_filter)
            if not weighted:
                raise MadGraph5Error('\nProcess orders cannot be determined automatically. \n' + \
                                      'Please specify them from the command line.')

            # this is a very rough attempt, and works only to guess QED/QCD
            qed, qcd = fks_common.get_qed_qcd_orders_from_weighted(len(myprocdef['legs']), 
                                                                   self._curr_model.get('order_hierarchy'), 
                                                                   weighted['WEIGHTED'])

            if qed < 0 or qcd < 0:
                raise MadGraph5Error('\nAutomatic process-order determination lead to negative constraints:\n' + \
                      ('QED: %d,  QCD: %d\n' % (qed, qcd)) + \
                      'Please specify the coupling orders from the command line.')
            if self.options['nlo_mixed_expansion']:
                orders = {'QED': 2*qed, 'QCD': 2*qcd}
                # set all the other coupling to zero
                for o in myprocdef['model'].get_coupling_orders():
                    if o not in ['QED', 'QCD']:
                        if self._curr_model.get('order_hierarchy')[o] == self._curr_model.get('order_hierarchy')['QCD']:
                            orders[o] = 2*qcd
                        elif self._curr_model.get('order_hierarchy')[o] == self._curr_model.get('order_hierarchy')['QED']:
                            orders[o] = 2*qed
                        else:
                            orders[o] = 0

                myprocdef.set('squared_orders', orders)
                # warn the user of what happened
                logger.info(('Setting the born squared orders automatically in the process definition to %s.\n' + \
                                'If this is not what you need, please regenerate with the correct orders.'), 
                                ' '.join(['%s^2<=%s' %(k,v) if v else '%s=%s' % (k,v) for k,v in myprocdef['squared_orders'].items()]), 
                                '$MG:BOLD')
            else: 
                orders = {'QED': qed, 'QCD': qcd}
                sqorders = {'QED': 2*qed, 'QCD': 2*qcd}
                # set all the other coupling to zero
                for o in myprocdef['model'].get_coupling_orders():
                    if o not in ['QED', 'QCD']:
                        if self._curr_model.get('order_hierarchy')[o] == self._curr_model.get('order_hierarchy')['QCD']:
                            orders[o] = qcd
                            sqorders[o] = 2*qcd
                        elif self._curr_model.get('order_hierarchy')[o] == self._curr_model.get('order_hierarchy')['QED']:
                            orders[o] = qed
                            sqorders[o] = 2*qed
                        elif o in self._curr_model.get('expansion_order') and self._curr_model.get('expansion_order')[o]<50:
                            orders[o] = self._curr_model.get('expansion_order')[o]
                            sqorders[o] = 2*self._curr_model.get('expansion_order')[o]
                        else:
                            orders[o] = 0
                            sqorders[o] = 0

                myprocdef.set('orders', orders)
                myprocdef.set('squared_orders', sqorders)
                # warn the user of what happened
                logger.info(('Setting the born orders automatically in the process definition to %s.\n' + \
                                'If this is not what you need, please regenerate with the correct orders.'), 
                                ' '.join(['%s<=%s' %(k,v) if v else '%s=%s' % (k,v) for k,v in myprocdef['orders'].items()]), 
                                '$MG:BOLD')                

        # now check that all couplings that are there in orders also appear
        # in squared_orders. If not, set the corresponding one
        for k, v in myprocdef['orders'].items():
            if k not in myprocdef['squared_orders'].keys():
                myprocdef['squared_orders'][k] = 2*v 
                logger.warning('Order %s is not constrained as squared_orders. Using: %s^2=%d' % (k,k,2*v) )

        # check that all the couplings of the model have been constrained
        # in the squared orders, otherwise set the others to zero
        # (not the hidden order an interference process adds to the model)
        model_orders = base_objects.visible_orders(
                                     myprocdef['model'].get('coupling_orders'))
        for o in model_orders:
            if o not in myprocdef['squared_orders'].keys():
                logger.warning('No squared order constraint for order %s. Setting to 0' % o)
                myprocdef['squared_orders'][o] = 0 

        myprocdef['born_sq_orders'] = copy.copy(myprocdef['squared_orders'])
        # split all orders in the model, for the moment it's the simplest solution
        # mz02/2014
        #if proc_type[1] != 'only':
        myprocdef['split_orders'] += [o for o in model_orders \
                if o not in myprocdef['split_orders']]

        # now set the squared orders
        if not myprocdef['squared_orders']:
            logger.warning('No squared orders have been provided, will be guessed by the order constraints')
            for ord, val in myprocdef['orders'].items():
                myprocdef['squared_orders'][ord] = 2 * val

        # then increase the orders which are perturbed
        for pert in myprocdef['perturbation_couplings']:

            if not self.options['nlo_mixed_expansion'] and pert not in proc_type[2]:
                    continue


            # if orders have been specified increase them
            if list(myprocdef['orders'].keys()) != ['WEIGHTED']:
                try:
                    myprocdef['orders'][pert] += 1
                except KeyError:
                    # if the order is not specified
                    # then MG does not put any bound on it
                    ###myprocdef['orders'][pert] = 99
                    pass
                try:
                    myprocdef['squared_orders'][pert] += 2
                except KeyError:
                    # the order is not provided, assume
                    # it is originally zero
                    myprocdef['squared_orders'][pert] = 2

        # update also the WEIGHTED entry
        if 'WEIGHTED' in list(myprocdef['orders'].keys()):
            myprocdef['orders']['WEIGHTED'] += 1 * \
                    max([myprocdef.get('model').get('order_hierarchy')[ord] for \
                    ord in myprocdef['perturbation_couplings']])

            myprocdef['squared_orders']['WEIGHTED'] += 2 * \
                    max([myprocdef.get('model').get('order_hierarchy')[ord] for \
                    ord in myprocdef['perturbation_couplings']])
        # if [orders] have not been specified, 
        # finally set perturbation_couplings to **all** the coupling orders 
        # available in the model.
        # This is necessary because when doing EW corrections one only specifies
        # squared-orders constraints. In that case, all kind of splittings/loop-particles
        # must be included
        if not myprocdef['orders'] and self.options['nlo_mixed_expansion']:
            myprocdef['perturbation_couplings'] = list(model_orders)

        self._curr_proc_defs.append(myprocdef)

        # check if any initial-state leg contains leptons. If so, switch 
        # include_lepton_initiated_processes to True
        lep_ids = [11, -11, 13, -13, 15, -15]
        initial_ids = []
        merged_particles = myprocdef['model'].get('merged_particles')
        for leg in [leg for leg in myprocdef['legs'] if not leg['state']]:
            if leg.get('flavor'):
                initial_ids.extend(leg['flavor'])
                continue
            for pdg in leg['ids']:
                members = merged_particles.get(abs(pdg))
                if members:
                    initial_ids.extend(member if pdg > 0 else -member
                                       for member in members)
                else:
                    initial_ids.append(pdg)
        if any([idd in lep_ids for idd in initial_ids]) and \
                not self.options['include_lepton_initiated_processes']:
            logger.warning('The process definition has leptons in the initial state')
            logger.warning('The flag \'include_lepton_initiated_processes\' will be set to True')
            self.do_set('include_lepton_initiated_processes True')

        # this is the options dictionary to pass to the FKSMultiProcess
        fks_options = {'OLP': self.options['OLP'],
                       'ignore_six_quark_processes': self.options['ignore_six_quark_processes'],
                       'init_lep_split': self.options['include_lepton_initiated_processes'],
                       'ncores_for_proc_gen': ncores_for_proc_gen,
                       'ewsudakov': self.ewsudakov,
                       'nlo_mixed_expansion': self.options['nlo_mixed_expansion'],
                       'loop_filter':self._fks_multi_proc['loop_filter'] if hasattr(self, '_fks_multi_proc') else None,
                       'diagram_filter': tree_diagram_filter,
                       'loop_diagram_filter': loop_diagram_filter}

        fksproc =fks_base.FKSMultiProcess(myprocdef,fks_options)
        try:
            self._fks_multi_proc.add(fksproc)
        except AttributeError: 
            self._fks_multi_proc = fksproc
            self._fks_multi_proc['loop_filter'] = fks_options['loop_filter']
        # the matrix elements of the previous output (which the serial mode
        # keeps for another output of the same processes) lack this process
        self._curr_matrix_elements = helas_objects.HelasMultiProcess()

        if proc_type[1] in ['all', 'virt'] and \
                                  not self.options['complex_mass_scheme']:
            self.warn_resonances_without_cms(fksproc, proc_type[2])

        if not aMCatNLOInterface.display_expansion and  self.options['nlo_mixed_expansion']:
            base = {}
            for amp in self._fks_multi_proc.get_born_amplitudes():
                nb_part = len(amp['process']['legs'])
                for diag in amp['diagrams']:   
                    if nb_part not in  base:
                        base[nb_part] = diag.get('orders')
                    elif base[nb_part] != diag.get('orders'):
                        aMCatNLOInterface.display_expansion = True
                        logger.info(
"""------------------------------------------------------------------------
This computation can involve not only purely SM-QCD corrections at NLO.
Please also cite ref. 'arXiv:1804.10017' when using results from this code.
------------------------------------------------------------------------
""", '$MG:BOLD')
                        break
                else:
                    continue
                break




    def do_output(self, line):
        """Main commands: Initialize a new Template or reinitialize one"""
        
        args = self.split_arg(line)
        me_exporters = [arg.split('=', 1)[1] for arg in args
                        if arg.startswith('--me_exporter=')]
        if len(me_exporters) > 1:
            raise self.InvalidCmd('Only one --me_exporter option is allowed')
        me_exporter = me_exporters[0] if me_exporters else None
        if me_exporter and me_exporter != 'mg7':
            raise self.InvalidCmd(
                'NLO real offloading supports only the built-in '
                '--me_exporter=mg7')
        # Check Argument validity
        self.check_output(args)
        
        noclean = '-noclean' in args
        force = '-f' in args 
        nojpeg = '-nojpeg' in args

        # For NLO, the group_subprocesses is automatically set to false
        group_processes = False
        # initialize the writer
        if self._export_format in ['NLO','ewsudsa','NLO_SA']:
            output_type_dict = {'NLO': 'amcatnlo', 'ewsudsa': 'ewsudsa',
                                'NLO_SA': 'amcatnlo_sa'}
            self._curr_exporter = export_v4.ExportV4Factory(self, noclean,
                      output_type=output_type_dict[self._export_format],
                      group_subprocesses=group_processes)
            output_options = dict(
                arg[2:].split('=', 1) if '=' in arg else (arg[2:], True)
                for arg in args if arg.startswith('--'))
            if 'vector_size' in output_options:
                self._curr_exporter.opt['nlo_real_vector_size'] = \
                    output_options['vector_size']
            if me_exporter:
                self._curr_exporter.opt['nlo_real_offload'] = True

            self._curr_exporter.pass_information_from_cmd(self)

        self._me_curr_exporter = None

        # check if a dir with the same name already exists
        if not force and not noclean and os.path.isdir(self._export_dir)\
               and self._export_format in ['NLO', 'ewsudsa', 'NLO_SA']:
            # Don't ask if user already specified force or noclean
            logger.info('INFO: directory %s already exists.' % self._export_dir)
            logger.info('If you continue this directory will be deleted and replaced.')
            answer = self.ask('Do you want to continue?', 'y', ['y','n'], 
                                                timeout=self.options['timeout'])
            if answer != 'y':
                raise self.InvalidCmd('Stopped by user request')

        # The FKS exporters keep the trace color basis (ddm_color_basis): this
        # also turns off a DDM basis left on by an earlier 'check'. (Before
        # the matrix elements are generated, which builds their color bases.)
        self.set_color_basis_mode(self._curr_exporter)

        # the matrix elements, before the directory is replaced (their
        # generation may fail: the low_mem_multicore_nlo_generation mode
        # generates the real and virtual amplitudes now)
        if self._export_format in ['NLO', 'ewsudsa', 'NLO_SA']:
            self.generate_fks_matrix_elements(group=group_processes)
        try:
            # if one gets here either used -f or answered yes to the question about
            # removing the dir
            if os.path.exists(self._export_dir):
                shutil.rmtree(self._export_dir)

            # Make a Template Copy
            if self._export_format in ['NLO', 'ewsudsa', 'NLO_SA']:
                self._curr_exporter.copy_fkstemplate(self._curr_model)
            if me_exporter:
                from madmatrix import output as madmatrix_output
                exporter_class = madmatrix_output.MadMatrixExporterFactory.\
                    get_exporter_class('nlo_real')
                exporter_options = dict(self.options)
                exporter_options['output_options'] = output_options
                self._me_curr_exporter = exporter_class(
                    self._export_dir, exporter_options)
                self._me_curr_exporter.pass_information_from_cmd(self)
                # The primary FKS/MadLoop exporter mutates model-level coupling
                # dependency caches while writing virtuals. Preserve a pristine
                # tree model for the one shared MadMatrix conversion pass.
                self._me_curr_exporter.prepare_nlo_tree(
                    copy.deepcopy(self._curr_model))

            # Reset _done_export, since we have new directory
            self._done_export = False

            # Perform export and finalize right away
            self.export(nojpeg, group_processes=group_processes)
        finally:
            # the low_mem_multicore_nlo_generation mode: the temporary files
            # of the matrix elements, which an output reads (a next output
            # generates them again)
            if isinstance(self._curr_matrix_elements, fks_helas.FKSHelasMultiProcess) \
               and self._fks_multi_proc['ncores_for_proc_gen']:
                self._curr_matrix_elements.remove_tmpdir()

        # Pass potential new information generated during the export.
        self._curr_exporter.pass_information_from_cmd(self)

        # Automatically run finalize
        # The NLO-real exporter has already converted its deliberately
        # tree-only dependency set. The generic dual-export finalizer would
        # feed it the full FKS/MadLoop dependency set a second time, including
        # UV and loop ALOHA content, and overwrite that isolated model pass.
        nlo_real_exporter = self._me_curr_exporter
        self._me_curr_exporter = None
        try:
            self.finalize(nojpeg)
        finally:
            self._me_curr_exporter = nlo_real_exporter
            
        # Generate the virtuals if from OLP
        if self.options['OLP']!='MadLoop':
            self._curr_exporter.generate_virtuals_from_OLP(
              self.born_processes_for_olp,self._export_dir,self.options['OLP'])
                
        # Remember that we have done export
        self._done_export = (self._export_dir, self._export_format)

        # Reset _export_dir, so we don't overwrite by mistake later
        self._export_dir = None

    def generate_fks_matrix_elements(self, group=False):
        """Generate the matrix elements of the processes (a
        FKSHelasMultiProcess), unless the previous output, of the same
        processes, did (the serial mode keeps them, until an 'add process')"""

        # Sort amplitudes according to number of diagrams,
        # to get most efficient multichannel output
        self._curr_amps.sort(key = lambda a: a.get_number_of_diagrams(), reverse=True)

        if self._curr_matrix_elements.get_matrix_elements():
            return
        if group:
            raise MadGraph5Error("Cannot group subprocesses when "+\
                                                      "exporting to NLO")
        # in the low_mem_multicore_nlo_generation mode, the processes which
        # finalize the matrix elements compute the Born support records of
        # the exporter (see write_directories_multicore)
        me_record = None
        if self._fks_multi_proc['ncores_for_proc_gen']:
            me_record = getattr(self._curr_exporter, 'born_support_record', None)
        self._curr_matrix_elements = fks_helas.FKSHelasMultiProcess(
                                self._fks_multi_proc, 
                                loop_optimized= self.options['loop_optimized_output'],
                                me_record=me_record)

    # Export a matrix element  
    def export(self, nojpeg=False, group_processes=False):
        """Export the matrix elements of generate_fks_matrix_elements to file"""

        self._curr_helas_model = helas_call_writers.FortranUFOHelasCallWriter(self._curr_model)
        if self._me_curr_exporter:
            self._me_curr_helas_model = \
                self._me_curr_exporter.helas_exporter(
                    self._curr_model, options=self.options)
        else:
            self._me_curr_helas_model = None
        if self._fks_multi_proc['ncores_for_proc_gen']:
            #new NLO generation
            if self._curr_matrix_elements['has_loops']:
                self._curr_exporter.opt['mp'] = True
            self._curr_exporter.model = self._curr_model

        calls = 0
        splitorders = []

        path = self._export_dir

        if self._export_format in ['NLO', 'ewsudsa', 'NLO_SA']:
            path = os.path.join(path, 'SubProcesses')

            #_curr_matrix_element is a FKSHelasMultiProcess Object 
            proc_charac = self._curr_exporter.proc_characteristic 
            for charac in ['has_isr', 'has_fsr', 'has_loops']:
                proc_charac[charac] = self._curr_matrix_elements[charac]

            # the matrix elements were generated in the
            # low_mem_multicore_nlo_generation mode (files written by
            # fks_helas_objects) or in the serial mode
            low_mem = bool(self._fks_multi_proc['ncores_for_proc_gen'])
            matrix_elements = self._curr_matrix_elements.get('matrix_elements')

            # Save processes instances generated
            self.born_processes_for_olp = []
            self.born_processes = []

            if low_mem:
                calls, splitorders, max_loop_vertex_ranks = \
                                          self.write_directories_multicore(path)
            else:
                # the Born providers, which the exporter assigns before
                # writing the directories (none for an exporter without them)
                born_support_record = getattr(self._curr_exporter,
                                              'born_support_record', None)
                if born_support_record:
                    self._curr_exporter.prepare_born_support(
                            [(export_fks.born_dir_name(me.born_me.get('processes')[0]),) + \
                             born_support_record(me) for me in matrix_elements])
                borndirs = []
                for ime, me in enumerate(matrix_elements):
                    #me is a FKSHelasProcessFromReals
                    calls_dir, splitorders_dir = \
                            self._curr_exporter.generate_directories_fks(me, 
                            self._curr_helas_model, 
                            ime, len(matrix_elements),
                            path,self.options['OLP'],
                            second_exporter=self._me_curr_exporter,
                            second_helas=self._me_curr_helas_model)
                    calls += calls_dir
                    splitorders += [so for so in splitorders_dir if so not in splitorders]
                    self.born_processes_for_olp.append(me.born_me.get('processes')[0])
                    self.born_processes.append(me.born_me.get('processes'))
                    borndirs.append(export_fks.born_dir_name(me.born_me.get('processes')[0]))
                self._curr_exporter.write_born_dirs(borndirs)

                max_loop_vertex_ranks = [me.get_max_loop_vertex_rank() for \
                                         me in self._curr_matrix_elements.get_virt_matrix_elements()]

            card_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                     'procdef_mg5.dat')
            
            if self.options['loop_optimized_output'] and \
               len(max_loop_vertex_ranks) > 0:
                self._curr_exporter.write_coef_specs_file(max_loop_vertex_ranks)
            if self._generate_info:
                self._curr_exporter.write_procdef_mg5(card_path, #
                                self._curr_model['name'],
                                self._generate_info)
                try:
                    cmd.Cmd.onecmd(self, 'history .')
                except Exception:
                    logger.debug('fail to run command \"history cmd\"')
                    pass
            subproc_path = os.path.join(path, os.path.pardir, 'SubProcesses', \
                                     'initial_states_map.dat')
            nmaxpdf = self._curr_exporter.write_init_map(subproc_path,
                                self._curr_matrix_elements.get('initial_states'))
            self._curr_exporter.write_maxproc_files(nmaxpdf, 
                                os.path.join(path, os.path.pardir, 'SubProcesses'))

            self._curr_exporter.write_orderstag_file(splitorders, self._export_dir)
            if self._me_curr_exporter:
                self._me_curr_exporter.finalize_nlo_model(self._curr_model)

        cpu_time1 = time.time()


    def write_directories_multicore(self, path):
        """low_mem_multicore_nlo_generation: write the directories of the
        matrix elements (the files written by FKSHelasMultiProcess) with
        processes forked from this one (each with its own copy of the
        exporter), then what the serial mode accumulates in the exporter
        while it writes them one after the other, in the same order.
        Returns calls, splitorders and max_loop_vertex_ranks, as the serial
        mode computes them."""

        matrix_elements = self._curr_matrix_elements.get('matrix_elements')
        me_processes = self._curr_matrix_elements['me_processes']

        # the Born providers, which the processes writing the directories take
        # from the exporter: the records of the matrix elements were computed
        # by the processes which finalized them (see generate_fks_matrix_elements)
        if getattr(self._curr_exporter, 'born_support_record', None):
            self._curr_exporter.prepare_born_support(
                    [(export_fks.born_dir_name(processes[0]),) + record for \
                     processes, record in zip(me_processes,
                                              self._curr_matrix_elements['me_records'])])

        logger.info('Writing directories...')
        if self._me_curr_exporter:
            for ime, mefile in enumerate(matrix_elements):
                me = fks_helas.load_matrix_element(mefile, ime + 1)
                self._me_curr_exporter.prime_nlo_tree_dependencies(
                    me, self._me_curr_helas_model)
                del me
        diroutputmap = fks_common.low_mem_map(generate_directories_fks_async,
                [(self._curr_exporter, mefile, self._curr_helas_model, ime,
                  len(matrix_elements), path, self.options['OLP'],
                  self._me_curr_exporter, self._me_curr_helas_model) \
                 for ime, mefile in enumerate(matrix_elements)],
                self._fks_multi_proc['ncores_for_proc_gen'])

        # what the serial mode accumulates, in the order of the matrix elements
        calls = 0
        splitorders = []
        max_loop_vertex_ranks = []
        for diroutput in diroutputmap:
            calls += diroutput['calls']
            if self._me_curr_exporter:
                self._me_curr_exporter.merge_nlo_tree_dependencies(
                    diroutput['second_dependencies'])
            splitorders += [so for so in diroutput['splitorders'] if so not in splitorders]
            if diroutput['max_loop_vertex_rank'] is not None:
                max_loop_vertex_ranks.append(diroutput['max_loop_vertex_rank'])
        merge_proc_characteristics(self._curr_exporter.proc_characteristic,
                [diroutput['proc_characteristic'] for diroutput in diroutputmap],
                [diroutput['splitting_types'] for diroutput in diroutputmap])

        self.replay_aloha_computations([info for diroutput in diroutputmap \
                                        for info in diroutput['aloha_infos']])
        pending = [diroutput['dp_terminal_pending'] for diroutput in diroutputmap \
                   if diroutput['dp_terminal_pending']]
        if pending:
            merged = dict(getattr(self._curr_exporter, '_dp_terminal_pending', {}))
            for entries in pending:
                merged.update(entries)
            self._curr_exporter._dp_terminal_pending = merged
        self._curr_exporter.write_born_dirs([diroutput['borndir'] \
                                             for diroutput in diroutputmap])

        for processes in me_processes:
            self.born_processes_for_olp.append(processes[0])
            self.born_processes.append(processes)

        return calls, splitorders, max_loop_vertex_ranks


    def replay_aloha_computations(self, aloha_infos):
        """Redo here the aloha computations of the workers, the analytic
        information of the loop wavefunctions (aloha_infos: the keys they
        added to AbstractALOHAModel.cached_interaction_infos), in the order of
        the serial mode, which does them in this process. The aloha routines
        written by finalize depend on them (through the aloha model of the
        exporter and the aloha kernel, whose order of the variables gives the
        order of the factors)."""

        if not aloha_infos:
            return
        aloha_model = self._curr_exporter.get_aloha_model(self._curr_model)
        for lorentz, outgoing, tag, info in aloha_infos:
            aloha_model.get_info(info, lorentz, outgoing, tag, cached=True)


    def do_launch(self, line):
        """Main commands: Ask for editing the parameters and then execute the code (NLO or aMC@(N)LO)
        """
        old_cwd = os.getcwd()
        argss = self.split_arg(line)
        # check argument validity and normalise argument
        (options, argss) = _launch_parser.parse_args(argss)
        options = options.__dict__
        self.check_launch(argss, options)
        if not os.path.isdir(os.path.join(os.getcwd(), argss[0], 'Events')):
            self.do_switch('ML5')
            return mg_interface.MadGraphCmd.do_launch(self,line)
#        if self.options['automatic_html_opening']:
#            misc.open_file(os.path.join(os.getcwd(), argss[0], 'crossx.html'))
#            self.options['automatic_html_opening'] = False

        if options['interactive']:
            if isinstance(self, extended_cmd.CmdShell):
                ME = run_interface.aMCatNLOCmdShell(me_dir=argss[0], options=self.options)
            else:
                ME = run_interface.aMCatNLOCmd(me_dir=argss[0],options=self.options)
                ME.pass_in_web_mode()
            # transfer interactive configuration.  Generation-time-only options
            # (e.g. zerowidth_tchannel, whose T-channel-width treatment is baked
            # into the matrix element at 'output' time) appear in the MG5 history
            # but are NOT valid run-time 'set' options -- replaying them would
            # raise in the run interface's check_set -- so set_lines_for_run
            # leaves them out; a genuine run-time 'set zerowidth_tchannel' typed
            # at the run prompt still goes straight to the run interface and
            # correctly crashes.
            for line in extended_cmd.set_lines_for_run(self.history):
                ME.exec_cmd(line)
            stop = self.define_child_cmd_interface(ME)                
            return stop

        ext_program = launch_ext.aMCatNLOLauncher(argss[0], self, run_mode=argss[1],
                                                  shell = isinstance(self, extended_cmd.CmdShell),
                                                  **options)
        ext_program.run()


class aMCatNLOInterfaceWeb(mg_interface.CheckValidForCmdWeb, aMCatNLOInterface):
    pass

_launch_usage = "launch [DIRPATH] [MODE] [options]\n" + \
                "-- execute the aMC@NLO output present in DIRPATH\n" + \
                "   By default DIRPATH is the latest created directory\n" + \
                "   MODE can be either LO, NLO, aMC@NLO or aMC@LO (if omitted, it is asked in a separate question)\n" + \
                "     If mode is set to LO/NLO, no event generation will be performed, but only the \n" + \
                "     computation of the total cross-section and the filling of parton-level histograms \n" + \
                "     specified in the DIRPATH/SubProcesses/madfks_plot.f file.\n" + \
                "     If mode is set to aMC@LO/aMC@NLO, after the cross-section computation, a .lhe \n" + \
                "     event file is generated which will be showered with the MonteCarlo specified \n" + \
                "     in the run_card.dat\n"

_launch_parser = misc.OptionParser(usage=_launch_usage)
_launch_parser.add_option("--allow_qed_ps", default=False, action='store_true',
                            help="Allow experimental NLO QED+PS matching by bypassing the launch restriction")
_launch_parser.add_option("-f", "--force", default=False, action='store_true',
                                help="Use the card present in the directory for the launch, without editing them")
_launch_parser.add_option("-c", "--cluster", default=False, action='store_true',
                            help="Submit the jobs on the cluster")
_launch_parser.add_option("-i", "--interactive", default=False, action='store_true',
                            help="Use the interactive console")
_launch_parser.add_option("-m", "--multicore", default=False, action='store_true',
                            help="Submit the jobs on multicore mode")
_launch_parser.add_option("-x", "--nocompile", default=False, action='store_true',
                            help="Skip compilation. Ignored if no executable is found")
_launch_parser.add_option("-r", "--reweightonly", default=False, action='store_true',
                            help="Skip integration and event generation, just run reweight on the" + \
                                 " latest generated event files (see list in SubProcesses/nevents_unweighted)")
_launch_parser.add_option("-p", "--parton", default=False, action='store_true',
                            help="Stop the run after the parton level file generation (you need " + \
                                    "to shower the file in order to get physical results)")
_launch_parser.add_option("-o", "--only_generation", default=False, action='store_true',
                            help="Skip grid set up, just generate events starting from " + \
                            "the last available results")
# the last option is different from the corresponding in amcatnlo_run_interface as it stores the 
# 'name' entry of the options, not the run_name one
_launch_parser.add_option("-n", "--name", default=False, dest='name',
                            help="Provide a name to the run")
_launch_parser.add_option("-R", "--reweight", default=False, action='store_true',
                            help="Run the reweight module (reweighting by different model parameter")
_launch_parser.add_option("-M", "--madspin", default=False, action='store_true',
                            help="Run the madspin package")
