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
"""Methods and classes to export matrix elements to v4 format."""

from __future__ import absolute_import
import copy
import fractions
import glob
import logging
import os
import stat
import sys
import re
import shutil
import subprocess
import itertools
import time
import datetime


import aloha

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.loop.loop_helas_objects as loop_helas_objects
import madgraph.loop.loop_onia as loop_onia
import madgraph.loop.onia_ep as onia_ep
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.various.banner as banner_mod
import madgraph.various.misc as misc
import madgraph.various.ninjars_install as ninjars_install
import madgraph.various.tirust_install as tirust_install
import madgraph.various.q_polynomial as q_polynomial
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.export_v4 as export_v4
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.process_checks as process_checks
import madgraph.various.progressbar as pbar
import madgraph.various.q_polynomial as q_polynomial
import madgraph.core.color_amp as color_amp
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.models.check_param_card as check_param_card
from madgraph.loop.loop_base_objects import LoopDiagram
from madgraph.loop.MadLoopBannerStyles import MadLoopBannerStyles
pjoin = os.path.join

import aloha.create_aloha as create_aloha
import madgraph.models.write_param_card as param_writer
from madgraph import MadGraph5Error, MG5DIR, InvalidCmd
from madgraph.iolibs.files import cp, ln, mv
pjoin = os.path.join
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.loop_exporter')

_FACTORIZED_WL_VERTEX = r'(?:FFV1L1_2|FFV1L2_1)'
_PRIVATE_MP_WL_ROUTINE = (r'(?:TERMINAL_WL_\d+_\d+'
    r'(?:_FACTOR_'+_FACTORIZED_WL_VERTEX+r')?|'
    r'UPDATE_WL_[2-4]_1_FACTOR_'+_FACTORIZED_WL_VERTEX+r')')
_PRIVATE_MP_WL_CALL = r'\bCALL MP_\w*'+_PRIVATE_MP_WL_ROUTINE+r'\('

#===============================================================================
# LoopExporterFortran
#===============================================================================
class LoopExporterFortran(object):
    """ Class to define general helper functions to the different 
        loop fortran exporters (ME, SA, MEGroup, etc..) which will inherit both 
        from this class AND from the corresponding ProcessExporterFortran(ME,SA,...).
        It plays the same role as ProcessExporterFrotran and simply defines here
        loop-specific helpers functions necessary for all loop exporters.
        Notice that we do not have LoopExporterFortran inheriting from 
        ProcessExporterFortran but give access to arguments like dir_path and
        clean using options. This avoids method resolution object ambiguity"""

    # The loop color bases (LoopColorBasis) are always trace ones, and the FKS
    # exporters, which inherit this, read the color flow of their Born and real
    # color bases directly (color_flow_decomposition): placed before the
    # ProcessExporterFortran(SA, ME, ...) in the MRO, this keeps every loop
    # output (MadLoop, loop-induced, NLO) on the trace basis.
    ddm_color_basis = 'never'

    default_opt = dict(export_v4.ProcessExporterFortran.default_opt)
    default_opt.update({'clean': False, 'complex_mass':False,
                        'export_format':'madloop', 'mp':True,
                        'loop_dir':'', 'cuttools_dir':'', 
                        'fortran_compiler':'gfortran',
                        'SubProc_prefix': 'P',
                        'output_dependencies': 'external',
                        'compute_color_flows': False,
                        'mode':'',
                        'vector_size':0})

    include_names    = {'ninja' : 'mninja.mod',
                        'ninjars' : 'mninjars.mod',
                        'tirust' : 'mtirust.mod',
                        'golem' : 'generic_function_1p.mod',
                        'samurai':'msamurai.mod',
                        'collier': 'collier.mod'}

    def __init__(self, dir_path = "", opt=None):
        """Initiate the LoopExporterFortran with directory information on where
        to find all the loop-related source files, like CutTools"""

        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)

        self.SubProc_prefix = self.opt['SubProc_prefix']
        self.loop_dir = self.opt['loop_dir']
        self.cuttools_dir = self.opt['cuttools_dir']
        self.fortran_compiler = self.opt['fortran_compiler']
        self.dependencies = self.opt['output_dependencies']
        self.compute_color_flows = self.opt['compute_color_flows']

        super(LoopExporterFortran,self).__init__(dir_path, self.opt)        


    def link_CutTools(self, targetPath):
        """Link the CutTools source directory inside the target path given
        in argument"""
        
        # special mode
        if 'heptools_install_dir' in self.opt and self.dependencies =='external':
            if os.path.exists(pjoin(self.opt['heptools_install_dir'], 'CutTools','lib')) and\
            not os.path.exists(pjoin(self.cuttools_dir,'includects', 'libcts.a')):
                misc.sprint('Going to use pre-compiled version of CutTools') 
                # Create the links to the lib folder
                linkfiles = ['libcts.a', 'mpmodule.mod']
                for file in linkfiles:
                    ln(os.path.join(self.opt['heptools_install_dir'],'CutTools','lib',file),
                                    os.path.join(targetPath,'lib'),abspath=True)
                return
            
        if self.dependencies=='internal':
            new_CT_path = pjoin(targetPath,'Source','CutTools')
            misc.copytree(self.cuttools_dir, new_CT_path, symlinks=True)
            
            current = misc.detect_current_compiler(os.path.join(new_CT_path,
                                                                    'makefile'))
            new = 'gfortran' if self.fortran_compiler is None else \
                                                          self.fortran_compiler
            if current != new:
                misc.mod_compilator(new_CT_path, new, current)
            
            # Create the links to the lib folder
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(pjoin(targetPath,'Source','CutTools','includects',file), 
                                                        pjoin(targetPath,'lib'))
            # Make sure it is recompiled at least once. Because for centralized
            # MadGraph7 installations, it might be that compiler differs.
            # Not necessary anymore because I check the compiler version from
            # the log compiler_version.log generated during CT compilation
            # misc.compile(['cleanCT'], cwd = pjoin(targetPath,'Source'))
 
        elif self.dependencies=='external':
            if not os.path.exists(os.path.join(self.cuttools_dir,'includects','libcts.a')):
                logger.info('Compiling CutTools. This has to be done only once and'+\
                                  ' can take a couple of minutes.','$MG:BOLD')
                current = misc.detect_current_compiler(os.path.join(\
                                                  self.cuttools_dir,'makefile'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                          self.fortran_compiler
                if current != new:
                    misc.mod_compilator(self.cuttools_dir, new, current)
                misc.compile(cwd=self.cuttools_dir, job_specs = False)
                
                if not os.path.exists(os.path.join(self.cuttools_dir,
                                                      'includects','libcts.a')):            
                    raise MadGraph5Error("CutTools could not be correctly compiled.")
    
            # Create the links to the lib folder
            linkfiles = ['libcts.a', 'mpmodule.mod']
            for file in linkfiles:
                ln(os.path.join(self.cuttools_dir,'includects',file),
                                    os.path.join(targetPath,'lib'),abspath=True)

        elif self.dependencies=='environment_paths':
            # Here the user chose to define the dependencies path in one of 
            # his environmental paths
            CTlib = misc.which_lib('libcts.a')
            CTmod = misc.which_lib('mpmodule.mod')
            if CTlib is None:
                raise InvalidCmd("Could not find the location of the file"+\
                    " libcts.a in your environment paths.")
            elif CTmod is None:
                raise InvalidCmd("Could not find the location of the file"+\
                    " mpmodule.mod in your environment paths.")
            else:
                logger.info('MadGraph7 is using CutTools installation found at %s.'%\
                                                         os.path.dirname(CTlib))
                ln(os.path.join(CTlib),os.path.join(targetPath,'lib'),abspath=True)
                ln(os.path.join(CTmod),os.path.join(targetPath,'lib'),abspath=True)

    
    def get_aloha_model(self, model):
        """ Caches the aloha model created here as an attribute of the loop 
        exporter so that it can later be used in the LoopHelasMatrixElement
        in the function compute_all_analytic_information for recycling aloha 
        computations across different LoopHelasMatrixElements steered by the
        same loop exporter.
        """
        if not hasattr(self, 'aloha_model'):
            self.aloha_model = create_aloha.AbstractALOHAModel(model.get('modelpath'))

        missing_lor = []
        for lor in model.get('lorentz'):
            if not hasattr(self.aloha_model.model.lorentz, lor.name):
                missing_lor.append(lor)
        if missing_lor:
            logger.debug("adding in aloha model %s lorentz struct" % len(missing_lor))
            self.aloha_model.add_Lorentz_object(missing_lor)
        
        return self.aloha_model

    #===========================================================================
    # write the multiple-precision header files
    #===========================================================================
    def write_mp_files(self, writer_mprec, writer_mpc):
        """Write the cts_mprec.h and cts_mpc.h"""

        file = open(os.path.join(self.cuttools_dir, 'src/cts/cts_mprec.h')).read()
        writer_mprec.writelines(file)

        file = open(os.path.join(self.cuttools_dir, 'src/cts/cts_mpc.h')).read()
        file = file.replace('&','')
        writer_mpc.writelines(file)

        return True
        
#===============================================================================
# LoopProcessExporterFortranSA
#===============================================================================
class LoopProcessExporterFortranSA(LoopExporterFortran,
                                   export_v4.ProcessExporterFortranSA):
                                
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format."""
       
    template_dir=os.path.join(_file_path,'iolibs/template_files/loop')
    madloop_makefile_name = 'makefile'

    MadLoop_banner = MadLoopBannerStyles.get_MadLoop_Banner(
               style='classic2', color='green', 
               top_frame_char = '=', bottom_frame_char = '=',
               left_frame_char = '{',right_frame_char = '}',
               print_frame=True, side_margin = 7, up_margin = 1)

    def __init__(self, *args, **opts):
        super(LoopProcessExporterFortranSA,self).__init__(*args,**opts)
        self.unique_id=0 # to allow collier to distinguish the various loop subprocesses
        self.has_loop_induced = False
        self.density_prefixes = set() # processes for which GET_DENSITY is written
        
    def copy_template(self, model):
        """Additional actions needed to setup the Template.
        """
        super(LoopProcessExporterFortranSA, self).copy_template(model)

        self.loop_additional_template_setup()
        
    def finalize(self, matrix_element, cmdhistory, MG5options, outputflag):
        """create the global information for loops"""
        
        super(LoopProcessExporterFortranSA,self).finalize(matrix_element,
                                             cmdhistory, MG5options, outputflag)
        

        MLCard = banner_mod.MadLoopParam(pjoin(self.dir_path, 'Cards', 'MadLoopParams.dat'))
        if getattr(self, 'has_onia', False):
            self.set_onia_madloop_defaults(MLCard)
        elif not self.has_loop_induced:
            self.set_default_reduction_library(MLCard)
        # For loop-induced processes and *only* when summing over all helicity configurations
        # (which is the default for standalone usage), COLLIER is faster than Ninja.
        if self.has_loop_induced:
            MLCard['MLReductionLib'] = "7|6|1"
            # Computing the poles with COLLIER also unnecessarily slows down the code
            # It should only be set to True for checks and it's acceptable to remove them
            # here because for loop-induced processes they should be zero anyway.
            # We keep it active for non-loop induced processes because COLLIER is not the
            # main reduction tool in that case, and the poles wouldn't be zero then
            MLCard['COLLIERComputeUVpoles'] = False
            MLCard['COLLIERComputeIRpoles'] = False

        MLCard.write(pjoin(self.dir_path, 'Cards', 'MadLoopParams_default.dat'))
        MLCard.write(pjoin(self.dir_path, 'Cards', 'MadLoopParams.dat'))
            
    def set_onia_reduction_library(self, card):
        """With bound states the constituents carry equal momenta: the Gram
        determinants of every loop with both constituents vanish and the
        Coulomb-exchange loops have linearly dependent propagators. Only
        TIRust (9) reduces them reliably: the other libraries, IREGI
        included, can return wrong values that pass MadLoop's stability
        tests, and onia_loop.f reports their results unstable. TIRust is
        therefore the only library of the default chain."""
        if 'mlreductionlib' in card.user_set:
            return False
        if not getattr(self, 'tir_available_dict', {}).get('tirust', False):
            logger.warning('Processes with bound states need TIRust for the '
                           'reduction of their threshold integrals (install tirust): '
                           'the results of the other libraries are reported unstable.')
            return False
        card.set('MLReductionLib', '9', user=True)
        return True

    def set_onia_madloop_defaults(self, card):
        """MadLoopParams.dat defaults of processes with bound states:
        - the reduction libraries (set_onia_reduction_library);
        - no helicity filter, since which constituent helicity configurations
          contribute depends on the frame (onia_loop.f skips them point by
          point).
        Values the user set are kept. (The rounding of the counterterm
        interference, which MadLoop's stability test does not vary, is
        estimated in loop_matrix.f instead: ONIA_CTSQ.)"""
        self.set_onia_reduction_library(card)
        if 'helicityfilterlevel' not in card.user_set:
            card.set('HelicityFilterLevel', 0, user=True)

    def set_default_reduction_library(self, card):
        """Prefer a usable NinjaRS installation for untouched ordinary defaults."""
        if (getattr(self, 'has_loop_induced', False) or
                not getattr(self, 'tir_available_dict', {}).get('ninjars', False) or
                'mlreductionlib' in card.user_set or card['MLReductionLib'] != '6|7|1'):
            return False
        preferred = ninjars_install.default_reduction_library(
            True, getattr(self, 'ninjars_dir', None))
        if preferred != [8]:
            return False
        # An active value is required: Fortran ignores commented default lines.
        card.set('MLReductionLib', '%s|%s' % (preferred[0], card['MLReductionLib']),
                 user=True)
        return True

    def write_f2py_makefile(self, model):
        return
    
    def write_f2py_check_sa(self, matrix_element, output_path):
        """ Write the general check_sa.py in SubProcesses that calls all processes successively."""
 
        # No need to further edit this file for now.
        file = open(os.path.join(self.template_dir,\
                                       'check_sa_all.py.inc')).read()
        open(output_path,'w').writelines(file)
        # Make it executable
        os.chmod(output_path, os.stat(output_path).st_mode | stat.S_IEXEC)
       
    
    f2py_matrix_splitter_template = pjoin(os.pardir,"loop", "f2py_wrapper_subproccesses.f")
    all_matrix_template = pjoin(os.pardir, "loop", "all_matrix.f")

    def write_f2py_splitter(self):
        """write a function to call the correct matrix element"""

        template_matrix = open(os.path.join(self.template_dir,
                                            self.all_matrix_template)).read()
        template_f2py = open(os.path.join(self.template_dir,
                                         self.f2py_matrix_splitter_template)).read()

        allids = list(self.prefix_info.keys())

        allprefix = [self.prefix_info[key][0] for key in allids]
        allncomb = [self.prefix_info[key][2] for key in allids]
        alliden = [self.prefix_info[key][3] for key in allids] 
        min_nexternal = min([len(ids[0]) for ids in allids])
        max_nexternal = max([len(ids[0]) for ids in allids])

        info = []
        for (key,pid), (prefix, tag, ncomb, iden) in self.prefix_info.items():
            info.append('#PY %s : %s # %s %s' % (tag, key, prefix, pid))
            

        def dispatch(body):
            """if-chain selecting the process from (pdgs, procid);
            body(prefix) gives the Fortran lines run for that process"""
            text = []
            for n_ext in range(min_nexternal, max_nexternal+1):
                current_id = [ids[0] for ids in allids if len(ids[0])==n_ext]
                current_pid = [ids[1] for ids in allids if len(ids[0])==n_ext]
                if not current_id:
                    continue
                if min_nexternal != max_nexternal:
                    if n_ext == min_nexternal:
                        text.append('       if (npdg.eq.%i)then' % n_ext)
                    else:
                        text.append('       else if (npdg.eq.%i)then' % n_ext)
                for ii,pdgs in enumerate(current_id):
                    pid =  current_pid[ii]
                    condition = '.and.'.join(['%i.eq.pdgs(%i)' %(pdg, i+1) for i, pdg in enumerate(pdgs)])
                    if ii==0:
                        text.append( ' if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition, pid, len(pdgs)))
                    else:
                        text.append( ' else if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition,pid,len(pdgs)))
                    text.append(body(self.prefix_info[(pdgs,pid)][0]))
                text.append(' endif')
            #close the function
            if min_nexternal != max_nexternal:
                text.append('endif')
            return '\n'.join(text)

        # MadLoop only writes <prefix>GET_DENSITY (in compute_color_flows.f)
        # for a process computing its colour flows in an output made with
        # --density, i.e. never for the virtual of an NLO reweighting
        # (rw_mevirt). Referencing it anyway leaves the shared library with
        # an undefined symbol, and importing the f2py module fails.
        def density_call(prefix):
            if prefix in self.density_prefixes:
                return ' call %sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, ALPHAS, SCALE2, INTER)' % prefix
            return " write(*,*) 'ERROR: no density matrix for %s'\n stop 1" % prefix
        has_density = any(prefix in self.density_prefixes for prefix in allprefix)

        params = self.get_model_parameter(self.model)
        parameter_setup =[]
        for key, var in params.items():
            parameter_setup.append('        CASE ("%s")\n          %s = value\n        MP__%s = value' 
                                   % (key, var, var))
        
               # part for the resetting of the helicity
        helreset_def = []
        helreset_setup = []
        for prefix in set(allprefix):
            helreset_setup.append(' %shelreset = .true. ' % prefix)
            helreset_def.append(' logical %shelreset \n common /%shelreset/ %shelreset' % (prefix, prefix, prefix))
        

        f2py_prefix = ''
        if self.opt['output_options'] and 'prefixf2py' in self.opt['output_options']:
            f2py_prefix = 'f%s_' % self.opt['output_options']['prefixf2py']


        # Build IDENS entries ONCE per ME slot (must align 1-to-1 with get_pdg_order / allids).
        all_iden = ''
        for i, iden in enumerate(alliden, start=1):
            all_iden += ' idens(%s) = %s \n' % (i, iden)

        formatting = {'python_information':'\n'.join(info), 
                      # get_me takes the squared scale (UPDATE_AS_PARAM2 sets MU_R = sqrt(SCALE2))
                      'smatrixhel': dispatch(lambda prefix: ' call %sget_me(p, ALPHAS, SCALES2, NHEL, ANS, RETURNCODE)' % prefix),
                      'maxpart': max_nexternal,
                      'nb_me': len(allids),
                      'pdgs': ','.join([str(pdg[i]) if i<len(pdg) else '0' 
                                    for i in range(max_nexternal) \
                                    for (pdg,pid) in allids]),
                      'prefix':'\',\''.join(allprefix),
                      'parameter_setup': '\n'.join(parameter_setup),
                      'pids':  ','.join(str(pid) for (pdg,pid) in allids),
                      'helreset_def' : '\n'.join(helreset_def),
                      'helreset_setup' : '\n'.join(helreset_setup),
                      'f2py_prefix': f2py_prefix,
                      'idens_value': all_iden,
                      'density_splitter': dispatch(density_call),
                      }
    
        formatting['lenprefix'] = len(formatting['prefix'])
        # without any GET_DENSITY, PY_GET_DENSITY is left out of the module
        context = {'has_density': has_density}
        text = template_matrix % formatting
        fsock = writers.FortranWriter(pjoin(self.dir_path, 'SubProcesses', 'all_matrix.f'),'w')
        fsock.writelines(text, context=context)
        fsock.close()
        
        text = template_f2py % formatting
        fsock = writers.FortranWriter(pjoin(self.dir_path, 'SubProcesses', 'f2py_wrapper.f'),'w')
        fsock.writelines(text, context=context)
        fsock.close()
    
    def loop_additional_template_setup(self, copy_Source_makefile = True):
        """ Perform additional actions specific for this class when setting
        up the template with the copy_template function."""

        # We must change some files to their version for NLO computations
        cpfiles= ["Cards/MadLoopParams.dat",
                  "SubProcesses/MadLoopParamReader.f",
                  "SubProcesses/MadLoopParams.inc",
                  "SubProcesses/mg5_citation.f"]
        if copy_Source_makefile:
            cpfiles.append("Source/makefile")
        
        for file in cpfiles:
            shutil.copy(os.path.join(self.loop_dir,'StandAlone/', file),
                        os.path.join(self.dir_path, file))

        cp(pjoin(self.loop_dir,'StandAlone/Cards/MadLoopParams.dat'),
           pjoin(self.dir_path, 'Cards/MadLoopParams_default.dat'))

        ln(pjoin(self.dir_path, 'Cards','MadLoopParams.dat'), pjoin(self.dir_path,'SubProcesses'))

        # We might need to give a different name to the MadLoop makefile
        shutil.copy(pjoin(self.loop_dir,'StandAlone','SubProcesses','makefile'),
                pjoin(self.dir_path, 'SubProcesses',self.madloop_makefile_name))

        # Write SubProcesses/MadLoop_makefile_definitions with dummy variables
        # for the non-optimized output
        link_tir_libs=[]
        tir_libs=[]

        filePath = pjoin(self.dir_path, 'SubProcesses',
                                                 'MadLoop_makefile_definitions')
        calls = self.write_loop_makefile_definitions(
                        writers.MakefileWriter(filePath),link_tir_libs,tir_libs)

        # We need minimal editing of MadLoopCommons.f
        # For the optimized output, this file will be overwritten once the 
        # availability of COLLIER has been determined.
        MadLoopCommon = open(os.path.join(self.loop_dir,'StandAlone', 
                                    "SubProcesses","MadLoopCommons.inc")).read()
        writer = writers.FortranWriter(os.path.join(self.dir_path, 
                                             "SubProcesses","MadLoopCommons.f"))
        writer.writelines(MadLoopCommon%{
          'print_banner_commands':self.MadLoop_banner}, context={
                'collier_available':False})
        writer.close()
        
        # Copy the whole MadLoop5_resources directory (empty at this stage)
        ml_path = pjoin(self.dir_path,'SubProcesses', 'MadLoop5_resources')
        if not os.path.exists(ml_path):
            os.mkdir(ml_path)

        # Link relevant cards from Cards inside the MadLoop5_resources
        ln(pjoin(self.dir_path,'SubProcesses','MadLoopParams.dat'), ml_path)
        ln(pjoin(self.dir_path,'Cards','param_card.dat'), ml_path)
        ln(pjoin(self.dir_path,'Cards','ident_card.dat'), ml_path) 

        # And remove check_sa in the SubProcess folder since now there is a
        # check_sa tailored to each subprocess.
        if os.path.isfile(pjoin(self.dir_path,'SubProcesses','check_sa.f')):
            os.remove(pjoin(self.dir_path,'SubProcesses','check_sa.f'))

        cwd = os.getcwd()
        dirpath = os.path.join(self.dir_path, 'SubProcesses')
        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0
                     
        # Write the cts_mpc.h and cts_mprec.h files imported from CutTools
        self.write_mp_files(writers.FortranWriter('cts_mprec.h'),\
                                            writers.FortranWriter('cts_mpc.h'))

        # Return to original PWD
        os.chdir(cwd)

        # We must link the CutTools to the Library folder of the active Template
        super(LoopProcessExporterFortranSA, self).link_CutTools(self.dir_path)

    # This function is placed here and not in optimized exporterd,
    # because the same makefile.inc should be used in all cases.
    def write_loop_makefile_definitions(self, writer, link_tir_libs,
                                                       tir_libs,tir_include=[]):
        """ Create the file makefile which links to the TIR libraries."""
            
        file = open(os.path.join(self.loop_dir,'StandAlone',
                      'SubProcesses','MadLoop_makefile_definitions.inc')).read()  
        replace_dict={}
        replace_dict['link_tir_libs']=' '.join(link_tir_libs)
        replace_dict['tir_libs']=' '.join(tir_libs)
        tir_libs = tir_libs[:]
        tir_libs = [lib for lib in tir_libs if 'iregi' not in lib.lower()]
        dylibs =' '.join(tir_libs).replace('ninja.$(libext)', 'ninja.$(dylibext)') 
        dylibs = dylibs.replace('collier.$(libext)', 'collier.$(dylibext)')
        replace_dict['tir_dylibs'] = dylibs
        replace_dict['dotf']='%.f'
        replace_dict['prefix'] = self.SubProc_prefix
        replace_dict['doto'] = '%.o'
        replace_dict['tir_include'] = ' '.join(tir_include)
        replace_dict['rpaths_libs'] = ''
        for lib in tir_libs:
            if lib.endswith(('libninjars.a', 'libtirust.a')):
                continue
            replace_dict['rpaths_libs'] += '-Wl,-rpath,%s '%os.path.dirname(lib)    

        file = file % replace_dict
        if writer:
            writer.writelines(file)
        else:
            return file
        
    def convert_model(self, model, wanted_lorentz = [], 
                                             wanted_couplings = [], **opts):
        """ Caches the aloha model created here when writing out the aloha 
        fortran subroutine.
        """
        self.get_aloha_model(model)
        super(LoopProcessExporterFortranSA, self).convert_model(model,
           wanted_lorentz = wanted_lorentz, wanted_couplings = wanted_couplings)

    def write_aloha_library(self, model, wanted_lorentz, npwave=0, gauge='',
                            loop_dual=False, loop_npwave=1):
        """The HELAS library of the loop output: with a P-wave bound state
        its double-precision routines are dual (export_v4), for all the
        processes of the output, which must then all have one."""
        pwave = getattr(self, 'onia_pwave_processes', [])
        if any(pwave):
            if not all(pwave):
                raise loop_onia.OniaLoopError(
                    'A process with a P-wave bound state must be output alone (or '
                    'with other P-wave processes): its HELAS library is in dual numbers.')
            loop_dual = True
            # the dual numbers of the library have one direction per P-wave
            # state: all the processes of the output must have the same number
            counts = getattr(self, 'onia_npwave', set()) or {1}
            if len(counts) > 1:
                raise loop_onia.OniaLoopError(
                    'Processes with one and with two P-wave bound states must be '
                    'output separately: their HELAS libraries differ.')
            return super(LoopProcessExporterFortranSA, self).write_aloha_library(
                model, wanted_lorentz, npwave=npwave, gauge=gauge, loop_dual=loop_dual,
                loop_npwave=max(counts), loop_ep=bool(getattr(self, 'onia_ep_dirs', {})))
        return super(LoopProcessExporterFortranSA, self).write_aloha_library(
            model, wanted_lorentz, npwave=npwave, gauge=gauge, loop_dual=loop_dual)

    def get_ME_identifier(self, matrix_element, 
                                 group_number = None, group_elem_number = None):
        """ A function returning a string uniquely identifying the matrix 
        element given in argument so that it can be used as a prefix to all
        MadLoop5 subroutines and common blocks related to it. This allows
        to compile several processes into one library as requested by the 
        BLHA (Binoth LesHouches Accord) guidelines.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent."""

        # When disabling the loop grouping in the LoopInduced MadEvent output,
        # we have only the group_number set and the proc_id set to None. In this
        # case we don't print the proc_id.
        if (not group_number is None) and group_elem_number is None:
            return 'ML5_%d_%s_'%(matrix_element.get('processes')[0].get('id'),
                                                                   group_number)            
        elif group_number is None or group_elem_number is None:
            return 'ML5_%d_'%matrix_element.get('processes')[0].get('id') 
        else:
            return 'ML5_%d_%s_%s_'%(matrix_element.get('processes')[0].get('id'),
                                                group_number, group_elem_number)

    def get_SubProc_folder_name(self, process, 
                                 group_number = None, group_elem_number = None):
        """Returns the name of the SubProcess directory, which can contain
        the process group and group element number for the case of loop-induced
        integration with MadEvent."""
        
        # When disabling the loop grouping in the LoopInduced MadEvent output,
        # we have only the group_number set and the proc_id set to None. In this
        # case we don't print the proc_id.
        if not group_number is None and group_elem_number is None:
            return "%s%d_%s_%s"%(self.SubProc_prefix, process.get('id'), 
                              group_number,process.shell_string(print_id=False))
        elif group_number is None or group_elem_number is None:
            return "%s%s" %(self.SubProc_prefix,process.shell_string())
        else:
            return "%s%d_%s_%s_%s"%(self.SubProc_prefix, process.get('id'), 
           group_number, group_elem_number,process.shell_string(print_id=False))

    #===========================================================================
    # Set the compiler to be gfortran for the loop processes.
    #===========================================================================
    def compiler_choice(self, compiler=export_v4.default_compiler):
        """ Different daughter classes might want different compilers.
        Here, the gfortran compiler is used throughout the compilation 
        (mandatory for CutTools written in f90) """
        if isinstance(compiler, str):
            fortran_compiler = compiler
            compiler = export_v4.default_compiler
            compiler['fortran'] = fortran_compiler
        
        if not compiler['fortran'] is None and not \
                       any([name in compiler['fortran'] for name in \
                                                         ['gfortran','ifort']]):
            logger.info('For loop processes, the compiler must be fortran90'+\
                        'compatible, like gfortran.')
            compiler['fortran'] = 'gfortran'
            self.set_compiler(compiler,True)
        else:
            self.set_compiler(compiler)
        
        self.set_cpp_compiler(compiler['cpp'])
    
    def turn_to_mp_calls(self, helas_calls_list):
        # Prepend 'MP_' to all the helas calls in helas_calls_list.
        # Might look like a brutal unsafe implementation, but it is not as 
        # these calls are built from the properties of the HELAS objects and
        # whether they are evaluated in double or quad precision is none of 
        # their business but only relevant to the output algorithm.
        # Also the cast to complex masses DCMPLX(*) must be replaced by
        # CMPLX(*,KIND=16)
        MP=re.compile(r"(?P<toSub>^.*CALL\s+)",re.IGNORECASE | re.MULTILINE)
        
        def replaceWith(match_obj):
            return match_obj.group('toSub')+'MP_'

        DCMPLX=re.compile(r"DCMPLX\((?P<toSub>([^\)]*))\)",\
                                                   re.IGNORECASE | re.MULTILINE)
        
        for i, helas_call in enumerate(helas_calls_list):
            new_helas_call=MP.sub(replaceWith,helas_call)
            helas_calls_list[i]=DCMPLX.sub(r"CMPLX(\g<toSub>,KIND=16)",\
                                                                 new_helas_call)

    def make_source_links(self):
        """ In the loop output, we don't need the files from the Source folder """
        pass

    def make_model_symbolic_link(self):
        """ Add the linking of the additional model files for multiple precision
        """
        super(LoopProcessExporterFortranSA, self).make_model_symbolic_link()
        model_path = self.dir_path + '/Source/MODEL/'
        ln(model_path + '/mp_coupl.inc', self.dir_path + '/SubProcesses')
        ln(model_path + '/mp_coupl_same_name.inc', self.dir_path + '/SubProcesses')
    
    def make(self):
        """ Compiles the additional dependences for loop (such as CutTools)."""
        super(LoopProcessExporterFortranSA, self).make()
        
        # make CutTools (only necessary with MG option output_dependencies='internal')
        libdir = os.path.join(self.dir_path,'lib')
        sourcedir = os.path.join(self.dir_path,'Source')
        if self.dependencies=='internal':
            if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
                if os.path.exists(pjoin(sourcedir,'CutTools')):
                    logger.info('Compiling CutTools (can take a couple of minutes) ...')
                    misc.compile(['CutTools','-j1'], cwd = sourcedir, nb_core=1)
                    logger.info('          ...done.')
                else:
                    raise MadGraph5Error('Could not compile CutTools because its'+\
                   ' source directory could not be found in the SOURCE folder.')
        if not os.path.exists(os.path.realpath(pjoin(libdir, 'libcts.a'))) or \
            not os.path.exists(os.path.realpath(pjoin(libdir, 'mpmodule.mod'))):
            raise MadGraph5Error('CutTools compilation failed.')
        
        # Verify compatibility between current compiler and the one which was
        # used when last compiling CutTools (if specified).
        compiler_log_path = pjoin(os.path.dirname((os.path.realpath(pjoin(
                                  libdir, 'libcts.a')))),'compiler_version.log')
        if os.path.exists(compiler_log_path):
            compiler_version_used = open(compiler_log_path,'r').read()
            if not str(misc.get_gfortran_version(misc.detect_current_compiler(\
                       pjoin(sourcedir,'make_opts')))) in compiler_version_used:
                if os.path.exists(pjoin(sourcedir,'CutTools')):
                    logger.info('CutTools was compiled with a different fortran'+\
                                            ' compiler. Re-compiling it now...')
                    misc.compile(['cleanCT'], cwd = sourcedir)
                    misc.compile(['CutTools','-j1'], cwd = sourcedir, nb_core=1)
                    logger.info('          ...done.')
                else:
                    raise MadGraph5Error("CutTools installation in %s"\
                                 %os.path.realpath(pjoin(libdir, 'libcts.a'))+\
                 " seems to have been compiled with a different compiler than"+\
                    " the one specified in MadGraph7. Please recompile it by typing"+\
                    " 'install cuttools_iregi' in the MadGraph7 interface.")
    
    def cat_coeff(self, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Concatenate the coefficient information to reduce it to 
        (fraction, is_imaginary) """

        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

        return (total_coeff, is_imaginary)       

    def get_amp_to_jamp_map(self, col_amps, n_amps):
        """ Returns a list with element 'i' being a list of tuples corresponding
        to all apparition of amplitude number 'i' in the jamp number 'j'
        with coeff 'coeff_j'. The format of each tuple describing an apparition 
        is (j, coeff_j). where coeff_j is of the form (Fraction, is_imag)."""

        if(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error("Incorrect col_amps argument passed to get_amp_to_jamp_map")
        else:
            raise MadGraph5Error("Incorrect col_amps argument passed to get_amp_to_jamp_map")
        
        # To store the result
        res_list = [[] for i in range(n_amps)]
        for i, coeff_list in enumerate(color_amplitudes):
                for (coefficient, amp_number) in coeff_list:
                    res_list[amp_number-1].append((i,self.cat_coeff(\
                      coefficient[0],coefficient[1],coefficient[2],coefficient[3])))

        return res_list

    def get_color_matrix(self, matrix_element):
        """Return the color matrix definition lines. This color matrix is of size
        NLOOPAMPSxNBORNAMPS and allows for squaring individually each Loop and Born
        amplitude."""

        logger.info('Computing diagram color coefficients')

        # The two lists have a list of tuples at element 'i' which correspond
        # to all apparitions of loop amplitude number 'i' in the jampl number 'j'
        # with coeff 'coeffj'. The format of each tuple describing an apparition 
        # is (j, coeffj).
        ampl_to_jampl=self.get_amp_to_jamp_map(\
          matrix_element.get_loop_color_amplitudes(),
          matrix_element.get_number_of_loop_amplitudes())
        if matrix_element.get('processes')[0].get('has_born'):
            ampb_to_jampb=self.get_amp_to_jamp_map(\
          matrix_element.get_born_color_amplitudes(),
          matrix_element.get_number_of_born_amplitudes())
        else:
            ampb_to_jampb=ampl_to_jampl
        # Below is the original color matrix multiplying the JAMPS
        if matrix_element.get('color_matrix'):
            ColorMatrixDenom = \
              matrix_element.get('color_matrix').get_line_denominators()
            ColorMatrixNum = [ matrix_element.get('color_matrix').\
                               get_line_numerators(index, denominator) for
                               (index, denominator) in enumerate(ColorMatrixDenom) ]
        else:
            ColorMatrixDenom= [1]
            ColorMatrixNum = [[1]]
            
        # Below is the final color matrix output
        ColorMatrixNumOutput=[]
        ColorMatrixDenomOutput=[]
        
        # Now we construct the color factors between each born and loop amplitude
        # by scanning their contributions to the different jamps.
        start = time.time()
        progress_bar = None
        time_info = False
        for i, jampl_list in enumerate(ampl_to_jampl):
            # This can be pretty long for processes with many color flows.
            # So, if necessary (i.e. for more than 15s), we tell the user the
            # estimated time for the processing.
            if i==5:
                elapsed_time = time.time()-start
                t = len(ampl_to_jampl)*(elapsed_time/5.0)
                if t > 10.0:
                    time_info = True
                    logger.info('The color factors computation will take '+\
                      ' about %s to run. '%str(datetime.timedelta(seconds=int(t)))+\
                      'Started on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))
                    if logger.getEffectiveLevel()<logging.WARNING:
                        widgets = ['Color computation:', pbar.Percentage(), ' ', 
                                                pbar.Bar(),' ', pbar.ETA(), ' ']
                        progress_bar = pbar.ProgressBar(widgets=widgets, 
                                       maxval=len(ampl_to_jampl), fd=sys.stdout)
            
            if not progress_bar is None:
                progress_bar.update(i+1)
                # Flush to force the printout of the progress_bar to be updated
                sys.stdout.flush()

            line_num=[]
            line_denom=[]

            # Treat the special case where this specific amplitude contributes to no
            # color flow at all. So it is zero because of color but not even due to
            # an accidental cancellation among color flows, but simply because of its
            # projection to each individual color flow is zero. In such case, the 
            # corresponding jampl_list is empty and all color coefficients must then
            # be zero. This happens for example in the Higgs Effective Theory model
            # for the bubble made of a 4-gluon vertex and the effective ggH vertex.
            if len(jampl_list)==0:
                line_num=[0]*len(ampb_to_jampb)
                line_denom=[1]*len(ampb_to_jampb)
                ColorMatrixNumOutput.append(line_num)
                ColorMatrixDenomOutput.append(line_denom)
                continue

            for jampb_list in ampb_to_jampb:
                real_num=0
                imag_num=0
                common_denom=color_amp.ColorMatrix.lcmm(*[abs(ColorMatrixDenom[jampl]*
                    ampl_coeff[0].denominator*ampb_coeff[0].denominator) for 
                    ((jampl, ampl_coeff),(jampb,ampb_coeff)) in 
                    itertools.product(jampl_list,jampb_list)])
                for ((jampl, ampl_coeff),(jampb, ampb_coeff)) in \
                                       itertools.product(jampl_list,jampb_list):
                    # take the numerator and multiply by lcm/denominator
                    # as we will later divide by the lcm.
                    buff_num=ampl_coeff[0].numerator*\
                        ampb_coeff[0].numerator*ColorMatrixNum[jampl][jampb]*\
                        abs(common_denom)/(ampl_coeff[0].denominator*\
                        ampb_coeff[0].denominator*ColorMatrixDenom[jampl])
                    # Remember that we must take the complex conjugate of
                    # the born jamp color coefficient because we will compute
                    # the square with 2 Re(LoopAmp x BornAmp*)
                    if ampl_coeff[1] and ampb_coeff[1]:
                        real_num=real_num+buff_num
                    elif not ampl_coeff[1] and not ampb_coeff[1]:
                        real_num=real_num+buff_num
                    elif not ampl_coeff[1] and ampb_coeff[1]:
                        imag_num=imag_num-buff_num
                    else:
                        imag_num=imag_num+buff_num
                assert not (real_num!=0 and imag_num!=0), "MadGraph7 found a "+\
                  "color matrix element which has both a real and imaginary part."
                if imag_num!=0:
                    assert int(imag_num) == imag_num and int(common_denom) == common_denom
                    res=fractions.Fraction(int(imag_num),int(common_denom))
                    line_num.append(res.numerator)
                    # Negative denominator means imaginary color coef of the
                    # final color matrix
                    line_denom.append(res.denominator*-1)
                else:
                    assert int(real_num) == real_num and int(common_denom) == common_denom
                    res=fractions.Fraction(int(real_num),int(common_denom))
                    line_num.append(res.numerator)
                    # Positive denominator means real color coef of the final color matrix
                    line_denom.append(res.denominator)

            ColorMatrixNumOutput.append(line_num)
            ColorMatrixDenomOutput.append(line_denom)

        if time_info:
            logger.info('Finished on %s.'%datetime.datetime.now().strftime(\
                                                              "%d-%m-%Y %H:%M"))            
        if progress_bar!=None:
            progress_bar.finish()

        return (ColorMatrixNumOutput,ColorMatrixDenomOutput)

    def get_context(self,matrix_element):
        """ Returns the contextual variables which need to be set when
        pre-processing the template files."""

        # The nSquaredSO entry of the general replace dictionary should have
        # been set in write_loopmatrix prior to the first call to this function
        # However, for cases where the TIRCaching contextual variable is 
        # irrelevant (like in the default output), this might not be the case
        # so we set it to 1. 
        try:
            n_squared_split_orders = matrix_element.rep_dict['nSquaredSO']
        except (KeyError, AttributeError):
            n_squared_split_orders = 1

        LoopInduced = not matrix_element.get('processes')[0].get('has_born')
        self.has_loop_induced = max(LoopInduced, self.has_loop_induced)
        # Force the computation of loop color flows for loop_induced processes
        ComputeColorFlows = self.compute_color_flows or LoopInduced
        # The variable AmplitudeReduction is just to make the contextual
        # conditions more readable in the include files.
        AmplitudeReduction = LoopInduced or ComputeColorFlows
        # Even when not reducing at the amplitude level, the TIR caching
        # is useful when there is more than one squared split order config.
        TIRCaching = AmplitudeReduction or n_squared_split_orders>1
        UseDensity = 'density' in self.cmd_options #detects if we want to compute the density matrix
        MadEventOutput = False
        return {'LoopInduced': LoopInduced,
                'ComputeColorFlows': ComputeColorFlows,
                'AmplitudeReduction': AmplitudeReduction,
                'TIRCaching': TIRCaching,
                'MadEventOutput': MadEventOutput,
                'UseDensity': UseDensity,
                'Onia': loop_onia.has_onia(matrix_element),
                'OniaPwave': loop_onia.has_pwave(matrix_element),
                'OniaPwave2': loop_onia.npwave(matrix_element) == 2,
                'OniaInitial': loop_onia.has_initial_onia(matrix_element),
                'OniaPwaveEP': False}


    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_loop_subprocess(self, matrix_element, fortran_model,
          group_number = None, proc_id = None, config_map=None, unique_id=None):
        """Generate the Pxxxxx directory for a loop subprocess in MG4 standalone,
        including the necessary loop_matrix.f, born_matrix.f and include files.
        Notice that this is too different from generate_subprocess_directory
        so that there is no point reusing this mother function.
        The 'group_number' and 'proc_id' options are only used for the LoopInduced
        MadEvent output and only to specify the ME_identifier and the P* 
        SubProcess directory name."""

        cwd = os.getcwd()
        proc_dir_name = self.get_SubProc_folder_name(
                        matrix_element.get('processes')[0],group_number,proc_id)
        dirpath = os.path.join(self.dir_path, 'SubProcesses', proc_dir_name)
        
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        try:
            os.chdir(dirpath)
        except os.error:
            logger.error('Could not cd to directory %s' % dirpath)
            return 0

        logger.info('Creating files in directory %s' % dirpath)

        if unique_id is None:
            raise MadGraph5Error('A unique id must be provided to the function'+\
                     'generate_loop_subprocess of LoopProcessExporterFortranSA.')
        # Create an include with the unique consecutive ID assigned
        open('unique_id.inc','w').write(
"""      integer UNIQUE_ID
      parameter(UNIQUE_ID=%d)"""%unique_id)
        
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        calls=self.write_loop_matrix_element_v4(None,matrix_element,
                         fortran_model, group_number = group_number, 
                                     proc_id = proc_id, config_map = config_map)

        # We assume here that all processes must share the same property of 
        # having a born or not, which must be true anyway since these are two
        # definite different classes of processes which can never be treated on
        # the same footing.
        if matrix_element.get('processes')[0].get('has_born'):
            filename = 'born_matrix.f'
            calls = self.write_bornmatrix(
                writers.FortranWriter(filename),
                matrix_element,
                fortran_model)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        # Do not draw the loop diagrams if they are too many.
        # The user can always decide to do it manually, if really needed
        loop_diags = [loop_diag for loop_diag in\
             matrix_element.get('base_amplitude').get('loop_diagrams')\
             if isinstance(loop_diag,LoopDiagram) and loop_diag.get('type') > 0]
        if len(loop_diags)>5000:
            logger.info("There are more than 5000 loop diagrams."+\
                                              "Only the first 5000 are drawn.")
        filename = "loop_matrix.ps"
        plot = draw.MultiEpsDiagramDrawer(base_objects.DiagramList(
            loop_diags[:5000]),filename,
            model=matrix_element.get('processes')[0].get('model'),amplitude='')
        logger.info("Drawing loop Feynman diagrams for " + \
                     matrix_element.get('processes')[0].nice_string())
        plot.draw()

        if matrix_element.get('processes')[0].get('has_born'):   
            filename = "born_matrix.ps"
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('born_diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude='')
            logger.info("Generating born Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string(\
                                                          print_weighted=False))
            plot.draw()

        self.link_files_from_Subprocesses(self.get_SubProc_folder_name(
                       matrix_element.get('processes')[0],group_number,proc_id))
        
        # Return to original PWD
        os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    def link_files_from_Subprocesses(self,proc_name):
        """ To link required files from the Subprocesses directory to the
        different P* ones"""
        
        linkfiles = ['coupl.inc',
                     'cts_mprec.h', 'cts_mpc.h', 'mp_coupl.inc', 
                     'mp_coupl_same_name.inc',
                     'MadLoopParamReader.f','MadLoopCommons.f',
                     'MadLoopParams.inc','global_specs.inc',
                     'mg5_citation.f']
        
        for file in linkfiles:
            ln('../%s' % file)
        
        ln('../%s'%self.madloop_makefile_name, name='makefile')

        # The mp module
        ln('../../lib/mpmodule.mod')
            
        # Also like the whole MadLoop5_files directory
        ln('../MadLoop5_resources')

    def generate_general_replace_dict(self,matrix_element,
                                           group_number = None, proc_id = None):
        """Generates the entries for the general replacement dictionary used
        for the different output codes for this exporter.The arguments 
        group_number and proc_id are just for the LoopInduced output with MadEvent."""

        # Helper
        def compute_iden_from_pdgs(ids, ninitial, model):
            """
            Helper function to compute denominator factor
            """
            def nhel_from_particle(p):
                spin = int(p.get('spin'))
                # for massless vectors use 2 helicities not 3
                mass = p.get('mass')
                if spin == 3 and (mass == 'ZERO' or str(mass).upper() == 'ZERO'):
                    return 2
                return spin

            def color_dim_from_particle(p):
                # In UFO, color is typically 1, 3, -3, 8, ...
                return abs(int(p.get('color')))

            incoming = ids[:ninitial]
            iden = 1
            for pid in incoming:
                p = model.get_particle(pid)
                iden *= nhel_from_particle(p) * color_dim_from_particle(p)
            return int(iden)
        
        dict={}
        # A general process prefix which appears in front of all MadLooop
        # subroutines and common block so that several processes can be compiled
        # together into one library, as necessary to follow BLHA guidelines.
        
        dict['proc_prefix'] = self.get_ME_identifier(matrix_element,
                       group_number = group_number, group_elem_number = proc_id)

        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        if 'prefix' in self.cmd_options and self.cmd_options['prefix'] in ['int','proc']:
            ncomb = matrix_element.get_helicity_combinations()
            for proc in matrix_element.get('processes'):
                ids = [l.get('id') for l in proc.get('legs_with_decays')]
                iden = compute_iden_from_pdgs(ids, ninitial, self.model)
                self.prefix_info[tuple(ids),proc.get('id')] = [dict['proc_prefix'], proc.get_tag(), ncomb, iden]

        # The proc_id is used for MadEvent grouping, so none of our concern here
        # and it is simply set to an empty string.        
        dict['proc_id'] = ''
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        dict['info_lines'] = info_lines
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        dict['process_lines'] = process_lines
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        dict['nexternal'] = nexternal
        dict['nincoming'] = ninitial
        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        dict['ncomb'] = ncomb
        # Extract nloopamps
        nloopamps = matrix_element.get_number_of_loop_amplitudes()
        dict['nloopamps'] = nloopamps
        # Extract nloopdiags
        nloopdiags = len(matrix_element.get('diagrams'))
        dict['nloopdiags'] = nloopdiags
        # Extract nctamps
        nctamps = matrix_element.get_number_of_CT_amplitudes()
        dict['nctamps'] = nctamps
        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_external_wavefunctions()
        dict['nwavefuncs'] = nwavefuncs
        # Set format of the double precision
        dict['real_dp_format']='real*8'
        dict['real_mp_format']='real*16'
        # Set format of the complex
        dict['complex_dp_format']='complex*16'
        dict['complex_mp_format']='complex*32'
        # Set format of the masses
        dict['mass_dp_format'] = dict['complex_dp_format']
        dict['mass_mp_format'] = dict['complex_mp_format']
        # Fill in default values for the placeholders for the madevent 
        # loop-induced output
        dict['nmultichannels'] = 0
        dict['nmultichannel_configs'] = 0
        dict['config_map_definition'] = ''
        dict['config_index_map_definition'] = ''        
        # Color matrix size
        # For loop induced processes it is NLOOPAMPSxNLOOPAMPS and otherwise
        # it is NLOOPAMPSxNBORNAMPS
        # Also, how to access the number of Born squared order contributions

        if matrix_element.get('processes')[0].get('has_born'):
            dict['color_matrix_size'] = 'nbornamps'
            dict['get_nsqso_born']=\
                           "include 'nsqso_born.inc'"
        else:
            dict['get_nsqso_born']="""INTEGER NSQSO_BORN
            PARAMETER (NSQSO_BORN=0)
            """
            dict['color_matrix_size'] = 'nloopamps'    
        
        # These placeholders help to have as many common templates for the
        # output of the loop induced processes and those with a born 
        # contribution.
        if matrix_element.get('processes')[0].get('has_born'):
            # Extract nbornamps
            nbornamps = matrix_element.get_number_of_born_amplitudes()
            dict['nbornamps'] = nbornamps
            dict['ncomb_helas_objs'] = ',ncomb'
            dict['nbornamps_decl'] = \
              """INTEGER NBORNAMPS
                 PARAMETER (NBORNAMPS=%d)"""%nbornamps
            dict['nBornAmps'] = nbornamps
                 
        else:
            dict['ncomb_helas_objs'] = ''  
            dict['dp_born_amps_decl'] = ''
            dict['dp_born_amps_decl_in_mp'] = ''
            dict['copy_mp_to_dp_born_amps'] = ''
            dict['mp_born_amps_decl'] = ''
            dict['nbornamps_decl'] = ''
            dict['nbornamps'] = 0
            dict['nBornAmps'] = 0
        
        return dict
    
    def write_loop_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        group_number = None, proc_id = None, config_map = None):
        """ Writes loop_matrix.f, CT_interface.f, loop_num.f and
        mp_born_amps_and_wfs.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent and only used in get_ME_identifier.
        """
        
        # Create the necessary files for the loop matrix element subroutine
        
        if config_map:
            raise MadGraph5Error('The default loop output cannot be used with'+\
              'MadEvent and cannot compute the AMP2 for multi-channeling.')

        if not isinstance(fortran_model,\
          helas_call_writers.FortranUFOHelasCallWriter):
            raise MadGraph5Error('The loop fortran output can only'+\
              ' work with a UFO Fortran model')
        
        LoopFortranModel = helas_call_writers.FortranUFOHelasCallWriter(
                     argument=fortran_model.get('model'),
                     hel_sum=matrix_element.get('processes')[0].get('has_born'))

        # Compute the analytical information of the loop wavefunctions in the
        # loop helas matrix elements using the cached aloha model to reuse
        # as much as possible the aloha computations already performed for
        # writing out the aloha fortran subroutines.
        matrix_element.compute_all_analytic_information(
          self.get_aloha_model(matrix_element.get('processes')[0].get('model')))

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        matrix_element.rep_dict = self.generate_general_replace_dict(
                 matrix_element, group_number = group_number, proc_id = proc_id)                                 
        
        # Extract max number of loop couplings (specific to this output type)
        matrix_element.rep_dict['maxlcouplings']= \
                                         matrix_element.find_max_loop_coupling()
        # The born amp declaration suited for also outputting the loop-induced
        # processes as well.
        if matrix_element.get('processes')[0].get('has_born'):
            matrix_element.rep_dict['dp_born_amps_decl_in_mp'] = \
                  matrix_element.rep_dict['complex_dp_format']+" DPAMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sAMPS/DPAMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['dp_born_amps_decl'] = \
                  matrix_element.rep_dict['complex_dp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sAMPS/AMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['mp_born_amps_decl'] = \
                  matrix_element.rep_dict['complex_mp_format']+" AMP(NBORNAMPS,NCOMB)"+\
                  "\n common/%sMP_AMPS/AMP"%matrix_element.rep_dict['proc_prefix']
            matrix_element.rep_dict['copy_mp_to_dp_born_amps'] = \
                   '\n'.join(['DO I=1,NBORNAMPS','DPAMP(I,H)=AMP(I,H)','ENDDO'])
        
        if writer:
            raise MadGraph5Error('Matrix output mode no longer supported.')
        
        filename = 'loop_matrix.f'
        calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                      matrix_element,
                                      LoopFortranModel)       

        # Write out the proc_prefix in a file, this is quite handy
        proc_prefix_writer = writers.FortranWriter('proc_prefix.txt','w')
        proc_prefix_writer.write(matrix_element.rep_dict['proc_prefix'])
        proc_prefix_writer.close()
                    
        filename = 'check_sa.f'
        self.write_check_sa(writers.FortranWriter(filename),matrix_element)
        
        filename = 'CT_interface.f'
        self.write_CT_interface(writers.FortranWriter(filename),\
                                matrix_element)
        
        
        
        filename = 'improve_ps.f'
        calls = self.write_improve_ps(writers.FortranWriter(filename),
                                                             matrix_element)
        
        filename = 'loop_num.f'
        self.write_loop_num(writers.FortranWriter(filename),\
                                matrix_element,LoopFortranModel)
        
        filename = 'mp_born_amps_and_wfs.f'
        self.write_born_amps_and_wfs(writers.FortranWriter(filename),\
                                     matrix_element,LoopFortranModel)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                                                            nexternal, ninitial)

        filename = 'process_info.inc'        
        self.write_process_info_file(writers.FortranWriter(filename),
                                                                 matrix_element)
        return calls

    def write_process_info_file(self, writer, matrix_element):
        """A small structural function to write the include file specifying some
        process characteristics."""

        model = matrix_element.get('processes')[0].get('model')
        process_info = {}
        # The maximum spin of any particle connected (or directly running in) 
        # any loop of this matrix element. This is important because there is
        # some limitation in the stability tests that can be performed when this
        # maximum spin is above 3 (vectors). Also CutTools has limitations in 
        # this regard.
        process_info['max_spin_connected_to_loop']=\
                                 matrix_element.get_max_spin_connected_to_loop()
        
        process_info['max_spin_external_particle']= max(
                model.get_particle(l.get('id')).get('spin') for l in 
                                 matrix_element.get('processes')[0].get('legs'))

        proc_include = \
"""
INTEGER MAX_SPIN_CONNECTED_TO_LOOP
PARAMETER(MAX_SPIN_CONNECTED_TO_LOOP=%(max_spin_connected_to_loop)d)
INTEGER MAX_SPIN_EXTERNAL_PARTICLE
PARAMETER(MAX_SPIN_EXTERNAL_PARTICLE=%(max_spin_external_particle)d)
"""%process_info

        writer.writelines(proc_include)
                                
    def generate_subprocess_directory(self, matrix_element, fortran_model,
                                      me_number=None, second_exporter=None):
        """ To overload the default name for this function such that the correct
        function is used when called from the command interface """
        # 3rd positional slot is the subprocess number (base-class convention);
        # loop_interface.ML5export omits it, madgraph_interface.export passes it.
        assert second_exporter is None
        self.unique_id +=1
        return self.generate_loop_subprocess(matrix_element,fortran_model,
                                                            unique_id=self.unique_id)

    def write_check_sa(self, writer, matrix_element):
        """Writes out the steering code check_sa. In the optimized output mode,
        All the necessary entries in the replace_dictionary have already been 
        set in write_loopmatrix because it is only there that one has access to
        the information about split orders."""        
        replace_dict = copy.copy(matrix_element.rep_dict)     
        for key in ['print_so_born_results','print_so_loop_results',
            'write_so_born_results','write_so_loop_results','set_coupling_target']:
            if key not in list(replace_dict.keys()):
                replace_dict[key]=''
        # With bound states the driver sees the physical legs (one per bound
        # state, of mass m1+m2: pmass.inc)
        onia = loop_onia.has_onia(matrix_element)
        if onia:
            replace_dict['nexternal'] = matrix_element.get_physical_nexternal_ninitial()[0]
        
        #initialisation of the parameters for density matrix (dont think it is needed)
        replace_dict['use_density'] = '.false.'
        replace_dict['dens_nchanging'] = 1
        replace_dict['dens_ncomb'] = 2
        replace_dict['dens_pos'] = '\n'
        replace_dict['dens_allow_hel'] = '\n'

        if 'density' in self.cmd_options:
            import math
            replace_dict['use_density'] = '.true.'
            changing = [int(i) for i in self.cmd_options['density'].split(',')]
            replace_dict['dens_nchanging'] = len(changing)
            replace_dict['dens_pos'] = '\n        '.join(
                   ['POS(%s) = %i' % (i+1, pos) for i,pos in enumerate(changing)])
            get_helicity_per_particle = matrix_element.get_helicity_per_particle()
            changing_hels = [get_helicity_per_particle[pos-1] for pos in changing]
            replace_dict['dens_ncomb'] = math.prod([len(hel) for hel in changing_hels])

            i = 0
            replace_dict['dens_allow_hel'] = ''
            for comb in  itertools.product(*changing_hels):
                for h in comb:
                    i += 1
                    replace_dict['dens_allow_hel'] += ' ALLOW_HEL(%i) = %i\n       ' % (i, h)

        if matrix_element.get('processes')[0].get('has_born'):
            file = open(os.path.join(self.template_dir,'check_sa.inc')).read()
        else:
            file = open(os.path.join(self.template_dir,\
                                          'check_sa_loop_induced.inc')).read()
            if self.opt['vector_size']:
                replace_dict["include_vector"] = "include '../../Source/vector.inc'"
            else:
               replace_dict["include_vector"] = '' 
        file=file%replace_dict
        #we add the context to distinguish density mode from regular mode
        writer.writelines(file, context=self.get_context(matrix_element))
         
        # We can always write the f2py wrapper if present (in loop optimized mode, it is)
        if not os.path.isfile(pjoin(self.template_dir,'check_py.f.inc')):
            return
        if onia:
            # The f2py wrapper passes arrays of the constituent legs
            logger.info('No python (f2py) wrapper for processes with bound states.')
            return
        
        file = open(os.path.join(self.template_dir,\
                                      'check_py.f.inc')).read()

        if 'prefix' in self.cmd_options and self.cmd_options['prefix'] in ['int','proc']:
            replace_dict['prefix_routine'] = replace_dict['proc_prefix']
        else:
            replace_dict['prefix_routine'] = ''
        file=file%replace_dict        
        new_path = writer.name.replace('check_sa.f', 'f2py_wrapper.f')
        new_writer = writer.__class__(new_path, 'w')
        new_writer.writelines(file)

        file = open(os.path.join(self.template_dir,\
                                      'check_sa.py.inc')).read()
        # For now just put in an empty PS point but in the future, maybe generate
        # a valid one already here by default
        curr_proc = matrix_element.get('processes')[0]
        random_PSpoint_python_formatted = \
"""# Specify your chosen PS point below. If you leave it filled with None, then the script will attempt to read it from the file PS.input.
p= [[None,]*4]*%d"""%len(curr_proc.get('legs'))

        process_definition_string = curr_proc.nice_string().replace('Process:','')
        file=file.format(random_PSpoint_python_formatted,process_definition_string,
                         replace_dict['proc_prefix'].lower())
        new_path = writer.name.replace('check_sa.f', 'check_sa.py')
        new_writer = open(new_path, 'w')
        new_writer.writelines(file)
        # Make it executable
        os.chmod(new_path, os.stat(new_path).st_mode | stat.S_IEXEC)

    def write_improve_ps(self, writer, matrix_element):
        """ Write out the improve_ps subroutines which modify the PS point
        given in input and slightly deform it to achieve exact onshellness on
        all external particles as well as perfect energy-momentum conservation""" 
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        (nexternal,ninitial)=matrix_element.get_nexternal_ninitial()
        replace_dict['ninitial']=ninitial

        # Here we check whether the external particles are on-shell or off-shell
        base_process_string = matrix_element.get('processes')[0].base_string()
        particles_process = base_process_string.replace(">", "", 1).split()
        offshell_or_not = ['.true.' if '*' in elem else '.false.' for elem in particles_process]
        
        mass_list=matrix_element.get_external_masses()[:-2]
        mp_variable_prefix = check_param_card.ParamCard.mp_prefix

        # Write the quadruple precision version of this routine only.
        replace_dict['real_format']=replace_dict['real_mp_format']
        replace_dict['mp_prefix']='MP_'
        replace_dict['exp_letter']='e'
        replace_dict['mp_specifier']='_16'
        replace_dict['coupl_inc_name']='mp_coupl.inc'
        replace_dict['masses_def'] = '\n'
        for i, m in enumerate(mass_list):
            if offshell_or_not[i] == '.false.':
                replace_dict['masses_def'] += f'MASSES({i+1})={mp_variable_prefix}{m}\n'
            else:
                replace_dict['masses_def'] += f'MASSES({i+1})=SQRT(ABS(P(0,{i+1})**2-P(1,{i+1})**2-P(2,{i+1})**2-P(3,{i+1})**2))\n'
        
        # misc.sprint('\n'.join(['MASSES(%(i)d)=%(prefix)s%(m)s'%{'i':i+1,'m':m, 'prefix':mp_variable_prefix} for i, m in enumerate(mass_list)]))
        
        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = '' 

        template = open(os.path.join(self.template_dir,'improve_ps.inc')).read()
        file_mp=template%replace_dict

        if loop_onia.has_onia(matrix_element):
            # With bound states, MP_IMPROVE_PS_POINT_PRECISION is that of
            # onia_loop.f: it puts the physical particles (a bound state of
            # mass m1+m2) on their mass shell with exact momentum
            # conservation, and splits the bound states again.
            name = '%(proc_prefix)sMP_IMPROVE_PS_POINT_PRECISION' % replace_dict
            definition = 'SUBROUTINE %s(' % name
            assert file_mp.count(definition) == 1
            file_mp = file_mp.replace(definition,
                                  'SUBROUTINE %s_CONSTITUENT_LEGS(' % name)
        writer.writelines(file_mp)

    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False
        
        file = open(os.path.join(self.template_dir,'loop_num.inc')).read()
        
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        loop_helas_calls=fortran_model.get_loop_amplitude_helas_calls(matrix_element)
        replace_dict['maxlcouplings']=matrix_element.find_max_loop_coupling()
        replace_dict['loop_helas_calls'] = "\n".join(loop_helas_calls) 
       
        # The squaring is only necessary for the processes with born where the 
        # sum over helicities is done before sending the numerator to CT.
        dp_squaring_lines=['DO I=1,NBORNAMPS',
            'CFTOT=DCMPLX(CF_N(AMPLNUM,I)/DBLE(ABS(CF_D(AMPLNUM,I))),0.0d0)',
            'IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1',
            'RES=RES+CFTOT*BUFF*DCONJG(AMP(I,H))','ENDDO']
        mp_squaring_lines=['DO I=1,NBORNAMPS',
'CFTOT=CMPLX(CF_N(AMPLNUM,I)/(1.0E0_16*ABS(CF_D(AMPLNUM,I))),0.0E0_16,KIND=16)',
            'IF(CF_D(AMPLNUM,I).LT.0) CFTOT=CFTOT*IMAG1',
            'QPRES=QPRES+CFTOT*BUFF*CONJG(AMP(I,H))','ENDDO']
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['dp_squaring']='\n'.join(dp_squaring_lines)
            replace_dict['mp_squaring']='\n'.join(mp_squaring_lines)
        else:
            replace_dict['dp_squaring']='RES=BUFF'
            replace_dict['mp_squaring']='QPRES=BUFF'                       

        # Prepend MP_ to all helas calls.
        self.turn_to_mp_calls(loop_helas_calls)
        replace_dict['mp_loop_helas_calls'] = "\n".join(loop_helas_calls)
        
        file=file%replace_dict
        
        if writer:
            writer.writelines(file)
        else:
            return file

    def write_CT_interface(self, writer, matrix_element, optimized_output=False):
        """ Create the file CT_interface.f which contains the subroutine defining
         the loop HELAS-like calls along with the general interfacing subroutine.
         It is used to interface against any OPP tool, including Samurai and Ninja."""

        files=[]

        # First write CT_interface which interfaces MG5 with CutTools.
        replace_dict=copy.copy(matrix_element.rep_dict)
        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = '' 
        
        # We finalize CT result differently whether we used the built-in
        # squaring against the born.
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['finalize_CT']='\n'.join([\
         'RES(%d)=NORMALIZATION*2.0d0*DBLE(RES(%d))'%(i,i) for i in range(1,4)])
        else:
            replace_dict['finalize_CT']='\n'.join([\
                     'RES(%d)=NORMALIZATION*RES(%d)'%(i,i) for i in range(1,4)])
        
        file = open(os.path.join(self.template_dir,'CT_interface.inc')).read()  

        file = file % replace_dict
        files.append(file)

        if optimized_output:
            ninjars_template = os.path.join(self.template_dir, 'NINJARS_interface.inc')
            with open(ninjars_template) as stream:
                files.append(stream.read() % replace_dict)
        
        # Now collect the different kind of subroutines needed for the
        # loop HELAS-like calls.
        HelasLoopAmpsCallKeys=matrix_element.get_used_helas_loop_amps()

        for callkey in HelasLoopAmpsCallKeys:
            replace_dict=copy.copy(matrix_element.rep_dict)
            # Add to this dictionary all other attribute common to all
            # HELAS-like loop subroutines.
            if matrix_element.get('processes')[0].get('has_born'):
                replace_dict['validh_or_nothing']=',validh'
            else:
                replace_dict['validh_or_nothing']=''
            # In the optimized output, the number of couplings in the loop is
            # not specified so we only treat it here if necessary:
            if len(callkey)>2:
                replace_dict['ncplsargs']=callkey[2]
                cplsargs="".join(["C%d,MP_C%d, "%(i,i) for i in range(1,callkey[2]+1)])
                replace_dict['cplsargs']=cplsargs
                cplsdecl="".join(["C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
                replace_dict['cplsdecl']=cplsdecl
                mp_cplsdecl="".join(["MP_C%d, "%i for i in range(1,callkey[2]+1)])[:-2]
                replace_dict['mp_cplsdecl']=mp_cplsdecl
                cplset="\n".join(["\n".join(["LC(%d)=C%d"%(i,i),\
                                         "MP_LC(%d)=MP_C%d"%(i,i)])\
                              for i in range(1,callkey[2]+1)])
                replace_dict['cplset']=cplset
            
            replace_dict['nloopline']=callkey[0]
            wfsargs="".join(["W%d, "%i for i in range(1,callkey[1]+1)])
            replace_dict['wfsargs']=wfsargs
            # We don't pass the multiple precision mass in the optimized_output
            if not optimized_output:
                margs="".join(["M%d,MP_M%d, "%(i,i) for i in range(1,callkey[0]+1)])
            else:
                margs="".join(["M%d, "%i for i in range(1,callkey[0]+1)])
            replace_dict['margs']=margs                
            wfsargsdecl="".join([("W%d, "%i) for i in range(1,callkey[1]+1)])[:-2]
            replace_dict['wfsargsdecl']=wfsargsdecl
            margsdecl="".join(["M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['margsdecl']=margsdecl
            mp_margsdecl="".join(["MP_M%d, "%i for i in range(1,callkey[0]+1)])[:-2]
            replace_dict['mp_margsdecl']=mp_margsdecl
            weset="\n".join([("WE("+str(i)+")=W"+str(i)) for \
                             i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            weset="\n".join([("WE(%d)=W%d"%(i,i)) for i in range(1,callkey[1]+1)])
            replace_dict['weset']=weset
            msetlines=["M2L(1)=M%d**2"%(callkey[0]),]
            mset="\n".join(msetlines+["M2L(%d)=M%d**2"%(i,i-1) for \
                             i in range(2,callkey[0]+1)])
            replace_dict['mset']=mset            
            mset2lines=["ML(1)=M%d"%(callkey[0]),"ML(2)=M%d"%(callkey[0]),
                  "MP_ML(1)=MP_M%d"%(callkey[0]),"MP_ML(2)=MP_M%d"%(callkey[0])]
            mset2="\n".join(mset2lines+["\n".join(["ML(%d)=M%d"%(i,i-2),
                                               "MP_ML(%d)=MP_M%d"%(i,i-2)]) for \
                                        i in range(3,callkey[0]+3)])
            replace_dict['mset2']=mset2           
            replace_dict['nwfsargs'] = callkey[1]
            if callkey[0]==callkey[1]:
                replace_dict['nwfsargs_header'] = ""
                replace_dict['pairingargs']=""
                replace_dict['pairingdecl']=""
                pairingset="""DO I=1,NLOOPLINE
                                PAIRING(I)=1
                              ENDDO
                           """
                replace_dict['pairingset']=pairingset               
            else:
                replace_dict['nwfsargs_header'] = '_%d'%callkey[1]
                pairingargs="".join([("P"+str(i)+", ") for i in \
                                                         range(1,callkey[0]+1)])
                replace_dict['pairingargs']=pairingargs
                pairingdecl="integer "+"".join([("P"+str(i)+", ") for i in \
                                                    range(1,callkey[0]+1)])[:-2]
                replace_dict['pairingdecl']=pairingdecl
                pairingset="\n".join([("PAIRING("+str(i)+")=P"+str(i)) for \
                             i in range(1,callkey[0]+1)])
                replace_dict['pairingset']=pairingset
            
            file = open(os.path.join(self.template_dir,\
                                             'helas_loop_amplitude.inc')).read()
            file = misc.apply_template(file, replace_dict)
            files.append(file)
        
        file="\n".join(files)
        
        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    # Helper function to split HELAS CALLS in dedicated subroutines placed
    # in different files.
    def split_HELASCALLS(self, writer, replace_dict, template_name, masterfile, \
                         helas_calls, entry_name, bunch_name,n_helas=2000,
                         required_so_broadcaster = 'LOOP_REQ_SO_DONE',
                         continue_label = 1000, momenta_array_name='P',
                         context={}):
        """ Finish the code generation with splitting.         
        Split the helas calls in the argument helas_calls into bunches of 
        size n_helas and place them in dedicated subroutine with name 
        <bunch_name>_i. Also setup the corresponding calls to these subroutine 
        in the replace_dict dictionary under the entry entry_name.
        The context specified will be forwarded to the fileWriter."""
        helascalls_replace_dict=copy.copy(replace_dict)
        helascalls_replace_dict.setdefault('terminal_ieee_use', '')
        helascalls_replace_dict.setdefault('terminal_ieee_declarations', '')
        helascalls_replace_dict['bunch_name']=bunch_name
        helascalls_files=[]
        for i, k in enumerate(range(0, len(helas_calls), n_helas)):
            helascalls_replace_dict['bunch_number']=i+1                
            helascalls_replace_dict['helas_calls']=\
                                           '\n'.join(helas_calls[k:k + n_helas])
            if replace_dict.get('terminal_ieee_use'):
                has_private = re.search(_PRIVATE_MP_WL_CALL,
                    helascalls_replace_dict['helas_calls'])
                for key in ('terminal_ieee_use', 'terminal_ieee_declarations'):
                    helascalls_replace_dict[key] = replace_dict[key] if has_private else ''
            helascalls_replace_dict['required_so_broadcaster']=\
                                                         required_so_broadcaster
            helascalls_replace_dict['continue_label']=continue_label
            new_helascalls_file = open(os.path.join(self.template_dir,\
                                                          template_name)).read()
            new_helascalls_file = new_helascalls_file % helascalls_replace_dict
            helascalls_files.append(new_helascalls_file)
        # Setup the call to these HELASCALLS subroutines in loop_matrix.f
        helascalls_calls = [ "CALL %s%s_%d(%s,NHEL,H,IC)"%\
              (replace_dict['proc_prefix'] ,bunch_name,a+1,momenta_array_name) \
                                          for a in range(len(helascalls_files))]
        replace_dict[entry_name]='\n'.join(helascalls_calls)
        if writer:
            for i, helascalls_file in enumerate(helascalls_files):
                filename = '%s_%d.f'%(bunch_name,i+1)
                writers.FortranWriter(filename).writelines(helascalls_file,
                                                                context=context)
        else:
                masterfile='\n'.join([masterfile,]+helascalls_files)                

        return masterfile

    def write_loopmatrix(self, writer, matrix_element, fortran_model,
                                                                 noSplit=False):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line
        # When the user asks for the polarized matrix element we must 
        # multiply back by the helicity averaging factor
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        replace_dict['beamone_helavgfactor'], replace_dict['beamtwo_helavgfactor'] =\
                                       matrix_element.get_beams_hel_avg_factor()

        # These entries are specific for the output for loop-induced processes
        # Also sets here the details of the squaring of the loop ampltiudes
        # with the born or the loop ones.
        if not matrix_element.get('processes')[0].get('has_born'):
            replace_dict['compute_born']=\
"""C There is of course no born for loop induced processes
ANS(0)=0.0d0
"""
            replace_dict['set_reference']='\n'.join([
              'C For loop-induced, the reference for comparison is set later'+\
              ' from the total contribution of the previous PS point considered.',
              'C But you can edit here the value to be used for the first PS point.',
                'if (NPSPOINTS.eq.0) then','ref=1.0d-50','else',
                'ref=nextRef/DBLE(NPSPOINTS)','endif'])
            replace_dict['loop_induced_setup'] = '\n'.join([
              'HELPICKED_BU=HELPICKED','HELPICKED=H','MP_DONE=.FALSE.',
              'IF(SKIPLOOPEVAL) THEN','GOTO 1227','ENDIF'])
            replace_dict['loop_induced_finalize'] = \
            ("""DO I=NCTAMPS+1,NLOOPAMPS
               IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN
                 WRITE(*,*) '##W03 WARNING Contribution ',I
                 WRITE(*,*) ' is unstable for helicity ',H
               ENDIF
C                IF(.NOT.%(proc_prefix)sISZERO(ABS(AMPL(2,I))+ABS(AMPL(3,I)),REF,-1,H)) THEN
C                  WRITE(*,*) '##W04 WARNING Contribution ',I,' for helicity ',H,' has a contribution to the poles.'
C                  WRITE(*,*) 'Finite contribution         = ',AMPL(1,I)
C                  WRITE(*,*) 'single pole contribution    = ',AMPL(2,I)
C                  WRITE(*,*) 'double pole contribution    = ',AMPL(3,I)
C                ENDIF
               ENDDO
               1227 CONTINUE
               HELPICKED=HELPICKED_BU""")%replace_dict
            replace_dict['loop_helas_calls']=""
            replace_dict['nctamps_or_nloopamps']='nloopamps'
            replace_dict['nbornamps_or_nloopamps']='nloopamps'
            replace_dict['squaring']=\
                    """ANS(1)=ANS(1)+DBLE(CFTOT*AMPL(1,I)*DCONJG(AMPL(1,J)))
                       IF (J.EQ.1) THEN
                         ANS(2)=ANS(2)+DBLE(CFTOT*AMPL(2,I))+DIMAG(CFTOT*AMPL(2,I))
                         ANS(3)=ANS(3)+DBLE(CFTOT*AMPL(3,I))+DIMAG(CFTOT*AMPL(3,I))                         
                       ENDIF"""      
        else:
            replace_dict['compute_born']=\
"""C Compute the born, for a specific helicity if asked so.
call %(proc_prefix)ssmatrixhel(P_USER,USERHEL,FLAVOR,ANS(0))
"""%matrix_element.rep_dict
            replace_dict['set_reference']=\
"""C We chose to use the born evaluation for the reference
call %(proc_prefix)ssmatrix(p,FLAVOR,ref)"""%matrix_element.rep_dict
            replace_dict['loop_induced_helas_calls'] = ""
            replace_dict['loop_induced_finalize'] = ""
            replace_dict['loop_induced_setup'] = ""
            replace_dict['nctamps_or_nloopamps']='nctamps'
            replace_dict['nbornamps_or_nloopamps']='nbornamps'
            replace_dict['squaring']='\n'.join(['DO K=1,3',
                   'ANS(K)=ANS(K)+2.0d0*DBLE(CFTOT*AMPL(K,I)*DCONJG(AMP(J,H)))',
                                                                       'ENDDO'])

        # Write a dummy nsquaredSO.inc which is used in the default
        # loop_matrix.f code (even though it does not support split orders evals)
        # just to comply with the syntax expected from the external code using MadLoop.
        writers.FortranWriter('nsquaredSO.inc').writelines(
"""INTEGER NSQUAREDSO
PARAMETER (NSQUAREDSO=0)""")

        # Actualize results from the loops computed. Only necessary for
        # processes with a born.
        actualize_ans=[]
        if matrix_element.get('processes')[0].get('has_born'):
            actualize_ans.append("DO I=NCTAMPS+1,NLOOPAMPS")
            actualize_ans.extend("ANS(%d)=ANS(%d)+AMPL(%d,I)"%(i,i,i) for i \
                                                                  in range(1,4)) 
            actualize_ans.append(\
               "IF((CTMODERUN.NE.-1).AND..NOT.CHECKPHASE.AND.(.NOT.S(I))) THEN")
            actualize_ans.append(\
                   "WRITE(*,*) '##W03 WARNING Contribution ',I,' is unstable.'")
            actualize_ans.extend(["ENDIF","ENDDO"])
            replace_dict['actualize_ans']='\n'.join(actualize_ans)
        else:
            replace_dict['actualize_ans']=\
            ("""C We add five powers to the reference value to loosen a bit the vanishing pole check.
C               IF(.NOT.(CHECKPHASE.OR.(.NOT.HELDOUBLECHECKED)).AND..NOT.%(proc_prefix)sISZERO(ABS(ANS(2))+ABS(ANS(3)),ABS(ANS(1))*(10.0d0**5),-1,H)) THEN
C                 WRITE(*,*) '##W05 WARNING Found a PS point with a contribution to the single pole.'
C                 WRITE(*,*) 'Finite contribution         = ',ANS(1)
C                 WRITE(*,*) 'single pole contribution    = ',ANS(2)
C                 WRITE(*,*) 'double pole contribution    = ',ANS(3)
C               ENDIF""")%replace_dict
        
        # Write out the color matrix
        (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
        CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorNumFactors.dat'%matrix_element.rep_dict),'w')
        for ColorLine in CMNum:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        CMWriter=open(pjoin('..','MadLoop5_resources',
          '%(proc_prefix)sColorDenomFactors.dat'%matrix_element.rep_dict),'w')
        for ColorLine in CMDenom:
            CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
        CMWriter.close()
        
        # Write out the helicity configurations
        HelConfigs=matrix_element.get_helicity_matrix()
        HelConfigWriter=open(pjoin('..','MadLoop5_resources',
                 '%(proc_prefix)sHelConfigs.dat'%matrix_element.rep_dict),'w')
        for HelConfig in HelConfigs:
            HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
        HelConfigWriter.close()
        
        # Extract helas calls
        loop_amp_helas_calls = fortran_model.get_loop_amp_helas_calls(\
                                                                 matrix_element)
        # The proc_prefix must be replaced
        loop_amp_helas_calls = [lc % matrix_element.rep_dict 
                                                 for lc in loop_amp_helas_calls]
        
        born_ct_helas_calls, UVCT_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element)
        # In the default output, we do not need to separate these two kind of
        # contributions
        born_ct_helas_calls = born_ct_helas_calls + UVCT_helas_calls
        file = open(os.path.join(self.template_dir,\
        
                        'loop_matrix_standalone.inc')).read()
        
        if matrix_element.get('processes')[0].get('has_born'):
            toBeRepaced='loop_helas_calls'
        else:
            toBeRepaced='loop_induced_helas_calls'

        # Decide here whether we need to split the loop_matrix.f file or not.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>1000)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'helas_calls_split.inc',file,born_ct_helas_calls,\
                            'born_ct_helas_calls','helas_calls_ampb')
            file=self.split_HELASCALLS(writer,replace_dict,\
                    'helas_calls_split.inc',file,loop_amp_helas_calls,\
                    toBeRepaced,'helas_calls_ampl')
        else:
            replace_dict['born_ct_helas_calls']='\n'.join(born_ct_helas_calls)
            replace_dict[toBeRepaced]='\n'.join(loop_amp_helas_calls)

        #In loop-induced, particles are put onshell to get a better precision on PS points. If we want to study processes with external off-shell particles we need
        #it to consider the offshell mass m^2 = p^2 to the on-shell mass.
        #KEEP_OFFSHELL_MASS contains the information based on the generation string of which external particle should be kept off-shell.
        base_process_string = matrix_element.get('processes')[0].base_string()
        particles_process = base_process_string.replace(">", "", 1).split()
        offshell_or_not = ['.true.' if '*' in elem else '.false.' for elem in particles_process]
        logger.info("Particles with .true. are generated off-shell: " + str(offshell_or_not))

        replace_dict["keep_offshell_mass"] = ""
        for i in range(len(offshell_or_not)):
            replace_dict["keep_offshell_mass"] += f"KEEP_OFFSHELL_MASS({i + 1}) = {offshell_or_not[i]}\n"
        
        file = file % replace_dict

        loop_calls_finder = re.compile(r'^\s*CALL\S*LOOP\S*')
        n_loop_calls = len([call for call in loop_amp_helas_calls if not loop_calls_finder.match(call) is None])
        if writer:
            # Write the file
            writer.writelines(file)  
            return n_loop_calls
        else:
            # Return it to be written along with the others
            return n_loop_calls, file
                  
    def write_bornmatrix(self, writer, matrix_element, fortran_model):
        """Create the born_matrix.f file for the born process as for a standard
        tree-level computation."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")

        # For now, we can use the exact same treatment as for tree-level
        # computations by redefining here a regular HelasMatrixElementf or the
        # born process.
        # It is important to make a deepcopy, as we don't want any possible 
        # treatment on the objects of the bornME to have border effects on
        # the content of the LoopHelasMatrixElement object.
        bornME = helas_objects.HelasMatrixElement()
        for prop in bornME.keys():
            bornME.set(prop,copy.deepcopy(matrix_element.get(prop)))
        bornME.set('base_amplitude',None,force=True)
        bornME.set('diagrams',copy.deepcopy(\
                                            matrix_element.get_born_diagrams()))        
        bornME.set('color_basis',copy.deepcopy(\
                                        matrix_element.get('born_color_basis')))
        bornME.set('color_matrix',copy.deepcopy(\
                              color_amp.ColorMatrix(bornME.get('color_basis'))))
        # This is to decide whether once to reuse old wavefunction to store new
        # ones (provided they are not used further in the code.)
        bornME.optimization = True
        replace_dict = super(LoopProcessExporterFortranSA,self).write_matrix_element_v4(
                                                  writer, bornME, fortran_model,
                           proc_prefix=matrix_element.rep_dict['proc_prefix'],
                           return_replace_dict=True)
        if not replace_dict:
            return 0
        # The MadLoop drivers include the number of squared split orders of
        # the born, next to born_matrix.f.
        writers.FortranWriter(pjoin(
            os.path.dirname(os.path.abspath(writer.name)),
            'nsqso_born.inc')).writelines(
                """INTEGER NSQSO_BORN
                   PARAMETER (NSQSO_BORN=%d)"""%replace_dict['nSqAmpSplitOrders'])
        return replace_dict['return_value']

    def write_born_amps_and_wfs(self, writer, matrix_element, fortran_model,
                                                                 noSplit=False): 
        """ Writes out the code for the subroutine MP_BORN_AMPS_AND_WFS which 
        computes just the external wavefunction and born amplitudes in 
        multiple precision. """

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        replace_dict = copy.copy(matrix_element.rep_dict)

        # For the wavefunction copy, check what suffix is needed for the W array
        if matrix_element.get('processes')[0].get('has_born'):
            replace_dict['h_w_suffix']=',H'
        else:
            replace_dict['h_w_suffix']=''            

        # Extract helas calls
        born_amps_and_wfs_calls , uvct_amp_calls = \
          fortran_model.get_born_ct_helas_calls(matrix_element, include_CT=True)
        # In the default output, these two kind of contributions do not need to
        # be differentiated
        born_amps_and_wfs_calls = born_amps_and_wfs_calls + uvct_amp_calls
        
        # Turn these HELAS calls to the multiple-precision version of the HELAS
        # subroutines.
        self.turn_to_mp_calls(born_amps_and_wfs_calls)

        file = open(os.path.join(self.template_dir,\
                        'mp_born_amps_and_wfs.inc')).read()   
        # Decide here whether we need to split the loop_matrix.f file or not.
        if (not noSplit and (len(matrix_element.get_all_amplitudes())>2000)):
            file=self.split_HELASCALLS(writer,replace_dict,\
                            'mp_helas_calls_split.inc',file,\
                            born_amps_and_wfs_calls,'born_amps_and_wfs_calls',\
                            'mp_helas_calls')
        else:
            replace_dict['born_amps_and_wfs_calls']=\
                                            '\n'.join(born_amps_and_wfs_calls)

        file = misc.apply_template(file, replace_dict)
        if writer:
            # Write the file
            writer.writelines(file)
        else:
            # Return it to be written along with the others
            return file

#===============================================================================
# LoopProcessOptimizedExporterFortranSA
#===============================================================================

class LoopProcessOptimizedExporterFortranSA(LoopProcessExporterFortranSA):
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format which exploits the Pozzorini method of representing
       the loop numerators as polynomial to render its evaluations faster."""

    template_dir=os.path.join(_file_path,'iolibs/template_files/loop_optimized')
    # The option below controls whether one wants to group together in one single
    # CutTools/TIR call the loops with same denominator structure
    forbid_loop_grouping = False
    
    # List of potential TIR library one wants to link to.
    # Golem and Samurai will typically get obtained from gosam_contrib
    # which might also contain a version of ninja. We must therefore
    # make sure that ninja appears first in the list of -L because 
    # it is the tool for which the user is most susceptible of 
    # using a standalone version independent of gosam_contrib
    all_tir=['pjfry','iregi','ninja','golem','samurai','collier','ninjars','tirust']
    
    def __init__(self, dir_path = "", opt=None):
        """Initiate the LoopProcessOptimizedExporterFortranSA with directory 
        information on where to find all the loop-related source files, 
        like CutTools and TIR"""

        super(LoopProcessOptimizedExporterFortranSA,self).__init__(dir_path, opt)

        # TIR available ones
        self.tir_available_dict={'pjfry':False,'iregi':True,'golem':True,
                                 'samurai':True,'ninja':True,'collier':True,'ninjars':True,'tirust':True}

        for tir in self.all_tir:
            tir_dir="%s_dir"%tir
            if tir_dir in self.opt and not self.opt[tir_dir] is None:
                # Make sure to defer the 'local path' to the current MadGraph7 root.
                tir_path = self.opt[tir_dir].strip()
                if tir_path.startswith('.') or (
                        tir in ('ninjars', 'tirust') and not os.path.isabs(tir_path)):
                    tir_path = os.path.abspath(pjoin(MG5DIR,tir_path))
                setattr(self,tir_dir,tir_path)
            else:
                setattr(self,tir_dir,'')

    def copy_template(self, model):
        """Additional actions needed to setup the Template. 
        """
        
        super(LoopProcessOptimizedExporterFortranSA, self).copy_template(model)
        
        self.loop_optimized_additional_template_setup()

    #===========================================================================
    # Bound states (onia)
    #===========================================================================
    def check_onia_loop_output(self, matrix_element):
        """Refuse the loop outputs of processes with bound states that the
        onia layer (onia_loop.f) does not handle."""
        if not matrix_element.get('processes')[0].get('has_born'):
            raise loop_onia.OniaLoopError('Loop-induced processes with bound states '
                                          'are not supported yet.')
        if self.compute_color_flows:
            raise loop_onia.OniaLoopError("The option 'loop_color_flows' is not "
                                          "supported for processes with bound states.")

    def write_onia_loop_files(self, matrix_element, fortran_model, loop_replace_dict):
        """Write onia_loop_data.inc and onia_loop.f, the projection of the
        constituent amplitudes onto the bound states (see loop_onia.py)."""

        self.check_onia_loop_output(matrix_element)
        self.has_onia = True
        info = loop_onia.onia_loop_info(matrix_element)
        nexternal = info['nexternal']
        ncomb = len(info['helc'])
        nonia = len(info['onia'])
        # MadFKS's MC over helicities (BinothLHA.f) reads
        # MadLoop5_resources/HelFilter.dat in the numbering of SLOOPMATRIXHEL,
        # the physical configurations: all of them are kept (multiplicity 1),
        # since which J_z of a massive bound state vanish depends on the frame.
        # MadLoop's own helicity filter is off for bound states
        # (HelicityFilterLevel=0, loop_matrix.f), so it neither reads nor
        # writes this file.
        resources = pjoin('..', 'MadLoop5_resources')
        if os.path.isdir(resources):
            with open(pjoin(resources, '%(proc_prefix)sHelFilter.dat'
                            % matrix_element.rep_dict), 'w') as hel_filter:
                hel_filter.write(''.join(' 1\n' for _ in info['phys_helc']))
        njzt = len(info['jz'])
        nborn = matrix_element.get_number_of_born_amplitudes()
        model = matrix_element.get('processes')[0].get('model')

        # The colour flows of the (projected) Born amplitudes and their
        # colour matrix (loop_onia.py, "Colour in the basis of colour flows")
        ampb_to_jampb = self.get_amp_to_jamp_map(
            matrix_element.get_born_color_amplitudes(), nborn)
        nflow = len(matrix_element.get('born_color_basis'))
        born_cm = loop_onia.flow_colour_matrix(
            color_amp.ColorMatrix(matrix_element.get('born_color_basis')))

        def ints(values):
            return ','.join('%d' % v for v in values)

        lines = ['C     Bound states of the process (loop_onia.py, onia_loop.f)',
                 'INTEGER ONIA_NONIA, ONIA_NJZT, ONIA_NOTHERS, ONIA_IDEN',
                 'PARAMETER (ONIA_NONIA=%d, ONIA_NJZT=%d, ONIA_NOTHERS=%d, ONIA_IDEN=%d)'
                 % (nonia, njzt, info['nothers'], matrix_element.get_denominator_factor()),
                 'C     physical leg of each constituent leg, and the other constituent',
                 'INTEGER ONIA_MAPPING(%d), ONIA_PARTNER(%d)' % (nexternal, nexternal),
                 'C     per bound state: particle and antiparticle legs, S, J, colour',
                 'INTEGER ONIA_Q1(%d), ONIA_Q2(%d), ONIA_S(%d), ONIA_J(%d), ONIA_C(%d)'
                 % ((nonia,) * 5),
                 'C     J_z of each bound state in each polarisation configuration',
                 'INTEGER ONIA_JZ(%d,%d)' % (nonia, njzt),
                 'C     the constituent helicity configurations (HelConfigs.dat) and',
                 'C     the index of the helicities of their other legs',
                 'INTEGER ONIA_HELC(%d,%d), ONIA_OTHERS(%d)' % (nexternal, ncomb, ncomb)]
        partner = [0] * nexternal
        for o in info['onia']:
            partner[o['q1'] - 1] = o['q2']
            partner[o['q2'] - 1] = o['q1']
        lines.append('DATA ONIA_MAPPING/%s/' % ints(info['mapping']))
        lines.append('DATA ONIA_PARTNER/%s/' % ints(partner))
        for key, attr in [('ONIA_Q1', 'q1'), ('ONIA_Q2', 'q2'), ('ONIA_S', 'S'),
                          ('ONIA_J', 'J'), ('ONIA_C', 'C')]:
            lines.append('DATA %s/%s/' % (key, ints([o[attr] for o in info['onia']])))
        lines.append('DATA ONIA_JZ/%s/' % ints([j for t in info['jz'] for j in t]))
        if info['ninitial_onia']:
            # a decaying bound state (its incoming spin projector, its decay
            # LDME): 1 for it, 0 for the others
            lines.append('C     1 for a decaying (initial-state) bound state')
            lines.append('INTEGER ONIA_IN(%d)' % nonia)
            lines.append('DATA ONIA_IN/%s/' % ints([1 if o['initial'] else 0
                                                    for o in info['onia']]))
        lines.append('DATA ONIA_HELC/%s/' % ints([h for hel in info['helc'] for h in hel]))
        lines.append('DATA ONIA_OTHERS/%s/' % ints(info['others']))
        nphys, nphyscomb = info['nexternal_phys'], len(info['phys_helc'])
        lines += ['C     the physical helicity configurations (SLOOPMATRIXHEL): the',
                  'C     helicity matrix of the physical process, a bound state with its',
                  'C     J_z = -J..J, the last leg fastest (as the tree-level output and',
                  'C     the FKS Born); per physical leg its constituent leg, or minus',
                  'C     the index of its bound state; per configuration the signature',
                  'C     (ONIA_OTHERS) of the helicities of the other legs',
                  'C     A selected configuration is not averaged over the initial',
                  'C     helicities (ONIA_HELAVG of them), as in MadLoop and the tree level',
                  'INTEGER ONIA_NPHYS, ONIA_PHYS_NCOMB, ONIA_HELAVG',
                  'PARAMETER (ONIA_NPHYS=%d, ONIA_PHYS_NCOMB=%d, ONIA_HELAVG=%d)'
                  % (nphys, nphyscomb, matrix_element.get_hel_avg_factor()),
                  'INTEGER ONIA_PHYS_HEL(%d,%d), ONIA_PHYS_LEG(%d)' % (nphys, nphyscomb, nphys),
                  'INTEGER ONIA_PHYS_OTHERS(%d)' % nphyscomb,
                  'DATA ONIA_PHYS_HEL/%s/' % ints([h for hel in info['phys_helc'] for h in hel]),
                  'DATA ONIA_PHYS_LEG/%s/' % ints(info['phys_leg']),
                  'DATA ONIA_PHYS_OTHERS/%s/' % ints(info['phys_others'])]
        lines.append('C     the colour flows of the projected Born amplitudes (ONIA_BF, one')
        lines.append('C     basis: ONIA_BORN_FLOWS) and the colour matrix between them')
        lines.append('INTEGER ONIA_NFLOW')
        lines.append('PARAMETER (ONIA_NFLOW=%d)' % nflow)
        lines.extend(loop_onia.flow_map_lines('ONIA_BF', [ampb_to_jampb], [nflow]))
        lines.extend(loop_onia.flow_matrix_lines('ONIA_FCM', [born_cm], (nflow, nflow)))
        # QCD loops hold both constituents of a bound state, and so meet the
        # degenerate threshold integrals, only when the constituents are
        # coloured (not for leptonia)
        process = matrix_element.get('processes')[0]
        coloured = any(model.get_particle(leg.get('id')).get('color') != 1
                       for leg in process.get('legs') if leg.get('onium'))
        lines.append('C     some loop holds both constituents of a bound state')
        lines.append('LOGICAL ONIA_THRESHOLD_LOOPS')
        lines.append('PARAMETER (ONIA_THRESHOLD_LOOPS=%s)'
                     % ('.TRUE.' if coloured else '.FALSE.'))
        lines.append('C     the P-wave bound state (0 if none), and L of each bound state')
        lines.append('INTEGER ONIA_IPW, ONIA_L(%d)' % nonia)
        lines.append('PARAMETER (ONIA_IPW=%d)' % info['ipwave'])
        if info['npwave'] == 2:
            lines.append('C     the second P-wave bound state (dual component 2)')
            lines.append('INTEGER ONIA_IPW2')
            lines.append('PARAMETER (ONIA_IPW2=%d)' % info['ipwaves'][1])
        lines.append('DATA ONIA_L/%s/' % ints([o['L'] for o in info['onia']]))
        writers.FortranWriter('onia_loop_data.inc').writelines(lines)
        context = self.get_context(matrix_element)
        if context.get('OniaPwave') and not context.get('tirust_raised_powers'):
            raise loop_onia.OniaLoopError(
                'The one-loop virtual of a P-wave bound state needs TIRust with '
                'raised propagator powers (installation metadata raised_powers): '
                "run 'install tirust --update'.")

        # The masses of the constituent legs
        masses = []
        for wf in sorted(matrix_element.get_external_wavefunctions(),
                         key=lambda wf: wf.get('number_external')):
            mass = model.get('particle_dict')[wf.get('pdg_code')].get('mass')
            masses.append('ZERO' if mass.lower() == 'zero' else 'ABS(%s)' % mass)
        mass_lines = '\n'.join('CMASS(%d)=%s' % (i + 1, m) for i, m in enumerate(masses))
        # In quadruple precision, the masses squared of the loop propagators
        # are the double-precision ones (M2L of the loop routines): the QP
        # constituent masses are their square roots, so that the
        # constituents stay exactly on the mass shell the reduction sees
        # (TIRust recognises the Coulomb-exchange degeneracy at the input
        # precision).
        mp_mass_lines = '\n'.join(
            'CMASS(%d)=%s' % (i + 1, m if m == 'ZERO' else
                              'SQRT(REAL(REAL(%s,KIND=8)**2,KIND=16))' % m)
            for i, m in enumerate(masses))

        # The Born amplitudes alone (no counterterm), in DP and QP
        born_calls = []
        for diagram in matrix_element.get_born_diagrams():
            born_calls.extend([fortran_model.get_wavefunction_call(wf)
                               for wf in diagram.get('wavefunctions')])
            for amplitude in diagram.get('amplitudes'):
                born_calls.append(fortran_model.get_amplitude_call(amplitude))
        mp_born_calls = list(born_calls)
        self.turn_to_mp_calls(mp_born_calls)
        if loop_onia.has_pwave(matrix_element):
            born_calls = loop_onia.pwave_dual_names(born_calls)
            mp_born_calls = loop_onia.pwave_dual_names(mp_born_calls, mp=True)

        ldme_product = self.get_ldme_product(matrix_element) or '1D0'
        # the entries of loop_matrix.f (split orders, colour matrix, ...) and
        # the bound states
        replace_dict = copy.copy(loop_replace_dict)
        replace_dict.update(matrix_element.rep_dict)
        replace_dict.update({
            'onia_nexternal_phys': info['nexternal_phys'],
            'nexternal': nexternal,
            'ncomb': ncomb,
            'nbornamps': nborn,
            'onia_dp_masses': mass_lines,
            'onia_mp_masses': mp_mass_lines,
            'onia_born_calls': '\n'.join(born_calls),
            'onia_mp_born_calls': '\n'.join(mp_born_calls),
            'onia_dp_norm': loop_onia.fortran_double(info['norm']),
            'onia_mp_norm': loop_onia.fortran_quad(info['norm']),
            'onia_ldme_product': ldme_product,
            'onia_ldme_perturbative': self.get_ldme_perturbative(matrix_element)})
        if ncomb != matrix_element.get_helicity_combinations():
            raise loop_onia.OniaLoopError('Inconsistent helicity configurations.')
        # the physical incoming particles (ONIA_IR_POLES,
        # MP_IMPROVE_PS_POINT_PRECISION): a decaying bound state is one
        replace_dict['nincoming'] = info['ninitial_phys']
        self.write_onia_loop_links(matrix_element, info, model)
        file = open(os.path.join(self.template_dir, 'onia_loop.inc')).read() % replace_dict
        writers.FortranWriter('onia_loop.f').writelines(file, context=context)
        # The LDMEs of the onia card (Source/MODEL/ldme.inc, export_onia_files)
        ln('../ldme.inc')

    def write_onia_loop_links(self, matrix_element, info, model):
        """onia_loop_links.inc: the colour links between the physical legs
        in the Born colour-flow basis (loop_onia.flow_color_links) and the
        coefficient of 1/eps of the Gamma_j of each physical leg, for
        ONIA_IR_POLES."""

        nflow = len(matrix_element.get('born_color_basis'))
        links = loop_onia.flow_color_links(matrix_element, info)
        keys = sorted(links)
        nlinks = max(len(keys), 1)
        # the massless quark flavours (gamma_g)
        nl = len(set(abs(p.get('pdg_code')) for p in model.get('particles')
                     if abs(p.get_color()) == 3 and p.get('spin') == 2
                     and p.get('mass').lower() == 'zero'))
        cf, ca, tr = fractions.Fraction(4, 3), fractions.Fraction(3), fractions.Fraction(1, 2)
        gamma = [fractions.Fraction(0)] * info['nexternal_phys']
        onia_legs = dict((o['phys'], o) for o in info['onia'])
        for wf in matrix_element.get_external_wavefunctions():
            n = wf.get('number_external')
            leg = info['mapping'][n - 1]
            part = model.get_particle(wf.get('pdg_code'))
            if leg in onia_legs:
                gamma[leg - 1] = ca if (onia_legs[leg]['C'] == 8 and
                                        abs(part.get_color()) == 3) else fractions.Fraction(0)
            elif abs(part.get_color()) == 8:
                gamma[leg - 1] = ca * 11 / 6 - tr * 2 * nl / 3 \
                    if part.get('mass').lower() == 'zero' else ca
            elif abs(part.get_color()) == 3:
                gamma[leg - 1] = cf * 3 / 2 if part.get('mass').lower() == 'zero' else cf

        def col(values):
            return ','.join(loop_onia.fortran_double(v) for v in values)

        lines = ['C     Colour links between the physical legs (ONIA_IR_POLES)',
                 'INTEGER ONIA_NLINKS',
                 'PARAMETER (ONIA_NLINKS=%d)' % len(keys),
                 'INTEGER ONIA_LINK_J(%d), ONIA_LINK_K(%d)' % (nlinks, nlinks),
                 'REAL*8 ONIA_LEG_GAMMA(%d)' % info['nexternal_phys'],
                 'DATA ONIA_LEG_GAMMA/%s/' % col(gamma)]
        if keys:
            lines.append('DATA ONIA_LINK_J/%s/' % ','.join('%d' % k[0] for k in keys))
            lines.append('DATA ONIA_LINK_K/%s/' % ','.join('%d' % k[1] for k in keys))
        else:
            lines.append('DATA ONIA_LINK_J/0/, ONIA_LINK_K/0/')
        lines.extend(loop_onia.flow_matrix_lines(
            'ONIA_FCL', [links[k] for k in keys], (nflow, nflow, nlinks),
            '<T_j.T_k> between the Born colour flows, per link'))
        writers.FortranWriter('onia_loop_links.inc').writelines(lines)
        if info['npwave'] == 2:
            self.write_onia_loop_pwsoft2(matrix_element, info)
        elif info['ipwave']:
            self.write_onia_loop_pwsoft(matrix_element, info)

    def write_onia_loop_pwsoft(self, matrix_element, info):
        """onia_loop_pwsoft.inc: the colour of the soft current of the P-wave
        bound state (loop_onia.pwave_soft_current), for the P -> S pole of
        ONIA_PW_SOFT_POLE: the colour bases of its S-wave states (PWS_SF,
        basis A for state A) and the flow matrices of the links (PWS_FX)."""

        nflow = len(matrix_element.get('born_color_basis'))
        aux = loop_onia.pwave_soft_current(matrix_element, info, self.get_amp_to_jamp_map)
        links = [(ia + 1, k, kind, x) for ia, a in enumerate(aux)
                 for (k, kind), x in sorted(a['links'].items())]
        naux, nlinks = max(len(aux), 1), max(len(links), 1)
        nfmax = max([a['nflow'] for a in aux] + [1])

        def col(values):
            return ','.join(loop_onia.fortran_double(v) for v in values)

        lines = ['C     The S-wave states of the soft current of the P-wave bound state',
                 'C     and their colour (ONIA_PW_SOFT_POLE)',
                 'INTEGER PWS_NAUX, PWS_NLINKS, PWS_NFMAX',
                 'PARAMETER (PWS_NAUX=%d, PWS_NLINKS=%d, PWS_NFMAX=%d)'
                 % (len(aux), len(links), nfmax),
                 "CHARACTER*2 PWS_LABEL(%d)" % naux,
                 'REAL*8 PWS_CNORM(%d)' % naux,
                 'INTEGER PWS_AUX(%d), PWS_LEG(%d), PWS_KIND(%d)' % (nlinks, nlinks, nlinks)]
        if aux:
            lines.append('DATA PWS_LABEL/%s/' % ','.join("'%s'" % a['label'] for a in aux))
            lines.append('DATA PWS_CNORM/%s/' % col(a['cnorm'] for a in aux))
        else:
            lines.append("DATA PWS_LABEL/'  '/, PWS_CNORM/0D0/")
        if links:
            lines.append('DATA PWS_AUX/%s/' % ','.join('%d' % l[0] for l in links))
            lines.append('DATA PWS_LEG/%s/' % ','.join('%d' % l[1] for l in links))
            lines.append('DATA PWS_KIND/%s/' % ','.join('%d' % l[2] for l in links))
        else:
            lines.append('DATA PWS_AUX/0/, PWS_LEG/0/, PWS_KIND/0/')
        lines.append('C     the colour flows of the S-wave states (ONIA_PWS_FLOWS)')
        lines.extend(loop_onia.flow_map_lines('PWS_SF', [a['amp_s'] for a in aux],
                                              [a['nflow'] for a in aux]))
        lines.extend(loop_onia.flow_matrix_lines(
            'PWS_FX', [l[3] for l in links], (nflow, nfmax, nlinks),
            'X between the Born flows and those of the S-wave state, per link'))
        writers.FortranWriter('onia_loop_pwsoft.inc').writelines(lines)

    def write_onia_loop_pwsoft2(self, matrix_element, info):
        """onia_loop_pwsoft.inc of two P-wave bound states: the colour of the
        soft current of each state (loop_onia.pwave_soft_current, the other
        state keeping its colour), indexed by the state (1: ONIA_IPW, 2:
        ONIA_IPW2), and of the dipole-dipole exchange between them
        (loop_onia.pwave_dipole_dipole), for ONIA_PW_SOFT_POLE. The colour
        bases (PWS_SF) are those of S-wave state A of state S, number
        A+(S-1)*PWS_NAUXMAX, then those of the dipole-dipole terms N, number
        2*PWS_NAUXMAX+N."""

        nflow = len(matrix_element.get('born_color_basis'))
        auxs = [loop_onia.pwave_soft_current(matrix_element, info, self.get_amp_to_jamp_map, ipw)
                for ipw in info['ipwaves']]
        links = [[(ia + 1, k, kind, x) for ia, a in enumerate(aux)
                  for (k, kind), x in sorted(a['links'].items())] for aux in auxs]
        dd = loop_onia.pwave_dipole_dipole(matrix_element, info, self.get_amp_to_jamp_map)
        naux = max([len(a) for a in auxs] + [1])
        nlinks = max([len(l) for l in links] + [1])
        ndd = max(len(dd), 1)
        nfmax = max([a['nflow'] for aux in auxs for a in aux] + [d['nflow'] for d in dd] + [1])

        def col(values):
            return ','.join(loop_onia.fortran_double(v) for v in values)

        def padded(rows, size, fill):
            return [r + [fill] * (size - len(r)) for r in rows]

        lines = ['C     Two P-wave bound states: the S-wave states of the soft current of',
                 'C     each state and their colour (index 1: ONIA_IPW, 2: ONIA_IPW2), and',
                 'C     the colour of the dipole-dipole exchange between the two',
                 'C     (ONIA_PW_SOFT_POLE)',
                 'INTEGER PWS_NAUXMAX, PWS_NLINKSMAX, PDD_N, PDD_NMAX, PWS_NFMAX',
                 'PARAMETER (PWS_NAUXMAX=%d, PWS_NLINKSMAX=%d, PDD_N=%d, PDD_NMAX=%d)'
                 % (naux, nlinks, len(dd), ndd),
                 'PARAMETER (PWS_NFMAX=%d)' % nfmax,
                 'INTEGER PWS_NAUX(2), PWS_NLINKS(2)',
                 'CHARACTER*2 PWS_LABEL(PWS_NAUXMAX,2)',
                 'REAL*8 PWS_CNORM(PWS_NAUXMAX,2)',
                 'INTEGER PWS_AUX(PWS_NLINKSMAX,2), PWS_LEG(PWS_NLINKSMAX,2)',
                 'INTEGER PWS_KIND(PWS_NLINKSMAX,2)',
                 'CHARACTER*2 PDD_LABEL1(PDD_NMAX), PDD_LABEL2(PDD_NMAX)',
                 'INTEGER PDD_KIND1(PDD_NMAX), PDD_KIND2(PDD_NMAX)',
                 'REAL*8 PDD_CNORM1(PDD_NMAX), PDD_CNORM2(PDD_NMAX)',
                 'DATA PWS_NAUX/%s/' % ','.join('%d' % len(a) for a in auxs),
                 'DATA PWS_NLINKS/%s/' % ','.join('%d' % len(l) for l in links)]
        aux_rows = padded([list(a) for a in auxs], naux, None)
        lines.append('DATA PWS_LABEL/%s/' % ','.join(
            "'%s'" % (a['label'] if a else '  ') for row in aux_rows for a in row))
        lines.append('DATA PWS_CNORM/%s/' % col(a['cnorm'] if a else 0 for row in aux_rows for a in row))
        link_rows = padded(links, nlinks, (0, 0, 0, {}))
        for field, name in ((0, 'AUX'), (1, 'LEG'), (2, 'KIND')):
            lines.append('DATA PWS_%s/%s/' % (name, ','.join('%d' % l[field] for row in link_rows
                                                            for l in row)))
        ddp = dd or [dict(label1='  ', label2='  ', kind1=0, kind2=0, cnorm1=0, cnorm2=0,
                          amp_s=[], nflow=0, x={})]
        lines.append('DATA PDD_LABEL1/%s/' % ','.join("'%s'" % d['label1'] for d in ddp))
        lines.append('DATA PDD_LABEL2/%s/' % ','.join("'%s'" % d['label2'] for d in ddp))
        lines.append('DATA PDD_KIND1/%s/' % ','.join('%d' % d['kind1'] for d in ddp))
        lines.append('DATA PDD_KIND2/%s/' % ','.join('%d' % d['kind2'] for d in ddp))
        lines.append('DATA PDD_CNORM1/%s/' % col(d['cnorm1'] for d in ddp))
        lines.append('DATA PDD_CNORM2/%s/' % col(d['cnorm2'] for d in ddp))
        lines.append('C     the colour flows of the S-wave states and of the dipole-dipole')
        lines.append('C     terms (ONIA_PWS_FLOWS)')
        bases = [a for row in aux_rows for a in row] + ddp
        lines.extend(loop_onia.flow_map_lines('PWS_SF', [b['amp_s'] if b else [] for b in bases],
                                              [b['nflow'] if b else 0 for b in bases]))
        lines.extend(loop_onia.flow_matrix_lines(
            'PWS_FX', [l[3] for row in link_rows for l in row], (nflow, nfmax, nlinks, 2),
            'X between the Born flows and those of the S-wave state, per link'))
        lines.extend(loop_onia.flow_matrix_lines(
            'PDD_FX', [d['x'] for d in ddp], (nflow, nfmax, ndd),
            'X between the Born flows and those of the dipole-dipole term'))
        writers.FortranWriter('onia_loop_pwsoft.inc').writelines(lines)

    def onia_public_entries(self, file, matrix_element):
        """loop_matrix.f of a process with bound states: MadLoop runs on the
        constituent legs and the public SLOOPMATRIX of onia_loop.f takes the
        physical momenta, so the core routine is SLOOPMATRIX_CONSTITUENTS and
        the public wrappers (SLOOPMATRIX_THRES, SLOOPMATRIXHEL(_THRES)) take
        one leg per bound state."""
        prefix = matrix_element.rep_dict['proc_prefix']
        nexternal = matrix_element.get_nexternal_ninitial()[0]
        nphys = matrix_element.get_physical_nexternal_ninitial()[0]
        core = '      SUBROUTINE %sSLOOPMATRIX(P_USER,ANS)\n' % prefix
        if file.count(core) not in (1, 2):
            raise loop_onia.OniaLoopError('Unexpected loop_matrix.f: core SLOOPMATRIX')
        file = file.replace(core, '      SUBROUTINE %sSLOOPMATRIX_CONSTITUENTS(P_USER,ANS)\n' % prefix)
        parameter = 'PARAMETER (NEXTERNAL=%d)' % nexternal
        for wrapper in ['SLOOPMATRIXHEL(P,HEL,ANS)',
                        'SLOOPMATRIXHEL_THRES(P,HEL,ANS,PREC_ASKED,PREC_FOUND,RET_CODE)',
                        'SLOOPMATRIX_THRES(P,ANS,PREC_ASKED,PREC_FOUND,RET_CODE)']:
            head = '      SUBROUTINE %s%s' % (prefix, wrapper)
            if file.count(head) != 1:
                raise loop_onia.OniaLoopError('Unexpected loop_matrix.f: %s' % wrapper)
            i = file.index(head)
            j = file.index(parameter, i)
            file = file[:j] + 'PARAMETER (NEXTERNAL=%d)' % nphys + file[j + len(parameter):]
        return file

    def write_bornmatrix(self, writer, matrix_element, fortran_model):
        """born_matrix.f. With bound states it is the Born of the constituent
        process, not projected (the projected Born is in onia_loop.f): the
        'onium' attributes are dropped from a copy of the matrix element."""

        if not loop_onia.has_onia(matrix_element):
            return super(LoopProcessOptimizedExporterFortranSA, self).write_bornmatrix(
                                              writer, matrix_element, fortran_model)
        constituent_me = copy.copy(matrix_element)
        for prop in ['diagrams', 'processes']:
            constituent_me.set(prop, copy.deepcopy(matrix_element.get(prop)), force=True)
        for diagram in constituent_me.get('diagrams'):
            for wf in diagram.get('wavefunctions'):
                wf.set('onium', {})
            for amp in diagram.get_regular_amplitudes() if hasattr(diagram, 'get_regular_amplitudes') else diagram.get('amplitudes'):
                for wf in amp.get('mothers'):
                    if wf.get('onium'):
                        wf.set('onium', {})
        for proc in constituent_me.get('processes'):
            for leg in proc.get('legs'):
                leg.set('onium', {})
        res = super(LoopProcessOptimizedExporterFortranSA, self).write_bornmatrix(
                                              writer, constituent_me, fortran_model)
        nsqso = open(pjoin(os.path.dirname(os.path.abspath(writer.name)),
                           'nsqso_born.inc')).read()
        if 'NSQSO_BORN=1)' not in nsqso.replace(' ', ''):
            raise loop_onia.OniaLoopError('Processes with bound states and more than '
                         'one squared split order in the Born are not supported.')
        return res

    def write_pmass_file(self, writer, matrix_element):
        """pmass.inc of the physical legs of the check drivers: a bound state
        has the mass m1+m2 of its constituents (MadLoop's onia layer puts them
        at zero relative momentum)."""

        if not loop_onia.has_onia(matrix_element):
            return super(LoopProcessOptimizedExporterFortranSA, self).write_pmass_file(
                                                             writer, matrix_element)
        info = loop_onia.onia_loop_info(matrix_element)
        model = matrix_element.get('processes')[0].get('model')
        masses = []
        for wf in sorted(matrix_element.get_external_wavefunctions(),
                         key=lambda wf: wf.get('number_external')):
            mass = model.get('particle_dict')[wf.get('pdg_code')].get('mass')
            masses.append('ZERO' if mass.lower() == 'zero' else 'ABS(%s)' % mass)
        phys = {}
        for i, m in enumerate(masses):
            phys.setdefault(info['mapping'][i], []).append(m)
        lines = []
        for leg in sorted(phys):
            ms = [m for m in phys[leg] if m != 'ZERO'] or ['ZERO']
            lines.append('PMASS(%d)=%s' % (leg, '+'.join(ms)))
        writer.writelines(lines)
        return True

    def get_context(self,matrix_element, **opts):
        """ Additional contextual information which needs to be created for
        the optimized output."""
        
        context = LoopProcessExporterFortranSA.get_context(self, matrix_element, 
                                                                         **opts)

        # For now assume Ninja always supports quadruple precision
        try:
            context['ninja_supports_quad_prec'] = \
                     misc.get_ninja_quad_prec_support(getattr(self,'ninja_dir'))
        except AttributeError:
            context['ninja_supports_quad_prec'] = False

        context['ninjars_supports_quad_prec'] = (
            self.tir_available_dict.get('ninjars', False) and
            ninjars_install.supports_quad_precision(getattr(self, 'ninjars_dir', '')))

        context['tirust_supports_quad_prec'] = (
            self.tir_available_dict.get('tirust', False) and
            tirust_install.supports_quad_precision(getattr(self, 'tirust_dir', '')))

        context['tirust_dp_route_statistics_available'] = (
            self.tir_available_dict.get('tirust', False) and
            tirust_install.supports_dp_route_statistics(
                getattr(self, 'tirust_dir', '')))

        # optional raised propagator powers (P-wave bound states)
        context['tirust_raised_powers'] = (
            self.tir_available_dict.get('tirust', False) and
            tirust_install.supports_raised_powers(getattr(self, 'tirust_dir', '')))

        # The extended-precision passes of one P-wave bound state
        # (#PWaveExtendedPrecision, onia_ep.py): standalone output, TIRust
        # with extended inputs (the EP integrals) and quadruple precision
        # (the MP code the EP code is derived from)
        context['OniaPwaveEP'] = bool(
            type(self) is LoopProcessOptimizedExporterFortranSA and
            context.get('OniaPwave') and not context.get('OniaPwave2') and
            not context.get('AmplitudeReduction') and
            context['tirust_supports_quad_prec'] and context['tirust_raised_powers'] and
            onia_ep.tirust_supports_extended_inputs(getattr(self, 'tirust_dir', '')))

        for tir in self.all_tir:
            context['%s_available'%tir]=self.tir_available_dict[tir]
            # safety check
            if tir not in ['golem','pjfry','iregi','samurai','ninja','collier','ninjars','tirust']:
                raise MadGraph5Error("%s was not a TIR currently interfaced."%tir_name)

        return context

    def loop_optimized_additional_template_setup(self):
        """ Perform additional actions specific for this class when setting
        up the template with the copy_template function."""
        
        # We must link the TIR to the Library folder of the active Template
        link_tir_libs=[]
        tir_libs=[]
        tir_include=[]
        
        for tir in self.all_tir:
            tir_dir="%s_dir"%tir
            libpath=getattr(self,tir_dir)
            libname="lib%s.a"%tir
            tir_name=tir
            libpath = self.link_TIR(os.path.join(self.dir_path, 'lib'),
                                              libpath,libname,tir_name=tir_name)
            if tir in ('ninjars', 'tirust'):
                setattr(self, tir + '_dir', libpath)
            if libpath != "":
                if tir in ['ninja','pjfry','golem','samurai','collier','ninjars','tirust']:
                    # It is cleaner to use the original location of the libraries
                    copied_ninjars = tir in ('ninjars', 'tirust') and self.dependencies == 'internal'
                    if tir == 'ninjars':
                        link_tir_libs.extend(ninjars_install.pre_link_flags(libpath))
                    if tir == 'tirust':
                        link_tir_libs.extend(tirust_install.pre_link_flags(libpath))
                    if copied_ninjars:
                        link_tir_libs.append('-L$(LIBDIR) -l' + tir)
                        tir_libs.append('$(LIBDIR)lib' + tir + '.a')
                    else:
                        link_tir_libs.append('-L%s/ -l%s'%(libpath,tir))
                        tir_libs.append('%s/lib%s.%s' % (
                            libpath, tir, 'a' if tir in ('ninjars', 'tirust') else '$(libext)'))
                    if tir == 'ninjars':
                        link_tir_libs.extend(ninjars_install.native_link_flags(libpath))
                    if tir == 'tirust':
                        link_tir_libs.extend(tirust_install.native_link_flags(libpath))
                        # Executables live in SubProcesses or its process directories;
                        # shared process libraries live in lib.
                        link_tir_libs.extend(tirust_install.runtime_search_flags(
                            libpath, copied_ninjars))
                    # For Ninja, we must also link against OneLoop.
                    if tir in ['ninja']:
                        if not any(os.path.isfile(pjoin(libpath,'libavh_olo.%s'%ext)) 
                                              for ext in ['a','dylib','so']):
                            raise MadGraph5Error(
"The OneLOop library 'libavh_olo.(a|dylib|so)' could no be found in path '%s'. Please place a symlink to it there."%libpath)
                        link_tir_libs.append('-L%s/ -l%s'%(libpath,'avh_olo'))
                        tir_libs.append('%s/lib%s.$(libext)'%(libpath,'avh_olo'))
                    if tir in ['ninja','golem', 'samurai','collier','ninjars','tirust']:
                        trgt_path = pjoin(os.path.dirname(libpath),'include')
                        if os.path.isdir(trgt_path):
                            to_include = misc.find_includes_path(trgt_path,
                                                        self.include_names[tir])
                        else:
                            to_include = None
                        if to_include is None and tir in ('ninjars', 'tirust'):
                            to_include = misc.find_includes_path(
                                libpath, self.include_names[tir])
                        # Special possible location for collier
                        if to_include is None and tir=='collier':
                            to_include = misc.find_includes_path(
                               pjoin(libpath,'modules'),self.include_names[tir])
                        if to_include is None:
                            logger.error(
'Could not find the include directory for %s, looking in %s.\n' % (tir, str(trgt_path))+
'Generation carries on but you will need to edit the include path by hand in the makefiles.')
                            to_include = '<Not_found_define_it_yourself>'                
                        tir_include.append('-I %s' % (
                            '$(LIBDIR)' if copied_ninjars else str(to_include)))
                        # To be able to easily compile a MadLoop library using
                        # makefiles built outside of the MadGraph7 framework
                        # (such as what is done with the Sherpa interface), we
                        # place here an easy handle on the golem includes
                        name_map = {'golem':'golem95','samurai':'samurai',
                                    'ninja':'ninja','collier':'collier','ninjars':'ninjars','tirust':'tirust'}
                        if not copied_ninjars:
                            ln(to_include, starting_dir=pjoin(self.dir_path,'lib'),
                                       name='%s_include'%name_map[tir],abspath=True)
                            ln(libpath, starting_dir=pjoin(self.dir_path,'lib'),
                                           name='%s_lib'%name_map[tir],abspath=True)
                else :
                    link_tir_libs.append('-l%s'%tir)
                    tir_libs.append('$(LIBDIR)lib%s.$(libext)'%tir)

        MadLoop_makefile_definitions = pjoin(self.dir_path,'SubProcesses',
                                                 'MadLoop_makefile_definitions')
        if os.path.isfile(MadLoop_makefile_definitions):
            os.remove(MadLoop_makefile_definitions)

        calls = self.write_loop_makefile_definitions(
                        writers.MakefileWriter(MadLoop_makefile_definitions),
                                link_tir_libs,tir_libs, tir_include=tir_include)

        # Finally overwrite MadLoopCommons.f now that we know the availability of
        # COLLIER.
        MadLoopCommon = open(os.path.join(self.loop_dir,'StandAlone', 
                                    "SubProcesses","MadLoopCommons.inc")).read()
        writer = writers.FortranWriter(os.path.join(self.dir_path, 
                                             "SubProcesses","MadLoopCommons.f"))
        writer.writelines(MadLoopCommon%{
          'print_banner_commands':self.MadLoop_banner}, context={
                'collier_available':self.tir_available_dict['collier']})
        writer.close()

    def link_files_from_Subprocesses(self,proc_name):
        """ Does the same as the mother routine except that it also links
        coef_specs.inc in the HELAS folder."""

        LoopProcessExporterFortranSA.link_files_from_Subprocesses(self,proc_name)
        
        # Link the coef_specs.inc for aloha to define the coefficient
        # general properties (of course necessary in the optimized mode only)
        ln(os.path.join(self.dir_path,'Source','DHELAS','coef_specs.inc'),
           os.path.join(self.dir_path, 'SubProcesses', proc_name),
           abspath=False, cwd=None)


    def link_TIR(self, targetPath,libpath,libname,tir_name='TIR'):
        """Link the TIR source directory inside the target path given
        in argument"""
        
        # PJFry's reduction ID is retired. Keep historical source helpers,
        # but never link or advertise it in a newly generated process.
        if tir_name == 'pjfry':
            self.tir_available_dict['pjfry'] = False
            return ""
        if tir_name in ['golem','samurai','ninja','collier','ninjars','tirust']:
            # not self-contained libraries
            if (not isinstance(libpath,str)) or (not os.path.exists(libpath)) \
            or (not os.path.isfile(pjoin(libpath,libname))):
                if isinstance(libpath,str) and libpath != '' and \
                (not os.path.isfile(pjoin(libpath,libname))):
                    # WARNING ONLY appears when the libpath is a wrong specific path.
                    logger.warning("The %s reduction library could not be found"%tir_name\
                                   +" with PATH:%s specified in mg7_configuration.txt."%libpath\
                                   +" It will not be available.")
                self.tir_available_dict[tir_name]=False
                return ""
            if tir_name == 'ninjars':
                if not ninjars_install.is_available(libpath):
                    logger.warning("NinjaRS installation at %s is unusable; "
                                   "run 'install ninjars --update'. It will not be available.", libpath)
                    self.tir_available_dict[tir_name] = False
                    return ""
            if tir_name == 'tirust' and not tirust_install.is_available(libpath):
                logger.warning("TIRust installation at %s is unusable; run "
                               "'install tirust --update'. It will not be available.", libpath)
                self.tir_available_dict[tir_name] = False
                return ""
            # Check the version of the tool, if the library was found
            if tir_name in ['ninja','samurai'] and self.tir_available_dict[tir_name]:
                # Make sure the librry was indeed installed in the source directory
                # of the tool, of course this check doesn't make sense.
                if os.path.isfile(pjoin(libpath,os.pardir,'AUTHORS')):
                    try:
                        version = open(pjoin(libpath,os.pardir,'VERSION'),'r').read()
                    except IOError:
                        version = None
                    if version is None :
                        logger.warning(
"Your version of '%s' in \n  %s\nseems too old %sto be compatible with MadGraph7."
%(tir_name, libpath ,'' if not version else '(v%s) '%version)+
("\nConsider updating it by hand or using the 'install' function of MadGraph7." if tir_name!='samurai'
 else "\nAsk the authors for the latest version compatible with MadGraph7."))
        else:
            # self-contained libraries
            if (not isinstance(libpath,str)) or (not os.path.exists(libpath)):
                # WARNING ONLY appears when the libpath is a wrong specific path.
                logger.warning("The %s reduction library could not be found"%tir_name\
                                   +" with PATH:%s specified in mg7_configuration.txt."%libpath\
                                   +" It will not be available.")
                self.tir_available_dict[tir_name]=False
                return ""
       
        if self.dependencies=='internal':
            if tir_name in ('ninjars', 'tirust'):
                os.makedirs(targetPath, exist_ok=True)
                installer = ninjars_install if tir_name == 'ninjars' else tirust_install
                # Include the installed ABI, runtime and its complete dependency
                # notices/provenance so internal output stays self-contained.
                for artifact in installer.export_files(libpath):
                    source = artifact if os.path.isabs(artifact) else pjoin(libpath, artifact)
                    destination = pjoin(targetPath, os.path.basename(source))
                    if os.path.abspath(source) != os.path.abspath(destination):
                        shutil.copy2(source, destination)
                self.tir_available_dict[tir_name] = True
                return os.path.abspath(targetPath)
            if tir_name in ['pjfry','golem','samurai','ninja','collier']:
                self.tir_available_dict[tir_name]=False
                logger.info("When using the 'output_dependencies=internal' "+\
" MadGraph7 option, the (optional) reduction library %s cannot be employed because"%tir_name+\
" it is not distributed with the MadGraph7 code so that it cannot be copied locally.")
                return ""
            elif tir_name == "iregi":
                # This is the right paths for IREGI
                new_iregi_path = pjoin(targetPath,os.path.pardir,'Source','IREGI')
                misc.copytree(pjoin(libpath,os.path.pardir), new_iregi_path, 
                                                                  symlinks=True)
                
                current = misc.detect_current_compiler(
                                 pjoin(new_iregi_path,'src','makefile_ML5_lib'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                        self.fortran_compiler
                if current != new:
                    misc.mod_compilator(pjoin(new_iregi_path,'src'), new,current)
                    misc.mod_compilator(pjoin(new_iregi_path,'src','oneloop'), 
                                                                   new, current)

                # Create the links to the lib folder
                ln(pjoin(targetPath,os.path.pardir,'Source','IREGI','src',
                                                            libname),targetPath)
            else:
                logger.info("Tensor integral reduction library "+\
                                            "%s not implemented yet."%tir_name)
            return libpath
 
        elif self.dependencies=='external':
            if not os.path.exists(pjoin(libpath,libname)) and tir_name=='iregi':
                        # special mode
                if 'heptools_install_dir' in self.opt and os.path.exists(pjoin(self.opt['heptools_install_dir'], 'IREGI')):
                    misc.sprint('Going to use pre-compiled version of IREGI') 
                    # Create the links to the lib folder
                    ln(os.path.join(self.opt['heptools_install_dir'],'IREGI','src','libiregi.a'),
                                    os.path.join(targetPath),abspath=True)
                    return os.path.join(targetPath, 'libiregi.a')
                
                
                
                logger.info('Compiling IREGI. This has to be done only once and'+\
                             ' can take a couple of minutes.','$MG:BOLD')
                
                current = misc.detect_current_compiler(os.path.join(\
                                                    libpath,'makefile_ML5_lib'))
                new = 'gfortran' if self.fortran_compiler is None else \
                                                        self.fortran_compiler
                if current != new:
                    misc.mod_compilator(libpath, new,current)
                    misc.mod_compilator(pjoin(libpath,'oneloop'), new, current)

                misc.compile(cwd=libpath, job_specs = False)

                if not os.path.exists(pjoin(libpath,libname)):            
                    logger.warning("IREGI could not be compiled. Check"+\
                      "the compilation errors at %s. The related "%libpath+\
                                              "functionalities are turned off.")
                    self.tir_available_dict[tir_name]=False
                    return ""
            # We link the tools below directly to the lib directory of the output 
            if not tir_name in ['pjfry','golem','samurai','ninja','collier','ninjars','tirust']:
                ln(os.path.join(libpath,libname),targetPath,abspath=True)

        elif self.dependencies=='environment_paths':
            # Here the user chose to define the dependencies path in one of 
            # his environmental paths
            newlibpath = misc.which_lib(libname)
            if not newlibpath is None:
                logger.info('MadGraph7 is using %s installation found at %s.'%\
                                                          (tir_name,newlibpath)) 
                # We link the tools below directly to directly where the library is detected
                if not tir_name in ['pjfry','golem','samurai','ninja','collier','ninjars','tirust']:
                    ln(newlibpath,targetPath,abspath=True)
                self.tir_available_dict[tir_name]=True
                return os.path.dirname(newlibpath)
            else:
                logger.warning("Could not find the location of the file"+\
                  " %s in you environment paths. The related "%libname+\
                                             "functionalities are turned off.")
                self.tir_available_dict[tir_name]=False
                return ""
            
        self.tir_available_dict[tir_name]=True
        return libpath
    
    def set_group_loops(self, matrix_element):
        """ Decides whether we must group loops or not for this matrix element"""

        # Decide if loops sharing same denominator structures have to be grouped
        # together or not.
        if self.forbid_loop_grouping:
            self.group_loops = False
        else:
            self.group_loops = (not self.get_context(matrix_element)['ComputeColorFlows'])\
                          and matrix_element.get('processes')[0].get('has_born')
        
        return self.group_loops
    
    def make(self):
        """Compile Source, its DHELAS and MODEL libraries at -O2 (identical
        results without -march or fast-math options, faster loop evaluation)
        for the plain standalone output, whose make_opts has no -O flag. The
        DP-terminal proof admits exactly these flags (dp_terminal_proof)."""
        if type(self) is LoopProcessOptimizedExporterFortranSA:
            self.set_library_optimization()
        super(LoopProcessOptimizedExporterFortranSA, self).make()

    def set_library_optimization(self, flag='-O2'):
        """ALOHA_FLAG (read by the DHELAS makefile) and MODEL_FLAG (MODEL
        makefile) in Source/make_opts."""
        make_opts = pjoin(self.dir_path, 'Source', 'make_opts')
        if os.path.exists(make_opts):
            from madgraph.interface.common_run_interface import CommonRunCmd
            CommonRunCmd.update_make_opts_full(
                make_opts, {'ALOHA_FLAG': flag, 'MODEL_FLAG': flag})

    def finalize(self, matrix_element, cmdhistory, MG5options, outputflag):
        """create the global information for loops"""
        
        super(LoopProcessOptimizedExporterFortranSA,self).finalize(matrix_element,
                                             cmdhistory, MG5options, outputflag)
        self.write_global_specs(matrix_element)
        # Model, ALOHA and shared include sources now exist. Until this final
        # rendered proof succeeds, every DP batch retains the ordinary call.
        self.finalize_dp_terminal_updates()
        self.write_onia_ep_files()
        if 2 in getattr(self, 'onia_npwave', set()):
            self.set_onia_pw2_code_model()

    ONIA_PW2_CODE_MODEL = (
        '# Two P-wave bound states: the loop polynomials of their derivatives can\n'
        '# exceed the 2 GB of static data of the default x86-64 code model\n'
        'ifeq ($(shell uname -m),x86_64)\n'
        'FFLAGS += -mcmodel=medium\n'
        'endif\n')

    def set_onia_pw2_code_model(self):
        """The subprocesses of two P-wave bound states are built with the
        medium code model on x86-64: their loop wavefunctions and those of
        their derivatives (four polynomials of rank R+2 per wavefunction, in
        double and in quadruple precision) can exceed 2 GB of static data,
        e.g. for g g > chic0 chic0."""
        path = pjoin(self.dir_path, 'SubProcesses', 'makefile')
        with open(path) as stream:
            text = stream.read()
        old = 'include $(ROOT)/Source/make_opts\n'
        if self.ONIA_PW2_CODE_MODEL in text:
            return
        if text.count(old) != 1:
            raise MadGraph5Error('Unexpected SubProcesses makefile')
        with open(path, 'w') as stream:
            stream.write(text.replace(old, old + self.ONIA_PW2_CODE_MODEL))

    def write_onia_ep_files(self):
        """The extended-precision passes of the processes with a P-wave bound
        state (#PWaveExtendedPrecision): their EP files, derived from the MP
        ones now that all are written (onia_ep.py), and their build."""
        ep_dirs = getattr(self, 'onia_ep_dirs', {})
        if not ep_dirs:
            return
        for pdir, proc_prefix in sorted(ep_dirs.items()):
            onia_ep.derive_subprocess(pdir, proc_prefix)
        onia_ep.patch_subprocesses_makefile(pjoin(self.dir_path, 'SubProcesses', 'makefile'))
        logger.info('Extended-precision passes of the P-wave bound state written '
                    '(off by default: #PWaveExtendedPrecision in MadLoopParams.dat).')
    

    
    def write_loop_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        group_number = None, proc_id = None, config_map = None):
        """ Writes loop_matrix.f, CT_interface.f,TIR_interface.f,GOLEM_inteface.f 
        and loop_num.f only but with the optimized FortranModel.
        The arguments group_number and proc_id are just for the LoopInduced
        output with MadEvent and only used in get_ME_identifier."""
                
        # Warn the user that the 'matrix' output where all relevant code is
        # put together in a single file is not supported in this loop output.
        if writer:
            raise MadGraph5Error('Matrix output mode no longer supported.')
        
        if not isinstance(fortran_model,\
          helas_call_writers.FortranUFOHelasCallWriter):
            raise MadGraph5Error('The optimized loop fortran output can only'+\
              ' work with a UFO Fortran model')
        OptimizedFortranModel=\
          helas_call_writers.FortranUFOHelasCallWriterOptimized(\
          fortran_model.get('model'),False)


        if not matrix_element.get('processes')[0].get('has_born') and \
                                                   not self.compute_color_flows:
            logger.debug("Color flows will be employed despite the option"+\
              " 'loop_color_flows' being set to False because it is necessary"+\
                                                          " for optimizations.")

        # Compute the analytical information of the loop wavefunctions in the
        # loop helas matrix elements using the cached aloha model to reuse
        # as much as possible the aloha computations already performed for
        # writing out the aloha fortran subroutines.
        matrix_element.compute_all_analytic_information(
          self.get_aloha_model(matrix_element.get('processes')[0].get('model')))

        self.set_group_loops(matrix_element)

        # Initialize a general replacement dictionary with entries common to 
        # many files generated here.
        matrix_element.rep_dict = LoopProcessExporterFortranSA.\
                        generate_general_replace_dict(self, matrix_element, 
                                 group_number = group_number, proc_id = proc_id)

        # and those specific to the optimized output
        self.set_optimized_output_specific_replace_dict_entries(matrix_element)

        # Create the necessary files for the loop matrix element subroutine      
        proc_prefix_writer = writers.FortranWriter('proc_prefix.txt','w')
        proc_prefix_writer.write(matrix_element.rep_dict['proc_prefix'])
        proc_prefix_writer.close()
                    
        filename = 'loop_matrix.f'
        calls = self.write_loopmatrix(writers.FortranWriter(filename),
                                      matrix_element,
                                      OptimizedFortranModel)    
        
        filename = 'check_sa.f'
        self.write_check_sa(writers.FortranWriter(filename),matrix_element)
        
        filename = 'polynomial.f'
        calls = self.write_polynomial_subroutines(
                                      writers.FortranWriter(filename),
                                      matrix_element)
        
        filename = 'improve_ps.f'
        calls = self.write_improve_ps(writers.FortranWriter(filename),
                                                                 matrix_element)
        
        filename = 'CT_interface.f'
        self.write_CT_interface(writers.FortranWriter(filename),\
                                matrix_element)
        
        filename = 'TIR_interface.f'
        self.write_TIR_interface(writers.FortranWriter(filename),
                                matrix_element)
        
        if 'golem' in self.tir_available_dict and self.tir_available_dict['golem']:
            filename = 'GOLEM_interface.f'
            self.write_GOLEM_interface(writers.FortranWriter(filename),
                                       matrix_element)

        if 'collier' in self.tir_available_dict and self.tir_available_dict['collier']:
            filename = 'COLLIER_interface.f'
            self.write_COLLIER_interface(writers.FortranWriter(filename),
                                       matrix_element)

        if self.tir_available_dict.get('tirust', False):
            self.write_TIRUST_interface(writers.FortranWriter('TIRUST_interface.f'),
                                        matrix_element)

        filename = 'loop_num.f'
        self.write_loop_num(writers.FortranWriter(filename),\
                                           matrix_element,OptimizedFortranModel)
        
        filename = 'mp_compute_loop_coefs.f'
        self.write_mp_compute_loop_coefs(writers.FortranWriter(filename),\
                                     matrix_element,OptimizedFortranModel)

        if self.get_context(matrix_element)['ComputeColorFlows']:
            filename = 'compute_color_flows.f'
            self.write_compute_color_flows(writers.FortranWriter(filename),
                                        matrix_element, config_map = config_map)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                                                            nexternal, ninitial)
        
        # Write general process information                        
        filename = 'process_info.inc'
        self.write_process_info_file(writers.FortranWriter(filename),
                                                                 matrix_element)

        if self.get_context(matrix_element)['TIRCaching']:
            filename = 'tir_cache_size.inc'
            self.write_tir_cache_size_include(writers.FortranWriter(filename))

        # The object graph is necessary but not sufficient: a custom template
        # or writer may introduce another COMMON consumer. Fail closed by
        # regenerating the ordinary MP batches if the rendered proof fails.
        terminal = OptimizedFortranModel.get_terminal_wavefunctions(matrix_element)
        if terminal and not self.validate_terminal_wavefunctions(
                '.', matrix_element.rep_dict['proc_prefix'], terminal):
            logger.debug("Keeping full MP updaters: generated terminal proof unavailable")
            self.write_mp_compute_loop_coefs(writers.FortranWriter('mp_compute_loop_coefs.f'),
                matrix_element, OptimizedFortranModel, _terminal=False)

        dp_terminal = self.get_dp_terminal_wavefunctions(matrix_element)
        if dp_terminal:
            pending = getattr(self, '_dp_terminal_pending', {})
            pending[os.path.abspath('.')] = (
                matrix_element.rep_dict['proc_prefix'], dp_terminal)
            self._dp_terminal_pending = pending

        return calls

    def get_dp_terminal_wavefunctions(self, matrix_element):
        """Select the qualified DP diagonal rank/shape pairs.

        Only the layouts the rendered proof knows: standalone and MC@NLO
        output of processes with a Born. Subclasses with other templates or
        build recipes keep the dense updaters.
        """
        from madgraph.iolibs import export_fks
        if (type(self) not in (LoopProcessOptimizedExporterFortranSA,
                               export_fks.ProcessOptimizedExporterFortranFKS) or
                loop_onia.has_pwave(matrix_element) or
                self.opt.get('vector_size') or
                self.get_context(matrix_element)['MadEventOutput'] or
                not matrix_element.get('processes')[0].get('has_born') or
                matrix_element.rep_dict.get('max_lwf_size') != 4 or
                len(matrix_element.rep_dict['proc_prefix']+'DP_TERMINAL_WL_1_1') > 63):
            return {}
        terminal = helas_call_writers.FortranUFOHelasCallWriterOptimized.\
            get_terminal_wavefunctions(matrix_element)
        rank = matrix_element.get_max_loop_rank()
        vertex_rank = matrix_element.get_max_loop_vertex_rank()
        return {number: info for number, info in terminal.items()
                if info['vertex_rank'] == 1 and
                (info['loop_mother_rank'] == 1 or
                 (info['loop_mother_rank'] in (2, 3) and vertex_rank == 1 and
                  info['loop_mother_rank'] + 1 <= rank <= 6)) and
                info['lcut_size'] == info['in_size'] == info['out_size'] == 4}

    # The IEEE environment of the private DP updaters (nearest rounding, no
    # halting exception), checked once per coefficient batch, before its first
    # private call. Each call ANDs it with the sticky INEXACT flag captured
    # just before it. dp_terminal_proof certifies these exact statements.
    DP_TERMINAL_MODE_DECLARATIONS = (
        'LOGICAL WL_MODES_OK', 'INTEGER WL_H',
        'LOGICAL WL_HALTING(SIZE(IEEE_ALL))', 'TYPE(IEEE_ROUND_TYPE) WL_ROUNDING')
    DP_TERMINAL_MODE_BLOCK = (
        'WL_MODES_OK=.FALSE.',
        'CALL IEEE_GET_ROUNDING_MODE(WL_ROUNDING)',
        'IF(WL_ROUNDING.EQ.IEEE_NEAREST) THEN',
        'WL_MODES_OK=.TRUE.',
        'DO WL_H=1,SIZE(IEEE_ALL)',
        'IF(.NOT.IEEE_SUPPORT_HALTING(IEEE_ALL(WL_H))) WL_MODES_OK=.FALSE.',
        'ENDDO',
        'IF(WL_MODES_OK) THEN',
        'CALL IEEE_GET_HALTING_MODE(IEEE_ALL,WL_HALTING)',
        'IF(ANY(WL_HALTING)) WL_MODES_OK=.FALSE.',
        'ENDIF',
        'ENDIF')
    DP_TERMINAL_MODE_AND = 'WL_ENTRY_INEXACT=WL_ENTRY_INEXACT.AND.WL_MODES_OK'

    @staticmethod
    def prepare_dp_terminal_batch(source, prefix, expected):
        """Propose exact fixed-form call replacements, leaving other bytes intact.

        This is only a proposal: the independent full-source validator must
        accept all consumers and aliases before these private calls survive.
        """
        cls = LoopProcessOptimizedExporterFortranSA
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*|', prefix) or
                re.search(r'(?<![A-Z0-9_])(?:WL_ENTRY_INEXACT|WL_MODES_OK|WL_HALTING|'
                          r'WL_ROUNDING|WL_H)(?![A-Z0-9_])', source.upper())):
            raise ValueError('Unknown DP terminal batch declaration')
        wanted = {}
        for target, info in expected.items():
            if (info['loop_mother_rank'] not in (1, 2, 3) or info['vertex_rank'] != 1 or
                    any(info[key] != 4 for key in ('lcut_size', 'in_size', 'out_size'))):
                raise ValueError('Unsupported DP terminal shape')
            call = ('CALL '+prefix+'UPDATE_WL_{loop_mother_rank}_{vertex_rank}'
                    '(WL(1,0,1,{loop_mother_number}),'
                    '4,COEFS,4,4,WL(1,0,1,{out}))').format(**info)
            wanted[re.sub(r'\s+', '', call).upper()] = target
        physical = source.splitlines(True)
        blocks = []
        for index, raw in enumerate(physical):
            if not raw.strip() or raw[0] in 'cC*!':
                continue
            if len(raw.rstrip('\n')) > 132 or len(raw) < 6 or '\t' in raw[:6]:
                raise ValueError('Unknown DP fixed-form statement')
            text = re.sub(r'\s+', '', raw[6:].strip()).upper()
            if raw[5] not in ' 0':
                if not blocks:
                    raise ValueError('Orphan DP continuation')
                blocks[-1][1] = index+1
                blocks[-1][2] += text
            else:
                blocks.append([index, index+1, text])

        def fixed(statement):
            return '      '+statement[:66]+'\n'+''.join(
                '     $'+statement[i:i+66]+'\n' for i in range(66, len(statement), 66))

        replacements, found = [], set()
        for start, end, text in blocks:
            if text not in wanted:
                continue
            target = wanted[text]
            if target in found or physical[start][:5].strip():
                raise ValueError('Duplicate or labeled DP terminal producer')
            if any(line[0] in 'cC*!' for line in physical[start:end]):
                raise ValueError('Comment inside DP terminal call')
            found.add(target)
            private = text.replace('CALL', 'CALL ', 1).replace(
                'UPDATE_WL_', 'DP_TERMINAL_WL_', 1)[:-1]+',WL_ENTRY_INEXACT)'
            replacements.append((start, end,
                fixed('CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)')+
                fixed(cls.DP_TERMINAL_MODE_AND)+fixed(private)))
        if not found:
            return source, found
        # The modes are checked once, before the first private call
        replacements.sort()
        start, end, replacement = replacements[0]
        replacements[0] = (start, end, ''.join(fixed(statement) for statement in
                                               cls.DP_TERMINAL_MODE_BLOCK) + replacement)
        for start, end, replacement in reversed(replacements):
            physical[start:end] = [replacement]
        candidate = ''.join(physical)
        pattern = r'(?mi)^ {6}IMPLICIT\s+NONE[ \t]*\n'
        insertion = ('      USE, INTRINSIC :: IEEE_ARITHMETIC\n'
                     '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n'
                     '      IMPLICIT NONE\n      LOGICAL WL_ENTRY_INEXACT\n' +
                     ''.join('      %s\n' % declaration for declaration in
                             cls.DP_TERMINAL_MODE_DECLARATIONS))
        candidate, count = re.subn(pattern, lambda _: insertion, candidate)
        if count != 1:
            raise ValueError('Unknown DP terminal batch scope')
        return candidate, found

    def finalize_dp_terminal_updates(self):
        """Promote only complete rendered proofs; otherwise restore dense bytes."""
        pending = getattr(self, '_dp_terminal_pending', {})
        self._dp_terminal_pending = {}
        for directory, (prefix, expected) in sorted(pending.items()):
            original, proposals, selected = {}, {}, set()
            accepted = False
            try:
                from madgraph.loop.dp_terminal_proof import validate_dp_terminal_wavefunctions
                if os.path.realpath(directory) != os.path.abspath(directory):
                    raise ValueError('Aliased DP export directory')
                for filename in sorted(glob.glob(pjoin(directory, 'coef_construction_*.f'))):
                    if os.path.islink(filename) or not stat.S_ISREG(os.stat(filename).st_mode):
                        raise ValueError('Aliased or nonregular DP batch')
                    with open(filename, 'rb') as stream:
                        source_bytes = stream.read()
                    source = source_bytes.decode('utf-8')
                    candidate, found = self.prepare_dp_terminal_batch(source, prefix, expected)
                    if selected.intersection(found):
                        raise ValueError('Repeated DP terminal producer across batches')
                    selected.update(found)
                    if found:
                        original[filename], proposals[filename] = source_bytes, candidate.encode('utf-8')
                if selected != set(expected):
                    raise ValueError('Not every DP terminal producer was recognized')
                for filename, source in proposals.items():
                    with open(filename, 'wb') as stream:
                        stream.write(source)
                accepted = validate_dp_terminal_wavefunctions(directory, prefix, expected)
            except (ImportError, OSError, ValueError, KeyError, TypeError):
                accepted = False
            finally:
                if not accepted:
                    for filename, source in original.items():
                        with open(filename, 'wb') as stream:
                            stream.write(source)
            if not accepted:
                logger.debug('Keeping full DP updaters: generated terminal proof unavailable')

    def set_optimized_output_specific_replace_dict_entries(self, matrix_element):
        """ Specify the entries of the replacement dictionary which are specific
        to the optimized output and only relevant to it (the more general entries
        are set in the mother class LoopProcessExporterFortranSA."""
        
        # P-wave bound states: the derivative of a loop propagator gives the
        # integral of rank R+1 with that propagator doubled, and with two
        # P-wave states the mixed derivative ranks up to R+2
        max_loop_rank = matrix_element.get_max_loop_rank() + \
                        loop_onia.loop_rank_increase(matrix_element)
        matrix_element.rep_dict['maxrank']=max_loop_rank
        matrix_element.rep_dict['loop_max_coefs']=\
                        q_polynomial.get_number_of_coefs_for_rank(max_loop_rank)
        max_loop_vertex_rank=matrix_element.get_max_loop_vertex_rank()
        matrix_element.rep_dict['vertex_max_coefs']=\
                 q_polynomial.get_number_of_coefs_for_rank(max_loop_vertex_rank)
                 
        matrix_element.rep_dict['nloopwavefuncs']=\
                               matrix_element.get_number_of_loop_wavefunctions()
        max_spin=matrix_element.get_max_loop_particle_spin()

        matrix_element.rep_dict['max_lwf_size']= 4 if max_spin <=3 else 16
        matrix_element.rep_dict['nloops']=len(\
                        [1 for ldiag in matrix_element.get_loop_diagrams() for \
                                           lamp in ldiag.get_loop_amplitudes()])
        
        if self.set_group_loops(matrix_element):
            matrix_element.rep_dict['nloop_groups']=\
                                          len(matrix_element.get('loop_groups'))
        else:
            matrix_element.rep_dict['nloop_groups']=\
                                              matrix_element.rep_dict['nloops']
    
    def write_loop_num(self, writer, matrix_element,fortran_model):
        """ Create the file containing the core subroutine called by CutTools
        which contains the Helas calls building the loop"""

        replace_dict=copy.copy(matrix_element.rep_dict)

        file = open(os.path.join(self.template_dir,'loop_num.inc')).read()  
        file = file % replace_dict
        writer.writelines(file,context=self.get_context(matrix_element))

    def write_CT_interface(self, writer, matrix_element):
        """ We can re-use the mother one for the loop optimized output."""
        LoopProcessExporterFortranSA.write_CT_interface(\
                            self, writer, matrix_element,optimized_output=True)

    def write_TIR_interface(self, writer, matrix_element):
        """ Create the file TIR_interface.f which does NOT contain the subroutine
         defining the loop HELAS-like calls along with the general interfacing 
         subroutine. """

        # First write TIR_interface which interfaces MG5 with TIR.
        replace_dict=copy.copy(matrix_element.rep_dict)
            
        file = open(os.path.join(self.template_dir,'TIR_interface.inc')).read()  

        # Check which loops have an Higgs effective vertex so as to correctly 
        # implement CutTools limitation
        loop_groups = matrix_element.get('loop_groups')
        has_HEFT_vertex = [False]*len(loop_groups)
        for i, (denom_structure, loop_amp_list) in enumerate(loop_groups):
            for lamp in loop_amp_list:
                final_lwf = lamp.get_final_loop_wavefunction()
                while not final_lwf is None:
                    # We define here an HEFT vertex as any vertex built up from
                    # only massless vectors and massive scalars (at least one of each)
                    # We ask for massive scalars in part to remove the gluon ghost false positive.
                    scalars = len([1 for wf in final_lwf.get('mothers') if 
                                    wf.get('spin')==1 and wf.get('mass')!='ZERO'])
                    vectors = len([1 for wf in final_lwf.get('mothers') if 
                                  wf.get('spin')==3 and wf.get('mass')=='ZERO'])
                    if scalars>=1 and vectors>=1 and \
                               scalars+vectors == len(final_lwf.get('mothers')):
                        has_HEFT_vertex[i] = True
                        break
                    final_lwf = final_lwf.get_loop_mother()
                else:
                    continue
                break

        has_HEFT_list = []
        chunk_size = 9
        for k in range(0, len(has_HEFT_vertex), chunk_size):
            has_HEFT_list.append("DATA (HAS_AN_HEFT_VERTEX(I),I=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(has_HEFT_vertex)),
                     ','.join(['.TRUE.' if l else '.FALSE.' for l in 
                                           has_HEFT_vertex[k:k + chunk_size]])))
        replace_dict['has_HEFT_list'] = '\n'.join(has_HEFT_list)
        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = ''
        file = file % replace_dict
        
        FPR = q_polynomial.FortranPolynomialRoutines(
        replace_dict['maxrank'],coef_format=replace_dict['complex_dp_format'],\
                                         sub_prefix=replace_dict['proc_prefix'])
        if self.tir_available_dict['pjfry']:
            file += '\n\n'+FPR.write_pjfry_mapping()
        if self.tir_available_dict['iregi']:
            file += '\n\n'+FPR.write_iregi_mapping()

        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    def write_TIRUST_interface(self, writer, matrix_element):
        """Emit the DP/QP tensor adapters, static permutation and plan registry."""
        replace_dict = copy.copy(matrix_element.rep_dict)
        replace_dict['include_vector'] = (
            "include '../../Source/vector.inc'" if self.opt['vector_size'] else '')
        generator = q_polynomial.FortranPolynomialRoutines(replace_dict['maxrank'])
        permutation = generator.get_TIRUST_mapping()
        replace_dict['tirust_coefmap'] = '\n'.join(
            'DATA (TIRUST_MAP(I),I=%d,%d)/%s/' %
            (start, min(start + 9, len(permutation) - 1),
             ','.join(str(value) for value in permutation[start:start + 10]))
            for start in range(0, len(permutation), 10))
        plans = sorted(set((amp.get_call_key()[1],
                            amp.get_analytic_info('wavefunction_rank'))
                           for diagram in matrix_element.get_loop_diagrams()
                           for amp in diagram.get_loop_amplitudes()))
        replace_dict['tirust_plan_registration'] = '\n'.join(
            '      CALL TIRUST_REGISTER_PLAN(%d,%d,STATUS)\n'
            '      IF(STATUS.NE.TIRUST_SUCCESS) RETURN' % pair
            for pair in plans if 1 <= pair[0] <= 6 and 0 <= pair[1] <= 6)
        with open(os.path.join(self.template_dir, 'TIRUST_interface.inc')) as stream:
            source = stream.read() % replace_dict
        if writer:
            writer.writelines(source, context=self.get_context(matrix_element))
        else:
            return source

    def write_COLLIER_interface(self, writer, matrix_element):
        """ Create the file COLLIER_interface.f"""

        # First write GOLEM_interface which interfaces MG5 with TIR.
        replace_dict=copy.copy(matrix_element.rep_dict)
            
        file = open(os.path.join(self.template_dir,'COLLIER_interface.inc')).read()
 
        FPR = q_polynomial.FortranPolynomialRoutines(replace_dict['maxrank'],\
                                                    coef_format=replace_dict['complex_dp_format'],\
                                                    sub_prefix=replace_dict['proc_prefix'])
        map_definition = []
        collier_map = FPR.get_COLLIER_mapping()
        
        chunk_size = 10
        for map_name, indices_list in \
                           [('COEFMAP_ZERO',[c[0] for c in collier_map]),
                            ('COEFMAP_ONE',[c[1] for c in collier_map]),
                            ('COEFMAP_TWO',[c[2] for c in collier_map]),
                            ('COEFMAP_THREE',[c[3] for c in collier_map])]:
            for k in range(0, len(indices_list), chunk_size):
                map_definition.append("DATA (%s(I),I=%3r,%3r) /%s/" % \
                (map_name,k, min(k + chunk_size, len(indices_list))-1,
                 ','.join('%2r'%ind for ind in indices_list[k:k + chunk_size])))

        replace_dict['collier_coefmap'] = '\n'.join(map_definition) 

        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = ''

        file = file % replace_dict
        
        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    def write_GOLEM_interface(self, writer, matrix_element):
        """ Create the file GOLEM_interface.f which does NOT contain the subroutine
         defining the loop HELAS-like calls along with the general interfacing 
         subroutine. """

        # First write GOLEM_interface which interfaces MG5 with TIR.
        replace_dict=copy.copy(matrix_element.rep_dict)
        
        # We finalize TIR result differently whether we used the built-in
        # squaring against the born.
        if not self.get_context(matrix_element)['AmplitudeReduction']:
            replace_dict['loop_induced_sqsoindex']=',SQSOINDEX'
        else:
            replace_dict['loop_induced_sqsoindex']=''


        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = '' 
            
        file = open(os.path.join(self.template_dir,'GOLEM_interface.inc')).read()

        file = misc.apply_template(file, replace_dict)

        FPR = q_polynomial.FortranPolynomialRoutines(replace_dict['maxrank'],\
                                                    coef_format=replace_dict['complex_dp_format'],\
                                                    sub_prefix=replace_dict['proc_prefix'])
        
        file += '\n\n'+FPR.write_golem95_mapping()
        
        if writer:
            writer.writelines(file,context=self.get_context(matrix_element))
        else:
            return file

    def write_polynomial_subroutines(self,writer,matrix_element):
        """ Subroutine to create all the subroutines relevant for handling
        the polynomials representing the loop numerator """
        
        # First create 'loop_max_coefs.inc'
        IncWriter=writers.FortranWriter('loop_max_coefs.inc','w')
        IncWriter.writelines("""INTEGER LOOPMAXCOEFS
                           PARAMETER (LOOPMAXCOEFS=%(loop_max_coefs)d)"""
                                                       %matrix_element.rep_dict)
        
        # Then coef_specs directly in DHELAS if it does not exist already
        # 'coef_specs.inc'. If several processes exported different files there,
        # it is fine because the overall maximum value will overwrite it in the
        # end
        coef_specs_path = pjoin(self.dir_path, 'Source','DHELAS','coef_specs.inc')
        if not os.path.isfile(coef_specs_path):
            IncWriter=writers.FortranWriter(coef_specs_path,'w')
            IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %matrix_element.rep_dict)
            IncWriter.close()
        
        # List of all subroutines to place there
        subroutines=[]
        
        # Start from the routine in the template
        replace_dict = copy.copy(matrix_element.rep_dict)
                
        dp_routine = open(os.path.join(self.template_dir,'polynomial.inc')).read()
        mp_routine = open(os.path.join(self.template_dir,'polynomial.inc')).read()
        # The double precision version of the basic polynomial routines, such as
        # create_loop_coefs
        replace_dict['complex_format'] = replace_dict['complex_dp_format']
        replace_dict['real_format'] = replace_dict['real_dp_format']
        replace_dict['mp_prefix'] = ''
        replace_dict['kind'] = 8
        replace_dict['zero_def'] = '0.0d0'
        replace_dict['one_def'] = '1.0d0'
        dp_routine = dp_routine % replace_dict 
        # The quadruple precision version of the basic polynomial routines
        replace_dict['complex_format'] = replace_dict['complex_mp_format']
        replace_dict['real_format'] = replace_dict['real_mp_format']
        replace_dict['mp_prefix'] = 'MP_'
        replace_dict['kind'] = 16
        replace_dict['zero_def'] = '0.0e0_16'
        replace_dict['one_def'] = '1.0e0_16'
        mp_routine = mp_routine % replace_dict
        subroutines.append(dp_routine)
        subroutines.append(mp_routine)        

        # Initialize the polynomial routine writer at the rank of
        # LOOPMAXCOEFS (rep_dict['maxrank']): R+1 for a P-wave bound state,
        # whose doubled propagators have one more power of l, and R+2 with
        # two P-wave states (tripled lines and two doubled lines, l^mu l^nu)
        poly_rank = matrix_element.get_max_loop_rank() + \
                    loop_onia.loop_rank_increase(matrix_element)
        poly_writer=q_polynomial.FortranPolynomialRoutines(
            poly_rank,
            updater_max_rank = matrix_element.get_max_loop_vertex_rank(), 
            sub_prefix=replace_dict['proc_prefix'],
            proc_prefix=replace_dict['proc_prefix'],
            mp_prefix='')
        # Write the polynomial constant module common to all
        writer.writelines(poly_writer.write_polynomial_constant_module()+'\n')
        ep = self.get_context(matrix_element).get('OniaPwaveEP', False)
        if loop_onia.has_pwave(matrix_element):
            # A P-wave bound state: the dual numbers have default
            # initialisation and type-bound procedures, so cannot be in COMMON
            # blocks; the dual objects shared by the routines are in a module
            # (polynomial.f is compiled before the other files)
            writer.writelines('\n'.join([
                'MODULE %(proc_prefix)sPW_DUAL',
                'C     A P-wave bound state: the dual (value, derivative in the relative',
                'C     momentum) wavefunctions, Born and counterterm amplitudes and loop',
                'C     momenta shared by the routines of the process (onia_loop.f), and',
                'C     the dual constituent momenta of the current direction. They are',
                'C     allocatable: gfortran 15.2 re-applies the default initialisation',
                'C     of the dual numbers to use-associated (non-allocatable) arrays on',
                'C     entry of the procedures that use them.',
                'USE ALOHA_OBJECT',
                'IMPLICIT NONE',
                'TYPE(ALOHA), ALLOCATABLE :: PWW(:)',
                'TYPE(DUAL), ALLOCATABLE :: PWAMP(:), PWAMPL(:,:), PWPL(:,:), PWPD(:,:)',
                'C     the same in quadruple precision',
                'TYPE(MP_ALOHA), ALLOCATABLE :: MPPWW(:)',
                'TYPE(MP_DUAL), ALLOCATABLE :: MPPWAMP(:), MPPWAMPL(:,:)',
                'TYPE(MP_DUAL), ALLOCATABLE :: MPPWPL(:,:), MPPWPD(:,:)'] +
                (['C     and in extended precision (#PWaveExtendedPrecision)',
                  'TYPE(EP_ALOHA), ALLOCATABLE :: EPPWW(:)',
                  'TYPE(EP_DUAL), ALLOCATABLE :: EPPWAMP(:), EPPWAMPL(:,:)',
                  'TYPE(EP_DUAL), ALLOCATABLE :: EPPWPL(:,:), EPPWPD(:,:)']
                 if ep else []) + [
                'CONTAINS',
                'SUBROUTINE %(proc_prefix)sPW_DUAL_INIT()',
                'IF (ALLOCATED(PWW)) RETURN',
                'ALLOCATE(PWW(%(nwavefuncs)d), PWAMP(%(nbornamps)d), PWAMPL(3,%(nctamps)d))',
                'ALLOCATE(PWPL(0:3,-1:%(nloopwavefuncs)d), PWPD(0:3,%(nexternal)d))',
                'ALLOCATE(MPPWW(%(nwavefuncs)d), MPPWAMP(%(nbornamps)d), MPPWAMPL(3,%(nctamps)d))',
                'ALLOCATE(MPPWPL(0:3,-1:%(nloopwavefuncs)d), MPPWPD(0:3,%(nexternal)d))'] +
                (['ALLOCATE(EPPWW(%(nwavefuncs)d), EPPWAMP(%(nbornamps)d), EPPWAMPL(3,%(nctamps)d))',
                  'ALLOCATE(EPPWPL(0:3,-1:%(nloopwavefuncs)d), EPPWPD(0:3,%(nexternal)d))']
                 if ep else []) + [
                'END SUBROUTINE %(proc_prefix)sPW_DUAL_INIT',
                'END MODULE %(proc_prefix)sPW_DUAL']) % replace_dict + '\n')

        mp_poly_writer=q_polynomial.FortranPolynomialRoutines(
            poly_rank,
            updater_max_rank = matrix_element.get_max_loop_vertex_rank(),        
            coef_format='complex*32', sub_prefix='MP_'+replace_dict['proc_prefix'],
            proc_prefix=replace_dict['proc_prefix'], mp_prefix='MP_')
        # The eval subroutine
        subroutines.append(poly_writer.write_polynomial_evaluator())
        subroutines.append(mp_poly_writer.write_polynomial_evaluator())
        # The add coefs subroutine
        subroutines.append(poly_writer.write_add_coefs())
        subroutines.append(mp_poly_writer.write_add_coefs())        
        # The merging one for creating the loop coefficients
        subroutines.append(poly_writer.write_wl_merger())
        subroutines.append(mp_poly_writer.write_wl_merger())
        for wl_update in matrix_element.get_used_wl_updates():
            # We pick here the most appropriate way of computing the 
            # tensor product depending on the rank of the two tensors.
            # The various choices below come out from a careful comparison of
            # the different methods using the valgrind profiler
            if wl_update[0]==wl_update[1]==1 or wl_update[0]==0 or wl_update[1]==0:
                # If any of the rank is 0, or if they are both equal to 1, 
                # then we are better off using the full expanded polynomial, 
                # and let the compiler optimize it.
                subroutines.append(poly_writer.write_expanded_wl_updater(\
                                                     wl_update[0],wl_update[1]))
                subroutines.append(mp_poly_writer.write_expanded_wl_updater(\
                                                     wl_update[0],wl_update[1]))
            elif wl_update[0] >= wl_update[1]:
                # If the loop polynomial is larger then we will filter and loop
                # over the vertex coefficients first. The smallest product for
                # which the routines below could be used is then 
                # loop_rank_2 x vertex_rank_1
                subroutines.append(poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=True))
                subroutines.append(mp_poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=True))
            else:
                # This happens only when the rank of the updater (vertex coef)
                # is larger than the one of the loop coef and none of them is
                # zero. This never happens in renormalizable theories but it
                # can happen in the HEFT ones or other effective ones. In this
                # case the typically use of this routine if for the product
                # loop_rank_1 x vertex_rank_2
                subroutines.append(poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=False))
                subroutines.append(mp_poly_writer.write_compact_wl_updater(\
                  wl_update[0],wl_update[1],loop_over_vertex_coefs_first=False))            
            if wl_update[1] == 1 and 2 <= wl_update[0] <= 4:
                for vertex in ('FFV1L1_2', 'FFV1L2_1'):
                    name = (mp_poly_writer.sub_prefix+'UPDATE_WL_%d_1_FACTOR_%s' %
                            (wl_update[0], vertex))
                    if len(name) <= 63:
                        subroutines.append(mp_poly_writer.write_factorized_wl_updater(
                            wl_update[0], vertex))
                
        terminal = helas_call_writers.FortranUFOHelasCallWriterOptimized.\
            get_terminal_wavefunctions(matrix_element)
        dp_terminal = self.get_dp_terminal_wavefunctions(matrix_element)
        for left, right in sorted(set((item['loop_mother_rank'], item['vertex_rank'])
                                      for item in dp_terminal.values())):
            subroutines.append(poly_writer.write_dp_terminal_wl_updater(left, right))
        for left, right in sorted(set((item['loop_mother_rank'], item['vertex_rank'])
                                      for item in terminal.values())):
            if left == right == 1 or left == 0 or right == 0:
                subroutines.append(mp_poly_writer.write_expanded_wl_updater(
                    left, right, terminal=True))
            else:
                subroutines.append(mp_poly_writer.write_compact_wl_updater(
                    left, right, terminal=True))
            if right == 1 and 2 <= left <= 4:
                for vertex in ('FFV1L1_2', 'FFV1L2_1'):
                    name = (mp_poly_writer.sub_prefix+'TERMINAL_WL_%d_1_FACTOR_%s' %
                            (left, vertex))
                    if len(name) <= 63:
                        subroutines.append(mp_poly_writer.write_factorized_wl_updater(
                            left, vertex, terminal=True))
        writer.writelines('\n\n'.join(subroutines),
                                       context=self.get_context(matrix_element))

    @staticmethod
    def validate_terminal_wavefunctions(directory, prefix, expected):
        """Check all rendered consumers before retaining private MP calls."""
        def logical_source(filename):
            statements = []
            with open(filename) as stream:
                for raw in stream:
                    if not raw.strip() or raw[0] in 'cC*!':
                        continue
                    statement = raw[6:].strip().upper()
                    if len(raw) > 5 and raw[5] not in ' 0':
                        if not statements:
                            raise ValueError("Orphan continuation")
                        statements[-1] += statement
                    else:
                        statements.append(statement)
            return '\n'.join(statements)
        try:
            sources = {os.path.basename(name): logical_source(name)
                       for name in glob.glob(pjoin(directory, '*.f'))}
            batches = sorted((name for name in sources
                              if re.fullmatch(r'mp_coef_construction_\d+\.f', name)),
                             key=lambda name: int(re.search(r'(\d+)\.f$', name)[1]))
            if not batches:
                return False
            stem = re.escape('MP_'+prefix.upper())
            update = re.compile(r'CALL '+stem+r'(UPDATE|TERMINAL)_WL_(\d+)_(\d+)'
                r'(_FACTOR_'+_FACTORIZED_WL_VERTEX+r')?'
                r'\(WL\(1,0,1,(-?\d+)\),(\d+),COEFS,(\d+),(\d+),'
                r'WL\(1,0,1,(\d+)\)(,WL_ENTRY_INEXACT)?\)')
            create = re.compile(r'CALL '+stem+r'CREATE_LOOP_COEFS\(WL\(1,0,1,(\d+)\),')
            producers, consumers, roots, private = {}, set(), set(), {}
            for name in batches:
                for line in sources[name].splitlines():
                    match = update.fullmatch(line)
                    if match:
                        mode, left, right, factor, origin, cut, ins, outs, target, flag = match.groups()
                        left, right, origin, cut, ins, outs, target = map(
                            int, (left, right, origin, cut, ins, outs, target))
                        if (target in producers or target == origin or
                                (origin not in producers and origin not in (-1, 0)) or
                                (mode == 'TERMINAL' or bool(factor)) != bool(flag) or
                                (factor and not (2 <= left <= 4 and right == 1 and
                                                 ins == outs == 4 and 0 <= cut <= 4))):
                            return False
                        info = dict(loop_mother_rank=left, vertex_rank=right,
                            loop_mother_number=origin, lcut_size=cut,
                            in_size=ins, out_size=outs, out=target)
                        producers[target] = info
                        consumers.add(origin)
                        if mode == 'TERMINAL':
                            private[target] = info
                    elif create.match(line):
                        if len(re.findall(r'\bWL\b', line)) != 1:
                            return False
                        root = int(create.match(line)[1])
                        if root not in producers:
                            return False
                        roots.add(root)
                    elif re.search(r'\bWL\b', line) and not line.startswith(('COMPLEX', 'COMMON')):
                        return False
            if private != expected or not set(private).issubset(roots-consumers):
                return False
            block = '/'+prefix.upper()+'MP_WL/'
            common = 'COMMON'+block+'WL,PL'
            for name, text in sources.items():
                declarations = [re.sub(r'\s+', '', line) for line in text.splitlines()
                                if block in re.sub(r'\s+', '', line)]
                if not declarations:
                    continue
                if any(declaration != common for declaration in declarations):
                    return False
                if name in batches:
                    continue
                for line in text.splitlines():
                    if line.startswith(('COMPLEX', 'COMMON')):
                        continue
                    if re.search(r'\bCALL\b[^\n]*\bWL\b', line):
                        return False
                    if len(re.findall(r'\bWL\b', line)) != len(re.findall(r'\bWL\(', line)):
                        return False
                    for use in re.finditer(r'\bWL\(([^)\n]*)\)', line):
                        indices = use[1].split(',')
                        if len(indices) != 4 or indices[-1] not in ('-1', '0'):
                            return False
            polynomial = sources.get('polynomial.f', '')
            merge = re.search(r'(?m)^SUBROUTINE '+stem+r'MERGE_WL\(.*?^END\s*$',
                              polynomial, re.S | re.M)
            create_body = re.search(r'(?m)^SUBROUTINE '+stem+r'CREATE_LOOP_COEFS\(.*?^END\s*$',
                                    polynomial, re.S | re.M)
            return bool(merge and create_body and
                'OUT(J)=OUT(J)+WL(I,J,I)*CONST' in merge[0].replace(' ', '') and
                len(re.findall(r'\bWL\(', merge[0])) == 2 and
                len(re.findall(r'\bWL\b', merge[0])) == 3 and
                len(re.findall(r'\bLOOP_WF\b', create_body[0])) == 3 and
                re.search(r'(?m)^CALL '+stem+r'MERGE_WL\(LOOP_WF,', create_body[0]))
        except (OSError, ValueError, KeyError):
            return False

    @staticmethod
    def factorize_mp_coef_calls(calls, prefix):
        """Select guarded private contractions after recognized vertex calls.

        Selection is local to one generated wavefunction call block. The
        Fortran helper verifies the actual stored derivative rows and falls
        back to the original updater whenever its numerical contract fails.
        """
        vertex_call = re.compile(r'CALL MP_('+_FACTORIZED_WL_VERTEX+r')\(.*?,COEFS\)')
        updater = re.compile(r'(CALL MP_'+re.escape(prefix)+
            r'(UPDATE|TERMINAL)_WL_([2-4])_1)'
            r'\(WL\(1,0,1,(-?\d+)\),([0-4]),COEFS,4,4,'
            r'WL\(1,0,1,(\d+)\)(,WL_ENTRY_INEXACT)?\)')
        result = []
        for block in calls:
            vertex = None
            lines = block.splitlines(keepends=True)
            for index, raw in enumerate(lines):
                line = raw.rstrip('\r\n')
                if not line.strip() or line.lstrip().startswith(('#', '!')):
                    continue
                producer = vertex_call.fullmatch(line)
                if producer:
                    vertex = producer[1]
                    continue
                match = updater.fullmatch(line)
                if (vertex and match and int(match[4]) != int(match[6]) and
                        int(match[6]) > 0 and
                        (match[2] == 'TERMINAL') == bool(match[7])):
                    name = match[1][5:]+'_FACTOR_'+vertex
                    if len(name) <= 63:
                        rewritten = line.replace(match[1], 'CALL '+name, 1)
                        if match[2] == 'UPDATE':
                            rewritten = rewritten[:-1]+',WL_ENTRY_INEXACT)'
                        lines[index] = rewritten+raw[len(line):]
                vertex = None
            result.append(''.join(lines))
        return result

    def write_mp_compute_loop_coefs(self, writer, matrix_element, fortran_model,
                                   _terminal=True):
        """Create the write_mp_compute_loop_coefs.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0
        
        # Set lowercase/uppercase Fortran code
        
        writers.FortranWriter.downcase = False

        replace_dict = copy.copy(matrix_element.rep_dict)                 

        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = ''

        # Extract helas calls
        squared_orders = matrix_element.get_squared_order_contribs()
        split_orders = matrix_element.get('processes')[0].get('split_orders')
        
        born_ct_helas_calls , uvct_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element,
                       squared_orders=squared_orders, split_orders=split_orders)
        self.turn_to_mp_calls(born_ct_helas_calls)
        self.turn_to_mp_calls(uvct_helas_calls)
        pwave = loop_onia.has_pwave(matrix_element)
        terminal = (fortran_model.get_terminal_wavefunctions(matrix_element)
                    if _terminal and not pwave else {})
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops,
                        squared_orders=squared_orders,split_orders=split_orders,
                        terminal_wavefunctions=terminal)
        # The proc_prefix must be replaced
        coef_construction = [c % matrix_element.rep_dict for c 
                                                           in coef_construction]
        self.turn_to_mp_calls(coef_construction)
        self.turn_to_mp_calls(coef_merging)        
        momenta = 'MP_P'
        if pwave:
            # A P-wave bound state: the dual quadruple-precision calls, with
            # the derivative of the loop wavefunctions (no factorised or
            # terminal updaters)
            momenta = 'MPPWPD'
            coef_construction = loop_onia.pwave_dual_names(
                loop_onia.pwave_coef_construction_calls(
                    coef_construction, loop_onia.npwave(matrix_element)), mp=True)
            born_ct_helas_calls = loop_onia.pwave_dual_names(born_ct_helas_calls, mp=True)
            uvct_helas_calls = loop_onia.pwave_dual_names(uvct_helas_calls, mp=True)
        else:
            coef_construction = self.factorize_mp_coef_calls(
                coef_construction, matrix_element.rep_dict['proc_prefix'])
        # Add the intrinsic call after MP conversion, which prefixes all CALLs.
        private_call = re.compile(r'(?m)^(CALL MP_'+re.escape(
            matrix_element.rep_dict['proc_prefix'])+_PRIVATE_MP_WL_ROUTINE+r'\()')
        has_private_updates = any(private_call.search(call) for call in coef_construction)
        coef_construction = [private_call.sub(
            r'CALL IEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)\n\1', call)
            for call in coef_construction]
                                         
        file = open(os.path.join(self.template_dir,\
                                           'mp_compute_loop_coefs.inc')).read()

        # Setup the contextual environment which is used in the splitting
        # functions below
        context = self.get_context(matrix_element)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'mp_helas_calls_split.inc',file,born_ct_helas_calls,\
                        'mp_born_ct_helas_calls','mp_helas_calls_ampb',
                        required_so_broadcaster = 'MP_CT_REQ_SO_DONE',
                        continue_label = 2000,
                        momenta_array_name = momenta,
                        context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'mp_helas_calls_split.inc',file,uvct_helas_calls,\
                        'mp_uvct_helas_calls','mp_helas_calls_uvct',
                        required_so_broadcaster = 'MP_UVCT_REQ_SO_DONE',
                        continue_label = 3000,
                        momenta_array_name = momenta,
                        context=context)
        if has_private_updates:
            replace_dict['terminal_ieee_use'] = '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n'
            replace_dict['terminal_ieee_declarations'] = '      LOGICAL WL_ENTRY_INEXACT\n'
        file=self.split_HELASCALLS(writer,replace_dict,\
                'mp_helas_calls_split.inc',file,coef_construction,\
                'mp_coef_construction','mp_coef_construction',
                required_so_broadcaster = 'MP_LOOP_REQ_SO_DONE',
                continue_label = 4000,
                momenta_array_name = momenta,
                context=context)

        replace_dict['mp_coef_merging']='\n'.join(coef_merging)

        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = ''

        file = misc.apply_template(file, replace_dict)

        # Write the file
        writer.writelines(file,context=context)

    def write_color_matrix_data_file(self, writer, col_matrix):
        """Writes out the files (Loop|Born)ColorFlowMatrix.dat corresponding
        to the color coefficients for JAMP(L|B)*JAMP(L|B)."""
        
        res = []
        for line in range(len(col_matrix._col_basis1)):
            numerators = []
            denominators = []
            for row in range(len(col_matrix._col_basis2)):
                coeff = col_matrix.col_matrix_fixed_Nc[(line,row)]
                numerators.append('%6r'%coeff[0].numerator)
                denominators.append('%6r'%(
                                  coeff[0].denominator*(-1 if coeff[1] else 1)))
            res.append(' '.join(numerators))
            res.append(' '.join(denominators))            
        
        res.append('EOF')
        
        writer.writelines('\n'.join(res))
    
    def write_color_flow_coefs_data_file(self, writer, color_amplitudes, 
                                                                   color_basis):
        """ Writes the file '(Loop|Born)ColorFlowCoefs.dat using the coefficients
        list of the color_amplitudes in the argument of this function."""

        my_cs = color.ColorString()        
        
        res = []

        for jamp_number, coeff_list in enumerate(color_amplitudes):
            my_cs.from_immutable(sorted(color_basis.keys())[jamp_number])
            # Order the ColorString so that its ordering is canonical.
            ordered_cs = color.ColorFactor([my_cs]).full_simplify()[0]    
            res.append('%d # Coefficient for flow number %d with expr. %s'\
                            %(len(coeff_list), jamp_number+1, repr(ordered_cs)))
            # A line element is a tuple (numerator, denominator, amplitude_id)
            line_element = []

            for (coefficient, amp_number) in coeff_list:
                coef = self.cat_coeff(\
                    coefficient[0],coefficient[1],coefficient[2],coefficient[3])
                line_element.append((coef[0].numerator,
                         coef[0].denominator*(-1 if coef[1] else 1),amp_number))
            # Sort them by growing amplitude number
            line_element.sort(key=lambda el:el[2])

            for i in range(3):
                res.append(' '.join('%6r'%elem[i] for elem in line_element))
        
        res.append('EOF')
        writer.writelines('\n'.join(res))
    
    def write_compute_color_flows(self, writer, matrix_element, config_map):
        """Writes the file compute_color_flows.f which uses the AMPL results
        from a common block to project them onto the color flow space so as 
        to compute the JAMP quantities. For loop induced processes, this file
        will also contain a subroutine computing AMPL**2 for madevent
        multichanneling."""
        
        loop_col_amps = matrix_element.get_loop_color_amplitudes()
        matrix_element.rep_dict['nLoopFlows'] = len(loop_col_amps)
        
        dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sLoopColorFlowCoefs.dat'
                                                %matrix_element.rep_dict),'w')
        self.write_color_flow_coefs_data_file(dat_writer,
                        loop_col_amps, matrix_element.get('loop_color_basis'))
        dat_writer.close()

        dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sLoopColorFlowMatrix.dat'
                                                %matrix_element.rep_dict),'w')
        self.write_color_matrix_data_file(dat_writer,
                                             matrix_element.get('color_matrix'))
        dat_writer.close() 

        if matrix_element.get('processes')[0].get('has_born'):
            born_col_amps = matrix_element.get_born_color_amplitudes()
            matrix_element.rep_dict['nBornFlows'] = len(born_col_amps)
            dat_writer = open(pjoin('..','MadLoop5_resources',
                                      '%(proc_prefix)sBornColorFlowCoefs.dat'
                                                %matrix_element.rep_dict),'w')
            self.write_color_flow_coefs_data_file(dat_writer,
                          born_col_amps, matrix_element.get('born_color_basis'))
            dat_writer.close()
            
            dat_writer = open(pjoin('..','MadLoop5_resources',
                                     '%(proc_prefix)sBornColorFlowMatrix.dat'
                                                %matrix_element.rep_dict),'w')
            self.write_color_matrix_data_file(dat_writer,
                  color_amp.ColorMatrix(matrix_element.get('born_color_basis')))
            dat_writer.close()
        else:
            matrix_element.rep_dict['nBornFlows'] = 0

        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # The following variables only have to be defined for the LoopInduced
        # output for madevent.
        if self.get_context(matrix_element)['MadEventOutput']:
            self.get_amp2_lines(matrix_element, replace_dict, config_map)
        else:
            replace_dict['config_map_definition'] = ''
            replace_dict['config_index_map_definition'] = ''            
            replace_dict['nmultichannels'] = 0
            replace_dict['nmultichannel_configs'] = 0
            
        # The nmultichannels entry will be used in the matrix<i> wrappers as 
        # well, so we add it to the general_replace_dict too.
        matrix_element.rep_dict['nmultichannels'] = \
                                                  replace_dict['nmultichannels']
        matrix_element.rep_dict['nmultichannel_configs'] = \
                                           replace_dict['nmultichannel_configs']        

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line

        # Helicity offset convention
        # For a given helicity, the attached integer 'i' means
        # 'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, 
        #                             to helicity number abs(i+HELOFFSET)
        # 'i' == -HELOFFSET        -> Helicity is analytically zero
        # 'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'.
        #                             If it is zero, it is skipped.
        # Typically, the hel_offset is 10000
        replace_dict['hel_offset'] = 10000
        
        
        file = misc.apply_template(
            open(os.path.join(self.template_dir,
                              'compute_color_flows.inc')).read(),
            replace_dict)

        context = self.get_context(matrix_element)
        writer.writelines(file,context=context)
        if context['UseDensity']:
            self.density_prefixes.add(matrix_element.rep_dict['proc_prefix'])
    
    def write_global_specs(self, matrix_element_list, output_path=None):
        """ From the list of matrix element, or the single matrix element, derive
        the global quantities to write in global_coef_specs.inc"""
        
        if isinstance(matrix_element_list, (group_subprocs.SubProcessGroupList,
                                            loop_helas_objects.LoopHelasProcess)):
            matrix_element_list = matrix_element_list.get_matrix_elements()
        
        if isinstance(matrix_element_list, list):
            me_list = matrix_element_list
        else:
            me_list = [matrix_element_list]    

        if output_path is None:
            out_path = pjoin(self.dir_path,'SubProcesses','global_specs.inc')
        else:
            out_path = output_path

        open(out_path,'w').write(
"""      integer MAXNEXTERNAL
      parameter(MAXNEXTERNAL=%d)
      integer OVERALLMAXRANK
      parameter(OVERALLMAXRANK=%d)
      integer NPROCS
      parameter(NPROCS=%d)"""%(
         max(me.get_nexternal_ninitial()[0] for me in me_list),
         max(me.get_max_loop_rank() for me in me_list),
         len(me_list)))    

    
    def fix_coef_specs(self, overall_max_lwf_spin, overall_max_loop_vert_rank):
        """ If processes with different maximum loop wavefunction size or
        different maximum loop vertex rank have to be output together, then
        the file 'coef.inc' in the HELAS Source folder must contain the overall
        maximum of these quantities. It is not safe though, and the user has 
        been appropriately warned at the output stage """
        
        # Remove the existing link
        coef_specs_path=os.path.join(self.dir_path,'Source','DHELAS',\
                                                               'coef_specs.inc')
        os.remove(coef_specs_path)
       
        spin_to_wf_size = {1:4,2:4,3:4,4:16,5:16}
        overall_max_lwf_size = spin_to_wf_size[overall_max_lwf_spin]
        overall_max_loop_vert_coefs = q_polynomial.get_number_of_coefs_for_rank(
                                                     overall_max_loop_vert_rank)
        # Replace it by the appropriate value
        IncWriter=writers.FortranWriter(coef_specs_path,'w')
        IncWriter.writelines("""INTEGER MAXLWFSIZE
                           PARAMETER (MAXLWFSIZE=%(max_lwf_size)d)
                           INTEGER VERTEXMAXCOEFS
                           PARAMETER (VERTEXMAXCOEFS=%(vertex_max_coefs)d)"""\
                           %{'max_lwf_size':overall_max_lwf_size,
                             'vertex_max_coefs':overall_max_loop_vert_coefs})
        IncWriter.close()

    def setup_check_sa_replacement_dictionary(self, matrix_element, \
                                       split_orders,squared_orders,amps_orders):
        """ Sets up the replacement dictionary for the writeout of the steering
        file check_sa.f"""
        if len(squared_orders)<1:
            matrix_element.rep_dict['print_so_loop_results']=\
                                         "write(*,*) 'No split orders defined.'"
        elif len(squared_orders)==1:
            matrix_element.rep_dict['set_coupling_target']=''
            matrix_element.rep_dict['print_so_loop_results']=\
              "write(*,*) 'All loop contributions are of split orders (%s)'"%(
                      ' '.join(['%s=%d'%(split_orders[i],squared_orders[0][i]) \
                                            for i in range(len(split_orders))]))
        else:
            matrix_element.rep_dict['set_coupling_target']='\n'.join([
'# Here we leave the default target squared split order to -1, meaning that we'+
' aim at computing all individual contributions. You can choose otherwise.',
'call %(proc_prefix)sSET_COUPLINGORDERS_TARGET(-1)'%matrix_element.rep_dict])
            matrix_element.rep_dict['print_so_loop_results'] = '\n'.join([
              '\n'.join(["write(*,*) '%dL) Loop ME for orders (%s) :'"%((j+1),(' '.join(
          ['%s=%d'%(split_orders[i],so[i]) for i in range(len(split_orders))]))),
              "IF (PREC_FOUND(%d).NE.-1.0d0) THEN"%(j+1),
              "write(*,*) ' > accuracy = ',PREC_FOUND(%d)"%(j+1),
              "ELSE",
              "write(*,*) ' > accuracy =   NA'",              
              "ENDIF",
              "write(*,*) ' > finite   = ',MATELEM(1,%d)"%(j+1),
              "write(*,*) ' > 1eps     = ',MATELEM(2,%d)"%(j+1),
              "write(*,*) ' > 2eps     = ',MATELEM(3,%d)"%(j+1)
              ]) for j, so in enumerate(squared_orders)])
        matrix_element.rep_dict['write_so_loop_results'] = '\n'.join(
          ["write (69,*) 'Split_Orders_Names %s'"%(' '.join(split_orders))]+
          ['\n'.join([
          "write (69,*) 'Loop_SO_Results %s'"%(' '.join(
                                           ['%d'%so_value for so_value in so])),
          "write (69,*) 'SO_Loop ACC  ',PREC_FOUND(%d)"%(j+1),
          "write (69,*) 'SO_Loop FIN  ',MATELEM(1,%d)"%(j+1),
          "write (69,*) 'SO_Loop 1EPS ',MATELEM(2,%d)"%(j+1),
          "write (69,*) 'SO_Loop 2EPS ',MATELEM(3,%d)"%(j+1),
          ]) for j, so in enumerate(squared_orders)])

        # We must reconstruct here the born squared orders.
        squared_born_so_orders = []
        for i, amp_order in enumerate(amps_orders['born_amp_orders']):
            for j in range(0,i+1):
                key = tuple([ord1 + ord2 for ord1,ord2 in \
                        zip(amp_order[0],amps_orders['born_amp_orders'][j][0])])
                if not key in squared_born_so_orders:
                    squared_born_so_orders.append(key)
        if len(squared_born_so_orders)<1:
            matrix_element.rep_dict['print_so_born_results'] = ''
        elif len(squared_born_so_orders)==1: 
            matrix_element.rep_dict['print_so_born_results'] = \
              "write(*,*) 'All Born contributions are of split orders (%s)'"%(
                ' '.join(['%s=%d'%(split_orders[i],squared_born_so_orders[0][i]) 
                                            for i in range(len(split_orders))]))
        else:
            matrix_element.rep_dict['print_so_born_results'] = '\n'.join([
          "write(*,*) '%dB) Born ME for orders (%s) = ',MATELEM(0,%d)"%(j+1,' '.join(
       ['%s=%d'%(split_orders[i],so[i]) for i in range(len(split_orders))]),j+1)
                                for j, so in enumerate(squared_born_so_orders)])
        matrix_element.rep_dict['write_so_born_results'] = '\n'.join(
          ['\n'.join([
          "write (69,*) 'Born_SO_Results %s'"%(' '.join(
                                           ['%d'%so_value for so_value in so])),
          "write (69,*) 'SO_Born BORN ',MATELEM(0,%d)"%(j+1),
          ]) for j, so in enumerate(squared_born_so_orders)])
        
        # Add a bottom bar to both print_so_[loop|born]_results 
        matrix_element.rep_dict['print_so_born_results'] += \
                             '\nwrite (*,*) "---------------------------------"'
        matrix_element.rep_dict['print_so_loop_results'] += \
                             '\nwrite (*,*) "---------------------------------"'
                             
    def write_tir_cache_size_include(self, writer):
        """Write the file 'tir_cache_size.inc' which sets the size of the TIR
        cache the user wishes to employ and the default value for it.
        This can have an impact on MadLoop speed when using stability checks
        but also impacts in a non-negligible way MadLoop's memory footprint.
        It is therefore important that the user can chose its size."""

        # For the standalone optimized output, a size of one is necessary.
        # The MadLoop+MadEvent output sets it to 2 because it can gain further
        # speed increase with a TIR cache of size 2 due to the structure of the
        # calls to MadLoop there.
        tir_cach_size = "parameter(TIR_CACHE_SIZE=1)"
        writer.writelines(tir_cach_size)

    def write_loopmatrix(self, writer, matrix_element, fortran_model, \
                                                   write_auxiliary_files=True,):
        """Create the loop_matrix.f file."""
        
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        # Set lowercase/uppercase Fortran code
        writers.FortranWriter.downcase = False

        # Starting off with the treatment of the split_orders since some 
        # of the information extracted there will come into the 
        # general_replace_dict. Split orders are abbreviated SO in all the 
        # keys of the replacement dictionaries.
        
        # Take care of the split_orders
        squared_orders, amps_orders = matrix_element.get_split_orders_mapping()
        # Creating here a temporary list containing only the information of 
        # what are the different squared split orders contributing
        # (i.e. not using max_contrib_amp_number and max_contrib_ref_amp_number)
        sqso_contribs = [sqso[0] for sqso in squared_orders]
        split_orders = matrix_element.get('processes')[0].get('split_orders')
        # The entries set in the function below are only for check_sa written
        # out in write_loop__matrix_element_v4 (it is however placed here because the
        # split order information is only available here).
        self.setup_check_sa_replacement_dictionary(matrix_element,
                                         split_orders,sqso_contribs,amps_orders)
        
        # Now recast the split order basis for the loop, born and counterterm
        # amplitude into one single splitorderbasis.
        overall_so_basis = misc.make_unique(
            [born_so[0] for born_so in amps_orders['born_amp_orders']]+
            [born_so[0] for born_so in amps_orders['loop_amp_orders']])
        # We must re-sort it to make sure it follows an increasing WEIGHT order
        order_hierarchy = matrix_element.get('processes')[0]\
                                            .get('model').get('order_hierarchy')
        if set(order_hierarchy.keys()).union(set(split_orders))==\
                                                    set(order_hierarchy.keys()):
            overall_so_basis.sort(key= lambda so: 
                         sum([order_hierarchy[split_orders[i]]*order_power for \
                                              i, order_power in enumerate(so)]))

        # Those are additional entries used throughout the different files of
        # MadLoop5
        matrix_element.rep_dict['split_order_str_list'] = str(split_orders)
        matrix_element.rep_dict['nSO'] = len(split_orders)
        matrix_element.rep_dict['nSquaredSO'] = len(sqso_contribs)
        matrix_element.rep_dict['nAmpSO'] = len(overall_so_basis)

        writers.FortranWriter('nsquaredSO.inc').writelines(
"""INTEGER NSQUAREDSO
PARAMETER (NSQUAREDSO=%d)"""%matrix_element.rep_dict['nSquaredSO'])
        files.cp('nsquaredSO.inc', '..')
        
        replace_dict = copy.copy(matrix_element.rep_dict)
        # Build the general array mapping the split orders indices to their
        # definition
        replace_dict['ampsplitorders'] = '\n'.join(self.get_split_orders_lines(\
                                             overall_so_basis,'AMPSPLITORDERS'))
        replace_dict['SquaredSO'] = '\n'.join(self.get_split_orders_lines(\
                                                  sqso_contribs,'SQPLITORDERS'))
        
        # Specify what are the squared split orders selected by the proc def.
        replace_dict['chosen_so_configs'] = self.set_chosen_SO_index(
                               matrix_element.get('processes')[0],sqso_contribs)
        
        # Now we build the different arrays storing the split_orders ID of each
        # amp.
        ampSO_list=[-1]*sum(len(el[1]) for el in amps_orders['loop_amp_orders'])
        for SO in amps_orders['loop_amp_orders']:
            for amp_number in SO[1]:
                ampSO_list[amp_number-1]=overall_so_basis.index(SO[0])+1

        replace_dict['loopAmpSO'] = '\n'.join(self.format_integer_list(
                                                    ampSO_list,'LOOPAMPORDERS'))
        ampSO_list=[-1]*sum(len(el[1]) for el in amps_orders['born_amp_orders'])
        for SO in amps_orders['born_amp_orders']:
            for amp_number in SO[1]:
                ampSO_list[amp_number-1]=overall_so_basis.index(SO[0])+1
        replace_dict['BornAmpSO'] = '\n'.join(self.format_integer_list(
                                                    ampSO_list,'BORNAMPORDERS'))

        # We then go to the TIR setup
        # The first entry is the CutTools, we make sure it is available
        looplibs_av=['.TRUE.']
        # one should be careful about the order in the following as it must match
        # the ordering in MadLoopParamsCard.
        for tir_lib in ['pjfry','iregi','golem','samurai','ninja','collier','ninjars','tirust']:
            looplibs_av.append('.TRUE.' if tir_lib in self.all_tir and \
                                self.tir_available_dict[tir_lib] else '.FALSE.')
        replace_dict['data_looplibs_av']=','.join(looplibs_av)

        # Helicity offset convention
        # For a given helicity, the attached integer 'i' means
        # 'i' in ]-inf;-HELOFFSET[ -> Helicity is equal, up to a sign, 
        #                             to helicity number abs(i+HELOFFSET)
        # 'i' == -HELOFFSET        -> Helicity is analytically zero
        # 'i' in ]-HELOFFSET,inf[  -> Helicity is contributing with weight 'i'.
        #                             If it is zero, it is skipped.
        # Typically, the hel_offset is 10000
        replace_dict['hel_offset'] = 10000

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line                  

        # When the user asks for the polarized matrix element we must 
        # multiply back by the helicity averaging factor
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        replace_dict['beamone_helavgfactor'], replace_dict['beamtwo_helavgfactor'] =\
                                       matrix_element.get_beams_hel_avg_factor()
        
        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = ''

        if 'density' in self.cmd_options:
            replace_dict['use_density'] = '.true.'
        else:
            replace_dict['use_density'] = '.false.'

        if write_auxiliary_files:
            # Write out the color matrix
            (CMNum,CMDenom) = self.get_color_matrix(matrix_element)
            CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorNumFactors.dat'%matrix_element.rep_dict),'w')
            for ColorLine in CMNum:
                CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
            CMWriter.close()
            CMWriter=open(pjoin('..','MadLoop5_resources',
            '%(proc_prefix)sColorDenomFactors.dat'%matrix_element.rep_dict),'w')
            for ColorLine in CMDenom:
                CMWriter.write(' '.join(['%d'%C for C in ColorLine])+'\n')
            CMWriter.close()
            
            # Write out the helicity configurations
            HelConfigs=matrix_element.get_helicity_matrix()
            HelConfigWriter=open(pjoin('..','MadLoop5_resources',
                 '%(proc_prefix)sHelConfigs.dat'%matrix_element.rep_dict),'w')
            for HelConfig in HelConfigs:
                HelConfigWriter.write(' '.join(['%d'%H for H in HelConfig])+'\n')
            HelConfigWriter.close()
        
        # Extract helas calls
        born_ct_helas_calls, uvct_helas_calls = \
                           fortran_model.get_born_ct_helas_calls(matrix_element,
                        squared_orders=squared_orders,split_orders=split_orders)
        coef_construction, coef_merging = fortran_model.get_coef_construction_calls(\
                                    matrix_element,group_loops=self.group_loops,
                        squared_orders=squared_orders,split_orders=split_orders)

        loop_CT_calls = fortran_model.get_loop_CT_calls(matrix_element,\
                       group_loops=self.group_loops,
                       squared_orders=squared_orders, split_orders=split_orders)
        # The proc_prefix must be replaced
        coef_construction = [c % matrix_element.rep_dict for c 
                                                           in coef_construction]
        loop_CT_calls = [lc % matrix_element.rep_dict for lc in loop_CT_calls]
        
        file = open(os.path.join(self.template_dir,\
                                           'loop_matrix_standalone.inc')).read()

        # Setup the contextual environment which is used in the splitting
        # functions below
        context = self.get_context(matrix_element)
        # A P-wave bound state: the double-precision amplitudes are computed
        # in dual numbers, from the dual constituent momenta PD
        momenta = 'P'
        if not hasattr(self, 'onia_pwave_processes'):
            self.onia_pwave_processes = []
        self.onia_pwave_processes.append(bool(context.get('OniaPwave')))
        if not hasattr(self, 'onia_npwave'):
            self.onia_npwave = set()
        if context.get('OniaPwaveEP'):
            # the EP files are derived from the MP ones at finalize (onia_ep.py)
            if not hasattr(self, 'onia_ep_dirs'):
                self.onia_ep_dirs = {}
            self.onia_ep_dirs[os.path.abspath('.')] = matrix_element.rep_dict['proc_prefix']
        if context.get('OniaPwave'):
            self.onia_npwave.add(loop_onia.npwave(matrix_element))
            momenta = 'PWPD'
            coef_construction = loop_onia.pwave_dual_names(
                loop_onia.pwave_coef_construction_calls(
                    coef_construction, loop_onia.npwave(matrix_element)))
            born_ct_helas_calls = loop_onia.pwave_dual_names(born_ct_helas_calls)
            uvct_helas_calls = loop_onia.pwave_dual_names(uvct_helas_calls)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'helas_calls_split.inc',file,born_ct_helas_calls,\
                        'born_ct_helas_calls','helas_calls_ampb',
                        required_so_broadcaster = 'CT_REQ_SO_DONE',
                        continue_label = 2000, momenta_array_name=momenta,
                        context = context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                        'helas_calls_split.inc',file,uvct_helas_calls,\
                        'uvct_helas_calls','helas_calls_uvct',
                        required_so_broadcaster = 'UVCT_REQ_SO_DONE',
                        continue_label = 3000, momenta_array_name=momenta,
                        context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                'helas_calls_split.inc',file,coef_construction,\
                'coef_construction','coef_construction',
                required_so_broadcaster = 'LOOP_REQ_SO_DONE',
                continue_label = 4000, momenta_array_name=momenta,
                context=context)
        file=self.split_HELASCALLS(writer,replace_dict,\
                'helas_calls_split.inc',file,loop_CT_calls,\
                'loop_CT_calls','loop_CT_calls',
                required_so_broadcaster = 'CTCALL_REQ_SO_DONE',
                continue_label = 5000, momenta_array_name=momenta,
                context=context)
       
        # Add the entries above to the general_replace_dict so that it can be 
        # used by write_mp_compute_loop_coefs later
        matrix_element.rep_dict['loop_CT_calls']=replace_dict['loop_CT_calls']            
        matrix_element.rep_dict['born_ct_helas_calls']=replace_dict['born_ct_helas_calls']            
        matrix_element.rep_dict['uvct_helas_calls']=replace_dict['uvct_helas_calls']            
        matrix_element.rep_dict['coef_construction']=replace_dict['coef_construction']            
        
        replace_dict['coef_merging']='\n'.join(coef_merging)
        if self.opt['vector_size']:
            replace_dict['include_vector'] = "include '../../Source/vector.inc'"
        else:
            replace_dict['include_vector'] = '' 

        #In loop-induced, particles are put onshell to get a better precision on PS points. If we want to study processes with external off-shell particles we need
        #it to consider the offshell mass m^2 = p^2 to the on-shell mass.
        #KEEP_OFFSHELL_MASS contains the information based on the generation string of which external particle should be kept off-shell.
        base_process_string = matrix_element.get('processes')[0].base_string()
        particles_process = base_process_string.replace(">", "", 1).split()
        offshell_or_not = ['.true.' if '*' in elem else '.false.' for elem in particles_process]
        logger.info("Particles with .true. are generated off-shell: " + str(offshell_or_not))

        replace_dict["keep_offshell_mass"] = ""
        for i in range(len(offshell_or_not)):
            replace_dict["keep_offshell_mass"] += f"KEEP_OFFSHELL_MASS({i + 1}) = {offshell_or_not[i]}\n"

        if context.get('Onia'):
            self.write_onia_loop_files(matrix_element, fortran_model, replace_dict)

        file = file % replace_dict
        if context.get('Onia'):
            file = self.onia_public_entries(file, matrix_element)
        number_of_calls = len([call for call in loop_CT_calls if call.find('CALL LOOP') != 0])   
        if writer:
            # Write the file
            writer.writelines(file,context=context)
            return number_of_calls
        else:
            # Return it to be written along with the others
            return number_of_calls, file

#===============================================================================
# LoopProcessExporterFortranSA
#===============================================================================
class LoopProcessExporterFortranMatchBox(LoopProcessOptimizedExporterFortranSA,
                                      export_v4.ProcessExporterFortranMatchBox):                                  
    """Class to take care of exporting a set of loop matrix elements in the
       Fortran format."""

    default_opt = dict(LoopProcessOptimizedExporterFortranSA.default_opt)
    default_opt['export_format'] = 'madloop_matchbox'
    default_opt['sa_symmetry'] = True
    



    def get_color_string_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        return export_v4.ProcessExporterFortranMatchBox.get_color_string_lines(matrix_element)


    def get_JAMP_lines(self, *args, **opts):
        """Adding leading color part of the colorflow"""
            
        return export_v4.ProcessExporterFortranMatchBox.get_JAMP_lines(self, *args, **opts)
      
    def get_ME_identifier(self, matrix_element, group_number = None, group_elem_number = None):
        """ To not mix notations between borns and virtuals we call it here also MG5 """
        return 'MG5_%d_'%matrix_element.get('processes')[0].get('id')         
      

    def finalize(self, matrix_element, cmdhistory, MG5options, outputflag):

        out = super().finalize(matrix_element, cmdhistory, MG5options, outputflag)
        misc.compile(cwd=pjoin(self.dir_path,'Source','MODEL'))
        return out

#===============================================================================
# LoopInducedExporter
#===============================================================================
class LoopInducedExporterME(LoopProcessOptimizedExporterFortranSA):
    """ A class to specify all the functions common to LoopInducedExporterMEGroup
    and LoopInducedExporterMENoGroup (but not relevant for the original
    Madevent exporters)"""

    madloop_makefile_name = 'makefile_MadLoop'
    
    
    def __init__(self, *args, **opts):
        """ Initialize the process, setting the proc characteristics."""
        super(LoopInducedExporterME, self).__init__(*args, **opts)
        self.proc_characteristic['loop_induced'] = True

        if self.opt and isinstance(self.opt['output_options'], dict) and \
                                       't_strategy' in self.opt['output_options']:
            self.opt['t_strategy'] = banner_mod.ConfigFile.format_variable(
                  self.opt['output_options']['t_strategy'], int, 't_strategy')
            
        self.opt['vector_size'] = 1 

    
    def get_context(self,*args,**opts):
        """ Make sure that the contextual variable MadEventOutput is set to
        True for this exporter"""
        
        context = super(LoopInducedExporterME,self).get_context(*args,**opts)
        context['MadEventOutput'] = True
        return context
        
    #===========================================================================
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===========================================================================
    def write_procdef_mg5(self, file_pos, modelname, process_str):
        """ write an equivalent of the MG4 proc_card in order that all the Madevent
        Perl script of MadEvent4 are still working properly for pure MG5 run.
        Not needed for StandAlone so we need to call the correct one 
        """
        
        return export_v4.ProcessExporterFortran.write_procdef_mg5(
            self, file_pos, modelname, process_str)

    def get_source_libraries_list(self):
        """ Returns the list of libraries to be compiling when compiling the
        SOURCE directory. It is different for loop_induced processes and 
        also depends on the value of the 'output_dependencies' option"""
        
        libraries_list = super(LoopInducedExporterME,self).\
                                                     get_source_libraries_list()

        if self.dependencies=='internal':
            libraries_list.append('$(LIBDIR)libcts.$(libext)')
            libraries_list.append('$(LIBDIR)libiregi.$(libext)')

        return libraries_list

    def link_files_in_SubProcess(self, Ppath):
        """ Add the loop-induced related links to the P* directory Ppath"""
        
        super(LoopInducedExporterME,self).link_files_in_SubProcess(Ppath)
        
        ln(pjoin('../MadLoop5_resources') , cwd=Ppath)

    def copy_template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        LoopProcessExporterFortranSA.loop_additional_template_setup(self,
                                                     copy_Source_makefile=False)

        LoopProcessOptimizedExporterFortranSA.\
                                  loop_optimized_additional_template_setup(self)
                                  
    
    #===========================================================================
    # Create jpeg diagrams, html pages,proc_card_mg5.dat and madevent.tar.gz
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """Function to finalize v4 directory, for inheritance.
        """
        
        self.proc_characteristic['loop_induced'] = True

        # This can be uncommented if one desires to have the MadLoop
        # initialization performed at the end of the output phase.
        # Alternatively, one can simply execute the command 'initMadLoop' in
        # the madevent interactive interface after the output.
        # from madgraph.interface.madevent_interface import MadLoopInitializer
        # MadLoopInitializer.init_MadLoop(self.dir_path,
        #                   subproc_prefix=self.SubProc_prefix, MG_options=None)
        
        self.write_global_specs(matrix_elements)
        
    def write_tir_cache_size_include(self, writer):
        """Write the file 'tir_cache_size.inc' which sets the size of the TIR
        cache the user wishes to employ and the default value for it.
        This can have an impact on MadLoop speed when using stability checks
        but also impacts in a non-negligible way MadLoop's memory footprint.
        It is therefore important that the user can chose its size."""

        # In this case of MadLoop+MadEvent output, we set it to 2 because we
        # gain further speed increase with a TIR cache of size 2 due to the 
        # the fact that we call MadLoop once per helicity configuration in this 
        # case.
        tir_cach_size = "parameter(TIR_CACHE_SIZE=2)"
        writer.writelines(tir_cach_size)

    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                        proc_id = None, config_map = [], subproc_number = None):
        """ Write it the wrapper to call the ML5 subroutine in the library.""" 
        
        # Generating the MadEvent wrapping ME's routines
        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if not isinstance(writer, writers.FortranWriter):
            raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            
        replace_dict = copy.copy(matrix_element.rep_dict)
        
        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
        
        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        # It can be set to None when write_matrix_element_v4 is called without
        # grouping. In this case the subroutine SMATRIX should take an empty
        # suffix.
        if proc_id is None:
            replace_dict['proc_id'] = ''
        else:
            replace_dict['proc_id'] = proc_id
        
        #set the average over the number of initial helicities
        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        replace_dict['beamone_helavgfactor'], replace_dict['beamtwo_helavgfactor'] =\
                                       matrix_element.get_beams_hel_avg_factor()

        # number of helicity states actually kept for each beam (can be reduced
        # by an explicit polarisation in the process definition). Used to avoid
        # applying the beam polarisation of the run_card on top of it.
        s1, s2 = matrix_element.get_spin_state_initial()
        replace_dict['nb_spin_state1'] = s1
        replace_dict['nb_spin_state2'] = s2

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines
        
        
        # Extract ndiags
        ndiags = len(matrix_element.get('diagrams'))
        replace_dict['ndiags'] = ndiags
        
        # Set define_iconfigs_lines
        replace_dict['define_iconfigs_lines'] = \
             """INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
             COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG"""

        if proc_id:
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_iconfigs_lines'] += \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
            # Set set_amp2_line
            replace_dict['configID_in_matrix'] = "SUBDIAG(%s)"%proc_id
        else:
            # Standard running
            # Set set_amp2_line
            replace_dict['configID_in_matrix'] = "MAPCONFIG(ICONFIG)"
        
        # If group_numer
        replace_dict['ml_prefix'] = \
                 self.get_ME_identifier(matrix_element, subproc_number, proc_id)
        
        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor
        
        n_tot_diags = len(matrix_element.get_loop_diagrams())
        replace_dict['n_tot_diags'] = n_tot_diags

        # Flavor table for GET_FLAVOR<proc_id>, called by the flavor dispatcher
        # in auto_dsig of the MadEvent output (otherwise auto_dsig fails to link
        # with 'undefined reference to get_flavor<proc_id>_'). Loop-induced
        # subprocesses carry only the trivial all-1 flavor (the loop ME exposes
        # no merged-flavor position data), matching the FLAVOR /NEXTERNAL*1/
        # convention used elsewhere in the loop matrix template.
        nexternal, _ = matrix_element.get_nexternal_ninitial()
        all_flav = matrix_element.get_external_flavors_with_iden()
        replace_dict['max_flavor'] = len(all_flav)
        get_flavor_matrix = ''
        for i in range(len(all_flav)):
            get_flavor_matrix += ' DATA (FLAVOR(i,  %d),i=  1, NEXTERNAL) /%s/\n' \
                                 % (i+1, ', '.join(['1'] * nexternal))
        replace_dict['get_flavor_matrix'] = get_flavor_matrix

        file = open(pjoin(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)).read()
        file = misc.apply_template(file, replace_dict)

        # Write the file
        writer.writelines(file)

        return 0, ncolor

    def get_amp2_lines(self, *args, **opts):
        """Make sure the function is implemented in the daughters"""

        raise NotImplemented('The function get_amp2_lines must be called in '+\
                                       ' the daughters of LoopInducedExporterME')

#===============================================================================
# LoopInducedExporterMEGroup
#===============================================================================
class LoopInducedExporterMEGroup(LoopInducedExporterME,
                                       export_v4.ProcessExporterFortranMEGroup):
    """Class to take care of exporting a set of grouped loop induced matrix 
    elements"""
    
    matrix_file = "matrix_loop_induced_madevent_group.inc"
    # GOODHEL(NCOMB, MAXSPROC) / NTRY(MAXSPROC) in matrix_file
    goodhel_dims = 'MAXSPROC'
    ntry_dims = 'MAXSPROC'

    def make_source_links(self,*args, **opts):
        """ In the loop-induced output with MadEvent, we need the files from the 
        Source folder """
        export_v4.ProcessExporterFortranMEGroup.make_source_links(
                                                            self, *args, **opts)

    def write_source_makefile(self, *args, **opts):
        """Pick the correct write_source_makefile function from 
        ProcessExporterFortranMEGroup"""
        
        export_v4.ProcessExporterFortranMEGroup.write_source_makefile(self,
                                                                  *args, **opts)

    def copy_template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        
        # Start with the MadEvent one
        export_v4.ProcessExporterFortranMEGroup.copy_template(self,*args,**opts)

        # Then the MadLoop-standalone related one
        LoopInducedExporterME.copy_template(self, *args, **opts)

    def finalize(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically what finalize_v4_directory must be used, so that the
        # MRO doesn't interfere.

        self.proc_characteristic['loop_induced'] = True
        
        export_v4.ProcessExporterFortranMEGroup.finalize(self,*args,**opts)

        # special handling for loop-induced processes for the template files
        #since some function are duplicated...
        text = open(pjoin(self.dir_path,'Source','setrun.f'), 'r').read()
        import madgraph.iolibs.file_writers as file_writers
        fsock = file_writers.FortranWriter(pjoin(self.dir_path,'Source', 'setrun.f'),'w')
        fsock.remove_routine(text, 'DDILOG')
        fsock.close()

        
        # And the finalize from LoopInducedExporterME which essentially takes
        # care of MadLoop virtuals initialization
        LoopInducedExporterME.finalize(self,*args,**opts)
        
    def generate_subprocess_directory(self, subproc_group,
                                                    fortran_model,group_number,
                                                    second_exporter=None,
                                                    second_helas=None):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files"""
        assert second_exporter is None
        # Generate the MadLoop files
        calls = 0
        matrix_elements = subproc_group.get('matrix_elements')
        for ime, matrix_element in enumerate(matrix_elements):
            self.unique_id +=1
            calls += self.generate_loop_subprocess(matrix_element,fortran_model,
                              group_number = group_number, proc_id = str(ime+1),
#          group_number = str(subproc_group.get('number')), proc_id = str(ime+1),
                            config_map = subproc_group.get('diagram_maps')[ime],
                            unique_id=self.unique_id)
        
        # Then generate the MadEvent files
        self.opt['vector_size'] =1 
        export_v4.ProcessExporterFortranMEGroup.generate_subprocess_directory(
                                 self, subproc_group,fortran_model,group_number)
        
        return calls
    
    def get_amp2_lines(self, matrix_element, replace_dict, config_map):
        """Return the various replacement dictionary inputs necessary for the 
        multichanneling amp2 definition for the loop-induced MadEvent output.
        """

        if not config_map:
            raise MadGraph5Error('A multi-channeling configuration map is '+\
              ' necessary for the MadEvent Loop-induced output with grouping.')

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()

        ret_lines = []
        # In this case, we need to sum up all amplitudes that have
        # identical topologies, as given by the config_map (which
        # gives the topology/config for each of the diagrams
        if isinstance(matrix_element, loop_helas_objects.LoopHelasMatrixElement):
            diagrams = matrix_element.get_loop_diagrams()            
        else:
            diagrams = matrix_element.get('diagrams')
                
        # Note that we need to use AMP2 number corresponding to the first 
        # diagram number used for that AMP2.
        # The dictionary below maps the config ID to this corresponding first 
        # diagram number
        config_index_map = {}
        # For each diagram number, the dictionary below gives the config_id it
        # belongs to or 0 if it doesn't belong to any.
        loop_amp_ID_to_config = {}
        
        # Combine the diagrams with identical topologies
        config_to_diag_dict = {}
        for idiag, diag in enumerate(diagrams):
            try:
                config_to_diag_dict[config_map[idiag]].append(idiag)
            except KeyError:
                config_to_diag_dict[config_map[idiag]] = [idiag]

        for config in sorted(config_to_diag_dict.keys()):
            config_index_map[config] = (config_to_diag_dict[config][0] + 1)
                   
            # First add the UV and R2 counterterm amplitudes of each selected
            # diagram for the multichannel config
            CT_amp_numbers = [a.get('number') for a in \
                                    sum([diagrams[idiag].get_ct_amplitudes() for \
                                     idiag in config_to_diag_dict[config]], [])]
            
            for CT_amp_number in CT_amp_numbers:
                loop_amp_ID_to_config[CT_amp_number] = config 

            # Now add here the loop amplitudes.
            loop_amp_numbers = [a.get('amplitudes')[0].get('number')
                       for a in sum([diagrams[idiag].get_loop_amplitudes() for \
                                     idiag in config_to_diag_dict[config]], [])]
            
            for loop_amp_number in loop_amp_numbers:
                loop_amp_ID_to_config[loop_amp_number] = config
                
        # Notice that the config_id's are not necessarily sequential here, so
        # the size of the config_index_map array has to be the maximum over all
        # config_ids.
        # config_index_map should never be empty unless there was no diagram,
        # so the expression below is ok.
        n_configs = max(config_index_map.keys())
        replace_dict['nmultichannel_configs'] = n_configs
                
        # We must fill the empty entries of the map with the dummy amplitude 
        # number 0.
        conf_list = [(config_index_map[i] if i in config_index_map else 0) \
                                                  for i in range(1,n_configs+1)]
        # Now the placeholder 'nmultichannels' refers to the number of 
        # multi-channels which are contributing, so we must filter out zeros. 
        replace_dict['nmultichannels'] = len([_ for _ in conf_list if _!=0])
        
        # Now write the amp2 related inputs in the replacement dictionary
        res_list = []
        chunk_size = 6
        for k in range(0, len(conf_list), chunk_size):
            res_list.append("DATA (config_index_map(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(conf_list)),
                    ','.join(["%6r" % i for i in conf_list[k:k + chunk_size]])))

        replace_dict['config_index_map_definition'] = '\n'.join(res_list)

        res_list = []
        n_loop_amps = max(loop_amp_ID_to_config.keys())
        amp_list = [loop_amp_ID_to_config[i] for i in \
                                   sorted(loop_amp_ID_to_config.keys()) if i!=0]
        chunk_size = 6
        for k in range(0, len(amp_list), chunk_size):
            res_list.append("DATA (CONFIG_MAP(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(amp_list)),
                    ','.join(["%6r" % i for i in amp_list[k:k + chunk_size]])))

        replace_dict['config_map_definition'] = '\n'.join(res_list)

        return
    
#===============================================================================
# LoopInducedExporterMENoGroup
#===============================================================================
class LoopInducedExporterMENoGroup(LoopInducedExporterME,
                                            export_v4.ProcessExporterFortranME):
    """Class to take care of exporting a set of individual loop induced matrix 
    elements"""

    matrix_file = "matrix_loop_induced_madevent.inc"

    def make_source_links(self,*args, **opts):
        """ In the loop-induced output with MadEvent, we need the files from the 
        Source folder """
        super(export_v4.ProcessExporterFortranME,self).\
                                                make_source_links(*args, **opts)

    def write_source_makefile(self, *args, **opts):
        """Pick the correct write_source_makefile function from 
        ProcessExporterFortran"""
        
        super(export_v4.ProcessExporterFortranME,self).\
                                            write_source_makefile(*args, **opts)

    def copy_template(self, *args, **opts):
        """Pick the right mother functions
        """
        # Call specifically the necessary building functions for the mixed
        # template setup for both MadEvent and MadLoop standalone
        
        # Start with the MadEvent one
        export_v4.ProcessExporterFortranME.copy_template(self,*args,**opts)

        # Then the MadLoop-standalone related one
        LoopInducedExporterME.copy_template(self, *args, **opts)

    def finalize(self, *args, **opts):
        """Pick the right mother functions
        """
        
        self.proc_characteristic['loop_induced'] = True
        # Call specifically what finalize must be used, so that the
        # MRO doesn't interfere.
        export_v4.ProcessExporterFortranME.finalize(self, *args, **opts)

        # And the finilize_v4 from LoopInducedExporterME which essentially takes
        # care of MadLoop virtuals initialization
        LoopInducedExporterME.finalize(self, *args, **opts)

    def generate_subprocess_directory(self, matrix_element, fortran_model, me_number):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files"""
        
        self.unique_id += 1
        # Then generate the MadLoop files
        calls = self.generate_loop_subprocess(matrix_element,fortran_model,                           
                                  group_number = me_number, 
                                  unique_id=self.unique_id)
        
        
        # First generate the MadEvent files
        calls += export_v4.ProcessExporterFortranME.generate_subprocess_directory(
                                 self, matrix_element, fortran_model, me_number)
        return calls

    def get_amp2_lines(self, matrix_element, replace_dict, config_map):
        """Return the amp2(i) = sum(amp for diag(i))^2 lines"""

        if config_map:
            raise MadGraph5Error('A configuration map should not be specified'+\
                              ' for the Loop induced exporter without grouping.')

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        # Get minimum legs in a vertex
        vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
        matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Note that we need to use AMP2 number corresponding to the first 
        # diagram number used for that AMP2.
        # The dictionary below maps the config ID to this corresponding first 
        # diagram number
        config_index_map = {}
        # For each diagram number, the dictionary below gives the config_id it
        # belongs to or 0 if it doesn't belong to any.
        loop_amp_ID_to_config = {}

        n_configs = 0
        for idiag, diag in enumerate(matrix_element.get('diagrams')):
            # Ignore any diagrams with 4-particle vertices.
            use_for_multichanneling = True
            if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                use_for_multichanneling = False
                curr_config = 0
            else:
                n_configs += 1
                curr_config = n_configs

            if not use_for_multichanneling:
                if 0 not in config_index_map: 
                    config_index_map[0] = idiag + 1
            else:
                config_index_map[curr_config] = idiag + 1
                   
            CT_amps = [ a.get('number') for a in diag.get_ct_amplitudes()]
            for CT_amp in CT_amps:
                loop_amp_ID_to_config[CT_amp] = curr_config
                            
            Loop_amps = [a.get('amplitudes')[0].get('number')
                                            for a in diag.get_loop_amplitudes()]
            for Loop_amp in Loop_amps:
                loop_amp_ID_to_config[Loop_amp] = curr_config
 
        # Now write the amp2 related inputs in the replacement dictionary
        n_configs = len([k for k in config_index_map.keys() if k!=0])
        replace_dict['nmultichannel_configs'] = n_configs
        # Now the placeholder 'nmultichannels' refers to the number of 
        # multi-channels which are contributing which, in the non-grouped case
        # is always equal to the total number of multi-channels. 
        replace_dict['nmultichannels'] = n_configs
        
        res_list = []
        conf_list = [config_index_map[i] for i in sorted(config_index_map.keys())
                                                                        if i!=0]
        chunk_size = 6
        for k in range(0, len(conf_list), chunk_size):
            res_list.append("DATA (config_index_map(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(conf_list)),
                    ','.join(["%6r" % i for i in conf_list[k:k + chunk_size]])))

        replace_dict['config_index_map_definition'] = '\n'.join(res_list)

        res_list = []
        n_loop_amps = max(loop_amp_ID_to_config.keys())
        amp_list = [loop_amp_ID_to_config[i] for i in \
                                   sorted(loop_amp_ID_to_config.keys()) if i!=0]
        chunk_size = 6
        for k in range(0, len(amp_list), chunk_size):
            res_list.append("DATA (CONFIG_MAP(i),i=%6r,%6r) /%s/" % \
                (k + 1, min(k + chunk_size, len(amp_list)),
                    ','.join(["%6r" % i for i in amp_list[k:k + chunk_size]])))

        replace_dict['config_map_definition'] = '\n'.join(res_list)
