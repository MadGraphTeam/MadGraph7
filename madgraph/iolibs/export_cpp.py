################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
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
"""Methods and classes to export models and matrix elements to Pythia 8
and C++ Standalone format."""

from __future__ import absolute_import
import copy
import fractions
import glob
import itertools
import logging
from math import fmod, factorial
import os
import re
import shutil
import subprocess
import json

import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.drawing_svg as draw_svg
import madgraph.iolibs.files as files
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.various.banner as banner_mod
from madgraph import MadGraph5Error, InvalidCmd, MG5DIR
from madgraph.iolibs.files import cp, ln, mv

from madgraph.iolibs.export_v4 import VirtualExporter, ProcessExporterFortran
import madgraph.various.misc as misc

import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_pythia8')
pjoin = os.path.join


def make_model_cpp(dir_path):
    """Make the model library in a C++ standalone directory"""

    source_dir = os.path.join(dir_path, "src")
    # Run standalone
    logger.info("Running make for src")
    misc.compile(cwd=source_dir)


#===============================================================================
# UFOModelConverterCPP
#===============================================================================

class UFOModelConverterCPP(object):
    """ A converter of the UFO-MG5 Model to the C++ format """

    # Static variables (for inheritance)
    output_name = 'C++ Standalone'
    namespace = 'MG5'
    aloha_writer = 'CPP'
    cc_ext = 'cc'

    # Dictionary from Python type to C++ type
    type_dict = {"real": "double",
                 "complex": "std::complex<double>",
                 "FLV_COUPLING": "FLV_COUPLING"}

    # Regular expressions for cleaning of lines from Aloha files
    compiler_option_re = re.compile(r'^#\w')
    namespace_re = re.compile(r'^using namespace')

    slha_to_depend = {('SMINPUTS', (3,)): ('aS',),
                      ('SMINPUTS', (1,)): ('aEM',)}

    # Template files to use
    include_dir = '.'
    cc_file_dir = '.'
    param_template_h = 'cpp_model_parameters_h.inc'
    param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = 'cpp_hel_amps_h.inc'
    aloha_template_cc = 'cpp_hel_amps_cc.inc'

    copy_include_files = []
    copy_cc_files = []

    def __init__(self, model, output_path, wanted_lorentz = [],
                 wanted_couplings = [], replace_dict={}):
        """ initialization of the objects """
        misc.sprint('Exporting model to C++ standalone format')
        self.model = model
        self.model_name = ProcessExporterCPP.get_model_name(model['name'])

        self.dir_path = output_path
        self.default_replace_dict = dict(replace_dict)
        # List of needed ALOHA routines
        self.wanted_lorentz = wanted_lorentz

        # For dependent couplings, only want to update the ones
        # actually used in each process. For other couplings and
        # parameters, just need a list of all.
        self.coups_dep = {}    # name -> base_objects.ModelVariable
        self.coups_indep = []  # base_objects.ModelVariable
        self.params_dep = []   # base_objects.ModelVariable
        self.params_indep = [] # base_objects.ModelVariable
        self.coups_flv_dep = []    # (name, object, [couplings])
        self.coups_flv_indep = []  # (name, object, [couplings]) 
        self.p_to_cpp = parsers.UFOExpressionParserCPP()

        # Prepare parameters and couplings for writeout in C++
        self.prepare_parameters()
        self.prepare_couplings(wanted_couplings)

    def write_files(self):
        """Create all necessary files"""

        # Write Helas Routines
        self.write_aloha_routines()

        # Write parameter (and coupling) class files
        self.write_parameter_class_files()

    # Routines for preparing parameters and couplings from the model

    def prepare_parameters(self):
        """Extract the parameters from the model, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += [p for p in self.model['parameters'][key] if p.name]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))
            else:
                for p in self.model['parameters'][key]:
                    if p.name == 'ZERO':
                        continue
                    self.params_indep.append(base_objects.ModelVariable(p.name,
                                              p.name + " = " + \
                                              self.p_to_cpp.parse(p.expr) + ";",
                                              p.type,
                                              p.depend))

        # For external parameters, want to read off the SLHA block code
        while params_ext:
            param = params_ext.pop(0)
            # Read value from the slha variable
            expression = ""
            assert param.value.imag == 0
            if len(param.lhacode) == 1:
                expression = "%s = slha.get_block_entry(\"%s\", %d, %e);" % \
                             (param.name, param.lhablock.lower(),
                              param.lhacode[0], param.value.real)
            elif len(param.lhacode) == 2:
                expression = "indices[0] = %d;\nindices[1] = %d;\n" % \
                             (param.lhacode[0], param.lhacode[1])
                expression += "%s = slha.get_block_entry(\"%s\", indices, %e);" \
                              % (param.name, param.lhablock.lower(), param.value.real)
            else:
                raise MadGraph5Error("Only support for SLHA blocks with 1 or 2 indices")
            self.params_indep.insert(0,
                                   base_objects.ModelVariable(param.name,
                                                   expression,
                                                              'real'))
            
    def prepare_couplings(self, wanted_couplings = []):
        """Extract the couplings from the model, and store them in
        the two lists coups_indep and coups_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['couplings'].keys())
        keys.sort(key=len)
        for key, coup_list in self.model['couplings'].items():
            if "aS" in key:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings \
                        or f"-{c.name}" in wanted_couplings:
                        self.coups_dep[c.name] = base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend)
            else:
                for c in coup_list:
                    if not wanted_couplings or c.name in wanted_couplings \
                        or f"-{c.name}" in wanted_couplings:
                        self.coups_indep.append(base_objects.ModelVariable(\
                                                                   c.name,
                                                                   c.expr,
                                                                   c.type,
                                                                   c.depend))

        # Handle flavor couplings
        # strategy picke one of the actual coupling and check if this is a running one or not
        flavor_couplings = [c for c in wanted_couplings if isinstance(c, base_objects.FLV_Coupling)]
        misc.sprint(self.coups_dep)
        deps = [c.name for c in self.coups_dep.values()]
        for one_flv in flavor_couplings:
            one_coupling = one_flv.get_one_coupling()
            if one_coupling in deps:
                self.coups_flv_dep.append( one_flv)
            else:
                self.coups_flv_indep.append(one_flv)

        # Convert coupling expressions from Python to C++
        for coup in list(self.coups_dep.values()) + self.coups_indep:
            coup.expr = coup.name + " = " + self.p_to_cpp.parse(coup.expr) + ";"


    # Routines for writing the parameter files

    def write_parameter_class_files(self):
        """Generate the parameters_model.h and parameters_model.cc
        files, which have the parameters and couplings for the model."""

        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        parameter_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'Parameters_%s.h' % self.model_name)
        parameter_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'Parameters_%s.cc' % self.model_name)

        file_h, file_cc = self.generate_parameters_class_files()

        # Write the files
        writers.CPPWriter(parameter_h_file).writelines(file_h)
        writers.CPPWriter(parameter_cc_file).writelines(file_cc)

        # Copy additional needed files
        for copy_file in self.copy_include_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.include_dir))
        # Copy additional needed files
        for copy_file in self.copy_cc_files:
            shutil.copy(os.path.join(_file_path, 'iolibs',
                                         'template_files',copy_file),
                        os.path.join(self.dir_path, self.cc_file_dir))

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(parameter_h_file)[-1],
                       os.path.split(parameter_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(parameter_h_file)[0],
                     os.path.split(parameter_cc_file)[0]))

    def generate_parameters_class_files(self):
        """Create the content of the Parameters_model.h and .cc files"""

        replace_dict = self.default_replace_dict

        replace_dict['info_lines'] = self.get_mg5_info_lines()
        replace_dict['model_name'] = self.model_name

        replace_dict['independent_parameters'] = \
                                   "// Model parameters independent of aS\n" + \
                                   self.write_parameters(self.params_indep)
        replace_dict['independent_couplings'] = \
                                    "// Model couplings independent of aS\n" + \
                                    self.write_parameters(self.coups_indep)
                                  
                                  
        replace_dict['dependent_parameters'] = \
                                    "// Model parameters dependent on aS\n" + \
                                    self.write_parameters(self.params_dep)
        replace_dict['dependent_couplings'] = \
                                   "// Model couplings dependent on aS\n" + \
                                   self.write_parameters(list(self.coups_dep.values()))

        replace_dict['flavor_independent_couplings'] = \
                                    "// Model flavor couplings independent of aS\n" + \
                                    self.write_parameters([c for c in self.coups_flv_indep])
        replace_dict['flavor_dependent_couplings'] = \
                                    "// Model flavor couplings dependent of aS\n" + \
                                    self.write_parameters([c for c in self.coups_flv_dep])                                    
        replace_dict['set_independent_parameters'] = \
                               self.write_set_parameters(self.params_indep)
        replace_dict['set_independent_couplings'] = \
                               self.write_set_parameters(self.coups_indep)
        replace_dict['set_dependent_parameters'] = \
                               self.write_set_parameters(self.params_dep)
        replace_dict['set_dependent_couplings'] = \
                               self.write_set_parameters(list(self.coups_dep.values()))
        # Only independent flavored couplings use the FLV_COUPLING value[] pointer mechanism;
        # dependent (running-alphas) ones are gathered event-by-event (see model_handling / Step 3).
        replace_dict['set_flv_couplings'] = \
                                self.write_flv_couplings(self.coups_flv_indep)

        replace_dict['print_independent_parameters'] = \
                               self.write_print_parameters(self.params_indep)
        replace_dict['print_independent_couplings'] = \
                               self.write_print_parameters(self.coups_indep)
        replace_dict['print_dependent_parameters'] = \
                               self.write_print_parameters(self.params_dep)
        replace_dict['print_dependent_couplings'] = \
                               self.write_print_parameters(list(self.coups_dep.values()))

        if 'include_prefix' not in replace_dict:
            replace_dict['include_prefix'] = ''


        file_h = self.read_template_file(self.param_template_h) % \
                 replace_dict
        file_cc = self.read_template_file(self.param_template_cc) % \
                  replace_dict
        
        return file_h, file_cc

    def write_parameters(self, params):
        """Write out the definitions of parameters"""

        # Create a dictionary from parameter type to list of parameter names
        type_param_dict = {}

        for param in params:
            if hasattr(param, 'type'):
                type_param_dict[param.type] = \
                    type_param_dict.setdefault(param.type, []) + [param.name]
            elif isinstance(param, base_objects.FLV_Coupling):
                type_param_dict['FLV_COUPLING'] = \
                    type_param_dict.setdefault('FLV_COUPLING', []) + [param.name]

        # For each parameter type, write out the definition string
        # type parameters;
        misc.sprint(type_param_dict)
        res_strings = []
        for key in type_param_dict:
            res_strings.append("%s %s;" % (self.type_dict[key],
                                          ",".join(type_param_dict[key])))

        return "\n".join(res_strings)

    def write_set_parameters(self, params):
        """Write out the lines of independent parameters"""

        # For each parameter, write name = expr;

        res_strings = []
        for param in params:
            res_strings.append("%s" % param.expr)

        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)        
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                res_strings.append("if (%s < 0)" % particle.get('mass'))
                res_strings.append("%(width)s = -abs(%(width)s);" % \
                                   {"width": particle.get('width')})

        return "\n".join(res_strings)

    def _assert_flv_couplings_supported(self, params):
        """Refuse, with a clear and actionable message, the merged-flavor
        coupling structures the C++ (mg7/standalone_mg7) backend cannot yet
        generate correctly, instead of crashing or emitting wrong/uncompilable
        code.

        Supported: one- and two-merged-leg "partner" vertices, with either
        flavor-*independent* or *dependent* (event-by-event, running-alphas)
        couplings. Single-merged-leg vertices (one merged fermion + an unmerged
        partner, e.g. the electroweak MSSM squark-quark-neutralino vertices) are
        serialized like the Fortran side (the unmerged partner is given flavor
        index 1) and gated by the merged leg (see get_coupling_def). Dependent
        flavored couplings (e.g. the SUSY-QCD MSSM gluino-squark-quark vertices)
        are gathered event-by-event into cDPF_* / flvCOUPs_dep (Step 3).

        Not yet supported (raises):

          * a vertex with more than two merged-flavor legs (never seen so far).

        The Fortran 'madevent'/'standalone' output supports the remaining cases.
        See docs/mg7_merged_flavor_mssm_design.md.
        """
        for coupl in params:
            for key in coupl.flavors:
                nb_merged = len([i for i in key if i != 0])
                if nb_merged in (1, 2):
                    continue
                raise InvalidCmd(
                    "merged-flavor C++ output (mg7/standalone_mg7) does not yet "
                    "support this process: flavor coupling %s connects %d "
                    "merged-flavor legs; only one or two are supported. Use "
                    "'output madevent' or 'output standalone' for this process. "
                    "See docs/mg7_merged_flavor_mssm_design.md for details."
                    % (coupl.name, nb_merged))

    def write_flv_couplings(self, params):
        """Write out the lines of independent parameters"""

        self._assert_flv_couplings_supported(params)
        def_flv = []
        # For each parameter, write name = expr;
        for coupl in params:
            for key, c in coupl.flavors.items():
                # Same (k1, k2) derivation as the Fortran/Python backends: for a
                # single merged leg the unmerged partner is flavor index 1 and
                # the PARTNER/PARTNER2 direction depends on which fermion carries
                # the merged leg (see FLV_Coupling.get_partner_indices).
                k1, k2 = base_objects.FLV_Coupling.get_partner_indices(key)
                def_flv.append('%(name)s.partner[%(in)i] = %(out)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.partner2[%(out)i] = %(in)i;' % {'name': coupl.name,'in': k1-1, 'out': k2-1})
                def_flv.append('%(name)s.val[%(in)i]  =  &%(coupl)s;' % {'name': coupl.name,'in': k1-1, 'coupl': c})

        return "\n".join(def_flv)


    def write_print_parameters(self, params):
        """Write out the lines of independent parameters"""

        # For each parameter, write name = expr;

        res_strings = []
        for param in params:
            res_strings.append("cout << setw(20) << \"%s \" << \"= \" << setiosflags(ios::scientific) << setw(10) << %s << endl;" % (param.name, param.name))

        return "\n".join(res_strings)

    # Routines for writing the ALOHA files

    def write_aloha_routines(self):
        """Generate the hel_amps_model.h and hel_amps_model.cc files, which
        have the complete set of generalized Helas routines for the model"""
        
        if not os.path.isdir(os.path.join(self.dir_path, self.include_dir)):
            os.makedirs(os.path.join(self.dir_path, self.include_dir))
        if not os.path.isdir(os.path.join(self.dir_path, self.cc_file_dir)):
            os.makedirs(os.path.join(self.dir_path, self.cc_file_dir))

        model_h_file = os.path.join(self.dir_path, self.include_dir,
                                    'HelAmps_%s.h' % self.model_name)
        model_cc_file = os.path.join(self.dir_path, self.cc_file_dir,
                                     'HelAmps_%s.%s' % (self.model_name, self.cc_ext))

        replace_dict = {}

        replace_dict['output_name'] = self.output_name
        replace_dict['info_lines'] = self.get_mg5_info_lines()
        replace_dict['namespace'] = self.namespace
        replace_dict['model_name'] = self.model_name

        # Read in the template .h and .cc files, stripped of compiler
        # commands and namespaces
        import aloha
        if aloha.unitary_gauge == 3:
            template_h_files = self.read_aloha_template_files(ext = 'fd_h')
            template_cc_files = self.read_aloha_template_files(ext = 'fd_cc')
        else:
            template_h_files = self.read_aloha_template_files(ext = 'h')
            template_cc_files = self.read_aloha_template_files(ext = 'cc')

        aloha_model = create_aloha.AbstractALOHAModel(self.model.get('name'),
                                                      explicit_combine=True)
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        
        if self.wanted_lorentz:
            aloha_model.compute_subset(self.wanted_lorentz)
        else:
            aloha_model.compute_all(save=False, custom_propa=True)
            
        for abstracthelas in dict(aloha_model).values():
            h_rout, cc_rout = abstracthelas.write(output_dir=None, 
                                                  language=self.aloha_writer, 
                                                  mode='no_include')

            template_h_files.append(h_rout)
            template_cc_files.append(cc_rout)
            
            #aloha_writer = aloha_writers.ALOHAWriterForCPP(abstracthelas,
            #                                               self.dir_path)
            #header = aloha_writer.define_header()
            #template_h_files.append(self.write_function_declaration(\
            #                             aloha_writer, header))
            #template_cc_files.append(self.write_function_definition(\
            #                              aloha_writer, header))

        replace_dict['function_declarations'] = '\n'.join(template_h_files)
        replace_dict['function_definitions'] = '\n'.join(template_cc_files)

        file_h = self.read_template_file(self.aloha_template_h) % replace_dict
        file_cc = self.read_template_file(self.aloha_template_cc) % replace_dict

        # Write the files
        writers.CPPWriter(model_h_file).writelines(file_h)
        writers.CPPWriter(model_cc_file).writelines(file_cc)

        logger.info("Created files %s and %s in directory" \
                    % (os.path.split(model_h_file)[-1],
                       os.path.split(model_cc_file)[-1]))
        logger.info("%s and %s" % \
                    (os.path.split(model_h_file)[0],
                     os.path.split(model_cc_file)[0]))


    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        template_files = []
        for filename in misc.glob('*.%s' % ext, pjoin(MG5DIR, 'aloha','template_files')):
            file = open(filename, 'r')
            template_file_string = ""
            while file:
                line = file.readline()
                if len(line) == 0: break
                line = self.clean_line(line)
                if not line:
                    continue
                template_file_string += line.strip() + '\n'
            template_files.append(template_file_string)

        return template_files

#    def write_function_declaration(self, aloha_writer, header):
#        """Write the function declaration for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_h(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)
#
#    def write_function_definition(self, aloha_writer, header):
#        """Write the function definition for the ALOHA routine"""
#
#        ret_lines = []
#        for line in aloha_writer.write_cc(header).split('\n'):
#            if self.compiler_option_re.match(line) or self.namespace_re.match(line):
#                # Strip out compiler flags and namespaces
#                continue
#            ret_lines.append(line)
#        return "\n".join(ret_lines)

    def clean_line(self, line):
        """Strip a line of compiler options and namespace options."""

        if self.compiler_option_re.match(line) or self.namespace_re.match(line):
            return ""

        return line

    def get_mg5_info_lines(self):
        """Return info lines for MG5, suitable to place at beginning of
        Fortran files"""

        return OneProcessExporterCPP.get_mg5_info_lines()

    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
         
        return OneProcessExporterCPP.read_template_file(filename, classpath)

#===============================================================================
# UFOModelConverterGPU
#===============================================================================

class UFOModelConverterGPU(UFOModelConverterCPP):
    
    aloha_writer = 'cudac'
    cc_ext = 'cu'
        # Template files to use
    #include_dir = '.'
    #c_file_dir = '.'
    #param_template_h = 'cpp_model_parameters_h.inc'
    #param_template_cc = 'cpp_model_parameters_cc.inc'
    aloha_template_h = pjoin('gpu','cpp_hel_amps_h.inc')
    aloha_template_cc = pjoin('gpu','cpp_hel_amps_cc.inc')
    helas_h = pjoin('gpu', 'helas.h')
    helas_cc = pjoin('gpu', 'helas.cu')

    def read_aloha_template_files(self, ext):
        """Read all ALOHA template files with extension ext, strip them of
        compiler options and namespace options, and return in a list"""

        path = pjoin(MG5DIR, 'aloha','template_files')
        out = []
        
        if ext == 'h':
            out.append(open(pjoin(path, self.helas_h)).read())
        else:
            out.append(open(pjoin(path, self.helas_cc)).read())
    
        return out

    def write_process_h_file(self, writer):
        
        replace_dict = UFOModelConverterCPP.write_process_h_file(self, None)
        replace_dict['include_for_complex'] = '#include "mgOnGpuTypes.h"'
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict

class OneProcessExporterCPP(object):
    """Class to take care of exporting a set of matrix elements to
    C++ format."""


    # Static variables (for inheritance)
    process_dir = '.'
    include_dir = '.'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files')
    _template_path = os.path.join(_file_path, 'iolibs', 'template_files')
    process_template_h = 'cpp_process_h.inc'
    process_template_cc = 'cpp_process_cc.inc'
    process_class_template = 'cpp_process_class.inc'
    process_definition_template = 'cpp_process_function_definitions.inc'
    process_wavefunction_template = 'cpp_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'cpp_process_sigmaKin_function.inc'
    single_process_template = 'cpp_process_matrix.inc'
    cc_ext = 'cc'
    support_multichannel = False
    imaginary_unit = "std::complex<double>(0,1)"
    use_flavor_mask = True

    class ProcessExporterCPPError(Exception):
        pass
    
    def __init__(self, matrix_elements, cpp_helas_call_writer, process_string = "",
                 process_number = 0, path = os.getcwd(), prefix=""):
        """Initiate with matrix elements, helas call writer, process
        string, path. Generate the process .h and .cc files."""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            self.matrix_elements = matrix_elements.get('matrix_elements')
        elif isinstance(matrix_elements, helas_objects.HelasMatrixElement):
            self.matrix_elements = \
                         helas_objects.HelasMatrixElementList([matrix_elements])
        elif isinstance(matrix_elements, helas_objects.HelasMatrixElementList):
            self.matrix_elements = matrix_elements
        else:
            raise base_objects.PhysicsObject.PhysicsObjectError("Wrong object type for matrix_elements: %s" % type(matrix_elements))

        if not self.matrix_elements:
            raise MadGraph5Error("No matrix elements to export")

        self._original_wf_numbers = []
        self._original_amp_numbers = []
        seen = set()
        for me in self.matrix_elements:
            for wf in me.get_all_wavefunctions():
                if id(wf) not in seen:
                    self._original_wf_numbers.append((wf, wf.get('number')))
                    seen.add(id(wf))
            for amp in me.get_all_amplitudes():
                if id(amp) not in seen:
                    self._original_amp_numbers.append((amp, amp.get('number')))
                    seen.add(id(amp))

        self.model = self.matrix_elements[0].get('processes')[0].get('model')
        self.model_name = ProcessExporterCPP.get_model_name(self.model.get('name'))

        self.processes = sum([me.get('processes') for \
                              me in self.matrix_elements], [])
        self.processes.extend(sum([me.get_mirror_processes() for \
                              me in self.matrix_elements], []))

        self.nprocesses = len(self.matrix_elements)
        if any([m.get('has_mirror_process') for m in self.matrix_elements]):
            self.nprocesses = 2*len(self.matrix_elements)

        if process_string:
            self.process_string = process_string
        else:
            self.process_string = self.processes[0].base_string()

        if process_number:
            self.process_number = process_number
        else:
            self.process_number = self.processes[0].get('id')

        self.process_name = self.get_process_name()
        self.process_class = "CPPProcess"
        # Emit the crossing-symmetry machinery (extended flavor_id carrying a
        # crossing). Off by default; ProcessExporterCPP.generate_subprocess_-
        # directory turns it on for standalone_cpp when --use_crossing is set.
        self.use_crossing = False

        self.path = path
        self.helas_call_writer = cpp_helas_call_writer

        if not isinstance(self.helas_call_writer, helas_call_writers.CPPUFOHelasCallWriter):
            raise self.ProcessExporterCPPError("helas_call_writer not CPPUFOHelasCallWriter")

        self.nexternal, self.ninitial = \
                        self.matrix_elements[0].get_nexternal_ninitial()
        self.nfinal = self.nexternal - self.ninitial

        # Check if we can use the same helicities for all matrix
        # elements
        
        self.single_helicities = True

        hel_matrix = self.get_helicity_matrix(self.matrix_elements[0])

        for me in self.matrix_elements[1:]:
            if self.get_helicity_matrix(me) != hel_matrix:
                self.single_helicities = False

        if self.single_helicities:
            # If all processes have the same helicity structure, this
            # allows us to reuse the same wavefunctions for the
            # different processes
            
            self.wavefunctions = []
            wf_number = 0

            for me in self.matrix_elements:
                for iwf, wf in enumerate(me.get_all_wavefunctions()):
                    try:
                        old_wf = \
                               self.wavefunctions[self.wavefunctions.index(wf)]
                        wf.set('number', old_wf.get('number'))
                    except ValueError:
                        wf_number += 1
                        wf.set('number', wf_number)
                        self.wavefunctions.append(wf)

            # Also combine amplitudes
            self.amplitudes = helas_objects.HelasAmplitudeList()
            amp_number = 0
            for me in self.matrix_elements:
                for iamp, amp in enumerate(me.get_all_amplitudes()):
                    try:
                        old_amp = \
                               self.amplitudes[self.amplitudes.index(amp)]
                        amp.set('number', old_amp.get('number'))
                    except ValueError:
                        amp_number += 1
                        amp.set('number', amp_number)
                        self.amplitudes.append(amp)
            diagram = helas_objects.HelasDiagram({'amplitudes': self.amplitudes})
            self.amplitudes = helas_objects.HelasMatrixElement({\
                'diagrams': helas_objects.HelasDiagramList([diagram])})


            self.include_multi_channel = False
    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename, classpath=False):
        """Open a template file and return the contents."""
         
        if isinstance(filename, tuple):
            file_path = filename[0]
            filename = filename[1]
        elif isinstance(filename, str):
            if classpath:
                file_path = cls._template_path
            else:
                file_path = cls.template_path
        else:
            raise MadGraph5Error('Argument should be string or tuple.')
        
        return open(os.path.join(file_path, filename)).read()
        

    @staticmethod
    def get_mg5_info_lines():
        info = misc.get_pkg_info()
        info_lines = ""
        if info and 'version' in info and  'date' in info:
            info_lines = "//  MadGraph5_aMC@NLO v. %s, %s\n" % \
                         (info['version'], info['date'])
            info_lines = info_lines + \
                         "//  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "//  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"
        else:
            info_lines = "//  MadGraph5_aMC@NLO\n" + \
                         "//  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "//  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"        

        return info_lines
        
                  
    @staticmethod
    def get_multi_channel_dictionary(matrix_element, config_map):

        return ProcessExporterFortran.get_multi_channel_dictionary(matrix_element, config_map)

    # Methods for generation of process files for C++
    def generate_process_files(self):
        """Generate the .h and .cc files needed for C++, for the
        processes described by multi_matrix_element"""

        try:
            # Create the files
            if not os.path.isdir(os.path.join(self.path, self.include_dir)):
                os.makedirs(os.path.join(self.path, self.include_dir))
            filename = os.path.join(self.path, self.include_dir,
                                    '%s.h' % self.process_class)

            self.write_process_h_file(writers.CPPWriter(filename))

            if not os.path.isdir(os.path.join(self.path, self.process_dir)):
                os.makedirs(os.path.join(self.path, self.process_dir))
            filename = os.path.join(self.path, self.process_dir,
                                    '%s.%s' % (self.process_class, self.cc_ext))
            self.write_process_cc_file(writers.CPPWriter(filename))

            logger.info('Created files %(process)s.h and %(process)s.cc in' % \
                        {'process': self.process_class} + \
                        ' directory %(dir)s' % {'dir': os.path.split(filename)[0]})
        finally:
            self.restore_original_numbering()

    def generate_process_files_madevent(self, proc_id, config_map, subproc_number):


        self.include_multi_channel = config_map
        self.generate_process_files() 
#        raise Exception("working fine but not fully implemented so far")

    def restore_original_numbering(self):
        for wf, number in self._original_wf_numbers:
            wf.set('number', number)
            wf.set('me_id', number)
        for amp, number in self._original_amp_numbers:
            amp.set('number', number)


    def get_default_converter(self):
        
        replace_dict = {}       

        
        return replace_dict
    
    #===========================================================================
    # write_process_h_file
    #===========================================================================
    def write_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""
        
        if writer and not isinstance(writer, writers.CPPWriter):
            raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = self.get_default_converter()

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract model name
        replace_dict['model_name'] = \
                         self.model_name

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract class definitions
        process_class_definitions = self.get_process_class_definitions()
        replace_dict['process_class_definitions'] = process_class_definitions
        replace_dict['include_for_complex'] = ''

        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    #===========================================================================
    # write_process_cc_file
    #===========================================================================
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""

        if writer:
            if not isinstance(writer, writers.CPPWriter):
                raise writers.CPPWriter.CPPWriterError(\
                "writer not CPPWriter")

        replace_dict = self.get_default_converter()

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process file name
        replace_dict['process_file_name'] = self.process_name

        # Extract model name
        replace_dict['model_name'] = self.model_name
                         

        # Extract class function definitions
        process_function_definitions = \
                         self.get_process_function_definitions()
        replace_dict['process_function_definitions'] = \
                                                   process_function_definitions

        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict

    #===========================================================================
    # Process export helper functions
    #===========================================================================
    def get_process_class_definitions(self, write=True):
        """The complete class definition for the process"""

        replace_dict = {}
        # Default (no-crossing) fill; overridden in the single_helicities branch.
        replace_dict['cross_member_decl'] = ''

        # Extract model name
        replace_dict['model_name'] = self.model_name

        # Extract process info lines for all processes
        process_lines = "\n".join([self.get_process_info_lines(me) for me in \
                                   self.matrix_elements])
        
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        replace_dict['nfinal'] = self.nfinal

        # Extract number of external particles
        replace_dict['ninitial'] = self.ninitial

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        # Extract process definition
        process_definition = "%s (%s)" % (self.process_string,
                                          self.model_name)
        replace_dict['process_definition'] = process_definition

        process = self.processes[0]

        replace_dict['process_code'] = self.process_number
        replace_dict['nexternal'] = self.nexternal
        replace_dict['nprocesses'] = self.nprocesses
        

        color_amplitudes = self.matrix_elements[0].get_color_amplitudes(
            merge_quartic_amplitudes=False)
        # Number of color flows
        replace_dict['ncolor'] = len(color_amplitudes)

        if self.single_helicities:
            wfct_size = 18
            # Set the size of Wavefunction
            if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
                wfct_size = 18
            else:
                wfct_size = 6
            
            replace_dict['wfct_size'] = wfct_size
            
            cross_repl = self.get_crossing_replace_dict(self.matrix_elements[0])
            replace_dict['cross_member_decl'] = cross_repl['cross_member_decl']
            replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          void calculate_wavefunctions(const int perm[], const int hel[], const int flavor[]%(cross_cw_sig_extra)s);
                          static const int nwavefuncs = %(nwfct)d;
                          MG5_%(model_name)s::ALOHAOBJ w[nwavefuncs];
                          """ % \
                          {'nwfct':len(self.wavefunctions),
                          'sizew': wfct_size,
                          'model_name': self.model_name,
                          'cross_cw_sig_extra': cross_repl['cross_cw_sig_extra'],
                          }

            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s();" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

            replace_dict['ncomb'] = (
                self.matrix_elements[0].get_helicity_combinations()
            )
            replace_dict['namp'] = len(
                self.amplitudes.get_all_amplitudes()
            )
            replace_dict['ndiag'] = len(
                self.matrix_elements[0].get("diagrams")
            )
            replace_dict['nflav'] = len(
                self.matrix_elements[0].get_external_flavors_with_iden()
            )

        else:
            replace_dict['all_sigma_kin_definitions'] = \
                          "\n".join(["void sigmaKin_%s(int* flavor );" % \
                                     me.get('processes')[0].shell_string().\
                                     replace("0_", "") \
                                     for me in self.matrix_elements])
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s(const int hel[]);" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        if write:
            file = self.read_template_file(self.process_class_template) % replace_dict
            return file
        else:
            return replace_dict
        
    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = {}

        # Extract model name
        replace_dict['model_name'] = self.model_name

        # Extract process info lines
        replace_dict['process_lines'] = \
                             "\n".join([self.get_process_info_lines(me) for \
                                        me in self.matrix_elements])

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        color_amplitudes = [me.get_color_amplitudes(merge_quartic_amplitudes=False) for me in \
                            self.matrix_elements]

        replace_dict['initProc_lines'] = \
                                self.get_initProc_lines(self.matrix_elements[0],
                                                        color_amplitudes)
        replace_dict['reset_jamp_lines'] = \
                                     self.get_reset_jamp_lines(color_amplitudes)
        replace_dict['sigmaKin_lines'], other_replace = \
                                     self.get_sigmaKin_lines(color_amplitudes)
        replace_dict.update(other_replace)
            
        replace_dict['sigmaHat_lines'] = self.get_sigmaHat_lines()

        replace_dict['all_sigmaKin'] = \
                                  self.get_all_sigmaKin_lines(color_amplitudes,
                                                              'CPPProcess')
        
        replace_dict['nexternal'] = len(self.matrix_elements[0].get('processes')[0].get('legs'))
        _, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        process = self.matrix_elements[0].get('processes')[0]
        sym_data = ProcessExporterFortran._get_broken_symmetry_data(process, nincoming)
        ProcessExporterFortran._fill_broken_sym_replace_dict(replace_dict, sym_data)

        # ident_cross() companion of broken_sym() (empty unless crossing is on).
        replace_dict['ident_cross_function'] = \
            self.get_crossing_replace_dict(self.matrix_elements[0])['ident_cross_function']

        if write:
            file = self.read_template_file(self.process_definition_template) %\
               replace_dict
            return file
        else:
            return replace_dict

    def get_process_name(self):
        """Return process file name for the process in matrix_element"""

        process_string = self.process_string

        # Extract process number
        proc_number_pattern = re.compile(r"^(.+)@\s*(\d+)\s*(.*)$")
        proc_number_re = proc_number_pattern.match(process_string)
        proc_number = 0
        if proc_number_re:
            proc_number = int(proc_number_re.group(2))
            process_string = proc_number_re.group(1) + \
                             proc_number_re.group(3)

        # Remove order information
        order_pattern = re.compile(r"^(.+)\s+(\w+)\s*=\s*(\d+)\s*$")
        order_re = order_pattern.match(process_string)
        while order_re:
            process_string = order_re.group(1)
            order_re = order_pattern.match(process_string)
        
        process_string = process_string.replace(' ', '')
        process_string = process_string.replace('>', '_')
        process_string = process_string.replace('+', 'p')
        process_string = process_string.replace('-', 'm')
        process_string = process_string.replace('~', 'x')
        process_string = process_string.replace('/', '_no_')
        process_string = process_string.replace('$', '_nos_')
        process_string = process_string.replace('|', '_or_')
        if proc_number != 0:
            process_string = "%d_%s" % (proc_number, process_string)

        #process_string = "Sigma_%s_%s" % (self.model_name,
                                          #process_string)
        return process_string

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "# " + process.nice_string().replace('\n', '\n# * ') \
                         for process in matrix_element.get('processes')])


    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars.%s);" % part.get('mass'))
        #for i, colamp in enumerate(color_amplitudes):
        #    initProc_lines.append("jamp2 = new double[%d];" % \
        #                          (i, len(colamp)))
        initProc_lines.append("jamp2 = new double[%d];" % len(color_amplitudes[0]))

        return "\n".join(initProc_lines)

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        #for icol, col_amp in enumerate(color_amplitudes):
        #    ret_lines+= """for(int i=0;i < %(ncolor)d; i++)
        #    jamp2[%(proc_number)d][i]=0.;\n""" % \
        #    {"ncolor": len(col_amp), "proc_number": icol}
        for icol, col_amp in enumerate(color_amplitudes):
            ret_lines+= """for(int i=0;i < %(ncolor)d; i++)
            jamp2[i]=0.;\n""" % \
            {"ncolor": len(col_amp), "proc_number": icol}
        return ret_lines
        

    def get_calculate_wavefunctions(self, wavefunctions, amplitudes, write=True):
        """Return the lines for optimized calculation of the
        wavefunctions for all subprocesses"""

        replace_dict = {}

        replace_dict['nwavefuncs'] = len(wavefunctions)
        replace_dict['flavor_mask_decl'] = ''
        replace_dict['flavor_mask_setup'] = ''
        
        #ensure no recycling of wavefunction ! incompatible with some output
        for me in self.matrix_elements:
            me.restore_original_wavefunctions()

        if len(self.matrix_elements) == 1:
            mask_decl, mask_setup, n_flavors, active_flavor_mask = \
                    self.get_flavor_mask_blocks(self.matrix_elements[0])
            replace_dict['flavor_mask_decl'] = mask_decl
            replace_dict['flavor_mask_setup'] = mask_setup
        else:
            n_flavors = 0
            active_flavor_mask = None

        self.helas_call_writer.use_flavor_mask = (n_flavors > 0)
        self.helas_call_writer.me_n_flavors = n_flavors
        self.helas_call_writer.me_active_flavor_mask = active_flavor_mask
        # When crossing is on, the external HELAS calls must permute the
        # helicity through perm[] and multiply their NSF flag by ic[] (both set
        # up by sigmaKin); mirror of the fortran use_crossing_ic gate.
        self.helas_call_writer.use_crossing_ic = getattr(self, 'use_crossing', False)
        try:
            replace_dict['wavefunction_calls'] = "\n".join(\
                self.helas_call_writer.get_wavefunction_calls(\
                helas_objects.HelasWavefunctionList(wavefunctions)))

            replace_dict['amplitude_calls'] = "\n".join(\
                self.helas_call_writer.get_amplitude_calls(amplitudes))
        finally:
            self.helas_call_writer.use_flavor_mask = False
            self.helas_call_writer.me_n_flavors = 0
            self.helas_call_writer.me_active_flavor_mask = None
            self.helas_call_writer.use_crossing_ic = False

        if write:
            file = self.read_template_file(self.process_wavefunction_template) % \
                replace_dict
            return file
        else:
            return replace_dict

    def _cpp_flav_rows(self, matrix_element, allowed_flavors):
        """Return the per-flavor group-position rows (as C++ initialiser
        strings like '{0, 0, 1, 1}') for *allowed_flavors*. Group positions are
        0-based, matching the convention used by the C++ flavor-mask table and
        the HELAS ixxxxx/oxxxxx flv argument."""
        model = matrix_element.get('processes')[0].get('model')
        merged_particles = model.get('merged_particles') or {}
        pdg_to_group_index = {}
        max_group_size = 0
        for merged_id, members in merged_particles.items():
            members = list(members)
            if members:
                max_group_size = max(max_group_size, len(members))
                pdg_to_group_index[int(merged_id)] = 0
                pdg_to_group_index[-int(merged_id)] = 0
            for pos, pdg in enumerate(members):
                pdg = int(pdg)
                pdg_to_group_index[pdg] = pos
                pdg_to_group_index[-pdg] = pos
        flav_rows = []
        for flavor in allowed_flavors:
            row = []
            for p in flavor:
                p = int(p)
                if p in pdg_to_group_index:
                    row.append(str(pdg_to_group_index[p]))
                elif abs(p) in pdg_to_group_index:
                    row.append(str(pdg_to_group_index[abs(p)]))
                elif max_group_size and 1 <= abs(p) <= max_group_size:
                    row.append(str(abs(p) - 1))
                else:
                    row.append('0')
            flav_rows.append('{%s}' % ', '.join(row))
        return flav_rows

    def _cpp_sigmakin_flavor(self, matrix_element):
        """Return (n_flavors, flav_rows, n_legs) for the always-on per-flavor
        good-helicity filter in sigmaKin. n_flavors is always >= 1: an ME with
        no merged-particle variants is a single flavor whose group index is 0 on
        every leg (C++ group indices are 0-based), matching the flavor[] = 0
        convention the callers use for an unmerged leg. n_legs is the number of
        external legs (the length of each flav_table row), returned explicitly
        so callers need not infer it from the initialiser string."""
        nexternal = matrix_element.get_nexternal_ninitial()[0]
        allowed_flavors = matrix_element.compute_flavor_masks()
        if not allowed_flavors:
            return (1, ['{%s}' % ', '.join(['0'] * nexternal)], nexternal)
        return (len(allowed_flavors),
                self._cpp_flav_rows(matrix_element, allowed_flavors),
                len(allowed_flavors[0]))

    def get_flavor_mask_blocks(self, matrix_element):
        """Return declaration/setup blocks for C++ flavor-mask guards."""

        if not getattr(self, 'use_flavor_mask', False):
            return ('', '', 0, 0)

        allowed_flavors = matrix_element.compute_flavor_masks()
        if not allowed_flavors:
            return ('', '', 0, 0)

        if matrix_element.flavor_mask_is_trivial():
            return ('', '', len(allowed_flavors), (1 << len(allowed_flavors)) - 1)

        n_flavors = len(allowed_flavors)
        n_wfs = matrix_element.get_number_of_wavefunctions()
        n_amps = matrix_element.get_number_of_amplitudes()
        nwords_wf = max(1, (n_wfs + 63) // 64)
        nwords_amp = max(1, (n_amps + 63) // 64)

        wf_masks = [0] * n_wfs
        amp_masks = [0] * n_amps
        for wf in matrix_element.get_all_wavefunctions():
            idx = wf.get('number')
            if isinstance(idx, int) and idx > 0:
                wf_masks[idx - 1] = wf['flavor_mask'] if 'flavor_mask' in wf else 0
        for amp in matrix_element.get_all_amplitudes():
            idx = amp.get('number')
            if isinstance(idx, int) and idx > 0:
                amp_masks[idx - 1] = amp['flavor_mask'] if 'flavor_mask' in amp else 0

        active_flavor_mask = 0
        for amp_mask in amp_masks:
            active_flavor_mask |= amp_mask
        if active_flavor_mask == 0:
            active_flavor_mask = (1 << n_flavors) - 1

        wf_index_masks = [[0] * nwords_wf for _ in range(n_flavors)]
        amp_index_masks = [[0] * nwords_amp for _ in range(n_flavors)]
        for flav_idx in range(n_flavors):
            bit = 1 << flav_idx
            for obj_idx, mask in enumerate(wf_masks):
                if mask & bit:
                    word = obj_idx // 64
                    pos = obj_idx % 64
                    wf_index_masks[flav_idx][word] |= (1 << pos)
            for obj_idx, mask in enumerate(amp_masks):
                if mask & bit:
                    word = obj_idx // 64
                    pos = obj_idx % 64
                    amp_index_masks[flav_idx][word] |= (1 << pos)

        active_wf_index_masks = [0] * nwords_wf
        active_amp_index_masks = [0] * nwords_amp
        for flav_mask in wf_index_masks:
            for word, value in enumerate(flav_mask):
                active_wf_index_masks[word] |= value
        for flav_mask in amp_index_masks:
            for word, value in enumerate(flav_mask):
                active_amp_index_masks[word] |= value

        def fmt_uint64_2d(dtype, name, matrix):
            rows = ['{%s}' % ', '.join('%dULL' % v for v in row) for row in matrix]
            return '%s %s[%d][%d] = {%s};' % (
                dtype, name, len(matrix), len(matrix[0]), ', '.join(rows))

        model = matrix_element.get('processes')[0].get('model')
        merged_particles = model.get('merged_particles') or {}
        pdg_to_group_index = {}
        max_group_size = 0
        for merged_id, members in merged_particles.items():
            members = list(members)
            if members:
                max_group_size = max(max_group_size, len(members))
                # C++ flavor indices are 0-based (first merged member -> 0).
                pdg_to_group_index[int(merged_id)] = 0
                pdg_to_group_index[-int(merged_id)] = 0
            for pos, pdg in enumerate(members):
                pdg = int(pdg)
                pdg_to_group_index[pdg] = pos
                pdg_to_group_index[-pdg] = pos

        flav_rows = []
        for flavor in allowed_flavors:
            row = []
            for p in flavor:
                p = int(p)
                if p in pdg_to_group_index:
                    row.append(str(pdg_to_group_index[p]))
                elif abs(p) in pdg_to_group_index:
                    row.append(str(pdg_to_group_index[abs(p)]))
                elif max_group_size and 1 <= abs(p) <= max_group_size:
                    row.append(str(abs(p) - 1))
                else:
                    row.append('0')
            flav_rows.append('{%s}' % ', '.join(row))
        decl_lines = [
            '// Flavor-mask machinery (compute_flavor_masks).',
            'const int nmask_flav = %d;' % n_flavors,
            'const int nwords_wf = %d;' % nwords_wf,
            'const int nwords_amp = %d;' % nwords_amp,
            fmt_uint64_2d('static const unsigned long long', 'wf_index_mask',
                          wf_index_masks),
            fmt_uint64_2d('static const unsigned long long', 'amp_index_mask',
                          amp_index_masks),
            'static const int flav_table[%d][%d] = {%s};' % (
                n_flavors, len(allowed_flavors[0]), ', '.join(flav_rows)),
            'static const unsigned long long active_wf_mask[%d] = {%s};' % (
                nwords_wf, ', '.join('%dULL' % v for v in active_wf_index_masks)),
            'static const unsigned long long active_amp_mask[%d] = {%s};' % (
                nwords_amp, ', '.join('%dULL' % v for v in active_amp_index_masks)),
            'unsigned long long current_wf_mask[nwords_wf];',
            'unsigned long long current_amp_mask[nwords_amp];',
        ]

        setup_lines = [
            'for (int mask_k = 0; mask_k < nwords_wf; ++mask_k) current_wf_mask[mask_k] = active_wf_mask[mask_k];',
            'for (int mask_k = 0; mask_k < nwords_amp; ++mask_k) current_amp_mask[mask_k] = active_amp_mask[mask_k];',
            'int flav_idx_lookup = -1;',
            'for (int mask_i = 0; mask_i < nmask_flav; ++mask_i) {',
            '  bool flav_match = true;',
            '  for (int mask_j = 0; mask_j < nexternal; ++mask_j) {',
            '    if (flavor[mask_j] != flav_table[mask_i][mask_j]) {',
            '      flav_match = false;',
            '      break;',
            '    }',
            '  }',
            '  if (flav_match) {',
            '    flav_idx_lookup = mask_i;',
            '    break;',
            '  }',
            '}',
            'if (flav_idx_lookup >= 0) {',
            '  for (int mask_k = 0; mask_k < nwords_wf; ++mask_k) current_wf_mask[mask_k] = wf_index_mask[flav_idx_lookup][mask_k];',
            '  for (int mask_k = 0; mask_k < nwords_amp; ++mask_k) current_amp_mask[mask_k] = amp_index_mask[flav_idx_lookup][mask_k];',
            '}',
        ]

        return ('\n'.join(decl_lines), '\n'.join(setup_lines),
                n_flavors, active_flavor_mask)
       

    @staticmethod
    def _cpp_int_array(values):
        """Flat C++ initialiser '{a, b, c}' for a list of ints."""
        return '{%s}' % ', '.join(str(int(v)) for v in values)

    @staticmethod
    def _cpp_int_array2d(flat, ncols):
        """Nested C++ initialiser '{{...}, {...}}' from a flat list, ncols wide."""
        rows = ['{%s}' % ', '.join(str(int(v)) for v in flat[i:i + ncols])
                for i in range(0, len(flat), ncols)]
        return '{%s}' % ', '.join(rows)

    def get_crossing_replace_dict(self, matrix_element):
        """Fill the crossing-machinery holes of the C++ standalone templates.

        Mirrors export_v4.fill_crossing_replace_dict for the standalone_cpp
        backend. When self.use_crossing is False every hole gets the plain,
        pre-crossing code so the output is byte-for-byte the old one; when it is
        True the extended flavor_id (a flavor AND a crossing) is decoded in
        sigmaKin, the momenta/helicities are permuted through the crossing and
        the swapped legs' NSF flag is flipped (via the ic[] array the HELAS
        calls now read), and the denominator is split into the crossing-
        dependent initial-state spin*color (spincol_cross) times the flavor-
        dependent identical-final-state factor (ident_cross).
        """
        # Plain (no-crossing) fills: identical to the historical template.
        plain = {
            'fidx': 'flavor_id',
            'cross_tables_decode': '',
            'cross_perm_block': ('int perm[nexternal];\n'
                                 'for(int i = 0; i < nexternal; i++){\n'
                                 '    perm[i]=i;\n'
                                 '}'),
            'cross_cw_args': '',
            'cross_return':
                'return matrix_element * broken_sym(flavor) / denominator;',
            'cross_cw_sig_extra': '',
            'cross_member_decl': '',
            'ident_cross_function': '',
            # No crossing: every call is the uncrossed process, so the C-parity
            # de-duplication is always allowed.
            'csym_dedup_ok': 'true',
            # Historical good-helicity filter (byte-identical to pre-crossing).
            'cross_ghidx_setup': '',
            'cross_goodhel_gate':
                'goodhel[flavor_id][ihel] || ntry[flavor_id] < 2',
            'cross_goodhel_train':
                'if (t != 0. && !goodhel[flavor_id][ihel]){\n'
                '                goodhel[flavor_id][ihel]=true;\n'
                '                ngood[flavor_id] ++;\n'
                '                igood[flavor_id][ngood[flavor_id]] = ihel;\n'
                '            }',
        }
        if not self.use_crossing:
            return plain

        tables = ProcessExporterFortran.compute_crossing_tables(
            self, matrix_element)
        nexternal = tables['nexternal']
        ninitial = tables['ninitial']
        ncross = (nexternal + 1) * (nexternal + 1)

        # Per-leg tables (one entry per external leg). The crossing's slot
        # permutation and NSF sign flips are decoded from the crossing code at
        # runtime (cross_perm_ic, mirroring the fortran GET_CROSS_PERM), and the
        # two halves of the denominator are rebuilt from these -- so no
        # cross-indexed table (spincol/basepid/src/perm/ic) is stored.
        spincol_part_init = self._cpp_int_array(tables['spincol_part'])
        ids_base_init = self._cpp_int_array(tables['ids_base'])
        antipid_base_init = self._cpp_int_array(tables['antipid_base'])
        # Good-helicity remap: instead of the baked ghremap[ncross*ncomb] row
        # table, keep only the per-crossing filterable flag and resolve the
        # gating identity row at runtime (see cross_ghidx_setup) -- the same
        # NCROSS*NCOMB -> NCROSS shrink the fortran path does via CROSS_GHIDX.
        # allow_reverse False so it matches the order helicities[] is emitted in.
        ghfilt_init = self._cpp_int_array(
            ProcessExporterFortran.compute_ghfilt(
                self, matrix_element, allow_reverse=False))

        cross_tables_decode = (
            "// Crossing symmetry: flavor_id carries a flavor AND a crossing.\n"
            "//   cross    = flavor_id / nflavors\n"
            "//   flav_use = flavor_id %% nflavors  (index used for masking)\n"
            "// A crossing permutes momenta/helicities between slots and flips\n"
            "// each swapped leg's NSF flag. The slot permutation is a fixed\n"
            "// relabelling decoded from the crossing code at runtime\n"
            "// (cross_perm_ic), so no cross-indexed table is stored; the\n"
            "// denominator splits into the crossing-dependent initial-state\n"
            "// spin*color (spincol_cross) and the flavor-dependent identical-\n"
            "// final-state factor (ident_cross), both rebuilt from per-leg data.\n"
            "const int ncross = %(ncross)d;\n"
            "// ghfilt[cross] = 1 if this crossing's good-helicity filter is a\n"
            "// clean bijection of the identity rows, 0 otherwise (initial-\n"
            "// initial swap, inapplicable, or non-bijection). Genuinely per-\n"
            "// crossing (not derivable from per-leg data), so kept as a table --\n"
            "// the fortran path tabulates it too. The gating identity row itself\n"
            "// is recomputed per row at runtime (see the good-helicity loop).\n"
            "// See ProcessExporterFortran.compute_ghfilt.\n"
            "static const int ghfilt[ncross] = %(ghfilt)s;\n"
            "int cross = flavor_id / nflavors;\n"
            "int flav_use = flavor_id %% nflavors;\n"
            "// A null spin*color entry (out of range, impossible, or an\n"
            "// overlapping swap) means an identically-zero matrix element.\n"
            "if (cross < 0 || cross >= ncross || spincol_cross(cross) == 0)\n"
            "    return 0.;"
        ) % {'ncross': ncross, 'ghfilt': ghfilt_init}

        cross_perm_block = (
            "int perm[nexternal];\n"
            "int ic[nexternal];\n"
            "cross_perm_ic(cross, perm, ic);")

        cross_return = (
            "// Uncrossed: historical path (IDEN via denominator, BROKEN_SYM\n"
            "// correcting the identical-particle count per flavor). Crossed:\n"
            "// rebuild the denominator from the crossed initial-state spin*color\n"
            "// and the identical final-state factor of the actual flavors.\n"
            "if (cross == 0)\n"
            "    return matrix_element * broken_sym(flavor) / denominator;\n"
            "return matrix_element / "
            "(spincol_cross(cross) * ident_cross(cross, flavor));")

        ident_cross_function = (
            "//------------------------------------------------------------------\n"
            "// Runtime crossing decode (mirrors the fortran GET_CROSS_PERM/\n"
            "// SWAP_LEGS): cross = i*(nexternal+1) + j swaps particle 1 with i\n"
            "// and particle 2 with j (0 = leave alone; i==1 / j==2 are self-swaps,\n"
            "// also no-ops). perm[k] is the input slot landing in crossed slot k\n"
            "// and ic[k] its NSF sign flip. perm/ic are always left a valid\n"
            "// permutation (identity for an inapplicable code) so a momentum\n"
            "// gather never reads out of range; the return value flags an\n"
            "// applicable crossing (false = overlapping swap / out of range).\n"
            "bool CPPProcess::cross_perm_ic(int cross, int* perm, int* ic)\n"
            "{\n"
            "    const int ncross = (nexternal + 1) * (nexternal + 1);\n"
            "    for (int k = 0; k < nexternal; k++) { perm[k] = k; ic[k] = 1; }\n"
            "    if (cross < 0 || cross >= ncross) return false;\n"
            "    const int xi = cross / (nexternal + 1);\n"
            "    const int xj = cross %% (nexternal + 1);\n"
            "    // Overlapping-swap codes compose into a 3-cycle the consumers\n"
            "    // read with opposite orientation: pure redundancy, invalid.\n"
            "    if (xi != 0 && xi != 1 && xj != 0 && xj != 2 &&\n"
            "        (xi == 2 || xj == 1 || xi == xj)) return false;\n"
            "    if (xi != 0 && xi != 1)\n"
            "    {\n"
            "        int t = perm[0]; perm[0] = perm[xi - 1]; perm[xi - 1] = t;\n"
            "        ic[0] = -ic[0]; ic[xi - 1] = -ic[xi - 1];\n"
            "    }\n"
            "    if (xj != 0 && xj != 2)\n"
            "    {\n"
            "        int t = perm[1]; perm[1] = perm[xj - 1]; perm[xj - 1] = t;\n"
            "        ic[1] = -ic[1]; ic[xj - 1] = -ic[xj - 1];\n"
            "    }\n"
            "    // A crossing may only conjugate a leg that CHANGES SIDE. Both\n"
            "    // legs of a same-side transposition are conjugated while\n"
            "    // neither moves across, which is no crossing at all: for a\n"
            "    // 2 -> N process that is the beam swap (xi==2 / xj==1), which\n"
            "    // must not conjugate anything; for a 1 -> N one it is every xj\n"
            "    // swap. Mirrors the fortran GET_CROSS_PERM.\n"
            "    for (int k = 0; k < nexternal; k++)\n"
            "        if (ic[k] == -1 &&\n"
            "            ((k < %(ninitial)d) == (perm[k] < %(ninitial)d)))\n"
            "            return false;\n"
            "    return true;\n"
            "}\n"
            "\n"
            "//------------------------------------------------------------------\n"
            "// Initial-state spin*color average of the crossed process: the\n"
            "// product of the per-leg spin*color (spincol_part, conjugation\n"
            "// invariant) over the legs the crossing puts in the initial state.\n"
            "// 0 for a crossing that cannot be applied.\n"
            "int CPPProcess::spincol_cross(int cross)\n"
            "{\n"
            "    static const int spincol_part[nexternal] = %(spincol_part)s;\n"
            "    int perm[nexternal], ic[nexternal];\n"
            "    if (!cross_perm_ic(cross, perm, ic)) return 0;\n"
            "    int factor = 1;\n"
            "    for (int k = 0; k < %(ninitial)d; k++)\n"
            "        factor *= spincol_part[perm[k]];\n"
            "    return factor;\n"
            "}\n"
            "\n"
            "//------------------------------------------------------------------\n"
            "// Identical-final-state factor (product of n!) of the crossed\n"
            "// process. Flavor dependent, so computed at runtime: two crossed\n"
            "// final legs are identical when they carry the same flavor group\n"
            "// (same representative PDG -- ids_base, conjugated to antipid_base\n"
            "// when the leg swapped side) and the same actual flavor. FLAVOR is\n"
            "// not permuted by the crossing, so slot k reads flavor[perm[k]].\n"
            "int CPPProcess::ident_cross(int cross, const int* flavor)\n"
            "{\n"
            "    static const int ids_base[nexternal] = %(ids_base)s;\n"
            "    static const int antipid_base[nexternal] = %(antipid_base)s;\n"
            "    int perm[nexternal], ic[nexternal];\n"
            "    cross_perm_ic(cross, perm, ic);\n"
            "    int bpid[nexternal];\n"
            "    for (int k = 0; k < nexternal; k++)\n"
            "        bpid[k] = (ic[k] == 1) ? ids_base[perm[k]] : antipid_base[perm[k]];\n"
            "    bool used[nexternal];\n"
            "    for (int k = 0; k < nexternal; k++) used[k] = false;\n"
            "    int fact = 1;\n"
            "    for (int k = %(ninitial)d; k < nexternal; k++)\n"
            "    {\n"
            "        if (used[k]) continue;\n"
            "        int n = 1;\n"
            "        for (int l = k + 1; l < nexternal; l++)\n"
            "        {\n"
            "            if (used[l]) continue;\n"
            "            if (bpid[k] == bpid[l] &&\n"
            "                flavor[perm[k]] == flavor[perm[l]])\n"
            "            {\n"
            "                used[l] = true;\n"
            "                n = n + 1;\n"
            "                fact = fact * n;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return fact;\n"
            "}"
        ) % {'spincol_part': spincol_part_init, 'ids_base': ids_base_init,
             'antipid_base': antipid_base_init, 'ninitial': ninitial}

        return {
            'fidx': 'flav_use',
            'cross_tables_decode': cross_tables_decode,
            'cross_perm_block': cross_perm_block,
            'cross_cw_args': ', ic',
            'cross_return': cross_return,
            'cross_cw_sig_extra': ', const int ic[]',
            'cross_member_decl':
                '  bool cross_perm_ic(int cross, int* perm, int* ic);\n'
                '  int spincol_cross(int cross);\n'
                '  int ident_cross(int cross, const int* flavor);',
            'ident_cross_function': ident_cross_function,
            # C-parity de-duplication only for the uncrossed process (cross 0):
            # a crossing permutes/sign-flips the helicities so a base-row flip
            # is not the crossed C-parity partner (crossed flavors: full sum).
            'csym_dedup_ok': 'cross == 0',
            # The good-helicity filter is shared per flavor but consulted and
            # trained through the crossing's row permutation sigma^-1: a crossed
            # row is good iff its identity counterpart is. Rather than store the
            # whole sigma^-1 (ghremap[ncross*ncomb]), recompute the gating
            # identity row here: inverse-permute + sign-flip the crossed row's
            # config (perm/ic already hold the runtime-decoded cross_perm_ic),
            # then find the identity row carrying it. ghidx = -1 disables the
            # filter for a non-filterable crossing (ghfilt[cross] == 0: compute
            # the row, never train). For cross 0 perm/ic are the identity so
            # ghidx == ihel, exactly the historical filter. The search is only
            # reached while scanning (ntry < 10), so it is off the hot path.
            'cross_ghidx_setup':
                'int ghidx = -1;\n'
                '        if (ghfilt[cross]){\n'
                '            int tgt[nexternal];\n'
                '            for(int k = 0; k < nexternal; k++){\n'
                '                tgt[perm[k]] = ic[k] * helicities[ihel][k];\n'
                '            }\n'
                '            for(int r = 0; r < ncomb; r++){\n'
                '                bool same = true;\n'
                '                for(int k = 0; k < nexternal; k++){\n'
                '                    if (helicities[r][k] != tgt[k]){\n'
                '                        same = false;\n'
                '                    }\n'
                '                }\n'
                '                if (same){\n'
                '                    ghidx = r;\n'
                '                    break;\n'
                '                }\n'
                '            }\n'
                '        }\n'
                '        ',
            'cross_goodhel_gate':
                'ghidx < 0 || goodhel[flav_use][ghidx] || ntry[flav_use] < 2',
            'cross_goodhel_train':
                'if (t != 0. && ghidx >= 0 && !goodhel[flav_use][ghidx]){\n'
                '                goodhel[flav_use][ghidx]=true;\n'
                '                ngood[flav_use] ++;\n'
                '                igood[flav_use][ngood[flav_use]] = ihel;\n'
                '            }',
        }

    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""

        if self.include_multi_channel and not self.support_multichannel:
            raise Exception("This standalone format does not support madevent interface")

        
        if self.single_helicities:
            replace_dict = {}
            assert len(self.matrix_elements) == 1

            # Crossing-symmetry holes (identity fills when use_crossing is off).
            replace_dict.update(
                self.get_crossing_replace_dict(self.matrix_elements[0]))

            # Number of helicity combinations
            replace_dict['ncomb'] = \
                            self.matrix_elements[0].get_helicity_combinations()

            # Process name
            replace_dict['process_class_name'] = self.process_name
        
            # Particle ids for the call to setupForME
            replace_dict['id1'] = self.processes[0].get('legs')[0].get('id')
            replace_dict['id2'] = self.processes[0].get('legs')[1].get('id')

            # Extract helicity matrix
            replace_dict['helicity_matrix'] = \
                            self.get_helicity_matrix(self.matrix_elements[0])

            replace_dict['flavor_table'] = self.get_flavor_table(self.matrix_elements[0])

            # Extract denominator
            den_factors = [str(me.get_denominator_factor()) for me in \
                               self.matrix_elements]
            #if self.nprocesses != len(self.matrix_elements):
            #    den_factors.extend(den_factors)
            replace_dict['den_factors'] = ",".join(den_factors)
            replace_dict['get_matrix_t_lines'] = "\n".join(
                     ["double t = matrix_%(proc_name)s();" % \
                     {"iproc": i, "proc_name": \
                      me.get('processes')[0].shell_string().replace("0_", "")} \
                     for i, me in enumerate(self.matrix_elements)])

            # temporary
            replace_dict['madE_var_reset'] = ''
            replace_dict['madE_caclwfcts_call'] = ''
            replace_dict['madE_update_answer'] = ''



            # Generate lines for mirror matrix element calculation
            mirror_matrix_lines = ""

            if any([m.get('has_mirror_process') for m in self.matrix_elements]):
                mirror_matrix_lines += \
"""             // Mirror initial state momenta for mirror process
                perm[0]=1;
                perm[1]=0;
                int flv_tmp = flavor[0];
                flavor[0] = flavor[1];  
                flavor[1] = flv_tmp;
                // Calculate wavefunctions
                calculate_wavefunctions(perm, helicities[ihel], flavor);
                // Mirror back
                perm[0]=0;
                perm[1]=1;
                flavor[1] = flavor[0];
                flavor[0] = flv_tmp;
                // Calculate matrix elements
                """
                
                mirror_matrix_lines += "\n".join(
                    ["t[%(iproc)d]=matrix_%(proc_name)s();" % \
                     {"iproc": i + len(self.matrix_elements), "proc_name": \
                      me.get('processes')[0].shell_string().replace("0_", "")} \
                     for i, me in enumerate(self.matrix_elements) if me.get('has_mirror_process')])
                    
            replace_dict['get_mirror_matrix_lines'] = mirror_matrix_lines

            replace_dict['nproc'] = sum([ 2 if m.get('has_mirror_process') else 1
                                        for m in self.matrix_elements])
            replace_dict['nb_amp'] = len(self.amplitudes.get_all_amplitudes())
            replace_dict['nexternal'] = len(self.processes[0].get('legs'))

            # Always-on per-flavor good-helicity filter: goodhel/ntry/igood/
            # ngood/sum_hel/jhel are indexed by a per-point flav_idx resolved
            # from flavor[] via sk_flav_table, so a helicity that is zero for one
            # flavor is never dropped for another. nflav is >= 1 (an unmerged ME
            # is a single flavor); for nproc > 1 there is no single flavor table,
            # so fall back to one flavor (flav_idx stays 0).
            single_me = len(self.matrix_elements) == 1
            if single_me:
                sk_nflav, sk_flav_rows, n_legs = \
                    self._cpp_sigmakin_flavor(self.matrix_elements[0])
            else:
                n_legs = replace_dict['nexternal']
                sk_nflav, sk_flav_rows = \
                    (1, ['{%s}' % ', '.join(['0'] * n_legs)])
            replace_dict['cpp_goodhel_decl'] = (
                "const int nflav = %d;\n" % sk_nflav +
                "static const int sk_flav_table[nflav][%d] = {%s};\n"
                % (n_legs, ', '.join(sk_flav_rows)) +
                "static bool goodhel[nflav][ncomb] = {};\n"
                "static int ntry[nflav] = {}, sum_hel[nflav] = {}, ngood[nflav] = {};\n"
                "static int igood[nflav][ncomb];\n"
                "static int jhel[nflav];")
            # Resolve flavor[] -> flav_idx via sk_flav_table. For the single-ME
            # case a flavor absent from the table is not an allowed combination
            # (its |M|^2 is zero): flav_idx stays -1 and we short-circuit to a
            # zero matrix element before indexing the per-flavor goodhel/ntry
            # arrays. For the multi-ME fallback there is a single (all-zero) row
            # and no real per-flavor split, so flav_idx defaults to 0.
            if single_me:
                replace_dict['cpp_flav_idx_compute'] = (
                    "int flav_idx = -1;\n"
                    "for (int fi = 0; fi < nflav; ++fi) {\n"
                    "  bool fmatch = true;\n"
                    "  for (int fj = 0; fj < %d; ++fj) {\n" % n_legs +
                    "    if (flavor[fj] != sk_flav_table[fi][fj]) { fmatch = false; break; }\n"
                    "  }\n"
                    "  if (fmatch) { flav_idx = fi; break; }\n"
                    "}\n"
                    "if (flav_idx < 0) {\n"
                    "  for (int i = 0; i < nprocesses; i++) matrix_element[i] = 0.;\n"
                    "  return;\n"
                    "}\n")
            else:
                replace_dict['cpp_flav_idx_compute'] = (
                    "int flav_idx = 0;\n"
                    "for (int fi = 0; fi < nflav; ++fi) {\n"
                    "  bool fmatch = true;\n"
                    "  for (int fj = 0; fj < %d; ++fj) {\n" % n_legs +
                    "    if (flavor[fj] != sk_flav_table[fi][fj]) { fmatch = false; break; }\n"
                    "  }\n"
                    "  if (fmatch) { flav_idx = fi; break; }\n"
                    "}\n")

            if write:
                file = \
                 self.read_template_file(\
                            self.process_sigmaKin_function_template) %\
                            replace_dict
                return file, replace_dict
            else:
                return replace_dict
        else:
            ret_lines = "// Call the individual sigmaKin for each process\n"
            ret_lines = ret_lines + \
                   "\n".join(["sigmaKin_%s();" % \
                              me.get('processes')[0].shell_string().\
                              replace("0_", "") for \
                              me in self.matrix_elements])
            if write:
                return ret_lines, replace_dict
            else:
                replace_dict['get_mirror_matrix_lines'] = ret_lines
                return replace_dict

    def get_flavor_table(self, matrix_element):
        print(list(matrix_element.get_external_flavors()))
        flavors = list(matrix_element.get_external_flavors_with_iden())
        print(flavors)
        flavor_dict = {
            1: 0, 2: 1, 3: 2, 4: 3, # quarks
            11: 0, 13: 1, 15: 2,    # charged leptons
            12: 0, 14: 1, 16: 2,    # neutrinos
        }
        flavor_table = []
        flavor_mirror_table = []
        for flavor in flavors:
            aloha_flavor = [flavor_dict.get(abs(f), 0) for f in flavor[0]]
            flavor_table.append(",".join(str(f) for f in aloha_flavor))
        full_flavor_table = "{{" + "}, {".join(flavor_table) + "}}"
        flavor_count = len(flavor_table)
        ext_count = len(flavors[0][0])
        return f"""
        static const int flavor_table[{flavor_count}][{ext_count}] = {full_flavor_table};
        """
              
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            cross_cw_sig_extra = \
                self.get_crossing_replace_dict(self.matrix_elements[0])['cross_cw_sig_extra']
            ret_lines.append(\
                "void %s::calculate_wavefunctions(const int perm[], const int hel[], const int flavor[]%s){" % \
                (class_name, cross_cw_sig_extra))
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append(self.get_calculate_wavefunctions(\
                self.wavefunctions, self.amplitudes))
            ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)


    def get_sigmaKin_single_process(self, i, matrix_element, write=True):
        """Write sigmaKin for each process"""

        # Write sigmaKin for the process

        replace_dict = {}

        # Process name
        replace_dict['proc_name'] = \
          matrix_element.get('processes')[0].shell_string().replace("0_", "")
        
        # Process name
        replace_dict['process_class_name'] = self.process_name
        
        # Process number
        replace_dict['proc_number'] = i

        # Number of helicity combinations
        replace_dict['ncomb'] = matrix_element.get_helicity_combinations()

        # Extract helicity matrix
        replace_dict['helicity_matrix'] = \
                                      self.get_helicity_matrix(matrix_element)
        # Extract denominator
        replace_dict['den_factor'] = matrix_element.get_denominator_factor()
        

        if write:
            file = \
            self.read_template_file('cpp_process_sigmaKin_subproc_function.inc') %\
            replace_dict
            return file
        else:
            return replace_dict
        
    def get_matrix_single_process(self, i, matrix_element, color_amplitudes,
                                  class_name, write=True):
        """Write matrix() for each process"""

        # Write matrix() for the process

        replace_dict = {}

        # Process name
        replace_dict['proc_name'] = \
          matrix_element.get('processes')[0].shell_string().replace("0_", "")
        

        # Wavefunction and amplitude calls
        if self.single_helicities:
            replace_dict['matrix_args'] = ""
            replace_dict['all_wavefunction_calls'] = "" 
        else:
            replace_dict['matrix_args'] = "const int hel[]"
            wavefunctions = matrix_element.get_all_wavefunctions()
            replace_dict['all_wavefunction_calls'] = \
                         """const int nwavefuncs = %d;
                         std::complex<double> w[nwavefuncs][18];
                         """ % len(wavefunctions)+ \
                         self.get_calculate_wavefunctions(wavefunctions, [])

        # Process name
        replace_dict['process_class_name'] = class_name
        
        # Process number
        replace_dict['proc_number'] = i

        # Number of color flows
        replace_dict['ncolor'] = len(color_amplitudes)

        replace_dict['ngraphs'] = matrix_element.get_number_of_amplitudes()

        # Extract color matrix
        replace_dict['color_matrix_lines'] = \
                                     self.get_color_matrix_lines(matrix_element)

                                     
        replace_dict['jamp_lines'] = self.get_jamp_lines(color_amplitudes)

        # The color sum may run on a smaller basis than the one the color flow
        # is picked among (see the madmatrix override)
        self.set_color_flow_lines_cpp(matrix_element, replace_dict)

        replace_dict['amp2_lines'] = self.get_amp2_lines(matrix_element)

        #specific exporter hack
        replace_dict =  self.get_class_specific_definition_matrix(replace_dict, matrix_element)
        
        if write:
            file = self.read_template_file(self.single_process_template) % \
                replace_dict
            return file
        else:
            return replace_dict
        
    def get_class_specific_definition_matrix(self, converter, matrix_element):
        """place to add some specific hack to a given exporter.
        Please always use Super in that case"""

        return converter

    def get_sigmaHat_lines(self):
        """Get sigmaHat_lines for function definition for Pythia 8 .cc file"""

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])
        beams = sorted(list(beams))
        res_lines = []

        # Write a selection routine for the different processes with
        # the same beam particles
        res_lines.append("// Select between the different processes")
        for ibeam, beam_parts in enumerate(beams):
            
            if ibeam == 0:
                res_lines.append("if(id1 == %d && id2 == %d){" % beam_parts)
            else:
                res_lines.append("else if(id1 == %d && id2 == %d){" % beam_parts)            
            
            # Pick out all processes with this beam pair
            beam_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get('processes')]]

            # Add mirror processes, 
            beam_processes.extend([(len(self.matrix_elements) + i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get_mirror_processes()]])

            # Now add matrix elements for the processes with the right factors
            res_lines.append("// Add matrix elements for processes with beams %s" % \
                             repr(beam_parts))
            res_lines.append("return %s;" % \
                             ("+".join(["matrix_element[%i]*%i" % \
                                        (i, len([proc for proc in \
                                         me.get('processes') if beam_parts == \
                                         (proc.get('legs')[0].get('id'),
                                          proc.get('legs')[1].get('id')) or \
                                         me.get('has_mirror_process') and \
                                         beam_parts == \
                                         (proc.get('legs')[1].get('id'),
                                          proc.get('legs')[0].get('id'))])) \
                                        for (i, me) in beam_processes]).\
                              replace('*1', '')))
            res_lines.append("}")
            

        res_lines.append("else {")
        res_lines.append("// Return 0 if not correct initial state assignment")
        res_lines.append(" return 0.;}")

        return "\n".join(res_lines)


    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line = "static const int helicities[ncomb][nexternal] = {";
        helicity_line_list = []

        for helicities in matrix_element.get_helicity_matrix(allow_reverse=False):
            helicity_line_list.append("{"+",".join(['%d'] * len(helicities)) % \
                                       tuple(helicities) + "}")

        return helicity_line + ",".join(helicity_line_list) + "};"

    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""

        return "const int denominator = %d;" % \
               matrix_element.get_denominator_factor()

    def get_color_matrix_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["static const double denom = 1;",
                              "static const int cf[1] = {1};"])
        else:
            color_denominators = matrix_element.get('color_matrix').\
                                                 get_line_denominators()
            denominator = min(color_denominators)
            denom_string = "static const int denom = %i;" % (denominator)

            matrix_strings = []
            my_cs = color.ColorString()
            for index in range(len(color_denominators)):
                # Then write the numerators for the matrix elements
                num_list = matrix_element.get('color_matrix').\
                                            get_line_numerators(index, denominator)

                matrix_strings+= ["%d" % (i if pos==0 else 2*i) for pos,i in enumerate(num_list[index:])]
            matrix_string = "static const int cf[ncolor*(ncolor+1)/2] = {" + \
                            ",".join(matrix_strings) + "};"
            return "\n".join([denom_string, matrix_string])


    @classmethod
    def coeff(cls, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""
    
        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power
    
        if total_coeff == 1:
            if is_imaginary:
                return f'+{cls.imaginary_unit}*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return f'-{cls.imaginary_unit}*'
            else:
                return '-'
    
        res_str = '%+i.' % total_coeff.numerator
    
        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%i.' % total_coeff.denominator
    
        if is_imaginary:
            res_str = res_str + f'*{cls.imaginary_unit}'
    
        return res_str + '*'



            
    def set_color_flow_lines_cpp(self, matrix_element, replace_dict):
        """Tell the process template that the color sum and the color flow use
        the same basis. Overridden by the backends which can put the color sum
        on a smaller one."""

        replace_dict['ncolor_flow'] = replace_dict['ncolor']
        replace_dict['jampflow_lines'] = ''
        replace_dict['jamp_flow'] = 'jamp_sv'

    def get_jamp_lines(self, color_amplitudes):
        """Return the jamp = sum(fermionfactor * amp[i]) lines"""

        res_list = []

        for i, coeff_list in enumerate(color_amplitudes):

            res = "jamp[%i]=" % i

            # Optimization: if all contributions to that color basis element have
            # the same coefficient (up to a sign), put it in front
            list_fracs = [abs(coefficient[0][1]) for coefficient in coeff_list]
            common_factor = False
            diff_fracs = misc.make_unique(list_fracs)
            if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                common_factor = True
                global_factor = diff_fracs[0]
                res = res + '%s(' % self.coeff(1, global_factor, False, 0)

            for (coefficient, amp_number) in coeff_list:

                if common_factor:
                    res = res + "%samp[%d]" % (self.coeff(coefficient[0],
                                               coefficient[1] / abs(coefficient[1]),
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number - 1)
                else:
                    res = res + "%samp[%d]" % (self.coeff(coefficient[0],
                                               coefficient[1],
                                               coefficient[2],
                                               coefficient[3]),
                                               amp_number - 1)

            if common_factor:
                res = res + ')'

            res += ';'
            res_list.append(res)

        return "\n".join(res_list)

    def get_amp2_lines(self, matrix_element):
        """Return the amp2(i) = sum(amp for diag(i))^2 lines"""

        ret_lines = []
        # Get minimum legs in a vertex
        
        #vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
           #matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
        #minvert = min(vert_list) if vert_list!=[] else 0

        for idiag, diag in enumerate(matrix_element.get('diagrams')):
            # Ignore any diagrams with 4-particle vertices.
            #if diag.get_vertex_leg_numbers()!=[] and \
                               #max(diag.get_vertex_leg_numbers()) > minvert:
                #continue
            # Now write out the expression for AMP2, meaning the sum of
            # squared amplitudes belonging to the same diagram
            line = "amp2[%d] += " % (idiag)
            line += "+".join(["norm(amp[%(num)d])" % \
                              {"num": a.get('number')-1} for a in \
                              diag.get('amplitudes')])
            line += ";"
            ret_lines.append(line)

        return "\n".join(ret_lines)
    
coeff = OneProcessExporterCPP.coeff

class OneProcessExporterMatchbox(OneProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Matchbox format."""

    # Static variables (for inheritance)
    process_class_template = 'matchbox_class.inc'
    single_process_template = 'matchbox_matrix.inc'
    process_definition_template = 'matchbox_function_definitions.inc'

    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars.%s);" % part.get('mass'))
        return "\n".join(initProc_lines)


    def get_class_specific_definition_matrix(self, converter, matrix_element):
        """ """
        
        converter = super(OneProcessExporterMatchbox, self).get_class_specific_definition_matrix(converter, matrix_element)
        
        # T(....)
        converter['color_sting_lines'] = \
                                     self.get_color_string_lines(matrix_element)
                                     
        return converter
        
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for MAtchbox .cc file"""

        ret_lines = []
        if self.single_helicities:
            ret_lines.append(\
                "void %s::calculate_wavefunctions(const int perm[], const int hel[]){" % \
                class_name)
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append(self.get_calculate_wavefunctions(\
                self.wavefunctions, self.amplitudes))
            ret_lines.append(self.get_jamp_lines(color_amplitudes[0]))
            ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)


    def get_color_string_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["static const double res[1][1] = {-1.};"])
        
        #start the real work
        color_denominators = matrix_element.get('color_matrix').\
                                                         get_line_denominators()
        matrix_strings = []
        my_cs = color.ColorString()
                
        for i_color in range(len(color_denominators)):
            # Then write the numerators for the matrix elements
            my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[i_color])
            t_str=repr(my_cs)
            t_match=re.compile(r"(\w+)\(([\s\d+\,]*)\)")
            # from '1 T(2,4,1) Tr(4,5,6) Epsilon(5,3,2,1) T(1,2)' returns with findall:
            # [('T', '2,4,1'), ('Tr', '4,5,6'), ('Epsilon', '5,3,2,1'), ('T', '1,2')]
            all_matches = t_match.findall(t_str)
            tmp_color = [] 
            for match in all_matches:
                ctype, arg = match[0], [m.strip() for m in match[1].split(',')]
                if ctype not in ['T', 'Tr']:
                    raise self.ProcessExporterCPPError('Color Structure not handle by Matchbox')
                tmp_color.append(arg)
            #compute the maximal size of the vector
            nb_index = sum(len(o) for o in tmp_color)
            max_len = nb_index + (nb_index//2) -1
            #create the list with the 0 separator
            curr_color = tmp_color[0]
            for tcolor in tmp_color[1:]:
                curr_color += ['0'] + tcolor
            curr_color += ['0'] * (max_len- len(curr_color)) 
            #format the output
            matrix_strings.append('{%s}' % ','.join(curr_color))

        matrix_string = 'static const double res[%s][%s] = {%s};' % \
            (len(color_denominators), max_len, ",".join(matrix_strings))    

        return matrix_string


#===============================================================================
# ProcessExporterPythia8
#===============================================================================
class OneProcessExporterPythia8(OneProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Pythia 8 format."""

    # Static variables (for inheritance)
    process_template_h = 'pythia8_process_h.inc'
    process_template_cc = 'pythia8_process_cc.inc'
    process_class_template = 'pythia8_process_class.inc'
    process_definition_template = 'pythia8_process_function_definitions.inc'
    process_wavefunction_template = 'pythia8_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'pythia8_process_sigmaKin_function.inc'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files', 'pythia8')     


    def __init__(self, *args, **opts):
        """Set process class name"""

        if 'version' in opts:
            self.version = opts['version']
            del opts['version']
        else:
            self.version='8.2'
        super(OneProcessExporterPythia8, self).__init__(*args, **opts)

        # Check if any processes are not 2->1,2,3
        for me in self.matrix_elements:
            if me.get_nexternal_ninitial() not in [(3,2),(4,2),(5,2)]:
                nex,nin = me.get_nexternal_ninitial()
                raise InvalidCmd("Pythia 8 can only handle 2->1,2,3 processes, not %d->%d" % \
                      (nin,nex-nin))
            
        self.process_class = self.process_name
        
    # Methods for generation of process files for Pythia 8

    def get_default_converter(self):
        
        replace_dict = {}       
        # Extract model name
        replace_dict['model_name'] = self.model_name
        if self.version =="8.2":
            replace_dict['include_prefix'] = 'Pythia8/'
        else:
            replace_dict['include_prefix'] = ''
            
        replace_dict['version'] = self.version
        
        return replace_dict
    #===========================================================================
    # Process export helper functions
    #===========================================================================
    def get_process_class_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = self.get_default_converter()


        # Extract process info lines for all processes
        process_lines = "\n".join([self.get_process_info_lines(me) for me in \
                                   self.matrix_elements])
        
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        replace_dict['nfinal'] = self.nfinal

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        # Extract process definition
        process_definition = "%s (%s)" % (self.process_string,
                                          self.model_name)
        replace_dict['process_definition'] = process_definition

        process = self.processes[0]
        replace_dict['process_code'] = 10000 + \
                                       100*process.get('id') + \
                                       self.process_number

        replace_dict['inFlux'] = self.get_process_influx()

        replace_dict['id_masses'] = self.get_id_masses(process)
        replace_dict['resonances'] = self.get_resonance_lines()

        replace_dict['nexternal'] = self.nexternal
        replace_dict['nprocesses'] = self.nprocesses
        
        if self.single_helicities:
            replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          void calculate_wavefunctions(const int perm[], const int hel[], const int flavor[]);
                          static const int nwavefuncs = %d;
                          std::complex<double> w[nwavefuncs][18];
                          static const int namplitudes = %d;
                          std::complex<double> amp[namplitudes];""" % \
                          (len(self.wavefunctions),
                           len(self.amplitudes.get_all_amplitudes()))
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s();" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        else:
            replace_dict['all_sigma_kin_definitions'] = \
                          "\n".join(["void sigmaKin_%s();" % \
                                     me.get('processes')[0].shell_string().\
                                     replace("0_", "") \
                                     for me in self.matrix_elements])
            replace_dict['all_matrix_definitions'] = \
                           "\n".join(["double matrix_%s(const int hel[]);" % \
                                      me.get('processes')[0].shell_string().\
                                      replace("0_", "") \
                                      for me in self.matrix_elements])

        if write:
            file = self.read_template_file('pythia8_process_class.inc') % replace_dict
            return file
        else:
            return replace_dict

    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""


        replace_dict = self.get_default_converter()

        # Extract process info lines
        replace_dict['process_lines'] = \
                             "\n".join([self.get_process_info_lines(me) for \
                                        me in self.matrix_elements])

        # Extract process class name (for the moment same as file name)
        replace_dict['process_class_name'] = self.process_name

        color_amplitudes = [me.get_color_amplitudes(merge_quartic_amplitudes=False) for me in \
                            self.matrix_elements]

        replace_dict['initProc_lines'] = \
                                     self.get_initProc_lines(color_amplitudes)
        replace_dict['reset_jamp_lines'] = \
                                     self.get_reset_jamp_lines(color_amplitudes)
        

        replace_dict['sigmaKin_lines'], _ = \
                                     self.get_sigmaKin_lines(color_amplitudes)
        replace_dict['sigmaHat_lines'] = \
                                     self.get_sigmaHat_lines()

        replace_dict['setIdColAcol_lines'] = \
                                   self.get_setIdColAcol_lines(color_amplitudes)

        replace_dict['weightDecay_lines'] = \
                                       self.get_weightDecay_lines()    

        replace_dict['all_sigmaKin'] = \
                                  self.get_all_sigmaKin_lines(color_amplitudes,
                                                              self.process_name)
        if write:
            file = self.read_template_file('pythia8_process_function_definitions.inc') %\
               replace_dict
            return file
        else:
            return replace_dict

    def get_process_influx(self):
        """Return process file name for the process in matrix_element"""

        # Create a set with the pairs of incoming partons in definite order,
        # e.g.,  g g >... u d > ... d~ u > ... gives ([21,21], [1,2], [-2,1])
        beams = set([tuple(sorted([process.get('legs')[0].get('id'),
                                   process.get('legs')[1].get('id')])) \
                          for process in self.processes])

        # Define a number of useful sets
        antiquarks = list(range(-1, -6, -1))
        quarks = list(range(1,6))
        antileptons = list(range(-11, -17, -1))
        leptons = list(range(11, 17, 1))
        allquarks = antiquarks + quarks
        antifermions = antiquarks + antileptons
        fermions = quarks + leptons
        allfermions = allquarks + antileptons + leptons
        downfermions = list(range(-2, -5, -2)) + list(range(-1, -5, -2)) + \
                       list(range(-12, -17, -2)) + list(range(-11, -17, -2)) 
        upfermions = list(range(1, 5, 2)) + list(range(2, 5, 2)) + \
                     list(range(11, 17, 2)) + list(range(12, 17, 2))

        # The following gives a list from flavor combinations to "inFlux" values
        # allowed by Pythia8, see Pythia 8 document SemiInternalProcesses.html
        set_tuples = [(set([(21, 21)]), "gg"),
                      (set(list(itertools.product(allquarks, [21]))), "qg"),
                      (set(zip(antiquarks, quarks)), "qqbarSame"),
                      (set(list(itertools.product(allquarks,
                                                       allquarks))), "qq"),
                      (set(zip(antifermions, fermions)),"ffbarSame"),
                      (set(zip(downfermions, upfermions)),"ffbarChg"),
                      (set(list(itertools.product(allfermions,
                                                       allfermions))), "ff"),
                      (set(list(itertools.product(allfermions, [22]))), "fgm"),
                      (set([(21, 22)]), "ggm"),
                      (set([(22, 22)]), "gmgm")]

        for set_tuple in set_tuples:
            if beams.issubset(set_tuple[0]):
                return set_tuple[1]

        raise InvalidCmd('Pythia 8 cannot handle incoming flavors %s' %\
                             repr(beams))

        return 

    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, filename):
        """Open a template file and return the contents."""
             
        try:
            return super(OneProcessExporterPythia8, cls).read_template_file(filename)     
        except:
            return super(OneProcessExporterPythia8, cls).read_template_file(filename, classpath=True)

        
    def get_id_masses(self, process):
        """Return the lines which define the ids for the final state particles,
        for the Pythia phase space"""

        if self.nfinal == 1:
            return ""
        
        mass_strings = []
        for i in range(2, len(process.get_legs_with_decays())):
            if self.model.get_particle(process.get_legs_with_decays()[i].get('id')).\
                   get('mass') not in  ['zero', 'ZERO']:
                mass_strings.append("int id%dMass() const {return %d;}" % \
                                (i + 1, abs(process.get_legs_with_decays()[i].get('id'))))

        return "\n".join(mass_strings)

    def get_resonance_lines(self):
        """Return the lines which define the ids for intermediate resonances
        for the Pythia phase space"""

        if self.nfinal == 1:
            return "virtual int resonanceA() const {return %d;}" % \
                           abs(self.processes[0].get('legs')[2].get('id'))
        
        res_strings = []
        res_letters = ['A', 'B']

        sids, singleres, schannel = self.get_resonances()

        for i, sid in enumerate(sids[:2]):
            res_strings.append("virtual int resonance%s() const {return %d;}"\
                                % (res_letters[i], sid))

        if schannel:
           res_strings.append("virtual bool isSChannel() const {return true;}")

        if singleres != 0:
            res_strings.append("virtual int idSChannel() const {return %d;}" \
                               % singleres)
            
        return "\n".join(res_strings)

    def get_resonances(self):
        """Return the PIDs for any resonances in 2->2 and 2->3 processes."""

        model = self.matrix_elements[0].get('processes')[0].get('model')
        new_pdg = model.get_first_non_pdg()
        # Get a list of all resonant s-channel contributions
        diagrams = sum([me.get('diagrams') for me in self.matrix_elements], [])
        resonances = []
        no_t_channels = True
        final_s_channels = []
        for diagram in diagrams:
            schannels, tchannels = diagram.get('amplitudes')[0].\
                                   get_s_and_t_channels(self.ninitial, model,
                                                        new_pdg)
            for schannel in schannels:
                sid = schannel.get('legs')[-1].get('id')
                part = self.model.get_particle(sid)
                if part:
                    width = self.model.get_particle(sid).get('width')
                    if width.lower() != 'zero':
                        # Only care about absolute value of resonance PIDs:
                        resonances.append(abs(sid))
                    else:
                        sid = 0
                    if len(tchannels) == 1 and schannel == schannels[-1]:
                        final_s_channels.append(abs(sid))

            if len(tchannels) > 1:
                # There are t-channel diagrams
                no_t_channels = False
            
        resonance_set = set(resonances)
        final_s_set = set(final_s_channels)

        singleres = 0
        # singleres is set if all diagrams have the same final resonance
        if len(final_s_channels) == len(diagrams) and len(final_s_set) == 1 \
                and final_s_channels[0] != 0:
            singleres = final_s_channels[0]

        resonance_set = misc.make_unique([pid for pid in resonance_set])

        # schannel is True if all diagrams are pure s-channel and there are
        # no QCD vertices
        schannel = no_t_channels and \
                   not any(['QCD' in d.calculate_orders() for d in diagrams])

        return resonance_set, singleres, schannel

    def get_initProc_lines(self, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set massive/massless matrix elements for c/b/mu/tau")
        # Add lines to set c/b/tau/mu kinematics massive/massless
        if not self.model.get_particle(4) or \
               self.model.get_particle(4).get('mass').lower() == 'zero':
            cMassiveME = "0."
        else:
            cMassiveME = "particleDataPtr->m0(4)"
        initProc_lines.append("mcME = %s;" % cMassiveME)
        if not self.model.get_particle(5) or \
               self.model.get_particle(5).get('mass').lower() == 'zero':
            bMassiveME = "0."
        else:
            bMassiveME = "particleDataPtr->m0(5)"
        initProc_lines.append("mbME = %s;" % bMassiveME)
        if not self.model.get_particle(13) or \
               self.model.get_particle(13).get('mass').lower() == 'zero':
            muMassiveME = "0."
        else:
            muMassiveME = "particleDataPtr->m0(13)"
        initProc_lines.append("mmuME = %s;" % muMassiveME)
        if not self.model.get_particle(15) or \
               self.model.get_particle(15).get('mass').lower() == 'zero':
            tauMassiveME = "0."
        else:
            tauMassiveME = "particleDataPtr->m0(15)"
        initProc_lines.append("mtauME = %s;" % tauMassiveME)
            
        for i, me in enumerate(self.matrix_elements):
            initProc_lines.append("jamp2[%d] = new double[%d];" % \
                                  (i, len(color_amplitudes[i])))

        return "\n".join(initProc_lines)

    def get_setIdColAcol_lines(self, color_amplitudes):
        """Generate lines to set final-state id and color info for process"""

        res_lines = []

        # Create a set with the pairs of incoming partons
        beams = set([(process.get('legs')[0].get('id'),
                      process.get('legs')[1].get('id')) \
                     for process in self.processes])
        beams = sorted(list(beams))
        # Now write a selection routine for final state ids
        for ibeam, beam_parts in enumerate(beams):
            if ibeam == 0:
                res_lines.append("if(id1 == %d && id2 == %d){" % beam_parts)
            else:
                res_lines.append("else if(id1 == %d && id2 == %d){" % beam_parts)            
            # Pick out all processes with this beam pair
            beam_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[0].get('id'),
                                process.get('legs')[1].get('id')) \
                               for process in me.get('processes')]]
            # Pick out all mirror processes for this beam pair
            beam_mirror_processes = []
            if beam_parts[0] != beam_parts[1]:
                beam_mirror_processes = [(i, me) for (i, me) in \
                              enumerate(self.matrix_elements) if beam_parts in \
                              [(process.get('legs')[1].get('id'),
                                process.get('legs')[0].get('id')) \
                               for process in me.get('processes')]]

            final_id_list = []
            final_mirror_id_list = []
            for (i, me) in beam_processes:
                final_id_list.extend([tuple([l.get('id') for l in \
                                             proc.get_legs_with_decays() if l.get('state')]) \
                                      for proc in me.get('processes') \
                                      if beam_parts == \
                                      (proc.get('legs')[0].get('id'),
                                       proc.get('legs')[1].get('id'))])
            for (i, me) in beam_mirror_processes:
                final_mirror_id_list.extend([tuple([l.get('id') for l in \
                                             proc.get_legs_with_decays() if l.get('state')]) \
                                      for proc in me.get_mirror_processes() \
                                      if beam_parts == \
                                      (proc.get('legs')[0].get('id'),
                                       proc.get('legs')[1].get('id'))])
            final_id_list = set(final_id_list)
            final_mirror_id_list = set(final_mirror_id_list)

            if final_id_list and final_mirror_id_list or \
               not final_id_list and not final_mirror_id_list:
                raise self.ProcessExporterCPPError("Missing processes, or both process and mirror process")


            ncombs = len(final_id_list)+len(final_mirror_id_list)

            res_lines.append("// Pick one of the flavor combinations %s" % \
                             ", ".join([repr(ids) for ids in final_id_list]))

            me_weight = []
            for final_ids in final_id_list:
                items = [(i, len([ p for p in me.get('processes') \
                             if [l.get('id') for l in \
                             p.get_legs_with_decays()] == \
                             list(beam_parts) + list(final_ids)])) \
                       for (i, me) in beam_processes]
                me_weight.append("+".join(["matrix_element[%i]*%i" % (i, l) for\
                                           (i, l) in items if l > 0]).\
                                 replace('*1', ''))
                if any([l>1 for (i, l) in items]):
                    raise self.ProcessExporterCPPError("More than one process with identical " + \
                          "external particles is not supported")

            for final_ids in final_mirror_id_list:
                items = [(i, len([ p for p in me.get_mirror_processes() \
                             if [l.get('id') for l in p.get_legs_with_decays()] == \
                             list(beam_parts) + list(final_ids)])) \
                       for (i, me) in beam_mirror_processes]
                me_weight.append("+".join(["matrix_element[%i]*%i" % \
                                           (i+len(self.matrix_elements), l) for\
                                           (i, l) in items if l > 0]).\
                                 replace('*1', ''))
                if any([l>1 for (i, l) in items]):
                    raise self.ProcessExporterCPPError("More than one process with identical " + \
                          "external particles is not supported")

            if final_id_list:
                res_lines.append("int flavors[%d][%d] = {%s};" % \
                                 (ncombs, self.nfinal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in ids]) + "}" for ids \
                                            in final_id_list])))
            elif final_mirror_id_list:
                res_lines.append("int flavors[%d][%d] = {%s};" % \
                                 (ncombs, self.nfinal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in ids]) + "}" for ids \
                                            in final_mirror_id_list])))
            res_lines.append("vector<double> probs;")
            res_lines.append("double sum = %s;" % "+".join(me_weight))
            for me in me_weight:
                res_lines.append("probs.push_back(%s/sum);" % me)
            res_lines.append("int choice = rndmPtr->pick(probs);")
            for i in range(self.nfinal):
                res_lines.append("id%d = flavors[choice][%d];" % (i+3, i))

            res_lines.append("}")

        res_lines.append("setId(%s);" % ",".join(["id%d" % i for i in \
                                                 range(1, self.nexternal + 1)]))

        # Now write a selection routine for color flows

        # We need separate selection for each flavor combination,
        # since the different processes might have different color
        # structures.
        
        # Here goes the color connections corresponding to the JAMPs
        # Only one output, for the first subproc!

        res_lines.append("// Pick color flow")

        res_lines.append("int ncolor[%d] = {%s};" % \
                         (len(color_amplitudes),
                          ",".join([str(len(colamp)) for colamp in \
                                    color_amplitudes])))
                                                 

        for ime, me in enumerate(self.matrix_elements):

            res_lines.append("if((%s)){" % \
                                 ")||(".join(["&&".join(["id%d == %d" % \
                                            (i+1, l.get('id')) for (i, l) in \
                                            enumerate(p.get_legs_with_decays())])\
                                           for p in me.get('processes')]))
            if ime > 0:
                res_lines[-1] = "else " + res_lines[-1]

            proc = me.get('processes')[0]
            if not me.get('color_basis'):
                # If no color basis, just output trivial color flow
                res_lines.append("setColAcol(%s);" % ",".join(["0"]*2*self.nfinal))
            else:
                # Else, build a color representation dictionnary
                repr_dict = {}
                legs = proc.get_legs_with_decays()
                for l in legs:
                    repr_dict[l.get('number')] = \
                        proc.get('model').get_particle(l.get('id')).get_color()
                # Get the list of color flows
                color_flow_list = \
                    me.get('color_basis').color_flow_decomposition(\
                                                      repr_dict, self.ninitial)
                # Select a color flow
                ncolor = len(me.get('color_basis'))
                res_lines.append("""vector<double> probs;
                  double sum = %s;
                  for(int i=0;i<ncolor[%i];i++)
                  probs.push_back(jamp2[%i][i]/sum);
                  int ic = rndmPtr->pick(probs);""" % \
                                 ("+".join(["jamp2[%d][%d]" % (ime, i) for i \
                                            in range(ncolor)]), ime, ime))

                color_flows = []
                for color_flow_dict in color_flow_list:
                    color_flows.append([int(fmod(color_flow_dict[l.get('number')][i], 500)) \
                                        for (l,i) in itertools.product(legs, [0,1])])

                # Write out colors for the selected color flow
                res_lines.append("static int colors[%d][%d] = {%s};" % \
                                 (ncolor, 2 * self.nexternal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in flows]) + "}" for flows \
                                            in color_flows])))

                res_lines.append("setColAcol(%s);" % \
                                 ",".join(["colors[ic][%d]" % i for i in \
                                          range(2 * self.nexternal)]))
            res_lines.append('}')

        # Same thing but for mirror processes
        for ime, me in enumerate(self.matrix_elements):
            if not me.get('has_mirror_process'):
                continue
            res_lines.append("else if((%s)){" % \
                                 ")||(".join(["&&".join(["id%d == %d" % \
                                            (i+1, l.get('id')) for (i, l) in \
                                            enumerate(p.get_legs_with_decays())])\
                                           for p in me.get_mirror_processes()]))

            proc = me.get('processes')[0]
            if not me.get('color_basis'):
                # If no color basis, just output trivial color flow
                res_lines.append("setColAcol(%s);" % ",".join(["0"]*2*self.nfinal))
            else:
                # Else, build a color representation dictionnary
                repr_dict = {}
                legs = proc.get_legs_with_decays()
                legs[0:2] = [legs[1],legs[0]]
                for l in legs:
                    repr_dict[l.get('number')] = \
                        proc.get('model').get_particle(l.get('id')).get_color()
                # Get the list of color flows
                color_flow_list = \
                    me.get('color_basis').color_flow_decomposition(\
                                                      repr_dict, self.ninitial)
                # Select a color flow
                ncolor = len(me.get('color_basis'))
                res_lines.append("""vector<double> probs;
                  double sum = %s;
                  for(int i=0;i<ncolor[%i];i++)
                  probs.push_back(jamp2[%i][i]/sum);
                  int ic = rndmPtr->pick(probs);""" % \
                                 ("+".join(["jamp2[%d][%d]" % (ime, i) for i \
                                            in range(ncolor)]), ime, ime))

                color_flows = []
                for color_flow_dict in color_flow_list:
                    color_flows.append([color_flow_dict[l.get('number')][i] % 500 \
                                        for (l,i) in itertools.product(legs, [0,1])])

                # Write out colors for the selected color flow
                res_lines.append("static int colors[%d][%d] = {%s};" % \
                                 (ncolor, 2 * self.nexternal,
                                  ",".join(["{" + ",".join([str(id) for id \
                                            in flows]) + "}" for flows \
                                            in color_flows])))

                res_lines.append("setColAcol(%s);" % \
                                 ",".join(["colors[ic][%d]" % i for i in \
                                          range(2 * self.nexternal)]))
            res_lines.append('}')

        return "\n".join(res_lines)


    def get_weightDecay_lines(self):
        """Get weightDecay_lines for function definition for Pythia 8 .cc file"""

        weightDecay_lines = "// Just use isotropic decay (default)\n"
        weightDecay_lines += "return 1.;"

        return weightDecay_lines

    #===============================================================================
    # Routines to export/output UFO models in Pythia8 format
    #===============================================================================
    def convert_model_to_pythia8(self, model, pythia_dir, wanted_lorentz = []):
        """Create a full valid Pythia 8 model from an MG5 model (coming from UFO)"""
    
        if not os.path.isfile(os.path.join(pythia_dir, 'include', 'Pythia.h'))\
           and not os.path.isfile(os.path.join(pythia_dir, 'include', 'Pythia8', 'Pythia.h')):
            logger.warning('Directory %s is not a valid Pythia 8 main dir.' % pythia_dir)
    
        # create the model parameter files
        model_builder = UFOModelConverterPythia8(model, pythia_dir, 
                                                 wanted_lorentz=wanted_lorentz,
                                                 replace_dict=self.get_default_converter())
        model_builder.cc_file_dir = "Processes_" + model_builder.model_name
        model_builder.include_dir = model_builder.cc_file_dir
    
        model_builder.write_files()
        # Write makefile
        model_builder.write_makefile()
        # Write param_card
        model_builder.write_param_card()
        return model_builder.model_name, model_builder.cc_file_dir


#===============================================================================
# ProcessExporterCPP
#===============================================================================
class ProcessExporterCPP(VirtualExporter):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    grouped_mode = False
    exporter = 'cpp'
    # Only the plain standalone_cpp exporter emits the crossing machinery; the
    # matchbox/pythia8/mg7 subclasses write their own templates and override
    # this back to False.
    supports_crossing = True

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True
                        }
    
    oneprocessclass = OneProcessExporterCPP
    s= _file_path + 'iolibs/template_files/'
    dirs_to_create = ['src', 'lib', 'Cards', 'SubProcesses']
    from_template = {'src': [s+'rambo.h', s+'rambo.cc', s+'read_slha.h', s+'read_slha.cc',
                             s+'mg5_citation.h', s+'mg5_citation.cc'],
                     'SubProcesses': []}
    to_link_in_P = ['Makefile']
    template_src_make = pjoin(_file_path, 'iolibs', 'template_files','Makefile_sa_cpp_src')
    template_Sub_make = pjoin(_file_path, 'iolibs', 'template_files','Makefile_sa_cpp_sp') 
    create_model_class =  UFOModelConverterCPP
    _check_sa_cpp_template = pjoin(_file_path, 'iolibs', 'template_files', 'check_sa.cpp')
    

    def __init__(self, dir_path = "", opt=None):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = MG5DIR
        self.dir_path = dir_path
        self.model = None

        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)
        
        #place holder to pass information to the run_interface
        self.proc_characteristic = banner_mod.ProcCharacteristic()    

    def copy_template(self, model):
        """Prepare export_dir as standalone_cpp directory, including:
        src (for RAMBO, model and ALOHA files + makefile)
        lib (with compiled libraries from src)
        SubProcesses (with check_sa.cpp + makefile and Pxxxxx directories)
        """

        try:
            os.mkdir(self.dir_path)
        except os.error as error:
            logger.warning(error.strerror + " " + self.dir_path)
        
        with misc.chdir(self.dir_path):
            logger.info('Creating subdirectories in directory %s' % self.dir_path)

            for d in self.dirs_to_create:
                try:
                    os.mkdir(d)
                except os.error as error:
                    logger.warning(error.strerror + " " + self.dir_path)
    
            # Write param_card
            open(os.path.join("Cards","param_card.dat"), 'w').write(\
                                                       model.write_param_card())

    
            # Copy the needed src files
            for key in self.from_template:
                for f in self.from_template[key]:
                    cp(f, key)

            if self.template_src_make:
                # Copy src Makefile
                makefile = self.read_template_file(self.template_src_make) % \
                                        self.get_makefile_replace_dict(model)
                open(os.path.join('src', 'Makefile'), 'w').write(makefile)

            if self.template_Sub_make:
                # Copy SubProcesses Makefile
                makefile = self.read_template_file(self.template_Sub_make) % \
                                        self.get_makefile_replace_dict(model)
                open(os.path.join('SubProcesses', 'Makefile'), 'w').write(makefile)

    def get_makefile_replace_dict(self, model):
        """Template replacements for the src and SubProcesses makefiles."""

        return {'model': self.get_model_name(model.get('name')),
                'cpp_compiler': self.opt['cpp_compiler'] if self.opt['cpp_compiler'] else 'g++'}

    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
            
        return False, matrix_element



    def convert_model(self, model, wanted_lorentz = [],
                         wanted_couplings = []):
        # create the model parameter files
        model_builder = self.create_model_class(model,
                                         os.path.join(self.dir_path, 'src'),
                                         wanted_lorentz,
                                         wanted_couplings)
        model_builder.write_files()
    
    def compile_model(self):
        make_model_cpp(self.dir_path)
    
    @classmethod
    def read_template_file(cls, *args, **opts):
        """Open a template file and return the contents."""
         
        return cls.oneprocessclass.read_template_file(*args, **opts) 

    @classmethod
    def get_mg5_info_lines(cls):
        return cls.oneprocessclass.get_mg5_info_lines()
        
    #===============================================================================
    # generate_subprocess_directory
    #===============================================================================
    def _get_check_sa_cpp_crossing_example(self, matrix_element, maxflavor,
                                           nexternal, use_crossing):
        """C++ block for check_sa.cpp demonstrating the crossed matrix elements.

        Returns '' when crossing is not active for this backend/matrix element,
        leaving the driver unchanged. Otherwise it mirrors the Fortran
        check_sa.f demonstration: a loop over every way of crossing particle 1
        and particle 2 with a final-state particle (and over each flavor) that,
        for each, evaluates the crossed matrix element and prints its signed
        PDGs and value. The whole section is gated behind `if(false)` so it is
        present only as a ready-to-enable example.

        flavor_id is 0-based in C++: flavor_id = cross*nflav + flav0, with
        cross = flip1*(nexternal+1) + flip2 (flip1/flip2 the partners of
        particle 1/2), matching sigmaKin's decode. standalone_cpp has no runtime
        PDG accessor, so the signed PDG of each flavor_id is precomputed here
        into demo_pdg[flavor_id*nexternal + slot] the same way
        GET_PDG_FOR_FLAVOR does (conjugating swapped legs, zeros for an
        impossible/overlapping crossing). Each evaluation uses a FRESH
        CPPProcess so the shared good-helicity cache cannot contaminate it.
        """
        if not use_crossing:
            return ''

        tables = ProcessExporterFortran.compute_crossing_tables(
            self, matrix_element)
        spincol = tables['spincol']
        perm = tables['perm']
        ic = tables['ic']
        nx = tables['nexternal']
        ncross = len(spincol)
        # The flavor count sigmaKin decodes against (CPPProcess::nflavors); read
        # from the same source that fills %(nflav)d so the demo_pdg table indexes
        # by flavor_id exactly as the runtime does.
        n_flav = len(matrix_element.get_external_flavors_with_iden())
        # Physical signed PDGs (basepid holds internal group codes like 81, not
        # the physical PDG the user expects).
        _, pdg_flat, antipdg_flat = \
            ProcessExporterFortran._build_flav_pdg_tables(self, matrix_element)
        # Those tables are indexed by physical flavor combination while flavor_id
        # counts coupling-equivalence classes; _flavor_rep_rows bridges the two
        # (the same lookup compute_crossing_pdg_entries does, kept shared so the
        # demo table and the fortran signatures cannot drift apart).
        rep_rows = ProcessExporterFortran._flavor_rep_rows(
            self, matrix_element)

        # demo_pdg[flavor_id*nexternal + slot], flavor_id = cross*nflav+flav0.
        demo_pdg = []
        for cross in range(ncross):
            for flav0 in range(n_flav):
                row = rep_rows[flav0]
                for k in range(nx):
                    if spincol[cross] == 0:
                        demo_pdg.append(0)
                        continue
                    src = perm[cross * nx + k]
                    if ic[cross * nx + k] == 1:
                        demo_pdg.append(pdg_flat[row * nx + src])
                    else:
                        demo_pdg.append(antipdg_flat[row * nx + src])

        sep = ('    cout << " ---------------------------------------------------'
               '--------------------------" << endl;')
        lines = [
            '  // Crossing-symmetry examples (crossed processes); see the',
            '  // matching block in the Fortran check_sa.f. Gated behind',
            '  // if(false): flip it to true to actually print them. Each',
            '  // flavor_id is evaluated on a fresh CPPProcess so the shared',
            '  // good-helicity cache cannot contaminate the crossed value.',
            '  if(false){',
            '    const int nflav = process.nflavors;',
            '    const int nin = process.ninitial;',
            '    const int nx = process.nexternal;',
            '    static const int demo_pdg[%d] = {%s};'
            % (len(demo_pdg), ', '.join(str(p) for p in demo_pdg)),
            '    cout << endl << " Crossing-symmetry examples (crossed '
            'processes):" << endl << endl;',
            '    for(int flip1 = nin+1; flip1 <= nx; flip1++){',
            '      for(int flip2 = nin+1; flip2 <= nx; flip2++){',
            '        for(int j = 1; j <= nflav; j++){',
            '          // cross = (partner of p1)*(nx+1) + (partner of p2)',
            '          int cross = flip1*(nx+1) + flip2;',
            '          int flavor_id = cross*nflav + (j-1);',
            '          CPPProcess xproc("../../Cards/param_card.dat");',
            '          xproc.setMomenta(p);',
            '          double xme = xproc.sigmaKin(flavor_id);',
            '          cout << "PARTICLE #1 crossed with particle # " '
            '<< flip1 << endl;',
            '          cout << "PARTICLE #2 crossed with particle # " '
            '<< flip2 << endl;',
            '          cout << "PDG";',
            '          for(int s = 0; s < nx; s++) cout << " " '
            '<< demo_pdg[flavor_id*nx + s];',
            '          cout << " FLAV_IDX " << flavor_id << endl;',
            '          cout << "Matrix element = " << xme'
            ' << " GeV^" << -(2*xproc.nexternal-8) << endl;',
            sep,
            '        }',
            '      }',
            '    }',
            '  }',
        ]
        return '\n'.join(lines)

    def write_check_sa_cpp(self, matrix_element, dirpath, use_crossing=False):
        """Write a per-process check_sa.cpp with flavor arrays filled in.

        This mirrors the Fortran ``write_check_sa`` in ``export_v4.py``:
        it reads the template ``check_sa.cpp``, fills in ``%(maxflavor)d``,
        ``%(nexternal)d``, ``%(flavor_arr)s``, and ``%(pdg_arr)s``, then
        writes the result into *dirpath*/check_sa.cpp.

        The resulting binary is invoked as ``./check [energy]``; when *energy*
        is omitted it defaults to 1500 GeV.
        """
        template = open(self._check_sa_cpp_template).read()

        # Get the model from the matrix element (self.model may not be set yet).
        model = (self.model if self.model is not None else
                 matrix_element.get('processes')[0].get('model'))

        all_flavors = matrix_element.get_external_flavors(all_perm=False)
        all_pdgs    = [l.get('id') for l in
                       matrix_element.get('processes')[0].get('legs_with_decays')]
        nexternal   = len(all_pdgs)

        # Deduplicate flavor combinations (same logic as the Fortran exporter):
        # two different (flv1, flv2, …) tuples that give the same coupling are
        # collapsed to a single entry.
        map_all_flv = {}
        for flv1 in all_flavors:
            coup = matrix_element.get_coupling_for_flv(flv1, model)
            if coup not in map_all_flv:
                map_all_flv[coup] = flv1

        unique_flavors = list(map_all_flv.values())
        maxflavor = max(len(unique_flavors), 1)

        # Map individual PDG → flavor index (0-based) inside each merged group.
        # The C++ ALOHA routines index their val[] and partner[] arrays from 0,
        # so the first member of a merged group gets index 0, the second gets 1,
        # etc.  Non-merged particles keep the sentinel value 0 (they never
        # participate in flavor-indexed val[] lookups).
        pdg_to_flv_index = {}
        merged = (model.get('merged_particles') or {}) if model is not None else {}
        for group_id, sub_ids in merged.items():
            for j, pdg in enumerate(sub_ids):
                pdg_to_flv_index[pdg] = j          # 0-based

        # Build the C++ 2-D array initialisers.
        if not unique_flavors:
            # Non-merged model: single default flavor (all zeros → default C++
            # sigmaKin behavior).
            flavor_rows = ['{' + ', '.join(['0'] * nexternal) + '}']
            pdg_rows    = ['{' + ', '.join(str(p) for p in all_pdgs) + '}']
        else:
            flavor_rows = []
            pdg_rows    = []
            for flv_tuple in unique_flavors:
                f_row = []
                p_row = []
                for j, flv_idx in enumerate(flv_tuple):
                    raw_pdg = all_pdgs[j]
                    sign    = 1 if raw_pdg >= 0 else -1
                    if abs(raw_pdg) in merged:
                        # Merged particle: look up 0-based flavor index.
                        f_row.append(str(pdg_to_flv_index.get(flv_idx, 0)))
                        p_row.append(str(sign * flv_idx))
                    else:
                        # Non-merged particle: 0 means "not flavor-merged".
                        f_row.append('0')
                        p_row.append(str(raw_pdg))
                flavor_rows.append('{' + ', '.join(f_row) + '}')
                pdg_rows.append('{' + ', '.join(p_row) + '}')

        flavor_arr_str = '{' + ', '.join(flavor_rows) + '}'
        pdg_arr_str    = '{' + ', '.join(pdg_rows)    + '}'

        content = template % {
            'maxflavor': maxflavor,
            'nexternal': nexternal,
            'flavor_arr': flavor_arr_str,
            'pdg_arr':    pdg_arr_str,
            'crossing_example': self._get_check_sa_cpp_crossing_example(
                matrix_element, maxflavor, nexternal, use_crossing),
        }
        with open(pjoin(dirpath, 'check_sa.cpp'), 'w') as fout:
            fout.write(content)

    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer,
                                      proc_number=None):
        """Generate the Pxxxxx directory for a subprocess in C++ standalone,
        including the necessary .h and .cc files"""

        #matrix_element = copy.deepcopy(matrix_element)
        process_exporter_cpp = self.oneprocessclass(matrix_element,cpp_helas_call_writer)

        # Enable the crossing machinery for standalone_cpp when the process was
        # generated with --use_crossing (default on) and the process does not
        # pin a specific s-channel (which a crossing would not preserve). Only a
        # single-ME directory carries the flavor tables the crossing needs.
        process_exporter_cpp.use_crossing = bool(
            getattr(self, 'supports_crossing', False)
            and self.opt.get('use_crossing', False)
            and len(process_exporter_cpp.matrix_elements) == 1
            and not ProcessExporterFortran.breaks_crossing_symmetry(
                process_exporter_cpp.matrix_elements[0].get('processes')[0]))


        # Create the directory PN_xx_xxxxx in the specified path
        proc_dir_name = "P%d_%s" % (process_exporter_cpp.process_number, 
                                    process_exporter_cpp.process_name)
        dirpath = pjoin(self.dir_path, 'SubProcesses', proc_dir_name)
        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)
    
        with misc.chdir(dirpath):
            logger.info('Creating files in directory %s' % dirpath)
            process_exporter_cpp.path = dirpath
            # Create the process .h and .cc files
            process_exporter_cpp.generate_process_files()
            for file in self.to_link_in_P:
                ln('../%s' % file)
        # Write a per-process check_sa.cpp with flavor info filled in
        self.write_check_sa_cpp(matrix_element, dirpath,
                                use_crossing=process_exporter_cpp.use_crossing)
        return proc_dir_name

    @staticmethod
    def get_model_name(name):
        """Replace - with _, + with _plus_ in a model name."""

        name = name.replace('-', '_')
        name = name.replace('+', '_plus_')
        return name
    
    def finalize(self, *args, **opts):
        """ """
        self.compile_model()
        pass

class ProcessExporterMatchbox(ProcessExporterCPP):
    oneprocessclass = OneProcessExporterMatchbox
    supports_crossing = False

class ProcessExporterPythia8(ProcessExporterCPP):
    oneprocessclass = OneProcessExporterPythia8
    grouped_mode = 'madevent'
    supports_crossing = False
     
    #===============================================================================
    # generate_process_files_pythia8
    #===============================================================================
    def generate_process_directory(self, multi_matrix_element, cpp_helas_call_writer,
                                   process_string = "",
                                   process_number = 0,
                                   version='8.2'):

        """Generate the .h and .cc files needed for Pythia 8, for the
        processes described by multi_matrix_element"""

        process_exporter_pythia8 = OneProcessExporterPythia8(multi_matrix_element,
                                                      cpp_helas_call_writer,
                                                      process_string,
                                                      process_number,
                                                      self.dir_path,
                                                      version=version)
    
        # Set process directory
        model = process_exporter_pythia8.model
        model_name = process_exporter_pythia8.model_name
        process_exporter_pythia8.process_dir = \
                       'Processes_%(model)s' % {'model': \
                        model_name}
        process_exporter_pythia8.include_dir = process_exporter_pythia8.process_dir
        process_exporter_pythia8.generate_process_files()
        return process_exporter_pythia8

    #===============================================================================
    # generate_example_file_pythia8
    #===============================================================================
    @staticmethod
    def generate_example_file_pythia8(path,
                                       model_path,
                                       process_names,
                                       exporter,
                                       main_file_name = "",
                                       example_dir = "examples",
                                       version="8.2"):
        """Generate the main_model_name.cc file and Makefile in the examples dir"""
    
        filepath = os.path.join(path, example_dir)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
    
        replace_dict = {}
    
        # Extract version number and date from VERSION file
        info_lines = ProcessExporterPythia8.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines
    
        # Extract model name
        replace_dict['model_name'] = exporter.model_name
    
        # Extract include line
        replace_dict['include_lines'] = \
                              "\n".join(["#include \"%s.h\"" % proc_name \
                                         for proc_name in process_names])
    
        # Extract setSigmaPtr line
        replace_dict['sigma_pointer_lines'] = \
               "\n".join(["pythia.setSigmaPtr(new %s());" % proc_name \
                         for proc_name in process_names])
    
        # Extract param_card path
        replace_dict['param_card'] = os.path.join(os.path.pardir,model_path,
                                                  "param_card_%s.dat" % \
                                                  exporter.model_name)
    
        # Create the example main file
        if version =="8.2":
            template_path = 'pythia8.2_main_example_cc.inc'
            makefile_path = 'pythia8.2_main_makefile.inc'
            replace_dict['include_prefix'] = 'Pythia8/'
        else:
            template_path = 'pythia8_main_example_cc.inc'
            makefile_path = 'pythia8_main_makefile.inc'
            replace_dict['include_prefix'] = ''
        
        
        file = ProcessExporterPythia8.read_template_file(template_path) % \
               replace_dict
    
        if not main_file_name:
            num = 1
            while os.path.exists(os.path.join(filepath,
                                        'main_%s_%i.cc' % (exporter.model_name, num))) or \
                  os.path.exists(os.path.join(filepath,
                                        'main_%s_%i' % (exporter.model_name, num))):
                num += 1
            main_file_name = str(num)
    
        main_file = 'main_%s_%s' % (exporter.model_name,
                                    main_file_name)
    
        main_filename = os.path.join(filepath, main_file + '.cc')
    
        # Write the file
        writers.CPPWriter(main_filename).writelines(file)
    
        replace_dict = {}
    
        # Extract version number and date from VERSION file
        replace_dict['info_lines'] = ProcessExporterPythia8.get_mg5_info_lines()
    
        replace_dict['main_file'] = main_file
    
        replace_dict['process_dir'] = model_path
    
        replace_dict['include_dir'] = exporter.include_dir
    
        # Create the makefile
        file = ProcessExporterPythia8.read_template_file(makefile_path) % replace_dict
    
        make_filename = os.path.join(filepath, 'Makefile_%s_%s' % \
                                (exporter.model_name, main_file_name))
    
        # Write the file
        open(make_filename, 'w').write(file)
    
        logger.info("Created files %s and %s in directory %s" \
                    % (os.path.split(main_filename)[-1],
                       os.path.split(make_filename)[-1],
                       os.path.split(make_filename)[0]))
        return main_file, make_filename

    def convert_model(self,*args,**opts):
        pass
    def finalize(self, *args, **opts):
        pass
  

#===============================================================================
# UFOModelConverterPythia8
#===============================================================================

class UFOModelConverterPythia8(UFOModelConverterCPP):
    """ A converter of the UFO-MG5 Model to the Pythia 8 format """

    # Static variables (for inheritance)
    output_name = 'Pythia 8'
    namespace = 'Pythia8'
    
    # Dictionaries for expression of MG5 SM parameters into Pythia 8
    slha_to_expr = {('SMINPUTS', (1,)): '1./csm->alphaEM(((pd->m0(23))*(pd->m0(23))))',
                    ('SMINPUTS', (2,)): 'M_PI*csm->alphaEM(((pd->m0(23))*(pd->m0(23))))*((pd->m0(23))*(pd->m0(23)))/(sqrt(2.)*((pd->m0(24))*(pd->m0(24)))*(((pd->m0(23))*(pd->m0(23)))-((pd->m0(24))*(pd->m0(24)))))',
                    ('SMINPUTS', (3,)): 'alpS',
                    ('CKMBLOCK', (1,)): 'csm->VCKMgen(1,2)',
                    }

    # Template files to use
    param_template_h = 'pythia8_model_parameters_h.inc'
    param_template_cc = 'pythia8_model_parameters_cc.inc'
    template_paths = os.path.join(_file_path, 'iolibs', 'template_files', 'pythia8')     

    def prepare_parameters(self):
        """Extract the model parameters from Pythia 8, and store them in
        the two lists params_indep and params_dep"""

        # Keep only dependences on alphaS, to save time in execution
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)
        params_ext = []
        for key in keys:
            if key == ('external',):
                params_ext += [p for p in self.model['parameters'][key] if p.name]
            elif 'aS' in key:
                for p in self.model['parameters'][key]:
                    self.params_dep.append(base_objects.ModelVariable(p.name,
                                                 p.name + " = " + \
                                                 self.p_to_cpp.parse(p.expr) + ';',
                                                 p.type,
                                                 p.depend))
            else:
                for p in self.model['parameters'][key]:
                    self.params_indep.append(base_objects.ModelVariable(p.name,
                                                 p.name + " = " + \
                                                 self.p_to_cpp.parse(p.expr) + ';',
                                                 p.type,
                                                 p.depend))

        # For external parameters, want to use the internal Pythia
        # parameters for SM params and masses and widths. For other
        # parameters, want to read off the SLHA block code
        while params_ext:
            param = params_ext.pop(0)
            key = (param.lhablock, tuple(param.lhacode))
            if 'aS' in self.slha_to_depend.setdefault(key, ()):
                # This value needs to be set event by event
                self.params_dep.insert(0,
                                       base_objects.ModelVariable(param.name,
                                                   param.name + ' = ' + \
                                                   self.slha_to_expr[key] + ';',
                                                   'real'))
            else:
                try:
                    # This is an SM parameter defined above
                    self.params_indep.insert(0,
                                             base_objects.ModelVariable(param.name,
                                                   param.name + ' = ' + \
                                                   self.slha_to_expr[key] + ';',
                                                   'real'))
                except Exception:
                    # For Yukawa couplings, masses and widths, insert
                    # the Pythia 8 value
                    if param.lhablock == 'YUKAWA':
                        self.slha_to_expr[key] = 'pd->mRun(%i, pd->m0(24))' \
                                                 % param.lhacode[0]
                    if param.lhablock == 'MASS':
                        self.slha_to_expr[key] = 'pd->m0(%i)' \
                                            % param.lhacode[0]
                    if param.lhablock == 'DECAY':
                        self.slha_to_expr[key] = \
                                            'pd->mWidth(%i)' % param.lhacode[0]
                    if key in self.slha_to_expr:
                        self.params_indep.insert(0,\
                                     base_objects.ModelVariable(param.name,
                                     param.name + "=" + self.slha_to_expr[key] \
                                                                + ';',
                                                                'real'))
                    else:
                        # This is a BSM parameter which is read from SLHA
                        if len(param.lhacode) == 1:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        elif len(param.lhacode) == 2:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.lhacode[1],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        elif len(param.lhacode) == 3:
                            expression = "if(!slhaPtr->getEntry<double>(\"%s\", %d, %d, %d, %s)){\n" % \
                                         (param.lhablock.lower(),
                                          param.lhacode[0],
                                          param.lhacode[1],
                                          param.lhacode[2],
                                          param.name) + \
                                          ("cout << \"Warning, setting %s to %e\" << endl;\n" \
                                          + "%s = %e;}") % (param.name, param.value.real,
                                                           param.name, param.value.real)
                        else:
                            raise MadGraph5Error("Only support for SLHA blocks with 1 or 2 indices")
                        self.params_indep.insert(0,
                                               base_objects.ModelVariable(param.name,
                                                                          expression,
                                                                          'real'))

    def write_makefile(self):
        """Generate the Makefile, which creates library files."""

        makefilename = os.path.join(self.dir_path, self.cc_file_dir,
                                    'Makefile')

        replace_dict = {}

        replace_dict['info_lines'] = self.get_mg5_info_lines()
        replace_dict['model'] = self.model_name

        if self.default_replace_dict['version'] == "8.2":
            path = 'pythia8.2_makefile.inc'
        else:
            path = 'pythia8_makefile.inc'
        makefile = self.read_template_file(path) % replace_dict

        # Write the files
        open(makefilename, 'w').write(makefile)

        logger.info("Created %s in directory %s" \
                    % (os.path.split(makefilename)[-1],
                       os.path.split(makefilename)[0]))

    def write_param_card(self):
        """Generate the param_card for the model."""

        paramcardname = os.path.join(self.dir_path, self.cc_file_dir,
                                    'param_card_%s.dat' % self.model_name)
        # Write out param_card
        open(paramcardname, 'w').write(\
            self.model.write_param_card())

        logger.info("Created %s in directory %s" \
                    % (os.path.split(paramcardname)[-1],
                       os.path.split(paramcardname)[0]))
        
    #===============================================================================
    # Global helper methods
    #===============================================================================
    @classmethod
    def read_template_file(cls, *args, **opts):
        """Open a template file and return the contents."""
         
        return OneProcessExporterPythia8.read_template_file(*args, **opts)


class ProcessExporterMG7(ProcessExporterCPP):
    """ Extends the standalone CPP exporter to add files needed to run madevent7 / madnis """

    supports_crossing = False
    s= _file_path + 'iolibs/template_files/'
    dirs_to_create = ['bin', 'src', 'lib', 'Cards', 'SubProcesses']
    # mg7_v5 builds api.so in the P* folders (instead of the standalone_cpp
    # 'check' driver)
    template_Sub_make = pjoin(_file_path, 'iolibs', 'template_files',
                              'Makefile_sa_cpp_sp_api')
    # NB: Cards/run_card.toml is NOT copied verbatim here; it is generated in
    # finalize() from the run_card.toml template via banner.RunCardMG7, so that
    # process-dependent defaults are filled in (see create_run_card).
    from_template = {'src': [s+'read_slha.h', s+'read_slha.cc', s+'mg7/api.h'],
                     'SubProcesses': [s+'mg7/api.cpp'],
                     'Cards': []}
    #from_template_simd = [
    #    s+"mg7/api.h",
    #    s+"mg7/simd/api_simd.cpp",
    #    s+"mg7/simd/cudacpp.mk",
    #    s+"mg7/simd/Makefile",
    #]
    #to_link_simd = ["api.h", "api_simd.cpp", "cudacpp.mk", "Makefile"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.me_lib_format = args[1].get("me_lib_format", None)
        self.process_info = []

    def generate_subprocess_directory(
        self, matrix_element, cpp_helas_call_writer, proc_number=None
    ):
        """ Override of super().generate_subprocess_directory """
        process_exporter_mg7 = self.oneprocessclass(matrix_element,cpp_helas_call_writer)

        # Enable the crossing machinery (extended flavor id) when the process was
        # generated with --use_crossing (default on) and the process does not pin
        # a specific s-channel (which a crossing would not preserve). Only a
        # single-ME directory carries the flavor tables the crossing needs. When
        # off, use_crossing stays False and the output is byte-identical.
        process_exporter_mg7.use_crossing = bool(
            getattr(self, 'supports_crossing', False)
            and self.opt.get('use_crossing', False)
            and len(process_exporter_mg7.matrix_elements) == 1
            and not ProcessExporterFortran.breaks_crossing_symmetry(
                process_exporter_mg7.matrix_elements[0].get('processes')[0]))

        # Create the directory PN_xx_xxxxx in the specified path
        proc_dir_name = process_exporter_mg7.name
        dirpath = pjoin(self.dir_path, 'SubProcesses', proc_dir_name)

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)
        with misc.chdir(dirpath):
            logger.info('Creating files in directory %s' % dirpath)
            process_exporter_mg7.path = dirpath
            # Create the process .h and .cc files
            process_exporter_mg7.generate_process_files()
            for file in self.to_link_in_P:
                ln('../%s' % file)

        # Generate SVG Feynman diagrams (diagrams.svg + diagrams.json)
        if not self.opt.get('output_options', {}).get('noeps') == 'True':
            svg_stem = pjoin(dirpath, 'diagrams')
            model = matrix_element.get('processes')[0].get('model')
            diagrams = matrix_element.get('base_amplitude').get('diagrams')
            logger.info('Generating Feynman diagrams for %s' %
                        matrix_element.get('processes')[0].nice_string())
            plot = draw_svg.MultiSVGDiagramDrawer(diagrams, svg_stem,
                                                  model=model, amplitude=True)
            plot.draw()

        me_lib_path = self.me_lib_format.format(process_id = proc_dir_name)
        self.process_info.append(process_exporter_mg7.get_subprocess_info(dirpath, me_lib_path))

    def copy_template(self, model):
        super().copy_template(model)

        # TODO: for now, we import the files from madgraph. eventually, we should copy
        # the files instead to allow for modification
        with misc.chdir(self.dir_path):
            madnis_bin = os.path.join("bin", "generate_events")
            with open(madnis_bin, "w") as f:
                f.write(
                    "#! /usr/bin/env python3\n"
                    "import sys, os\n"
                    f"sys.path.append('{MG5DIR}')\n"
                    "from madgraph.iolibs.template_files.mg7.madevent import main\n"
                    "if __name__ == '__main__':\n"
                    "    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))\n"
                    "    try:\n"
                    "        main()\n"
                    "    except KeyboardInterrupt:\n"
                    "        pass\n"
                )
            os.chmod(madnis_bin, 0o755)

    def finalize(self, matrix_elements=None, history='', *args, **kwargs):
        file_name = os.path.normpath(os.path.join(
            self.dir_path, "SubProcesses", "subprocesses.json"
        ))
        with open(file_name, 'w') as f:
            json.dump(self.process_info, f)

        # Generate Cards/run_card.toml from the template, filling in
        # process-dependent defaults (mirrors the LO run_card.dat logic).
        self.create_run_card(matrix_elements, history)

        # SubProcesses/proc_characteristics: needed by the CommonRunCmd-based
        # post-processing driver (get_characteristics) so that the madevent
        # tool interface can run on this directory.
        self.create_proc_characteristics(matrix_elements)

        # Cards/me5_configuration.txt: read by CommonRunCmd.set_configuration.
        # Point it at the MG5 install so tool paths (pythia8, etc.) and the
        # cluster/run-mode settings resolve from the central configuration.
        try:
            with open(pjoin(self.dir_path, 'Cards',
                            'me5_configuration.txt'), 'w') as fsock:
                fsock.write('# configuration for the mg7 post-processing tools\n'
                            'mg5_path = %s\n' % MG5DIR)
        except Exception as error:
            logger.warning('could not write me5_configuration.txt: %s', error)

        # MadAnalysis5 default analysis cards, tailored to this process. This
        # must run *before* history.write() below: writing the proc_card cleans
        # the history in place (dropping the multiparticle 'define' lines that
        # MA5 needs to resolve p/j/... in the process).
        self.create_ma5_default_cards(matrix_elements, history)

        # Record the generation commands (proc_card_mg5.dat) so that the model
        # and process end up in the LHE banner (needed by MadSpin/reweight/...).
        try:
            if history:
                history.write(os.path.join(self.dir_path, 'Cards',
                                           'proc_card_mg5.dat'))
        except Exception as error:
            logger.warning('could not write proc_card_mg5.dat: %s', error)

        # we don't call super().finalize() since it would call ProcessExporterCPP.finalize()
        # which would compile the model in src/, and we don't want that

    def pass_information_from_cmd(self, cmd):
        """Capture the process definitions from the command interface; needed to
        generate the MadAnalysis5 default cards at output time."""
        self.proc_defs = getattr(cmd, '_curr_proc_defs', None)

    def create_ma5_default_cards(self, matrix_elements, history):
        """Call MadAnalysis5 to write process-tailored default analysis cards
        (parton + hadron), like the madevent exporter does. Falls back silently
        to the generic default cards if MA5 is unavailable or fails."""
        ma5_path = self.opt.get('madanalysis5_path')
        proc_defs = getattr(self, 'proc_defs', None)
        if not ma5_path or proc_defs is None:
            return

        processes = None
        try:
            if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
                processes = [me.get('processes') for megroup in matrix_elements
                             for me in megroup['matrix_elements']]
            elif matrix_elements:
                processes = [me.get('processes')
                             for me in matrix_elements['matrix_elements']]
        except (KeyError, TypeError):
            processes = None

        # expand merged-flavor beam codes (81/82/...) so MA5 recognises the legs
        proc_defs = self.expand_merged_particle_legs(proc_defs)

        try:
            from madgraph.interface import common_run_interface as common_run
            ma5 = common_run.CommonRunCmd.get_MadAnalysis5_interpreter(
                MG5DIR, ma5_path, loglevel=100)
            if ma5 is None:
                return
            logger.info('Generating MadAnalysis5 default cards tailored to this process')
            for lvl in ('parton', 'hadron'):
                try:
                    text = ma5.main.madgraph.generate_card(history, proc_defs,
                                                           processes, lvl)
                except (Exception, SystemExit):
                    import traceback as _tb
                    logger.debug('MA5 %s card error:\n%s', lvl, _tb.format_exc())
                    logger.warning('MadAnalysis5 failed to write a %s-level default '
                                   'analysis card for this process.', lvl)
                    continue
                out = os.path.join(self.dir_path, 'Cards',
                                   'madanalysis5_%s_card_default.dat' % lvl)
                with open(out, 'w') as fsock:
                    fsock.write(text)
        except (Exception, SystemExit) as error:
            logger.warning('MadAnalysis5 default card generation failed: %s', error)

    def create_run_card(self, matrix_elements, history):
        """Write Cards/run_card.toml from the run_card.toml template via
        banner.RunCardMG7, applying process-dependent defaults."""

        run_card = banner_mod.RunCardMG7()

        processes = None
        try:
            if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
                processes = [me.get('processes') for megroup in matrix_elements
                             for me in megroup['matrix_elements']]
            elif matrix_elements:
                processes = [me.get('processes')
                             for me in matrix_elements['matrix_elements']]
        except (KeyError, TypeError):
            processes = None

        if processes:
            run_card.create_default_for_process(self.proc_characteristic,
                                                history, processes)
            # persist the model so the runtime can compute widths set to 'auto'
            # in the param_card (and recompute them at each scan point). A hash
            # of the model's python source is stored on the second line so the
            # runtime can detect a model that changed since output.
            try:
                model = processes[0][0].get('model')
                model_path = model.get('modelpath')
                model_ref = model_path or model.get('name')
                if model_ref:
                    model_hash = misc.hash_model_files(model_path) if model_path else None
                    with open(pjoin(self.dir_path, 'SubProcesses', 'model.txt'), 'w') as f:
                        f.write(model_ref + '\n' + (model_hash or '') + '\n')
            except Exception:
                pass

        template = pjoin(_file_path, 'iolibs', 'template_files',
                         'mg7', 'run_card.toml')
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card.toml'),
                       template=template)
        # Also write a concrete default card so the interactive card editor
        # can offer "set <param> default" (mirrors run_card_default.dat at LO).
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.toml'),
                       template=template)

    def create_proc_characteristics(self, matrix_elements):
        """Populate and write SubProcesses/proc_characteristics. This is the
        file CommonRunCmd.get_characteristics() reads to learn ninitial /
        nexternal / initial-state PDGs, so that the reused madevent tool
        interface can drive post-processing on this (C++/madspace) output."""
        pc = self.proc_characteristic

        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
            me_list = [me for megroup in matrix_elements
                       for me in megroup['matrix_elements']]
        elif matrix_elements:
            me_list = list(matrix_elements['matrix_elements'])
        else:
            me_list = []

        procs = []
        qcd_orders = set()
        for me in me_list:
            if not me.get('processes'):
                continue
            nexternal, ninitial = me.get_nexternal_ninitial()
            pc['nexternal'] = max(pc['nexternal'], nexternal)
            pc['ninitial'] = ninitial
            procs.extend(me.get('processes'))
            # power of alpha_s in |M|^2 = QCD coupling order of the amplitude;
            # collect it over every diagram so we can tell whether it is uniform.
            # Never let this break the output: on any surprise just fall back to
            # -1 (systematics then simply cannot reconstruct the reweighting).
            try:
                for diagram in me.get('diagrams'):
                    qcd_orders.add(diagram.calculate_orders().get('QCD', 0))
            except Exception as error:
                logger.debug('could not determine the QCD order: %s', error)
                qcd_orders.add(None)

        # a single value of alpha_s (uniform QCD power) lets systematics
        # reconstruct the LO reweighting info without an <mgrwt> block
        pc['single_qcd_order'] = (qcd_orders.pop()
                                  if len(qcd_orders) == 1 and None not in qcd_orders
                                  else -1)

        if procs:
            pc['pdg_initial1'] = [p.get_initial_pdg(1) for p in procs
                                  if p.get_initial_pdg(1)]
            pc['pdg_initial2'] = [p.get_initial_pdg(2) for p in procs
                                  if p.get_initial_pdg(2)]
            model = procs[0].get('model')
            colored = set(abs(p.get('pdg_code')) for p in model.get('particles')
                          if p.get('color') > 1)
            pc['colored_pdgs'] = sorted(colored)
            # ISR/FSR presence drives (e.g.) the shower's initial/final radiation
            pc['has_isr'] = any(abs(pid) in colored
                                for pid in pc['pdg_initial1'] + pc['pdg_initial2'])
            pc['has_fsr'] = any(abs(fid) in colored
                                for p in procs for fid in p.get_final_ids())

        pc.write(pjoin(self.dir_path, 'SubProcesses', 'proc_characteristics'))

def ExportCPPFactory(cmd, group_subprocesses=False, cmd_options={}):
    """ Determine which Export class is required. cmd is the command 
        interface containing all potential usefull information.
    """

    opt = dict(cmd.options)
    opt['output_options'] = cmd_options
    # --use_crossing of the generate/add process command, and of the output
    # command for this output (both default on). Only the exporters that set
    # supports_crossing (standalone_cpp, standalone_mg7/madmatrix) read this
    # key; the others ignore it.
    opt['use_crossing'] = getattr(cmd, '_use_crossing', True) \
                          and getattr(cmd, '_output_use_crossing', True)
    cformat = cmd._export_format
    
    if cformat == 'pythia8':
        return ProcessExporterPythia8(cmd._export_dir, opt)
    elif cformat == 'standalone_cpp':
        return  ProcessExporterCPP(cmd._export_dir, opt)
    elif cformat == 'matchbox_cpp':
        return  ProcessExporterMatchbox(cmd._export_dir, opt)
    elif cformat == 'mg7_v5':
        return ProcessExporterMG7(cmd._export_dir, opt)
    elif cformat == 'mg7':
        from madmatrix.output import ProcessExporterMadMatrix
        return ProcessExporterMadMatrix(cmd._export_dir, opt)
    elif cformat == 'standalone_mg7':
        from madmatrix.output import ProcessExporterMadMatrixStandalone
        return ProcessExporterMadMatrixStandalone(cmd._export_dir, opt)
    else:
        return cmd._export_plugin(cmd._export_dir, opt)

    
