# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Nov 2026) for project MadVFold (arXiv:2609.23500).
# (The classes below are from export_cpp.py of mg5amcnlo v3.8.1.)

# The three items of export_cpp.py of mg5amcnlo v3.8.1 which the cudacpp plugin derives
# from, and which MadGraph7 removed from its export_cpp.py. The plugin loads its own
# independent copy of the MadGraph7 module and adds these to it (see patchPlugin.py):
# the module of MadGraph7 itself, used by madmatrix, is not modified. This file is
# executed in the namespace of that copy, so it has no imports of its own.
class OneProcessExporterGPU(OneProcessExporterCPP):

    # Static variables (for inheritance)
    process_dir = '.'
    include_dir = '.'
    template_path = os.path.join(_file_path, 'iolibs', 'template_files')
    __template_path = os.path.join(_file_path, 'iolibs', 'template_files') 
    process_template_h = 'gpu/process_h.inc'
    process_template_cc = 'gpu/process_cc.inc'
    process_class_template = 'gpu/process_class.inc'
    process_definition_template = 'gpu/process_function_definitions.inc'
    process_wavefunction_template = 'cpp_process_wavefunctions.inc'
    process_sigmaKin_function_template = 'gpu/process_sigmaKin_function.inc'
    single_process_template = 'gpu/process_matrix.inc'
    cc_ext = 'cu'
    support_multichannel = True
    multichannel_var = ',fptype& multi_chanel_num, fptype& multi_chanel_denom'

    def __init__(self, *args, **opts):
        
        super(OneProcessExporterGPU, self).__init__(*args, **opts)
        self.process_class = "gCPPProcess"

    def generate_process_files(self):
        
        if self.matrix_elements[0].get('has_mirror_process'):
            self.matrix_elements[0].set('has_mirror_process', False)
            self.nprocesses/=2

        super(OneProcessExporterGPU, self).generate_process_files()

        self.edit_check_sa()
        self.edit_mgonGPU()
        
        # add symbolic link for C++
        files.ln(pjoin(self.path, 'gcheck_sa.cu'), self.path, 'check_sa.cc')
        files.ln(pjoin(self.path, 'gCPPProcess.cu'), self.path, 'CPPProcess.cc')
    

    def edit_check_sa(self):
        
        template = open(pjoin(self.template_path,'gpu','check_sa.cu'),'r').read()
        replace_dict = {}
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['model'] = self.model_name
        replace_dict['numproc'] = len(self.matrix_elements)

        ff = open(pjoin(self.path, 'gcheck_sa.cu'),'w')
        ff.write(template)
        ff.close()
        
    def edit_mgonGPU(self):
        
        template = open(pjoin(self.template_path,'gpu','mgOnGpuConfig.h'),'r').read()
        replace_dict = {}
        nexternal, nincoming = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nincoming'] = nincoming
        replace_dict['noutcoming'] = nexternal - nincoming
        
        # Number of helicity combinations
        replace_dict['nbhel'] = \
                            self.matrix_elements[0].get_helicity_combinations()
        replace_dict['nwavefunc'] = \
                          self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['wavefuncsize'] = 6
        
        
        ff = open(pjoin(self.path, '..','..','src','mgOnGpuConfig.h'),'w')
        ff.write(template % replace_dict)
        ff.close()        
        

    def get_initProc_lines(self, matrix_element, color_amplitudes):
        """Get initProc_lines for function definition for Pythia 8 .cc file"""

        initProc_lines = []

        initProc_lines.append("// Set external particle masses for this matrix element")

        for part in matrix_element.get_external_wavefunctions():
            initProc_lines.append("mME.push_back(pars->%s);" % part.get('mass'))
        #for i, colamp in enumerate(color_amplitudes):
        #    initProc_lines.append("jamp2[%d] = new double[%d];" % \
        #                          (i, len(colamp)))

        return "\n".join(initProc_lines)

    def get_reset_jamp_lines(self, color_amplitudes):
        """Get lines to reset jamps"""

        ret_lines = ""
        return ret_lines
    
    def get_sigmaKin_lines(self, color_amplitudes, write=True):
        """Get sigmaKin_lines for function definition for Pythia 8 .cc file"""



        replace_dict =  super().get_sigmaKin_lines(color_amplitudes, write=False)

        if self.include_multi_channel:
            replace_dict['madE_var_reset'] = """
            fptype multi_chanel_num = 0.;
            fptype multi_chanel_denom = 0.;
            """
            replace_dict['madE_caclwfcts_call'] = '&multi_chanel_num, &multi_chanel_denom'
            replace_dict['madE_update_answer'] = '   allMEs[iproc*nprocesses + ievt] *= multi_chanel_num/multi_chanel_denom;'

            multi_channel = self.get_multi_channel_dictionary(self.matrix_elements[0].get('diagrams'), self.include_multi_channel)
            replace_dict['nb_channel'] = len(multi_channel)
            replace_dict['nb_color'] = max(1, len(self.matrix_elements[0].get('color_basis')))


        if write:
            file = \
                self.read_template_file(\
                        self.process_sigmaKin_function_template) %\
                        replace_dict
            return file, replace_dict
        else:
            return replace_dict

        if not write:
            return replace_dict, other_replace
        else:
            raise Exception


    @staticmethod
    def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""
    
        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power
    
        if total_coeff == 1:
            if is_imaginary:
                return '+cxtype(0,1)*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-cxtype(0,1)*'
            else:
                return '-'
    
        res_str = '%+i.' % total_coeff.numerator
    
        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%i.' % total_coeff.denominator
    
        if is_imaginary:
            res_str = res_str + '*cxtype(0,1)'
    
        return res_str + '*'



    def get_process_function_definitions(self, write=True):
        """The complete Pythia 8 class definition for the process"""

        replace_dict = super(OneProcessExporterGPU,self).get_process_function_definitions(write=False)


        replace_dict['ncouplings'] = len(self.couplings2order)
        replace_dict['ncouplingstimes2'] = 2 *  replace_dict['ncouplings']
        replace_dict['nparams'] = len(self.params2order)
        replace_dict['nmodels'] = replace_dict['nparams'] + replace_dict['ncouplings']
        replace_dict['coupling_list'] = ' '

        coupling = [''] * len(self.couplings2order)
        params = [''] * len(self.params2order)
        for coup, pos in self.couplings2order.items():
            coupling[pos] = coup
        coup_str = "static cxtype tIPC[%s] = {pars->%s};\n"\
            %(len(self.couplings2order), ',pars->'.join(coupling))
        for para, pos in self.params2order.items():
            params[pos] = para            
        param_str = "static double tIPD[%s] = {pars->%s};\n"\
            %(len(self.params2order), ',pars->'.join(params))            
        
        
        replace_dict['assign_coupling'] = coup_str + param_str
        replace_dict['all_helicities'] = self.get_helicity_matrix(self.matrix_elements[0])
        replace_dict['all_helicities'] = replace_dict['all_helicities'] .replace("helicities", "tHel")
        
        file = self.read_template_file(self.process_definition_template) %\
               replace_dict

        return file

    def get_process_class_definitions(self, write=True):
        
        replace_dict = super(OneProcessExporterGPU,self).get_process_class_definitions(write=False)

        replace_dict['nwavefuncs'] = replace_dict['wfct_size']
        replace_dict['namp'] = len(self.amplitudes.get_all_amplitudes())
        replace_dict['model'] = self.model_name
        
        replace_dict['sizew'] = self.matrix_elements[0].get_number_of_wavefunctions()
        replace_dict['nexternal'], _ = self.matrix_elements[0].get_nexternal_ninitial()
        replace_dict['ncomb'] = len([x for x in self.matrix_elements[0].get_helicity_matrix()])
        
        replace_dict['all_sigma_kin_definitions'] = \
                          """// Calculate wavefunctions
                          __device__ void calculate_wavefunctions(int ihel, double local_mom[%(nexternal)i][3],
                                        cxtype amp[%(namp)d])
                          {
                          const int ncolor =  %(ncolor)d;
                          cxtype jamp[ncolor];

                            cxtype w[%(nwfct)d][%(sizew)d];
                            """ % \
                          {'nwfct':len(self.wavefunctions),
                          'sizew': replace_dict['wfct_size'],
                          'nexternal':replace_dict['nexternal'],
                          'namp':len(self.amplitudes),
                          'ncolor': len(self.matrix_elements[0].get_color_amplitudes())
                          }

        if write:
            file = self.read_template_file(self.process_class_template) % replace_dict
            return file
        else:
            return replace_dict
        
        
#     def get_calculate_wavefunctions(self, wavefunctions, amplitudes, write=True):
#         """Return the lines for optimized calculation of the
#         wavefunctions for all subprocesses"""
# 
#         raise Exception
#         replace_dict = {}
# 
#         replace_dict['nwavefuncs'] = len(wavefunctions)
#         
#         #ensure no recycling of wavefunction ! incompatible with some output
#         #for me in self.matrix_elements:
#         #    me.restore_original_wavefunctions()
# 
#         replace_dict['wavefunction_calls'] = "\n".join(\
#             self.helas_call_writer.get_wavefunction_calls(\
#             helas_objects.HelasWavefunctionList(wavefunctions)))
# 
#         replace_dict['amplitude_calls'] = "\n".join(\
#             self.helas_call_writer.get_amplitude_calls(amplitudes))
# 
#         if write:
#             file = self.read_template_file(self.process_wavefunction_template) % \
#                 replace_dict
#             return file
#         else:
#             return replace_dict
    
    def get_all_sigmaKin_lines(self, color_amplitudes, class_name):
        """Get sigmaKin_process for all subprocesses for Pythia 8 .cc file"""

        ret_lines = []
        if self.single_helicities:
            
            template = "__device__ void calculate_wavefunctions(int ihel, const fptype* allmomenta,fptype &meHelSum \n#ifndef __CUDACC__\n                                , const int ievt\n#endif\n %(multi_channel)s                               )\n{"
            
            if self.include_multi_channel:
                info = {'multi_channel': self.multichannel_var}  
            else:
                info = {'multi_channel': ''}

            ret_lines.append( template % info)

            ret_lines.append(" using namespace MG5_%s;" % self.model_name)
            ret_lines.append("mgDebug( 0, __FUNCTION__ );")
            ret_lines.append("cxtype amp[1]; // was %i" % len(self.matrix_elements[0].get_all_amplitudes()))
            ret_lines.append("const int ncolor =  %i;" % len(color_amplitudes[0]))
            ret_lines.append("cxtype jamp[ncolor];")
            ret_lines.append("// Calculate wavefunctions for all processes")
            ret_lines.append("using namespace MG5_%s;" % self.model_name)


            multi_channel = None
            if self.include_multi_channel:
                if not self.support_multichannel:
                    raise Exception("link with madevent not supported")
                multi_channel = self.get_multi_channel_dictionary(self.matrix_elements[0].get('diagrams'), self.include_multi_channel)

            helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    self.matrix_elements[0],
                                                    color_amplitudes[0],
                                                    multi_channel_map = multi_channel
                                                    )
            assert len(self.matrix_elements) == 1 # how to handle if this is not true?

            self.couplings2order = self.helas_call_writer.couplings2order
            self.params2order = self.helas_call_writer.params2order
            nwavefuncs = self.matrix_elements[0].get_number_of_wavefunctions()
            ret_lines.append("cxtype w[nwf][nw6];")


            ret_lines += helas_calls
            #ret_lines.append(self.get_calculate_wavefunctions(\
            #    self.wavefunctions, self.amplitudes))
            #ret_lines.append("}")
        else:
            ret_lines.extend([self.get_sigmaKin_single_process(i, me) \
                                  for i, me in enumerate(self.matrix_elements)])
        to_add = []
        to_add.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        ret_lines.extend([self.get_matrix_single_process(i, me,
                                                         color_amplitudes[i],
                                                         class_name) \
                                for i, me in enumerate(self.matrix_elements)])
        return "\n".join(ret_lines)

    def write_process_h_file(self, writer):
        """Write the class definition (.h) file for the process"""
        
        replace_dict = super(OneProcessExporterGPU, self).write_process_h_file(False)
        try:
            replace_dict['helamps_h'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.h' % self.model_name)).read()
        except FileNotFoundError:
            replace_dict['helamps_h'] = "\n#include \"../../src/HelAmps_%s.h\"" % self.model_name
        
        if writer:
            file = self.read_template_file(self.process_template_h) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    
    def write_process_cc_file(self, writer):
        """Write the class member definition (.cc) file for the process
        described by matrix_element"""
        
                
        replace_dict = super(OneProcessExporterGPU, self).write_process_cc_file(False)
        #try:
        #    replace_dict['hel_amps_def'] = open(pjoin(self.path, os.pardir, os.pardir,'src','HelAmps_%s.cu' % self.model_name)).read()
        #except FileNotFoundError:
        replace_dict['hel_amps_def'] = "\n#include \"../../src/HelAmps_%s.cu\"" % self.model_name
            
        if writer:
            file = self.read_template_file(self.process_template_cc) % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict


    def get_icolamp_lines(self, mapconfigs, matrix_element, num_matrix_element):
        """Return the ICOLAMP matrix, showing which JAMPs contribute to
        which configs (diagrams)."""

        ret_list = []

        booldict = {False: "false", True: "true"}

        # Only want to include leading color flows, so find max_Nc
        color_basis = matrix_element.get('color_basis')
        if not color_basis:
            # No color, so only one color factor. Simply write a ".true." 
            # for each config (i.e., each diagram with only 3 particle
            # vertices
            text = " {{%s}}" % ','.join(['true']* len(mapconfigs))
            return text


        # There is a color basis - create a list showing which JAMPs have
        # contributions to which configs
        # We don't want to include the power of Nc's which come from the potential
        # loop color trace (i.e. in the case of a closed fermion loop for example)
        # so we subtract it here when computing max_Nc
        max_Nc = max(sum([[(v[4]-v[5]) for v in val] for val in 
                                                      color_basis.values()],[]))

        # Create dictionary between diagram number and JAMP number
        diag_jamp = {}
        for ijamp, col_basis_elem in \
                enumerate(sorted(matrix_element.get('color_basis').keys())):
            for diag_tuple in matrix_element.get('color_basis')[col_basis_elem]:
                # Only use color flows with Nc == max_Nc. However, notice that
                # we don't want to include the Nc power coming from the loop
                # in this counting.
                if (diag_tuple[4]-diag_tuple[5]) == max_Nc:
                    diag_num = diag_tuple[0] + 1
                    # Add this JAMP number to this diag_num
                    diag_jamp[diag_num] = diag_jamp.setdefault(diag_num, []) + \
                                          [ijamp+1]
                #else:
                #    self.proc_characteristic['single_color'] = False

        colamps = ijamp + 1
        for iconfig, num_diag in enumerate(mapconfigs):  
            # mapconfigs can be a list or a dictionary.
            # In case of dictionary the num_diag will be the key of the dictionary.    
            if num_diag == 0:
                continue

            # List of True or False 
            bool_list = [(i + 1 in diag_jamp[num_diag]) for i in range(colamps)]
            # Add line
            ret_list.append("{%s};" % ','.join(["%s" % booldict[b] for b in bool_list]))



        return '{%s};' % ','.join(ret_list)

def get_mg5_info_lines():
    """Return info lines for MG5, suitable to place at beginning of
    Fortran files"""

    info = misc.get_pkg_info()
    info_lines = ""
    if info and 'version' in info and  'date' in info:
        info_lines = "#  MadGraph5_aMC@NLO v. %s, %s\n" % \
                     (info['version'], info['date'])
        info_lines = info_lines + \
                     "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                     "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"
    else:
        info_lines = "#  MadGraph5_aMC@NLO\n" + \
                     "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                     "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"        

    return info_lines




#===============================================================================
# ProcessExporterGPU
#===============================================================================
class ProcessExporterGPU(ProcessExporterCPP):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    grouped_mode = False
    exporter = 'gpu'

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True
                        }
    
    oneprocessclass = OneProcessExporterGPU
    s= _file_path + 'iolibs/template_files/'
    from_template = {'src': [s+'gpu/rambo.h', s+'gpu/rambo.cc', s+'read_slha.h', s+'read_slha.cc',
                             s+'gpu/mgOnGpuTypes.h', s+'gpu/grambo.cu'],
                    'SubProcesses': [s+'gpu/timer.h', s+'gpu/Makefile', s+'gpu/nvtx.h',
                                     s+'gpu/nvtx.h', s+'gpu/check.cc',
                                     s+'gpu/timermap.h', s+'gpu/profile.sh',
                                     s+'gpu/perf.py', s+ 'gpu/Memory.h', s + 'gpu/runTest.cc']}
    to_link_in_P = ['Makefile', 'timer.h', 'timermap.h', 'nvtx.h', 'perf.py', 'Memory.h', 'runTest.cc']

    template_src_make = pjoin(_file_path, 'iolibs', 'template_files','gpu','Makefile_src')
    template_Sub_make = pjoin(_file_path, 'iolibs', 'template_files','gpu','Makefile')
    create_model_class =  UFOModelConverterGPU
    
    def compile_model(self):
        return

