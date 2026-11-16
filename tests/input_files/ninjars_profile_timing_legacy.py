"""Pre-metadata timer method fixture for mocked call/float regression checks.

Exact method source from the audited pre-metadata v2 timing implementation.
Only definitions are compiled by the unit tests; no compiler/physics is invoked.
"""
class LoopMatrixElementTimer:
    def skip_loop_evaluation_setup(self, dir_name, skip=True):
        """ Edit loop_matrix.f in order to skip the loop evaluation phase.
        Notice this only affects the double precision evaluation which is
        normally fine as we do not make the timing check on mp."""

        file = open(pjoin(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(pjoin(dir_name,'loop_matrix.f'), 'w')
        loop_matrix = re.sub(r"SKIPLOOPEVAL=\S+\)","SKIPLOOPEVAL=%s)"%('.TRUE.' 
                                           if skip else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def boot_time_setup(self, dir_name, bootandstop=True):
        """ Edit loop_matrix.f in order to set the flag which stops the
        execution after booting the program (i.e. reading the color data)."""

        file = open(pjoin(dir_name,'loop_matrix.f'), 'r')
        loop_matrix = file.read()
        file.close()
        
        file = open(pjoin(dir_name,'loop_matrix.f'), 'w')        
        loop_matrix = re.sub(r"BOOTANDSTOP=\S+\)","BOOTANDSTOP=%s)"%('.TRUE.' 
                                    if bootandstop else '.FALSE.'), loop_matrix)
        file.write(loop_matrix)
        file.close()

    def time_matrix_element(self, matrix_element, reusing = False,
                       param_card = None, keep_folder = False, options=None,
                       MLOptions = {}):
        """ Output the matrix_element in argument and give detail information
        about the timing for its output and running"""

        # If True, then force three PS points only and skip the test on
        # unpolarized PS point 
        make_it_quick=False

        if options and 'split_orders' in list(options.keys()):
            split_orders = options['split_orders']
        else:
            split_orders = -1

        assert ((not reusing and isinstance(matrix_element, \
                 helas_objects.HelasMatrixElement)) or (reusing and 
                              isinstance(matrix_element, base_objects.Process)))
        if not reusing:
            proc_name = matrix_element['processes'][0].shell_string()[2:]
        else:
            proc_name = matrix_element.shell_string()[2:]
        
        export_dir=pjoin(self.output_path,('SAVED' if keep_folder else '')+\
                                                temp_dir_prefix+"_%s"%proc_name)

        res_timings = self.setup_process(matrix_element,export_dir, \
            reusing, param_card,MLOptions = MLOptions,clean=True)
        
        if res_timings == None:
            return None
        dir_name=res_timings['dir_path']

        def check_disk_usage(path):
            return subprocess.check_output(
                [f"du -shc {path}"], shell=True, encoding="utf8"
            ).split()[-2]

        res_timings['du_source']=check_disk_usage(pjoin(\
                                                 export_dir,'Source','*','*.f'))
        res_timings['du_process']=check_disk_usage(pjoin(dir_name,'*.f'))
        res_timings['du_color']=check_disk_usage(pjoin(dir_name,
                                                  'MadLoop5_resources','*.dat'))
        res_timings['du_exe']=check_disk_usage(pjoin(dir_name,'check'))

        if not res_timings['Initialization']==None:
            time_per_ps_estimate = (res_timings['Initialization']/4.0)/2.0
        elif make_it_quick:
            time_per_ps_estimate = -1.0            
        else:
            # We cannot estimate from the initialization, so we run just a 3
            # PS point run to evaluate it.
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                                  read_ps = False, npoints = 3, hel_config = -1, 
                                                      split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
            time_per_ps_estimate = run_time/3.0
        
        self.boot_time_setup(dir_name,bootandstop=True)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        res_timings['Booting_time'] = run_time
        self.boot_time_setup(dir_name,bootandstop=False)

        # Detect one contributing helicity
        contributing_hel=0
        n_contrib_hel=0
        proc_prefix_file = open(pjoin(dir_name,'proc_prefix.txt'),'r')
        proc_prefix = proc_prefix_file.read()
        proc_prefix_file.close()
        helicities = open(pjoin(dir_name,'MadLoop5_resources',
                                  '%sHelFilter.dat'%proc_prefix)).read().split()
        for i, hel in enumerate(helicities):
            if (self.loop_optimized_output and int(hel)>-10000) or hel=='T':
                if contributing_hel==0:
                    contributing_hel=i+1
                n_contrib_hel += 1
                    
        if contributing_hel==0:
            logger.error("Could not find a contributing helicity "+\
                                     "configuration for process %s."%proc_name)
            return None
        
        res_timings['n_contrib_hel']=n_contrib_hel
        res_timings['n_tot_hel']=len(helicities)
        
        # We aim at a 30 sec run
        if not make_it_quick:
            target_pspoints_number = max(int(30.0/time_per_ps_estimate)+1,50)
        else:
            target_pspoints_number = 10
        
        logger.info("Checking timing for process %s "%proc_name+\
                                    "with %d PS points."%target_pspoints_number)
        
        MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        
        if compile_time == None: return None
        
        res_timings['run_polarized_total']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)

        if make_it_quick:
            res_timings['run_unpolarized_total'] = 1.0
            res_timings['ram_usage'] = 0.0
        else:
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                 read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                          split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name, 
                                                                      checkRam=True)

            if compile_time == None: return None
            res_timings['run_unpolarized_total']=\
                       (run_time-res_timings['Booting_time'])/target_pspoints_number
            res_timings['ram_usage'] = ram_usage       
        
        if not self.loop_optimized_output:
            return res_timings
        
        # For the loop optimized output, we also check the time spent in
        # computing the coefficients of the loop numerator polynomials.
        
        # So we modify loop_matrix.f in order to skip the loop evaluation phase.
        self.skip_loop_evaluation_setup(dir_name,skip=True)

        if make_it_quick:
            res_timings['run_unpolarized_coefs'] = 1.0
        else:
            MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                 read_ps = False, npoints = target_pspoints_number, hel_config = -1,
                                                          split_orders=split_orders)
            compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
            if compile_time == None: return None
            res_timings['run_unpolarized_coefs']=\
                       (run_time-res_timings['Booting_time'])/target_pspoints_number
        
        MadLoopInitializer.fix_PSPoint_in_check(pjoin(export_dir,'SubProcesses'),
                          read_ps = False, npoints = target_pspoints_number*2, \
                       hel_config = contributing_hel, split_orders=split_orders)
        compile_time, run_time, ram_usage = MadLoopInitializer.make_and_run(dir_name)
        if compile_time == None: return None
        res_timings['run_polarized_coefs']=\
               (run_time-res_timings['Booting_time'])/(target_pspoints_number*2)    

        # Restitute the original file.
        self.skip_loop_evaluation_setup(dir_name,skip=False)
        
        return res_timings

