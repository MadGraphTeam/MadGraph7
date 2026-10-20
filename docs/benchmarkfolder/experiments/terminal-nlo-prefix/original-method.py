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
    The context specified will be forwarded to the the fileWriter."""
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
            has_private = re.search(r'\bCALL MP_\w+TERMINAL_WL_\d+_\d+\(',
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
