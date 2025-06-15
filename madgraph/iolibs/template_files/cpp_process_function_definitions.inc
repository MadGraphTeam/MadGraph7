//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

//--------------------------------------------------------------------------
// Initialize process. 
  
CPPProcess::CPPProcess(string param_card_name) :
    goodhel(), ntry(), sum_hel(), ngood(), igood(), jhel()
{
// Instantiate the model class and set parameters that stay fixed during run
    SLHAReader slha(param_card_name, false);
    pars.setIndependentParameters(slha);
    pars.setIndependentCouplings();
    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

double CPPProcess::sigmaKin(int flavor_id, bool mirror) { 
    // Set the parameters which change event by event
    pars.setDependentParameters();
    pars.setDependentCouplings();

    for (int i = 0; i < ndiagrams; ++i) {
        amp2[i] = 0;
    }

    // Reset color flows
    %(reset_jamp_lines)s
    %(sigmaKin_lines)s
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

int CPPProcess::broken_sym(int* flavor)
{
    int old_factor = %(get_old_symmmetry_value)s;
    %(get_pid)s
    int nincoming = %(nincoming)d;
    for (int i = 0; i < (nexternal - nincoming - 1); i++)
    {
        if(pid[i] == 0)
            continue;
        int n_tot = 1;
        for (int j = i + 1; j < (nexternal - nincoming - 1); j++)
        {
            if((pid[i] == pid[j]) && (flavor[i] == flavor[j]))
            {
                pid[j] = 0;
                n_tot = n_tot + 1;
                old_factor = old_factor / n_tot;
            }
        }
    }
    return old_factor;
}
