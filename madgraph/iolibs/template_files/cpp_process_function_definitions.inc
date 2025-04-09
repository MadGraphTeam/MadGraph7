//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s

//--------------------------------------------------------------------------
// Initialize process. 
  
CPPProcess::CPPProcess(string param_card_name) :
    goodhel(), ntry(0), sum_hel(0), ngood(0), igood(), jhel(0)
{
// Instantiate the model class and set parameters that stay fixed during run
    SLHAReader slha(param_card_name);
    pars.setIndependentParameters(slha);
    pars.setIndependentCouplings();
    %(initProc_lines)s
} 

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour. 

void CPPProcess::sigmaKin( int* flavor) { 
    // Set the parameters which change event by event
    pars.setDependentParameters();
    pars.setDependentCouplings();

    // Reset color flows
    %(reset_jamp_lines)s
    %(sigmaKin_lines)s
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence. 

double CPPProcess::sigmaHat() {  
    %(sigmaHat_lines)s
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

%(all_sigmaKin)s
