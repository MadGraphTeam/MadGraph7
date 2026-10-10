//==========================================================================
// A class for calculating the matrix elements for
%(process_lines)s
//--------------------------------------------------------------------------

class CPPProcess
{
public:

  // Constants for array limits
  static const int ninitial = %(ninitial)d;
  static const int nexternal = %(nexternal)d;
  static const int nprocesses = %(nprocesses)d;
  static const int ncomb = %(ncomb)d;
  static const int ndiagrams = %(ndiag)d;
  static const int namplitudes = %(namp)d;
  static const int nflavors = %(nflav)d;

  // Constructor.
  CPPProcess(string param_card_name);

  // Calculate flavour-independent parts of cross section.
  double sigmaKin(int flavor_id);

  // Info on the subprocess.
  string name()    const {return "%(process_definition)s";}

  int    code()    const {return %(process_code)s;}

  const vector<double>& getMasses() const {return mME;}

  const double* getAmp2() const { return amp2; }

  Parameters_%(model_name)s& getParameters() { return pars; }

  // Get and set momenta for matrix element evaluation
  vector<double*>& getMomenta(){return p;}
  void setMomenta(vector<double*>& momenta){p = momenta;}
  void setInitial(int inid1, int inid2){id1 = inid1; id2 = inid2;}

private:

  // Private functions to calculate the matrix element for all subprocesses
  %(all_sigma_kin_definitions)s
  %(all_matrix_definitions)s

  std::complex<double> amp[namplitudes];
  double amp2[ndiagrams];

  // Color flows, used when selecting color
  double* jamp2;

  // Pointer to the model parameters
  Parameters_%(model_name)s pars;

  // vector with external particle masses
  vector<double> mME;

  // vector with momenta (to be changed each event)
  vector<double*> p;
  // Initial particle ids
  int id1, id2;

  // Good-helicity filter: goodhel[flav][ihel] marks the rows with a non-zero
  // |M|^2, igood[flav][0..ngood-1] lists them (0-based, so ncomb entries hold
  // every row).
  bool goodhel[nflavors][ncomb];
  int ntry[nflavors], sum_hel[nflavors], ngood[nflavors];
  int igood[nflavors][ncomb];
  int jhel[nflavors];

  // C-parity de-duplication of the helicity sum (uncrossed process only, see
  // sigmaKin): flip[ihel] is the helicity row with every helicity negated (an
  // involution, built once); ntry_csym[flav] counts the calls the reuse may
  // apply to, the first ones validating it; csym_bad[flav] latches true once
  // ANY row fails to pair up or ANY pair shows |M(ihel)| != |M(flip)| at a scan
  // point, so the reuse is all-or-nothing per flavor and halves the loop
  // uniformly. Once validated, the full sum computes the lower-index row of
  // each pair only and counts it twice; for the helicity-sampling mode the good
  // helicities of a flavor are reduced to those representatives
  // (igoodrep[flav][0..nrep-1]) carrying a doubled weight (repwgt).
  int flip[ncomb];
  bool flip_ready;
  int ntry_csym[nflavors];
  bool csym_bad[nflavors];
  int igoodrep[nflavors][ncomb];
  int nrep[nflavors];
  int repwgt[nflavors][ncomb];

  // function to compute missing symmetry factors after flavor consolidation
  int broken_sym(const int* flavor);
%(cross_member_decl)s

};
