! Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MG5aMC CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
! Further modified by: D. Massaro, A. Thete, A. Valassi (2021-2025).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================
  //==========================================================================
  // A class for calculating the matrix elements for
  %(process_lines)s
  //--------------------------------------------------------------------------

  class CPPProcess
  {
  public: /* clang-format off */

    // Constructor (from command line arguments)
    CPPProcess( bool verbose = false, bool debug = false );

    // Destructor
    ~CPPProcess();

    // Initialize process (read model parameters from file)
    virtual void initProc( const std::string& param_card_name );

    // Retrieve the compiler that was used to build this module
    static const std::string getCompiler();

    // PDG id for a given flavor combination index and particle index (0-based).
    // Useful for tools that need to enumerate flavor combinations without going through UMAMI.
    static int flavorPDG( int iflavor, int ipar );

    // Other methods of this instance (???)
    const std::vector<fptype>& getMasses() const { return m_masses; }
    //virtual int code() const{ return 1; }
    //void setInitial( int inid1, int inid2 ){ id1 = inid1; id2 = inid2; }
    //int getDim() const { return dim; }
    //int getNIOParticles() const { return nexternal; } // nexternal was nioparticles

    // Accessors (unused so far: add four of them only to fix a clang build warning)
    //bool verbose() const { return m_verbose; }
    bool debug() const { return m_debug; }

  public:

    // Compile-time constants: single source of truth is ProcessData.h
    static constexpr int np4 = ProcessData::np4;
    static constexpr int nw6 = ProcessData::nw6;
    static constexpr int npari = ProcessData::npari;
    static constexpr int nparf = ProcessData::nparf;
    static constexpr int npar = ProcessData::npar;
    static constexpr int ncomb = ProcessData::ncomb;
    static constexpr int ndiagrams = ProcessData::ndiagrams;
    static constexpr int ncolor = ProcessData::ncolor;
    static constexpr int nmaxflavor = ProcessData::nmaxflavor;

  private: /* clang-format on */

    // Command line arguments (constructor)
    bool m_verbose;
    bool m_debug;

    // Physics model parameters to be read from file (initProc function)
#ifndef MGONGPU_HARDCODE_PARAM
    Parameters* m_pars;
#endif
    std::vector<fptype> m_masses; // external particle masses

    // Other variables of this instance (???)
    //int id1, id2; // initial particle ids
    //cxtype** amp; // ???
  };
