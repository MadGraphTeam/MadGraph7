// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin, calculate_jamps, good-helicity filtering.
// The diagram/vertex-call sequence (helas_calls) is process-specific and
// lives in the P1-generated EvaluateDiagrams.inc, #include'd below.
//
// This is a first, foundational slice: the process-independent storage the
// rest of this file operates on (cHel/cFlavors/cIPD/cIPC/cIPF/bsmIndepParam),
// its setters, and the two helpers (getChannelId, computeDependentCouplings)
// that don't depend on calculate_jamps/sigmaKin, which land in a later slice.

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessDenominators.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessNumerators.h"
#include "MemoryAccessWavefunctions.h"
#include "ColorData.h"       // for shouldUseBlas/mgOnGpu::nchannels/channel2iconfig/icolamp/nconfigSDE
#include "color_sum.h"       // for color_sum_cpu/color_sum_cpu_blas
#include "mgOnGpuConfig.h"

#include <algorithm>
#include <cassert>
#include <cstdlib> // for std::abort (crossing guard in EvaluateDiagrams.inc)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace madmatrix
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // The number of SIMD vectors of events processed by calculate_jamps
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
  constexpr int nParity = 2;
#else
  constexpr int nParity = 1;
#endif

  // ncolor_flow (unlike ncolor) is not in ProcessData.h: the color sum can run
  // on a smaller (DDM) basis than the color flow probabilities do, so this stays
  // a CPPProcess-generated constant (see process_class.inc/set_color_flow_lines_cpp).
  constexpr int ncolor_flow = CPPProcess::ncolor_flow;

  // Helicity/flavor tables and SM parameter/coupling storage, populated once
  // by CPPProcess's constructor/initProc via the setters below.
  static short cHel[ncomb][npar];
  static short cFlavors[nmaxflavor][npar];
  static int cNGoodHel;
  static int cGoodHel[ncomb];
  static fptype cIPD[nIPD > 0 ? nIPD : 1];
  static fptype cIPC[nIPC > 0 ? nIPC * 2 : 1];
  static int cIPF_partner1[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  static int cIPF_partner2[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  static fptype cIPF_value[ProcessTables::nMF * nIPF * 2 > 0 ? ProcessTables::nMF * nIPF * 2 : 1];
  static double bsmIndepParam[Parameters::nBsmIndepParam > 0 ? Parameters::nBsmIndepParam : 1];
  // BWCUTOFF of the $-excluded propagators (ALOHA P1D tag), see setBwCutoff
  static fptype cBWCUTOFF = 15.;

  //--------------------------------------------------------------------------
  // Crossing symmetry (ProcessTables::use_crossing, from --use_crossing).
  // An event's flavor index is then the EXTENDED id K*nmaxflavor + flav:
  // flav picks the flavor combination as usual (and is constant across a SIMD
  // page, see umami.cc), while the crossing-table row K may differ from lane
  // to lane. calculate_jamps permutes each lane's momenta into the base slot
  // order (the preamble of EvaluateDiagrams.inc) and evaluates, in every lane,
  // the SAME helicity row of the base table: a leg that changed side is fed
  // the opposite helicity (with its NSF flag, see _crossing_external_block),
  // which is the same all-outgoing configuration. The crossed amplitude being
  // the analytic continuation of the base one, the crossings share their zero
  // rows, so ONE good-helicity list (cGoodHel) serves every lane: the scan
  // below takes the union over the crossings this ME records, which only
  // differs from the base's own by the frame-dependent rows of a massive leg.

  // C-parity good-helicity de-duplication. Two helicity rows that are exact
  // mirrors (every helicity negated) give an identical |M|^2 under a parity/C
  // conserving amplitude, so only one of the two need ever be computed: the
  // good-helicity list is REDUCED to the lower-index representative of each
  // C-pair, every representative is counted twice, and the event-by-event
  // helicity choice returns the representative or its cFlip partner at equal
  // rate. That halves the sigmaKin trip count and the calculate_jamps + colour
  // sum calls. The verdict is taken in the (serial) good-helicity scan, so
  // sigmaKin only ever reads these tables and stays thread-safe. It is
  // all-or-nothing: one unpaired good row or one mismatched pair turns it off.
  // With crossing the pairing is the same for every crossing (each evaluates
  // the base row, so the mirror of a row is the base mirror), and the scan
  // tests every crossing: one verdict for all of them.
  static int cFlip[ncomb];            // C-parity partner: every helicity negated (an involution)
  static bool cCsymScanned;           // the validating scan actually ran (never trust a default)
  static bool cCsymBad;               // latched when ANY pair mismatched at a scan point (any crossing)
  static bool cCsymOk;                // the de-duplication is on

  // Initial-state spin*color average of the process crossing-table row
  // `cross` crosses into (the product of the per-leg spin*color of the base
  // legs it puts in the initial state, tabulated by the exporter). 0 for a
  // row out of range, which the per-event denominator turns into a zero ME.
  inline int
  spincol_cross( int cross )
  {
    return ( cross >= 0 && cross < ncross ) ? xspincol_tab[cross] : 0;
  }

  // Identical-final-state factor (product of n!) of the crossed process.
  // Flavor dependent, hence runtime: two crossed final legs are identical when
  // they carry the same flavor group (same representative PDG -- ids_base,
  // conjugated to antipid_base when the leg changes side) and the same actual
  // flavor. FLAVOR is not permuted, so input slot k reads the base leg
  // pinv[k] it is fed to: cFlavors[iflavor][pinv[k]]. A decay-block leaf
  // (countable_tab 0) is skipped -- a crossing never moves one, and the
  // resonance-level symmetry of the blocks is the constant ident_resonance --
  // exactly as the fortran GET_IDENT_CROSS.
  int
  ident_cross( int cross, int iflavor )
  {
    int perm[npar], ic[npar];
    cross_pinv( cross, perm, ic );
    int bpid[npar];
    for( int k = 0; k < npar; k++ )
      bpid[k] = ( ic[k] == 1 ) ? ids_base[perm[k]] : antipid_base[perm[k]];
    bool used[npar];
    for( int k = 0; k < npar; k++ ) used[k] = false;
    int fact = ident_resonance;
    for( int k = npari; k < npar; k++ )
    {
      if( used[k] || !countable_tab[perm[k]] ) continue;
      int n = 1;
      for( int l = k + 1; l < npar; l++ )
      {
        if( used[l] || !countable_tab[perm[l]] ) continue;
        if( bpid[k] == bpid[l] && cFlavors[iflavor][perm[k]] == cFlavors[iflavor][perm[l]] )
        {
          used[l] = true;
          n = n + 1;
          fact = fact * n;
        }
      }
    }
    return fact;
  }

  // Crossed-event selected helicity code (allselhel). For a crossed event the
  // reported helicity is the CROSSED process's own canonical code, not the
  // base row: input slot k carries base leg pinv[k], which the crossed lane
  // evaluated with the base row's helicity negated when the leg changed side
  // (xic[k] = -1, see _crossing_external_block). The NSF flag keeps that fed
  // value physical, so it is the crossed leg's helicity, encoded in mixed
  // radix over the crossed process's own per-leg states (xhel_states row K,
  // see the exporter). That is the code the expanded output of the crossed
  // process reports, the one the madevent output writes for a routed event,
  // and the one an `output mg7` crossed entry decodes with its own helicity
  // table. Row 0 is the identity (base row+1), so the non-crossing path is
  // unchanged; a row out of the table reports 0 (its |M|^2 is 0).
  //
  // xhel_states MUST be the allow_reverse=True per-leg order (see the exporter).
  inline int
  selected_hel_code( int base_ihel, unsigned int flavor_id )
  {
    const int xcross = (int)( flavor_id / nmaxflavor );
    if( xcross == 0 ) return base_ihel + 1;
    if( xcross >= ncross ) return 0;
    int xperm[npar], xic[npar];
    cross_pinv( xcross, xperm, xic );
    int code = 0;
    for( int k = 0; k < npar; k++ )
    {
      const int val = (int)cHel[base_ihel][xperm[k]] * xic[k];
      const int slot = xcross * npar + k; // this crossing's own states for slot k
      int d = 0;
      for( int dd = 0; dd < xhel_nhstate[slot]; dd++ )
      {
        if( xhel_states[slot * xhel_maxhel + dd] == val )
        {
          d = dd;
          break;
        }
      }
      code = code * xhel_nhstate[slot] + d;
    }
    return code + 1;
  }

  // Pick the helicity row to report for the ighel-th good helicity of a lane.
  // The row stands for a C-parity PAIR counted twice when the de-duplication
  // is on, so either member must come out at equal rate or the event-level
  // helicity distribution is biased while |M|^2 and the cross section stay
  // perfectly correct. The fair coin is recycled from the selection variate
  // itself: given that the (unnormalised) CDF landed in [lo,hi), rnd is exactly
  // uniform on that interval, so its position within the bin is an independent
  // U(0,1). Drawing a fresh random number instead would desynchronise the
  // stream shared with the Fortran integrator.
  // Returns the 0-based base row; every lane, crossed or not, evaluated the
  // ighel-th row of the one good-helicity list.
  inline int
  csym_selected_row( const int ihel, const fptype rnd, const fptype lo, const fptype hi )
  {
    if( !cCsymOk ) return ihel;
    const fptype w = hi - lo;
    if( !( w > (fptype)0 ) ) return ihel; // degenerate bin: cannot be selected anyway
    return ( ( rnd - lo ) < (fptype)0.5 * w ) ? ihel : cFlip[ihel];
  }

  // The reported (Fortran-indexed, [1,ncomb]) helicity of one lane
  inline int
  selected_helicity( int ighel, unsigned int flavor_id, fptype rnd, fptype lo, fptype hi )
  {
    const int row = csym_selected_row( cGoodHel[ighel], rnd, lo, hi );
    if constexpr( use_crossing )
      return selected_hel_code( row, flavor_id );
    else
      return row + 1;
  }

  // Whether the helicity sum of the lane with this flavor id is C-parity
  // de-duplicated (its representatives then each stand for two rows): one
  // verdict for every crossing, see sigmaKin_getGoodHel
  inline bool
  csym_lane_on( unsigned int flavor_id )
  {
    (void)flavor_id;
    return cCsymOk;
  }

  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors )
  {
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
    memcpy( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
  }

  void setIndependentParams( const fptype* tIPD )
  {
    if( nIPD > 0 ) memcpy( cIPD, tIPD, nIPD * sizeof( fptype ) );
  }

  void setIndependentCouplings( const cxtype* tIPC )
  {
    if( nIPC > 0 ) memcpy( cIPC, tIPC, nIPC * sizeof( cxtype ) );
  }

  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value )
  {
    if( nIPF == 0 ) return;
    memcpy( cIPF_partner1, tIPF_partner1, ProcessTables::nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_partner2, tIPF_partner2, ProcessTables::nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_value, tIPF_value, ProcessTables::nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) memcpy( bsmIndepParam, values, n * sizeof( double ) );
  }

  void setBwCutoff( const double bwcutoff )
  {
    cBWCUTOFF = bwcutoff;
  }

  const short* getHelicities()
  {
    return &cHel[0][0];
  }

  //--------------------------------------------------------------------------

  // SCALAR channelId for the whole SIMD neppV2 event page (C++), i.e. one or two neppV event page(s)
  // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a neppV2 SIMD event page
  // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
  __device__ INLINE unsigned int
  getChannelId( const unsigned int* allChannelIds, const int ievt00, bool sanityCheckMixedPrecision = true )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = HostAccessChannelIds; // non-trivial access: buffer includes all events
    if( allChannelIds != nullptr )
    {
      // First - and/or only - neppV page of channels (iParity=0 => ievt0 = ievt00 + 0 * neppV)
      const unsigned int* channelIds = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 ); // fix bug #899/#911
      uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds );                           // fix #895 (compute this only once for all diagrams)
#ifndef MGONGPU_CPPSIMD
      // NB: channelIds_sv is a scalar in no-SIMD C++
      channelId = channelIds_sv;
#else
      // NB: channelIds_sv is a vector in SIMD C++
      channelId = channelIds_sv[0];    // element[0]
      for( int i = 1; i < neppV; ++i ) // elements[1...neppV-1]
      {
        assert( channelId == channelIds_sv[i] ); // SANITY CHECK #898: check that all events in a SIMD vector have the same channelId
      }
#endif
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
      if( sanityCheckMixedPrecision )
      {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        // Second neppV page of channels (iParity=1 => ievt0 = ievt00 + 1 * neppV)
        const unsigned int* channelIds2 = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 + neppV ); // fix bug #899/#911
        uint_v channelIds2_v = CID_ACCESS::kernelAccessConst( channelIds2 );                                    // fix #895 (compute this only once for all diagrams)
        // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
        for( int i = 0; i < neppV; ++i )
        {
          assert( channelId == channelIds2_v[i] ); // SANITY CHECKS #898 #924: all events in the 2nd SIMD vector have the same channelId as that of the 1st SIMD vector
        }
#else
        (void)sanityCheckMixedPrecision; // no second SIMD page to cross-check outside mixed precision
#endif
      }
    }
    return channelId;
  }

  //--------------------------------------------------------------------------

  __global__ void
  computeDependentCouplings( const fptype* allgs, fptype* allcouplings, const int nevt )
  {
    using G_ACCESS = HostAccessGs;
    using C_ACCESS = HostAccessCouplings;
    for( int ipagV = 0; ipagV < nevt / neppV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      const fptype* gs = MemoryAccessGs::ieventAccessRecordConst( allgs, ievt0 );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<G_ACCESS, C_ACCESS>( gs, couplings, bsmIndepParam );
    }
  }

  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------

  // The mass an external leg marked off-shell ('*': MadSpin's decaying particles)
  // is evaluated with. The Fortran output and the scalar backend take the
  // invariant mass of its momentum; here one call evaluates several events, whose
  // invariant masses differ, with a single mass, so the pole mass stays. That is
  // the same as long as the momenta are on shell, which is all mg7 generates;
  // off-shell momenta need the scalar backend (madmatrix.density refuses others).
  template<class M_ACCESS>
  inline fptype_amp
  offshell_mass( const fptype_amp poleMass, const fptype_momenta*, const int )
  {
    return poleMass;
  }

  //--------------------------------------------------------------------------

  // Accumulate a multichannel numerator contribution in place. In C++ each
  // event page is processed serially within the helicity loop, so a plain
  // sum suffices (CUDA needs atomicAdd instead: see backend/gpu/SigmaKin.cc).
#define NUM_ATOMIC_ADD( DST, VAL ) ( DST ) += ( VAL )

  // --hel_recycling warm-up probe (madmatrix/output.py). The recycled build drops
  // the amplitudes that vanish at a given helicity -- in particular the ones a
  // gauge choice (--axial_gauge) makes vanish identically -- and it has to be told
  // which those are. A build with -DMG_HR_PROBE has every amplitude of the
  // generated diagrams record |amp|^2 (MG_HR_PROBE_AMP), and an amplitude is alive
  // at a helicity once it is above zeroamp_tol of the largest amplitude of that
  // helicity at the same point: a cancellation zero is a rounding residual, never
  // an exact 0 (the same rule as the fortran hel_zeroamp detection). The answer is
  // the union over the points and extended flavor ids (every crossing) that
  // getGoodHel samples, so an amplitude is dropped at a base row only if it
  // vanishes for every crossing served; sigmaKin_setGoodHel prints it under
  // MG_DUMP_GOODHEL. Scalar double builds only.
#ifdef MG_HR_PROBE
#if defined MGONGPU_CPPSIMD or !defined MGONGPU_FPTYPE_DOUBLE
#error "MG_HR_PROBE needs BACKEND=scalar FPTYPE=d"
#endif
  static std::vector<double> hrProbeAbs2;           // |amp|^2 at the current point, by amplitude number
  static std::vector<std::vector<char>> hrProbeSeen;  // [ihel][namp]: evaluated at this helicity
  static std::vector<std::vector<char>> hrProbeAlive; // [ihel][namp]: not a cancellation zero
  inline void
  hr_probe_record( const int namp, const cxtype_amp_sv& amp )
  {
    if( (int)hrProbeAbs2.size() <= namp ) hrProbeAbs2.resize( namp + 1, -1. );
    hrProbeAbs2[namp] = cxabs2( amp );
  }
  inline void
  hr_probe_close( const int ihel )
  {
    constexpr double zeroamp_tol = 1e-12;
    if( hrProbeSeen.empty() )
    {
      hrProbeSeen.resize( ncomb );
      hrProbeAlive.resize( ncomb );
    }
    double ampmax = 0;
    for( double a : hrProbeAbs2 ) ampmax = std::max( ampmax, a );
    std::vector<char>& seen = hrProbeSeen[ihel];
    std::vector<char>& alive = hrProbeAlive[ihel];
    if( seen.size() < hrProbeAbs2.size() )
    {
      seen.resize( hrProbeAbs2.size(), 0 );
      alive.resize( hrProbeAbs2.size(), 0 );
    }
    for( size_t namp = 0; namp < hrProbeAbs2.size(); namp++ )
    {
      if( hrProbeAbs2[namp] < 0 ) continue;
      seen[namp] = 1;
      if( hrProbeAbs2[namp] > zeroamp_tol * zeroamp_tol * ampmax ) alive[namp] = 1;
      hrProbeAbs2[namp] = -1.;
    }
  }
#define MG_HR_PROBE_AMP( NAMP ) hr_probe_record( NAMP, amp_sv[0] )
#else
#define MG_HR_PROBE_AMP( NAMP )
#endif

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // Also compute running sums over helicities adding jamp2, numerator, denominator
  // (NB: this function no longer handles matrix elements as the color sum has now been
  // moved to a separate function/kernel). This function processes a single event "page"
  // or SIMD vector (or for two in "mixed" precision mode, nParity=2). Accepts a SCALAR
  // channelId because it is GUARANTEED that all events in a SIMD vector have the same
  // channelId #898.
  void
  calculate_jamps( const short* hel,                 // input: helicity of each leg, hel[npar] (a row of cHel, or not: see Density.cc)
                   const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,     // input: indices of the flavor combinations
                   cxtype_amp_sv* allJamp_sv,          // output: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed) for this helicity
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,          // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype_amp* allDenominators,        // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype_amp_sv* jamp2_sv,            // output: jamp2[nParity][ncolor_flow][neppV] for color choice (nullptr if disabled)
                   const int ievt00 )                  // input: first event number in current C++ event page
  {
    using M_ACCESS = HostAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = HostAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = HostAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = HostAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = HostAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using CDPF_ACCESS = CD_ACCESS;              // the running flavor couplings gathered into dpf_value (one event page, as CD_ACCESS)
    using F_ACCESS = HostAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;    // non-trivial access: buffer includes all events
    mgDebug( 0, __FUNCTION__ );

    // EvaluateDiagrams.inc reads the helicities as cHel[ihel][ipar]. Shadow the
    // helicity table with the one row this call evaluates, so that the row need
    // not be one of cHel at all: the density matrix builds combinations that a
    // polarised process leaves out of its table (sigmaKin_density, Density.cc).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
    const short( *cHel )[npar] = reinterpret_cast<const short( * )[npar]>( hel );
#pragma GCC diagnostic pop
    constexpr int ihel = 0;

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given C++ event
    // page (ipagV) [NB these variables are reused several times (and re-initialised each
    // time) within the same event or event page]. Create memory for both momenta and
    // wavefunctions separately, and later wrap them in ALOHAOBJ.
    fptype_momenta_sv pvec_sv[nwf][np4];
    cxtype_amp_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams
    cxtype_amp_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype_amp* amp_fp = reinterpret_cast<fptype_amp*>( amp_sv );

    // special temporary ALOHAOBJ to hold F/Vtmp values in the combined vertex functions
    // while using the FD gauge (harmless, unused, when the model doesn't need it)
    fptype_momenta_sv pvec_sv_tmp[1][np4];
    cxtype_amp_sv w_sv_tmp[1][nw6];
    ALOHAOBJ aloha_obj_tmp[1];
    aloha_obj_tmp[0] = ALOHAOBJ{ pvec_sv_tmp[0], w_sv_tmp[0] };
    cxtype_amp_sv amp_tmp_sv[1]; // to ensure proper aligment for vector instructions
    fptype_amp* amp_tmp_fp = reinterpret_cast<fptype_amp*>( amp_tmp_sv );

    // jamp: sum (for one event or event page) of the invariant amplitudes for all Feynman
    // diagrams in a given color combination (NB: vector cxtype_v IS initialized to 0, but
    // scalar cxtype is NOT, if "= {}" is missing!)
    // (njampso = ncolor * nampso: one vector per amplitude split order, just ncolor without them)
    cxtype_amp_sv jamp_sv[njampso] = {};
    // jampTmp: partial sums of amplitudes that several color flows share, so that they are
    // computed only once (see MadMatrixUFOHelasCallWriter.build_jamp_plan); no "= {}", each
    // one is assigned before it is ever read.
    cxtype_amp_sv jampTmp_sv[ProcessTables::nb_tmp_jamp > 0 ? ProcessTables::nb_tmp_jamp : 1];

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes
    // === (for one event page in C++, or for two in mixed mode)

    // START LOOP ON IPARITY
    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      const int ievt0 = ievt00 + iParity * neppV;

#include "JampsPreamble.h" // a fragment of the iParity loop body, not a header
      fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );

      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i = 0; i < njampso; i++ ) { jamp_sv[i] = cxzero_sv<cxtype_amp_sv>(); }

      // Numerators for the current event page (C++); denominators are no longer
      // accumulated here: they are derived as the sum of numerators later.
      fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
#include "EvaluateDiagrams.inc"
#include "ColorFlows.inc" // defines jampflow_sv[ncolor_flow], which is not jamp_sv on the DDM basis

      // *** COLOR CHOICE BELOW ***
      // Store the leading color flows for choice of color
      if( jamp2_sv ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor_flow; icol++ )
          jamp2_sv[ncolor_flow * iParity + icol] += cxabs2( jampflow_sv[icol] ); // may underflow #831
      }

      // *** PREPARE OUTPUT JAMPS ***
      // In C++, copy the local jamp to the output array passed as function argument
      for( int icol = 0; icol < njampso; icol++ )
        allJamp_sv[iParity * njampso + icol] = jamp_sv[icol];
    }
    // END LOOP ON IPARITY

    mgDebug( 1, __FUNCTION__ );
  }

#undef NUM_ATOMIC_ADD
#undef MG_HR_PROBE_AMP

  //--------------------------------------------------------------------------

  // --hel_recycling (CPU only; on GPU blockIdx.y already parallelises over the
  // helicities). A wavefunction only depends on the helicities of the external
  // legs it descends from, so the helicity rows that agree on those legs can
  // share it. calculate_jamps rebuilds everything once per row; the recycled
  // build, generated by madevent's hel_recycle into HelRecyclingCalls.inc, makes
  // each wavefunction once per COPY, skips the amplitudes the warm-up probe found
  // vanishing at a row, and folds the others into per-row tables that load_jamps
  // then hands back to the helicity loop one row at a time. Without the generated
  // files (no --hel_recycling, or a warm-up that could not run) this is not
  // compiled and the helicity loop calls calculate_jamps as before.
#if __has_include( "HelRecycling.inc" ) and !defined MG_HR_PROBE
#define MGONGPU_HEL_RECYCLING 1
#include "HelRecycling.inc" // nHrRows, cHrRow[ncomb], hrNwf (and the recycled chunks, if any)

  // The rows that were built are the rows that contributed AT THE PARAMETERS THE
  // WARM-UP RAN AT. getGoodHel still runs at run time, and may ask for a row that
  // is not in the table: a different param_card, a C-parity verdict that differs
  // from the warm-up's (only representatives are built), MGONGPU_NOCSYM. That row
  // is computed the ordinary way (load_jamps), exact but not recycled; say so once.
  inline void
  hr_unbaked_row( int ihel )
  {
    static bool warned = false;
    if( warned ) return;
    warned = true;
    printf( "INFO: helicity row %d is not in the --hel_recycling table (measured at\n"
            "INFO: other parameters): it is computed without recycling.\n",
            ihel );
  }

  // Build every helicity row of the table at once, for one event page (two in
  // mixed precision): jampAll_sv[nHrRows][nParity][njampso] and, if
  // storeChannelWeights, numAll_sv[nHrRows][nParity][ndiagrams].
  void
  calculate_all_jamps( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: indices of the flavor combinations
                       cxtype_amp_sv* jampAll_sv,        // output: jamps of every row
                       fptype_amp_sv* numAll_sv,         // output: |amp|^2 by diagram of every row
                       bool storeChannelWeights,
                       const int ievt00 )                // input: first event number in current C++ event page
  {
    using M_ACCESS = HostAccessMomenta;
    using W_ACCESS = HostAccessWavefunctions;
    using A_ACCESS = HostAccessAmplitudes;
    using CD_ACCESS = HostAccessCouplings;
    using CI_ACCESS = HostAccessCouplingsFixed;
    using CDPF_ACCESS = CD_ACCESS; // the running flavor couplings gathered into dpf_value (as in calculate_jamps)
    using F_ACCESS = HostAccessIflavorVec;
    mgDebug( 0, __FUNCTION__ );
    // one wavefunction slot per COPY, which is more than the nwf of one row and
    // far fewer than one row's worth per row
    fptype_momenta_sv pvec_sv[hrNwf][np4];
    cxtype_amp_sv w_sv[hrNwf][nw6];
    cxtype_amp_sv amp_sv[1]; // every amplitude goes through this one scratch, then into the tables
    ALOHAOBJ aloha_obj[hrNwf];
    for( int iwf = 0; iwf < hrNwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype_amp* amp_fp = reinterpret_cast<fptype_amp*>( amp_sv );
    // the vertex with one leg left off, shared by every row that differs only in
    // that leg (the P1N amplitude split)
    fptype_momenta_sv p1n_pvec_sv[np4];
    cxtype_amp_sv p1n_w_sv[nw6];
    ALOHAOBJ _p1n = ALOHAOBJ{ p1n_pvec_sv, p1n_w_sv };
    (void)_p1n;
    (void)amp_fp;
    for( int i = 0; i < nHrRows * nParity * njampso; i++ ) jampAll_sv[i] = cxzero_sv<cxtype_amp_sv>();
    if( storeChannelWeights )
      for( int i = 0; i < nHrRows * nParity * ndiagrams; i++ ) numAll_sv[i] = fptype_amp_sv{ 0 };
    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      const int ievt0 = ievt00 + iParity * neppV;
#include "JampsPreamble.h" // a fragment of the iParity loop body, not a header
#include "HelRecyclingCalls.inc"
    }
    mgDebug( 1, __FUNCTION__ );
  }

  // Hand the helicity loop the row it asked for, out of the tables above, with the
  // bookkeeping calculate_jamps does for its own row: the multichannel numerators
  // and the color flows for the choice of color. A row the table lacks is
  // computed by calculate_jamps instead (see hr_unbaked_row).
  void
  load_jamps( int ihel,
              const cxtype_amp_sv* jampAll_sv,     // input: jamps of every row
              const fptype_amp_sv* numAll_sv,      // input: |amp|^2 by diagram of every row
              const fptype_momenta* allmomenta,    // input: momenta[nevt*npar*4] (for a row not in the table)
              const fptype* allcouplings,          // input: couplings[nevt*ndcoup*2] (idem)
              const unsigned int* iflavorVec,      // input: indices of the flavor combinations (idem)
              cxtype_amp_sv* allJamp_sv,           // output: jamp_sv[nParity*njampso] for this helicity
              bool storeChannelWeights,
              fptype_amp* allNumerators,           // input/output: multichannel numerators[nevt], add helicity ihel
              fptype_amp* allDenominators,         // input/output: multichannel denominators[nevt] (idem)
              fptype_amp_sv* jamp2_sv,             // output: jamp2[nParity][ncolor_flow][neppV] for color choice (nullptr if disabled)
              const int ievt00 )                   // input: first event number in current C++ event page
  {
    using NUM_ACCESS = HostAccessNumerators;
    const int row = cHrRow[ihel];
    if( row < 0 )
    {
      hr_unbaked_row( ihel );
      calculate_jamps( cHel[ihel], allmomenta, allcouplings, iflavorVec, allJamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
      return;
    }
    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      const int ievt0 = ievt00 + iParity * neppV;
      const size_t slot = (size_t)row * nParity + iParity;
      cxtype_amp_sv jamp_sv[njampso];
      for( int icol = 0; icol < njampso; icol++ )
        jamp_sv[icol] = jampAll_sv[slot * njampso + icol];
      if( storeChannelWeights )
      {
        fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
        fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
        for( int idiag = 0; idiag < ndiagrams; idiag++ )
          numerators_sv[idiag] += numAll_sv[slot * ndiagrams + idiag];
      }
#include "ColorFlows.inc" // defines jampflow_sv[ncolor_flow], which is not jamp_sv on the DDM basis
      if( jamp2_sv ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor_flow; icol++ )
          jamp2_sv[ncolor_flow * iParity + icol] += cxabs2( jampflow_sv[icol] ); // may underflow #831
      }
      for( int icol = 0; icol < njampso; icol++ )
        allJamp_sv[iParity * njampso + icol] = jamp_sv[icol];
    }
  }
#endif // HelRecycling.inc

  //--------------------------------------------------------------------------

  // The good-helicity filter of madevent (matrix_madevent_group_v4.inc): a
  // helicity contributes if, at one of the sampled events, its |M|^2 is more than
  // limhel/ncomb of the sum over all helicities there (limhel is the hidden
  // run_card parameter of madevent, 1e-8 by default). Comparing to an exact 0 kept
  // every helicity whose amplitudes only cancel to rounding -- the all-plus and
  // one-minus helicities of a purely gluonic process, 12 of the 32 of g g > g g g --
  // so that every evaluation then computed them for nothing.
  //
  // Not for a crossing that moves a massive leg into a slot of another mass
  // (xonshell_tab 0). Every crossing is scanned on the points of the base
  // process (choose_good_helicities), which are off shell for that one, and a
  // row it needs can come out at rounding level there: a longitudinal Z fed a
  // lightlike momentum is proportional to it. A relative test would drop that
  // row for every event, so there a helicity is good when it is not exactly 0.
#ifndef MG_HR_PROBE
  constexpr double limhel = 1e-8;
#else
  // The --hel_recycling warm-up bakes the rows it finds into the recycled table.
  // A row the run time asks for and the table lacks is still computed, without
  // recycling (load_jamps), but at a cost on every event. So the probe keeps
  // every row above a much lower threshold -- still far above a rounding
  // residual (~1e-30 of the sum) -- on more events (goodHelScanEvents).
  constexpr double limhel = 1e-14;
#endif

  //--------------------------------------------------------------------------

  void
  sigmaKin_getGoodHel( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,        // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,      // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array
                       const int nevt )                  // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    // Allocate arrays at build time to contain at least 16 events (or at least neppV events if neppV>16, e.g. in future VPUs)
    constexpr int maxtry0 = goodHelScanEvents; // 16, but at least neppV
    // Loop over only nevt events if nevt is < 16 (note that nevt is always >= neppV)
    assert( nevt >= neppV );
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    // HELICITY LOOP: CALCULATE WAVEFUNCTIONS
    const int npagV = maxtry / neppV;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT /* clang-format off */
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 ); // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV; // loop on one SIMD page (neppV events) at a time
#endif /* clang-format on */
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): a helicity
    // that vanishes for the sampled flavor may be non-zero for another merged
    // flavor and must not be dropped. Sample every flavor combination on the
    // same momenta and OR the result, so cGoodHel becomes the union over all
    // flavors (extra helicities simply contribute 0 for a given flavor at run
    // time, exactly as in the scalar standalone_cpp per-flavor good-hel filter).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hgFlavorVec[maxtry0] = {}; // forced single-flavor index buffer
    // |M|^2 of every helicity at every sampled event (of the current extended
    // flavor id), compared at the end to the sum over helicities at the same event
    std::vector<fptype> helMEs( (size_t)ncomb * maxtry0, 0. );
    // C-parity partner of every helicity row, and the per-hel |M|^2 of the
    // current scan page for the C-parity test
    fptype me_scan[ncomb][neppV];
    cCsymScanned = false;
    cCsymBad = false;
    for( int h = 0; h < ncomb; h++ )
    {
      cFlip[h] = h;
      for( int j = 0; j < ncomb; j++ )
      {
        bool same = true;
        for( int k = 0; k < npar; k++ ) if( cHel[j][k] != -cHel[h][k] ) same = false;
        if( same ) { cFlip[h] = j; break; }
      }
    }
    // Crossing: sample every extended flavor id, i.e. every row of the
    // crossing table -- the crossings this ME records (every applicable one
    // only with --crossing_table=all) -- times every flavor, and OR them into
    // the one list: each crossing is evaluated in its own frame, where a
    // massive leg can need a row the base does not. Each row costs a full ncomb-helicity
    // calculate_jamps scan, which is why the table carries no merely
    // applicable crossing by default (scanning those was a 46x one-off
    // startup cost on g g > t t~ g g g: 48 applicable, 0 recorded).
    constexpr int nscan = use_crossing ? ProcessTables::ncross * nmaxflavor : nmaxflavor;
    for( int iflav = 0; iflav < nscan; ++iflav )
    {
    const int xcross = iflav / nmaxflavor; // always 0 without crossing
    if constexpr( use_crossing )
    {
      if( spincol_cross( xcross ) == 0 ) continue;
    }
    for( int i = 0; i < maxtry0; ++i ) hgFlavorVec[i] = (unsigned int)iflav;
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT /* clang-format off */
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif /* clang-format on */
      for( int ihel = 0; ihel < ncomb; ihel++ )
      {
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allMEs[ievt] = 0;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          allMEs[ievt2] = 0;
#endif
        }
        constexpr fptype_amp_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
#if defined MGONGPU_CPPSIMD and !( defined MGONGPU_FPTYPE_AMP_FLOAT ) and defined MGONGPU_FPTYPE2_FLOAT
        cxtype_amp_sv jamp_sv[2 * njampso] = {}; // all zeros
#else
        cxtype_amp_sv jamp_sv[njampso] = {}; // all zeros
#endif
        calculate_jamps( cHel[ihel], allmomenta, allcouplings, hgFlavorVec, jamp_sv, false, allNumerators, allDenominators, jamp2_sv, ievt00 );
#ifdef MG_HR_PROBE
        // file this evaluation's amplitudes under its helicity row: here, as
        // calculate_jamps only sees the row it was handed, not its number
        hr_probe_close( ihel );
#endif
        color_sum_cpu( allMEs, jamp_sv, ievt00 );
        for( int ie = 0; ie < neppV; ++ie ) me_scan[ihel][ie] = allMEs[ievt00 + ie];
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          helMEs[(size_t)ihel * maxtry0 + ievt] = allMEs[ievt];
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          helMEs[(size_t)ihel * maxtry0 + ievt2] = allMEs[ievt2];
#endif
        }
      }
      // C-parity test of this (flavor, page). The largest |M|^2 is the scale a
      // difference has to be significant against: a RELATIVE test alone
      // compares the roundoff noise of two numerically-zero rows against itself
      // and fails at random -- which latched "not C-symmetric" on manifestly
      // C-symmetric processes (the MHV-vanishing gluon configurations of
      // u u~ > g g sit at |M|^2 ~ 1e-30 out of ~10), silently disabling the
      // dedup. A row that far below the largest cannot bias the helicity sum
      // whichever way it is paired, while a genuine parity violation shows up
      // at the relative level. One verdict for every crossing.
      {
        fptype mmax = (fptype)0.;
        for( int h = 0; h < ncomb; h++ )
          for( int ie = 0; ie < neppV; ++ie )
          {
            const fptype v = me_scan[h][ie] < (fptype)0. ? -me_scan[h][ie] : me_scan[h][ie];
            if( v > mmax ) mmax = v;
          }
        for( int h = 0; h < ncomb; h++ )
        {
          if( cFlip[h] > h )
          {
            for( int ie = 0; ie < neppV; ++ie )
            {
              const fptype a = me_scan[h][ie];
              const fptype b = me_scan[cFlip[h]][ie];
              fptype d = a - b;
              if( d < (fptype)0. ) d = -d;
              const fptype aa = a < (fptype)0. ? -a : a;
              const fptype bb = b < (fptype)0. ? -b : b;
              if( d > (fptype)1e-6 * ( aa + bb ) && d > (fptype)1e-12 * mmax )
                cCsymBad = true;
            }
          }
        }
      }
      cCsymScanned = true; // a full ncomb-row comparison has been made
    }
    // a helicity is good for this extended flavor id if it is above limhel/ncomb
    // of the sum over helicities at one of the events -- not exactly 0 where the
    // points are off shell for the crossing (see limhel) -- OR-ed over the ids:
    // one list for every crossing
    const double relhel = xonshell_tab[xcross] ? limhel : 0.;
    for( int ievt = 0; ievt < maxtry; ++ievt )
    {
      fptype sumhel = 0;
      for( int ihel = 0; ihel < ncomb; ihel++ ) sumhel += std::abs( helMEs[(size_t)ihel * maxtry0 + ievt] );
      for( int ihel = 0; ihel < ncomb; ihel++ )
        if( std::abs( helMEs[(size_t)ihel * maxtry0 + ievt] ) > sumhel * relhel / ncomb ) isGoodHel[ihel] = true;
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    {
      // All-or-nothing C-parity verdict. cCsymScanned is the load-bearing term:
      // if the validating scan never ran (cached good helicities, an API caller
      // reaching setGoodHel on its own) the flag must default to OFF, never to
      // ON -- trusting an un-run scan is how this dedup was once silently
      // enabled on a parity-violating process.
      cCsymOk = cCsymScanned && !cCsymBad;
      for( int h = 0; h < ncomb; h++ )
        if( isGoodHel[h] && ( cFlip[h] == h || !isGoodHel[cFlip[h]] ) ) cCsymOk = false;
#ifdef MGONGPU_NOCSYM
      cCsymOk = false; // ablation knob: force the full helicity sum
#endif
      if( cCsymOk )
      {
        // Keep only the lower-index representative of every C-parity pair.
        // sigmaKin counts each one twice and csym_selected_row hands back the
        // representative or its mirror at equal rate, so this is exact rather
        // than approximate: the dropped rows have an identical |M|^2.
        int n = 0;
        for( int g = 0; g < nGoodHel; g++ )
          if( goodHel[g] < cFlip[goodHel[g]] ) { cGoodHel[n] = goodHel[g]; n++; }
        for( int h = n; h < ncomb; h++ ) cGoodHel[h] = 0;
        cNGoodHel = n;
        nGoodHel = n;
      }
    }
#ifdef MG_HR_PROBE
    // --hel_recycling warm-up: the good helicities and, for each, the amplitudes
    // the probe never saw alive, in the protocol the madevent warm-up prints.
    // The rows the helicity loop reads: after the C-parity reduction above,
    // only the representatives (cGoodHel), as madevent bakes them; a row the
    // run time asks for and the table lacks is computed without recycling.
    if( getenv( "MG_DUMP_GOODHEL" ) != nullptr )
    {
      for( int ighel = 0; ighel < cNGoodHel; ighel++ )
        printf( "Matrix Element/Good Helicity: %d %d\n", 1, cGoodHel[ighel] + 1 );
      for( int ighel = 0; ighel < cNGoodHel; ighel++ )
      {
        const int ihel = cGoodHel[ighel];
        if( hrProbeSeen.empty() ) break;
        for( size_t namp = 0; namp < hrProbeSeen[ihel].size(); namp++ )
          if( hrProbeSeen[ihel][namp] and !hrProbeAlive[ihel][namp] )
            printf( "HEL/ZEROAMP: %d %d %d\n", 1, ihel + 1, (int)namp );
      }
    }
#endif
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

  int
  sigmaKin_getNGoodHel()
  {
    return cNGoodHel;
  }

  //--------------------------------------------------------------------------

  // Decay-aware identical-particle (broken-)symmetry factor, shared with the
  // Fortran / standalone_cpp exporters (_get_broken_symmetry_data). Two
  // entries contribute to the over-counting factor only when they have the
  // same top-level PID AND the same full decay/flavour block, so e.g. two Z
  // bosons decaying to different families are correctly distinguished.
  int
  broken_symmetry_factor( const int iflavor )
  {
    int pid_work[broken_sym_nentries];
    for( int i = 0; i < broken_sym_nentries; i++ )
      pid_work[i] = broken_sym_pid_list[i];

    int total_factor = 1;
    for( int icomp = 0; icomp < broken_sym_ncomponents; icomp++ )
    {
      int old_factor = broken_sym_component_old_factors[icomp];
      if( broken_sym_component_old_factors[icomp] > 1 )
      {
        for( int i = broken_sym_component_starts[icomp] - 1; i < broken_sym_component_ends[icomp]; i++ )
        {
          if( pid_work[i] == 0 )
            continue;
          int n_tot = 1;
          for( int j = i + 1; j < broken_sym_component_ends[icomp]; j++ )
          {
            if( pid_work[i] != pid_work[j] )
              continue;
            bool same_block = ( broken_sym_block_lengths[i] == broken_sym_block_lengths[j] );
            for( int k = 0; same_block && k < broken_sym_block_lengths[i]; k++ )
            {
              if( cFlavors[iflavor][broken_sym_block_starts[i] - 1 + k] != cFlavors[iflavor][broken_sym_block_starts[j] - 1 + k] )
                same_block = false;
            }
            if( same_block )
            {
              pid_work[j] = 0;
              n_tot = n_tot + 1;
              old_factor = old_factor / n_tot;
            }
          }
        }
      }
      total_factor = total_factor * old_factor;
    }
    return total_factor;
  }

  //--------------------------------------------------------------------------
  // Helicity choice when a helicity can contribute negatively: a squared split-order
  // selection that keeps an interference (nampso > 1). As madevent does
  // (matrix_madevent_group_v4.inc), helicity i is chosen with probability
  // |T_i| / sum_j |T_j| and the event gets sign(T_i) * sum_j |T_j|, whose average over
  // the choice is sum_j T_j: the chosen helicity then reproduces the signed per-helicity
  // |M|^2. Choosing on the signed running sum, as the positive case does, would not:
  // with mixed signs it is not a distribution, and a negative total turns it upside down.
  // ighelSum[ighel] is the running sum of T_j over the good helicities j <= ighel, as the
  // helicity loop leaves it. Returns the |M|^2 of the event (before the helicity/color
  // average and the channel weight) and sets selhel; a vanishing |M|^2 returns 0 and
  // leaves selhel alone, as the positive case does.
  static inline fptype
  select_helicity_signed( const fptype* ighelSum,       // input: running sums over the good helicities
                          const int ngood,              // input: number of good helicities in ighelSum
                          const unsigned int flavor_id, // input: flavor (crossing) index of the event
                          const fptype rndhel,          // input: random number for the helicity choice
                          int& selhel )                 // output: chosen helicity (Fortran range [1,ncomb])
  {
    fptype absTotal = 0;
    for( int ighel = 0; ighel < ngood; ighel++ )
      absTotal += std::abs( ighelSum[ighel] - ( ighel > 0 ? ighelSum[ighel - 1] : 0 ) );
    if( absTotal == 0 ) return 0;
    fptype absSum = 0, absLo = 0, absHi = 0;
    int ighelLast = -1; // last non-vanishing helicity, in case rndhel is not below 1
    for( int ighel = 0; ighel < ngood; ighel++ )
    {
      const fptype t = ighelSum[ighel] - ( ighel > 0 ? ighelSum[ighel - 1] : 0 );
      if( t == 0 ) continue;
      ighelLast = ighel;
      absLo = absSum;
      absSum += std::abs( t );
      absHi = absSum;
      if( rndhel < absSum / absTotal ) // the last ratio is exactly 1 (same terms, same order)
        break;
    }
    const fptype tLast = ighelSum[ighelLast] - ( ighelLast > 0 ? ighelSum[ighelLast - 1] : 0 );
    // The same row choice as the positive case (selected_helicity): the lane's own
    // crossed helicity code with crossing, and the C-parity partner of a
    // de-duplicated representative picked within its |T| bin [absLo, absHi)
    selhel = selected_helicity( ighelLast, flavor_id, rndhel * absTotal, absLo, absHi ); // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
    return std::copysign( absTotal, tLast );
  }

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour

  void
  sigmaKin( const fptype_momenta* allmomenta,  // input: momenta[nevt*npar*4]
            const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,    // input: index of the flavor combination
            const fptype* allrndhel,           // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,           // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899)
            const fptype* allrnddiagram,       // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                    // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                    // output: helicity selection[nevt]
            int* allselcol,                    // output: helicity selection[nevt]
            fptype_amp* allNumerators,         // tmp: multichannel numerators[nevt], running_sum_over_helicities
            fptype_amp* allDenominators,       // tmp: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,    // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,             // if true, multiply channel weight to ME output
            const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            const bool sampleSignedHelicity )  // input: choose the helicity on |T_i| and return sign(T_i)*sum|T| (split orders only)
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      static_assert( nproc == 1 || nproc == 2, "Assume nprocesses == 1 or 2" );
      static_assert( proc_id == 1, "Assume process_id == 1" );
    }

    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;   // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators; // non-trivial access: buffer includes all events

    // === PART 0 - INITIALISATION (before calculate_jamps) ===
    // Reset the "matrix elements" - running sums of |M|^2 over helicities for the given event
    const int npagV = nevt / neppV;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = fptype_sv{ 0 };
      fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
      fptype_amp* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
      fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_amp_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      for( int i = 0; i < ndiagrams; ++i )
      {
        numerators_sv[i] = fptype_amp_sv{ 0 };
      }
      denominators_sv = fptype_amp_sv{ 0 };
    }

    // === PART 1 - HELICITY LOOP: CALCULATE WAVEFUNCTIONS ===
    // (using precomputed good helicities)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    // Mixed fptypes #537: float for color algebra and double elsewhere
    // Delay color algebra and ME updates (only on even pages)
    assert( npagV % 2 == 0 );     // SANITY CHECK for mixed fptypes: two neppV-pages are merged to one 2*neppV-page
    const int npagV2 = npagV / 2; // loop on two SIMD pages (neppV events) at a time
#else
    const int npagV2 = npagV; // loop on one SIMD page (neppV events) at a time
#endif
    // The good-helicity loop bound: the (possibly C-parity halved) good
    // helicity count, the same for every lane (crossed or not)
    const int nGoodLoop = cNGoodHel;
#ifdef _OPENMP
    // OMP multithreading #575 (NB: tested only with gcc11 so far)
#define _OMPLIST0 allcouplings, allMEs, allmomenta, allrndcol, allrndhel, allselcol, allselhel, cGoodHel, nGoodLoop, npagV2, sampleSignedHelicity
#define _OMPLIST1 , allDenominators, allNumerators, allChannelIds, allDiagramIdsOut, allrnddiagram, iflavorVec, mgOnGpu::icolamp, mgOnGpu::channel2iconfig
#pragma omp parallel for default( none ) shared( _OMPLIST0 _OMPLIST1 )
#undef _OMPLIST0
#undef _OMPLIST1
#endif // _OPENMP
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      const int ievt00 = ipagV2 * neppV * 2; // loop on two SIMD pages (neppV events) at a time
#else
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
#endif
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      fptype_amp_sv jamp2_sv[nParity * ncolor_flow] = {};
      fptype_sv MEs_ighel[ncomb] = {}; // sum of MEs for all good helicities up to ighel (first - and/or only - neppV page)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv MEs_ighel2[ncomb] = {}; // sum of MEs for all good helicities up to ighel (second neppV page)
#endif

      // Per-lane C-parity weight: 1 where this lane's helicity sum was
      // de-duplicated (every good helicity then stands for two rows), 0
      // otherwise. Materialised once per page and consumed by BOTH the scalar
      // helicity loop and the BLAS batch: letting the two paths drift apart is
      // what once made the batch return exactly half of |M|^2.
      fptype_sv csymExtra{};
      for( int ie = 0; ie < neppV; ie++ )
        reinterpret_cast<fptype*>( &csymExtra )[ie] = csym_lane_on( iflavorVec[ievt00 + ie] ) ? (fptype)1. : (fptype)0.;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      fptype_sv csymExtra2{};
      for( int ie = 0; ie < neppV; ie++ )
        reinterpret_cast<fptype*>( &csymExtra2 )[ie] = csym_lane_on( iflavorVec[ievt00 + neppV + ie] ) ? (fptype)1. : (fptype)0.;
#endif

#ifdef MGONGPU_HEL_RECYCLING
      // --hel_recycling: build every helicity row of this event page at once, then
      // read the rows back inside the helicity loop (see calculate_all_jamps)
      static thread_local std::vector<cxtype_amp_sv> hrJamp_sv( (size_t)nHrRows * nParity * njampso );
      static thread_local std::vector<fptype_amp_sv> hrNum_sv( (size_t)nHrRows * nParity * ndiagrams );
      calculate_all_jamps( allmomenta, allcouplings, iflavorVec, hrJamp_sv.data(), hrNum_sv.data(),
                           allChannelIds != nullptr || allrnddiagram != nullptr, ievt00 );
#endif

      // The color matrix does not depend on the helicity, so for a large-enough color matrix
      // (ColorMatrixData::shouldUseBlas) it pays to keep the jamps of every good helicity and
      // hand them all to BLAS in one call after the loop instead of the per-helicity color sum.
#ifdef MGONGPU_CPP_HAS_BLAS
      // The BLAS buffers below hold one ncolor jamp vector per helicity (cpp_blas_wanted is false once split)
      static_assert( !ColorMatrixData::shouldUseBlas || nampso == 1, "the BLAS color sum does not pair split amplitude orders" );
      if( ColorMatrixData::shouldUseBlas )
      {
        static thread_local std::vector<cxtype_sv> ghelJamp_sv( (size_t)ncomb * nParity * ncolor );
        for( int ighel = 0; ighel < nGoodLoop; ighel++ )
        {
          const int ihel = cGoodHel[ighel];
          cxtype_sv* jamp_sv = ghelJamp_sv.data() + (size_t)ighel * nParity * ncolor;
          for( int i = 0; i < nParity * ncolor; i++ ) jamp_sv[i] = cxzero_sv<cxtype_sv>(); // calculate_jamps accumulates into jamp_sv
          // **NB! in "mixed" precision, using SIMD, calculate_jamps computes MEs for TWO neppV pages with a single channelId! #924
          bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
#ifdef MGONGPU_HEL_RECYCLING
          load_jamps( ihel, hrJamp_sv.data(), hrNum_sv.data(), allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
#else
          calculate_jamps( cHel[ihel], allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
#endif
        }
        // The C-parity weight is NOT optional here: with the de-duplication on,
        // every helicity this loop just computed stands for two. The scalar loop
        // below adds the second copy per helicity; the batch has no per-helicity
        // step to hang that on, so the same per-lane 0/1 vector goes into
        // color_sum_cpu_blas and is applied while it builds the running sums.
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        color_sum_cpu_blas( allMEs, MEs_ighel, MEs_ighel2, ghelJamp_sv.data(), nGoodLoop, ievt00,
                            reinterpret_cast<const fptype*>( &csymExtra ), reinterpret_cast<const fptype*>( &csymExtra2 ) );
#else
        color_sum_cpu_blas( allMEs, MEs_ighel, nullptr, ghelJamp_sv.data(), nGoodLoop, ievt00,
                            reinterpret_cast<const fptype*>( &csymExtra ), nullptr );
#endif
      }
      else
#endif // MGONGPU_CPP_HAS_BLAS
      {
        for( int ighel = 0; ighel < nGoodLoop; ighel++ )
        {
          const int ihel = cGoodHel[ighel];
          // Snapshot the running |M|^2 sum before this helicity's contribution is
          // added, so the C-parity step below can add the very same contribution again
          const fptype_sv me1before = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const fptype_sv me2before = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 + neppV ) );
#endif
          cxtype_amp_sv jamp_sv[nParity * njampso] = {}; // fixed nasty bug (omitting 'nParity' caused memory corruptions after calling calculate_jamps)
          // **NB! in "mixed" precision, using SIMD, calculate_jamps computes MEs for TWO neppV pages with a single channelId! #924
          bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
#ifdef MGONGPU_HEL_RECYCLING
          load_jamps( ihel, hrJamp_sv.data(), hrNum_sv.data(), allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
#else
          calculate_jamps( cHel[ihel], allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
#endif
          color_sum_cpu( allMEs, jamp_sv, ievt00 );
          MEs_ighel[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          MEs_ighel2[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 + neppV ) );
#endif
          // C-parity weight 2 where the lane's sum was de-duplicated: the mirror
          // row this representative stands for has an identical |M|^2. MEs_ighel
          // is updated too -- it is the running CDF the helicity choice samples.
          {
            fptype_sv& me1 = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
            me1 = me1 + ( MEs_ighel[ighel] - me1before ) * csymExtra;
            MEs_ighel[ighel] = me1;
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
            fptype_sv& me2 = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 + neppV ) );
            me2 = me2 + ( MEs_ighel2[ighel] - me2before ) * csymExtra2;
            MEs_ighel2[ighel] = me2;
#endif
          }
        }
      }

      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV )
      {
        const int ievt = ievt00 + ieppV;
        if constexpr( nampso > 1 )
        {
          if( sampleSignedHelicity )
          {
            fptype ighelSum[ncomb];
            for( int ighel = 0; ighel < nGoodLoop; ighel++ )
#if defined MGONGPU_CPPSIMD
              ighelSum[ighel] = MEs_ighel[ighel][ieppV];
#else
              ighelSum[ighel] = MEs_ighel[ighel];
#endif
            *E_ACCESS::ieventAccessRecord( allMEs, ievt ) = select_helicity_signed( ighelSum, nGoodLoop, iflavorVec[ievt], allrndhel[ievt], allselhel[ievt] );
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
            const int ievt2 = ievt00 + ieppV + neppV;
            for( int ighel = 0; ighel < nGoodLoop; ighel++ )
              ighelSum[ighel] = MEs_ighel2[ighel][ieppV];
            *E_ACCESS::ieventAccessRecord( allMEs, ievt2 ) = select_helicity_signed( ighelSum, nGoodLoop, iflavorVec[ievt2], allrndhel[ievt2], allselhel[ievt2] );
#endif
            continue;
          }
        }
        for( int ighel = 0; ighel < nGoodLoop; ighel++ )
        {
#if defined MGONGPU_CPPSIMD
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel][ieppV] / MEs_ighel[nGoodLoop - 1][ieppV] );
#else
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel] / MEs_ighel[nGoodLoop - 1] );
#endif
          if( okhel )
          {
            // Unnormalised CDF bin [clo,chi) of the selected ighel, and the total
            // ctot the stored variate is normalised by (okhel tested rnd < hi/tot)
            fptype clo = (fptype)0;
#if defined MGONGPU_CPPSIMD
            const fptype ctot = MEs_ighel[nGoodLoop - 1][ieppV];
            const fptype chi = MEs_ighel[ighel][ieppV];
            if( ighel > 0 ) clo = MEs_ighel[ighel - 1][ieppV];
#else
            const fptype ctot = MEs_ighel[nGoodLoop - 1];
            const fptype chi = MEs_ighel[ighel];
            if( ighel > 0 ) clo = MEs_ighel[ighel - 1];
#endif
            const int ihelF = selected_helicity( ighel, iflavorVec[ievt], allrndhel[ievt] * ctot, clo, chi ); // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            break;
          }
        }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        const int ievt2 = ievt00 + ieppV + neppV;
        for( int ighel = 0; ighel < nGoodLoop; ighel++ )
        {
          if( allrndhel[ievt2] < ( MEs_ighel2[ighel][ieppV] / MEs_ighel2[nGoodLoop - 1][ieppV] ) )
          {
            fptype clo = (fptype)0;
            const fptype ctot = MEs_ighel2[nGoodLoop - 1][ieppV];
            const fptype chi = MEs_ighel2[ighel][ieppV];
            if( ighel > 0 ) clo = MEs_ighel2[ighel - 1][ieppV];
            const int ihelF = selected_helicity( ighel, iflavorVec[ievt2], allrndhel[ievt2] * ctot, clo, chi ); // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt2] = ihelF;
            break;
          }
        }
#endif
      }
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
      const int vecsize = 2 * neppV;
#else
      const int vecsize = neppV;
#endif
      unsigned int channelIdVec[vecsize];
      if( allChannelIds != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          channelIdVec[ieppV] = allChannelIds[ievt];
        }
      }

      // Event-by-event random choice of channel
      if( allrnddiagram != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          fptype numerator_sum = 0., normalization = 0.;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            normalization += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
          }
          channelIdVec[ieppV] = mgOnGpu::nchannels;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            numerator_sum += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
            if( allrnddiagram[ievt] < numerator_sum / normalization )
            {
              channelIdVec[ieppV] = ichan + 1;
              break;
            }
          }
          allDiagramIdsOut[ievt] = channelIdVec[ieppV];
        }
      }

      // Event-by-event random choice of color #402
      if( allChannelIds != nullptr || allrnddiagram != nullptr ) // no event-by-event choice of color if channelId == 0 (fix FPE #783)
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          unsigned int channelId = channelIdVec[ieppV];
          if( channelId > mgOnGpu::nchannels )
          {
            printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which is greater than nchannels=%d\n", channelId, mgOnGpu::nchannels );
            assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
          }
          // NB (see #877): in the array channel2iconfig, the input index uses C indexing (channelId -1), the output index uses F indexing (iconfig)
          const int iconfig = mgOnGpu::channel2iconfig[channelId - 1]; // map N_diagrams to N_config <= N_diagrams configs (fix LHE color mismatch #856: see also #826, #852, #853)
          if( iconfig <= 0 )
          {
            printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which has no associated SDE iconfig\n", channelId );
            assert( iconfig > 0 ); // SANITY CHECK #917
          }
          else if( iconfig > (int)mgOnGpu::nconfigSDE )
          {
            printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d (invalid SDE iconfig=%d\n > nconfig=%d)", channelId, iconfig, mgOnGpu::nconfigSDE );
            assert( iconfig <= (int)mgOnGpu::nconfigSDE ); // SANITY CHECK #917
          }
          fptype_amp targetamp[ncolor_flow] = { 0 };
          // NB (see #877): explicitly use 'icolC' rather than 'icol' to indicate that icolC uses C indexing in [0, N_colors-1]
          for( int icolC = 0; icolC < ncolor_flow; icolC++ )
          {
            if( icolC == 0 )
              targetamp[icolC] = 0;
            else
              targetamp[icolC] = targetamp[icolC - 1];
#ifdef MGONGPU_CPPSIMD
            if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] +=
              jamp2_sv[icolC + ncolor_flow * ( ieppV / neppV )][ieppV % neppV];
#else
            if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] +=
              jamp2_sv[icolC + ncolor_flow * ( ieppV / neppV )];
#endif
          }
          const int ievt = ievt00 + ieppV;
          for( int icolC = 0; icolC < ncolor_flow; icolC++ )
          {
            if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor_flow - 1] ) )
            {
              allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
              break;
            }
          }
        }
      }
      else
      {
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
          const int ievt2 = ievt00 + ieppV + neppV;
          allselcol[ievt2] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
#endif
        }
      }
    }
    // *** END OF PART 1 - C++ (loop on event pages)

    // PART 2 - FINALISATION (after calculate_jamps)
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    const bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      if constexpr( !use_crossing )
        MEs_sv = MEs_sv * static_cast<fptype>( broken_symmetry_factor( iflavorVec[ievt0] ) ) / static_cast<fptype>( helcolDenominators[0] );
      else
      {
        // Per-event crossing-aware denominator: the crossing may differ per
        // lane. cross==0 keeps the historical IDEN/BROKEN_SYM path; a genuine
        // crossing rebuilds it from the crossed initial-state spin*color times
        // the identical-final-state factor of the actual flavors. An invalid
        // crossing must ASSIGN 0 (not multiply), because its unphysical
        // momentum relabelling can make the lane's |M|^2 a NaN and nan*0 = nan.
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const unsigned int fid = iflavorVec[ievt0 + ieppV];
          const int dcr = (int)( fid / nmaxflavor );
          const int dfl = (int)( fid % nmaxflavor );
          fptype& me = reinterpret_cast<fptype*>( &MEs_sv )[ieppV];
          if( dcr == 0 )
            me *= (fptype)broken_symmetry_factor( dfl ) / helcolDenominators[0];
          else if( spincol_cross( dcr ) == 0 )
            me = (fptype)0.; // no such crossing-table row -> ME 0
          else
            me *= (fptype)1. / ( (fptype)spincol_cross( dcr ) * (fptype)ident_cross( dcr, dfl ) );
        }
      }
      if( storeChannelWeights ) // fix segfault #892 (not 'channelIds[0] != 0')
      {
        // The numerators have already been accumulated over all good helicities in place (running sum
        // over the helicity loop), so there is no helicity dimension to sum here. The denominator is
        // just the sum of all numerators for this event page: derive it once and store it for the
        // downstream consumers (e.g. the multichannel amp2 output).
        fptype_amp* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
        fptype_amp* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
        fptype_amp_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
        fptype_amp_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
        denominators_sv = fptype_amp_sv{ 0 };
        for( int idiag = 0; idiag < ndiagrams; ++idiag )
          denominators_sv += numerators_sv[idiag];
        if( mulChannelWeight && allChannelIds != nullptr )
        {
          const unsigned int channelId = getChannelId( allChannelIds, ievt0, false );
          // A vanishing denominator means every diagram's |amp|^2 is zero for this event; clamp to the
          // smallest normal value so the channel-weight division stays finite (0/tiny is 0) rather than
          // turning a matrix element that is simply zero into a nan (see color_sum.cc fix #435 note).
          const fptype_sv safe_den = denominators_sv + std::numeric_limits<fptype>::min();
          MEs_sv *= numerators_sv[channelId - 1] / safe_den;
        }
      }
    }
    mgDebugFinalise();
  }

  //--------------------------------------------------------------------------

} // end namespace
