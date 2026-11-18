// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin and everything it calls (calculate_jamps,
// good-helicity filtering, color/channel selection). Declared here so that
// umami.cc/MatrixElementKernels.cc/Density.cc can call them. They all take the
// parameters of the process they evaluate (ProcessParameters.h) as an argument.

#ifndef SIGMAKIN_H
#define SIGMAKIN_H 1

#include "mgOnGpuConfig.h"
#include "mgOnGpuCxtypes.h" // for cxtype
#include "mgOnGpuVectors.h" // for fptype_amp_sv

namespace madmatrix
{
  struct ProcessParameters;

  __global__ void
  computeDependentCouplings( const ProcessParameters& params, // input: parameters of the process
                             const fptype* allgs,  // input: Gs[nevt]
                             fptype* allcouplings, // output: couplings[nevt*ndcoup*2]
                             const int nevt );     // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  // The events sigmaKin_getGoodHel looks at, which umami_initialize makes for it
  // (choose_good_helicities): one constant, so that the two never disagree.
  // 16, but at least neppV (otherwise the npagV loop does not even start).
  // The --hel_recycling warm-up probe (scalar, MG_HR_PROBE) bakes what it finds
  // with no run-time fallback for a dropped amplitude, so it samples 64.
#ifndef MG_HR_PROBE
  constexpr int goodHelScanEvents = ( neppV > 16 ? neppV : 16 );
#else
  constexpr int goodHelScanEvents = 64;
#endif

  int // output: the number of events it looked at, the first ones
  sigmaKin_getGoodHel( const ProcessParameters& params,   // input: parameters of the process
                       const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: index of the flavor combination
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,         // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,       // output: multichannel denominators[nevt], running_sum_over_helicities
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array (C++ implementation)
                       const int nevt );                 // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( ProcessParameters& params, // output: the good helicities of the process
                       const bool* isGoodHel );  // input: isGoodHel[ncomb] - host array

  // A crossing-table row with no phase space at its own masses (a crossed
  // decay lighter than its products, see scan_points in umami.cc): set before
  // sigmaKin_getGoodHel, which then leaves it out of the good-helicity scan.
  void sigmaKin_setNoPhaseSpace( const int cross );

  // The C-parity verdict of the last sigmaKin_getGoodHel scan of this thread (true:
  // a pair differed), and the verdict the next sigmaKin_setGoodHel of this thread
  // uses instead of a scan of its own: an instance that chose its good helicities
  // over several scans (umami.cc, learn_good_helicities) ANDs their verdicts, and
  // one that is given a choice (madmatrix_set_helicity_choice) takes its verdict.
  bool sigmaKin_csymScanBad();
  void sigmaKin_setCsymVerdict( const bool bad );

  void
  sigmaKin( const ProcessParameters& params,   // input: parameters of the process
            const fptype_momenta* allmomenta,  // input: momenta[nevt*npar*4]
            const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,    // input: index of the flavor combination
            const fptype* allrndhel,           // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,           // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899)
            const fptype* allrnddiagram,       // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                    // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                    // output: helicity selection[nevt]
            int* allselcol,                    // output: helicity selection[nevt]
            fptype_amp* allNumerators,          // tmp: multichannel numerators[nevt], running_sum_over_helicities
            fptype_amp* allDenominators,        // tmp: multichannel denominators[nevt], running_sum_over_helicities
            unsigned int* allDiagramIdsOut,    // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,             // if true, multiply channel weight to ME output
            const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
            // input: when the |M|^2 keeps an interference (squared split orders, nampso > 1),
            // choose the helicity on |T_i| and return sign(T_i) * sum_j |T_j| as madevent does,
            // so that the chosen helicity is exact; false returns sum_j T_j (no effect otherwise)
            const bool sampleSignedHelicity = false,
            // input: the helicity index [0,ncomb) of each event, which is then evaluated for that
            // helicity alone (and returns its contribution to the helicity-summed |M|^2, see
            // umami_matrix_element); nullptr, the default, sums over the good helicities
            const int* allihel = nullptr );

  // What the spin-density matrix (Density.cc) needs on top of sigmaKin: the
  // jamps of one event page for any helicity combination hel[npar], the
  // helicity table cHel[ncomb][npar] (row major) the combinations are built
  // from, and the identical-particle factor of a flavor combination.
  void
  calculate_jamps( const ProcessParameters& params,  // input: parameters of the process
                   const short* hel,                 // input: helicity of each leg, hel[npar], of all events (if allihel == nullptr)
                   const int* allihel,               // input: helicity index [0,ncomb) of each event[nevt] (nullptr: hel for all events)
                   const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,   // input: indices of the flavor combinations
                   cxtype_amp_sv* allJamp_sv,        // output: jamp_sv[njampso] (float/double) or jamp_sv[2*njampso] (mixed)
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,        // input/output: multichannel numerators[nevt]
                   fptype_amp* allDenominators,      // input/output: multichannel denominators[nevt]
                   fptype_amp_sv* jamp2_sv,          // output: jamp2 for the color choice (nullptr if disabled)
                   const int ievt00 );               // input: first event number in the event page
  const short* getHelicities();
  int broken_symmetry_factor( const int iflavor );
  // With crossing (ProcessTables::use_crossing): the identical-final-state
  // factor of the process crossing-table row cross crosses into, for the
  // flavor iflavor (the crossed denominator, see sigmaKin)
  int ident_cross( int cross, int iflavor );
}

#endif // SIGMAKIN_H
