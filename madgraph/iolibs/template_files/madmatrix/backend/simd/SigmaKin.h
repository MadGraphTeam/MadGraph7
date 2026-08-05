// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin and everything it calls (calculate_jamps,
// good-helicity filtering, color/channel selection). Declared here so
// CPPProcess.cc's constructor/initProc (P1-generated) can call the setters
// that populate this file's storage, and so umami.cc/MatrixElementKernels.cc
// can call sigmaKin/computeDependentCouplings.

#ifndef SIGMAKIN_H
#define SIGMAKIN_H 1

#include "mgOnGpuConfig.h"
#include "mgOnGpuCxtypes.h" // for cxtype

namespace mg5amcCpu
{
  __global__ void
  computeDependentCouplings( const fptype* allgs,
                             fptype* allcouplings,
                             const int nevt );

  void
  sigmaKin_getGoodHel( const fptype* allmomenta,
                       const fptype* allcouplings,
                       const unsigned int* iflavorVec,
                       fptype* allMEs,
                       fptype* allNumerators,
                       fptype* allDenominators,
                       bool* isGoodHel,
                       const int nevt );

  int
  sigmaKin_setGoodHel( const bool* isGoodHel );

  void
  sigmaKin( const fptype* allmomenta,
            const fptype* allcouplings,
            const unsigned int* iflavorVec,
            const fptype* allrndhel,
            const fptype* allrndcol,
            const unsigned int* allChannelIds,
            const fptype* allrnddiagram,
            fptype* allMEs,
            int* allselhel,
            int* allselcol,
            fptype* allNumerators,
            fptype* allDenominators,
            unsigned int* allDiagramIdsOut,
            bool mulChannelWeight,
            const int nevt );

  // Setters: called once by CPPProcess (P1-generated) to populate this file's
  // otherwise-internal storage, since it can no longer be written directly
  // from a different translation unit.
  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors );
  void setIndependentParams( const fptype* tIPD );
  void setIndependentCouplings( const cxtype* tIPC );
  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value );
  void setBsmIndepParam( const double* values, int n );
}

#endif // SIGMAKIN_H
