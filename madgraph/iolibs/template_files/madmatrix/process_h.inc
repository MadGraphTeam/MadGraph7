// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2025).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#ifndef MG5_%(process_file_name)s_H
#define MG5_%(process_file_name)s_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "Parameters.h"
#include "ProcessData.h"
#include "SigmaKin.h" // for the setters CPPProcess's constructor/initProc call
%(helamps_h)s

#include <vector>

//--------------------------------------------------------------------------

#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
%(process_class_definitions)s
  //--------------------------------------------------------------------------
  // sigmaKin, computeDependentCouplings and friends are backend-owned:
  // see backend/{cpu,simd,gpu}/SigmaKin.h for their declarations.
  //--------------------------------------------------------------------------
}

#endif // MG5_%(process_file_name)s_H
