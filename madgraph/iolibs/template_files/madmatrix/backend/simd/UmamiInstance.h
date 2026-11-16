// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Nov 2026.
//
// What a UMAMI handle (umami.cc) points to. Everything the matrix element of an
// instance depends on is in here, and not shared with the other instances of the
// library: the parameters (its CPPProcess) and the good helicities. Declared in a
// header for the madmatrix extensions taking a handle (Density.cc).

#ifndef UMAMIINSTANCE_H
#define UMAMIINSTANCE_H 1

#include "CPPProcess.h"

#include <cstddef>
#include <memory>

namespace madmatrix
{
  struct UmamiInstance
  {
    // The model parameters, and the parameters the matrix element is evaluated with
    CPPProcess process;
    // The good helicities of process.parameters(), chosen on the first events the
    // instance evaluates (umami.cc, nlearn) or set from the cards
    // (madmatrix_choose_good_helicities, madmatrix_set_helicity_choice): how many of
    // them it has looked at, and the helicities that contribute to one of them
    std::size_t learned = 0;
    bool good_hel[ProcessData::ncomb] = {};
    // where RAMBO has no phase space (a decay below threshold), every helicity is
    // kept: the first events cannot choose (see umami.cc)
    bool keep_every_helicity = false;
    // the C-parity verdict of the scans that chose good_hel: a pair that differed at
    // one of them disables the C-parity dedup for the instance (see csymOk)
    bool csym_bad = false;
    // parameter interference_helicity_summed (run card [generation]
    // interference_helicity = "summed"): an interference |M|^2 stays the helicity
    // sum even when the caller draws the helicity (see umami_matrix_element)
    bool summed_helicity = false;
    // The state of the spin-density matrices of this instance (Density.cc), made at
    // the first madmatrix_density* call: opaque here, Density.o is not in every build
    std::shared_ptr<void> density;
  };
}

#endif // UMAMIINSTANCE_H
