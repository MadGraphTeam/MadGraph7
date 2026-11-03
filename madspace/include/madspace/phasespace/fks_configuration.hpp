#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "madspace/util.hpp"

namespace madspace {

/**
 * One FKS configuration of a Born: the real process and the FKS pair
 * @f$(i, j)@f$ whose sector it integrates [2, 3].
 *
 * The FKS classes of madspace take a list of configurations, in the order of
 * `fks_index`: @ref FKSConfigurationMapping, @ref FKSSectorFunction,
 * @ref FKSEikonals, @ref FKSSplittingKernels and @ref FKSPrefactors. Each uses
 * the fields it needs, and check_fks_configurations() checks them once.
 * madspace.fks builds the list from the metadata of the FKS exporter
 * (fks_configurations.json) and the masses of the Born mapping.
 *
 * Particles are numbered from 0, incoming first, in the order of the real
 * process. Its Born is the real process without @f$i@f$, with the mother
 * @f$i + j@f$ in the place of @f$j@f$.
 *
 * **References**
 * - [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
 *   https://arxiv.org/abs/2602.06895
 * - [2] S. Frixione, Z. Kunszt, A. Signer, "Three-jet cross sections to
 *   next-to-leading order", https://arxiv.org/abs/hep-ph/9512328
 * - [3] R. Frederix, S. Frixione, F. Maltoni, T. Stelzer, "Automation of
 *   next-to-leading order computations in QCD: the FKS subtraction",
 *   https://arxiv.org/abs/0908.4272
 */
struct FKSConfiguration {
    /// Position of the radiated parton i, a massless final-state particle.
    std::size_t i_fks = 0;
    /// Position of the emitter j, another particle of the real process.
    std::size_t j_fks = 0;
    /// PDG codes of the real process.
    std::vector<int> pdgs;
    /// Color representation of each particle (1, 3, -3, 8).
    std::vector<int> colors;
    /// Electric charge of each particle.
    std::vector<double> charges;
    /// Mass of each particle.
    std::vector<double> masses;
    /// For each particle k, the particles l it forms an FKS pair with
    /// (FKS_J_FROM_I_D). The pairs (k, l) of all of them partition the phase
    /// space of the real process into the sectors of its configurations.
    nested_vector2<std::size_t> fks_j_from_i;

    /// Number of particles of the real process, incoming included.
    std::size_t particle_count() const { return pdgs.size(); }
    /// Whether the emitter is incoming.
    bool initial_state() const { return j_fks < 2; }
    /// Position of the emitter in the Born.
    std::size_t born_j_fks() const { return j_fks < i_fks ? j_fks : j_fks - 1; }
};

/**
 * Check a list of FKS configurations: at least one, all with real processes of
 * the same number of particles and with every field of that size, i a
 * massless final-state particle, j another particle that forms an FKS pair
 * with it, and FKS pairs within the real process.
 *
 * @param configurations  The configurations.
 * @param user            Name of the caller, for the error messages.
 * @return                The number of particles of the real processes.
 */
std::size_t check_fks_configurations(
    const std::vector<FKSConfiguration>& configurations, const std::string& user
);

} // namespace madspace
