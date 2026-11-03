#include "madspace/phasespace/fks_configuration.hpp"

#include <algorithm>
#include <stdexcept>

using namespace madspace;

std::size_t madspace::check_fks_configurations(
    const std::vector<FKSConfiguration>& configurations, const std::string& user
) {
    auto fail = [&](const std::string& message) {
        throw std::invalid_argument(user + ": " + message);
    };
    if (configurations.empty()) {
        fail("needs at least one FKS configuration");
    }
    std::size_t n = configurations.at(0).particle_count();
    for (const auto& c : configurations) {
        if (c.particle_count() != n) {
            fail(
                "the real processes of the configurations must have the same "
                "number of particles"
            );
        }
        if (c.colors.size() != n || c.charges.size() != n || c.masses.size() != n ||
            c.fks_j_from_i.size() != n) {
            fail(
                "the colors, charges, masses and FKS pairs of a configuration "
                "need one entry per particle of its real process"
            );
        }
        if (c.i_fks < 2 || c.i_fks >= n || c.j_fks >= n || c.i_fks == c.j_fks) {
            fail(
                "i_fks must be a final-state particle and j_fks another particle "
                "of the real process"
            );
        }
        if (c.masses.at(c.i_fks) != 0.) {
            fail("the radiated parton i must be massless");
        }
        for (const auto& partners : c.fks_j_from_i) {
            for (auto l : partners) {
                if (l >= n) {
                    fail("FKS pair out of range");
                }
            }
        }
        auto pairs_with = [&](std::size_t k, std::size_t l) {
            const auto& partners = c.fks_j_from_i.at(k);
            return std::find(partners.begin(), partners.end(), l) != partners.end();
        };
        if (!pairs_with(c.i_fks, c.j_fks) && !pairs_with(c.j_fks, c.i_fks)) {
            fail(
                "(i_fks, j_fks) of a configuration must be one of the FKS pairs "
                "of its real process"
            );
        }
    }
    return n;
}
