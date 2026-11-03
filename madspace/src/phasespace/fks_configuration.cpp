#include "madspace/phasespace/fks_configuration.hpp"

#include <algorithm>
#include <stdexcept>

using namespace madspace;

namespace {

constexpr std::size_t INCOMING_COUNT = 2;

// product of k! over the flavors of k particles
double identical_factor(std::vector<int> pdgs) {
    std::sort(pdgs.begin(), pdgs.end());
    double factor = 1.;
    std::size_t run = 1;
    for (std::size_t k = 1; k < pdgs.size(); ++k) {
        run = pdgs.at(k) == pdgs.at(k - 1) ? run + 1 : 1;
        factor *= static_cast<double>(run);
    }
    return factor;
}

} // namespace

std::vector<int> FKSConfiguration::born_pdgs() const {
    int pdg_i = pdgs.at(i_fks), pdg_j = pdgs.at(j_fks);
    // the gauge boson a pair of i and j of opposite flavors comes from
    int boson = colors.at(i_fks) != 1 ? 21 : 22;
    int mother;
    if (pdg_i == 21 || pdg_i == 22) {
        mother = pdg_j;
    } else if (!initial_state() && pdg_j == -pdg_i) {
        mother = boson;
    } else if (initial_state() && pdg_j == pdg_i) {
        mother = boson;
    } else if (initial_state() && (pdg_j == 21 || pdg_j == 22)) {
        mother = -pdg_i;
    } else {
        throw std::invalid_argument(
            "FKSConfiguration: i_fks and j_fks do not come from a splitting"
        );
    }
    std::vector<int> born;
    for (std::size_t k = 0; k < pdgs.size(); ++k) {
        if (k != i_fks) {
            born.push_back(k == j_fks ? mother : pdgs.at(k));
        }
    }
    return born;
}

double FKSConfiguration::symmetry_factor() const {
    auto final_count = [&](int pdg) {
        return static_cast<double>(
            std::count(pdgs.begin() + INCOMING_COUNT, pdgs.end(), pdg)
        );
    };
    int pdg_i = pdgs.at(i_fks), pdg_j = pdgs.at(j_fks);
    double f_j = 1.;
    if (!initial_state()) {
        f_j = final_count(pdg_j) - (pdg_i == pdg_j ? 1. : 0.);
    }
    return final_count(pdg_i) * f_j;
}

double FKSConfiguration::identical_particle_ratio() const {
    auto born = born_pdgs();
    return identical_factor({born.begin() + INCOMING_COUNT, born.end()}) /
        identical_factor({pdgs.begin() + INCOMING_COUNT, pdgs.end()});
}

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
