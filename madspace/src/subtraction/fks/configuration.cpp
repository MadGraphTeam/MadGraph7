#include "madspace/subtraction/fks/configuration.hpp"

#include <algorithm>
#include <stdexcept>

using namespace madspace;

namespace {

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

std::vector<int> FKSConfiguration::born_pdgs(std::size_t splitting) const {
    if (splitting >= splitting_types.size()) {
        throw std::invalid_argument(
            "FKSConfiguration: the configuration has no such splitting type"
        );
    }
    int pdg_i = pdgs.at(i_fks), pdg_j = pdgs.at(j_fks);
    // the gauge boson a pair of i and j of opposite flavors comes from
    int boson = splitting_types.at(splitting) == "QED" ? 22 : 21;
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
            std::count(pdgs.begin() + incoming, pdgs.end(), pdg)
        );
    };
    int pdg_i = pdgs.at(i_fks), pdg_j = pdgs.at(j_fks);
    double f_j = 1.;
    if (!initial_state()) {
        f_j = final_count(pdg_j) - (pdg_i == pdg_j ? 1. : 0.);
    }
    return final_count(pdg_i) * f_j;
}

double FKSConfiguration::identical_particle_ratio(std::size_t splitting) const {
    auto born = born_pdgs(splitting);
    return identical_factor({born.begin() + incoming, born.end()}) /
        identical_factor({pdgs.begin() + incoming, pdgs.end()});
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
    std::size_t incoming = configurations.at(0).incoming;
    for (const auto& c : configurations) {
        if (c.incoming != incoming || (incoming != 1 && incoming != 2)) {
            fail(
                "the configurations need the same number of incoming particles, "
                "two or one for a decay"
            );
        }
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
        if (c.i_fks < incoming || c.i_fks >= n || c.j_fks >= n || c.i_fks == c.j_fks) {
            fail(
                "i_fks must be a final-state particle and j_fks another particle "
                "of the real process"
            );
        }
        if (incoming == 1 && c.initial_state()) {
            fail("a decay radiates off final-state particles only");
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
        if (!c.tags.empty() && c.tags.size() != n) {
            fail("the tags of a configuration must be none or one per particle");
        }
        bool known_types = !c.splitting_types.empty() && c.splitting_types.size() <= 2;
        for (const auto& type : c.splitting_types) {
            known_types = known_types && (type == "QCD" || type == "QED");
        }
        if (!known_types ||
            (c.splitting_types.size() == 2 &&
             c.splitting_types.at(0) == c.splitting_types.at(1))) {
            fail(
                "the splitting types of a configuration must be one or two "
                "different of QCD and QED"
            );
        }
        if ((c.extra_counterterm >= 0) != (c.splitting_types.size() == 2)) {
            fail(
                "a configuration has an extra counterterm with two splitting "
                "types and only then"
            );
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

std::size_t FKSCouplingOrders::born_input_count() const {
    std::size_t count = born.size();
    for (const auto& orders : extra_born) {
        count += orders.size();
    }
    return count;
}

std::size_t FKSCouplingOrders::extra_born_offset(std::size_t extra) const {
    std::size_t offset = born.size();
    for (std::size_t k = 0; k < extra; ++k) {
        offset += extra_born.at(k).size();
    }
    extra_born.at(extra);
    return offset;
}

namespace {

// the position in nlo of an order plus two powers of a splitting type, -1
// where that is beyond max_nlo (keep_order_cnt of born.f and born_cnt_<n>.f)
int shifted_order(
    const FKSCouplingOrders& orders, std::vector<int> order, const std::string& type
) {
    const auto& names = orders.split_orders;
    auto position = std::find(names.begin(), names.end(), type);
    if (position == names.end()) {
        throw std::invalid_argument(
            "FKSCouplingOrders: the splitting type " + type + " is not a split order"
        );
    }
    std::size_t p = position - names.begin();
    order.at(p) += 2;
    for (std::size_t k = 0; k < order.size(); ++k) {
        if (order.at(k) > orders.max_nlo.at(k)) {
            return -1;
        }
    }
    auto found = std::find(orders.nlo.begin(), orders.nlo.end(), order);
    if (found == orders.nlo.end()) {
        throw std::invalid_argument(
            "FKSCouplingOrders: the counterterms of a " + type +
            " splitting of a Born order have no NLO order"
        );
    }
    return static_cast<int>(found - orders.nlo.begin());
}

} // namespace

int FKSCouplingOrders::counterterm_order(
    std::size_t born_index, const std::string& type
) const {
    return shifted_order(*this, born.at(born_index), type);
}

int FKSCouplingOrders::extra_counterterm_order(
    std::size_t extra, std::size_t index, const std::string& type
) const {
    return shifted_order(*this, extra_born.at(extra).at(index), type);
}

FKSCouplingOrders FKSCouplingOrders::qcd(int alpha_s_power) {
    if (alpha_s_power < 0) {
        throw std::invalid_argument(
            "FKSCouplingOrders: the power of alpha_s cannot be negative"
        );
    }
    int power = 2 * alpha_s_power;
    return {{"QCD"}, {{power}}, {{power + 2}}, {power + 2}};
}

void madspace::check_fks_coupling_orders(
    const FKSCouplingOrders& orders, const std::string& user
) {
    auto fail = [&](const std::string& message) {
        throw std::invalid_argument(user + ": " + message);
    };
    std::size_t n = orders.split_orders.size();
    if (n == 0 || orders.born.empty() || orders.nlo.empty()) {
        fail("the coupling orders need split orders and Born and NLO orders");
    }
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t l = 0; l < k; ++l) {
            if (orders.split_orders.at(k) == orders.split_orders.at(l)) {
                fail("a split order appears twice");
            }
        }
    }
    bool sizes = orders.max_nlo.size() == n;
    for (const auto& order : orders.born) {
        sizes = sizes && order.size() == n;
    }
    for (const auto& order : orders.nlo) {
        sizes = sizes && order.size() == n;
    }
    for (const auto& extra : orders.extra_born) {
        sizes = sizes && !extra.empty();
        for (const auto& order : extra) {
            sizes = sizes && order.size() == n;
        }
    }
    if (!sizes) {
        fail("every coupling order needs a power of each split order");
    }
}
