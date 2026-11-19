#include "madspace/phasespace/eva.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <set>
#include <stdexcept>

#include "../kernels/eva.hpp"

using namespace madspace;

namespace {

// lepton masses of madevent's ElectroweakFlux.inc (PDG 2020)
constexpr double electron_mass = 0.5109989461e-3;
constexpr double muon_mass = 105.6583745e-3;

bool is_photon(int pid) { return pid == 22 || pid == 7; }

} // namespace

EvaFlux::EvaFlux(
    const std::array<int, 2>& beam_leptons,
    const std::array<double, 2>& beam_energies,
    double w_mass,
    double z_mass,
    double e2,
    const std::array<double, 2>& left_fractions,
    int order,
    bool pt_evolution,
    bool xcut
) :
    _beam_leptons(beam_leptons),
    _beam_energies(beam_energies),
    _w_mass(w_mass),
    _z_mass(z_mass),
    _e2(e2),
    _left_fractions(left_fractions),
    _order(order),
    _pt_evolution(pt_evolution),
    _xcut(xcut) {
    for (std::size_t beam = 0; beam < 2; ++beam) {
        int lepton = std::abs(beam_leptons.at(beam));
        if (lepton != 11 && lepton != 13) {
            throw std::invalid_argument(std::format(
                "EVA needs electron or muon beams, not PDG id {} on beam {}",
                beam_leptons.at(beam),
                beam + 1
            ));
        }
        if (!(beam_energies.at(beam) > 0.)) {
            throw std::invalid_argument("EVA needs positive beam energies");
        }
        double fraction = left_fractions.at(beam);
        if (!(fraction >= 0. && fraction <= 1.)) {
            throw std::invalid_argument(std::format(
                "EVA: left-handed fraction {} of beam {} is not in [0, 1]",
                fraction,
                beam + 1
            ));
        }
    }
    if (order < 0 || order > 2) {
        throw std::invalid_argument("EVA order must be 0 (LLA), 1 (LP) or 2 (NLP)");
    }
    if (!(w_mass > 0. && z_mass > w_mass && e2 > 0.)) {
        throw std::invalid_argument("EVA needs 0 < M_W < M_Z and e^2 > 0");
    }
}

bool EvaFlux::has_density(int vector_pid, std::size_t beam) const {
    int lepton = _beam_leptons.at(beam);
    if (std::abs(vector_pid) == 24) {
        // a W+ from a positive lepton (charge +1 - 0), a W- from a negative one
        return (vector_pid > 0) == (lepton < 0);
    }
    return vector_pid == 23 || is_photon(vector_pid);
}

std::vector<double> EvaFlux::parameters(int vector_pid, std::size_t beam) const {
    using namespace kernels;
    if (std::abs(vector_pid) != 24 && vector_pid != 23 && !is_photon(vector_pid)) {
        throw std::invalid_argument(std::format(
            "EVA densities exist for W, Z and photons only, not for PDG id {}",
            vector_pid
        ));
    }
    std::vector<double> row(parameter_count, 0.);
    int lepton = _beam_leptons.at(beam);
    double ebeam = _beam_energies.at(beam);
    // ElectroweakFlux.inc: aEM, MW, MZ scheme
    double mw2 = _w_mass * _w_mass, mz2 = _z_mass * _z_mass;
    double cw2 = mw2 / mz2;
    double sw2 = 1. - cw2;
    double gw2 = _e2 / sw2;
    double gz2 = gw2 / cw2;
    double lepton_mass = std::abs(lepton) == 11 ? electron_mass : muon_mass;
    double gg2, gl2, gr2, mv2, mu2min, xmin;
    int order = _order;
    if (!has_density(vector_pid, beam)) {
        return row; // kind 0: no density
    }
    if (std::abs(vector_pid) == 24) {
        gg2 = gw2 / 2.;
        gl2 = 1.;
        gr2 = 0.;
        mv2 = mw2;
        mu2min = mw2;
        xmin = _xcut ? _w_mass / ebeam : 0.;
    } else if (vector_pid == 23) {
        double zle = -0.5 + sw2; // tLe - qe sw2
        double zre = sw2;        // -qe sw2
        gg2 = gz2;
        gl2 = zle * zle;
        gr2 = zre * zre;
        mv2 = mz2;
        mu2min = mz2;
        xmin = _xcut ? _z_mass / ebeam : 0.;
    } else {
        gg2 = _e2; // e^2 Q_e^2
        gl2 = 1.;
        gr2 = 1.;
        mv2 = lepton_mass * lepton_mass;
        mu2min = mv2;
        xmin = 0.;
        order = 0; // m_photon = 0: always LLA
    }
    if (lepton < 0) {
        std::swap(gl2, gr2); // an antilepton: parity inverted
    }
    row.at(eva_kind) = is_photon(vector_pid) ? 2. : 1.;
    row.at(eva_gg2) = gg2;
    row.at(eva_gl2) = gl2;
    row.at(eva_gr2) = gr2;
    row.at(eva_flpol) = _left_fractions.at(beam);
    row.at(eva_mv2) = mv2;
    row.at(eva_mu2min) = mu2min;
    row.at(eva_xmin) = xmin;
    row.at(eva_ebeam) = ebeam;
    row.at(eva_order) = order;
    row.at(eva_evolution) = _pt_evolution ? 1. : 0.;
    return row;
}

double EvaFlux::density(
    int vector_pid, std::size_t beam, int helicity, double x, double q
) const {
    auto row = parameters(vector_pid, beam);
    return kernels::eva_density_value<double>(x, q, helicity, row);
}

EvaLegTables::EvaLegTables(
    const EvaFlux& flux,
    const nested_vector2<me_int_t>& pid_options,
    const nested_vector3<me_int_t>& helicities
) {
    if (helicities.size() != pid_options.size()) {
        throw std::invalid_argument(std::format(
            "EVA: {} helicity tables for {} flavours",
            helicities.size(),
            pid_options.size()
        ));
    }
    std::vector<std::pair<int, std::size_t>> config_keys;
    auto config_index = [&](int pid, std::size_t beam) -> me_int_t {
        if (!flux.has_density(pid, beam)) {
            return -1;
        }
        std::pair<int, std::size_t> key{pid, beam};
        auto found = std::find(config_keys.begin(), config_keys.end(), key);
        if (found != config_keys.end()) {
            return static_cast<me_int_t>(found - config_keys.begin());
        }
        config_keys.push_back(key);
        params.push_back(flux.parameters(pid, beam));
        return static_cast<me_int_t>(config_keys.size() - 1);
    };
    for (std::size_t flavor = 0; flavor < pid_options.size(); ++flavor) {
        auto& option = pid_options.at(flavor);
        auto& rows = helicities.at(flavor);
        if (option.size() < 2) {
            throw std::invalid_argument("EVA needs two incoming particles");
        }
        if (rows.empty()) {
            throw std::invalid_argument("EVA: empty helicity table");
        }
        for (std::size_t leg = 0; leg < 2; ++leg) {
            int pid = option.at(leg);
            // throws for a particle that is not a vector boson
            flux.parameters(pid, 0);
            row_offsets.at(leg).push_back(
                static_cast<me_int_t>(leg_helicities.at(leg).size())
            );
            row_counts.at(leg).push_back(static_cast<me_int_t>(rows.size()));
            std::set<me_int_t> states;
            for (auto& row : rows) {
                if (row.size() < 2) {
                    throw std::invalid_argument("EVA: helicity row too short");
                }
                leg_helicities.at(leg).push_back(row.at(leg));
                states.insert(row.at(leg));
            }
            helicity_counts.at(leg).push_back(static_cast<double>(states.size()));
            for (std::size_t mirror = 0; mirror < 2; ++mirror) {
                configs.at(leg).push_back(config_index(pid, leg ^ mirror));
            }
        }
    }
    if (params.empty()) {
        // the kernels need a table even when no combination has a density
        params.push_back(std::vector<double>(EvaFlux::parameter_count, 0.));
    }
}
