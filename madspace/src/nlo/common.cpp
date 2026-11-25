#include "madspace/nlo/common.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>

using namespace madspace;

std::pair<Value, Value> madspace::born_frame_light_cone(
    FunctionBuilder& fb,
    const PhaseSpaceMapping& born,
    Value x1,
    Value x2,
    const NamedVector<Value>& conditions
) {
    // the rapidity log(x1 / x2) / 2 of the Born in the beams' frame, which
    // moves with beam_rapidity in the lab. With mirror_index 1 the first leg
    // comes from the second beam, along -z: then x2 belongs to the parton
    // along +z. The products with 1 and 0 select exactly.
    double rapidity = born.beam_rapidity();
    double exp_plus = std::exp(rapidity), exp_minus = std::exp(-rapidity);
    // the momentum fractions of lepton beams or of a decay are one
    if (born.leptonic() || born.is_decay()) {
        return {exp_plus, exp_minus};
    }
    Value x_plus = x1, x_minus = x2;
    if (born.mirror_beams()) {
        auto mirror = conditions.at("mirror_index");
        auto keep = fb.gather(mirror, std::vector<double>{1., 0.});
        auto swap = fb.gather(mirror, std::vector<double>{0., 1.});
        x_plus = fb.add(fb.mul(keep, x1), fb.mul(swap, x2));
        x_minus = fb.add(fb.mul(keep, x2), fb.mul(swap, x1));
    }
    if (rapidity == 0.) {
        return {x_plus, x_minus};
    }
    return {fb.mul(x_plus, exp_plus), fb.mul(x_minus, exp_minus)};
}

ColorMatrix::ColorMatrix(
    const std::vector<int>& color_representations, const std::string& who
) {
    // the coloured legs with the index of their Casimir
    std::size_t n = color_representations.size();
    std::vector<std::size_t> colored;
    std::vector<std::size_t> casimir_index;
    for (std::size_t leg = 0; leg < n; ++leg) {
        int representation = color_representations.at(leg);
        if (representation == 1) {
            continue;
        }
        double casimir;
        if (std::abs(representation) == 3) {
            casimir = QCD_CF;
        } else if (representation == 8) {
            casimir = QCD_CA;
        } else {
            throw std::invalid_argument(
                std::format(
                    "{}: no Casimir of the colour representation {}",
                    who,
                    representation
                )
            );
        }
        auto found = std::find(_casimirs.begin(), _casimirs.end(), casimir);
        casimir_index.push_back(static_cast<std::size_t>(found - _casimirs.begin()));
        if (found == _casimirs.end()) {
            _casimirs.push_back(casimir);
        }
        colored.push_back(leg);
    }
    // the rows of the colour matrix from the pairs i < j of the coloured legs,
    // pair (a, c) at a (2 n_c - a - 1) / 2 + c - a - 1, the Casimirs times the
    // Born after them and a zero for the colourless legs
    std::size_t n_c = colored.size();
    std::size_t pairs = n_c * (n_c - 1) / 2;
    _pair_count = pairs;
    me_int_t zero = static_cast<me_int_t>(pairs + _casimirs.size());
    _rows.assign(n, std::vector<me_int_t>(n, zero));
    for (std::size_t a = 0; a < n_c; ++a) {
        _rows.at(colored.at(a)).at(colored.at(a)) =
            static_cast<me_int_t>(pairs + casimir_index.at(a));
        for (std::size_t c = a + 1; c < n_c; ++c) {
            auto pair = static_cast<me_int_t>(a * (2 * n_c - a - 1) / 2 + c - a - 1);
            _rows.at(colored.at(a)).at(colored.at(c)) = pair;
            _rows.at(colored.at(c)).at(colored.at(a)) = pair;
        }
    }
    _has_color = n_c >= 2;
}

Value ColorMatrix::build(FunctionBuilder& fb, Value born, Value pairs) const {
    Value batch_size = fb.batch_size({born});
    ValueVec born_columns = fb.unstack(born);
    ValueVec pair_columns = fb.unstack(pairs);
    Value zero = fb.full({0., batch_size});
    ValueVec matrices;
    for (std::size_t k = 0; k < born_columns.size(); ++k) {
        ValueVec diagonal;
        for (double casimir : _casimirs) {
            diagonal.push_back(
                fb.mul(fb.full({casimir, batch_size}), born_columns.at(k))
            );
        }
        diagonal.push_back(zero);
        Value entries = fb.cat({pair_columns.at(k), fb.stack(diagonal)});
        ValueVec rows;
        for (auto& row : _rows) {
            rows.push_back(fb.select(entries, row));
        }
        matrices.push_back(fb.stack(rows));
    }
    return fb.stack(matrices);
}

std::pair<ValueVec, ValueVec> madspace::virtual_trick(
    FunctionBuilder& fb,
    Value virtual_value,
    Value born,
    Value ratio,
    Value fraction,
    Value evaluated
) {
    // r B everywhere, (V - r B) / f where evaluated
    Value born_sum = fb.reduce_sum(born);
    ValueVec ratios = fb.unstack(ratio);
    ValueVec virtual_columns = fb.unstack(virtual_value);
    ValueVec used, approximations;
    for (std::size_t k = 0; k < ratios.size(); ++k) {
        Value approximation = fb.mul(ratios.at(k), born_sum);
        Value residual = fb.mul(
            evaluated, fb.div(fb.sub(virtual_columns.at(k), approximation), fraction)
        );
        approximations.push_back(approximation);
        used.push_back(fb.add(approximation, residual));
    }
    return {used, approximations};
}
