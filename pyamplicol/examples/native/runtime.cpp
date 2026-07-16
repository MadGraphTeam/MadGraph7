// SPDX-License-Identifier: 0BSD

#include <rusticol.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: runtime_cpp ARTIFACT [PROCESS [PARAMETERS.json]]\n";
        return 2;
    }

    try {
        const std::string process = argc >= 3 ? argv[2] : "";
        rusticol::Runtime runtime(argv[1], process);
        if (argc == 4) {
            runtime.set_model_parameters_json(argv[3]);
        }
        runtime.set_model_parameter("normalization.alpha_s_me_check", 0.118);

        // One d d~ > z g point, flattened as [point][particle][E,px,py,pz].
        const std::vector<double> momenta{
            500.0, 0.0, 0.0, 500.0,
            500.0, 0.0, 0.0, -500.0,
            504.157625672, -304.1084262865, 208.76026523528103,
            331.35611794513767,
            495.842374328, 304.1084262865, -208.76026523528103,
            -331.35611794513767,
        };
        constexpr std::size_t point_count = 1;
        const auto totals = runtime.evaluate(momenta, point_count);
        const auto resolved = runtime.evaluate_resolved(momenta, point_count);
        const auto checked_totals = resolved.total();
        const double scale = std::max(1.0, std::abs(totals.at(0)));
        if (std::abs(totals.at(0) - checked_totals.at(0)) > 1.0e-12 * scale) {
            std::cerr << "resolved components do not reproduce the total\n";
            return 1;
        }

        std::cout << std::setprecision(17)
                  << "process=" << runtime.process_key() << "\n"
                  << "color_accuracy=" << runtime.color_accuracy() << "\n"
                  << "total=" << totals.at(0) << "\n";
    } catch (const std::exception &error) {
        std::cerr << "Rusticol error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
