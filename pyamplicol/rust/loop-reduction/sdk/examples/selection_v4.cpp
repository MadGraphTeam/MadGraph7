// SPDX-License-Identifier: 0BSD
// Loop-induced selections through the C++ wrapper (VirtualRuntime::evaluate_selected):
// ./selection_v4_cpp artifact process input.txt [binary64], with selection_v4.c's
// input and output formats.
#include "rusticol_virtual.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Call {
    const char *name, *arithmetic;
    std::uint32_t digits;
    bool resolved;
    int ids;
    bool by_point, colour;
    const char *resolution;
    std::size_t points;
};
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const bool binary64_only = argc > 4 && std::string(argv[4]) == "binary64";
    std::FILE *input = std::fopen(argv[3], "r");
    std::size_t points = 0, legs = 0, count = 0;
    if (!input || std::fscanf(input, "%zu %zu", &points, &legs) != 2) return 2;
    std::vector<double> momenta(points * legs * 4), scales(points);
    for (std::size_t p = 0; p < points; p++) {
        for (std::size_t i = 0; i < legs * 4; i++)
            if (std::fscanf(input, "%la", &momenta[p * legs * 4 + i]) != 1) return 2;
        if (std::fscanf(input, "%la", &scales[p]) != 1) return 2;
    }
    if (std::fscanf(input, "%zu", &count) != 1) return 2;
    std::vector<std::string> ids;
    for (std::size_t k = 0; k < count; k++) {
        char buffer[256];
        if (std::fscanf(input, "%255s", buffer) != 1) return 2;
        ids.emplace_back(buffer);
    }
    std::vector<std::uint32_t> rows(points);
    for (std::size_t p = 0; p < points; p++)
        if (std::fscanf(input, "%u", &rows[p]) != 1) return 2;
    std::fclose(input);
    try {
        rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
        const Call calls[] = {
            {"anchors", "double", 0, true, 0, false, false, "amplitude-basis", 2},
            {"one", "double", 0, false, 1, false, false, nullptr, 2},
            {"subset", "double", 0, false, 2, false, false, nullptr, 2},
            {"resolved-subset", "double", 0, true, 2, false, false, "contracted", 2},
            {"anchors-subset", "double", 0, true, 2, false, false, "amplitude-basis", 2},
            {"by-point", "double", 0, false, 0, true, false, nullptr, 2},
            {"colour", "double", 0, false, 0, false, true, nullptr, 2},
            {"anchors-dd", "double-double", 0, true, 0, false, false, "amplitude-basis", 1},
            {"by-point-dd", "double-double", 0, false, 0, true, false, nullptr, 1},
            {"anchors-ap", "arbitrary", 32, true, 0, false, false, "amplitude-basis", 1},
            {"subset-ap", "arbitrary", 32, false, 2, false, false, nullptr, 1},
        };
        for (const Call &call : calls) {
            if (binary64_only && std::string(call.arithmetic) != "double") continue;
            const std::size_t n = call.points < points ? call.points : points;
            runtime.set_arithmetic(call.arithmetic, call.digits);
            std::printf("call %s\n", call.name);
            rusticol::VirtualSelection selection;
            if (call.ids == 1) selection.helicity_ids = {ids.front()};
            if (call.ids == 2) selection.helicity_ids = ids;
            if (call.by_point) selection.helicity_by_point.assign(rows.begin(), rows.begin() + n);
            if (call.colour) selection.color_ids = {"1"};
            if (call.resolution) selection.color_resolution = std::string(call.resolution);
            std::vector<double> batch(momenta.begin(), momenta.begin() + n * legs * 4);
            std::vector<double> mu(scales.begin(), scales.begin() + n);
            try {
                auto result = runtime.evaluate_selected(batch, n, mu, selection, call.resolved);
                std::printf("statuses");
                for (auto status : result.statuses()) std::printf(" %u", status);
                std::printf("\nvirtual");
                for (double value : result.virtual_values()) std::printf(" %a", value);
                std::printf("\ndecimal %s\naxes %s\n", result.decimal_json().c_str(),
                            result.axes_json().c_str());
            } catch (const rusticol::VirtualError &error) {
                std::printf("refused %s\n", error.what());
            }
        }
    } catch (const std::exception &error) {
        std::printf("error %s\n", error.what());
        return 1;
    }
    return 0;
}
