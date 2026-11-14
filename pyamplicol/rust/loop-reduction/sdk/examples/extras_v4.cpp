// SPDX-License-Identifier: 0BSD
// Plan v4 SDK extras through the C++ wrapper: ./extras_v4_cpp artifact process
// input.txt, with extras_v4.c's input and output formats (resolved,
// colour- and spin-correlated, double-double and arbitrary-precision calls).
#include "rusticol_virtual.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
double hex(std::istream &in) {
  std::string text;
  if (!(in >> text)) throw std::runtime_error("short input");
  return std::strtod(text.c_str(), nullptr);
}
struct Call {
  const char *name, *arithmetic;
  unsigned digits;
  const char *poles;
  int kind;  // 0 ordinary, 1 correlated, 2 correlated with spin, 3 resolved
  std::size_t points;
};
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  // An optional fourth argument "binary64" runs the binary64 calls only.
  const bool binary64_only = argc > 4 && std::string(argv[4]) == "binary64";
  std::ifstream in(argv[3]);
  std::size_t points = 0, legs = 0, spin_legs = 0;
  in >> points >> legs;
  std::vector<double> momenta(points * legs * 4), scales(points);
  for (std::size_t p = 0; p < points; ++p) {
    for (std::size_t i = 0; i < legs * 4; ++i) momenta[p * legs * 4 + i] = hex(in);
    scales[p] = hex(in);
  }
  in >> spin_legs;
  std::vector<std::size_t> labels(spin_legs);
  std::vector<std::complex<double>> vectors;
  for (std::size_t l = 0; l < spin_legs; ++l) {
    in >> labels[l];
    for (int c = 0; c < 4; ++c) {
      double re = hex(in), im = hex(in);
      vectors.emplace_back(re, im);
    }
  }
  try {
    rusticol::VirtualRuntime runtime(argv[1], std::string(argv[2]));
    const Call calls[] = {
        {"ordinary", "double", 0, "all", 0, 2},
        {"correlated", "double", 0, "all", 1, 2},
        {"correlated-spin", "double", 0, "all", 2, 2},
        {"correlated-uv", "double", 0, "uv", 1, 2},
        {"correlated-none", "double", 0, "none", 1, 2},
        {"resolved", "double", 0, "all", 3, 2},
        {"resolved-ir", "double", 0, "ir", 3, 2},
        {"ordinary-dd", "double-double", 0, "all", 0, 1},
        {"correlated-spin-dd", "double-double", 0, "all", 2, 1},
        {"resolved-dd", "double-double", 0, "all", 3, 1},
        {"ordinary-ap", "arbitrary", 32, "all", 0, 1},
        {"correlated-spin-ap", "arbitrary", 32, "all", 2, 1},
        {"resolved-ap", "arbitrary", 32, "all", 3, 1},
    };
    for (const auto &call : calls) {
      if (binary64_only && std::string(call.arithmetic) != "double") continue;
      const std::size_t n = call.points < points ? call.points : points;
      runtime.set_arithmetic(call.arithmetic, call.digits);
      std::printf("call %s\n", call.name);
      try {
        runtime.set_poles(call.poles);
      } catch (const rusticol::VirtualError &refused) {
        // A pole selection the process does not serve (loop-induced: "all"
        // and "none"): the refusal, then the next call.
        std::printf("refused %s\n", refused.what());
        continue;
      }
      std::vector<double> p(momenta.begin(), momenta.begin() + n * legs * 4);
      std::vector<double> s(scales.begin(), scales.begin() + n);
      if (call.kind == 0 || call.kind == 3) {
        auto result = runtime.evaluate(p, n, s, {}, call.kind == 3);
        std::printf("born");
        for (double v : result.born()) std::printf(" %a", v);
        std::printf("\nvirtual");
        for (double v : result.virtual_values()) std::printf(" %a", v);
        std::printf("\ndecimal %s\n", result.decimal_json().c_str());
      } else {
        auto result = call.kind == 2 ? runtime.evaluate_correlated_with_sources(p, n, s, labels, vectors)
                                     : runtime.evaluate_correlated(p, n, s);
        std::printf("born");
        for (auto v : result.born()) std::printf(" %a %a", v.real(), v.imag());
        std::printf("\nvirtual");
        for (auto v : result.virtual_values()) std::printf(" %a %a", v.real(), v.imag());
        std::printf("\ndecimal %s\n", result.decimal_json().c_str());
      }
    }
  } catch (const rusticol::VirtualError &error) {
    std::printf("error %d %s\n", error.status(), error.what());
    return 1;
  }
  return 0;
}
