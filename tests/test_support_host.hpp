#pragma once

// Test helpers that need no CUDA headers and no device. Everything here builds
// in the host only configuration, so the validation cases that use only these
// helpers run in continuous integration on a machine with no toolkit at all.
// The device helpers live in test_support_gpu.hpp, which includes this file.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#include "cps/core/snapshot.hpp"

namespace cps_test {

// Independent double precision reference for the gravitational acceleration.
//
// Deliberately the most literal possible transcription of the force law: no
// tiling, no reciprocal square root, no single precision anywhere. Its only
// job is to be obviously correct, so that when it disagrees with a kernel the
// kernel is what is wrong.
//
// The positions and masses are taken as arrays of any floating type, because a
// HostParticleState holds single precision and an integrator that must stay in
// double precision cannot go through one.
template <typename Real>
inline void reference_accelerations(const std::vector<Real>& x, const std::vector<Real>& y,
                                    const std::vector<Real>& z, const std::vector<Real>& mass,
                                    double softening, double gravitational_constant,
                                    std::vector<double>& ax, std::vector<double>& ay,
                                    std::vector<double>& az) {
    const std::size_t n = x.size();
    ax.assign(n, 0.0);
    ay.assign(n, 0.0);
    az.assign(n, 0.0);
    const double softening_sq = softening * softening;

    for (std::size_t i = 0; i < n; ++i) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const double dx = static_cast<double>(x[j]) - static_cast<double>(x[i]);
            const double dy = static_cast<double>(y[j]) - static_cast<double>(y[i]);
            const double dz = static_cast<double>(z[j]) - static_cast<double>(z[i]);
            const double dist_sq = dx * dx + dy * dy + dz * dz + softening_sq;
            const double inv = 1.0 / std::sqrt(dist_sq);
            const double scale = static_cast<double>(mass[j]) * inv * inv * inv;
            sum_x += dx * scale;
            sum_y += dy * scale;
            sum_z += dz * scale;
        }
        ax[i] = gravitational_constant * sum_x;
        ay[i] = gravitational_constant * sum_y;
        az[i] = gravitational_constant * sum_z;
    }
}

inline void reference_accelerations(const cps::HostParticleState& state, double softening,
                                    double gravitational_constant, std::vector<double>& ax,
                                    std::vector<double>& ay, std::vector<double>& az) {
    reference_accelerations(state.x, state.y, state.z, state.mass, softening,
                            gravitational_constant, ax, ay, az);
}

template <typename Real>
inline double reference_potential(const std::vector<Real>& x, const std::vector<Real>& y,
                                  const std::vector<Real>& z, const std::vector<Real>& mass,
                                  double softening, double gravitational_constant) {
    const std::size_t n = x.size();
    const double softening_sq = softening * softening;
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dx = static_cast<double>(x[j]) - static_cast<double>(x[i]);
            const double dy = static_cast<double>(y[j]) - static_cast<double>(y[i]);
            const double dz = static_cast<double>(z[j]) - static_cast<double>(z[i]);
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + softening_sq);
            total -= static_cast<double>(mass[i]) * static_cast<double>(mass[j]) / dist;
        }
    }
    return gravitational_constant * total;
}

inline double reference_potential(const cps::HostParticleState& state, double softening,
                                  double gravitational_constant) {
    return reference_potential(state.x, state.y, state.z, state.mass, softening,
                               gravitational_constant);
}

// A reproducible cloud with no two particles on top of each other.
inline cps::HostParticleState random_cloud(int count, uint32_t seed, float extent = 1.0f) {
    cps::HostParticleState state;
    state.resize(static_cast<std::size_t>(count));
    uint32_t rng = seed == 0u ? 1u : seed;
    auto next = [&rng]() {
        // xorshift32, written out so the sequence does not depend on the
        // standard library version.
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return (static_cast<double>(rng) + 0.5) / 4294967296.0;
    };
    for (int i = 0; i < count; ++i) {
        state.x[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.y[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.z[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * extent);
        state.vx[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.vy[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.vz[static_cast<std::size_t>(i)] = static_cast<float>((next() * 2.0 - 1.0) * 0.1);
        state.mass[static_cast<std::size_t>(i)] = static_cast<float>(0.5 + next());
    }
    return state;
}

inline double relative_error(double measured, double expected) {
    const double scale = std::abs(expected);
    return scale > 0.0 ? std::abs(measured - expected) / scale : std::abs(measured);
}

// Upper tail of the chi squared distribution, needed by the Plummer radial
// density gate. Implemented here rather than pulled in from a statistics
// library because it is forty lines and adding a dependency to evaluate one
// special function would be a poor trade.
//
// Q(a, x) is the regularised upper incomplete gamma function. The series
// converges quickly below x = a + 1 and the continued fraction above it, which
// is the standard split.
inline double incomplete_gamma_series(double a, double x) {
    const int max_iterations = 1000;
    const double epsilon = 3.0e-14;
    double ap = a;
    double sum = 1.0 / a;
    double term = sum;
    for (int n = 0; n < max_iterations; ++n) {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if (std::abs(term) < std::abs(sum) * epsilon) {
            break;
        }
    }
    return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

inline double incomplete_gamma_continued_fraction(double a, double x) {
    const int max_iterations = 1000;
    const double epsilon = 3.0e-14;
    const double tiny = 1.0e-300;

    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i <= max_iterations; ++i) {
        const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
        b += 2.0;
        d = an * d + b;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = b + an / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::abs(delta - 1.0) < epsilon) {
            break;
        }
    }
    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

inline double regularised_upper_gamma(double a, double x) {
    if (x < 0.0 || a <= 0.0) {
        return std::nan("");
    }
    if (x == 0.0) {
        return 1.0;
    }
    if (x < a + 1.0) {
        return 1.0 - incomplete_gamma_series(a, x);
    }
    return incomplete_gamma_continued_fraction(a, x);
}

// Probability of seeing a chi squared statistic at least this large by chance,
// given the number of degrees of freedom.
inline double chi_square_p_value(double chi_square, int degrees_of_freedom) {
    if (degrees_of_freedom <= 0) {
        return std::nan("");
    }
    return regularised_upper_gamma(0.5 * degrees_of_freedom, 0.5 * chi_square);
}

// ---------------------------------------------------------------------------
// Recorded results
// ---------------------------------------------------------------------------
//
// The validation suite writes what it measures: one row per physics gate in
// validation_gates.csv, plus a few series the report plots. The canonical
// copies under experiments/results are committed, so an ordinary test run must
// never touch them. By default everything goes to a results directory in the
// build tree. Only CPS_RECORD_GATES=1 in the environment redirects it to
// experiments/results, and even then the gate file is merged rather than
// rewritten, so a partial run replaces only the gates it measured.

inline constexpr const char* GATE_CSV_HEADER = "gate,measured,limit,comparison,passed";

// True only for CPS_RECORD_GATES=1 exactly. Any other value, including 0 and
// an empty string, leaves the committed results alone.
inline bool recording_committed_results() {
    const char* value = std::getenv("CPS_RECORD_GATES");
    return value != nullptr && std::string(value) == "1";
}

// Where this process writes recorded results. Empty when the test was built
// without the location macros, in which case nothing is written.
inline std::filesystem::path results_directory() {
#if defined(CPS_PROJECT_ROOT) && defined(CPS_BINARY_DIR)
    if (recording_committed_results()) {
        return std::filesystem::path(CPS_PROJECT_ROOT) / "experiments" / "results";
    }
    return std::filesystem::path(CPS_BINARY_DIR) / "results";
#else
    return {};
#endif
}

// The registered gate names, one per line, in the order the CSV lists them.
// Blank lines and lines starting with # are ignored. An unreadable file gives
// an empty list.
inline std::vector<std::string> read_gate_registry(const std::filesystem::path& path) {
    std::vector<std::string> names;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') {
            continue;
        }
        const auto last = line.find_last_not_of(" \t\r");
        names.push_back(line.substr(first, last - first + 1));
    }
    return names;
}

inline std::filesystem::path gate_registry_path() {
#if defined(CPS_PROJECT_ROOT)
    return std::filesystem::path(CPS_PROJECT_ROOT) / "tests" / "validation" / "gates.txt";
#else
    return {};
#endif
}

// One CSV row, formatted exactly as the report generator expects to read it.
inline std::string format_gate_row(const std::string& name, double measured, double limit,
                                   const std::string& comparison, bool passed) {
    const int size = std::snprintf(nullptr, 0, "%s,%.17g,%.17g,%s,%d", name.c_str(), measured,
                                   limit, comparison.c_str(), passed ? 1 : 0);
    std::string row(static_cast<std::size_t>(size), '\0');
    std::snprintf(row.data(), row.size() + 1, "%s,%.17g,%.17g,%s,%d", name.c_str(), measured,
                  limit, comparison.c_str(), passed ? 1 : 0);
    return row;
}

struct GateRow {
    std::string gate;
    std::string line;  // the full CSV row, without a newline
};

// Merges freshly measured rows into the gate file at path.
//
// Rows already in the file are kept byte for byte unless this call measured
// the same gate again. The output lists registered gates in registry order,
// then any unregistered rows in the order they were found, each with a warning,
// because a row nobody registered is either stale or a test that forgot to
// register its gate. The file is written to a sibling temporary and renamed
// over the original, so an interrupted run leaves either the old file or the
// new one, never half of either. Returns false if the file could not be
// written.
inline bool merge_gate_rows(const std::filesystem::path& path,
                            const std::vector<std::string>& registry,
                            const std::vector<GateRow>& measured, std::ostream& warnings) {
    namespace fs = std::filesystem;

    std::vector<std::string> existing_order;
    std::map<std::string, std::string> existing;
    {
        std::ifstream in(path, std::ios::binary);
        std::string line;
        bool first_line = true;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            const bool header = first_line && line.rfind("gate,", 0) == 0;
            first_line = false;
            if (header || line.empty()) {
                continue;
            }
            const std::string gate = line.substr(0, line.find(','));
            if (existing.emplace(gate, line).second) {
                existing_order.push_back(gate);
            } else {
                warnings << "warning: " << path.string() << " lists gate " << gate
                         << " more than once; keeping the first row\n";
            }
        }
    }

    std::map<std::string, std::string> fresh;
    std::vector<std::string> fresh_order;
    for (const GateRow& row : measured) {
        if (fresh.find(row.gate) == fresh.end()) {
            fresh_order.push_back(row.gate);
        }
        fresh[row.gate] = row.line;
    }

    const std::set<std::string> registered(registry.begin(), registry.end());
    std::vector<std::string> lines;
    for (const std::string& gate : registry) {
        if (const auto it = fresh.find(gate); it != fresh.end()) {
            lines.push_back(it->second);
        } else if (const auto old = existing.find(gate); old != existing.end()) {
            lines.push_back(old->second);
        }
    }
    for (const std::string& gate : existing_order) {
        if (registered.count(gate) != 0) {
            continue;
        }
        warnings << "warning: gate " << gate << " in " << path.string()
                 << " is not registered in tests/validation/gates.txt; keeping its row\n";
        const auto it = fresh.find(gate);
        lines.push_back(it != fresh.end() ? it->second : existing[gate]);
    }
    for (const std::string& gate : fresh_order) {
        if (registered.count(gate) != 0 || existing.count(gate) != 0) {
            continue;
        }
        warnings << "warning: measured gate " << gate
                 << " is not registered in tests/validation/gates.txt; appending its row\n";
        lines.push_back(fresh[gate]);
    }

    std::error_code error;
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path(), error);
    }
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out << GATE_CSV_HEADER << '\n';
        for (const std::string& line : lines) {
            out << line << '\n';
        }
        out.flush();
        if (!out) {
            out.close();
            fs::remove(temporary, error);
            return false;
        }
    }
    fs::rename(temporary, path, error);
    if (error) {
        // Rename over an existing file fails on some filesystems; remove and
        // retry rather than leaving the temporary behind.
        std::error_code ignored;
        fs::remove(path, ignored);
        fs::rename(temporary, path, error);
        if (error) {
            fs::remove(temporary, ignored);
            return false;
        }
    }
    return true;
}

// Records the measured value of a physics gate alongside its limit.
//
// The report quotes gate results from the committed copy of this file rather
// than from anything typed by hand, so a gate that moves cannot silently
// disagree with the prose that describes it. Each call merges one row, so a
// filtered run, a run that skips the device cases, or a crash part way through
// leaves every gate it did not reach exactly as it was.
inline void record_gate(const std::string& name, double measured, double limit,
                        const std::string& comparison, bool passed) {
    const std::filesystem::path directory = results_directory();
    if (directory.empty()) {
        return;
    }
    const std::filesystem::path path = directory / "validation_gates.csv";
    const std::vector<std::string> registry = read_gate_registry(gate_registry_path());
    if (registry.empty()) {
        std::cerr << "warning: no registered gates could be read from "
                  << gate_registry_path().string() << "\n";
    }
    static bool announced = false;
    if (!announced) {
        std::cerr << "recording validation gates to " << path.string() << "\n";
        announced = true;
    }
    const GateRow row{name, format_gate_row(name, measured, limit, comparison, passed)};
    if (!merge_gate_rows(path, registry, {row}, std::cerr)) {
        std::cerr << "warning: could not write " << path.string() << "\n";
    }
}

// Writes a two column artifact the report plots, so it can show a gate rather
// than merely assert it. The path should come from results_directory().
inline void write_series_csv(const std::string& path, const std::string& header,
                             const std::vector<double>& first,
                             const std::vector<double>& second) {
    std::FILE* file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return;
    }
    std::fprintf(file, "%s\n", header.c_str());
    const std::size_t n = std::min(first.size(), second.size());
    for (std::size_t i = 0; i < n; ++i) {
        std::fprintf(file, "%.17g,%.17g\n", first[i], second[i]);
    }
    std::fclose(file);
}

}  // namespace cps_test
