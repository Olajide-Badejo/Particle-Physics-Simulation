#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/config.hpp"

namespace cps {

// Command line of cuda_physics_sim. Overrides are collected here and applied
// after the configuration file is read, so a command line switch always wins
// over the file regardless of order. Parsing is host only so it can be tested
// without a GPU.
struct CliOptions {
    bool show_help = false;
    bool show_version = false;
    bool resume = false;
    // Resume even when the snapshot's physics differ from the configuration.
    // Implies resume.
    bool force_resume = false;
    bool device_info = false;
    bool print_config = false;
    // Downgrades the fluid solver's refusal of an unstable time step to a
    // warning. Only ever sets the configuration field, never clears it.
    bool allow_unstable_dt = false;
    std::string config_path;

    std::optional<SolverType> solver;
    std::optional<NBodyVariant> variant;
    std::optional<long long> steps;
    std::optional<int> particles;
    std::optional<float> dt;
    std::optional<uint32_t> seed;
    std::optional<std::string> snapshot_dir;
    std::optional<long long> snapshot_every;
    std::optional<std::string> diagnostics_path;
    std::optional<long long> diag_every;
};

// Throws std::runtime_error naming the flag for an unknown option, a missing
// value, a value that is not a number, or a value out of range. Stops at
// --help or --version, as the original loop did.
CliOptions parse_cli(int argc, const char* const* argv);

// Copies every override that was given into config. Does not validate; call
// Config::validate() afterwards.
void apply_cli_overrides(const CliOptions& options, Config& config);

std::string cli_usage();

}  // namespace cps
