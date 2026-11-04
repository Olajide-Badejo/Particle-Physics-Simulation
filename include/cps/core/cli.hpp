#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "cps/core/config.hpp"

namespace cps {

// Command line of cps. Overrides are collected here and applied
// after the configuration file is read, so a command line switch always wins
// over the file regardless of order. Parsing is host only so it can be tested
// without a GPU.
//
// Which form the command line took. Legacy is the flags with no subcommand, kept
// as an alias for one major version; Usage is no arguments at all.
enum class CliCommand {
    Legacy,
    Usage,
    Run,
    Presets,
    Info,
    Convert,
};

struct CliOptions {
    CliCommand command = CliCommand::Legacy;
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
    // For convert this is the configuration that fingerprints a CPS1 snapshot.
    std::string config_path;
    std::string run_target;
    std::string convert_input;
    std::string convert_output;

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
    std::optional<std::string> vtk_dir;
    std::optional<long long> vtk_every;
};

// Throws std::runtime_error naming the flag for an unknown option, a missing
// value, a value that is not a number, or a value out of range. Stops at
// --help or --version, as the original loop did.
CliOptions parse_cli(int argc, const char* const* argv);

// The whole command line. The first argument picks the subcommand, and one that
// starts with a dash goes to parse_cli as the legacy form. Throws
// std::runtime_error naming the subcommand, argument or flag at fault.
CliOptions parse_command_line(int argc, const char* const* argv);

// Whether cps run should look this target up as an embedded preset rather than
// read it as a file: it has no path separator and does not end in .json.
bool run_target_is_preset(const std::string& target);

// The line the legacy form prints to stderr.
std::string legacy_form_notice();

// Copies every override that was given into config. Does not validate; call
// Config::validate() afterwards.
void apply_cli_overrides(const CliOptions& options, Config& config);

std::string cli_usage();
// The help of one subcommand. Legacy and Usage give the whole message.
std::string cli_usage(CliCommand command);

}  // namespace cps
